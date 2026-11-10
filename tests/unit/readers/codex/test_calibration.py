"""External calibration declarations and exact XY affine requirements."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from spatialdata.transformations import Affine

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.codex._calibration import read_experiment, resolve_calibration, validate_calibration

if TYPE_CHECKING:
    from pathlib import Path


class TestValidateCalibration:
    """Reject invalid public assumptions before I/O."""

    @pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_scale(self, value: float) -> None:
        with pytest.raises(ValueError, match="positive finite"):
            validate_calibration(value, None)

    @pytest.mark.parametrize("value", [True, "0.5"])
    def test_wrong_scale_type(self, value: object) -> None:
        with pytest.raises(TypeError, match="positive finite"):
            validate_calibration(cast("float", value), None)

    def test_valid(self) -> None:
        validate_calibration(None, None)
        validate_calibration(0.5, Affine(np.eye(3), input_axes=("x", "y"), output_axes=("x", "y")))

    def test_wrong_transform_type(self) -> None:
        with pytest.raises(TypeError, match="Affine"):
            validate_calibration(None, cast("Affine", "identity"))

    def test_wrong_axes(self) -> None:
        with pytest.raises(ValueError, match="exactly XY"):
            validate_calibration(None, Affine(np.eye(3), input_axes=("y", "x"), output_axes=("x", "y")))

    @pytest.mark.parametrize(
        "matrix", [[[1, 0, float("nan")], [0, 1, 0], [0, 0, 1]], [[0, 0, 0], [0, 1, 0], [0, 0, 1]]]
    )
    def test_invalid_matrix(self, matrix: list[list[float]]) -> None:
        with pytest.raises(ValueError, match=r"finite|invertible"):
            validate_calibration(
                None, Affine(np.asarray(matrix, dtype=np.float64), input_axes=("x", "y"), output_axes=("x", "y"))
            )

    def test_near_projective_original_matrix(self) -> None:
        matrix = np.eye(3)
        matrix[2, 0] = 1e-9  # SpatialData's constructor accepts this through allclose.
        mapping = Affine(matrix, input_axes=("x", "y"), output_axes=("x", "y"))
        with pytest.raises(ValueError, match="3-by-3 affine"):
            validate_calibration(None, mapping)

    @pytest.mark.parametrize("mutation", ["shape", "last_row", "nonfinite"])
    def test_externally_mutated_matrix(self, mutation: str) -> None:
        mapping = Affine(np.eye(3), input_axes=("x", "y"), output_axes=("x", "y"))
        if mutation == "shape":
            mapping.matrix = np.eye(2)
        elif mutation == "last_row":
            mapping.matrix[2, 2] = 2
        else:
            mapping.matrix[0, 0] = np.inf
        with pytest.raises(ValueError, match="3-by-3 affine"):
            validate_calibration(None, mapping)

    def test_mutated_complex_matrix_cannot_lose_its_imaginary_component(self) -> None:
        mapping = Affine(np.eye(3), input_axes=("x", "y"), output_axes=("x", "y"))
        mapping.matrix = np.diag([1j, 1, 1])
        with pytest.raises(ValueError, match="float64 3-by-3 affine"):
            validate_calibration(None, mapping)


class TestReadExperiment:
    """Interpret explicitly selected producer declarations without association guesses."""

    @pytest.mark.parametrize(
        ("version", "resolution"), [("1.5.0.38", 377.4463383838384), ("1.7.0.6", 377.40384615384613)]
    )
    def test_known_declarations(self, tmp_path: Path, version: str, resolution: float) -> None:
        path = tmp_path / "experiment.json"
        path.write_text(json.dumps({"version": version, "xyResolution": resolution}))

        calibration = read_experiment(ArtifactFile(path))

        assert calibration.pixel_size_um == resolution / 1000
        assert calibration.version == version
        assert "Akoya experiment xyResolution" in calibration.source

    @pytest.mark.parametrize("value", [None, True, "377", 0, -1])
    def test_invalid_resolution(self, tmp_path: Path, value: object) -> None:
        path = tmp_path / "experiment.json"
        path.write_text(json.dumps({"version": "1.7.0.6", "xyResolution": value}))

        with pytest.raises(ReaderFormatError, match="positive") as caught:
            read_experiment(ArtifactFile(path))

        assert caught.value.context.detected_version == "1.7.0.6"

    def test_huge_integer_resolution(self, tmp_path: Path) -> None:
        path = tmp_path / "experiment.json"
        path.write_text('{"version": "1.7.0.6", "xyResolution": 1' + "0" * 400 + "}")

        with pytest.raises(ReaderFormatError):
            read_experiment(ArtifactFile(path))

    @pytest.mark.parametrize(
        "metadata", [{}, {"version": 1.7}, {"version": "2.0", "xyResolution": 377}, {"per_pixel_XY_resolution": 377}]
    )
    def test_unsupported_profile(self, tmp_path: Path, metadata: dict[str, object]) -> None:
        path = tmp_path / "experiment.json"
        path.write_text(json.dumps(metadata))

        with pytest.raises(ReaderFormatError, match="Akoya version"):
            read_experiment(ArtifactFile(path))

    def test_large_file_is_not_capped(self, tmp_path: Path) -> None:
        """Size is not a validity criterion; the strict JSON reader parses any size."""
        path = tmp_path / "experiment.json"
        path.write_text(json.dumps({"version": "1.7.0.6", "xyResolution": 500, "notes": "x" * (2 * 1024 * 1024)}))

        assert read_experiment(ArtifactFile(path)).pixel_size_um == 0.5


class TestResolveCalibration:
    """Choose the caller scale or the experiment declaration and require agreement."""

    def test_uncalibrated(self) -> None:
        assert resolve_calibration(None, None) is None

    def test_caller_scale(self) -> None:
        calibration = resolve_calibration(0.5, None)

        assert calibration is not None
        assert calibration.pixel_size_um == 0.5
        assert calibration.version is None
        assert calibration.source.startswith("Caller-declared")

    @pytest.mark.parametrize("pixel_size_um", [None, 0.5])
    def test_experiment(self, tmp_path: Path, pixel_size_um: float | None) -> None:
        path = tmp_path / "experiment.json"
        path.write_text('{"version": "1.5.0.38", "xyResolution": 500}')

        calibration = resolve_calibration(pixel_size_um, ArtifactFile(path))

        assert calibration is not None
        assert calibration.pixel_size_um == 0.5
        assert calibration.version == "1.5.0.38"

    def test_conflict(self, tmp_path: Path) -> None:
        path = tmp_path / "experiment.json"
        path.write_text('{"version": "1.5.0.38", "xyResolution": 500}')

        with pytest.raises(ValueError, match=r"0\.6 conflicts .* \(0\.5\)"):
            resolve_calibration(0.6, ArtifactFile(path))
