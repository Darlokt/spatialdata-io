"""Bounded affine parsing with explicit direction and numerically stable invertibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import spatialdata_io.readers.merscope._transforms as transform_module
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.merscope._transforms import read_transform

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestReadTransform:
    """Interpret source microns as mosaic pixels without changing or repairing the matrix."""

    def test_nonidentity_affine_direction(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "transform.csv")
        expected = np.asarray([[2, 0.25, 10], [0, 3, -7], [0, 0, 1]])
        np.savetxt(path, expected)
        actual = read_transform(path).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual @ [2, 2, 1], [14.5, -1, 1])

    @pytest.mark.parametrize("scale", [1e-200, float(2**600)])
    def test_invertible_scale_is_not_rejected_by_determinant_underflow_or_overflow(
        self, tmp_path: Path, scale: float
    ) -> None:
        path = ArtifactFile(tmp_path / "transform.csv")
        expected = np.diag([scale, scale, 1])
        coefficient = str(2**600) if scale > 1 else repr(scale)
        path.write_text(f"{coefficient} 0 0\n0 {coefficient} 0\n0 0 1\n", encoding="utf-8")
        np.testing.assert_array_equal(read_transform(path).matrix, expected)

    @pytest.mark.parametrize(
        "content",
        [
            b"not numeric",
            b"\xff",
            b"1 0\n0 1\n",
            b"1 0 0\n0 nan 0\n0 0 1\n",
            b"1 0 0\n0 1 0\n0 0 2\n",
            b"1 2 0\n2 4 0\n0 0 1\n",
        ],
    )
    def test_invalid_transform(self, tmp_path: Path, content: bytes) -> None:
        path = ArtifactFile(tmp_path / "transform.csv")
        path.write_bytes(content)
        with pytest.raises(ReaderFormatError):
            read_transform(path)

    @pytest.mark.parametrize("coefficient", ["1e-400", "9007199254740993", "nan", "inf"])
    def test_coefficient_conversion_cannot_change_magnitude_or_integer_value(
        self, tmp_path: Path, coefficient: str
    ) -> None:
        path = ArtifactFile(tmp_path / "transform.csv")
        path.write_text(f"1 {coefficient} 0\n0 1 0\n0 0 1\n", encoding="utf-8")
        with pytest.raises(ReaderFormatError, match="malformed transform"):
            read_transform(path)

    def test_unrepresentable_inverse_fails(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "transform.csv")
        path.write_text("1e-320 0 0\n0 1 0\n0 0 1\n", encoding="utf-8")
        with pytest.raises(ReaderFormatError, match="inverse"):
            read_transform(path)

    def test_size_bound(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = ArtifactFile(tmp_path / "transform.csv")
        path.write_bytes(b"1 0 0\n0 1 0\n0 0 1\n")
        mocker.patch.object(transform_module, "_MAX_TRANSFORM_BYTES", 8)
        with pytest.raises(ReaderFormatError, match="oversized"):
            read_transform(path)

    def test_io_error_retains_identity_context_and_resource_cleanup(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        path = ArtifactFile(tmp_path / "transform.csv")
        error = OSError("injected")
        handle = mocker.MagicMock()
        handle.read.side_effect = error
        owner = mocker.patch("pathlib.Path.open")
        owner.return_value.__enter__.return_value = handle
        with pytest.raises(OSError, match="injected") as caught:
            read_transform(path)
        assert caught.value is error
        assert any("merscope" in note and str(path) in note for note in error.__notes__)
        owner.return_value.__exit__.assert_called_once()
