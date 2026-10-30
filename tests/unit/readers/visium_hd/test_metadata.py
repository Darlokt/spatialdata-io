"""Calibration, nullable registration pairs and count identity."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.visium_hd import _metadata as metadata
from tests.integration.readers.visium_hd._fixtures import write_bundle

if TYPE_CHECKING:
    from pathlib import Path


def _source(tmp_path: Path) -> ArtifactFile:
    return ArtifactFile(tmp_path / "feature_slice.h5")


def _replace_metadata(path: ArtifactFile, text: str) -> None:
    with h5py.File(path, "r+") as handle:
        handle.attrs["metadata_json"] = text


class TestReadSourceMetadata:
    """Verify read source metadata contracts."""

    @pytest.mark.parametrize("layout", [None, {}, {"unconsumed": "annotation"}, "not JSON"])
    def test_ignores_unused_nested_layout(self, tmp_path: Path, layout: object) -> None:
        write_bundle(tmp_path)
        path = _source(tmp_path)
        expected = metadata.read_source_metadata(path)
        with h5py.File(path, "r+") as handle:
            values = json.loads(handle.attrs["metadata_json"])
        if layout is None:
            values.pop("hd_layout_json")
        else:
            values["hd_layout_json"] = layout
        _replace_metadata(path, json.dumps(values))
        actual = metadata.read_source_metadata(path)
        assert (actual.nrows, actual.ncols, actual.sample_id, actual.spot_pitch) == (
            expected.nrows,
            expected.ncols,
            expected.sample_id,
            expected.spot_pitch,
        )
        assert actual.microscope is not None
        assert expected.microscope is not None
        np.testing.assert_array_equal(actual.microscope.spot_to_image, expected.microscope.spot_to_image)

    @pytest.mark.parametrize("null", [None, "microscope", "cytassist"])
    def test_metadata(self, tmp_path: Path, null: str | None) -> None:
        write_bundle(tmp_path)
        if null is not None:
            with h5py.File(_source(tmp_path), "r+") as handle:
                values = json.loads(handle.attrs["metadata_json"])
                values["transform_matrices"][f"spot_colrow_to_{null}_colrow"] = None
                values["transform_matrices"][f"{null}_colrow_to_spot_colrow"] = None
                handle.attrs["metadata_json"] = json.dumps(values)
        result = metadata.read_source_metadata(_source(tmp_path))
        assert (result.nrows, result.ncols, result.sample_id, result.spot_pitch) == (17, 17, "synthetic", 2.0)
        assert (result.microscope is None) == (null == "microscope")
        assert (result.cytassist is None) == (null == "cytassist")
        for pair in (result.microscope, result.cytassist):
            if pair is not None:
                assert not pair.spot_to_image.flags.writeable
                assert not pair.image_to_spot.flags.writeable
                np.testing.assert_allclose(pair.spot_to_image @ pair.image_to_spot, np.eye(3), rtol=0, atol=1e-12)
        with h5py.File(_source(tmp_path), "r+"):
            pass

    @pytest.mark.parametrize("text", ["[]", "{", '{"a":1,"a":2}', '"string"', "{}", ""])
    def test_bad_json(self, tmp_path: Path, text: str) -> None:
        write_bundle(tmp_path)
        _replace_metadata(_source(tmp_path), text)
        with pytest.raises(ReaderFormatError):
            metadata.read_source_metadata(_source(tmp_path))

    @pytest.mark.parametrize(
        "mode",
        [
            "nrows",
            "ncols",
            "pitch",
            "unsupported-pitch",
            "sample",
            "matrices",
            "missing",
            "part-null",
            "shape",
            "singular",
            "boolean",
            "text",
            "nonfinite",
            "inverse",
        ],
    )
    def test_bad_contracts(self, tmp_path: Path, mode: str) -> None:
        write_bundle(tmp_path)
        path = _source(tmp_path)
        with h5py.File(path, "r+") as handle:
            values = json.loads(handle.attrs["metadata_json"])
        matrices = values["transform_matrices"]
        key = "spot_colrow_to_microscope_colrow"
        replacements: dict[str, object] = {
            "nrows": True,
            "ncols": 0,
            "pitch": -1,
            "unsupported-pitch": 4,
            "sample": None,
            "matrices": [],
        }
        keys = {
            "pitch": "spot_pitch",
            "unsupported-pitch": "spot_pitch",
            "sample": "sample_id",
            "matrices": "transform_matrices",
        }
        if mode in replacements:
            values[keys.get(mode, mode)] = replacements[mode]
        elif mode == "missing":
            del matrices[key]
        else:
            matrix_replacements: dict[str, object] = {
                "part-null": None,
                "shape": [1, 2],
                "singular": [0] * 9,
                "boolean": [True, 0, 0, 0, 1, 0, 0, 0, 1],
                "text": ["1", 0, 0, 0, 1, 0, 0, 0, 1],
                "nonfinite": [float("nan")] * 9,
                "inverse": np.eye(3).ravel().tolist(),
            }
            matrices[key] = matrix_replacements[mode]
        _replace_metadata(path, json.dumps(values))
        with pytest.raises(ReaderFormatError):
            metadata.read_source_metadata(path)

    def test_invalid_bytes_and_io(self, tmp_path: Path) -> None:
        write_bundle(tmp_path)
        path = _source(tmp_path)
        with h5py.File(path, "r+") as handle:
            handle.attrs["metadata_json"] = np.bytes_(b"\xff")
        with pytest.raises(ReaderFormatError):
            metadata.read_source_metadata(path)
        path.unlink()
        with pytest.raises(FileNotFoundError):
            metadata.read_source_metadata(path)


class TestReadScaleFactors:
    """Verify read scale factors contracts."""

    def test_values_and_optional_bin(self, tmp_path: Path) -> None:
        write_bundle(tmp_path)
        path = ArtifactFile(tmp_path / "binned_outputs/square_002um/spatial/scalefactors_json.json")
        result = metadata.read_scale_factors(path)
        assert result.bin_size_um == 2
        assert result.hires == 0.5
        assert result.lowres == 0.25
        values = json.loads(path.read_text())
        del values["bin_size_um"]
        path.write_text(json.dumps(values))
        assert metadata.read_scale_factors(path).bin_size_um is None

    @pytest.mark.parametrize("value", [None, True, "2", 0, -1, float("inf")])
    def test_bad_values(self, tmp_path: Path, value: object) -> None:
        write_bundle(tmp_path)
        path = ArtifactFile(tmp_path / "binned_outputs/square_002um/spatial/scalefactors_json.json")
        values = json.loads(path.read_text())
        values["microns_per_pixel"] = value
        path.write_text(json.dumps(values))
        with pytest.raises(ReaderFormatError):
            metadata.read_scale_factors(path)


class TestReadIdentity:
    """Verify read identity contracts."""

    @pytest.mark.parametrize("version", ["spaceranger-3.0.0", "spaceranger-3.1.2", "spaceranger-4.0.1", "4.1.0"])
    def test_versions(self, tmp_path: Path, version: str) -> None:
        path = ArtifactFile(tmp_path / "identity.h5")
        with h5py.File(path, "w") as handle:
            handle.attrs["software_version"] = version
            handle.attrs["library_ids"] = np.array([b"sample"])
        result = metadata.read_identity(path)
        assert result.format_version == version.removeprefix("spaceranger-")
        assert result.library_id == "sample"
        with h5py.File(path, "r+") as handle:
            del handle.attrs["library_ids"]
        assert metadata.read_identity(path).library_id is None

    @pytest.mark.parametrize(
        "mode", ["version", "future", "unobserved3", "missing", "libraries", "bytes", "kind", "io"]
    )
    def test_invalid(self, tmp_path: Path, mode: str) -> None:
        path = ArtifactFile(tmp_path / "identity.h5")
        with h5py.File(path, "w") as handle:
            if mode != "missing":
                handle.attrs["software_version"] = {
                    "version": "spaceranger-2.0.0",
                    "future": "4.2.0",
                    "unobserved3": "3.99.0",
                    "bytes": np.bytes_(b"\xff"),
                    "kind": 3,
                }.get(mode, "4.1.0")
            if mode == "libraries":
                handle.attrs["library_ids"] = np.array([b"one", b"two"])
        if mode == "io":
            path.unlink()
        with pytest.raises((ReaderFormatError, OSError)):
            metadata.read_identity(path)


class TestReadBinScaleFactors:
    """Verify read bin scale factors contracts."""

    @pytest.mark.parametrize("mode", ["valid", "size", "diameter"])
    def test_calibration(self, tmp_path: Path, mode: str) -> None:
        write_bundle(tmp_path)
        path = ArtifactFile(tmp_path / "binned_outputs/square_002um/spatial/scalefactors_json.json")
        values = json.loads(path.read_text())
        if mode == "size":
            values["bin_size_um"] = 8
        elif mode == "diameter":
            values["spot_diameter_fullres"] = 9
        path.write_text(json.dumps(values))
        if mode == "valid":
            assert metadata.read_bin_scale_factors(path, 2).bin_size_um == 2
        else:
            with pytest.raises(ReaderFormatError):
                metadata.read_bin_scale_factors(path, 2)
