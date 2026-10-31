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

    from pytest_mock import MockerFixture


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
        assert (actual.nrows, actual.ncols) == (expected.nrows, expected.ncols)
        np.testing.assert_array_equal(actual.cytassist, expected.cytassist)

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
        assert (result.nrows, result.ncols) == (17, 17)
        assert (result.cytassist is None) == (null == "cytassist")
        if result.cytassist is not None:
            assert not result.cytassist.flags.writeable
            np.testing.assert_array_equal(result.cytassist, [[2, 0, 2], [0, 2, 3], [0, 0, 1]])
        with h5py.File(_source(tmp_path), "r+"):
            pass

    @pytest.mark.parametrize("text", ["[]", "{", '{"a":1,"a":2}', '{"a":1e309}', '"string"', "{}", ""])
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

    def test_values(self, tmp_path: Path) -> None:
        write_bundle(tmp_path)
        path = ArtifactFile(tmp_path / "binned_outputs/square_002um/spatial/scalefactors_json.json")
        assert metadata.read_bin_scale_factors(path, 2) == metadata.ScaleFactors(
            hires=0.5, lowres=0.25, microns_per_pixel=0.5
        )

    @pytest.mark.parametrize(
        "field",
        ["tissue_hires_scalef", "tissue_lowres_scalef", "spot_diameter_fullres", "microns_per_pixel", "bin_size_um"],
    )
    @pytest.mark.parametrize("value", [None, True, "2", 0, -1, float("inf")])
    def test_bad_values(self, tmp_path: Path, field: str, value: object) -> None:
        write_bundle(tmp_path)
        path = ArtifactFile(tmp_path / "binned_outputs/square_002um/spatial/scalefactors_json.json")
        values = json.loads(path.read_text())
        values[field] = value
        path.write_text(json.dumps(values))
        with pytest.raises(ReaderFormatError):
            metadata.read_bin_scale_factors(path, 2)

    def test_missing_bin_size(self, tmp_path: Path) -> None:
        write_bundle(tmp_path)
        path = ArtifactFile(tmp_path / "binned_outputs/square_002um/spatial/scalefactors_json.json")
        values = json.loads(path.read_text())
        del values["bin_size_um"]
        path.write_text(json.dumps(values))
        with pytest.raises(ReaderFormatError, match="bin_size_um"):
            metadata.read_bin_scale_factors(path, 2)

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
            assert metadata.read_bin_scale_factors(path, 2).microns_per_pixel == 0.5
        else:
            with pytest.raises(ReaderFormatError):
                metadata.read_bin_scale_factors(path, 2)


class TestReconcileIdentities:
    """Every consumed source must belong to one coherent producer export."""

    @staticmethod
    def _write(path: ArtifactFile, version: str, library_id: str | None) -> None:
        with h5py.File(path, "w") as handle:
            handle.attrs["software_version"] = version
            if library_id is not None:
                handle.attrs["library_ids"] = np.array([library_id.encode()])

    @pytest.mark.parametrize("reverse", [False, True])
    def test_missing_ids_and_normalized_versions(self, tmp_path: Path, *, reverse: bool) -> None:
        paths = [ArtifactFile(tmp_path / name) for name in ("feature_slice.h5", "counts.h5")]
        self._write(paths[0], "spaceranger-4.0.1", None)
        self._write(paths[1], "4.0.1", "sample")
        if reverse:
            paths.reverse()
        assert metadata.reconcile_identities(paths) == metadata.Identity("4.0.1", "sample")

    def test_all_missing_ids_and_distinct_path_reads(self, tmp_path: Path, mocker: MockerFixture) -> None:
        paths = [ArtifactFile(tmp_path / name) for name in ("raw.h5", "feature_slice.h5")]
        for path in paths:
            self._write(path, "3.1.2", None)
        spy = mocker.spy(metadata, "read_identity")
        assert metadata.reconcile_identities([paths[0], paths[1], paths[0]]) == metadata.Identity("3.1.2", None)
        assert spy.call_count == 2
        assert [call.args[0] for call in spy.call_args_list] == paths

    @pytest.mark.parametrize("version", ["2.0.0", "4.1.0", "4.0.2"])
    def test_nonfirst_version_conflict(self, tmp_path: Path, version: str) -> None:
        paths = [ArtifactFile(tmp_path / name) for name in ("first.h5", "second.h5")]
        self._write(paths[0], "4.0.1", "sample")
        self._write(paths[1], version, "sample")
        with pytest.raises(ReaderFormatError) as caught:
            metadata.reconcile_identities(paths)
        assert caught.value.path == paths[1]
        assert caught.value.context.reader == "visium_hd"
        assert caught.value.context.artifact == "count identity"
        assert caught.value.context.detected_version == version

    def test_known_library_conflict_after_unknown(self, tmp_path: Path) -> None:
        paths = [ArtifactFile(tmp_path / name) for name in ("unknown.h5", "first.h5", "second.h5")]
        for path, library_id in zip(paths, (None, "sample", "other"), strict=True):
            self._write(path, "4.0.1", library_id)
        with pytest.raises(ReaderFormatError) as caught:
            metadata.reconcile_identities(paths)
        assert caught.value.path == paths[2]
        assert caught.value.context.detected_version == "4.0.1"
        assert "sample" in caught.value.expected
        assert "other" in caught.value.found

    def test_empty_sources(self) -> None:
        with pytest.raises(ReaderFormatError, match="identity-bearing"):
            metadata.reconcile_identities([])
