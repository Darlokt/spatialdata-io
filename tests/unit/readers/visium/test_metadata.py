"""Tests for Visium metadata parsing."""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.visium._metadata import CountMetadata, read_count_metadata, read_scale_factors

if TYPE_CHECKING:
    from pathlib import Path


class TestReadScaleFactors:
    """Validate required numerical scale-factor semantics and mapping ownership."""

    def test_preserves_complete_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "scales.json"
        path.write_text(
            json.dumps(
                {
                    "tissue_hires_scalef": 0.5,
                    "tissue_lowres_scalef": 0.25,
                    "spot_diameter_fullres": 6,
                    "fiducial_diameter_fullres": 8.5,
                }
            ),
            encoding="utf-8",
        )

        result = read_scale_factors(ArtifactFile(path), version=None)

        assert result.hires == 0.5
        assert result.lowres == 0.25
        assert result.spot_diameter == 6.0
        assert result.values["fiducial_diameter_fullres"] == 8.5

    @pytest.mark.parametrize("value", [None, True, 0, -1, float("inf"), float("nan"), 10**400])
    def test_rejects_missing_or_nonpositive_required_values(self, tmp_path: Path, value: object) -> None:
        path = tmp_path / "scales.json"
        values: dict[str, object] = {
            "tissue_hires_scalef": value,
            "tissue_lowres_scalef": 0.25,
            "spot_diameter_fullres": 6,
        }
        path.write_text(json.dumps(values), encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="finite"):
            read_scale_factors(ArtifactFile(path), version=None)

    @pytest.mark.parametrize("key", ["tissue_hires_scalef", "tissue_lowres_scalef"])
    def test_rejects_finite_scale_with_nonfinite_inverse(self, tmp_path: Path, key: str) -> None:
        path = tmp_path / "scales.json"
        values = {"tissue_hires_scalef": 0.5, "tissue_lowres_scalef": 0.25, "spot_diameter_fullres": 6.0}
        values[key] = float(np.nextafter(0.0, 1.0))
        path.write_text(json.dumps(values), encoding="utf-8")

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(ReaderFormatError, match=key) as caught:
                read_scale_factors(ArtifactFile(path), version="2.1.0")

        assert caught.value.path == path
        assert caught.value.context == ReaderErrorContext("visium", "scale factors", "2.1.0")

    def test_accepts_small_scales_with_finite_inverse(self, tmp_path: Path) -> None:
        path = tmp_path / "scales.json"
        tiny = 1e-308
        path.write_text(
            json.dumps({"tissue_hires_scalef": tiny, "tissue_lowres_scalef": tiny, "spot_diameter_fullres": 6}),
            encoding="utf-8",
        )

        result = read_scale_factors(ArtifactFile(path), version=None)

        assert result.hires == tiny
        assert result.lowres == tiny
        np.testing.assert_array_equal([1 / result.hires, 1 / result.lowres], [1e308, 1e308])

    def test_rejects_display_radius_underflow_with_artifact_context(self, tmp_path: Path) -> None:
        path = tmp_path / "scales.json"
        path.write_text(
            json.dumps(
                {
                    "tissue_hires_scalef": 0.5,
                    "tissue_lowres_scalef": 0.25,
                    "spot_diameter_fullres": float(np.nextafter(0.0, 1.0)),
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ReaderFormatError, match="spot_diameter_fullres") as caught:
            read_scale_factors(ArtifactFile(path), version=None)

        assert caught.value.path == path
        assert caught.value.context.artifact == "scale factors"

    def test_accepts_smallest_representable_positive_radius(self, tmp_path: Path) -> None:
        path = tmp_path / "scales.json"
        radius = float(np.nextafter(0.0, 1.0))
        path.write_text(
            json.dumps({"tissue_hires_scalef": 0.5, "tissue_lowres_scalef": 0.25, "spot_diameter_fullres": 2 * radius}),
            encoding="utf-8",
        )

        result = read_scale_factors(ArtifactFile(path), version=None)

        assert result.spot_diameter / 2 == radius


class TestCountMetadata:
    """Normalize the Space Ranger software version once."""

    @pytest.mark.parametrize(
        ("software_version", "expected"),
        [
            ("spaceranger-1.1.0", "1.1.0"),
            ("spaceranger-2.1.0", "2.1.0"),
            ("3.0.1", "3.0.1"),
            ("spaceranger-2.1", None),
            ("spaceranger-2.1.0-rc1", None),
            ("cellranger-7.0.0", None),
            (None, None),
        ],
    )
    def test_format_version(self, software_version: str | None, expected: str | None) -> None:
        assert CountMetadata(None, None, software_version).format_version == expected


class TestReadCountMetadata:
    """Read optional attributes while the HDF5 owner is open."""

    @pytest.mark.parametrize(
        "library_ids", [np.asarray([b"library"]), b"library", "library"], ids=["array", "bytes", "str"]
    )
    def test_reads_single_library_and_optional_text(self, tmp_path: Path, library_ids: object) -> None:
        path = tmp_path / "counts.h5"
        with h5py.File(path, "w") as source:
            source.attrs["library_ids"] = library_ids
            source.attrs["chemistry_description"] = "chemistry"
            source.attrs["software_version"] = b"spaceranger-2.1.0"

        result = read_count_metadata(ArtifactFile(path))

        assert result == CountMetadata("library", "chemistry", "spaceranger-2.1.0")

    def test_absent_attributes_are_none(self, tmp_path: Path) -> None:
        path = tmp_path / "counts.h5"
        h5py.File(path, "w").close()

        assert read_count_metadata(ArtifactFile(path)) == CountMetadata(None, None, None)

    @pytest.mark.parametrize("library_ids", [np.asarray([], dtype="S1"), np.asarray([b"a", b"b"])])
    def test_rejects_non_single_library_h5(self, tmp_path: Path, library_ids: np.ndarray) -> None:
        path = tmp_path / "counts.h5"
        with h5py.File(path, "w") as source:
            source.attrs["library_ids"] = library_ids

        with pytest.raises(ReaderFormatError, match="exactly one library ID"):
            read_count_metadata(ArtifactFile(path))

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            # h5py stores bytes as a variable-length string and returns it decoded with surrogateescape.
            (b"\xff", "non-UTF-8 bytes"),
            (np.bytes_(b"\xff"), "non-UTF-8 bytes"),
            (b"", "empty attribute"),
            (np.int64(1), "value type int64"),
        ],
        ids=["variable-length", "fixed-length", "empty", "integer"],
    )
    def test_rejects_malformed_text(self, tmp_path: Path, value: object, message: str) -> None:
        path = tmp_path / "counts.h5"
        with h5py.File(path, "w") as source:
            source.attrs["software_version"] = value

        with pytest.raises(ReaderFormatError, match=message) as caught:
            read_count_metadata(ArtifactFile(path))

        assert caught.value.context.artifact == "count metadata"
        assert caught.value.path == path

    def test_unreadable_file_keeps_its_error_with_context(self, tmp_path: Path) -> None:
        path = tmp_path / "counts.h5"
        path.write_bytes(b"not hdf5")

        with pytest.raises(OSError, match="signature") as caught:
            read_count_metadata(ArtifactFile(path))

        assert any("count metadata" in note for note in caught.value.__notes__)
