"""Tests for bounded Visium metadata parsing."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactDirectory, ArtifactFile
from spatialdata_io.readers.visium._layout import H5Counts, MtxCounts
from spatialdata_io.readers.visium._metadata import (
    CountMetadata,
    read_count_metadata,
    read_scale_factors,
    resolve_dataset_id,
)

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

        result = read_scale_factors(ArtifactFile(path))

        assert result.hires == 0.5
        assert result.lowres == 0.25
        assert result.spot_diameter == 6.0
        assert result.values["fiducial_diameter_fullres"] == 8.5

    @pytest.mark.parametrize("value", [None, True, 0, -1, float("inf"), float("nan")])
    def test_rejects_missing_or_nonpositive_required_values(self, tmp_path: Path, value: object) -> None:
        path = tmp_path / "scales.json"
        values: dict[str, object] = {
            "tissue_hires_scalef": value,
            "tissue_lowres_scalef": 0.25,
            "spot_diameter_fullres": 6,
        }
        path.write_text(json.dumps(values), encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="finite"):
            read_scale_factors(ArtifactFile(path))


class TestReadCountMetadata:
    """Read only bounded attributes while the HDF5 owner is open."""

    def test_reads_single_library_and_optional_text(self, tmp_path: Path) -> None:
        path = tmp_path / "counts.h5"
        with h5py.File(path, "w") as source:
            source.attrs["library_ids"] = np.asarray([b"library"])
            source.attrs["chemistry_description"] = "chemistry"
            source.attrs["software_version"] = b"spaceranger-2.1.0"

        result = read_count_metadata(H5Counts(path=ArtifactFile(path), filename_prefix=None))

        assert result == CountMetadata("library", "chemistry", "spaceranger-2.1.0")

    def test_mex_has_no_embedded_metadata(self, tmp_path: Path) -> None:
        result = read_count_metadata(
            MtxCounts(
                path=ArtifactDirectory(tmp_path),
                prefix=None,
                compressed=True,
                filename_prefix="sample",
            )
        )

        assert result == CountMetadata(None, None, None)

    @pytest.mark.parametrize("library_ids", [np.asarray([], dtype="S1"), np.asarray([b"a", b"b"])])
    def test_rejects_non_single_library_h5(self, tmp_path: Path, library_ids: np.ndarray) -> None:
        path = tmp_path / "counts.h5"
        with h5py.File(path, "w") as source:
            source.attrs["library_ids"] = library_ids

        with pytest.raises(ReaderFormatError, match="exactly one library ID"):
            read_count_metadata(H5Counts(path=ArtifactFile(path), filename_prefix=None))

    def test_rejects_unbounded_attribute_text(self, tmp_path: Path) -> None:
        path = tmp_path / "counts.h5"
        with h5py.File(path, "w") as source:
            source.attrs["software_version"] = "x" * 4_097

        with pytest.raises(ReaderFormatError, match="at most 4096"):
            read_count_metadata(H5Counts(path=ArtifactFile(path), filename_prefix=None))


class TestResolveDatasetId:
    """Resolve precedence and reject identifiers unsafe for element names."""

    def test_explicit_id_precedes_embedded_and_filename_ids(self, tmp_path: Path) -> None:
        counts = H5Counts(path=ArtifactFile(tmp_path / "counts.h5"), filename_prefix="prefix")

        result = resolve_dataset_id("explicit", counts, CountMetadata("embedded", None, None))

        assert result == "explicit"

    def test_uses_embedded_then_filename_prefix(self, tmp_path: Path) -> None:
        counts = H5Counts(path=ArtifactFile(tmp_path / "counts.h5"), filename_prefix="prefix")

        assert resolve_dataset_id(None, counts, CountMetadata("embedded", None, None)) == "embedded"
        assert resolve_dataset_id(None, counts, CountMetadata(None, None, None)) == "prefix"

    @pytest.mark.parametrize("value", [None, "", " sample", "sample/one", "sample\\one", "sample\x00one"])
    def test_rejects_missing_or_unsafe_id(self, tmp_path: Path, value: str | None) -> None:
        counts = H5Counts(path=ArtifactFile(tmp_path / "counts.h5"), filename_prefix=None)

        with pytest.raises(ValueError, match="dataset_id"):
            resolve_dataset_id(value, counts, CountMetadata(None, None, None))
