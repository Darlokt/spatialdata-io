"""DBiT positional schemas, imported annotations, and exact output collisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.dbit._schemas import (
    BARCODE_SOURCE_DTYPE,
    _normalize_barcodes,
    _source_linkage_columns,
    _validate_output_column,
    _validate_table,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestNormalizeBarcodes:
    """Keep positional meaning strict and normalized storage explicit."""

    def test_typed_case_normalization(self, tmp_path: Path) -> None:
        """Retain row order and extract typed positions once."""
        frame = pd.DataFrame([["B1", "cccccccc"], ["A1", "aaaaaaaa"]], dtype=BARCODE_SOURCE_DTYPE)
        _normalize_barcodes(frame, ArtifactFile(tmp_path / "barcodes.txt"))
        assert frame["position"].tolist() == [1, 1]
        assert frame["sequence"].tolist() == ["CCCCCCCC", "AAAAAAAA"]
        assert frame["axis"].tolist() == ["B", "A"]
        assert frame.dtypes.astype(str).tolist() == ["int64[pyarrow]", "string[pyarrow]", "string[pyarrow]"]

    @pytest.mark.parametrize("column", [0, 1])
    def test_null_source_fields(self, tmp_path: Path, column: int) -> None:
        """Reject missing tokens before case conversion or position parsing."""
        frame = pd.DataFrame([["A1", "AAAAAAAA"], ["B1", "CCCCCCCC"]], dtype=BARCODE_SOURCE_DTYPE)
        frame.iloc[0, column] = None
        with pytest.raises(ReaderFormatError, match="invalid barcode"):
            _normalize_barcodes(frame, ArtifactFile(tmp_path / "barcodes.txt"))

    def test_empty_rows(self, tmp_path: Path) -> None:
        """Reject empty positional input before extracting fields."""
        with pytest.raises(ReaderFormatError, match="barcode map shape"):
            _normalize_barcodes(
                pd.DataFrame(columns=[0, 1], dtype=BARCODE_SOURCE_DTYPE), ArtifactFile(tmp_path / "barcodes.txt")
            )


class TestValidateTable:
    """Preserve imported extensions and reject ambiguous column/index names."""

    @pytest.mark.parametrize("destination", ["spatial_source", "spatialdata_io_positions", "spatial_grid"])
    def test_reserved_coordinate_destinations(self, tmp_path: Path, destination: str) -> None:
        table = ad.AnnData(np.ones((2, 2)))
        native = np.arange(6, dtype=np.int16).reshape(2, 3)
        table.obsm["spatial"] = native
        if destination == "spatialdata_io_positions":
            table.uns[destination] = {"coordinates": {"spatial": {"units": "micrometer"}}}
        else:
            table.obsm[destination] = native
        with pytest.raises(ReaderFormatError, match="occupied"):
            _validate_table(table, ArtifactFile(tmp_path / "source.h5ad"))
        assert table.obsm["spatial"] is native

    @pytest.mark.parametrize("axis", ["obs", "var"])
    def test_duplicate_names(self, tmp_path: Path, axis: str) -> None:
        """Reject duplicate annotations before named access."""
        table = ad.AnnData(np.ones((1, 1)))
        setattr(table, axis, pd.DataFrame([[1, 2]], columns=["same", "same"], index=["0"]))
        with pytest.raises(ReaderFormatError, match="duplicate annotation"):
            _validate_table(table, ArtifactFile(tmp_path / "table.h5ad"))

    @pytest.mark.parametrize("axis", ["obs", "var"])
    def test_null_index(self, tmp_path: Path, axis: str) -> None:
        """Reject persisted nullable-string identifiers."""
        table = ad.AnnData(np.ones((1, 1)))
        frame = table.obs if axis == "obs" else table.var
        assert isinstance(frame, pd.DataFrame)
        frame.index = pd.Index(pd.array([None], dtype="string"))
        with pytest.raises(ReaderFormatError, match="identifiers"):
            _validate_table(table, ArtifactFile(tmp_path / "table.h5ad"))


class TestValidateOutputColumn:
    """Reject distinct numeric families or nulls in reserved metadata."""

    @pytest.mark.parametrize("existing", [True, None, "1", 1.5])
    def test_conflicts(self, tmp_path: Path, *, existing: bool | str | float | None) -> None:
        """Keep integer output semantics exact despite Python coercion rules."""
        with pytest.raises(ReaderFormatError, match="conflicting reserved"):
            _validate_output_column(
                pd.DataFrame({"pixel_id": [existing]}),
                "pixel_id",
                pd.Series([1], dtype="int64[pyarrow]"),
                path=ArtifactFile(tmp_path / "table.h5ad"),
            )


class TestSourceLinkageColumns:
    """Recognize a declared old target without deleting unrelated metadata."""

    @pytest.mark.parametrize("regions", ["old", ["old"], ("old",), np.asarray(["old"])])
    def test_valid_previous_linkage(self, tmp_path: Path, regions: str | list[str] | tuple[str] | np.ndarray) -> None:
        """Accept supported persisted region declaration representations."""
        table = ad.AnnData(np.ones((1, 1)))
        table.obs["region"], table.obs["pixel_id"] = ["old"], [90]
        table.uns["spatialdata_attrs"] = {"region": regions, "region_key": "region", "instance_key": "pixel_id"}
        assert _source_linkage_columns(table, ArtifactFile(tmp_path / "table.h5ad")) == ("region", "pixel_id")

    @pytest.mark.parametrize("regions", [None, "", [], [None]])
    def test_malformed_previous_linkage(self, tmp_path: Path, regions: str | list[None] | None) -> None:
        """Reject incomplete declarations before removing source linkage fields."""
        table = ad.AnnData(np.ones((1, 1)))
        table.obs["region"], table.obs["pixel_id"] = ["old"], [90]
        table.uns["spatialdata_attrs"] = {"region": regions, "region_key": "region", "instance_key": "pixel_id"}
        with pytest.raises(ReaderFormatError, match="malformed previous"):
            _source_linkage_columns(table, ArtifactFile(tmp_path / "table.h5ad"))

    def test_missing_declared_columns(self, tmp_path: Path) -> None:
        """Reject a declaration lacking its corresponding annotation fields."""
        table = ad.AnnData(np.ones((1, 1)))
        table.uns["spatialdata_attrs"] = {"region": "old", "region_key": "region", "instance_key": "pixel_id"}
        with pytest.raises(ReaderFormatError, match="malformed previous"):
            _source_linkage_columns(table, ArtifactFile(tmp_path / "table.h5ad"))
