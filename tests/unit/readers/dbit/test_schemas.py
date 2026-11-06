"""DBiT positional schemas, imported annotations, and exact output collisions."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.dbit._schemas import (
    BARCODE_SOURCE_DTYPE,
    GRID_DTYPE,
    normalize_barcodes,
    validate_barcode_prefixes,
    validate_mapped_positions,
    validate_table,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestNormalizeBarcodes:
    """Keep positional meaning strict and normalized storage explicit."""

    def test_typed_case_normalization(self, tmp_path: Path) -> None:
        """Retain row order and extract typed positions once."""
        frame = pd.DataFrame([["B1", "cccccccc"], ["A1", "aaaaaaaa"]], dtype=BARCODE_SOURCE_DTYPE)
        normalize_barcodes(frame, ArtifactFile(tmp_path / "barcodes.txt"))
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
            normalize_barcodes(frame, ArtifactFile(tmp_path / "barcodes.txt"))

    def test_empty_rows(self, tmp_path: Path) -> None:
        """Reject empty positional input before extracting fields."""
        with pytest.raises(ReaderFormatError, match="barcode map shape"):
            normalize_barcodes(
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
            validate_table(table, ArtifactFile(tmp_path / "source.h5ad"))
        assert table.obsm["spatial"] is native

    @pytest.mark.parametrize("kind", ["equal", "conflicting", "null", "categorical"])
    def test_generated_observation_columns_are_accepted_unchanged(self, tmp_path: Path, kind: str) -> None:
        """Source values and storage are irrelevant to generated output destinations."""
        table = ad.AnnData(np.ones((2, 2)))
        native = np.arange(6, dtype=np.int16).reshape(2, 3)
        table.obsm["spatial"] = native
        for field in ("array_A", "array_B", "sample", "region", "pixel_id"):
            if kind == "categorical":
                table.obs[field] = pd.Categorical(["stale", None], categories=["stale"], ordered=True)
            elif kind == "null":
                table.obs[field] = pd.array([None, None], dtype="Int16")
            elif field in {"sample", "region"}:
                table.obs[field] = ["sample" if kind == "equal" else "other"] * 2
            else:
                table.obs[field] = [0, 1] if field == "pixel_id" and kind == "equal" else [1, 2]
        assert isinstance(table.obs, pd.DataFrame)
        original = table.obs.copy(deep=True)

        validate_table(table, ArtifactFile(tmp_path / "source.h5ad"))

        pd.testing.assert_frame_equal(table.obs, original)
        assert table.obsm["spatial"] is native
        assert "spatial_source" not in table.obsm
        assert "spatial_grid" not in table.obsm

    @pytest.mark.parametrize(
        "linkage",
        [
            None,
            False,
            "",
            {},
            {"region": 42},
            {"region": "old", "region_key": "region", "instance_key": "pixel_id"},
            {"region": "old", "region_key": "custom_region", "instance_key": "custom_id"},
        ],
    )
    def test_inherited_linkage_is_accepted_without_mutation(
        self, tmp_path: Path, *, linkage: bool | str | dict[str, str | int] | None
    ) -> None:
        """Schema validation leaves arbitrary prior linkage for final replacement."""
        table = ad.AnnData(np.ones((2, 2)))
        native = np.arange(6).reshape(2, 3)
        table.obsm["spatial"] = native
        table.uns["spatialdata_attrs"] = linkage
        original = deepcopy(linkage)

        validate_table(table, ArtifactFile(tmp_path / "source.h5ad"))

        assert table.obsm["spatial"] is native
        assert "spatial_source" not in table.obsm
        assert table.uns["spatialdata_attrs"] is linkage
        assert table.uns["spatialdata_attrs"] == original
        np.testing.assert_array_equal(native, np.arange(6).reshape(2, 3))

    def test_independent_source_coordinates_remain_available(self, tmp_path: Path) -> None:
        """An imported source embedding needs no displacement without spatial."""
        table = ad.AnnData(np.ones((2, 2)))
        native = np.arange(6).reshape(2, 3)
        table.obsm["spatial_source"] = native

        validate_table(table, ArtifactFile(tmp_path / "source.h5ad"))

        assert table.obsm["spatial_source"] is native

    @pytest.mark.parametrize("axis", ["obs", "var"])
    def test_duplicate_names(self, tmp_path: Path, axis: str) -> None:
        """Reject duplicate annotations before named access."""
        table = ad.AnnData(np.ones((1, 1)))
        setattr(table, axis, pd.DataFrame([[1, 2]], columns=["same", "same"], index=["0"]))
        with pytest.raises(ReaderFormatError, match="duplicate annotation"):
            validate_table(table, ArtifactFile(tmp_path / "table.h5ad"))

    @pytest.mark.parametrize("axis", ["obs", "var"])
    def test_null_index(self, tmp_path: Path, axis: str) -> None:
        """Reject persisted nullable-string identifiers."""
        table = ad.AnnData(np.ones((1, 1)))
        frame = table.obs if axis == "obs" else table.var
        assert isinstance(frame, pd.DataFrame)
        frame.index = pd.Index(pd.array([None], dtype="string"))
        with pytest.raises(ReaderFormatError, match="identifiers"):
            validate_table(table, ArtifactFile(tmp_path / "table.h5ad"))

    @pytest.mark.parametrize("axis", ["obs", "var"])
    def test_annotations_must_be_eager_dataframes(self, tmp_path: Path, axis: str) -> None:
        """AnnData types annotations as eager or lazy frames; the reader needs eager ones."""
        frames: dict[str, object] = {"obs": pd.DataFrame(index=["0"]), "var": pd.DataFrame(index=["0"])}
        frames[axis] = object()  # Stands in for a lazy Dataset2D.
        table = SimpleNamespace(obsm={}, uns={}, X=np.ones((1, 1)), n_obs=1, n_vars=1, **frames)

        with pytest.raises(TypeError, match="eager annotation DataFrames"):
            validate_table(table, ArtifactFile(tmp_path / "table.h5ad"))  # pyrefly: ignore[bad-argument-type] - Duck-typed AnnData stand-in.


class TestValidateBarcodePrefixes:
    """Require unique upper-case 16-base B/A prefixes."""

    def test_valid_prefixes(self, tmp_path: Path) -> None:
        validate_barcode_prefixes(pd.Index(["ACGTACGTACGTACGT", "TTTTTTTTAAAAAAAA"]), ArtifactFile(tmp_path / "t.h5ad"))

    @pytest.mark.parametrize(
        "names",
        [["ACGTACGTACGTACG"], ["ACGTACGTACGTACGN"], ["acgtacgtacgtacgt"], ["ACGTACGTACGTACGT", "ACGTACGTACGTACGT"]],
        ids=["short", "non_base", "lower_case", "duplicate"],
    )
    def test_invalid_prefixes(self, tmp_path: Path, names: list[str]) -> None:
        with pytest.raises(ReaderFormatError, match="malformed or duplicate barcode pairs"):
            validate_barcode_prefixes(pd.Index(names), ArtifactFile(tmp_path / "t.h5ad"))


class TestValidateMappedPositions:
    """Require both barcode halves of every observation in the map."""

    def test_mapped(self, tmp_path: Path) -> None:
        positions = pd.Series([1, 2], dtype=GRID_DTYPE)
        validate_mapped_positions(positions, positions, ArtifactFile(tmp_path / "t.h5ad"))

    @pytest.mark.parametrize("axis", ["a", "b"])
    def test_unknown_barcode(self, tmp_path: Path, axis: str) -> None:
        known = pd.Series([1, 2], dtype=GRID_DTYPE)
        unknown = pd.Series([1, None], dtype=GRID_DTYPE)
        a, b = (unknown, known) if axis == "a" else (known, unknown)
        with pytest.raises(ReaderFormatError, match="unknown observation barcodes"):
            validate_mapped_positions(a, b, ArtifactFile(tmp_path / "t.h5ad"))
