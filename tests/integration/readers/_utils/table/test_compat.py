"""Compose Arrow annotations with linkage and real AnnData persistence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from scipy import sparse
from shapely.geometry import Point
from spatialdata import SpatialData
from spatialdata.models import ShapesModel, TableModel

from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table, prepare_anndata_dataframe

if TYPE_CHECKING:
    from pathlib import Path


def _roundtrip(table: ad.AnnData, path: Path, storage: Literal["h5ad", "zarr"]) -> ad.AnnData:
    """Write and read a table through the selected public AnnData storage API."""
    if storage == "h5ad":
        table.write_h5ad(path, convert_strings_to_categoricals=False)
        return ad.read_h5ad(path)
    table.write_zarr(path, convert_strings_to_categoricals=False)
    return ad.read_zarr(path)


def _assert_expression(actual: ad.AnnData, expected: sparse.csr_array) -> None:
    """Compare sparse buffers, shape, and dtype without densifying expression."""
    expression = actual.X
    assert isinstance(expression, sparse.csr_array | sparse.csr_matrix)
    assert expression.shape == expected.shape
    assert expression.dtype == expected.dtype
    np.testing.assert_array_equal(expression.data, expected.data)
    np.testing.assert_array_equal(expression.indices, expected.indices)
    np.testing.assert_array_equal(expression.indptr, expected.indptr)


class TestPrepareAnndataDataframe:
    """Two reader-owned annotation contracts share the same storage boundary."""

    @pytest.mark.parametrize("consumer", ["visium", "codex"])
    @pytest.mark.parametrize("storage", ["h5ad", "zarr"])
    def test_reader_annotations_linkage_and_roundtrip(
        self, tmp_path: Path, consumer: str, storage: Literal["h5ad", "zarr"]
    ) -> None:
        """Persist two reader-shaped annotation contracts with exact target linkage."""
        region = "spots" if consumer == "visium" else "cells"
        context = ReaderErrorContext(reader=consumer, artifact="final annotations")
        identifiers = (
            pd.Series(["002-1", "001-1", "003-1"], dtype=pd.ArrowDtype(pa.string())).array
            if consumer == "visium"
            else pd.Series([2**64 - 1, 2**63 + 1, 2**53 + 1], dtype=pd.ArrowDtype(pa.uint64())).array
        )
        obs = pd.DataFrame(
            {
                "instance": identifiers,
                "region": pd.Categorical([region] * 3),
                "in_tissue": pd.Series([True, None, False], dtype="bool[pyarrow]").array,
                "array_row": pd.Series([2, 0, 1], dtype="uint16[pyarrow]").array,
                "quality": pd.Series([127, None, -128], dtype="int8[pyarrow]").array,
                "area": pd.Series([1.25, 2.5, 3.75], dtype="float32[pyarrow]").array,
                "annotation": pd.Series(["  β ", None, "001"], dtype=pd.ArrowDtype(pa.string())).array,
                "compatible_string": pd.Series(["a", None, "c"], dtype=pd.StringDtype(storage="pyarrow")).array,
            },
            index=pd.Index(
                pd.Series(["obs-b", "obs-a", "obs-c"], dtype=pd.ArrowDtype(pa.string())).array, name="barcode"
            ),
        )
        var = pd.DataFrame(
            {
                "feature_id": pd.Series(["0002", "0001"], dtype=pd.ArrowDtype(pa.string())).array,
                "feature_rank": pd.Series([2, 1], dtype="uint32[pyarrow]").array,
            },
            index=pd.Index(pd.Series(["gene_b", "gene_a"], dtype=pd.ArrowDtype(pa.string())).array, name="feature"),
        )
        metadata = pd.DataFrame(
            {"source": pd.Series([consumer], dtype=pd.ArrowDtype(pa.string())).array}, index=["sample"]
        )
        compatible_string = obs["compatible_string"].array

        prepare_anndata_dataframe(obs, context=context)
        prepare_anndata_dataframe(var, context=context)
        prepare_anndata_dataframe(metadata, context=context)

        assert obs["compatible_string"].array is compatible_string
        assert obs["instance"].dtype == (
            pd.StringDtype(storage="pyarrow") if consumer == "visium" else np.dtype(np.uint64)
        )
        assert obs["array_row"].dtype == np.dtype(np.uint16)
        assert obs["area"].dtype == np.dtype(np.float32)
        assert obs["quality"].dtype == pd.Int8Dtype()
        assert obs["in_tissue"].dtype == pd.BooleanDtype()

        target_index = (
            pd.Index(["001-1", "003-1", "002-1"], dtype=pd.StringDtype(storage="pyarrow"))
            if consumer == "visium"
            else pd.Index(np.asarray([2**63 + 1, 2**53 + 1, 2**64 - 1], dtype=np.uint64))
        )
        shapes = ShapesModel.parse(
            gpd.GeoDataFrame(
                {ShapesModel.RADIUS_KEY: [1.0, 1.0, 1.0]},
                geometry=[Point(1, 2), Point(3, 4), Point(5, 6)],
                index=target_index,
            )
        )
        expression = sparse.csr_array(np.asarray([[1, 0], [0, 2], [3, 4]], dtype=np.uint16))
        table = ad.AnnData(expression, obs=obs, var=var, uns={"metadata": metadata})
        result = parse_linked_table(
            table,
            TableLinkage((region,), "region", "instance"),
            context=context,
            expected_instance_ids={region: shapes.index},
        )
        spatial = SpatialData(shapes={region: shapes}, tables={"table": result})

        assert spatial.tables["table"] is table
        assert table.X is expression
        assert isinstance(table.X, sparse.sparray)
        assert not isinstance(table.X, sparse.spmatrix)
        assert set(table.obs["instance"]) == set(shapes.index)

        restored = _roundtrip(table, tmp_path / f"{consumer}.{storage}", storage)

        assert isinstance(restored.obs, pd.DataFrame)
        pd.testing.assert_frame_equal(restored.obs, obs, check_exact=True)
        assert isinstance(restored.var, pd.DataFrame)
        pd.testing.assert_frame_equal(restored.var, var, check_exact=True)
        pd.testing.assert_frame_equal(restored.uns["metadata"], metadata, check_exact=True)
        assert restored.uns[TableModel.ATTRS_KEY] == table.uns[TableModel.ATTRS_KEY]
        _assert_expression(restored, expression)

    @pytest.mark.parametrize("storage", ["h5ad", "zarr"])
    def test_all_integer_widths_and_null_masks_persist(self, tmp_path: Path, storage: Literal["h5ad", "zarr"]) -> None:
        """Round-trip integer extrema and null masks for every supported width."""
        frame = pd.DataFrame(index=["a", "b", "c"])
        for dtype in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
            limits = np.iinfo(dtype)
            values = [limits.max, limits.min, 1]
            frame[dtype] = pd.Series(values, dtype=f"{dtype}[pyarrow]").array
            frame[f"nullable_{dtype}"] = pd.Series([*values[:2], None], dtype=f"{dtype}[pyarrow]").array

        prepare_anndata_dataframe(frame, context=ReaderErrorContext(reader="example", artifact="obs"))
        table = ad.AnnData(obs=frame)

        restored = _roundtrip(table, tmp_path / f"integer_widths.{storage}", storage)

        assert isinstance(restored.obs, pd.DataFrame)
        pd.testing.assert_frame_equal(restored.obs, frame, check_exact=True)

    @pytest.mark.parametrize("storage", ["h5ad", "zarr"])
    def test_default_writer_preserves_nullable_string_values_without_setting_changes(
        self, tmp_path: Path, storage: Literal["h5ad", "zarr"]
    ) -> None:
        """Persist nullable strings through default writers without changing settings."""
        frame = pd.DataFrame(
            {
                "general_arrow": pd.Series(["001", None, "β"], dtype=pd.ArrowDtype(pa.string())).array,
                "arrow_string": pd.Series(["a", None, "c"], dtype=pd.StringDtype(storage="pyarrow")).array,
            },
            index=["a", "b", "c"],
        )

        prepare_anndata_dataframe(frame, context=ReaderErrorContext(reader="example", artifact="obs"))
        table = ad.AnnData(obs=frame)
        path = tmp_path / f"default.{storage}"
        setting = ad.settings.allow_write_nullable_strings

        if storage == "h5ad":
            table.write_h5ad(path)
            restored = ad.read_h5ad(path)
        else:
            table.write_zarr(path)
            restored = ad.read_zarr(path)

        assert ad.settings.allow_write_nullable_strings is setting
        assert isinstance(restored.obs, pd.DataFrame)
        for name in frame:
            np.testing.assert_array_equal(restored.obs[name].isna(), frame[name].isna())
            assert restored.obs[name].dropna().tolist() == frame[name].dropna().tolist()
