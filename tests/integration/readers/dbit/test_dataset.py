"""Validate experimental DBiT-seq counts with the authors' actual barcode map."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from spatialdata.models import get_table_keys

from spatialdata_io.readers.dbit import dbit

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _prepare_layout(source_root: Path, destination: Path) -> ad.AnnData:
    """Restore nucleotide identifiers in the published coordinate-indexed H5AD."""
    table = ad.read_h5ad(source_root / "source.h5ad")
    barcode_map = pd.read_csv(
        source_root / "barcode_coordinates.tsv",
        sep="\t",
        header=None,
        names=["barcode", "B", "A"],
        dtype={"barcode": str, "B": np.int64, "A": np.int64},
    )
    assert barcode_map.shape == (2500, 3)
    assert barcode_map["barcode"].str.fullmatch("[ACGT]{16}").all()

    barcode_map["source_id"] = barcode_map["B"].astype(str) + "x" + barcode_map["A"].astype(str)
    lookup = barcode_map.set_index("source_id")["barcode"]
    assert lookup.index.is_unique
    assert table.obs_names.isin(lookup.index).all()

    coordinates = np.asarray([name.split("x") for name in table.obs_names], dtype=np.int64)
    np.testing.assert_array_equal(table.obsm["spatial"], coordinates)

    # The published mapping's first coordinate indexes the first eight bases
    # (B), and its second indexes the last eight (A). Retain that source frame.
    table.obs["source_coordinate_id"] = table.obs_names
    table.obsm["source_spatial"] = table.obsm["spatial"]
    table.obs_names = lookup.loc[table.obs_names].tolist()

    # Preserve the archived raw counts and all features. The reader owns the
    # conversion from the source's dense expression to canonical sparse output.
    counts = table.X
    assert isinstance(counts, np.ndarray)
    assert counts.shape == (936, 22802)
    assert counts.dtype == np.float32

    destination.mkdir()
    table.write_h5ad(destination / "mouse_e10.h5ad")

    # Collapse the full 50-by-50 pairing map into its two original barcode axes.
    axis_lines: list[str] = []
    for axis, selection in (("A", slice(8, 16)), ("B", slice(0, 8))):
        axis_map = barcode_map[[axis, "barcode"]].copy()
        axis_map["barcode"] = axis_map["barcode"].str.slice(selection.start, selection.stop)
        axis_map = axis_map.drop_duplicates().sort_values(axis)
        assert axis_map[axis].tolist() == list(range(1, 51))
        axis_lines.extend(f"{axis}{position}\t{sequence}\n" for position, sequence in axis_map.itertuples(index=False))

    (destination / "barcode_list.txt").write_text("".join(axis_lines), encoding="utf-8")
    return table


class TestDbitDataset:
    """Exercise experimental values without inventing source identifiers."""

    def test_mouse_e10_experimental_counts(self, require_test_dataset: Callable[[str], Path], tmp_path: Path) -> None:
        source_root = require_test_dataset("dbit_mouse_e10")
        prepared_root = tmp_path / "prepared"
        source = _prepare_layout(source_root, prepared_root)

        result = dbit(prepared_root)

        assert not result.images
        assert list(result.shapes) == ["mouse_e10"]
        assert list(result.tables) == ["table"]

        table = result.tables["table"]
        assert isinstance(table.X, sparse.csr_array)
        assert not isinstance(table.X, sparse.spmatrix)
        assert table.X.has_canonical_format
        assert table.shape == (936, 22802)
        assert table.X.dtype == np.float32

        source_counts = source.X
        assert isinstance(source_counts, np.ndarray)
        order = source.obs_names.get_indexer(table.obs_names)
        expected_counts = sparse.csr_array(source_counts)[order]
        np.testing.assert_array_equal(table.X.data, expected_counts.data)
        np.testing.assert_array_equal(table.X.indices, expected_counts.indices)
        np.testing.assert_array_equal(table.X.indptr, expected_counts.indptr)

        assert isinstance(table.var, pd.DataFrame)
        assert isinstance(source.var, pd.DataFrame)
        pd.testing.assert_frame_equal(table.var, source.var)

        assert isinstance(table.obs, pd.DataFrame)
        assert isinstance(source.obs, pd.DataFrame)
        pd.testing.assert_series_equal(table.obs["leiden"], source.obs["leiden"].iloc[order])
        assert table.uns["hvg"] == source.uns["hvg"] == {"flavor": "seurat"}
        pd.testing.assert_frame_equal(table.uns["moranI"], source.uns["moranI"])
        np.testing.assert_array_equal(table.obsm["X_pca"], source.obsm["X_pca"][order])
        np.testing.assert_array_equal(table.varm["PCs"], source.varm["PCs"])

        original_coordinates = source.obsm["source_spatial"]
        assert isinstance(original_coordinates, np.ndarray)
        np.testing.assert_array_equal(table.obsm["spatial_grid"], original_coordinates[order, ::-1])
        assert table.obs["source_coordinate_id"].tolist() == source.obs["source_coordinate_id"].iloc[order].tolist()

        assert set(table.obs["pixel_id"]) == set(result.shapes["mouse_e10"].index)
        assert get_table_keys(table) == ("mouse_e10", "region", "pixel_id")
