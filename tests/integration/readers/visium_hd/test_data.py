"""Independent sparse and geometry oracles for the optional public tiny bundle."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import TYPE_CHECKING, Literal

import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_array
from shapely.geometry import Polygon

from spatialdata_io import visium_hd

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pandas as pd
    from numpy.typing import NDArray

pytestmark = [pytest.mark.data, pytest.mark.slow]


def _source_counts(path: Path) -> tuple[csr_array, list[str]]:
    with h5py.File(path, "r") as source:
        matrix = source["matrix"]
        shape = tuple(int(value) for value in matrix["shape"][:])
        counts = csr_array((matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]), shape=(shape[1], shape[0]))
        names = matrix["barcodes"].asstr()[:].tolist()
    return counts, names


def _nucleus_oracle(root: Path, names: pd.Index[str], nvars: int) -> csr_array:
    # Read only the tiny numeric sparse buffers and pointer endpoints needed by
    # projected nuclear mappings. Python integers make summation independent of
    # the implementation's sparse multiplication and accumulation dtype.
    mapping = root / "barcode_mappings.parquet"
    if not mapping.exists():
        mapping = root / "segmented_outputs/barcode_mappings.parquet"
    output_rows = {name: i for i, name in enumerate(names)}
    entries: defaultdict[tuple[int, int], int] = defaultdict(int)
    with (
        h5py.File(root / "binned_outputs/square_002um/raw_feature_bc_matrix.h5", "r") as source,
        pq.ParquetFile(mapping) as mappings,
    ):
        matrix = source["matrix"]
        data, features = matrix["data"][:], matrix["indices"][:]
        for batch in mappings.iter_batches(batch_size=65_536, columns=["square_002um", "cell_id", "in_nucleus"]):
            selected = batch.filter(batch.column("in_nucleus"))
            barcodes = selected.column("square_002um").to_pylist()
            cells = selected.column("cell_id").to_pylist()
            indices = np.array([int(name[8:13]) * 3350 + int(name[14:19]) for name in barcodes], dtype=np.int64)
            if not len(indices):
                continue
            order = np.argsort(indices)
            starts = matrix["indptr"][indices[order]]
            ends = matrix["indptr"][indices[order] + 1]
            for position, start, end in zip(order, starts, ends, strict=True):
                target = output_rows.get(cells[position])
                if target is None:
                    continue
                for i in range(int(start), int(end)):
                    entries[target, int(features[i])] += int(data[i])
    rows: list[int] = []
    cols: list[int] = []
    values: list[int] = []
    for (row, col), value in entries.items():
        rows.append(row)
        cols.append(col)
        values.append(value)
    return csr_array((np.array(values, dtype=np.int64), (rows, cols)), shape=(len(names), nvars))


class TestVisiumHDData:
    """Exact source counts and geometry on the public tiny bundle."""

    @pytest.mark.parametrize("kind", ["raw", "filtered"])
    def test_bin_counts_and_order(
        self, require_test_dataset: Callable[[str], Path], kind: Literal["raw", "filtered"]
    ) -> None:
        root = require_test_dataset("visium_hd_tiny")
        result = visium_hd(
            root,
            bin_sizes=(16,),
            bin_counts_kind=kind,
            include_cell_boundaries=False,
            include_nucleus_boundaries=False,
            include_images=False,
        )
        expected, names = _source_counts(root / f"binned_outputs/square_016um/{kind}_feature_bc_matrix.h5")
        table = result.tables["square_016um_table"]
        assert isinstance(table.X, csr_array)
        assert table.obs_names.tolist() == names
        assert (table.X.astype(np.int64) - expected.astype(np.int64)).nnz == 0
        assert table.shape == ((175561 if kind == "raw" else 5262), 32285)
        assert table.var["gene_ids"].is_unique
        assert table.var["feature_types"].unique().tolist() == ["Gene Expression"]
        assert result.coordinate_systems == ["global"]

    def test_nucleus_counts_and_polygon_vertices(self, require_test_dataset: Callable[[str], Path]) -> None:
        root = require_test_dataset("visium_hd_tiny")
        result = visium_hd(root, bin_sizes=(8,), include_images=False, include_nucleus_table=True)
        cells, names = _source_counts(root / "segmented_outputs/filtered_feature_cell_matrix.h5")
        table = result.tables["cell_table"]
        assert table.obs_names.tolist() == names
        assert isinstance(table.X, csr_array)
        assert (table.X.astype(np.int64) - cells.astype(np.int64)).nnz == 0
        nucleus = result.tables["nucleus_table"]
        assert isinstance(nucleus.X, csr_array)
        expected = _nucleus_oracle(root, nucleus.obs_names, nucleus.n_vars)
        assert (nucleus.X.astype(np.int64) - expected).nnz == 0
        assert nucleus.obs_names.equals(table.obs_names)
        assert nucleus.shape == (612, 32285)
        bins = result.tables["square_008um_table"]
        grid = np.column_stack(
            (
                8 * (bins.obs["array_col"].to_numpy() + 0.5),
                8 * (bins.obs["array_row"].to_numpy() + 0.5),
                np.ones(bins.n_obs),
            )
        )
        spatial = bins.obsm["spatial"]
        assert isinstance(spatial, np.ndarray)
        rigid = np.linalg.lstsq(grid, spatial, rcond=None)[0]
        affine = np.column_stack((rigid, [0.0, 0.0, 1.0]))
        with h5py.File(root / "feature_slice.h5", "r") as source:
            metadata = json.loads(source.attrs["metadata_json"])
        homography = np.array(metadata["transform_matrices"]["spot_colrow_to_microscope_colrow"]).reshape(3, 3)
        for kind in ("cell", "nucleus"):
            source_features = json.loads((root / f"segmented_outputs/{kind}_segmentations.geojson").read_text())[
                "features"
            ]
            expected_rings = {
                feature["properties"]["cell_id"]: feature["geometry"]["coordinates"][0] for feature in source_features
            }
            for identifier, polygon in result.shapes[f"{kind}_boundaries"].geometry.items():
                assert isinstance(polygon, Polygon)
                vertices: NDArray[np.float64] = np.array(polygon.exterior.coords)
                physical = np.column_stack((vertices, np.ones(len(vertices)))) @ np.linalg.inv(affine)
                spot = physical[:, :2] / 2 - 0.5
                pixels = np.column_stack((spot, np.ones(len(spot)))) @ homography.T
                assert_allclose(pixels[:, :2] / pixels[:, 2:], expected_rings[identifier], atol=2e-9, rtol=0)
            assert_array_equal(result.shapes[f"{kind}_boundaries"].index, table.obs["instance_id"])
