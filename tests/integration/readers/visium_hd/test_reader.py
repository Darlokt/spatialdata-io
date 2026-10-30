"""Synthetic public-reader contracts for supported Space Ranger layouts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

import anndata as ad
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from anndata.tests.helpers import assert_equal
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_array, csr_matrix, spmatrix
from spatialdata import read_zarr
from spatialdata.transformations import get_transformation

from spatialdata_io import visium_hd
from spatialdata_io.readers._utils.errors import ReaderFormatError
from tests.integration.readers.visium_hd._fixtures import CELL_IDS, write_bundle

if TYPE_CHECKING:
    from pathlib import Path


def _invalidate(root: Path, kind: str, ids: tuple[int, ...]) -> None:
    path = root / f"segmented_outputs/{kind}_segmentations.geojson"
    # This is fixture-owned JSON; its shape is already specified by write_bundle.
    document = json.loads(path.read_text())
    for feature in document["features"]:
        if feature["properties"]["cell_id"] in ids:
            feature["geometry"]["coordinates"] = [[[7, 17], [13, 23], [7, 23], [13, 17], [7, 17]]]
    path.write_text(json.dumps(document))


class TestVisiumHD:
    """Supported layouts and linked products agree with independent source truth."""

    @pytest.mark.parametrize(
        ("version", "segmented", "sizes", "prefix", "feature_slice_name", "null_pair"),
        [
            ("3.0.0", False, (2, 8, 16), "", "hd_feature_slice.h5", None),
            ("4.0.1", True, (2, 8, 16), "", "feature_slice.h5", None),
            ("4.1.0", True, (2, 6), "", "feature_slice.h5", None),
            ("4.1.0", False, (2, 8), "sample_", "feature_slice.h5", None),
            ("4.1.0", False, (2, 8), "", "feature_slice.h5", "microscope"),
            ("4.1.0", False, (2, 8), "", "feature_slice.h5", "cytassist"),
        ],
    )
    def test_layouts(  # noqa: PLR0913, RUF100 - Parametrized layout and version contract.
        self,
        tmp_path: Path,
        *,
        version: str,
        segmented: bool,
        sizes: tuple[int, ...],
        prefix: str,
        feature_slice_name: Literal["feature_slice.h5", "hd_feature_slice.h5"],
        null_pair: Literal["microscope", "cytassist"] | None,
    ) -> None:
        bundle = write_bundle(
            tmp_path,
            version=version,
            segmented=segmented,
            sizes=sizes,
            prefix=prefix,
            feature_slice_name=feature_slice_name,
            null_pair=null_pair,
        )
        result = visium_hd(bundle.root, include_images=False)
        bin_names = {f"square_{size:03d}um" for size in sizes}
        assert set(result.tables) == {f"{name}_table" for name in bin_names} | ({"cell_table"} if segmented else set())
        assert set(result.shapes) == bin_names | ({"cell_boundaries", "nucleus_boundaries"} if segmented else set())
        assert result.coordinate_systems == ["global"]
        for size in sizes:
            table = result.tables[f"square_{size:03d}um_table"]
            assert isinstance(table.X, csr_array)
            assert not isinstance(table.X, spmatrix)
            assert_array_equal(table.X.toarray(), bundle.raw[size][bundle.filtered_rows[size]])
            assert table.var_names.tolist() == ["Gene", "Gene-1", "Other"]
            assert table.var["genome"].tolist() == ["mouse", "human", "mouse"]
            assert table.var["feature_types"].tolist() == ["Gene Expression"] * 3
            assert table.obsm["spatial"].dtype == np.float64
            assert table.uns["spatialdata_io_positions"]["coordinates"]["spatial"]["position"] == "nominal bin centre"
            transforms = get_transformation(result.shapes[f"square_{size:03d}um"], get_all=True)
            assert isinstance(transforms, dict)
            assert set(transforms) == {"global"}
        if segmented:
            table = result.tables["cell_table"]
            assert table.obs_names.tolist() == [CELL_IDS[1], CELL_IDS[0]]
            assert isinstance(table.X, csr_array)
            assert_array_equal(table.X.toarray(), bundle.cells[[1, 0]])

    @pytest.mark.parametrize("kind", ["raw", "filtered"])
    def test_nucleus_population_and_raw_counts(self, tmp_path: Path, kind: Literal["raw", "filtered"]) -> None:
        bundle = write_bundle(tmp_path)
        result = visium_hd(
            bundle.root, bin_sizes=(), include_images=False, include_nucleus_table=True, cell_counts_kind=kind
        )
        indices = [0, 1, 2] if kind == "raw" else [1, 0]
        table = result.tables["nucleus_table"]
        assert table.obs_names.tolist() == [CELL_IDS[i] for i in indices]
        assert isinstance(table.X, csr_array)
        assert_array_equal(table.X.toarray(), bundle.nuclei[indices])
        cell_x = result.tables["cell_table"].X
        assert isinstance(cell_x, csr_array)
        assert_array_equal(cell_x.toarray(), bundle.cells[indices])

    def test_raw_bin_order_and_selection(self, tmp_path: Path) -> None:
        bundle = write_bundle(tmp_path, segmented=False)
        result = visium_hd(bundle.root, bin_sizes=(8,), bin_counts_kind="raw", include_images=False)
        assert set(result.tables) == {"square_008um_table"}
        table = result.tables["square_008um_table"]
        assert table.obs_names[:2].tolist() == ["s_008um_00000_00000-1", "s_008um_00000_00001-1"]
        assert isinstance(table.X, csr_array)
        assert_array_equal(table.X.toarray(), bundle.raw[8])
        # Edge squares retain the nominal full bin width.
        assert_allclose(result.shapes["square_008um"].geometry.iloc[-1].area, 64.0, rtol=1e-10)

    @pytest.mark.parametrize("kind", ["filtered", "raw"])
    @pytest.mark.parametrize("size", [6, 8, 16])
    def test_contradictory_bin_registration(self, tmp_path: Path, kind: Literal["filtered", "raw"], size: int) -> None:
        bundle = write_bundle(tmp_path, sizes=(2, 6, 8, 16), segmented=False)
        path = tmp_path / f"binned_outputs/square_{size:03d}um/spatial/tissue_positions.parquet"
        source = pq.read_table(path)
        values = source["pxl_col_in_fullres"].to_numpy().copy()
        # This partial last-row bin is absent from every filtered fixture matrix.
        values[-2] += 1_000_000
        pq.write_table(source.set_column(5, source.schema.field(5), pa.array(values)), path)
        with pytest.raises(ReaderFormatError, match="cross-bin registration residual"):
            visium_hd(bundle.root, bin_sizes=(size,), bin_counts_kind=kind, include_images=False)

    def test_round_trip(self, tmp_path: Path) -> None:
        bundle = write_bundle(tmp_path / "source")
        result = visium_hd(bundle.root, bin_sizes=(8,), include_images=False, include_nucleus_table=True)
        destination = tmp_path / "output.zarr"
        result.write(destination)
        restored = read_zarr(destination)
        assert set(restored.tables) == set(result.tables)
        assert set(restored.shapes) == set(result.shapes)
        for name, table in result.tables.items():
            assert_equal(
                restored.tables[name].uns["spatialdata_io_positions"], table.uns["spatialdata_io_positions"], exact=True
            )
            for storage in ("h5ad", "zarr"):
                table_path = tmp_path / f"positions{name}.{storage}"
                source_table = table
                if storage == "h5ad":
                    source_table.write_h5ad(table_path)
                    restored_positions = ad.read_h5ad(table_path)
                else:
                    source_table.write_zarr(table_path)
                    restored_positions = ad.read_zarr(table_path)
                assert_equal(
                    restored_positions.uns["spatialdata_io_positions"],
                    source_table.uns["spatialdata_io_positions"],
                    exact=True,
                )
                assert_equal(restored_positions.obsm, source_table.obsm, exact=True)
            assert isinstance(table.X, csr_array)
            restored_x = restored.tables[name].X
            assert isinstance(restored_x, (csr_array, csr_matrix))
            assert_array_equal(restored_x.toarray(), table.X.toarray())
            assert restored.tables[name].obs_names.equals(table.obs_names)

    def test_bins_without_feature_slice(self, tmp_path: Path) -> None:
        bundle = write_bundle(tmp_path, segmented=False)
        (tmp_path / "feature_slice.h5").unlink()
        result = visium_hd(bundle.root, bin_sizes=(8,), include_images=False)
        table = result.tables["square_008um_table"]
        assert isinstance(table.X, csr_array)
        assert_array_equal(table.X.toarray(), bundle.raw[8][bundle.filtered_rows[8]])

    def test_empty_product_selection(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, segmented=False)
        with pytest.raises(ValueError, match="at least one available product"):
            visium_hd(tmp_path, bin_sizes=(), include_images=False)

    def test_drop_cell_propagates_to_both_tables_and_nucleus(self, tmp_path: Path) -> None:
        bundle = write_bundle(tmp_path)
        _invalidate(tmp_path, "cell", (2,))
        result = visium_hd(
            tmp_path, bin_sizes=(), include_images=False, include_nucleus_table=True, invalid_geometries="drop"
        )
        assert result.shapes["cell_boundaries"].index.tolist() == [1]
        assert result.shapes["nucleus_boundaries"].index.tolist() == [1]
        assert result.tables["cell_table"].obs_names.tolist() == [CELL_IDS[0]]
        assert result.tables["nucleus_table"].obs_names.tolist() == [CELL_IDS[0]]
        cell_x, nucleus_x = result.tables["cell_table"].X, result.tables["nucleus_table"].X
        assert isinstance(cell_x, csr_array)
        assert isinstance(nucleus_x, csr_array)
        np.testing.assert_array_equal(cell_x.toarray(), bundle.cells[[0]])
        np.testing.assert_array_equal(nucleus_x.toarray(), bundle.nuclei[[0]])
        assert result.tables["cell_table"].obsm["spatial"].shape == (1, 2)

    def test_drop_nucleus_preserves_cell_population(self, tmp_path: Path) -> None:
        write_bundle(tmp_path)
        _invalidate(tmp_path, "nucleus", (1,))
        result = visium_hd(
            tmp_path, bin_sizes=(), include_images=False, include_nucleus_table=True, invalid_geometries="drop"
        )
        assert result.shapes["cell_boundaries"].index.tolist() == [2, 1]
        assert result.shapes["nucleus_boundaries"].index.tolist() == [2]
        assert result.tables["cell_table"].obs_names.tolist() == [CELL_IDS[1], CELL_IDS[0]]
        assert result.tables["nucleus_table"].obs_names.tolist() == [CELL_IDS[1]]

    def test_default_keep_and_raise(self, tmp_path: Path) -> None:
        write_bundle(tmp_path)
        _invalidate(tmp_path, "cell", (1,))
        kept = visium_hd(tmp_path, bin_sizes=(), include_images=False)
        assert not kept.shapes["cell_boundaries"].loc[1].geometry.is_valid
        assert kept.tables["cell_table"].obs_names.tolist() == [CELL_IDS[1], CELL_IDS[0]]
        with pytest.raises(ReaderFormatError, match="invalid"):
            visium_hd(tmp_path, bin_sizes=(), include_images=False, invalid_geometries="raise")

    def test_drop_rejects_empty_linkage(self, tmp_path: Path) -> None:
        write_bundle(tmp_path)
        _invalidate(tmp_path, "cell", (1, 2, 3))
        with pytest.raises(ReaderFormatError, match="empty boundary selection"):
            visium_hd(tmp_path, bin_sizes=(), include_images=False, invalid_geometries="drop")

    def test_missing_source_polygon_raises_before_linkage(self, tmp_path: Path) -> None:
        write_bundle(tmp_path)
        path = tmp_path / "segmented_outputs/cell_segmentations.geojson"
        document = json.loads(path.read_text())
        document["features"] = document["features"][1:]
        path.write_text(json.dumps(document))
        with pytest.raises(ReaderFormatError, match="matrix IDs absent from polygons"):
            visium_hd(tmp_path, bin_sizes=(), include_images=False)

    def test_nucleus_without_centroid_does_not_inherit_cell_positions(self, tmp_path: Path) -> None:
        write_bundle(tmp_path)
        path = tmp_path / "segmented_outputs/nucleus_segmentations.geojson"
        document = json.loads(path.read_text())
        for feature in document["features"]:
            feature["properties"].pop("nucleus_centroid")
        path.write_text(json.dumps(document))
        result = visium_hd(tmp_path, bin_sizes=(), include_images=False, include_nucleus_table=True)
        assert "spatial" in result.tables["cell_table"].obsm
        assert "spatialdata_io_positions" in result.tables["cell_table"].uns
        assert "spatial" not in result.tables["nucleus_table"].obsm
        assert "spatialdata_io_positions" not in result.tables["nucleus_table"].uns

    def test_nuclei_only_and_older_centroid_absence(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, version="4.0.1")
        result = visium_hd(tmp_path, bin_sizes=(), include_images=False, include_cell_boundaries=False)
        assert set(result.shapes) == {"nucleus_boundaries"}
        assert not result.tables
        assert result.shapes["nucleus_boundaries"].index.tolist() == [1, 2, 3]
        result = visium_hd(tmp_path, bin_sizes=(), include_images=False, include_nucleus_boundaries=False)
        assert set(result.tables) == {"cell_table"}
        assert "spatial" not in result.tables["cell_table"].obsm
