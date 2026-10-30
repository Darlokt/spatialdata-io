"""Registered company and study counts retain exact sparse values and source identity."""

from __future__ import annotations

from shutil import copyfile
from typing import TYPE_CHECKING, cast

import h5py
import numpy as np
import pytest
import tifffile
import zarr
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from scipy.sparse import csc_array, csr_array
from spatialdata.transformations import Scale, get_transformation
from xarray import DataArray

from spatialdata_io import stereoseq
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.image.tiff import _dask as tiff_dask

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy.typing as npt
    from anndata import AnnData
    from pytest_mock import MockerFixture
    from spatialdata import SpatialData

pytestmark = [pytest.mark.data, pytest.mark.slow]
_ORACLE_RECORD_BYTES = 2**23


def _forbid_densification(mocker: MockerFixture) -> None:
    for sparse_type in (csr_array, csc_array):
        mocker.patch.object(sparse_type, "toarray", side_effect=AssertionError("Expression must remain sparse."))


def _assert_native_cells(table: AnnData, source: Path) -> None:
    """Compare every native cell span and exon value with source-indexed CSR rows."""
    matrix, exons = table.X, table.layers["exon"]
    assert isinstance(matrix, csr_array)
    assert isinstance(exons, csr_array)
    assert matrix.dtype == exons.dtype == np.uint64
    assert matrix.has_canonical_format
    assert exons.has_canonical_format
    source_total = exon_total = 0
    with h5py.File(source, "r") as handle:
        group = handle["cellBin"]
        cells = group["cell"][:]
        genes = group["gene"][:]
        expression = group["cellExp"]
        source_exons = group["cellExpExon"]
        assert matrix.shape == (len(cells), len(genes))
        assert matrix.nnz == len(expression)
        assert table.obs_names.tolist() == [str(value) for value in cells["id"]]
        identity_field = "geneID" if "geneID" in genes.dtype.names else "geneName"
        assert table.var_names.tolist() == [bytes(value).decode("utf-8") for value in genes[identity_field]]
        np.testing.assert_array_equal(table.obs["instance_id"], cells["id"])
        np.testing.assert_array_equal(table.obsm["spatial_grid"], np.column_stack((cells["x"], cells["y"])))
        np.testing.assert_array_equal(table.obs["cellExon"], group["cellExon"][:])
        np.testing.assert_array_equal(table.var["geneExon"], group["geneExon"][:])
        assert (
            int(cells["geneCount"].max()) * (expression.dtype.itemsize + source_exons.dtype.itemsize)
            < _ORACLE_RECORD_BYTES
        )
        for row, cell in enumerate(cells):
            start, stop = int(cell["offset"]), int(cell["offset"]) + int(cell["geneCount"])
            records = expression[start:stop]
            order = np.argsort(records["geneID"], kind="stable")
            output_start, output_stop = matrix.indptr[row : row + 2]
            exon_start, exon_stop = exons.indptr[row : row + 2]
            np.testing.assert_array_equal(matrix.indices[output_start:output_stop], records["geneID"][order])
            np.testing.assert_array_equal(matrix.data[output_start:output_stop], records["count"][order])
            np.testing.assert_array_equal(exons.indices[exon_start:exon_stop], records["geneID"][order])
            np.testing.assert_array_equal(exons.data[exon_start:exon_stop], source_exons[start:stop][order])
            source_total += sum(int(value) for value in records["count"])
            exon_total += sum(int(value) for value in source_exons[start:stop])
    assert source_total == int(matrix.sum())
    assert exon_total == int(exons.sum())


def _assert_square_column(  # noqa: PLR0913, RUF100 - one assertion over paired source columns.
    counts: csc_array,
    exons: csc_array,
    coordinates: npt.NDArray[np.int64],
    feature: int,
    *,
    records: npt.NDArray[np.void],
    source_exons: npt.NDArray[np.uint8],
) -> int:
    """Group one bounded feature by coordinate using an independent exact sort oracle."""
    start, stop = counts.indptr[feature : feature + 2]
    exon_start, exon_stop = exons.indptr[feature : feature + 2]
    if len(records) == 0:
        assert start == stop
        assert exon_start == exon_stop
        return 0
    source_order = np.lexsort((records["y"], records["x"]))
    sorted_coordinates = np.column_stack((records["x"][source_order], records["y"][source_order]))
    starts = np.r_[0, np.flatnonzero(np.any(sorted_coordinates[1:] != sorted_coordinates[:-1], axis=1)) + 1]
    expected_coordinates = sorted_coordinates[starts]
    # This source has uint8 values and fewer than 2**32 records: any duplicate sum is <2**40.
    expected_counts = np.add.reduceat(records["count"][source_order].astype(np.uint64), starts)
    expected_exons = np.add.reduceat(source_exons[source_order].astype(np.uint64), starts)
    output_coordinates = coordinates[counts.indices[start:stop]]
    output_order = np.lexsort((output_coordinates[:, 1], output_coordinates[:, 0]))
    exon_coordinates = coordinates[exons.indices[exon_start:exon_stop]]
    exon_order = np.lexsort((exon_coordinates[:, 1], exon_coordinates[:, 0]))
    np.testing.assert_array_equal(output_coordinates[output_order], expected_coordinates)
    np.testing.assert_array_equal(counts.data[start:stop][output_order], expected_counts)
    np.testing.assert_array_equal(exon_coordinates[exon_order], expected_coordinates)
    np.testing.assert_array_equal(exons.data[exon_start:exon_stop][exon_order], expected_exons)
    return len(records) - len(starts)


def _assert_native_square(table: AnnData, source: Path) -> None:
    """Check all source features with a bounded source batch per feature, without densification."""
    assert isinstance(table.X, csr_array)
    assert isinstance(table.layers["exon"], csr_array)
    assert table.X.dtype == table.layers["exon"].dtype == np.uint64
    assert table.X.has_canonical_format
    counts, exons = csc_array(table.X), csc_array(table.layers["exon"])
    coordinates = np.asarray(table.obsm["spatial_grid"])
    assert coordinates.dtype == np.int64
    assert table.shape == (12604918, 45537)
    assert counts.nnz == 18829578
    total = exon_total = duplicate_records = checked_records = 0
    with h5py.File(source, "r") as handle:
        group = handle["geneExp/bin1"]
        genes = group["gene"][:]
        expression, source_exons = group["expression"], group["exon"]
        assert expression.dtype["count"] == source_exons.dtype == np.dtype(np.uint8)
        assert len(expression) < 2**32
        assert (
            int(genes["count"].max()) * (expression.dtype.itemsize + source_exons.dtype.itemsize) < _ORACLE_RECORD_BYTES
        )
        assert table.var_names.tolist() == [bytes(value).decode("utf-8") for value in genes["gene"]]
        for feature, gene in enumerate(genes):
            start, stop = int(gene["offset"]), int(gene["offset"]) + int(gene["count"])
            records = cast("npt.NDArray[np.void]", expression[start:stop])
            exon_records = cast("npt.NDArray[np.uint8]", source_exons[start:stop])
            duplicate_records += _assert_square_column(
                counts, exons, coordinates, feature, records=records, source_exons=exon_records
            )
            total += sum(int(value) for value in records["count"])
            exon_total += sum(int(value) for value in exon_records)
            checked_records += len(records)
    assert checked_records == 18829608
    assert duplicate_records == 30
    assert total == int(counts.sum()) == 28085627
    assert exon_total == int(exons.sum())
    assert table.obs_names[0] == "0"
    assert table.obs_names[-1] == "12604917"
    for start in range(0, table.n_obs, 1000000):
        stop = min(start + 1000000, table.n_obs)
        np.testing.assert_array_equal(table.obs["instance_id"].iloc[start:stop], np.arange(start, stop))


def _write_square_prefix(source: Path, destination: Path) -> None:
    """Copy complete leading feature spans, capped at 24 features and 50,000 events.

    Original downloads remain untouched. Native dtypes, root metadata, source
    extents, physical pitch, expression attributes and paired exons are retained.
    This deliberately tests a reduced product, not whole-file reader execution.
    """
    with h5py.File(source, "r") as original, h5py.File(destination, "w") as reduced:
        group = original["geneExp/bin1"]
        genes = group["gene"][:24]
        ends = genes["offset"].astype(np.uint64) + genes["count"]
        genes = genes[ends <= 50000]
        assert len(genes) > 0
        stop = int(genes[-1]["offset"]) + int(genes[-1]["count"])

        reduced.attrs.update(original.attrs)
        target = reduced.create_group("geneExp/bin1")
        target.create_dataset("gene", data=genes)
        expression = target.create_dataset("expression", data=group["expression"][:stop])
        expression.attrs.update(group["expression"].attrs)
        target.create_dataset("exon", data=group["exon"][:stop])


def _assert_square_prefix(table: AnnData, source: Path) -> None:
    """Compare the reduced output directly with original per-feature source spans."""
    assert isinstance(table.X, csr_array)
    assert isinstance(table.layers["exon"], csr_array)
    assert table.X.dtype == table.layers["exon"].dtype == np.uint64
    assert table.X.has_canonical_format
    assert table.layers["exon"].has_canonical_format
    counts, exons = csc_array(table.X), csc_array(table.layers["exon"])
    coordinates = np.asarray(table.obsm["spatial_grid"])

    expected_coordinates: dict[tuple[int, int], None] = {}
    total = exon_total = record_count = 0
    with h5py.File(source, "r") as original:
        group = original["geneExp/bin1"]
        genes = group["gene"][: table.n_vars]
        identity_field = "geneID" if "geneID" in genes.dtype.names else "gene"
        assert table.var_names.tolist() == [bytes(value).decode("utf-8") for value in genes[identity_field]]
        if "geneName" in genes.dtype.names:
            assert table.var["geneName"].tolist() == [bytes(value).decode("utf-8") for value in genes["geneName"]]

        for feature, gene in enumerate(genes):
            start, stop = int(gene["offset"]), int(gene["offset"]) + int(gene["count"])
            records = cast("npt.NDArray[np.void]", group["expression"][start:stop])
            source_exons = cast("npt.NDArray[np.uint8]", group["exon"][start:stop])
            assert records.nbytes + source_exons.nbytes < _ORACLE_RECORD_BYTES
            _assert_square_column(counts, exons, coordinates, feature, records=records, source_exons=source_exons)
            expected_coordinates.update(((int(row["x"]), int(row["y"])), None) for row in records)
            total += sum(int(value) for value in records["count"])
            exon_total += sum(int(value) for value in source_exons)
            record_count += len(records)

    assert record_count <= 50000
    np.testing.assert_array_equal(coordinates, list(expected_coordinates))
    assert coordinates.dtype == np.int64
    np.testing.assert_array_equal(table.obs["instance_id"], np.arange(table.n_obs))
    assert table.obs_names.tolist() == [str(value) for value in range(table.n_obs)]
    assert int(counts.sum()) == total
    assert int(exons.sum()) == exon_total
    assert table.uns["stereoseq"]["origin"] == [0, 0]
    assert table.uns["stereoseq"]["resolution"] == 500


def _assert_layout_equivalent(actual: SpatialData, expected: SpatialData) -> None:
    """Compare Stereo-seq directory discovery with the same explicit flat products."""
    assert set(actual.tables) == set(expected.tables)
    assert set(actual.points) == set(expected.points)
    assert set(actual.images) == set(expected.images)
    assert not actual.shapes
    assert not expected.shapes

    for name, table in actual.tables.items():
        reference = expected.tables[name]
        assert table.shape == reference.shape
        assert isinstance(table.obs, DataFrame)
        assert isinstance(reference.obs, DataFrame)
        assert isinstance(table.var, DataFrame)
        assert isinstance(reference.var, DataFrame)
        assert_frame_equal(table.obs, reference.obs)
        assert_frame_equal(table.var, reference.var)
        np.testing.assert_array_equal(table.obsm["spatial_grid"], reference.obsm["spatial_grid"])
        assert table.uns["stereoseq"] == reference.uns["stereoseq"]
        assert table.uns["spatialdata_attrs"] == reference.uns["spatialdata_attrs"]
        # AnnData also exposes X through its None layer key; X is compared below.
        assert {name for name in table.layers if name is not None} == {"exon"}
        assert set(table.layers) == set(reference.layers)

        for matrix, source in ((table.X, reference.X), (table.layers["exon"], reference.layers["exon"])):
            assert isinstance(matrix, csr_array)
            assert isinstance(source, csr_array)
            assert matrix.dtype == source.dtype == np.uint64
            np.testing.assert_array_equal(matrix.data, source.data)
            np.testing.assert_array_equal(matrix.indices, source.indices)
            np.testing.assert_array_equal(matrix.indptr, source.indptr)

    for name, points in actual.points.items():
        reference_points = expected.points[name]
        assert_frame_equal(points.compute(), reference_points.compute())
        for target in (points, reference_points):
            transformations = get_transformation(target, get_all=True)
            assert isinstance(transformations, dict)
            assert set(transformations) == {"global"}
            assert get_transformation(target, to_coordinate_system="global") == Scale([0.5, 0.5], axes=("x", "y"))

    for name, image in actual.images.items():
        reference_image = expected.images[name]
        assert isinstance(image, DataArray)
        assert isinstance(reference_image, DataArray)
        assert image.dims == reference_image.dims == ("c", "y", "x")
        assert image.shape == reference_image.shape
        assert image.dtype == reference_image.dtype
        np.testing.assert_array_equal(image.coords["c"], reference_image.coords["c"])
        for raster in (image, reference_image):
            transformations = get_transformation(raster, get_all=True)
            assert isinstance(transformations, dict)
            assert set(transformations) == {"global"}
            assert get_transformation(raster, to_coordinate_system="global") == Scale([0.5, 0.5], axes=("x", "y"))


class TestStereoseqData:
    """Manifest-managed originals and bounded reductions exercise public-reader workflows."""

    def test_vendor_cells_and_registered_image(
        self, require_test_dataset: Callable[[str], Path], mocker: MockerFixture
    ) -> None:
        root = require_test_dataset("stereoseq_mouse_eyeball")
        _forbid_densification(mocker)
        reads = mocker.spy(tiff_dask, "_read_tiff_selection")
        # Chunks equal to the 475-row full-width strip bands of the stored image;
        # a default 1024-row tile spans three bands.
        result = stereoseq(
            root,
            cell_bins="unadjusted",
            cell_counts_file="C04687E314.cellbin.gef",
            image_roles=("registered",),
            chunks=(1, 475, 11760),
        )
        reads.assert_not_called()
        assert set(result.tables) == {"cells_table"}
        assert set(result.points) == {"cells_centroids"}
        assert set(result.images) == {"HE_registered"}
        _assert_native_cells(result.tables["cells_table"], root / "C04687E314.cellbin.gef")
        table = result.tables["cells_table"]
        assert isinstance(table.X, csr_array)
        assert isinstance(table.layers["exon"], csr_array)
        assert table.shape == (7541, 27694)
        assert table.X.nnz == 3301510
        assert int(table.X.sum()) == 5857957
        assert int(table.layers["exon"].sum()) == 4667531
        assert len(set(table.var["geneName"])) == table.n_vars - 18
        points = result.points["cells_centroids"].compute()
        np.testing.assert_array_equal(points[["x", "y"]], result.tables["cells_table"].obsm["spatial_grid"])
        np.testing.assert_array_equal(points.index, result.tables["cells_table"].obs["instance_id"])
        image = result.images["HE_registered"]
        assert isinstance(image, DataArray)
        assert image.dims == ("c", "y", "x")
        assert image.shape == (3, 14700, 11760)
        assert image.dtype == np.uint8
        selected = image.data[:, 5000:5004, 4000:4006].compute(scheduler="synchronous")
        reads.assert_called_once()
        selection = reads.call_args.args[1]
        assert selection[0].stop - selection[0].start <= 475
        assert (selection[0].stop - selection[0].start) * image.shape[2] * 3 < 2**24
        with tifffile.TiffFile(root / "C04687E314_HE_regist.tif") as tiff:
            store = tiff.aszarr()
            try:
                expected = np.asarray(zarr.open_array(store=store, mode="r")[5000:5004, 4000:4006, :])
            finally:
                store.close()
        np.testing.assert_array_equal(selected, expected.transpose(2, 0, 1))
        assert "spatialdata_io_format_version" not in result.attrs

    def test_vendor_saw7_complete_cells(
        self, require_test_dataset: Callable[[str], Path], mocker: MockerFixture
    ) -> None:
        root = require_test_dataset("stereoseq_mouse_kidney_saw7")
        _forbid_densification(mocker)
        result = stereoseq(root, cell_bins="unadjusted", cell_counts_file="C02533C1.cellbin.gef")

        assert set(result.tables) == {"cells_table"}
        assert set(result.points) == {"cells_centroids"}
        table = result.tables["cells_table"]
        _assert_native_cells(table, root / "C02533C1.cellbin.gef")
        assert isinstance(table.X, csr_array)
        assert table.shape == (110646, 26220)
        assert table.X.nnz == 27926335
        assert int(table.X.sum()) == 50226213
        assert isinstance(table.layers["exon"], csr_array)
        assert int(table.layers["exon"].sum()) == 48075283

        points = result.points["cells_centroids"].compute()
        np.testing.assert_array_equal(points[["x", "y"]], table.obsm["spatial_grid"])
        np.testing.assert_array_equal(points.index, table.obs["instance_id"])
        assert set(table.obs["region"]) == {"cells_centroids"}
        assert table.uns["stereoseq"]["origin"] == [0, 0]
        assert table.uns["stereoseq"]["resolution"] == 500

    def test_vendor_complete_adjusted_cells(
        self, require_test_dataset: Callable[[str], Path], mocker: MockerFixture
    ) -> None:
        root = require_test_dataset("stereoseq_mouse_eyeball")
        _forbid_densification(mocker)
        source = "C04687E314.adjusted.cellbin.gef"
        result = stereoseq(root, cell_bins="adjusted", adjusted_cell_counts_file=source)

        assert set(result.tables) == {"adjusted_cells_table"}
        assert set(result.points) == {"adjusted_cells_centroids"}
        table = result.tables["adjusted_cells_table"]
        _assert_native_cells(table, root / source)
        assert isinstance(table.X, csr_array)
        assert table.shape == (7546, 27694)
        assert table.X.nnz == 5581462
        assert int(table.X.sum()) == 10459997
        assert isinstance(table.layers["exon"], csr_array)
        assert int(table.layers["exon"].sum()) == 8243004

        points = result.points["adjusted_cells_centroids"].compute()
        np.testing.assert_array_equal(points[["x", "y"]], table.obsm["spatial_grid"])
        np.testing.assert_array_equal(points.index, table.obs["instance_id"])
        assert set(table.obs["region"]) == {"adjusted_cells_centroids"}

    @pytest.mark.parametrize(
        "case",
        [
            ("stereoseq_mouse_kidney_saw7", "C02533C1.tissue.gef", 8, 49995),
            ("stereoseq_saw7_1_square", "A03994E6.tissue.gef", 24, 28684),
            ("stereoseq_mouse_eyeball", "C04687E314.raw.gef", 24, 16145),
        ],
    )
    def test_real_square_feature_prefix(
        self,
        require_test_dataset: Callable[[str], Path],
        tmp_path: Path,
        mocker: MockerFixture,
        case: tuple[str, str, int, int],
    ) -> None:
        dataset, filename, features, events = case
        root = require_test_dataset(dataset)
        _forbid_densification(mocker)
        _write_square_prefix(root / filename, tmp_path / filename)
        with h5py.File(tmp_path / filename, "r") as reduced, h5py.File(root / filename, "r") as original:
            assert len(reduced["geneExp/bin1/expression"]) == events
            for name in ("gene", "expression", "exon"):
                assert reduced[f"geneExp/bin1/{name}"].dtype == original[f"geneExp/bin1/{name}"].dtype

        result = stereoseq(tmp_path, bin_sizes=(1,), square_counts_file=filename)
        assert set(result.tables) == {"bin1_table"}
        assert set(result.points) == {"bin1_genes"}
        table = result.tables["bin1_table"]
        assert table.n_vars == features
        _assert_square_prefix(table, root / filename)

        points = result.points["bin1_genes"].compute()
        np.testing.assert_array_equal(points[["x", "y"]], table.obsm["spatial_grid"])
        np.testing.assert_array_equal(points.index, table.obs["instance_id"])
        assert set(table.obs["region"]) == {"bin1_genes"}

    def test_vendor_nonsimple_contours_fail_strict_polygon_mode(
        self, require_test_dataset: Callable[[str], Path]
    ) -> None:
        root = require_test_dataset("stereoseq_mouse_eyeball")
        with pytest.raises(ReaderFormatError, match=r"invalid geometries.*55"):
            stereoseq(
                root,
                cell_bins="unadjusted",
                cell_geometry="polygons",
                invalid_geometries="raise",
                cell_counts_file="C04687E314.cellbin.gef",
            )

    def test_company_saw7_numbered_tree_matches_flat_export(
        self, require_test_dataset: Callable[[str], Path], tmp_path: Path, mocker: MockerFixture
    ) -> None:
        root = require_test_dataset("stereoseq_mouse_kidney_saw7")
        _forbid_densification(mocker)
        flat, numbered = tmp_path / "flat", tmp_path / "numbered"
        flat.mkdir()
        (numbered / "05.tissuecut").mkdir(parents=True)
        (numbered / "051.cellcut").mkdir()

        # Reconstruct the documented tree from original individual downloads;
        # this is not an untouched vendor directory archive. Only squares are reduced.
        cells, square = "C02533C1.cellbin.gef", "C02533C1.tissue.gef"
        copyfile(root / cells, flat / cells)
        copyfile(root / cells, numbered / "051.cellcut" / cells)
        _write_square_prefix(root / square, flat / square)
        copyfile(flat / square, numbered / "05.tissuecut" / square)

        expected = stereoseq(
            flat, bin_sizes=(1,), cell_bins="unadjusted", square_counts_file=square, cell_counts_file=cells
        )
        actual = stereoseq(numbered, bin_sizes=(1,), cell_bins="unadjusted")

        assert set(actual.tables) == {"bin1_table", "cells_table"}
        assert set(actual.points) == {"bin1_genes", "cells_centroids"}
        _assert_layout_equivalent(actual, expected)

    @pytest.mark.parametrize("direct_outs", [False, True], ids=["task_root", "direct_outs"])
    def test_company_saw8_outs_tree_matches_flat_export(
        self,
        require_test_dataset: Callable[[str], Path],
        tmp_path: Path,
        mocker: MockerFixture,
        *,
        direct_outs: bool,
    ) -> None:
        root = require_test_dataset("stereoseq_mouse_eyeball")
        _forbid_densification(mocker)
        reads = mocker.spy(tiff_dask, "_read_tiff_selection")
        flat, task = tmp_path / "flat", tmp_path / "task"
        flat.mkdir()
        expressions, images = task / "outs/feature_expression", task / "outs/image"
        expressions.mkdir(parents=True)
        images.mkdir()

        # Place unchanged company cell/image originals in the documented layout;
        # only the square source uses the same bounded prefix in both trees.
        cells = "C04687E314.cellbin.gef"
        adjusted = "C04687E314.adjusted.cellbin.gef"
        square = "C04687E314.raw.gef"
        registered = "C04687E314_HE_regist.tif"
        for filename in (cells, adjusted):
            copyfile(root / filename, flat / filename)
            copyfile(root / filename, expressions / filename)
        copyfile(root / registered, flat / registered)
        copyfile(root / registered, images / registered)
        _write_square_prefix(root / square, flat / square)
        copyfile(flat / square, expressions / square)

        expected = stereoseq(
            flat,
            bin_sizes=(1,),
            cell_bins="both",
            square_counts_file=square,
            cell_counts_file=cells,
            adjusted_cell_counts_file=adjusted,
            image_roles=("registered",),
        )
        selected_root = task / "outs" if direct_outs else task
        # Whole-chip raw counts always require an explicit source selection.
        actual = stereoseq(
            selected_root,
            bin_sizes=(1,),
            cell_bins="both",
            square_counts_file=(expressions / square).relative_to(selected_root),
            image_roles=("registered",),
        )

        reads.assert_not_called()
        assert set(actual.tables) == {"bin1_table", "cells_table", "adjusted_cells_table"}
        assert set(actual.points) == {"bin1_genes", "cells_centroids", "adjusted_cells_centroids"}
        assert set(actual.images) == {"HE_registered"}
        _assert_layout_equivalent(actual, expected)
        reads.assert_not_called()

        image, reference = actual.images["HE_registered"], expected.images["HE_registered"]
        assert isinstance(image, DataArray)
        assert isinstance(reference, DataArray)
        assert image.data.chunks == reference.data.chunks
        assert image.data.chunksize == (1, 1024, 1024)
        selection = image.data[:, 5000:5004, 4000:4006].compute(scheduler="synchronous")
        expected_selection = reference.data[:, 5000:5004, 4000:4006].compute(scheduler="synchronous")
        np.testing.assert_array_equal(selection, expected_selection)
        # Each reader decodes only the three 475-row strip bands of one default tile.
        assert reads.call_count == 6
        assert {call.args[1][0] for call in reads.call_args_list} == {
            slice(3800, 4275),
            slice(4275, 4750),
            slice(4750, 5225),
        }

    def test_study_saw6_complete_square_counts_and_exons(
        self, require_test_dataset: Callable[[str], Path], mocker: MockerFixture
    ) -> None:
        root = require_test_dataset("stereoseq_saw6_square")
        _forbid_densification(mocker)
        result = stereoseq(root, bin_sizes=(1,), square_counts_file="A01785D6.tissue.gef")
        assert set(result.tables) == {"bin1_table"}
        assert set(result.points) == {"bin1_genes"}
        assert not result.images
        assert not result.shapes
        _assert_native_square(result.tables["bin1_table"], root / "A01785D6.tissue.gef")
        first_points = result.points["bin1_genes"].head(5)
        np.testing.assert_array_equal(first_points[["x", "y"]], result.tables["bin1_table"].obsm["spatial_grid"][:5])
        assert "spatialdata_io_format_version" not in result.attrs
