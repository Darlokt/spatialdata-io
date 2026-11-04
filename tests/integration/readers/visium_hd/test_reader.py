"""Synthetic public-reader contracts for supported Space Ranger layouts."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Literal

import anndata as ad
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from anndata.tests.helpers import assert_equal
from numpy.testing import assert_allclose, assert_array_equal
from scanpy import read_10x_h5
from scipy.sparse import csr_array, csr_matrix, spmatrix
from spatialdata import read_zarr
from spatialdata.models import get_table_keys
from spatialdata.transformations import get_transformation
from xarray import DataArray

from spatialdata_io import visium_hd
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.visium_hd import _metadata, _reader
from tests.integration.readers.visium_hd._fixtures import CELL_IDS, write_bundle

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _library(path: Path, value: str) -> None:
    with h5py.File(path, "r+") as source:
        source.attrs["library_ids"] = np.asarray([value], dtype="S32")


def _version(path: Path, value: str | None) -> None:
    with h5py.File(path, "r+") as source:
        if value is None:
            del source.attrs["software_version"]
        else:
            source.attrs["software_version"] = f"spaceranger-{value}"


def _remove_counts(root: Path) -> None:
    for path in root.glob("**/*matrix.h5"):
        path.unlink()


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
        ("source_dtype", "first", "second", "expected"),
        [("int32", 2**24, 1, 2**24 + 1), ("uint32", 2**32 - 1, 2, 2**32 + 1)],
    )
    def test_duplicate_h5_bin_counts_preserve_order_positions_and_linkage(
        self, tmp_path: Path, source_dtype: str, first: int, second: int, expected: int
    ) -> None:
        bundle = write_bundle(tmp_path, sizes=(2,), segmented=False)
        baseline = visium_hd(bundle.root, bin_sizes=(2,), include_images=False).tables["square_002um_table"]
        positions = baseline.obsm["spatial"]
        assert isinstance(positions, np.ndarray)
        path = bundle.root / "binned_outputs/square_002um/filtered_feature_bc_matrix.h5"
        with h5py.File(path, "r+") as handle:
            data = handle["matrix/data"][:].astype(source_dtype)
            data[:2] = [first, second]
            del handle["matrix/data"]
            handle.create_dataset("matrix/data", data=data)
            handle["matrix/indices"][1] = 0
        decoded = read_10x_h5(path, gex_only=False)
        assert isinstance(decoded.X, csr_matrix | csr_array)
        assert decoded.X.dtype == np.dtype("float32" if source_dtype == "int32" else "uint32")
        assert not decoded.X.has_canonical_format
        assert_array_equal(decoded.X.data[:2], [first, second])

        result = visium_hd(bundle.root, bin_sizes=(2,), include_images=False)

        table = result.tables["square_002um_table"]
        assert isinstance(table.X, csr_array)
        assert table.X.dtype == np.dtype(np.float64)
        assert table.X.has_canonical_format
        truth = bundle.raw[2][bundle.filtered_rows[2]].astype(np.float64)
        truth[0] = [expected, 0, 0]
        assert_array_equal(table.X.toarray(), truth)
        assert table.obs_names.tolist() == [
            "s_002um_00001_00003-1",
            "s_002um_00000_00000-1",
            "s_002um_00001_00000-1",
            "s_002um_00000_00005-1",
        ]
        assert table.var_names.tolist() == ["Gene", "Gene-1", "Other"]
        assert get_table_keys(table) == ("square_002um", "region", "instance_id")
        assert table.obs["instance_id"].tolist() == result.shapes["square_002um"].index.tolist()
        assert_array_equal(table.obsm["spatial"], positions)

    @pytest.mark.parametrize(
        ("source_dtype", "first", "second", "expected"),
        [("int32", 2**24, 1, 2**24 + 1), ("uint32", 2**32 - 1, 2, 2**32 + 1)],
    )
    def test_duplicate_h5_raw_counts_feed_exact_nucleus_reconstruction(
        self, tmp_path: Path, source_dtype: str, first: int, second: int, expected: int
    ) -> None:
        bundle = write_bundle(tmp_path, sizes=(2,))
        baseline = visium_hd(bundle.root, bin_sizes=(), include_images=False, include_nucleus_table=True).tables[
            "nucleus_table"
        ]
        positions = baseline.obsm["spatial"]
        assert isinstance(positions, np.ndarray)
        path = bundle.root / "binned_outputs/square_002um/raw_feature_bc_matrix.h5"
        with h5py.File(path, "r+") as handle:
            data = handle["matrix/data"][:].astype(source_dtype)
            indices = handle["matrix/indices"][:]
            indptr = handle["matrix/indptr"][:]
            # Raw bin 1 belongs to cell 1's nucleus; it has one Gene-1 entry.
            start = int(indptr[1])
            data[start] = first
            data = np.insert(data, start + 1, second)
            indices = np.insert(indices, start + 1, indices[start])
            indptr[2:] += 1
            for name, values in (("data", data), ("indices", indices), ("indptr", indptr)):
                del handle[f"matrix/{name}"]
                handle.create_dataset(f"matrix/{name}", data=values)
        decoded = read_10x_h5(path, gex_only=False)
        assert isinstance(decoded.X, csr_matrix | csr_array)
        assert decoded.X.dtype == np.dtype("float32" if source_dtype == "int32" else "uint32")
        assert not decoded.X.has_canonical_format
        assert_array_equal(decoded.X.data[start : start + 2], [first, second])

        result = visium_hd(bundle.root, bin_sizes=(), include_images=False, include_nucleus_table=True)

        table = result.tables["nucleus_table"]
        assert isinstance(table.X, csr_array)
        assert table.X.dtype == np.dtype(np.float64)
        assert table.X.has_canonical_format
        assert_array_equal(table.X.toarray(), [[0, 0, 4], [0, expected, 0]])
        assert table.obs_names.tolist() == [CELL_IDS[1], CELL_IDS[0]]
        assert table.var_names.tolist() == ["Gene", "Gene-1", "Other"]
        assert get_table_keys(table) == ("nucleus_boundaries", "region", "instance_id")
        assert table.obs["instance_id"].tolist() == result.shapes["nucleus_boundaries"].index.tolist()
        assert_array_equal(table.obsm["spatial"], positions)
        cell_expression = result.tables["cell_table"].X
        assert isinstance(cell_expression, csr_array)
        assert_array_equal(cell_expression.toarray(), bundle.cells[[1, 0]])

    @pytest.mark.parametrize(
        ("version", "segmented", "sizes", "prefix", "feature_slice_name", "null_pair"),
        [
            ("3.0.0", False, (2, 8, 16), "", "hd_feature_slice.h5", None),
            ("3.1.0", False, (2, 8, 16), "", "feature_slice.h5", None),
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
        assert result.attrs["visium_hd"]["library_id"] is None
        assert result.attrs["spatialdata_io_format_version"] == version
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

    @pytest.mark.parametrize("version", ["4.2.0", "4.0.1"])
    def test_nonfirst_count_version_rejected_before_calibration(
        self, tmp_path: Path, mocker: MockerFixture, version: str
    ) -> None:
        write_bundle(tmp_path, segmented=False)
        conflicting = tmp_path / "binned_outputs/square_016um/filtered_feature_bc_matrix.h5"
        _version(conflicting, version)
        fit = mocker.patch.object(_reader, "fit_slide_frame", side_effect=AssertionError("identity before fitting"))
        with pytest.raises(ReaderFormatError) as error:
            visium_hd(tmp_path, bin_sizes=(8, 16), include_images=False)
        assert str(conflicting) in str(error.value)
        assert version in str(error.value)
        if version == "4.0.1":
            assert "4.1.0" in str(error.value)
        fit.assert_not_called()

    def test_selected_bin_calibration_rejected_before_fitting(self, tmp_path: Path, mocker: MockerFixture) -> None:
        write_bundle(tmp_path, segmented=False)
        path = tmp_path / "binned_outputs/square_008um/spatial/scalefactors_json.json"
        calibration = json.loads(path.read_text())
        calibration["bin_size_um"] = 16
        path.write_text(json.dumps(calibration))
        fit = mocker.patch.object(_reader, "fit_slide_frame", side_effect=AssertionError("calibration before fitting"))
        with pytest.raises(ReaderFormatError, match="bin_size_um") as error:
            visium_hd(tmp_path, bin_sizes=(8,), include_images=False)
        assert str(path) in str(error.value)
        fit.assert_not_called()

    def test_coarse_bin_cannot_reuse_native_calibration(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, segmented=False)
        native = tmp_path / "binned_outputs/square_002um/spatial/scalefactors_json.json"
        coarse = tmp_path / "binned_outputs/square_008um/spatial/scalefactors_json.json"
        coarse.unlink()
        coarse.symlink_to(native)
        with pytest.raises(ReaderFormatError, match="bin_size_um=8") as error:
            visium_hd(tmp_path, bin_sizes=(8,), include_images=False)
        assert str(native) in str(error.value)

    @pytest.mark.parametrize(
        "relative",
        [
            "binned_outputs/square_016um/filtered_feature_bc_matrix.h5",
            "segmented_outputs/filtered_feature_cell_matrix.h5",
            "binned_outputs/square_002um/raw_feature_bc_matrix.h5",
        ],
        ids=["bin", "cell", "raw-nucleus"],
    )
    def test_selected_library_conflict_before_calibration(
        self, tmp_path: Path, mocker: MockerFixture, relative: str
    ) -> None:
        write_bundle(tmp_path)
        _library(tmp_path / "binned_outputs/square_008um/filtered_feature_bc_matrix.h5", "expected-library")
        conflicting = tmp_path / relative
        _library(conflicting, "other-library")
        fit = mocker.patch.object(_reader, "fit_slide_frame", side_effect=AssertionError("identity before fitting"))
        with pytest.raises(ReaderFormatError) as error:
            visium_hd(tmp_path, bin_sizes=(8, 16), include_images=False, include_nucleus_table=True)
        message = str(error.value)
        assert str(conflicting) in message
        assert "expected-library" in message
        assert "other-library" in message
        fit.assert_not_called()

    @pytest.mark.parametrize("conflict", ["version", "library", "missing-version"])
    def test_consumed_feature_identity_before_calibration(
        self, tmp_path: Path, mocker: MockerFixture, conflict: str
    ) -> None:
        write_bundle(tmp_path, segmented=False)
        feature = tmp_path / "feature_slice.h5"
        _library(tmp_path / "binned_outputs/square_008um/filtered_feature_bc_matrix.h5", "expected-library")
        if conflict == "library":
            _library(feature, "other-library")
        else:
            _version(feature, "4.0.1" if conflict == "version" else None)
        fit = mocker.patch.object(_reader, "fit_slide_frame", side_effect=AssertionError("identity before fitting"))
        with pytest.raises(ReaderFormatError) as error:
            visium_hd(tmp_path, bin_sizes=(8,), include_images=False)
        message = str(error.value)
        assert str(feature) in message
        if conflict == "library":
            assert "expected-library" in message
            assert "other-library" in message
        elif conflict == "version":
            assert "4.0.1" in message
            assert "4.1.0" in message
        fit.assert_not_called()

    @pytest.mark.parametrize("sizes", [(8, 16), (16, 8)])
    @pytest.mark.parametrize(
        "relative",
        [
            "binned_outputs/square_016um/filtered_feature_bc_matrix.h5",
            "segmented_outputs/filtered_feature_cell_matrix.h5",
            "binned_outputs/square_002um/raw_feature_bc_matrix.h5",
            "feature_slice.h5",
        ],
        ids=["bin", "cell", "raw-nucleus", "feature"],
    )
    def test_sole_known_library_is_propagated(self, tmp_path: Path, relative: str, sizes: tuple[int, ...]) -> None:
        write_bundle(tmp_path)
        _library(tmp_path / relative, "known-library")
        result = visium_hd(tmp_path, bin_sizes=sizes, include_images=False, include_nucleus_table=True)
        assert result.attrs["visium_hd"]["library_id"] == "known-library"
        assert result.attrs["spatialdata_io_format_version"] == "4.1.0"

    def test_unselected_matrices_do_not_contribute_identity(self, tmp_path: Path) -> None:
        write_bundle(tmp_path)
        selected = tmp_path / "binned_outputs/square_008um/filtered_feature_bc_matrix.h5"
        _library(selected, "selected-library")
        for path in tmp_path.glob("**/*matrix.h5"):
            if path != selected:
                _version(path, "9.0.0")
                _library(path, "unselected-library")
        result = visium_hd(
            tmp_path,
            bin_sizes=(8,),
            include_cell_boundaries=False,
            include_nucleus_boundaries=False,
            include_images=False,
        )
        assert set(result.tables) == {"square_008um_table"}
        assert result.attrs["visium_hd"]["library_id"] == "selected-library"

    def test_selected_raw_two_micron_identity_is_read_once(self, tmp_path: Path, mocker: MockerFixture) -> None:
        write_bundle(tmp_path)
        raw = tmp_path / "binned_outputs/square_002um/raw_feature_bc_matrix.h5"
        _library(raw, "raw-library")
        identity = mocker.spy(_metadata, "read_identity")
        result = visium_hd(
            tmp_path, bin_sizes=(2,), bin_counts_kind="raw", include_images=False, include_nucleus_table=True
        )
        assert result.attrs["visium_hd"]["library_id"] == "raw-library"
        paths = [call.args[0] for call in identity.call_args_list]
        assert paths.count(raw) == 1
        assert set(paths) == {
            raw,
            tmp_path / "segmented_outputs/filtered_feature_cell_matrix.h5",
            tmp_path / "feature_slice.h5",
        }

    @pytest.mark.parametrize("nucleus_only", [False, True], ids=["images", "nucleus-geometry"])
    def test_products_without_count_files(self, tmp_path: Path, *, nucleus_only: bool) -> None:
        write_bundle(tmp_path, segmented=nucleus_only, images=not nucleus_only)
        _remove_counts(tmp_path)
        _library(tmp_path / "feature_slice.h5", "feature-library")
        result = visium_hd(
            tmp_path,
            bin_sizes=(),
            include_cell_boundaries=False,
            include_nucleus_boundaries=nucleus_only,
            include_images=not nucleus_only,
            include_cytassist_image=not nucleus_only,
        )
        assert not result.tables
        assert set(result.shapes) == ({"nucleus_boundaries"} if nucleus_only else set())
        assert set(result.images) == (set() if nucleus_only else {"hires_image", "lowres_image", "cytassist_image"})
        assert result.attrs["visium_hd"]["library_id"] == "feature-library"
        assert result.attrs["spatialdata_io_format_version"] == "4.1.0"

    @pytest.mark.parametrize("version", [None, "4.2.0"])
    def test_count_free_feature_requires_supported_identity(self, tmp_path: Path, version: str | None) -> None:
        write_bundle(tmp_path, segmented=False, images=True)
        _remove_counts(tmp_path)
        feature = tmp_path / "feature_slice.h5"
        _version(feature, version)
        with pytest.raises(ReaderFormatError) as error:
            visium_hd(tmp_path, bin_sizes=())
        assert str(feature) in str(error.value)

    @pytest.mark.parametrize("unused_counts", [False, True], ids=["no-count-files", "unused-count-files"])
    def test_count_free_read_requires_feature_identity(self, tmp_path: Path, *, unused_counts: bool) -> None:
        write_bundle(tmp_path, segmented=False, images=True)
        if not unused_counts:
            _remove_counts(tmp_path)
        (tmp_path / "feature_slice.h5").unlink()
        with pytest.raises(FileNotFoundError, match="feature_slice"):
            visium_hd(tmp_path, bin_sizes=())

    @pytest.mark.parametrize("split", [False, True], ids=["empty-root", "split-root-and-bin"])
    def test_partial_root_spatial_keeps_available_images(self, tmp_path: Path, *, split: bool) -> None:
        write_bundle(tmp_path, segmented=False, images=True)
        root_spatial = tmp_path / "spatial"
        root_spatial.mkdir()
        if split:
            shutil.move(
                tmp_path / "binned_outputs/square_002um/spatial/tissue_hires_image.png",
                root_spatial / "tissue_hires_image.png",
            )
        result = visium_hd(tmp_path, bin_sizes=(8,), include_cytassist_image=True)
        assert set(result.images) == {"hires_image", "lowres_image", "cytassist_image"}
        hires = result.images["hires_image"]
        lowres = result.images["lowres_image"]
        assert isinstance(hires, DataArray)
        assert isinstance(lowres, DataArray)
        assert hires.dims == lowres.dims == ("c", "y", "x")
        assert hires.shape == (3, 48, 48)
        assert lowres.shape == (3, 24, 24)
        assert hires.dtype == lowres.dtype == np.uint8
        assert_array_equal(hires.data[:, 0, 0].compute(), [0, 1, 2])
        assert_array_equal(lowres.data[:, 0, 0].compute(), [0, 0, 0])

    def test_missing_requested_product_still_fails_without_counts(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, segmented=False, images=True)
        _remove_counts(tmp_path)
        with pytest.raises(FileNotFoundError, match="cell_segmentations"):
            visium_hd(tmp_path, bin_sizes=(), include_cell_boundaries=True)

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
