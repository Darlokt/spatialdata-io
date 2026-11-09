"""End-to-end synthetic and repository-managed Visium workflows."""

from __future__ import annotations

import gzip
import json
import shutil
import warnings
from typing import TYPE_CHECKING, Literal, cast

import anndata as ad
import h5py
import imagecodecs
import numpy as np
import pandas as pd
import pytest
import tifffile
from anndata import read_h5ad
from anndata.tests.helpers import assert_equal
from scanpy import read_10x_h5
from scipy import sparse
from scipy.io import mmwrite
from spatialdata import SpatialData
from spatialdata.models import get_table_keys
from spatialdata.testing import assert_spatial_data_objects_are_identical
from spatialdata.transformations import BaseTransformation, get_transformation
from xarray import DataArray, DataTree

from spatialdata_io import __version__
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.visium import visium

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _write_h5(path: Path, *, library_id: bytes | None = b"sample") -> None:
    with h5py.File(path, "w") as source:
        if library_id is not None:
            source.attrs["library_ids"] = np.asarray([library_id])
        source.attrs["chemistry_description"] = b"Synthetic chemistry"
        source.attrs["software_version"] = b"spaceranger-9.9.9"

        matrix = source.create_group("matrix")
        matrix.create_dataset("barcodes", data=np.asarray([b"barcode-a", b"barcode-b", b"barcode-c"]))
        matrix.create_dataset("data", data=np.asarray([1, 5, 2, 3, 4, 6], dtype=np.int32))
        matrix.create_dataset("indices", data=np.asarray([0, 2, 1, 0, 1, 2], dtype=np.int32))
        matrix.create_dataset("indptr", data=np.asarray([0, 2, 3, 6], dtype=np.int32))
        matrix.create_dataset("shape", data=np.asarray([3, 3], dtype=np.int64))

        features = matrix.create_group("features")
        features.create_dataset("id", data=np.asarray([b"gene-id-a", b"gene-id-b", b"antibody-id"]))
        features.create_dataset("name", data=np.asarray([b"gene-a", b"gene-b", b"protein-a"]))
        features.create_dataset(
            "feature_type",
            data=np.asarray([b"Gene Expression", b"Gene Expression", b"Antibody Capture"]),
        )


def _gzip(source: Path) -> None:
    with source.open("rb") as input_stream, gzip.open(f"{source}.gz", "wb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream)
    source.unlink()


def _write_mex(path: Path, *, compressed: bool) -> None:
    path.mkdir()
    mmwrite(
        path / "matrix.mtx",
        sparse.coo_array(np.asarray([[1, 0, 3], [0, 2, 4], [5, 0, 6]], dtype=np.int32)),
    )
    (path / "barcodes.tsv").write_text("barcode-a\nbarcode-b\nbarcode-c\n", encoding="utf-8")
    (path / "features.tsv").write_text(
        (
            "gene-id-a\tgene-a\tGene Expression\n"
            "gene-id-b\tgene-b\tGene Expression\n"
            "antibody-id\tprotein-a\tAntibody Capture\n"
        ),
        encoding="utf-8",
    )

    if compressed:
        for artifact in path.iterdir():
            _gzip(artifact)


def _write_spatial(
    root: Path,
    *,
    version: Literal["v1", "v1_headered", "v2"],
    in_tissue: tuple[int, int, int, int] = (1, 1, 1, 0),
) -> None:
    spatial = root / "spatial"
    spatial.mkdir()
    rows = (
        f"barcode-a,{in_tissue[0]},0,2,10,20\n"
        f"barcode-b,{in_tissue[1]},1,1,30,40\n"
        f"barcode-c,{in_tissue[2]},2,0,50,60\n"
        f"unused-grid-spot,{in_tissue[3]},3,3,70,80\n"
    )
    if version == "v1":
        (spatial / "tissue_positions_list.csv").write_text(rows, encoding="utf-8")
    else:
        header = "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n"
        filename = "tissue_positions_list.csv" if version == "v1_headered" else "tissue_positions.csv"
        (spatial / filename).write_text(header + rows, encoding="utf-8")

    (spatial / "scalefactors_json.json").write_text(
        json.dumps(
            {
                "tissue_hires_scalef": 0.5,
                "tissue_lowres_scalef": 0.25,
                "spot_diameter_fullres": 6.0,
                "fiducial_diameter_fullres": 8.0,
            }
        ),
        encoding="utf-8",
    )

    hires = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    lowres = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    (spatial / "tissue_hires_image.png").write_bytes(imagecodecs.png_encode(hires))
    (spatial / "tissue_lowres_image.png").write_bytes(imagecodecs.png_encode(lowres))


def _affine_matrix(element: DataArray, coordinate_system: str) -> np.ndarray:
    transformation = cast(
        "BaseTransformation",
        get_transformation(element, to_coordinate_system=coordinate_system),
    )
    return np.asarray(transformation.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")))


def _assert_table_contract(result: SpatialData) -> None:
    table = result.tables["table"]
    assert isinstance(table.X, sparse.csr_array)
    assert not isinstance(table.X, sparse.spmatrix)
    assert table.X.format == "csr"
    np.testing.assert_array_equal(
        table.X.toarray(),
        np.asarray([[1, 0, 5], [0, 2, 0], [3, 4, 6]], dtype=np.float32),
    )

    assert table.obs_names.tolist() == ["barcode-a", "barcode-b", "barcode-c"]
    assert table.var_names.tolist() == ["gene-a", "gene-b", "protein-a"]
    assert table.var["gene_ids"].tolist() == ["gene-id-a", "gene-id-b", "antibody-id"]
    assert table.var["feature_types"].tolist() == ["Gene Expression", "Gene Expression", "Antibody Capture"]
    assert table.obs["in_tissue"].tolist() == [1, 1, 1]
    assert table.obs["spot_id"].tolist() == ["barcode-a", "barcode-b", "barcode-c"]
    assert table.obs["region"].tolist() == ["sample", "sample", "sample"]
    np.testing.assert_array_equal(table.obsm["spatial_pixels"], np.asarray([[20, 10], [40, 30], [60, 50]]))

    assert table.uns["spatial"]["sample"]["scalefactors"]["fiducial_diameter_fullres"] == 8.0
    assert get_table_keys(table) == ("sample", "region", "spot_id")


class TestVisiumSyntheticWorkflow:
    """Prove every supported count and position layout through the public API."""

    @pytest.mark.parametrize(
        ("source_dtype", "first", "second", "expected"),
        [("int32", 2**24, 1, 2**24 + 1), ("uint32", 2**32 - 1, 2, 2**32 + 1)],
    )
    def test_duplicate_h5_counts_preserve_exact_sum_and_public_linkage(
        self, tmp_path: Path, source_dtype: str, first: int, second: int, expected: int
    ) -> None:
        path = tmp_path / "filtered_feature_bc_matrix.h5"
        _write_h5(path)
        _write_spatial(tmp_path, version="v2")
        with h5py.File(path, "r+") as handle:
            data = handle["matrix/data"][:].astype(source_dtype)
            data[:2] = [first, second]
            del handle["matrix/data"]
            handle.create_dataset("matrix/data", data=data)
            handle["matrix/indices"][1] = 0
        # Establish retained decoded contributions; upstream rounding/summation
        # would be outside the local count boundary's recovery guarantee.
        decoded = read_10x_h5(path, gex_only=False)
        assert isinstance(decoded.X, sparse.csr_matrix | sparse.csr_array)
        assert decoded.X.dtype == np.dtype("float32" if source_dtype == "int32" else "uint32")
        assert not decoded.X.has_canonical_format
        np.testing.assert_array_equal(decoded.X.data[:2], [first, second])

        result = visium(tmp_path)

        table = result.tables["table"]
        assert isinstance(table.X, sparse.csr_array)
        assert table.X.dtype == np.dtype(np.float64)
        assert table.X.has_canonical_format
        np.testing.assert_array_equal(table.X.toarray(), [[expected, 0, 0], [0, 2, 0], [3, 4, 6]])
        assert table.obs_names.tolist() == ["barcode-a", "barcode-b", "barcode-c"]
        assert table.var_names.tolist() == ["gene-a", "gene-b", "protein-a"]
        assert get_table_keys(table) == ("sample", "region", "spot_id")
        assert table.obs["spot_id"].tolist() == result.shapes["sample"].index.tolist()
        np.testing.assert_array_equal(table.obsm["spatial_pixels"], [[20, 10], [40, 30], [60, 50]])

    def test_rejects_mex_legacy_selector_before_silent_feature_relabeling(self, tmp_path: Path) -> None:
        mex = tmp_path / "filtered_feature_bc_matrix"
        _write_mex(mex, compressed=False)
        _write_spatial(tmp_path, version="v2")
        conflict = mex / "genes.tsv"
        conflict.write_text("wrong-id-a\twrong-a\nwrong-id-b\twrong-b\nwrong-id-c\twrong-c\n", encoding="utf-8")

        with pytest.raises(ReaderFormatError, match=r"genes\.tsv") as caught:
            visium(tmp_path, dataset_id="sample")

        assert caught.value.context.reader == "visium"
        assert caught.value.context.artifact == "count matrix"
        assert str(conflict) in str(caught.value)

    def test_mex_retains_exact_feature_identity_with_benign_legacy_named_artifacts(self, tmp_path: Path) -> None:
        mex = tmp_path / "filtered_feature_bc_matrix"
        _write_mex(mex, compressed=True)
        _write_spatial(tmp_path, version="v2")
        (mex / "genes.tsv").mkdir()
        (mex / "genes.tsv.gz").write_bytes(gzip.compress(b"wrong-id\twrong-name\n"))
        (mex / "other_genes.tsv").write_text("wrong-id\twrong-name\n", encoding="utf-8")

        result = visium(tmp_path, dataset_id="sample")

        _assert_table_contract(result)

    @pytest.mark.parametrize(("version", "barcode_index"), [("v1", 0), ("v1", 1), ("v2", 1)])
    def test_preserves_quoted_multiline_barcodes(
        self, tmp_path: Path, version: Literal["v1", "v2"], barcode_index: int
    ) -> None:
        counts = tmp_path / "filtered_feature_bc_matrix.h5"
        _write_h5(counts)
        _write_spatial(tmp_path, version=version)
        barcodes = ["barcode-a", "barcode-b", "barcode-c"]
        barcodes[barcode_index] += "\ncontinued"
        with h5py.File(counts, "r+") as source:
            del source["matrix/barcodes"]
            source.create_dataset("matrix/barcodes", data=np.asarray([barcode.encode() for barcode in barcodes]))
        filename = "tissue_positions_list.csv" if version == "v1" else "tissue_positions.csv"
        positions = tmp_path / "spatial" / filename
        header = "" if version == "v1" else positions.read_text(encoding="utf-8").splitlines()[0] + "\n"
        rows = [
            f'"{barcode}",1,{index},{2 - index},{10 + 20 * index},{20 + 20 * index}\n'
            for index, barcode in enumerate(barcodes)
        ]
        positions.write_text(header + "".join(rows), encoding="utf-8")

        result = visium(tmp_path)

        table = result.tables["table"]
        assert table.obs_names.tolist() == barcodes
        assert table.obs["spot_id"].tolist() == barcodes
        assert result.shapes["sample"].index.tolist() == barcodes
        assert get_table_keys(table) == ("sample", "region", "spot_id")
        assert isinstance(table.X, sparse.csr_array)
        np.testing.assert_array_equal(table.X.data, np.asarray([1, 5, 2, 3, 4, 6], dtype=np.float32))
        np.testing.assert_array_equal(table.obsm["spatial_pixels"], [[20, 10], [40, 30], [60, 50]])

    @pytest.mark.parametrize("key", ["tissue_hires_scalef", "tissue_lowres_scalef"])
    def test_rejects_scales_that_would_return_nonfinite_image_transform(self, tmp_path: Path, key: str) -> None:
        _write_h5(tmp_path / "filtered_feature_bc_matrix.h5")
        _write_spatial(tmp_path, version="v2")
        path = tmp_path / "spatial" / "scalefactors_json.json"
        values = json.loads(path.read_text(encoding="utf-8"))
        values[key] = float(np.nextafter(0.0, 1.0))
        path.write_text(json.dumps(values), encoding="utf-8")

        with pytest.raises(ReaderFormatError, match=key) as caught:
            visium(tmp_path)

        assert caught.value.path == path
        assert caught.value.context.artifact == "scale factors"

    @pytest.mark.parametrize("values", [(0.5, 0.5), (-1.0, 2.0)])
    def test_rejects_malformed_duplicate_counts(self, tmp_path: Path, values: tuple[float, float]) -> None:
        source = tmp_path / "filtered_feature_bc_matrix.h5"
        _write_h5(source)
        _write_spatial(tmp_path, version="v2")
        with h5py.File(source, "r+") as handle:
            data = handle["matrix/data"][:].astype(np.float64)
            data[:2] = values
            del handle["matrix/data"]
            handle.create_dataset("matrix/data", data=data)
            handle["matrix/indices"][1] = 0
        with pytest.raises(ReaderFormatError, match=r"negative|fractional"):
            visium(tmp_path)

    def test_named_positions_accept_reordered_fields_and_project_extensions(self, tmp_path: Path) -> None:
        """Named positions accept reordered fields and project extensions."""
        _write_h5(tmp_path / "filtered_feature_bc_matrix.h5")
        _write_spatial(tmp_path, version="v2")
        path = tmp_path / "spatial" / "tissue_positions.csv"
        positions = pd.read_csv(path)
        positions["quality"] = "good"
        positions[list(reversed(positions.columns))].to_csv(path, index=False)

        result = visium(tmp_path)

        _assert_table_contract(result)
        assert "quality" not in result.tables["table"].obs
        assert result.tables["table"].obs["in_tissue"].dtype == np.dtype(np.uint8)
        for column in ("array_row", "array_col"):
            assert result.tables["table"].obs[column].dtype == np.dtype(np.uint64)

    @pytest.mark.parametrize("version", ["v1_headered", "v2"])
    def test_named_positions_preserve_multiline_extension_header(
        self, tmp_path: Path, version: Literal["v1_headered", "v2"]
    ) -> None:
        """Classify the complete logical header before projecting an unused multiline field."""
        _write_h5(tmp_path / "filtered_feature_bc_matrix.h5")
        _write_spatial(tmp_path, version=version)
        filename = "tissue_positions_list.csv" if version == "v1_headered" else "tissue_positions.csv"
        path = tmp_path / "spatial" / filename
        header, *rows = path.read_text(encoding="utf-8").splitlines()
        path.write_text(
            '"quality\ncontinued",' + header + "\n" + "".join("good," + row + "\n" for row in rows), encoding="utf-8"
        )

        result = visium(tmp_path)

        _assert_table_contract(result)
        assert "quality\ncontinued" not in result.tables["table"].obs

    def test_duplicate_raw_header_names_are_rejected_before_projection(self, tmp_path: Path) -> None:
        """Duplicate raw header names are rejected before projection."""
        _write_h5(tmp_path / "filtered_feature_bc_matrix.h5")
        _write_spatial(tmp_path, version="v2")
        path = tmp_path / "spatial" / "tissue_positions.csv"
        header, *rows = path.read_text().splitlines()
        path.write_text(header + ",array_row\n" + "\n".join(row + ",0" for row in rows) + "\n")

        with pytest.raises(ValueError, match="duplicate column"):
            visium(tmp_path)

    @pytest.mark.parametrize("grid_row", [0, 2**53 + 1, 2**63 - 1, 2**63, 2**64 - 1])
    def test_table_h5ad_roundtrip_preserves_values_and_annotations(self, tmp_path: Path, grid_row: int) -> None:
        """Table h5ad roundtrip preserves values and annotations."""
        _write_h5(tmp_path / "filtered_feature_bc_matrix.h5")
        _write_spatial(tmp_path, version="v2")
        positions = tmp_path / "spatial" / "tissue_positions.csv"
        positions.write_text(positions.read_text().replace("barcode-a,1,0,2,10,20", f"barcode-a,1,{grid_row},2,10,20"))
        result = visium(tmp_path)
        path = tmp_path / "table.h5ad"

        result.tables["table"].write_h5ad(path, convert_strings_to_categoricals=False)
        restored = read_h5ad(path)

        assert isinstance(restored.obs, pd.DataFrame)
        assert isinstance(restored.var, pd.DataFrame)
        assert isinstance(result.tables["table"].obs, pd.DataFrame)
        assert isinstance(result.tables["table"].var, pd.DataFrame)
        assert restored.obs["array_row"].tolist()[0] == grid_row
        assert restored.obs["array_row"].dtype == np.dtype(np.uint64)
        assert restored.obs["in_tissue"].dtype == np.dtype(np.uint8)
        pd.testing.assert_frame_equal(restored.obs, result.tables["table"].obs, check_exact=True)
        pd.testing.assert_frame_equal(restored.var, result.tables["table"].var, check_exact=True)
        expression = restored.X
        expected = result.tables["table"].X
        assert isinstance(expression, sparse.csr_array | sparse.csr_matrix)
        assert isinstance(expected, sparse.csr_array)
        np.testing.assert_array_equal(expression.data, expected.data)
        np.testing.assert_array_equal(expression.indices, expected.indices)
        np.testing.assert_array_equal(expression.indptr, expected.indptr)

    def test_full_uint64_grid_preserves_spatialdata_zarr_linkage(self, tmp_path: Path) -> None:
        """Persist the largest grid coordinate without changing textual spot identity."""
        _write_h5(tmp_path / "filtered_feature_bc_matrix.h5")
        _write_spatial(tmp_path, version="v2")
        positions = tmp_path / "spatial" / "tissue_positions.csv"
        positions.write_text(
            positions.read_text().replace("barcode-a,1,0,2,10,20", f"barcode-a,1,{2**64 - 1},{2**64 - 1},10,20")
        )
        result = visium(tmp_path)
        store = tmp_path / "result.zarr"

        result.write(store)
        restored = SpatialData.read(store)

        for actual in (result, restored):
            table = actual.tables["table"]
            assert table.obs_names.tolist() == ["barcode-a", "barcode-b", "barcode-c"]
            assert actual.shapes["sample"].index.name == "spot_id"
            assert table.obs["spot_id"].tolist() == actual.shapes["sample"].index.tolist()
            assert get_table_keys(table) == ("sample", "region", "spot_id")
            assert table.obs["in_tissue"].dtype == np.dtype(np.uint8)
            for column in ("array_row", "array_col"):
                assert table.obs[column].dtype == np.dtype(np.uint64)
                assert table.obs[column].tolist()[0] == 2**64 - 1
            np.testing.assert_array_equal(table.obsm["spatial_pixels"], [[20, 10], [40, 30], [60, 50]])

    @pytest.mark.parametrize(
        ("counts_kind", "position_version"),
        [("h5", "v1"), ("mex_gz", "v2"), ("mex_plain", "v1_headered")],
    )
    def test_exact_scientific_contract(
        self,
        tmp_path: Path,
        counts_kind: Literal["h5", "mex_gz", "mex_plain"],
        position_version: Literal["v1", "v1_headered", "v2"],
    ) -> None:
        if counts_kind == "h5":
            _write_h5(tmp_path / "filtered_feature_bc_matrix.h5")
            dataset_id = None
        else:
            _write_mex(tmp_path / "filtered_feature_bc_matrix", compressed=counts_kind == "mex_gz")
            dataset_id = "sample"

        _write_spatial(tmp_path, version=position_version)
        fullres = tmp_path / "fullres.tif"
        fullres_data = np.arange(3 * 6 * 8, dtype=np.uint16).reshape(3, 6, 8)
        tifffile.imwrite(fullres, fullres_data, metadata={"axes": "CYX"}, photometric="minisblack", tile=(16, 16))

        result = visium(
            tmp_path,
            dataset_id=dataset_id,
            fullres_image_file=fullres,
            scale_factors=(2,),
            chunks=(1, 3, 4),
        )

        assert set(result.images) == {"sample_full_image", "sample_hires_image", "sample_lowres_image"}
        assert set(result.shapes) == {"sample"}
        assert set(result.tables) == {"table"}
        # Without a verifiable calibration no element is placed in micrometres.
        assert sorted(result.coordinate_systems) == ["sample_hires_pixels", "sample_lowres_pixels", "sample_pixels"]

        assert result.attrs["spatialdata_io_software_version"] == __version__
        assert result.attrs["spatialdata_io_reader"] == "visium"
        if counts_kind == "h5":
            assert result.attrs["spatialdata_io_format_version"] == "9.9.9"
        else:
            assert "spatialdata_io_format_version" not in result.attrs

        full_pyramid = result.images["sample_full_image"]
        assert isinstance(full_pyramid, DataTree)
        assert list(full_pyramid.children) == ["scale0", "scale1"]
        assert full_pyramid["scale1"]["image"].shape == (3, 3, 4)

        full_image = full_pyramid["scale0"]["image"]
        assert full_image.dims == ("c", "y", "x")
        assert full_image.shape == (3, 6, 8)
        assert full_image.dtype == np.dtype(np.uint16)
        assert full_image.chunksizes == {"c": (1, 1, 1), "y": (3, 3), "x": (4, 4)}
        np.testing.assert_array_equal(full_image.data.compute(), fullres_data)

        hires = result.images["sample_hires_image"]
        assert isinstance(hires, DataArray)
        assert hires.shape == (3, 4, 5)
        assert hires.chunksizes == {"c": (1, 1, 1), "y": (3, 1), "x": (4, 1)}
        np.testing.assert_array_equal(_affine_matrix(hires, "sample_pixels"), np.diag([2.0, 2.0, 1.0]))

        circles = result.shapes["sample"]
        assert circles.index.tolist() == ["barcode-a", "barcode-b", "barcode-c"]
        assert circles.index.name == "spot_id"
        np.testing.assert_array_equal(circles.geometry.x, np.asarray([20.0, 40.0, 60.0]))
        np.testing.assert_array_equal(circles.geometry.y, np.asarray([10.0, 30.0, 50.0]))
        np.testing.assert_array_equal(circles["radius"], np.asarray([3.0, 3.0, 3.0]))

        _assert_table_contract(result)

    def test_raw_counts_preserve_non_tissue_barcodes(self, tmp_path: Path) -> None:
        raw_counts = tmp_path / "raw_feature_bc_matrix.h5"
        _write_h5(raw_counts)
        _write_spatial(tmp_path, version="v2", in_tissue=(1, 0, 1, 0))

        result = visium(tmp_path, counts_source=raw_counts.name)

        table = result.tables["table"]
        assert table.obs_names.tolist() == ["barcode-a", "barcode-b", "barcode-c"]
        assert table.obs["in_tissue"].tolist() == [1, 0, 1]
        assert result.shapes["sample"].index.tolist() == ["barcode-a", "barcode-b", "barcode-c"]

    def test_legacy_h5_filename_prefix_supplies_dataset_id(self, tmp_path: Path) -> None:
        _write_h5(tmp_path / "legacy_filtered_feature_bc_matrix.h5", library_id=None)
        _write_spatial(tmp_path, version="v1")

        result = visium(tmp_path)

        assert set(result.shapes) == {"legacy"}
        assert result.tables["table"].obs["region"].tolist() == ["legacy", "legacy", "legacy"]

    def test_absolute_artifact_overrides_may_live_outside_root(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        external = tmp_path / "external"
        root.mkdir()
        external.mkdir()
        counts = external / "counts.h5"
        _write_h5(counts)
        _write_spatial(external, version="v2")

        result = visium(
            root,
            counts_source=counts,
            positions_file=external / "spatial" / "tissue_positions.csv",
            scale_factors_file=external / "spatial" / "scalefactors_json.json",
        )

        assert set(result.shapes) == {"sample"}
        assert result.tables["table"].n_obs == 3

    def test_store_roundtrip_preserves_public_elements(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        _write_h5(source / "filtered_feature_bc_matrix.h5")
        _write_spatial(source, version="v2")
        fullres = source / "fullres.tif"
        tifffile.imwrite(fullres, np.arange(35, dtype=np.uint16).reshape(5, 7), metadata={"axes": "YX"})
        result = visium(source, fullres_image_file=fullres.name)
        store = tmp_path / "result.zarr"

        for storage in ("h5ad", "zarr"):
            table_path = tmp_path / f"positions_visium.{storage}"
            source_table = result.tables["table"]
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
        result.write(store)
        restored = SpatialData.read(store)

        assert set(restored.images) == {"sample_full_image", "sample_hires_image", "sample_lowres_image"}
        # Without explicit scale factors the full-resolution image is single-scale.
        assert isinstance(restored.images["sample_full_image"], DataArray)
        assert set(restored.shapes) == {"sample"}
        assert restored.tables["table"].shape == (3, 3)


class TestVisiumImagesAndNames:
    """Channel names follow declared colour models and derived names are repaired."""

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            ("rgb", ["r", "g", "b"]),
            ("if_ome", ["DAPI", "CD3", "CD8", "PanCK"]),
            ("if_plain", [0, 1, 2, 3]),
        ],
    )
    def test_channel_coordinates_are_exact(self, tmp_path: Path, kind: str, expected: list[object]) -> None:
        _write_h5(tmp_path / "filtered_feature_bc_matrix.h5")
        _write_spatial(tmp_path, version="v2")
        (tmp_path / "spatial" / "tissue_lowres_image.png").write_bytes(
            imagecodecs.png_encode(np.zeros((2, 3, 4), dtype=np.uint8))
        )
        fullres = tmp_path / "fullres.tif"
        if kind == "rgb":
            values = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
            tifffile.imwrite(fullres, values, photometric="rgb")
            values = np.moveaxis(values, -1, 0)
        else:
            values = np.arange(4 * 6 * 8, dtype=np.uint16).reshape(4, 6, 8)
            metadata: dict[str, object] = {"axes": "CYX"}
            if kind == "if_ome":
                metadata["Channel"] = {"Name": expected}
            tifffile.imwrite(fullres, values, ome=kind == "if_ome", photometric="minisblack", metadata=metadata)

        result = visium(tmp_path, fullres_image_file=fullres.name)

        full = result.images["sample_full_image"]
        assert full.coords["c"].to_numpy().tolist() == expected
        assert full.dtype == values.dtype
        np.testing.assert_array_equal(full.data.compute(), values)
        assert result.images["sample_hires_image"].coords["c"].to_numpy().tolist() == ["r", "g", "b"]
        assert result.images["sample_lowres_image"].coords["c"].to_numpy().tolist() == ["r", "g", "b", "a"]

        store = tmp_path / "result.zarr"
        result.write(store)
        restored = SpatialData.read(store)
        assert restored.images["sample_full_image"].coords["c"].to_numpy().tolist() == expected

    def test_multi_series_tiff_requires_a_selection(self, tmp_path: Path) -> None:
        _write_h5(tmp_path / "filtered_feature_bc_matrix.h5")
        _write_spatial(tmp_path, version="v2")
        fullres = tmp_path / "fullres.tif"
        with tifffile.TiffWriter(fullres) as writer:
            writer.write(np.zeros((5, 7), dtype=np.uint8), photometric="minisblack")
            writer.write(np.ones((9, 11), dtype=np.uint8), photometric="minisblack")

        with pytest.raises(ReaderFormatError, match="2 TIFF series") as caught:
            visium(tmp_path, fullres_image_file=fullres)
        assert caught.value.context.detected_version == "9.9.9"

        result = visium(tmp_path, fullres_image_file=fullres, tiff_series=1)
        assert result.images["sample_full_image"].shape == (1, 9, 11)

    def test_unusable_library_id_is_sanitized(self, tmp_path: Path) -> None:
        _write_h5(tmp_path / "filtered_feature_bc_matrix.h5", library_id=b"V1 Mouse:Brain")
        _write_spatial(tmp_path, version="v2")

        with pytest.warns(UserWarning, match=r"HDF5 library ID 'V1 Mouse:Brain' .* using 'V1_Mouse_Brain'"):
            result = visium(tmp_path)

        assert set(result.shapes) == {"V1_Mouse_Brain"}
        assert set(result.images) == {"V1_Mouse_Brain_hires_image", "V1_Mouse_Brain_lowres_image"}
        assert get_table_keys(result.tables["table"]) == ("V1_Mouse_Brain", "region", "spot_id")
        assert result.tables["table"].uns["spatialdata_io_positions"]["coordinates"]["spatial_pixels"]["frame"] == (
            "V1_Mouse_Brain_pixels"
        )
        assert sorted(result.coordinate_systems) == [
            "V1_Mouse_Brain_hires_pixels",
            "V1_Mouse_Brain_lowres_pixels",
            "V1_Mouse_Brain_pixels",
        ]
        result.write(tmp_path / "result.zarr")
        assert set(SpatialData.read(tmp_path / "result.zarr").shapes) == {"V1_Mouse_Brain"}

    @pytest.mark.parametrize("counts", ["h5", "mex"])
    @pytest.mark.parametrize(
        ("make_unique", "expected"),
        [(True, ["gene-a", "gene-a-1", "protein-a"]), (False, ["gene-a", "gene-a", "protein-a"])],
    )
    def test_feature_name_uniqueness_is_optional(
        self, tmp_path: Path, counts: str, expected: list[str], *, make_unique: bool
    ) -> None:
        if counts == "h5":
            _write_h5(tmp_path / "filtered_feature_bc_matrix.h5")
            with h5py.File(tmp_path / "filtered_feature_bc_matrix.h5", "r+") as source:
                source["matrix/features/name"][1] = b"gene-a"
        else:
            mex = tmp_path / "filtered_feature_bc_matrix"
            _write_mex(mex, compressed=False)
            features = (mex / "features.tsv").read_text(encoding="utf-8").replace("\tgene-b\t", "\tgene-a\t")
            (mex / "features.tsv").write_text(features, encoding="utf-8")
        _write_spatial(tmp_path, version="v2")

        with warnings.catch_warnings():
            # AnnData warns about the duplicate names it is asked to keep.
            warnings.filterwarnings("ignore", "Variable names are not unique", UserWarning)
            result = visium(tmp_path, dataset_id="sample", make_var_names_unique=make_unique)

        assert result.tables["table"].var_names.tolist() == expected
        assert result.tables["table"].var["gene_ids"].tolist() == ["gene-id-a", "gene-id-b", "antibody-id"]

    def test_raw_software_version_stays_in_uns(self, tmp_path: Path) -> None:
        _write_h5(tmp_path / "filtered_feature_bc_matrix.h5")
        _write_spatial(tmp_path, version="v2")

        result = visium(tmp_path)

        assert result.tables["table"].uns["spatial"]["sample"]["metadata"] == {
            "chemistry_description": "Synthetic chemistry",
            "software_version": "spaceranger-9.9.9",
        }
        assert result.attrs["spatialdata_io_format_version"] == "9.9.9"


class TestVisiumDownloadedWorkflow:
    """Smoke-test exact versioned vendor artifacts managed by the repository."""

    @pytest.mark.data
    @pytest.mark.parametrize(
        ("dataset_key", "dataset_id", "software_version"),
        [
            ("visium_v1_human_lymph_node", "V1_Human_Lymph_Node", "1.1.0"),
            ("visium_v2_1_mouse_brain", "V1_Adult_Mouse_Brain", "2.1.0"),
        ],
    )
    def test_official_10x_dataset(
        self,
        require_test_dataset: Callable[[str], Path],
        dataset_key: str,
        dataset_id: str,
        software_version: str,
    ) -> None:
        result = visium(require_test_dataset(dataset_key))

        assert set(result.shapes) == {dataset_id}
        assert result.tables["table"].n_obs == len(result.shapes[dataset_id])
        assert isinstance(result.tables["table"].X, sparse.sparray)
        assert result.tables["table"].var_names.is_unique
        assert result.attrs["spatialdata_io_format_version"] == software_version


class TestVisiumPathPolicy:
    """Every discovered artifact follows the dataset root's symlink policy.

    Earlier releases accepted these out-of-root symlinks without a policy.
    """

    @pytest.mark.parametrize("counts", ["h5", "mex"])
    def test_out_of_root_symlinks_require_the_relaxed_policy(
        self, tmp_path: Path, link_outside_root: Callable[[Path], Path], counts: Literal["h5", "mex"]
    ) -> None:
        if counts == "h5":
            _write_h5(tmp_path / "filtered_feature_bc_matrix.h5")
            linked = [tmp_path / "filtered_feature_bc_matrix.h5"]
        else:
            _write_mex(tmp_path / "filtered_feature_bc_matrix", compressed=True)
            # The MEX reader opens these three files below the discovered directory.
            linked = [tmp_path / "filtered_feature_bc_matrix" / "matrix.mtx.gz"]
        _write_spatial(tmp_path, version="v2")
        linked += [tmp_path / "spatial" / "tissue_positions.csv", tmp_path / "spatial" / "tissue_hires_image.png"]
        expected = visium(tmp_path, dataset_id="sample")
        for artifact in linked:
            link_outside_root(artifact)

        with pytest.raises(ReaderFormatError, match="path_policy='relaxed'"):
            visium(tmp_path, dataset_id="sample")

        relaxed = visium(tmp_path, dataset_id="sample", path_policy="relaxed")
        assert_spatial_data_objects_are_identical(relaxed, expected)

    def test_relative_overrides_follow_the_path_policy(
        self, tmp_path: Path, link_outside_root: Callable[[Path], Path]
    ) -> None:
        _write_h5(tmp_path / "selected.h5")
        _write_spatial(tmp_path, version="v2")
        (tmp_path / "spatial" / "tissue_positions.csv").rename(tmp_path / "positions.csv")
        options = {"counts_source": "selected.h5", "positions_file": "positions.csv"}
        expected = visium(tmp_path, dataset_id="sample", **options)
        link_outside_root(tmp_path / "selected.h5")
        link_outside_root(tmp_path / "positions.csv")

        with pytest.raises(ReaderFormatError, match="path_policy='relaxed'"):
            visium(tmp_path, dataset_id="sample", **options)

        relaxed = visium(tmp_path, dataset_id="sample", path_policy="relaxed", **options)
        assert_spatial_data_objects_are_identical(relaxed, expected)

    def test_strict_rejects_each_discovered_artifact(
        self, tmp_path: Path, link_outside_root: Callable[[Path], Path]
    ) -> None:
        _write_mex(tmp_path / "filtered_feature_bc_matrix", compressed=False)
        _write_spatial(tmp_path, version="v2")
        artifacts = [
            "filtered_feature_bc_matrix/barcodes.tsv",
            "filtered_feature_bc_matrix/features.tsv",
            "spatial/scalefactors_json.json",
            "spatial/tissue_lowres_image.png",
        ]
        for artifact in artifacts:
            target = link_outside_root(tmp_path / artifact)
            with pytest.raises(ReaderFormatError, match="resolving outside the dataset root"):
                visium(tmp_path, dataset_id="sample")
            (tmp_path / artifact).unlink()
            target.rename(tmp_path / artifact)
