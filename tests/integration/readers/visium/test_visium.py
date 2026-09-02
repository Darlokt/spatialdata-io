"""End-to-end synthetic and repository-managed Visium workflows."""

from __future__ import annotations

import gzip
import json
import shutil
from typing import TYPE_CHECKING, Literal, cast

import h5py
import imagecodecs
import numpy as np
import pytest
import tifffile
from scipy import sparse
from scipy.io import mmwrite
from spatialdata import SpatialData
from spatialdata.models import get_table_keys
from spatialdata.transformations import BaseTransformation, get_transformation
from xarray import DataArray, DataTree

from spatialdata_io import __version__
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
    assert table.var["feature_types"].tolist() == ["Gene Expression", "Gene Expression", "Antibody Capture"]
    assert table.obs["in_tissue"].tolist() == [1, 1, 1]
    assert table.obs["spot_id"].tolist() == ["barcode-a", "barcode-b", "barcode-c"]
    assert table.obs["region"].tolist() == ["sample", "sample", "sample"]
    np.testing.assert_array_equal(table.obsm["spatial"], np.asarray([[20, 10], [40, 30], [60, 50]]))
    assert table.uns["spatial"]["sample"]["scalefactors"]["fiducial_diameter_fullres"] == 8.0
    assert get_table_keys(table) == ("sample", "region", "spot_id")


class TestVisiumSyntheticWorkflow:
    """Prove every supported count and position layout through the public API."""

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

        result = visium(tmp_path, dataset_id=dataset_id, fullres_image_file=fullres, chunks=(1, 3, 4))

        assert set(result.images) == {"sample_full_image", "sample_hires_image", "sample_lowres_image"}
        assert set(result.shapes) == {"sample"}
        assert set(result.tables) == {"table"}
        assert result.attrs["spatialdata_io_software_version"] == __version__
        assert result.attrs["spatialdata_io_reader"] == "visium"
        if counts_kind == "h5":
            assert result.attrs["spatialdata_io_format_version"] == "spaceranger-9.9.9"
        else:
            assert "spatialdata_io_format_version" not in result.attrs

        full_pyramid = result.images["sample_full_image"]
        assert isinstance(full_pyramid, DataTree)
        assert list(full_pyramid.children) == ["scale0", "scale1", "scale2"]
        full_image = full_pyramid["scale0"]["image"]
        assert full_image.dims == ("c", "y", "x")
        assert full_image.shape == (3, 6, 8)
        assert full_image.dtype == np.dtype(np.uint16)
        np.testing.assert_array_equal(full_image.data[:, :2, :3].compute(), fullres_data[:, :2, :3])
        hires = result.images["sample_hires_image"]
        assert isinstance(hires, DataArray)
        assert hires.shape == (3, 4, 5)
        np.testing.assert_array_equal(_affine_matrix(hires, "sample"), np.diag([2.0, 2.0, 1.0]))

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

        result.write(store)
        restored = SpatialData.read(store)

        assert set(restored.images) == {"sample_full_image", "sample_hires_image", "sample_lowres_image"}
        assert isinstance(restored.images["sample_full_image"], DataTree)
        assert set(restored.shapes) == {"sample"}
        assert restored.tables["table"].shape == (3, 3)


class TestVisiumDownloadedWorkflow:
    """Smoke-test exact versioned vendor artifacts managed by the repository."""

    @pytest.mark.data
    @pytest.mark.parametrize(
        ("dataset_key", "dataset_id", "software_version"),
        [
            ("visium_v1_human_lymph_node", "V1_Human_Lymph_Node", "spaceranger-1.1.0"),
            ("visium_v2_1_mouse_brain", "V1_Adult_Mouse_Brain", "spaceranger-2.1.0"),
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
