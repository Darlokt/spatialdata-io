"""Characterize the pre-centralization Visium reader contract."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import geopandas as gpd
import h5py
import imagecodecs
import numpy as np
from scipy import sparse
from spatialdata.models import ShapesModel, get_table_keys
from spatialdata.transformations import BaseTransformation, get_transformation

from spatialdata_io import __version__
from spatialdata_io._constants._constants import VisiumKeys
from spatialdata_io.readers.visium import visium

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from pytest_mock import MockerFixture
    from xarray import DataArray


def _write_10x_h5(path: Path) -> None:
    """Write the smallest Scanpy-compatible 10x v3 feature matrix."""
    with h5py.File(path, "w") as file:
        file.attrs["library_ids"] = np.asarray([b"source-library"])
        file.attrs["chemistry_description"] = b"Synthetic chemistry"
        file.attrs["software_version"] = b"9.9.9"

        matrix = file.create_group("matrix")
        matrix.create_dataset("barcodes", data=np.asarray([b"barcode-a", b"barcode-b", b"barcode-c"]))
        matrix.create_dataset("data", data=np.asarray([1, 2, 3, 4], dtype=np.int32))
        matrix.create_dataset("indices", data=np.asarray([0, 1, 0, 1], dtype=np.int32))
        matrix.create_dataset("indptr", data=np.asarray([0, 1, 2, 4], dtype=np.int32))
        # 10x stores feature-by-barcode shape; Scanpy returns barcode-by-feature.
        matrix.create_dataset("shape", data=np.asarray([2, 3], dtype=np.int64))

        features = matrix.create_group("features")
        features.create_dataset("id", data=np.asarray([b"gene-id-a", b"gene-id-b"]))
        features.create_dataset("name", data=np.asarray([b"gene-a", b"gene-b"]))
        features.create_dataset("feature_type", data=np.asarray([b"Gene Expression", b"Gene Expression"]))


def _write_png(path: Path, data: np.ndarray) -> None:
    path.write_bytes(imagecodecs.png_encode(data))


def _write_visium_layout(path: Path) -> None:
    _write_10x_h5(path / f"sample_{VisiumKeys.FILTERED_COUNTS_FILE}")

    spatial = path / "spatial"
    spatial.mkdir()
    (spatial / VisiumKeys.SPOTS_FILE_2).write_text(
        "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n"
        "barcode-a,1,0,2,10.0,20.0\n"
        "barcode-b,0,1,1,30.0,40.0\n"
        "barcode-c,1,2,0,50.0,60.0\n",
        encoding="utf-8",
    )
    (spatial / VisiumKeys.SCALEFACTORS_FILE).write_text(
        json.dumps(
            {
                VisiumKeys.SCALEFACTORS_HIRES: 0.5,
                VisiumKeys.SCALEFACTORS_LOWRES: 0.25,
                "spot_diameter_fullres": 6.0,
            }
        ),
        encoding="utf-8",
    )

    hires = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    lowres = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    _write_png(path / VisiumKeys.IMAGE_HIRES_FILE, hires)
    _write_png(path / VisiumKeys.IMAGE_LOWRES_FILE, lowres)


def _affine_matrix(element: DataArray | gpd.GeoDataFrame, coordinate_system: str) -> np.ndarray:
    transformation = cast(
        "BaseTransformation",
        get_transformation(element, to_coordinate_system=coordinate_system),
    )
    return np.asarray(transformation.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")))


class TestVisium:
    """Characterize the public Visium workflow before its migration cohort."""

    def test_synthetic_layout_preserves_numerical_and_structural_contract(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        _write_visium_layout(tmp_path)

        # The current reader passes a pandas DataFrame to ShapesModel.parse(), a
        # boundary rejected by the current SpatialData release. Keep Phase 0 a
        # characterization-only change by adapting only that call in the test;
        # the completion record tracks the reader defect for the Visium cohort.
        parse_shapes = ShapesModel.parse

        def parse_legacy_circles(
            data: pd.DataFrame,
            *,
            geometry: int,
            radius: float,
            index: pd.Series[int],
            transformations: dict[str, object],
        ) -> gpd.GeoDataFrame:
            assert geometry == 0
            circles = gpd.GeoDataFrame(
                {
                    "geometry": gpd.points_from_xy(data[VisiumKeys.SPOTS_X], data[VisiumKeys.SPOTS_Y]),
                    "radius": radius,
                },
                index=index.to_numpy(),
            )
            return parse_shapes(circles, transformations=transformations)

        mocker.patch("spatialdata_io.readers.visium.ShapesModel.parse", side_effect=parse_legacy_circles)

        sdata = visium(tmp_path)

        assert set(sdata.images) == {"sample_hires_image", "sample_lowres_image"}
        assert set(sdata.shapes) == {"sample"}
        assert set(sdata.tables) == {"table"}
        assert set(sdata.coordinate_systems) == {
            "sample",
            "sample_downscaled_hires",
            "sample_downscaled_lowres",
        }
        assert sdata.attrs == {
            "spatialdata_io_software_version": __version__,
            "spatialdata_io_reader": "visium",
        }

        hires = sdata.images["sample_hires_image"]
        lowres = sdata.images["sample_lowres_image"]
        assert hires.dims == ("c", "y", "x")
        assert hires.shape == (3, 4, 5)
        assert hires.dtype == np.dtype(np.uint8)
        assert lowres.dims == ("c", "y", "x")
        assert lowres.shape == (3, 2, 3)
        np.testing.assert_array_equal(hires.data[:, 1, 2].compute(), np.asarray([21, 22, 23], dtype=np.uint8))

        np.testing.assert_array_equal(
            _affine_matrix(hires, "sample_downscaled_hires"),
            np.eye(3),
        )
        np.testing.assert_array_equal(
            _affine_matrix(hires, "sample"),
            np.diag([2.0, 2.0, 1.0]),
        )
        np.testing.assert_array_equal(
            _affine_matrix(lowres, "sample_downscaled_lowres"),
            np.eye(3),
        )
        np.testing.assert_array_equal(
            _affine_matrix(lowres, "sample"),
            np.diag([4.0, 4.0, 1.0]),
        )

        shapes = sdata.shapes["sample"]
        assert shapes.index.tolist() == [0, 1, 2]
        np.testing.assert_array_equal(shapes.geometry.x.to_numpy(), np.asarray([20.0, 40.0, 60.0]))
        np.testing.assert_array_equal(shapes.geometry.y.to_numpy(), np.asarray([10.0, 30.0, 50.0]))
        np.testing.assert_array_equal(shapes["radius"].to_numpy(), np.asarray([3.0, 3.0, 3.0]))
        np.testing.assert_array_equal(
            _affine_matrix(shapes, "sample_downscaled_hires"),
            np.diag([0.5, 0.5, 1.0]),
        )
        np.testing.assert_array_equal(
            _affine_matrix(shapes, "sample_downscaled_lowres"),
            np.diag([0.25, 0.25, 1.0]),
        )

        table = sdata.tables["table"]
        assert sparse.isspmatrix_csr(table.X)
        # The fixture is intentionally bounded at 3 observations by 2 features.
        np.testing.assert_array_equal(table.X.toarray(), np.asarray([[1, 0], [0, 2], [3, 4]], dtype=np.float32))
        assert table.obs_names.tolist() == ["barcode-a", "barcode-b", "barcode-c"]
        assert table.var_names.tolist() == ["gene-a", "gene-b"]
        assert table.var["gene_ids"].tolist() == ["gene-id-a", "gene-id-b"]
        assert table.var["feature_types"].tolist() == ["Gene Expression", "Gene Expression"]
        assert table.obs["in_tissue"].tolist() == [1, 0, 1]
        assert table.obs["array_row"].tolist() == [0, 1, 2]
        assert table.obs["array_col"].tolist() == [2, 1, 0]
        assert table.obs["spot_id"].tolist() == [0, 1, 2]
        assert table.obs["spot_id"].tolist() == shapes.index.tolist()
        assert table.obs["region"].tolist() == ["sample", "sample", "sample"]
        np.testing.assert_array_equal(
            table.obsm["spatial"],
            np.asarray([[20.0, 10.0], [40.0, 30.0], [60.0, 50.0]]),
        )
        assert table.uns["spatial"] == {
            "sample": {
                "metadata": {
                    "chemistry_description": "Synthetic chemistry",
                    "software_version": "9.9.9",
                }
            }
        }
        assert get_table_keys(table) == ("sample", "region", "spot_id")
