"""Characterize the current public MERSCOPE workflow."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, cast

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import tifffile
from dask.callbacks import Callback
from shapely.geometry import MultiPolygon, Polygon
from spatialdata.models import get_table_keys
from spatialdata.transformations import get_transformation

from spatialdata_io import __version__
from spatialdata_io.readers.merscope import merscope

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata.transformations import BaseTransformation


def _write_layout(path: Path) -> np.ndarray:
    images_directory = path / "images"
    images_directory.mkdir()
    np.savetxt(images_directory / "micron_to_mosaic_pixel_transform.csv", np.eye(3))

    image = np.arange(4 * 5, dtype=np.uint16).reshape(4, 5)
    tifffile.imwrite(images_directory / "mosaic_DAPI_z0.tif", image)

    pd.DataFrame(
        {
            "global_x": [0.5, 3.5],
            "global_y": [1.5, 2.5],
            "global_z": [0.0, 0.0],
            "gene": ["gene-a", "gene-b"],
        }
    ).to_csv(path / "detected_transcripts.csv", index=False)

    polygons = gpd.GeoDataFrame(
        {
            "EntityID": ["cell-1", "cell-2"],
            "ZIndex": [0, 0],
            "Geometry": [
                MultiPolygon([Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]),
                MultiPolygon([Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])]),
            ],
        },
        geometry="Geometry",
    )
    polygons.to_parquet(path / "cell_boundaries.parquet")

    pd.DataFrame(
        {
            "gene-a": [1, 3],
            "gene-b": [2, 4],
            "Blank-1": [5, 6],
        },
        index=pd.Index(["cell-1", "cell-2"], name="cell"),
    ).to_csv(path / "cell_by_gene.csv")
    pd.DataFrame(
        {
            "center_x": [1.0, 3.0],
            "center_y": [1.0, 3.0],
            "volume": [10.0, 20.0],
        },
        index=pd.Index(["cell-1", "cell-2"], name="EntityID"),
    ).to_csv(path / "cell_metadata.csv")
    return image


class TestMerscope:
    """Characterize current raster, point, geometry, and linkage products."""

    def test_synthetic_layout_preserves_public_contract(self, tmp_path: Path) -> None:
        image = _write_layout(tmp_path)

        result = merscope(
            tmp_path,
            z_layers=0,
            region_name="region",
            slide_name="slide",
            backend="dask_image",
            image_models_kwargs={"chunks": (1, 4, 5), "scale_factors": None},
        )

        assert list(result.images) == ["slide_region_z0"]
        assert list(result.points) == ["slide_region_transcripts"]
        assert list(result.shapes) == ["slide_region_polygons"]
        assert list(result.tables) == ["table"]
        assert not result.labels
        assert result.attrs == {
            "spatialdata_io_reader": "merscope",
            "spatialdata_io_software_version": __version__,
        }

        image_element = result.images["slide_region_z0"]
        assert image_element.dims == ("c", "y", "x")
        assert image_element.shape == (1, 4, 5)
        assert image_element.dtype == np.dtype(np.uint16)
        assert image_element.chunksizes == {"c": (1,), "y": (4,), "x": (5,)}
        assert image_element.coords["c"].to_numpy().tolist() == ["DAPI"]
        np.testing.assert_array_equal(image_element.data[:, 1:3, 2:4].compute(), image[None, 1:3, 2:4])

        points = result.points["slide_region_transcripts"]
        assert isinstance(points, dd.DataFrame)
        assert points.npartitions == 1
        assert points.columns.tolist() == ["x", "y", "z", "gene"]
        assert str(points.dtypes["gene"]) == "category"
        computed_points = points.compute()
        assert computed_points["gene"].tolist() == ["gene-a", "gene-b"]
        np.testing.assert_array_equal(
            computed_points[["x", "y", "z"]].to_numpy(),
            np.asarray([[0.5, 1.5, 0.0], [3.5, 2.5, 0.0]]),
        )

        shapes = result.shapes["slide_region_polygons"]
        assert shapes.index.tolist() == ["cell-1", "cell-2"]
        assert shapes.geometry.geom_type.tolist() == ["MultiPolygon", "MultiPolygon"]
        np.testing.assert_array_equal(shapes.total_bounds, np.asarray([0.0, 0.0, 4.0, 4.0]))
        for element in (points, shapes):
            transformation = cast(
                "BaseTransformation",
                get_transformation(element, to_coordinate_system="global"),
            )
            np.testing.assert_array_equal(
                transformation.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
                np.eye(3),
            )

        table = result.tables["table"]
        np.testing.assert_array_equal(np.asarray(table.X), np.asarray([[1, 2], [3, 4]]))
        assert table.obs_names.tolist() == ["cell-1", "cell-2"]
        assert table.var_names.tolist() == ["gene-a", "gene-b"]
        assert table.obs["volume"].tolist() == [10.0, 20.0]
        assert table.obs["EntityID"].tolist() == ["cell-1", "cell-2"]
        np.testing.assert_array_equal(table.obsm["blank"].to_numpy(), np.asarray([[5], [6]]))
        np.testing.assert_array_equal(table.obsm["spatial"], np.asarray([[1.0, 1.0], [3.0, 3.0]]))
        assert get_table_keys(table) == ("slide_region_polygons", "cells_region", "EntityID")
        assert set(table.obs["EntityID"]) == set(shapes.index)

    def test_mosaic_construction_does_not_execute_pixel_tasks(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)
        executed: list[object] = []

        with Callback(pretask=lambda key, *_: executed.append(key)):
            result = merscope(
                tmp_path,
                z_layers=0,
                backend="dask_image",
                transcripts=False,
                cells_boundaries=False,
                cells_table=False,
                image_models_kwargs={"chunks": (1, 4, 5), "scale_factors": None},
            )

        assert not executed
        assert list(result.images) == [f"{tmp_path.parent.name}_{tmp_path.name}_z0"]

    def test_vpt_directory_selects_postprocessed_artifacts(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)
        vpt_directory = tmp_path / "vpt"
        vpt_directory.mkdir()
        shutil.copyfile(tmp_path / "cell_by_gene.csv", vpt_directory / "cell_by_gene.csv")
        shutil.copyfile(tmp_path / "cell_metadata.csv", vpt_directory / "cell_metadata.csv")
        shutil.copyfile(tmp_path / "cell_boundaries.parquet", vpt_directory / "cellpose_micron_space.parquet")

        result = merscope(
            tmp_path,
            vpt_outputs=vpt_directory,
            region_name="region",
            slide_name="slide",
            transcripts=False,
            mosaic_images=False,
        )

        assert not result.images
        assert not result.points
        assert list(result.shapes) == ["slide_region_polygons"]
        assert list(result.tables) == ["table"]
        assert result.tables["table"].obs_names.tolist() == ["cell-1", "cell-2"]
