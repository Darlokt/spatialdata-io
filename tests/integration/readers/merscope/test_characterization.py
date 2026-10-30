"""Preserve valid MERSCOPE values while testing explicit scientific corrections."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Literal, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from dask.callbacks import Callback
from scipy import sparse
from shapely.geometry import MultiPolygon, Polygon
from spatialdata.models import get_table_keys
from spatialdata.transformations import get_transformation

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.merscope import merscope
from tests.integration.readers.merscope._fixtures import MerscopeFixture, write_merscope_dataset

if TYPE_CHECKING:
    from pathlib import Path
    from typing import NotRequired, TypedDict

    from anndata import AnnData
    from dask.dataframe import DataFrame
    from spatialdata import SpatialData
    from spatialdata.transformations import BaseTransformation


if TYPE_CHECKING:

    class _PointMetadata(TypedDict):
        feature_key: str
        instance_key: NotRequired[str]

    class _PointAttributes(TypedDict):
        spatialdata_attrs: _PointMetadata


def _instance_key(points: DataFrame) -> str | None:
    """Read SpatialData's dynamically attached metadata absent from Dask stubs."""
    return cast("_PointAttributes", points.attrs)["spatialdata_attrs"].get("instance_key")


def _read(
    fixture: MerscopeFixture,
    *,
    images: bool = True,
    transcripts: bool = True,
    boundaries: bool = True,
    table: bool = True,
) -> SpatialData:
    """Exercise all supported artifacts with fixed names and requested z order."""
    return merscope(
        fixture.root,
        vpt_directory=None if fixture.profile == "standard" else fixture.analysis,
        z_layers=(2, 0),
        region_name="region",
        slide_name="slide",
        include_transcripts=transcripts,
        include_cell_boundaries=boundaries,
        include_table=table,
        include_images=images,
        chunks=(1, 16, 20),
    )


def _assert_table(table: AnnData, fixture: MerscopeFixture) -> None:
    """Compare sparse source measurements without constructing dense expression."""
    increment = 0 if fixture.profile == "standard" else 100
    assert isinstance(table.X, sparse.csr_array)
    assert not isinstance(table.X, sparse.spmatrix)
    assert table.X.shape == (3, 3)
    assert table.X.dtype == np.dtype("int64")
    np.testing.assert_array_equal(table.X.data, [1 + increment, 4 + increment, 3 + increment, 1 + increment])
    np.testing.assert_array_equal(table.X.indices, [0, 1, 0, 1])
    np.testing.assert_array_equal(table.X.indptr, [0, 1, 2, 4])
    assert table.obs_names.tolist() == list(fixture.cell_ids)
    assert table.var_names.tolist() == ["GeneB", "GeneA", "ZeroGene"]
    blanks = table.obsm["blank"]
    assert isinstance(blanks, sparse.csr_array)
    assert blanks.shape == (3, 1)
    np.testing.assert_array_equal(blanks.data, [2 + increment, 5 + increment])
    np.testing.assert_array_equal(blanks.indices, [0, 0])
    np.testing.assert_array_equal(blanks.indptr, [0, 1, 2, 2])
    assert table.uns["blank_names"] == ["Blank-1"]
    np.testing.assert_array_equal(table.obsm["spatial"], [[2, 2], [10, 1], [21, 1]])
    assert table.obs["annotation"].tolist() == ["A", "B", "C"]
    assert table.obs["EntityID"].tolist() == list(fixture.cell_ids)
    assert get_table_keys(table) == ("slide_region_polygons", "cells_region", "EntityID")


class TestMerscopeCharacterization:
    """Read actual standard and VPT artifacts through the public transaction."""

    @pytest.mark.parametrize("profile", ["standard", "cellpose", "watershed"])
    def test_supported_profiles_preserve_values_and_frames(
        self, tmp_path: Path, profile: Literal["standard", "cellpose", "watershed"]
    ) -> None:
        fixture = write_merscope_dataset(tmp_path, profile=profile)
        result = _read(fixture)
        assert list(result.images) == ["slide_region_z2", "slide_region_z0"]
        assert list(result.points) == ["slide_region_transcripts"]
        assert list(result.shapes) == ["slide_region_polygons"]
        assert list(result.tables) == ["table"]
        assert not result.labels
        for z in (2, 0):
            image = result.images[f"slide_region_z{z}"]
            assert image.dims == ("c", "y", "x")
            assert image.shape == (2, 64, 80)
            assert image.dtype == np.dtype("uint16")
            assert image.coords["c"].to_numpy().tolist() == ["Cyto", "DAPI"]
            image_transform = get_transformation(image, to_coordinate_system="global")
            assert not isinstance(image_transform, dict)
            np.testing.assert_array_equal(
                image_transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
                np.linalg.inv([[2, 0.25, 10], [0, 3, -7], [0, 0, 1]]),
            )
            for channel, offset in (("DAPI", 0), ("Cyto", 10_000)):
                np.testing.assert_array_equal(
                    image.sel(c=channel).data[2:4, 3:5].compute(),
                    np.asarray([[163, 164], [243, 244]]) + offset + z * 100,
                )
        _assert_table(result.tables["table"], fixture)
        boundaries = result.shapes["slide_region_polygons"]
        assert boundaries.index.tolist() == list(fixture.cell_ids)
        first = boundaries.geometry.iloc[0]
        assert isinstance(first, MultiPolygon)
        assert len(first.geoms) == 2
        assert sum(len(polygon.interiors) for polygon in first.geoms) == 1
        np.testing.assert_array_equal(sorted(polygon.area for polygon in first.geoms), [1, 18])
        np.testing.assert_array_equal(boundaries.geometry.area.to_numpy(), [19, 4, 4])
        points = result.points["slide_region_transcripts"]
        for element in (boundaries, points):
            transform = cast("BaseTransformation", get_transformation(element, to_coordinate_system="global"))
            np.testing.assert_array_equal(
                transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
                np.eye(3),
            )
        materialized = points.compute()
        assert materialized["gene"].tolist() == ["GeneB", "GeneA", "Blank-1", "ZeroGene"]
        assert materialized["gene"].cat.categories.tolist() == ["GeneB", "Blank-1", "GeneA", "ZeroGene"]
        assignment = "cell_id" if profile == "standard" else "original_cell_id"
        assert materialized[assignment].tolist() == ["1001", "2002", "3003", "-1"]
        assert _instance_key(points) == ("cell_id" if profile == "standard" else None)
        assert "z" not in materialized
        np.testing.assert_array_equal(materialized[["x", "y"]], [[2, 2], [10, 1], [21, 1], [8, 6]])
        np.testing.assert_array_equal(materialized["z_raw"], [0, 2, 2, 0])

    @pytest.mark.parametrize("profile", ["standard", "cellpose", "watershed"])
    @pytest.mark.parametrize("included", list(product((False, True), repeat=4)))
    def test_all_product_selections(
        self,
        tmp_path: Path,
        profile: Literal["standard", "cellpose", "watershed"],
        included: tuple[bool, bool, bool, bool],
    ) -> None:
        fixture = write_merscope_dataset(tmp_path, profile=profile)
        images, transcripts, boundaries, table = included
        result = _read(fixture, images=images, transcripts=transcripts, boundaries=boundaries, table=table)
        assert bool(result.images) is images
        assert bool(result.points) is transcripts
        assert bool(result.shapes) is boundaries
        assert bool(result.tables) is table
        if table:
            assert ("spatialdata_attrs" in result.tables["table"].uns) is boundaries

    def test_all_z_footprint_retains_linked_cells_and_extending_slices(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path, incomplete_z0=True)
        result = _read(fixture, images=False, transcripts=False)
        shapes = result.shapes["slide_region_polygons"]
        assert shapes.index.tolist() == ["1001", "2002", "3003"]
        assert set(result.tables["table"].obs_names) == set(shapes.index)
        np.testing.assert_array_equal(shapes.geometry.area, [19, 4, 4])

    def test_blank_substrings_remain_expression(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path, extra_gene=True)
        result = _read(fixture, images=False, transcripts=False)
        table = result.tables["table"]
        assert table.var_names.tolist() == ["GeneB", "GeneA", "ZeroGene", "BlanketGene", "blank-2"]
        assert table.uns["blank_names"] == ["Blank-1"]
        assert isinstance(table.obsm["blank"], sparse.csr_array)

    def test_vpt_root_assignments_are_honest_original_annotations(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path, profile="cellpose")
        result = _read(fixture, images=False)
        points = result.points["slide_region_transcripts"]
        assert "cell_id" not in points.columns
        original = set(points.compute()["original_cell_id"]) - {"-1"}
        assert original.isdisjoint(result.shapes["slide_region_polygons"].index)
        assert _instance_key(points) is None

    def test_construction_executes_no_transcript_or_pixel_tasks(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path)
        completed: list[str] = []
        with Callback(posttask=lambda key, *_: completed.append(str(key))):
            _read(fixture)
        assert completed == []

    def test_reordered_cell_metadata_aligns_by_exact_identity(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path)
        metadata_path = fixture.analysis / "cell_metadata.csv"
        metadata = pd.read_csv(metadata_path, index_col=0)
        metadata.iloc[::-1].to_csv(metadata_path)
        result = _read(fixture, images=False, transcripts=False)
        assert result.tables["table"].obs_names.tolist() == list(fixture.cell_ids)
        assert result.tables["table"].obs["annotation"].tolist() == ["A", "B", "C"]
        np.testing.assert_array_equal(result.tables["table"].obsm["spatial"], [[2, 2], [10, 1], [21, 1]])

    def test_ambiguous_vpt_boundaries_fail(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path, profile="watershed")
        cellpose = gpd.read_parquet(fixture.analysis / "watershed_micron_space.parquet")
        cellpose.to_parquet(fixture.analysis / "cellpose_micron_space.parquet")
        with pytest.raises(ReaderFormatError, match=r"ambiguous|both|boundar"):
            _read(fixture, images=False, transcripts=False)

    def test_available_reassigned_vpt_transcripts_are_preferred(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path, profile="cellpose", reassigned_transcripts=True)
        result = _read(fixture, images=False)
        points = result.points["slide_region_transcripts"]
        assert "original_cell_id" not in points.columns
        assert points.compute()["cell_id"].tolist() == ["9001", "9002", "9003", "-1"]
        assert _instance_key(points) == "cell_id"

    @pytest.mark.parametrize("artifact", ["detected_transcripts.csv", "cell_boundaries.parquet"])
    def test_missing_requested_products_fail(self, tmp_path: Path, artifact: str) -> None:
        fixture = write_merscope_dataset(tmp_path)
        (fixture.root / artifact).unlink()
        with pytest.raises(FileNotFoundError, match=artifact.replace(".", r"\.")):
            _read(fixture, images=False, table=False)

    def test_none_z_layers_loads_all_discovered_layers_in_order(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path)
        result = merscope(
            fixture.root, z_layers=None, include_transcripts=False, include_cell_boundaries=False, include_table=False
        )
        assert list(result.images) == ["slide_region_z0", "slide_region_z2"]

    def test_all_products_disabled_needs_no_affine(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path)
        (fixture.root / "images" / "micron_to_mosaic_pixel_transform.csv").unlink()
        result = _read(fixture, images=False, transcripts=False, boundaries=False, table=False)
        assert not result.images
        assert not result.points
        assert not result.shapes
        assert not result.tables

    def test_invalid_geometry_fails_without_dropping_a_linked_cell(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path)
        path = fixture.analysis / "cell_boundaries.parquet"
        boundaries = gpd.read_parquet(path)
        geometries = boundaries.geometry.to_list()
        geometries[2] = MultiPolygon([Polygon([(9, 0), (11, 2), (9, 2), (11, 0)])])
        boundaries.geometry = gpd.GeoSeries(geometries, index=boundaries.index)
        boundaries.to_parquet(path)
        with pytest.raises(ReaderFormatError, match=r"invalid|valid|geometry"):
            merscope(fixture.root, include_images=False, include_transcripts=False, invalid_geometries="raise")
