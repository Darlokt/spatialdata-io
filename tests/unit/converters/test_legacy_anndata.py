from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from anndata import AnnData
from anndata.tests.helpers import assert_equal
from scipy.sparse import csr_array
from spatialdata import SpatialData, get_element_instances
from spatialdata.datasets import blobs
from spatialdata.models import TableModel
from spatialdata.testing import assert_spatial_data_objects_are_identical
from spatialdata.transformations import BaseTransformation, Identity, Scale, get_transformation, set_transformation

from spatialdata_io.converters.legacy_anndata import (
    _linked_circles,
    from_legacy_anndata,
    to_legacy_anndata,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

BlobsTypes = Literal["blobs_labels", "blobs_circles", "blobs_polygons", "blobs_multipolygons"]


@pytest.fixture
def spatial_table() -> SpatialData:
    adata = AnnData(X=csr_array([[1], [2]], dtype=np.int16))
    adata.obsm["spatial"] = np.array([[1.0, 2.0], [3.0, 4.0]])
    return from_legacy_anndata(adata)


def blobs_annotating_element(name: BlobsTypes) -> SpatialData:
    sdata = blobs(length=50)
    instance_id = (
        get_element_instances(sdata.labels[name]).tolist()
        if name == "blobs_labels"
        else sdata.shapes[name].index.tolist()
    )
    n = len(instance_id)
    new_table = AnnData(shape=(n, 0), obs={"region": [name for _ in range(n)], "instance_id": instance_id})
    new_table = TableModel.parse(new_table, region=name, region_key="region", instance_key="instance_id")
    del sdata["table"]
    sdata["table"] = new_table
    return sdata


def _adjust_table_for_idempotency_check(sdata: SpatialData, *, include_images: bool) -> SpatialData:
    """We need to adjust the coordinate systems from from_legacy_anndata() to perform the idempotency check."""
    if not include_images:
        # nothing needs to be adjusted
        return sdata
    cs_name = "blobs_multiscale_image_downscaled_hires"
    sdata = sdata.filter_by_coordinate_system(cs_name)
    sdata.rename_coordinate_systems({cs_name: "global"})
    t = get_transformation(sdata.shapes["locations"], "global")
    assert isinstance(t, BaseTransformation)
    set_transformation(sdata.shapes["locations"], {"global": t}, set_all=True)
    image_name = "blobs_multiscale_image_hires_image"
    im = sdata.images[image_name]
    del sdata.images[image_name]
    sdata.images["blobs_multiscale_image"] = im
    return sdata


def idempotency_check_to_anndata(sdata0: SpatialData, *, include_images: bool) -> None:
    adata0 = to_legacy_anndata(sdata0, include_images=include_images)
    sdata1 = from_legacy_anndata(adata0)
    sdata1 = _adjust_table_for_idempotency_check(sdata1, include_images=include_images)
    adata2 = to_legacy_anndata(sdata1, include_images=include_images)
    assert_equal(adata0, adata2)


def idempotency_check_from_anndata(adata0: AnnData, *, include_images: bool) -> None:
    sdata0 = from_legacy_anndata(adata0)
    sdata0 = _adjust_table_for_idempotency_check(sdata0, include_images=include_images)
    adata1 = to_legacy_anndata(sdata0, include_images=include_images)
    sdata1 = from_legacy_anndata(adata1)
    sdata1 = _adjust_table_for_idempotency_check(sdata1, include_images=include_images)
    assert_spatial_data_objects_are_identical(sdata0, sdata1)


@pytest.mark.parametrize("name", ["blobs_labels", "blobs_circles", "blobs_polygons", "blobs_multipolygons"])
@pytest.mark.parametrize("include_images", [False, True])
def test_bidirectional_conversion(name: BlobsTypes, *, include_images: bool) -> None:
    if include_images:
        pytest.skip(
            "include_images=True can't be tested because the bug "
            "https://github.com/scverse/spatialdata/issues/165 causes a large error"
        )
    sdata0 = blobs_annotating_element(name)
    adata0 = to_legacy_anndata(sdata0, include_images=include_images)
    sdata1 = from_legacy_anndata(adata0)
    sdata1 = _adjust_table_for_idempotency_check(sdata1, include_images=include_images)

    idempotency_check_to_anndata(sdata1, include_images=include_images)
    idempotency_check_from_anndata(adata0, include_images=include_images)


class TestFromLegacyAnndata:
    @pytest.mark.parametrize(
        ("scalefactors", "expected_scale"),
        [
            ({"hires": 2.0, "tissue_hires_scalef": 3.0}, 2.0),
            ({"tissue_hires_scalef": 3.0}, 3.0),
            ({"hires": None, "tissue_hires_scalef": 3.0}, 1.0),
            ({}, 1.0),
        ],
    )
    def test_preserves_images_coordinates_counts_and_caller(
        self, scalefactors: dict[str, float | None], expected_scale: float
    ) -> None:
        adata = AnnData(
            X=csr_array([[3], [5]], dtype=np.int16),
            obs=pd.DataFrame(index=["a", "b"]),
            var=pd.DataFrame(index=["gene"]),
        )
        coordinates = np.array([[1.0, 2.0], [3.0, 4.0]])
        image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        adata.obsm["spatial"] = coordinates
        adata.uns["spatial"] = {
            "sample": {"images": {"hires": image}, "scalefactors": {**scalefactors, "spot_diameter_fullres": 6.0}}
        }
        original = adata.copy()

        sdata = from_legacy_anndata(adata)

        assert set(sdata.images) == {"sample_hires"}
        assert set(sdata.shapes) == {"locations"}
        assert set(sdata.tables) == {"table"}
        actual_image = sdata.images["sample_hires"]
        assert actual_image.dims == ("c", "y", "x")
        np.testing.assert_array_equal(actual_image.data.compute(), image.transpose(2, 0, 1))
        locations = sdata.shapes["locations"]
        np.testing.assert_array_equal(locations.geometry.x, coordinates[:, 0])
        np.testing.assert_array_equal(locations.geometry.y, coordinates[:, 1])
        np.testing.assert_array_equal(locations["radius"], [3.0, 3.0])
        transformation = get_transformation(locations, "sample_hires")
        assert isinstance(transformation, Scale)
        np.testing.assert_array_equal(transformation.scale, [expected_scale, expected_scale])
        table = sdata.tables["table"]
        assert table.obs_names.tolist() == ["a", "b"]
        assert table.var_names.tolist() == ["gene"]
        assert table.obs["region"].tolist() == ["locations", "locations"]
        assert table.obs["instance_id"].tolist() == locations.index.tolist()
        assert_equal(table.X, adata.X)
        assert_equal(adata, original)

    def test_copies_nonspatial_table(self) -> None:
        adata = AnnData(X=csr_array([[1]], dtype=np.int16))

        sdata = from_legacy_anndata(adata)

        assert not sdata.images
        assert not sdata.shapes
        assert sdata.tables["table"] is not adata
        assert_equal(sdata.tables["table"], adata)

    def test_rejects_xarray_observation_metadata(self) -> None:
        adata = AnnData(shape=(1, 0))
        adata.obs = xr.Dataset({"kind": ("cell", ["one"])}, coords={"cell": ["a"]})
        adata.obsm["spatial"] = np.array([[1.0, 2.0]])

        with pytest.raises(TypeError, match="pandas DataFrame observation metadata"):
            from_legacy_anndata(adata)


class TestToLegacyAnndata:
    def test_rejects_missing_coordinate_system(self) -> None:
        with pytest.raises(ValueError, match="exactly one coordinate system"):
            to_legacy_anndata(SpatialData())

    def test_rejects_ambiguous_coordinate_system(self, spatial_table: SpatialData) -> None:
        set_transformation(spatial_table.shapes["locations"], Identity(), to_coordinate_system="other")

        with pytest.raises(ValueError, match="exactly one coordinate system"):
            to_legacy_anndata(spatial_table)

    def test_rejects_unknown_coordinate_system(self, spatial_table: SpatialData) -> None:
        with pytest.raises(ValueError, match="does not have the coordinate system missing"):
            to_legacy_anndata(spatial_table, coordinate_system="missing")

    @pytest.mark.parametrize("table_names", [(), ("first", "second")])
    def test_rejects_missing_or_ambiguous_table(self, spatial_table: SpatialData, table_names: tuple[str, ...]) -> None:
        table = spatial_table.tables.pop("table")
        for name in table_names:
            spatial_table.tables[name] = table.copy()

        with pytest.raises(ValueError, match="exactly one table"):
            to_legacy_anndata(spatial_table)

    def test_rejects_unknown_table(self, spatial_table: SpatialData) -> None:
        with pytest.raises(ValueError, match="table missing is not present"):
            to_legacy_anndata(spatial_table, table_name="missing")

    def test_accepts_explicit_coordinate_system_and_table(self, spatial_table: SpatialData) -> None:
        original = spatial_table.tables["table"].copy()

        adata = to_legacy_anndata(spatial_table, coordinate_system="global", table_name="table")

        np.testing.assert_array_equal(adata.obsm["spatial"], [[1.0, 2.0], [3.0, 4.0]])
        assert_equal(spatial_table.tables["table"], original)
        assert_equal(adata.X, original.X)


class TestLinkedCircles:
    def test_rejects_empty_linkage(self, spatial_table: SpatialData, mocker: MockerFixture) -> None:
        # Exercise the converter's guard independently of upstream empty-join handling.
        empty_table = spatial_table.tables["table"][:0].copy()
        mocker.patch(
            "spatialdata_io.converters.legacy_anndata.join_spatialelement_table",
            return_value=({"locations": spatial_table.shapes["locations"].iloc[:0]}, empty_table),
        )

        with pytest.raises(ValueError, match="does not annotate any geometry"):
            _linked_circles(spatial_table, "table")
