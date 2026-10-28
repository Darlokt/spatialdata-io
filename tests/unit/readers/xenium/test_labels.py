"""Path-backed lazy labels: exact values, chunks, serializable graphs, bounded reads."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
from dask.callbacks import Callback
from spatialdata.transformations import Scale, get_transformation
from xarray import DataArray, DataTree

from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.xenium import _labels
from spatialdata_io.readers.xenium._labels import MaskSource, read_labels
from tests.integration.readers.xenium import _fixtures as fx

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

# Requested chunks equal to the vendor chunks keep one task per stored chunk.
NATIVE_CHUNKS = make_raster_options(chunks=(1, *fx.MASK_CHUNKS))


def _source(tmp_path: Path) -> tuple[MaskSource, fx.XeniumFixture]:
    bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_2_0)
    source = MaskSource(ArtifactFile(tmp_path / "cells.zarr.zip"), "masks/1", fx.SHAPE, fx.MASK_CHUNKS)
    return source, bundle


class TestMaskSource:
    """A mask source reads one selection through a store it opens and closes itself."""

    def test_reads_one_selection_and_pickles_as_a_path(self, tmp_path: Path) -> None:
        source, bundle = _source(tmp_path)

        np.testing.assert_array_equal(source[slice(2, 10), slice(0, 8)], bundle.cell_mask[2:10, 0:8])
        assert pickle.loads(pickle.dumps(source)) == source  # noqa: S301, RUF100 - Test-owned source.
        assert source.ndim == 2


class TestReadLabels:
    """Labels are exact, lazy, chunked as requested and serializable without open stores."""

    def test_values_chunks_transform_and_laziness(self, tmp_path: Path) -> None:
        source, bundle = _source(tmp_path)
        tasks: list[object] = []

        with Callback(posttask=lambda *args: tasks.append(args)):
            labels = read_labels(source, pixel_size=fx.PIXEL_SIZE, raster=make_raster_options())

        assert tasks == []
        assert isinstance(labels, DataArray)
        assert labels.dims == ("y", "x")
        assert labels.dtype == np.uint32
        # Default 1024-pixel tiles merge the native mask chunks of this small mask.
        assert labels.data.chunksize == fx.SHAPE
        transformation = get_transformation(labels, to_coordinate_system="global")
        assert isinstance(transformation, Scale)
        np.testing.assert_array_equal(transformation.scale, [fx.PIXEL_SIZE, fx.PIXEL_SIZE])
        assert transformation.axes == ("y", "x")
        np.testing.assert_array_equal(labels.data.compute(), bundle.cell_mask)

    def test_graph_holds_no_open_store(self, tmp_path: Path) -> None:
        source, _ = _source(tmp_path)
        labels = read_labels(source, pixel_size=fx.PIXEL_SIZE, raster=NATIVE_CHUNKS)

        graph = dict(labels.data.__dask_graph__())

        assert pickle.loads(pickle.dumps(graph)).keys() == graph.keys()  # noqa: S301, RUF100 - Test-owned graph.

    def test_partial_read_opens_only_intersecting_chunks(self, tmp_path: Path, mocker: MockerFixture) -> None:
        source, bundle = _source(tmp_path)
        labels = read_labels(source, pixel_size=fx.PIXEL_SIZE, raster=NATIVE_CHUNKS)
        opened = mocker.spy(_labels, "ZipStore")

        window = labels.data[0:8, 8:16].compute()

        np.testing.assert_array_equal(window, bundle.cell_mask[0:8, 8:16])
        assert opened.call_count == 1

    def test_raster_options(self, tmp_path: Path) -> None:
        source, bundle = _source(tmp_path)

        labels = read_labels(source, pixel_size=fx.PIXEL_SIZE, raster=make_raster_options((2,), (1, 4, 16)))

        assert isinstance(labels, DataTree)
        scale0 = labels["scale0"]["image"]
        assert scale0.data.chunksize == (4, 16)
        np.testing.assert_array_equal(scale0.data.compute(), bundle.cell_mask)
        assert labels["scale1"]["image"].shape == (12, 16)
