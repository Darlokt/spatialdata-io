"""Prove DBiT histology composition, lazy decoding, and task-owned resources."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import imagecodecs
import numpy as np
import pytest
import tifffile
from dask.callbacks import Callback
from spatialdata.transformations import Identity, get_transformation
from tifffile.zarr import ZarrTiffStore
from xarray import DataArray

import spatialdata_io.readers._utils.image._encoded as encoded
import spatialdata_io.readers._utils.image.tiff._dask as tiff_tasks
from spatialdata_io.readers.dbit import dbit
from tests.integration.readers.dbit.test_reader import _write_layout

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestDbitImages:
    """Keep image semantics and grid coordinates correct without eager pixel I/O."""

    @pytest.mark.parametrize("compressed", [False, True])
    def test_native_chunks_partial_reads_serialization_and_closure(
        self, tmp_path: Path, mocker: MockerFixture, *, compressed: bool
    ) -> None:
        _write_layout(tmp_path)
        source = np.arange(2 * 32 * 48, dtype=np.uint16).reshape(2, 32, 48)
        path = tmp_path / "histology.tif"
        tifffile.imwrite(
            path,
            source,
            tile=(16, 16),
            compression="deflate" if compressed else None,
            photometric="minisblack",
            metadata={"axes": "CYX"},
        )
        decode = mocker.spy(tiff_tasks, "_read_tiff_selection")
        tasks: list[str] = []

        with Callback(pretask=lambda key, *_: tasks.append(str(key))):
            result = dbit(tmp_path, image_file="histology.tif")

        assert not tasks
        assert decode.call_count == 0

        image = result.images["sample_image"]
        assert isinstance(image, DataArray)
        assert image.dims == ("c", "y", "x")
        assert image.shape == source.shape
        assert image.dtype == np.dtype(np.uint16)
        assert image.data.chunksize == (1, 16, 16)
        assert image.coords["c"].to_numpy().tolist() == [0, 1]
        assert get_transformation(image, get_all=True) == {"global": Identity()}

        # Pickle before installing resource trackers: the graph must retain only paths and selection state.
        restored = pickle.loads(pickle.dumps(image.data))  # noqa: S301, RUF100 - Only deserialize this test's own graph.
        close_file = mocker.spy(tifffile.FileHandle, "close")
        close_store = mocker.spy(ZarrTiffStore, "close")

        np.testing.assert_array_equal(restored[:1, 1:3, 2:5].compute(scheduler="synchronous"), source[:1, 1:3, 2:5])

        assert decode.call_count == 1
        assert decode.call_args.args[1] == (slice(0, 1), slice(0, 16), slice(0, 16))
        assert close_file.call_count >= 1
        assert close_store.call_count == 1

    @pytest.mark.parametrize("suffix", [".png", ".jpeg"])
    def test_encoded_rgb_is_lazy_and_closes_whole_frame_decode(
        self, tmp_path: Path, mocker: MockerFixture, suffix: str
    ) -> None:
        _write_layout(tmp_path)
        source = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
        path = tmp_path / f"histology{suffix}"
        if suffix == ".png":
            path.write_bytes(imagecodecs.png_encode(source))
            expected = source
        else:
            path.write_bytes(imagecodecs.jpeg8_encode(source, level=95))
            expected = imagecodecs.jpeg8_decode(path.read_bytes())
        decode = mocker.spy(encoded.imagecodecs, "png_decode" if suffix == ".png" else "jpeg8_decode")

        result = dbit(tmp_path, image_file=path, chunks=(1, 4, 5))

        assert decode.call_count == 0
        image = result.images["sample_image"]
        assert isinstance(image, DataArray)
        assert image.data.chunksize == (1, 4, 5)
        assert image.coords["c"].to_numpy().tolist() == [0, 1, 2]

        np.testing.assert_array_equal(
            image.data[:, :2, :3].compute(scheduler="synchronous"), expected[:2, :3].transpose(2, 0, 1)
        )

        assert decode.call_count == 1
        assert decode.call_args.args[0].closed is True

    def test_selected_series_and_level_determine_spatial_scale(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)
        path = tmp_path / "histology.tif"
        with tifffile.TiffWriter(path) as writer:
            writer.write(np.zeros((12, 16), dtype=np.uint8), metadata={"axes": "YX"})
            writer.write(
                np.ones((4, 12, 16), dtype=np.uint16),
                subifds=1,
                photometric="minisblack",
                metadata={"axes": "CYX"},
            )
            writer.write(
                np.full((4, 6, 8), 17, dtype=np.uint16),
                subfiletype=1,
                photometric="minisblack",
                metadata={"axes": "CYX"},
            )

        result = dbit(tmp_path, image_file=path, tiff_series=1, tiff_level=1, include_border=False)

        image = result.images["sample_image"]
        assert image.shape == (4, 6, 8)
        np.testing.assert_array_equal(image.data.compute(), np.full((4, 6, 8), 17, dtype=np.uint16))
        np.testing.assert_array_equal(result.shapes["sample"].total_bounds, [0, 0, 8, 8])

    def test_channel_count_never_sets_grid_scale(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)
        path = tmp_path / "histology.tif"
        tifffile.imwrite(
            path,
            np.zeros((11, 2, 3), dtype=np.uint8),
            metadata={"axes": "CYX"},
            photometric="minisblack",
        )

        result = dbit(tmp_path, image_file=path, include_border=False)

        np.testing.assert_array_equal(result.shapes["sample"].total_bounds, [0, 0, 3, 3])

    @pytest.mark.parametrize("fail", [False, True])
    def test_compute_closes_tiff_and_store_on_success_and_failure(
        self, tmp_path: Path, mocker: MockerFixture, *, fail: bool
    ) -> None:
        _write_layout(tmp_path)
        path = tmp_path / "histology.tif"
        tifffile.imwrite(path, np.ones((16, 16), dtype=np.uint8), tile=(16, 16))
        image = dbit(tmp_path, image_file=path).images["sample_image"]

        # Install trackers after metadata inspection to observe only task-owned resources.
        closed_handles: list[tifffile.FileHandle] = []
        closed_stores: list[ZarrTiffStore] = []
        original_file_close = tifffile.FileHandle.close
        original_store_close = ZarrTiffStore.close

        def close_file(handle: tifffile.FileHandle) -> None:
            original_file_close(handle)
            closed_handles.append(handle)

        def close_store(store: ZarrTiffStore) -> None:
            original_store_close(store)
            closed_stores.append(store)

        mocker.patch.object(tifffile.FileHandle, "close", new=close_file)
        mocker.patch.object(ZarrTiffStore, "close", new=close_store)

        if fail:
            mocker.patch.object(tiff_tasks.zarr, "open_array", side_effect=RuntimeError("injected source failure"))
            with pytest.raises(RuntimeError, match="injected source failure"):
                image.data.compute(scheduler="synchronous")
        else:
            np.testing.assert_array_equal(
                image.data.compute(scheduler="synchronous"), np.ones((1, 16, 16), dtype=np.uint8)
            )

        assert closed_handles
        assert all(handle.closed for handle in closed_handles)
        assert len(closed_stores) == 1
