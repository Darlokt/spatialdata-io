from __future__ import annotations

from typing import TYPE_CHECKING

from dask.callbacks import Callback
from spatialdata.models import Image2DModel, Labels2DModel

from spatialdata_io.readers._utils.image import inspect_tiff, normalize_raster_axes, read_tiff
from tests.integration.readers._utils.image.conftest import TiffOptions

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from tests.integration.readers._utils.image.conftest import TiffFactory


class TestModelComposition:
    """Compose the model-independent boundary with representative models."""

    def test_tiff_arrays_remain_lazy_through_image_and_labels_models(
        self,
        cyx_data: NDArray[np.uint16],
        write_tiff: TiffFactory,
    ) -> None:
        image_path = write_tiff(
            cyx_data,
            TiffOptions(compression="deflate", tile=(16, 16)),
        )
        image_result = read_tiff(inspect_tiff(image_path), series=0, level=0)
        image = normalize_raster_axes(
            image_result.data,
            image_result.axes,
            target_axes=("c", "y", "x"),
        )
        labels_path = write_tiff(
            cyx_data[0],
            TiffOptions(name="labels.tif", rowsperstrip=5, axes="YX"),
        )
        labels_result = read_tiff(inspect_tiff(labels_path), series=0, level=0)
        labels = normalize_raster_axes(
            labels_result.data,
            labels_result.axes,
            target_axes=("y", "x"),
        )
        executed: list[object] = []

        with Callback(pretask=lambda key, *_: executed.append(key)):
            parsed_image = Image2DModel.parse(image, dims=("c", "y", "x"), rgb=None)
            parsed_labels = Labels2DModel.parse(labels, dims=("y", "x"))

        assert not executed
        assert parsed_image.shape == cyx_data.shape
        assert parsed_labels.shape == cyx_data.shape[1:]
