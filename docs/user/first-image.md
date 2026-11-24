# Convert your first image

In this tutorial you will create a small two-channel TIFF, load it as a SpatialData
image, and save it in a Zarr store. Use an empty working directory so the
example has a fresh destination.

## Install spatialdata-io

Use a Python 3.12 or newer environment. To install the published release, run:

```bash
python -m pip install spatialdata-io
```

If you are following the documentation for a development checkout, install that
checkout with `python -m pip install -e .` from the repository root.

## Create and read the image

Run the Python blocks in order in the same session. Make a TIFF with two channels,
three rows and four columns. The second row of the first channel contains the
values 4 through 7.

```python
import numpy as np
import tifffile

pixels = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
tifffile.imwrite("slide.tif", pixels, photometric="minisblack")
```

Read the file with the generic reader. `data_axes="cyx"` describes the order of
the axes **in the file**. We call the coordinate system `slide_pixels` because
this image has no physical calibration.

```python
from spatialdata_io import generic

image = generic("slide.tif", data_axes="cyx", coordinate_system="slide_pixels")
print(image.dims, image.shape)
```

You should see `('c', 'y', 'x') (2, 3, 4)`. Each channel has the
original three rows and four columns. You can inspect the shape and dtype
before loading any pixels.

## Write and reopen the store

Give the image a name inside a SpatialData container and write it. Writing reads
the pixels from the TIFF, so keep the source file available until this finishes.

```python
from spatialdata import SpatialData, read_zarr

sdata = SpatialData(images={"slide": image})
sdata.write("slide.zarr")
reopened = read_zarr("slide.zarr")
print(reopened.images["slide"].isel(c=0, y=1).values.tolist())
```

The result is `[4, 5, 6, 7]`, the second row of the original image. Future
sessions can open `slide.zarr` directly.

For an instrument export, use the [reader for that technology](../readers/index.md).
Dataset readers return a SpatialData object containing the selected products,
so you can call its `write()` method directly. To convert files from a shell,
see the [command-line reference](../cli.md).
