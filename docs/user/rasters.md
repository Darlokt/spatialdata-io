# Choose image chunks and pyramid levels

Use `chunks` to choose how much data a viewer or spatial query reads at a time.
Use `scale_factors` to add lower-resolution levels for viewing a whole slide.
Both options apply when converting the source. Write the result to Zarr to reuse
that layout across sessions.

For example, given a Xenium output directory called `run`, create two additional
pyramid levels and store image channels separately in 512 × 512 pixel tiles:

```bash
python -m spatialdata_io xenium --input run --output run.zarr \
  --scale-factors 2 --scale-factors 2 \
  --chunks 1 --chunks 512 --chunks 512
```

In Python the same options are `scale_factors=(2, 2)` and `chunks=(1, 512, 512)`.
Each factor of two halves the size of the preceding level along the spatial axes.

## Choose chunk sizes

| Raster | Default chunk shape | Axis order |
| --- | --- | --- |
| 2D image | `(1, 1024, 1024)` | channel, y, x |
| 2D labels | `(1024, 1024)` | y, x |
| Generic 3D image | `(1, 1, 1024, 1024)` | channel, z, y, x |

One positive integer sets the same size on all axes. For dataset readers,
a tuple gives the sizes in `(c, y, x)` order. Labels take the spatial sizes
from that tuple. The [generic reader](../readers/generic.md) also accepts sizes
for each canonical axis of a volume, or a mapping keyed by axis name.
The same chunk sizes apply at every generated level, up to the level's extent.

Smaller spatial chunks can help if you repeatedly read small windows. A single
channel per chunk lets you read a channel independently. The original file's
encoding still determines how much data each read decodes. A TIFF can supply
tiles or full-width strips. PNG and JPEG require decoding the whole frame.

## Choose resolutions

The default `scale_factors=()` returns a single scale. Each generated level takes
an integer factor of at least two. Images use averaging and labels use nearest
neighbour, preserving label IDs. Use a single scale for 64-bit labels.

For classic Visium and Visium HD, the factors apply to the full-resolution
tissue image. Space Ranger hires and lowres images are separate elements.
For generic TIFF and DBiT inputs, `tiff_level` selects a resolution already
stored in the TIFF. Any generated levels start from that selection.

Reading a dataset and constructing its pyramids leave the pixels lazy. Keep the
source files in place until you compute the data or finish writing the store.
