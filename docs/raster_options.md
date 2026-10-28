# Raster options

Every reader that returns images or labels exposes the same two options with
one meaning: `scale_factors` for generated pyramid levels and `chunks` for the
chunk sizes of the returned rasters. Both are visible in the reader signature
and in `--help`; no reader chooses a pyramid or chunk size from the size of the
image.

## Defaults

| Option | Default | Meaning |
| --- | --- | --- |
| `scale_factors` | `()` | A single full-resolution scale; no generated levels. |
| `chunks` | `(1, 1024, 1024)` | Images: one channel per chunk and square 1024-pixel `y, x` tiles. Labels: `(1024, 1024)`. |

Generic 3D images (`c, z, y, x`) default to `(1, 1, 1024, 1024)`: one channel and
one plane per chunk.

The same tile sizes apply to every pyramid level. The values are defined once,
in `spatialdata_io._raster_defaults`, and used by every reader and command.

## `scale_factors`

A sequence of integer downsampling factors of at least two, one per generated
level, applied equally to the images and labels of a dataset. For example
`(2, 2, 2)` adds three levels, each half the size of the previous one. Images
are downsampled by interpolation and labels by nearest neighbour, so generated
label levels keep the original label values. Generated label pyramids require
labels of at most 32 bits; 64-bit labels are only read as a single scale.
`visium` applies the factors to the full-resolution tissue image only: its
`hires` and `lowres` images are already downsampled by Space Ranger and stay
single scale.

Levels are generated from the selected source level, which is the full
resolution unless a reader lets you select a stored TIFF level (`tiff_level`
in `generic` and `dbit`); other vendor-written pyramid levels are not read.

## `chunks`

One positive size for every axis, or three positive sizes in `c, y, x` order.
Labels use the `y` and `x` sizes. Requests may be non-square or group several
channels; any regular grid is valid Zarr. Readers whose images can be 3D or use
other axes (`generic`, `visium`) also accept one size per canonical axis or a
mapping from axis name to size; see their documentation.

Chunks are always passed explicitly to the SpatialData models. Without them
the multiscale image builder rechunks to its own 256-pixel tiles and merges all
channels into one chunk. Native storage chunks are not kept as the default
either: they are often irregular or unsuited to the written store (TIFF strips
are full-width bands, PNG and JPEG frames are one chunk), and the SpatialData
writer regularizes irregular chunks to the largest chunk per axis.

## Why square tiles and one channel per chunk

Viewers and spatial queries read rectangular windows, and square tiles touch
the fewest chunks for a random window. Pyramid levels halve `y` and `x`
equally, so square tiles keep their shape at every level, and 1024 is a
multiple of common TIFF tile sizes (256, 512, 1024), so rechunking merges whole
storage tiles. One channel per chunk lets a viewer that shows one channel read
only that channel, and keeps the chunk size independent of the channel count.

## Laziness

Rechunking and pyramid construction are lazy: constructing the reader result
executes no pixel tasks. Source decoding still follows the storage layout:
computing a small region decodes every TIFF tile or strip band of the output
chunks it intersects, and PNG and JPEG sources decode their whole frame. For
example, a 1024-row output chunk of a TIFF stored in full-width strips spans
several strip bands. Request smaller chunks for workflows dominated by small random
reads.

## Command line

The options use the same names in kebab case:

```bash
spatialdata_io xenium --input run --output run.zarr \
  --scale-factors 2 --scale-factors 2 \
  --chunks 1 --chunks 512 --chunks 512
```

Repeat `--scale-factors` once per generated level. Pass `--chunks` once for
every axis or three times in `c, y, x` order.
