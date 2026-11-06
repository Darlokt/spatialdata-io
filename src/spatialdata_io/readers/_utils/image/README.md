# Raster I/O boundary

This package is the nominal pixel-decoding boundary for `spatialdata-io` readers.
It turns one validated raster file into a lazy Dask array.

## Responsibilities

The package owns:

- metadata inspection without pixel decoding before graph construction;
- TIFF series, pyramid-level, axes, shape, dtype, and native-chunk discovery;
- complete typed OME metadata parsing;
- path-based, serializable Dask graphs;
- task-local file and store ownership;
- delayed PNG and JPEG decoding;
- explicit, lazy axis selection and normalization;
- the colour model each source declares for its samples, and the one rule
  that labels raster channels from it or from OME metadata.

A vendor reader owns:

- layout and version discovery;
- choosing a file, TIFF series, and pyramid level;
- interpreting or overriding source axes;
- selecting channels, time points, and z planes;
- stacking files, and deciding which source axis becomes the channel axis and
  whether OME channel records describe it;
- image-versus-label semantics;
- transformations, coordinate systems, and SpatialData model construction;
- requested/model rechunking after the native read layer.

## Shared raster options

Every reader that returns images or labels exposes `scale_factors` and `chunks`
with one meaning, validated by `make_raster_options()`:

- `scale_factors`: integer downsampling factors of at least two, one per
  generated pyramid level, applied equally to images and labels;
  `DEFAULT_SCALE_FACTORS = ()` returns a single scale.
- `chunks`: one positive size for every axis or three `c, y, x` sizes; labels
  use the `y, x` sizes. `RasterOptions.chunks` always holds the three sizes.
  `DEFAULT_CHUNKS = (1, 1024, 1024)` gives one channel per chunk and square
  1024-pixel tiles.

Readers import the constants directly from `spatialdata_io._raster_defaults`
as signature defaults, so a default changes in one place.
`RasterOptions.image_chunks()` and `label_chunks()` return explicit
per-axis sizes for the SpatialData model: without them the multiscale image
builder rechunks to 256-pixel tiles and merges all channels. Square is only
the default; any regular chunk grid is valid Zarr.
`RasterOptions.validate_image_pyramid()` rejects factors that would shrink an
element below one pixel, and `validate_label_pyramid()` additionally rejects
generated levels of labels wider than 32 bits, whose IDs SpatialData's
downsampling can round. Readers name the element in the message.

## Public workflow

TIFF inspection and reading are separate because readers may need metadata to
choose a series or level:

```python
from spatialdata.models import Image2DModel

from spatialdata_io.readers._utils.image import (
    inspect_tiff,
    normalize_raster_axes,
    ome_channel_names,
    raster_channel_names,
    read_tiff,
)

metadata = inspect_tiff(path, context=context)
raster = read_tiff(metadata, series=0, level=0, context=context)
axes = tuple(axis.lower() for axis in raster.axes)
data = normalize_raster_axes(
    raster.data,
    tuple("c" if axis == "s" else axis for axis in axes),
    target_axes=("c", "y", "x"),
    add_missing=("c",),
)
samples = "s" in axes
ome = ome_channel_names(metadata, series=0, count=int(data.shape[0]), samples=samples, context=context)
names = raster_channel_names(raster.color, samples=samples, ome=ome)
image = Image2DModel.parse(data, dims=("c", "y", "x"), c_coords=names, rgb=False)
```

`series` and `level` are intentionally required. The utility must not silently
choose the first image in a scientific file that contains several images. An
out-of-range series or level raises `IndexError` with a note naming the reader
context and the TIFF path.

PNG and JPEG have no selectable series or levels, so their entry points combine
header inspection with lazy graph construction. Ancillary PNG and JPEG payloads
are skipped without materialization; no total header-size ceiling is imposed:

```python
from spatialdata_io.readers._utils.image import read_jpeg, read_png

png = read_png(png_path, context=context)
jpeg = read_jpeg(jpeg_path, context=context)
```

All three readers return `LazyRaster(data, axes, color)`. Shape and dtype are
available from `data.shape` and `data.dtype`; the result does not duplicate
them. Interleaved samples are the axis `s` for every format: TIFF `S`, and the
gray+alpha, RGB or RGBA components of PNG and JPEG. `color` is the colour model
the source declares for that axis (see below).

Entry points take an `ArtifactFile` already validated by the shared path
package and do not resolve or kind-check it again; the canonical path is
stored in delayed tasks. Every entry point takes the reader's
`ReaderErrorContext`: inspection failures keep their type and gain a note
naming the reader, artifact and path.

Malformed encoded raster storage raises `RasterFormatError` from
`spatialdata_io.readers._utils.errors`. The error type is centralized with other
reader utility errors, while its raster-specific meaning and all decisions about
when to raise it remain owned by this package. Axis-option and array-contract
failures retain their precise `TypeError` or `ValueError` contracts.

## TIFF and OME metadata

`TiffMetadata.series` describes the arrays that `tifffile` can read. Every
level records its actual `shape`, `dtype`, declared `axes`, `native_chunks` and
samples `color`. These are storage facts and may differ between pyramid levels.

For OME-TIFF, `TiffMetadata.ome` contains the complete parsed `ome_types` model.
It provides logical XYZCT dimensions, physical sizes and units (including Z),
channels, planes, timing, acquisition metadata, and structured annotations:

```python
metadata = inspect_tiff(path, context=context)
if metadata.ome is not None:
    pixels = metadata.ome.images[0].pixels
    logical_size = (
        pixels.size_x,
        pixels.size_y,
        pixels.size_z,
        pixels.size_c,
        pixels.size_t,
    )
    physical_size = (
        pixels.physical_size_x,
        pixels.physical_size_y,
        pixels.physical_size_z,
    )
```

Readers must treat the returned OME model as read-only. It is never embedded in
pixel tasks. The package does not copy selected OME fields into parallel custom
types, and readers must not reopen the TIFF to parse metadata already available
through this model.

Linked OME `UUID FileName` references are resolved relative to the inspected
TIFF and must remain within that file's directory. Absolute references, parent
traversal, and symlinks resolving outside that directory are rejected before
linked pixels are opened. A `TiffData` UUID equal to the document's own UUID
refers to the inspected file, whatever its recorded `FileName`, so renamed
single-file OME-TIFFs remain readable.

Baseline TIFF resolution tags are not automatically interpreted as microscopy
calibration. A supported non-OME format may use them only when its documented
contract establishes their scientific meaning.

## Laziness and resource ownership

TIFF tiles are native chunks. Strips, and tiles as wide as the image, store
whole rows; consecutive ones are grouped into row bands of at most
`STRIP_BAND_BYTES` (16 MiB) decoded bytes, aligned to their boundaries, and a
larger strip is one chunk. Every task opens the TIFF, so one task per small
strip (vendor exports often store one row per strip) would dominate read time.

Dask stores one private path-backed source in a compact blockwise layer. Each
computed selection opens the TIFF, reads only the intersecting tiles or row
bands, and closes every resource before returning. Most layouts are read
through a short-lived tifffile Zarr store. A single page stored in the
inspected file that tifffile marks final (uncompressed, contiguous, no
predictor, default fill order, no subsampling) and whose segments hold exactly
the page is read directly, one bounded row band per task, because Zarr's
per-strip overhead exceeds the read itself. No `TiffFile`, Zarr store, file
handle, or parsed OME model is stored in the graph.

PNG and JPEG do not provide equivalent random tile access here. Their headers
are inspected eagerly with a fixed byte bound, but pixels are decoded in one
delayed whole-frame task. Computing even a small selection therefore decodes the
whole frame.

Axis normalization performs only lazy singleton removal, singleton insertion,
and transposition. It never guesses an axis from dimension size. A
non-singleton axis omitted from the target is an error, so no plane is ever
selected implicitly, and a missing target axis requires explicit permission to
insert it. Axis labels are abstract
single-letter names at this boundary: the calling reader must interpret unknown
TIFF axes or replace them with an explicit, validated format-specific override.

## Adding another format

Add a format only for a demonstrated reader requirement. Do not add a generic
reader class, inheritance hierarchy, registry, backend selector, or loose
`read_raster()` dispatcher.

Use this process:

1. Define the format's factual storage contract and supported encodings.
2. Inspect enough bounded metadata to know exact output shape, dtype, and axes
   without decoding pixels.
3. Keep format-specific metadata private unless vendor readers need it to make
   an explicit scientific decision.
4. Return `LazyRaster`; do not construct SpatialData models in this package.
5. Put only paths and small immutable values in Dask tasks.
6. Use bounded selection tasks when the storage format supports them. Otherwise,
   document and test a delayed whole-frame task.
7. Validate external bytes and metadata once at inspection. Trust precisely
   typed private state afterward, while verifying decoded shape and dtype at the
   task boundary.
8. Add independent unit tests for metadata parsing and integration tests for
   real decoding, laziness, graph serialization, partial-read behavior where
   applicable, and resource cleanup on success and failure.

Share code across formats only when the resource and decoding contracts are
actually identical. PNG and JPEG, for example, share one whole-frame task but
keep independent header parsers. TIFF remains separate because its series,
pyramids, OME metadata, and native selection behavior are fundamentally richer.

## Prohibited patterns

Reader modules must not:

- call `tifffile.imread`, `tifffile.memmap`, `imageio.imread`, or another pixel
  decoder directly;
- place open resources in a Dask graph;
- call `compute()`, `load()`, or `np.asarray()` on raster data without a
  documented, tested size bound;
- infer semantic axes from the smallest dimension or use unqualified
  `squeeze()`;
- pass arbitrary decoder keyword mappings through public reader APIs;
- introduce alternate TIFF backends or compatibility paths.

## OME physical pixel sizes and channel identities

`ome_pixel_sizes_um(metadata, series=..., axes=..., context=...)` returns one
entry per requested axis: the declared OME length converted from µm, nm or mm
to finite positive micrometres with one correctly rounded operation (9 nm is
exactly `0.009`), or `None` when that size is absent. Every entry
is `None` without OME metadata. A malformed present value or unsupported unit
fails even when other sizes are absent, so partial declarations can still be
compared between linked artifacts.

Generic base images, ISS aligned pairs, MACSima channels, MCMICRO local grids,
seqFISH and Steinbock exports share this normalization. Readers decide whether an incomplete set
calibrates anything and continue to own axis association, selected
series/levels, dimensions, cross-artifact agreement, registration and
transformations. The helper assigns no frame and reads no pixels.

`ome_channel_names(metadata, series=..., count=..., samples=..., context=...)`
returns one OME identifier per raster channel. Each OME channel is identified by
its name when all names are nonempty and unique, otherwise by its OME ID, which
must then be unique. OME declares an interleaved RGB image as one `Channel`
with `SamplesPerPixel="3"`, which `tifffile` exposes as a samples axis `S`.
The reader states with `samples` whether the raster's channel axis is that
samples axis. If it is, every OME channel expands to one identifier per sample,
`<identifier>:sample:<index>` (an unnamed RGB channel gives
`Channel:0:0:sample:0` to `Channel:0:0:sample:2`); these keep the source
record and sample order without assuming a colour interpretation, and are
unique whenever the channel identifiers are. A channel or sample count that
differs from the raster raises instead of falling back to positions. It returns
`None` without OME metadata. CODEX, DBiT, ISS, seqFISH and Steinbock use it;
readers decide which source axis becomes the raster channel axis.

## Channel labels from declared colour

SpatialData labels colour channels `r, g, b(, a)`, and viewers composite images
with those labels. A raster gets them only from its file's own declaration of
the samples, exposed as `LazyRaster.color` and `TiffLevelMetadata.color`
(`"rgb"`, `"rgba"` or `None`):

| Source | `"rgb"` | `"rgba"` |
| --- | --- | --- |
| TIFF | `PhotometricInterpretation` RGB without `ExtraSamples`, or JPEG-compressed YCbCr with interleaved samples (decoded to RGB) | RGB with one `ExtraSamples` value of associated (1) or unassociated (2) alpha |
| PNG | colour type 2 (RGB) or 3 (palette) | colour type 6, or type 2 or 3 with `tRNS` |
| JPEG | three components (decoded to RGB) | never |

Everything else declares no colour: `MinIsBlack`/`MinIsWhite` samples, RGB with
an unspecified (0) or a second extra sample, uncompressed YCbCr, palette TIFF,
grayscale and grayscale+alpha PNG, and one-component JPEG.

`raster_channel_names(color, samples=..., ome=...)` applies the single
labelling rule; readers must not reimplement it, infer RGB from a channel
count or dtype, or pass `rgb=None` to SpatialData:

1. `samples` (the raster's channel axis is the source's `s` axis) and a declared
   `color`: `["r", "g", "b"]` or `["r", "g", "b", "a"]`;
2. otherwise `ome`, the result of `ome_channel_names` when the reader has
   established that the channel axis corresponds to the OME channels;
3. otherwise `None`, which gives positional coordinates `0..n-1`.

OME identifiers are computed and validated before the rule is applied, so
corrupt OME channel records fail even when a colour declaration supplies the
names. CODEX, DBiT, ISS and the generic image reader use the rule; pass
`rgb=False` with the returned names to `Image2DModel.parse`.
