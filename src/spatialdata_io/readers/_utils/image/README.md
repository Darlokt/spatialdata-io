# Raster I/O boundary

This package is the nominal pixel-decoding boundary for `spatialdata-io` readers.
It turns one validated raster file into a lazy Dask array.

## Responsibilities

The package owns:

- bounded metadata inspection before graph construction;
- TIFF series, pyramid-level, axes, shape, dtype, and native-chunk discovery;
- complete typed OME metadata parsing;
- path-based, serializable Dask graphs;
- task-local file and store ownership;
- delayed PNG and JPEG decoding;
- explicit, lazy axis selection and normalization.

A vendor reader owns:

- layout and version discovery;
- choosing a file, TIFF series, and pyramid level;
- interpreting or overriding source axes;
- selecting channels, time points, and z planes;
- stacking files and assigning channel names;
- image-versus-label semantics;
- transformations, coordinate systems, and SpatialData model construction;
- requested/model rechunking after the native read layer.

## Public workflow

TIFF inspection and reading are separate because readers may need metadata to
choose a series or level:

```python
from spatialdata.models import Image2DModel

from spatialdata_io.readers._utils.image import (
    inspect_tiff,
    normalize_raster_axes,
    read_tiff,
)

metadata = inspect_tiff(path)
raster = read_tiff(metadata, series=0, level=0)
data = normalize_raster_axes(
    raster.data,
    raster.axes,
    target_axes=("c", "y", "x"),
    add_missing={"c"},
)
image = Image2DModel.parse(data, dims=("c", "y", "x"), rgb=None)
```

`series` and `level` are intentionally required. The utility must not silently
choose the first image in a scientific file that contains several images.

PNG and JPEG have no selectable series or levels, so their entry points combine
bounded header inspection with lazy graph construction:

```python
from spatialdata_io.readers._utils.image import read_jpeg, read_png

png = read_png(png_path)
jpeg = read_jpeg(jpeg_path)
```

All three readers return `LazyRaster(data, axes)`. Shape and dtype are available
from `data.shape` and `data.dtype`; the result does not duplicate them.

## TIFF and OME metadata

`TiffMetadata.series` describes the arrays that `tifffile` can read. Every
level records its actual `shape`, `dtype`, declared `axes`, and `native_chunks`.
These are storage facts and may differ between pyramid levels.

For OME-TIFF, `TiffMetadata.ome` contains the complete parsed `ome_types` model.
It provides logical XYZCT dimensions, physical sizes and units (including Z),
channels, planes, timing, acquisition metadata, and structured annotations:

```python
metadata = inspect_tiff(path)
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
linked pixels are opened.

Baseline TIFF resolution tags are not automatically interpreted as microscopy
calibration. A supported non-OME format may use them only when its documented
contract establishes their scientific meaning.

## Laziness and resource ownership

TIFF arrays use native tile or strip chunks. Dask stores one private path-backed
source in a compact blockwise layer. Each computed selection opens the TIFF and
its Zarr selection store, reads only intersecting segments, and closes both
before returning. No `TiffFile`, Zarr store, file handle, or parsed OME model is
stored in the graph.

PNG and JPEG do not provide equivalent random tile access here. Their headers
are inspected eagerly with a fixed byte bound, but pixels are decoded in one
delayed whole-frame task. Computing even a small selection therefore decodes the
whole frame.

Axis normalization performs only lazy slicing, singleton insertion, and
transposition. It never guesses an axis from dimension size. A non-singleton
axis omitted from the target requires an explicit selector, and a missing target
axis requires explicit permission to insert it. Axis labels are abstract
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
