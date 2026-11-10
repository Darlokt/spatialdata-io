# Generic images and GeoJSON

`generic()` reads one caller-selected file as one SpatialData image or shapes
element; the case-insensitive suffix selects the kind. `image()` and
`geojson()` read one kind directly. `generic_to_zarr()` creates a store or
appends one element, and the `generic` CLI exposes the same options. There is
no dataset root: no filename convention, layout or software version is
inferred, and `~` in a path is expanded.

| Suffix | Element | Reader |
| --- | --- | --- |
| `.tif`, `.tiff`, `.btf` | lazy image (TIFF, OME-TIFF, BigTIFF) | `image()` |
| `.png`, `.jpg`, `.jpeg` | lazy image | `image()` |
| `.geojson` | eager polygon shapes | `geojson()` |

Options of the other kind must keep their defaults; for example `data_axes`
with a GeoJSON file or `invalid_geometries="drop"` with an image fails.

## Images

Images require explicit `data_axes`: one string, a case-insensitive
permutation of `yx`, `cyx`, `zyx` or `czyx` naming every axis of the decoded
array, such as `"yxc"`. Another type, such as a tuple of letters, raises
`TypeError`; axes that do not match the rank of the decoded array raise
`ValueError` with a note naming the image. Images keep
source values, dtype and channel order in canonical `c, y, x` or `c, z, y, x`
order, with a singleton channel for grayscale. `tiff_series` selects a TIFF
series; when omitted, the only series is read and a file with several series
fails instead of silently returning the first. `tiff_level` selects a stored
pyramid level (default `0`). Neither applies to PNG or JPEG. Pixel decoding
remains lazy. TIFF selection reads source tiles or
strips; PNG and JPEG computations decode the whole encoded frame. Chunking and
generated pyramids follow the [raster options](raster_options.md); generated
levels may not shrink the image below one pixel.

For the selected level-zero OME series, complete positive physical sizes in
µm, nm or mm calibrate all retained spatial axes into micrometres. Anisotropy
is preserved as a transformation; pixels are never resampled. The existing
raster origin is retained, without stage registration or a half-pixel shift.
Absent or incomplete calibration, nonzero stored levels, and caller
reinterpretations of spatial axes retain source pixels. Invalid declared
sizes or unsupported units fail when used for calibration. In particular,
XY spacing without Z spacing leaves an entire 3D image in pixels.

Channel coordinates follow the file's own declarations, never the channel
count or dtype:

- When the `c` axis in `data_axes` is the file's interleaved samples axis and
  the file declares those samples RGB, the channels are `r, g, b`, or
  `r, g, b, a` when a fourth sample is declared alpha. SpatialData viewers
  composite such images as colour. The declarations are TIFF
  `PhotometricInterpretation` RGB (with `ExtraSamples` associated or
  unassociated alpha for `a`) or JPEG-compressed YCbCr, which decodes to RGB;
  PNG colour types 2 (RGB), 3 (palette) and 6 (RGBA), where a `tRNS` chunk
  also adds alpha; and three-component JPEG.
- Otherwise channels come from the OME channel records when the `c` axis is
  the file's declared OME channel axis: OME names when all are nonempty and
  unique, otherwise OME IDs. An interleaved OME image declares one channel with
  several samples; when the TIFF does not declare them RGB (for example
  `MinIsBlack` samples), reading its samples axis as `c` names them
  `<name or ID>:sample:0`, `:sample:1`, and so on. A grayscale OME image
  without a channel axis uses its single channel record for the added channel.
- All other cases use positional indexes `0, 1, …`: non-OME TIFFs without an
  RGB declaration, grayscale and grayscale-with-alpha PNG, one-component JPEG,
  and `data_axes` that read a different axis (for example Z) as channels or
  the channel axis as something else.

When the axes do correspond, OME channel records that disagree with the raster
channel count, or duplicate OME IDs, are corrupt metadata and fail rather than
falling back, even when a colour declaration supplies the names.

RGB TIFF and OME-TIFF, RGB, RGBA and palette PNG, and three-component JPEG
images previously had positional or `<name or ID>:sample:<k>` channel
coordinates; they now use `r, g, b(, a)`. `data_axes` previously also accepted
a sequence of axis names such as `("c", "y", "x")`; join the letters into one
string such as `"cyx"`, the form every reader and the CLI accept.

## GeoJSON

GeoJSON supports `FeatureCollection` and `GeometryCollection` documents with
2D Polygons or MultiPolygons. The file must be strict UTF-8 JSON: duplicate
object names and `NaN`/`Infinity` constants fail instead of being resolved
silently. Output rows retain source order and positional integer IDs; source
Feature IDs, properties and CRS declarations are discarded. Coordinates become
float64; holes and multipart order follow the source. Units remain those of the source,
potentially unknown; geographic coordinates are never reprojected or described
as micrometres. Shapes are necessarily eager; the document is decoded once and
its geometries are built in one vectorized step.

`invalid_geometries="keep"` preserves invalid topology. `"drop"` removes entire
invalid feature rows without renumbering survivors; `"raise"` reports invalid
positional IDs. Missing, empty, nonpolygon, nonfinite, nonplanar or malformed
geometries fail under every mode. Open rings are rejected, never closed or
repaired. Dropping every feature also fails. Coordinates are stored as float64,
so integers beyond `2**53` may round.

## Coordinate systems and stores

`generic()` defaults to the frame `global` when `coordinate_system` is omitted
or `None`; `image()` and `geojson()` require an explicit name. Explicit names
must be strings containing at least one non-whitespace character; accepted
names are preserved literally. A name does not prove registration: calibrated
images use micrometres while GeoJSON may use pixels or other units. Appending
with `generic_to_zarr()` therefore requires an explicit `coordinate_system`,
including when called through the CLI. Choose a distinct frame for unrelated
coordinates; choosing a shared name asserts that the coordinates already map
to that frame. It does not convert or register GeoJSON coordinates. Duplicate
element names fail before the source is read.

```python
from spatialdata_io import generic_to_zarr

generic_to_zarr("image.ome.tif", "sample.zarr", data_axes="cyx")
generic_to_zarr(
    "regions.geojson",
    "sample.zarr",
    coordinate_system="source_pixels",
    invalid_geometries="keep",
)
```

Library calls print no status messages and do not mutate caller-owned options.
Lazy raster computation continues to require the source image. Files close
after parsing or inside individual raster tasks; no temporary source is retained.
