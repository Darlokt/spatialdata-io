# Generic images and GeoJSON

`generic()` returns one SpatialData image or shapes element. `image()` and
`geojson()` expose the same format-specific operations. `generic_to_zarr()`
creates a store or appends one element; the `generic` CLI exposes the same
options. No filename convention or software version is inferred.

TIFF, OME-TIFF, BigTIFF/BTF, PNG and JPEG images require explicit `data_axes`:
a permutation of `yx`, `cyx`, `zyx` or `czyx`. Images retain source values,
dtype and channel order, with canonical CYX or CZYX output and a singleton
channel for grayscale. TIFF series and stored levels are selected explicitly.
Pixel decoding remains lazy. TIFF selection reads source tiles or strips;
PNG and JPEG computations decode the whole encoded frame. Chunking and generated
pyramids follow the [raster options](raster_options.md).

For the selected level-zero OME series, complete positive physical sizes in
µm, nm or mm calibrate all retained spatial axes into micrometres. Anisotropy
is preserved as a transformation; pixels are never resampled. The existing
raster origin is retained, without stage registration or a half-pixel shift.
Absent or incomplete calibration, nonzero stored levels, and caller
reinterpretations of spatial axes retain source pixels. Invalid declared
sizes or unsupported units fail when used for calibration. In particular,
XY spacing without Z spacing leaves an entire 3D image in pixels.

GeoJSON supports `FeatureCollection` and `GeometryCollection` documents with
2D Polygons or MultiPolygons. Output rows retain source order and positional
integer IDs; source Feature IDs, properties and CRS declarations are discarded.
Coordinates, holes and multipart order are preserved. Units remain those of
the source, potentially unknown; geographic coordinates are never reprojected
or described as micrometres. Shapes are necessarily eager; the document is
decoded once and its geometries are built in one vectorized step.

`invalid_geometries="keep"` preserves invalid topology. `"drop"` removes entire
invalid feature rows without renumbering survivors; `"raise"` reports invalid
positional IDs. Missing, empty, nonpolygon, nonfinite, nonplanar or malformed
geometries fail under every mode, as do integer coordinates that float64
cannot represent exactly. Open rings are rejected, never closed or repaired;
coordinates are never rounded. Dropping every feature also fails. Nondefault polygon policies are
invalid for image inputs.

The default frame name for a new element is `global`. A name does not prove
registration: calibrated images use micrometres while GeoJSON may use pixels
or other units. Appending with `generic_to_zarr()` therefore requires an
explicit `coordinate_system`, including when called through the CLI. Choose
a distinct frame for unrelated coordinates; choosing a shared name asserts
that the coordinates already map to that frame. It does not convert or register
GeoJSON coordinates. Duplicate element names fail before the source is read.

```python
from spatialdata_io import generic_to_zarr

generic_to_zarr("image.ome.tif", "sample.zarr", data_axes="cyx")
generic_to_zarr(
    "regions.geojson", "sample.zarr",
    coordinate_system="source_pixels", invalid_geometries="keep",
)
```

Library calls print no status messages and do not mutate caller-owned options.
Lazy raster computation continues to require the source image. Files close
after parsing or inside individual raster tasks; no temporary source is retained.
