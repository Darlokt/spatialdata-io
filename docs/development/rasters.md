# Implement raster and geometry outputs

Use `image.inspect_tiff()` and `image.read_tiff()` for TIFF-family metadata
and pixels. The shared loader builds lazy tasks from paths and immutable
selections. Each task opens and closes its own store. Native tile or strip
alignment lets Dask cull unneeded reads. Apply requested rechunking after
forming the native graph. `image.read_png()` and `image.read_jpeg()` defer
whole-frame decoding. Raster construction runs zero pixel tasks.

The reader selects axes, channel identity, series, level, model options and
transformations from format evidence or explicit user options. Use
`image.normalize_raster_axes()` or `image.normalize_cyx_image()` where the
layout fits their validated selection and axis contracts. Use named axes
for singleton removal. Keep open handles out of results and lazy graphs.
Preserve the native coordinate meaning and provide `global` in micrometres
when verified calibration establishes it. Document the pixel frame otherwise,
along with origin, orientation and any registration assumptions.
[Coordinate systems](../user/coordinates.md)

Polygonal readers share the `invalid_geometries` policy through `geometry`.
Validate structural requirements, such as present, nonempty geometry,
before checking validity. `"keep"` preserves the source topology by default,
`"raise"` reports invalid status and identifiers, and `"drop"` removes
entities as documented by each reader. Apply drops consistently to all
affected products before table linkage.


## Verify a bounded image read

Use a deterministic tiled TIFF with distinct values in at least two channels
and several spatial tiles. Assert axes, dtype, channel order and the transform
before computing a selection. Count executed pixel tasks during construction,
then compare one off-origin crop with independently known pixels. Test the
full array on a small fixture to catch swapped axes that a single crop can
miss.

For polygons, include a valid object and an invalid object with distinct
source IDs. Exercise the three policy settings and check the surviving IDs in
both shapes and table. If a geometry uses native pixel coordinates, check a
known vertex in its registered frame. [Testing readers](testing.md)
