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

## Choose TIFF pages and physical coordinates

A TIFF series groups arrays by storage properties. Some instruments also assign
individual TIFF image file directories (IFDs) roles such as acquisition band,
thumbnail, slide label or overview. Request page metadata from `inspect_tiff()`
only for formats that need it. The vendor reader interprets the page descriptions,
selects the physical IFDs in the specified order, and passes those selections
to `read_tiff(metadata, page=index, context=context)`. Equal shape and dtype
alone cannot establish page identity.
[PhenoCycler example](../readers/phenocycler.md)

Carry acquisition channel identity, a display name, stored dtype and signal
calibration separately when the source distinguishes them. A repeated target
can represent several measurements. An intensity scale describes stored pixel
values. Applying it is a numerical conversion that needs an explicit output
contract. Retain the format fields used to interpret measurements in
writeable metadata and exercise a Zarr round trip. Exporting complete XML or
sidecar objects can also export operator names, paths and hostnames.

Before pairing alternate encodings of one acquisition, compare the ordered
measurements as well as the spatial grid. For example, two channels can have
the same marker and optical band but different reagent barcodes. Compare
optional identifiers wherever both sources declare them. Test both a conflict
and an older export that omits the identifier.

Pixel size converts distances within one grid. Registration places two grids
in the same frame. For a multi-region scan, verify slide identity, units,
origins and the direction of the coordinate axes before composing scale and
translation. For a regional cell table, verify the mapping from the image
grid to the grid used to report cell positions. Test an off-origin landmark
or pixel centre. A filename match, equal dimensions or an integer size ratio
alone leaves the origin and crop undetermined.
[CODEX calibration](../readers/codex.md#calibration-and-registration)

## Polygonal outputs

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
miss. Include a large uncompressed contiguous plane written as a single strip.
A Dask graph may execute just one task while reading the entire plane for one
pixel. Measure the source selection or bytes read, before and after output
rechunking. Also check opposite file byte order and the last row or plane when
a reader relies on direct byte reads.

For polygons, include a valid object and an invalid object with distinct
source IDs. Exercise the three policy settings and check the surviving IDs in
both shapes and table. If a geometry uses native pixel coordinates, check a
known vertex in its registered frame. [Testing readers](testing.md)
