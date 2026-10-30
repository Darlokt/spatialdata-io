# Geometry construction and invalid-geometry policy

This private package owns one cross-reader output policy: what happens to
source polygons that are not valid geometries. It also owns exact construction
of decoded GeoJSON Polygon and MultiPolygon geometry objects.

```python
from spatialdata_io.readers._utils.geometry import (
    InvalidGeometryPolicy,
    geojson_polygons,
    invalid_geometry_mask,
    raise_invalid_geometries,
    validate_invalid_geometry_policy,
)
```

## Contract

- Every reader returning polygonal shapes exposes
  `invalid_geometries: Literal["keep", "drop", "raise"] = "keep"`.
  `validate_invalid_geometry_policy()` checks the public value once and raises
  `TypeError` or `ValueError`.
- A geometry is invalid when it is not `shapely.is_valid`, for example a
  self-intersecting ring or a bow-tie. `invalid_geometry_mask()` computes this
  for a one-dimensional geometry array. Missing and empty geometries are
  malformed input, not invalid geometry; readers reject them before
  classification and the mask raises `ValueError` if they reach it.
- `raise_invalid_geometries()` raises one `ReaderFormatError` naming the
  invalid count and at most ten reader-owned identifiers.

`geojson_polygons()` turns a sequence of decoded GeoJSON geometry objects
into one Shapely geometry per entry, in source order. Generic GeoJSON and
seqFISH boundaries share this contract:

- each entry is a Polygon or MultiPolygon with nonempty coordinates;
- each ring is a closed list of at least four 2D vertices; open rings are
  rejected rather than closed by Shapely;
- coordinates are finite JSON numbers (not booleans or strings) and integers
  must be exactly representable in float64;
- vertices, holes and part order are preserved, and topology is neither
  classified nor repaired.

Violations raise `ReaderFormatError` with the geometry's position. Validation
is a Python pass over the decoded coordinates; construction is one vectorized
ragged-array call. Document forms, feature identity, properties and CRS stay
reader-owned.

`"keep"` preserves source geometry exactly and is the default, because invalid
polygons occur in otherwise valid vendor data (for example simplified Xenium
boundaries). Repair is never implicit.

## Ownership boundaries

Readers own what `"drop"` removes (a single polygon, or an entity together with
its other shapes and table rows), how input that cannot be constructed as a
polygon is classified, identifier construction, vendor naming and
transformations. Structurally malformed input (nonfinite coordinates, broken
vertex grouping, unknown identifiers) raises in every policy. The package
performs no I/O and no repair.
