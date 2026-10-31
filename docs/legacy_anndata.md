# AnnData access and legacy conversion

Reader tables already are AnnData objects. Use them directly for expression
analysis; conversion is useful when importing older spatial AnnData datasets or
explicitly choosing positions derived from linked elements. Both converters are
available from `spatialdata_io.experimental` and
`spatialdata_io.converters.legacy_anndata`.

## Use a reader table directly

```python
table = sdata.tables["table"]  # Borrow the existing object.
independent = table.copy()  # Own a copy before independent mutation.
```

Mutating `table` also changes the table held by `sdata`. Access alone performs
no conversion, expression scan, geometry join or coordinate allocation. A copy
preserves observation/feature order, annotations, embeddings, layers and `raw`
while isolating subsequent mutation.

Inspect `table.uns["spatialdata_io_positions"]` before physical analysis.
Canonical `obsm["spatial"]`, when present, is XY micrometres; native-only
readers retain their documented pixel, grid or source embeddings instead.
Neither the name `global` nor legacy `uns["spatial"]` image metadata proves
physical calibration, image alignment or a Visium lattice. Choose coordinate
keys and downstream graph modes explicitly.

Several linkage regions may share one physical frame. Independent frames can
have identical numerical origins. If the position declaration uses `frame_key`,
partition by that observation column before an analysis that does not handle
independent frames:

```python
description = table.uns["spatialdata_io_positions"]["coordinates"]["spatial"]
frame_key = description["frame_key"]  # For a table declaring row-specific frames.
frame = table.obs[frame_key].cat.categories[0]
one_frame = table[table.obs[frame_key] == frame].copy()
```

A scalar `frame` declaration already identifies one frame. Slicing preserves
row-aligned embeddings and frame columns, but reserved SpatialData linkage
metadata can retain surplus region names. Ordinary AnnData concatenation does
not reconcile coordinate declarations or register independent samples.

Positions are snapshots. Registering spatial elements later does not update
stored vendor centres. Geometry centroids may have a different definition even
before registration; disagreement alone does not establish that a snapshot is
stale.

## Export with an explicit position source

The public signature is:

```text
to_legacy_anndata(
    sdata: SpatialData,
    *,
    table_name: str | None = None,
    coordinate_system: str | None = None,
    position_source: Literal["table", "centroids"] = "table",
) -> AnnData
```

```python
from spatialdata_io.experimental import to_legacy_anndata

exported = to_legacy_anndata(sdata, table_name="table")
```

The result is an owned AnnData copy. Select a table when there is more than
one. A declared frame is inferred only when unambiguous. For independent
row frames, select `coordinate_system` explicitly or subset the table first.
Explicit frame selection retains matching rows in their original relative order.
Default export preserves copied `spatialdata_attrs` linkage metadata without
validating or embedding its targets. Only centroid mode requires those targets.

| `position_source` | Coordinate behavior |
| --- | --- |
| `"table"` (default) | Preserve stored arrays, declarations, population and order. No element positions are computed. A requested frame must describe the stored snapshot; it does not reproject it. |
| `"centroids"` | Derive positions from linked Shapes, Labels or Points in the selected frame, align by `(region, instance_id)`, and retain table row order. Missing, duplicate or ambiguous target IDs fail. |

For the default mode, a declared physical `spatial` is preserved. An incoming
undeclared `spatial` is preserved exactly as `spatial_source`, with no inferred
units; an occupied destination fails. Existing `spatial_pixels`, `spatial_grid`
and `spatial_source` remain unchanged. There is no fallback from missing table
positions to geometry.

```python
centroids = to_legacy_anndata(
    sdata,
    table_name="table",
    coordinate_system="global",
    position_source="centroids",
)
```

Centroid mode applies the complete selected-frame transformation. Physical
centroids become float64 XY micrometre `spatial` only when that frame's physical
units are established. A replaced physical array and its declaration move to
`spatial_original`. Established native centroids use `spatial_centroids` with
their declared units and frame, preserving an existing physical snapshot.
Occupied preservation destinations fail. A frame name alone cannot establish
units, and unknown frame/unit contracts fail rather than acquire a scale.
Uncalibrated coordinates never become canonical `spatial`.
Canonical `spatial` is absent unless every returned row has established
micrometre calibration and a known frame; no partly physical array is produced.

Default table export executes no image, label or point tasks. Explicit label
centroids can compute the complete label raster, including its associated I/O
and memory cost; lazy Points also require materialization for their positions.
Polygon centroids, label centroids and source vendor centres remain distinct
position definitions.

Image export and the former `include_images` option have been removed. Keep
images and their transformations in SpatialData for registered overlays. The
exporter does not render, resize or create image-library metadata.
Existing legacy `uns["spatial"]` is copied with the table, including any embedded
image buffers; it does not establish alignment with the selected positions.

## Import older spatial AnnData

The public signature is:

```text
from_legacy_anndata(
    adata: AnnData,
    *,
    library_id: str | None = None,
    library_key: str | None = None,
    rgb: bool | None = None,
) -> SpatialData
```

```python
from spatialdata_io.experimental import from_legacy_anndata

imported = from_legacy_anndata(adata)
```

The importer accepts in-memory AnnData with eager annotations and numerical
coordinate arrays. Backed or lazy payloads fail before implicit computation.
It takes one owned AnnData copy, preserving expression values, feature order,
annotations, arbitrary layers and `raw`. Both converters normalize `X` to a
canonical CSR sparse array without densifying expression; `X=None` remains
`None`. Arbitrary layers and `raw` retain their source storage.

Plain legacy `obsm["spatial"]` means full-resolution pixel-index XY. It moves
exactly to `spatial_pixels`, with pixel units declared and no canonical physical
`spatial`. Existing destination keys fail rather than be overwritten. Import
creates a `locations` Points element, not spot/cell circles. New
`spatialdata_region` and `spatialdata_instance_id` observation columns link the
returned `table` to these points. Original observation names, identifiers and
columns, including existing `region`/`instance_id`, remain intact. Generated IDs
use original row ordinals and are local linkage identifiers; collisions with
the generated column names fail.

A valid physical position declaration is preserved for a single-frame import
without legacy images. It is never relabelled as pixels. Physical positions
with embedded legacy images fail because scalar resolution factors do not
establish their physical mapping. Declared pixel `spatial` follows the lossless
rename rule using its declared frame. Declared grid/unknown `spatial`, malformed
declarations and unsupported declaration versions fail. Other native
embeddings are preserved without guessing which should supply positions.

### Libraries and images

One call imports one library. A single library can be inferred. For multiple
libraries, supply both a selected ID and an observation column associating rows
with libraries:

```python
imported = from_legacy_anndata(
    adata,
    library_id="sample_a",
    library_key="library_id",
)
```

Selection retains original row order. Null/unknown memberships and empty
explicit selections fail. A supplied `library_key` is validated and honored
even for one library. Repeat calls for independent libraries; they do not gain
a common physical origin through import. Multiple declared physical frames
require explicit caller subsetting.

Only the selected library's existing in-memory images are imported, at stored
resolution, with their values and numerical dtype preserved. Inputs use YX for
grayscale and YXC for channel images; outputs use CYX. With `rgb=None`, three/four
channels are interpreted as RGB(A); `False` keeps numerical channel identities,
and `True` requires three/four channels. Images are not rendered or resampled.
Consumed image mappings, dimensions, dtypes and scale factors are validated.
For an image named `hires`, `scalefactors["hires"]` takes precedence when present;
otherwise `scalefactors["tissue_hires_scalef"]` is used. Each consumed scale must
be finite, positive and representable with a finite reciprocal. Missing values,
`None`, booleans and other invalid types raise `TypeError`; invalid numerical
values raise `ValueError`. An invalid direct entry never falls back to the other
key.
Legacy diameter annotations remain metadata and do not create or average
footprints.

For image scale `s`, image array indices equal `s * legacy_xy`. SpatialData
raster centres use intrinsic coordinates `(column + 0.5, row + 0.5)`, so the
image-to-legacy-frame map is `(intrinsic_xy - 0.5) / s`. This offset reconciles
pixel-index and raster-centre conventions; it does not calibrate physical units
or shift the stored source positions.

Without positions, the result contains an owned unlinked table and any
independently interpretable images. Empty positioned tables retain their
`(0, 2)` arrays and declarations but remain unlinked. Empty physical row-frame
declarations are retained without an inferred frame when no images require one.
Stale reserved linkage
metadata is removed on the owned import, while original observation fields
remain. A malformed claimed position array fails instead of becoming a
positionless partial result.

## Persist an AnnData table

Use AnnData's H5AD or Zarr writers directly. Ordinary data and supported
position declarations persist; `spatialdata_attrs` still refers to source
SpatialData targets and does not embed those targets in an H5AD. To save an
explicitly unlinked table, remove that declaration on an owned copy:

```python
standalone = sdata.tables["table"].copy()
standalone.uns.pop("spatialdata_attrs", None)
standalone.write_h5ad("table.h5ad")
```

Keep the SpatialData container when target elements and their transformations
must persist together. See [table persistence limits](table_contracts.md#persistence-ownership-and-numerical-limits)
and [coordinate calibration](calibration.md) for downstream coordinate selection
and storage limitations.
