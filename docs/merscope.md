# MERSCOPE

{func}`spatialdata_io.merscope` reads Vizgen instrument output and the Cellpose or Watershed output layouts of the Vizgen Post-processing Tool (VPT). It returns selected mosaics, transcript points, cell footprints and a linked expression table. Support describes the layouts and schemas below; filenames do not establish an instrument or software release number.

## Instrument layout

```text
slide/
└── region/
    ├── cell_by_gene.csv
    ├── cell_metadata.csv
    ├── cell_boundaries.parquet
    ├── detected_transcripts.csv
    └── images/
        ├── micron_to_mosaic_pixel_transform.csv
        ├── mosaic_Cyto_z0.tif
        ├── mosaic_DAPI_z0.tif
        ├── mosaic_Cyto_z2.tif
        └── mosaic_DAPI_z2.tif
```

Each product and its dependencies are required only when selected. Image-only loading requires mosaics and uses the optional micron-to-mosaic affine in `images/` when present. Native micron transcript points and cell boundaries require neither the affine nor an images directory. Table-only loading requires the count and metadata CSVs. Boundary-only loading requires no count or metadata file. All four inclusion flags default to `True`; selecting no products returns an empty SpatialData container with reader provenance.

`z_layers=None` selects every discovered image layer in ascending order. A nonempty sequence such as `(2, 0)` selects layers in the caller's order. Repeated, negative or noninteger indices fail; a scalar is not accepted. Stains are ordered lexically within each image. Every selected layer must have one file for every discovered stain, and all selected source planes must agree in spatial shape and dtype. Missing or duplicate stain/layer pairs fail.

`region_name` defaults to the region directory name and `slide_name` to its parent name. Their joined form, `<slide>_<region>`, supplies the element prefix:

| Element | Contract |
| --- | --- |
| `images["<slide>_<region>_z<index>"]` | Lazy `(c, y, x)` mosaic with source pixel dtype, lexical stain channels and source spatial shape. |
| `points["<slide>_<region>_transcripts"]` | Lazy projected XY transcript events. |
| `shapes["<slide>_<region>_polygons"]` | Eager cell boundaries indexed by exact textual EntityID, with a categorical geometry representation. |
| `tables["table"]` | Eager owned sparse expression, sparse control measurements and aligned cell annotations. |

## VPT layouts and transcript provenance

Set `vpt_directory` to the replacement analysis directory. Relative paths resolve against the instrument region root; absolute paths may be outside that root. Discovered artifacts, including symlink targets, remain within their owning instrument or analysis directory.

```text
region/
├── detected_transcripts.csv          # original instrument events
├── cell_by_gene.csv                  # original event vocabulary
├── images/                           # original mosaics and calibration
└── vpt/
    ├── cell_by_gene.csv              # replacement cell counts
    ├── cell_metadata.csv             # replacement cell metadata
    ├── cellpose_micron_space.parquet # Cellpose layout
    └── detected_transcripts.csv      # optional reassigned events
```

The Watershed layout uses the same instrument root and this analysis directory:

```text
region/
└── vpt/
    ├── cell_by_gene.csv
    ├── cell_metadata.csv
    ├── watershed_micron_space.parquet
    └── detected_transcripts.csv      # optional reassigned events
```

Exactly one supported boundary candidate is required when boundaries are included. The reader rejects ambiguity instead of giving one segmentation method priority.

An analysis-local `detected_transcripts.csv`, when present, supplies current assignments and uses the analysis-local count header as its feature vocabulary. Otherwise the reader uses the original instrument transcript CSV and requires the original instrument count header. The selected VPT count header cannot establish the vocabulary of an unrelated original event file. Missing required vocabulary files fail; the reader does not scan all transcript rows or substitute another vocabulary.

When original events accompany VPT cell replacements, their assignments become `original_cell_id` annotations. They have no current instance key and do not claim correspondence to replacement boundaries. Reassigned VPT events and standard instrument events retain `cell_id` as current assignment when that column exists. The `-1` unassigned sentinel is preserved. Current assignments are validated against the original complete table or boundary population when that independent population is available, including cells removed by the polygon policy; original assignments are not validated against replacement IDs.

## Tables and identities

`cell_by_gene.csv` begins with a positional cell-identifier field, followed by unique nonempty feature headers. Counts are finite nonnegative 64-bit integers, using an exact signed or unsigned count dtype as required by the source values. Count row and feature order are preserved, while cell metadata are aligned by exact identity. Duplicate, missing or conflicting cell populations fail. Integer identities are decoded without a floating intermediate and retained as exact text across observations, geometry indexes and point assignment columns.

`cell_metadata.csv` requires `EntityID`, `center_x` and `center_y`; named columns accept arbitrary order and additional annotations. Known numeric fields retain validated numeric representations; extra annotation fields are preserved as text. Centroids must be finite. Volume, when present, is finite and nonnegative. The output table retains micron centroids in `obsm["spatial"]` and source annotations, together with categorical `region`, `slide` and `dataset_id` fields. [Vizgen's metadata schema](https://vizgen.github.io/vizgen-postprocessing/output_data_formats/entity_metadata_format.html) describes centroid and volume units.

Table `uns["spatialdata_io_positions"]` describes unchanged vendor cell-centroid XY micrometres in the resolved vector frame. A table together with an unregistered mosaic uses `microns` for the table and `mosaic_pixels` for images, including when boundaries and transcripts are disabled.

Expression is a finite canonical SciPy CSR sparse array in `table.X`. Only case-sensitive names matching `Blank-[1-9][0-9]*` are separated as controls. Thus `Blank-1` is a control, while `Blank-0`, `Blank-01`, `blank-1` and `BlanketGene` remain expression features. Control measurements reside in CSR `obsm["blank"]`; `uns["blank_names"]` gives their source-header order. They are retained even though they are excluded from expression feature names. Both count products remain sparse.

With footprints included, the table links through categorical `cells_region` and exact textual `EntityID` to the independent footprint index. With footprints excluded, the table is normalized and unlinked; it has no stale SpatialData linkage metadata. Source annotations named `region`, `slide`, `dataset_id` or `cells_region` must agree exactly with the corresponding reader-created value; conflicting or null values fail. A source `cells_region` field is rejected when boundaries are excluded because the output table is unlinked. Equal values are preserved and normalized to the documented categorical representation. The reader owns newly created tables and does not mutate source files or caller selections.

## Cell boundaries and coordinates

Boundary Parquet requires `EntityID`, `ZIndex` and WKB `Geometry`, with optional
`Type` restricted to Cell/cell. Required columns are projected. Geometry must
be nonempty finite planar Polygon or MultiPolygon, and source IDs and plane
annotations must be valid under every policy. `ZIndex` may be null, as permitted by the [VPT geometry schema](https://vizgen.github.io/vizgen-postprocessing/output_data_formats/geometry_file_format.html); populated values must be nonnegative int64 plane indexes. Every slice participates in the XY footprint independently of its plane annotation. This applies to the supported instrument, Cellpose and Watershed layouts; it does not add support for other entity types. No geometry is repaired.

`invalid_geometries="keep"` retains ordinary all-z union footprints for cells
whose source slices are valid. If any slice is invalid, that cell instead
retains every literal Polygon component from all of its slices, flattening
MultiPolygons in source row/component order without changing vertices or holes.
The categorical `geometry_representation` column distinguishes `union_footprint`
from `source_components`. Literal components may overlap or remain invalid;
their summed area does not represent projected cell area. A failed or invalid
union of valid source slices raises rather than switching representations.

`"raise"` reports invalid cell IDs. `"drop"` removes every slice of affected
cells and their complete table rows, including aligned sparse blank measurements.
Original count/boundary identity agreement and reserved source annotations are
validated before filtering. Surviving IDs and row order are preserved. Lazy
transcripts retain their original assignments and population checks; their
assignments are not rewritten. Requested all-dropped boundaries fail. With
boundaries disabled no polygons are read and no cells are dropped.

Missing geometry and empty slices remain unsupported structural input, even
when another slice supplies a valid complete cell footprint. Empty occupancy
is a separate source-format decision from invalid topology; it is not silently
ignored or treated as an invalid polygon. Valid-cell union footprints retain
cells absent from z0, holes and disconnected components. Z occupancy and
per-slice annotations are not aggregated into false cell-wide values. Boundaries
remain two-dimensional projections, independent of selected image layers.

The calibrated `global` frame is source micrometres. Native vector coordinates
and table `obsm["spatial"]` remain unchanged; vectors carry identity. Mosaic
images use the full **inverse** of the micron-to-mosaic affine. Its bounded
64 KiB UTF-8 file contains a finite invertible homogeneous 3×3 matrix. For example:

```text
2  0.25  10
0  3     -7
0  0      1
```

This source matrix maps micron position `(2, 2)` to mosaic pixel `(14.5, -1)`;
the image's inverse maps that pixel back to `(2, 2)` in `global`. Anisotropy,
shear, reflection and translation are retained without resampling pixels.
Physical-only reads never inspect the pixel transform. Present malformed
calibration fails when images require it.

Without calibration, images alone retain the documented `global` mosaic-pixel
frame. If native vectors are also selected, images instead use `mosaic_pixels`
and vectors use `microns`, keeping unresolved registration explicit. A shared
frame is never inferred from dimensions. Source slice indices remain `z_raw`
annotations rather than physical Z.

## Transcript schema and lazy execution

Transcript CSVs require `gene`, `global_x` and `global_y`. Output coordinates `x` and `y` are NumPy float64 in micrometers. `gene` is categorical in the complete source count-header order, including controls, zero-count and unassigned-only features. Unknown names fail when their partition is computed.

Optional source fields are normalized as follows:

| Source | Output |
| --- | --- |
| `global_z` | Nonnegative integer `z_raw` annotation. |
| `cell_id` | Exact text current assignment, or `original_cell_id` for original events accompanying VPT replacement cells. |
| `fov`, `barcode_id` | Nonnegative int64 annotations. |
| `transcript_id` | Nonempty exact text codebook target identifier, including identifiers such as `ENST...`. |

The first unnamed source index and additional fields, including local pixel `x` and `y`, are excluded. The source `global_z` describes a stack index, so points have two spatial coordinates and no asserted physical Z calibration. The Dask row index is partition-local storage state and may repeat; it is not a biological event identifier. `transcript_id` identifies a codebook target and is not claimed to uniquely identify an event. [Vizgen's transcript schema](https://vizgen.github.io/vizgen-postprocessing/output_data_formats/detected_transcripts_format.html) distinguishes global micron XY, local pixel XY, stack Z and target identifiers.

Raster metadata and CSV headers are read eagerly. Construction of SpatialData executes zero pixel or transcript tasks. Expression uses bounded working batches into sparse arrays; tables, cell metadata and final geometry are eager. Event rows remain path-backed and projected, with exact lexical decoding before numeric or identifier conversion. Transcript byte partitions target approximately 8 MiB; Dask balances file blocks and extends boundaries through complete physical records, so actual partitions can be larger. This target describes partitioning rather than a total-memory ceiling. Count parsing uses numerical batches and CSV decoder buffers; final sparse arrays, annotations and geometry scale with the input. Each transcript record and its header must occupy one physical line. Quoted commas and quotes are supported; embedded newlines within quoted fields are unsupported. Every record must match the full source header width, even for fields excluded from the output.

CSV input must be valid. The Arrow decoder checks full record widths but does not enforce strict quoting: malformed quotes, including text after a closing quote, may be accepted and change decoded values. The reader does not perform a separate strict-syntax pass. Numeric precision, range, identifier and linkage checks still apply to decoded values.

Malformed unread coordinates, features and assignments raise with source context when the affected partition is computed. TIFF tasks open their source on execution, decode native tiles or bands of whole strips where supported, and close resources. Keep source files available until lazy computation or writing finishes; no open source handles are retained in the returned graphs.

`scale_factors` and `chunks` follow the shared [raster options](raster_options.md): by default a single scale and images in `(1, 1024, 1024)` chunks. A sequence such as `(2, 2)` creates a lazy image pyramid. Computing a region decodes only the TIFF tiles, or strips grouped into bands of at most 16 MiB, that intersect its chunks. The full-resolution source series and level remain fixed; each stain file must supply one grayscale plane with only singleton extra axes.

Missing or wrong-kind paths raise the corresponding filesystem exception. Ambiguous layouts, malformed schemas, inconsistent populations and invalid scientific values raise `ReaderFormatError`, a `ValueError` subclass. Invalid option types raise `TypeError`; invalid values raise `ValueError`. Validation failures do not return partial results. Provenance records the reader and package version without absolute input paths.

## Python and command line

```python
from spatialdata_io import merscope

sdata = merscope(
    "slide/region",
    vpt_directory="vpt",
    z_layers=(2, 0),
    scale_factors=(2, 2),
)
sdata.write("sample.zarr")
```

The CLI mirrors the reader, with repeatable layer, pyramid and chunk options:

```bash
spatialdata_io merscope --input slide/region --output sample.zarr \
  --vpt-directory vpt --z-layers 2 --z-layers 0 \
  --scale-factors 2 --scale-factors 2

spatialdata_io merscope --input slide/region --output counts.zarr \
  --no-include-images --no-include-transcripts --no-include-cell-boundaries
```

The former `vpt_outputs` mapping, `backend`, `imread_kwargs` and `image_models_kwargs` have been removed. Use `vpt_directory` and explicit raster options. Inclusion options are now `include_images`, `include_transcripts`, `include_cell_boundaries` and `include_table`. The default selects all image layers instead of z3; `z_layers=None` no longer disables images. Use `include_images=False` for that choice. Automatic four-level pyramids have been removed; request downsampling explicitly. Images default to `(1, 1024, 1024)` chunks instead of `(1, 4096, 4096)`, and `scale_factors=None` and `chunks=None` are no longer accepted: pass `()` for a single scale and explicit sizes for other chunks. Image loading follows the same shared TIFF path in every environment.
