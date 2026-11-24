# MERSCOPE

{func}`spatialdata_io.merscope` reads Vizgen instrument output and the Cellpose, Cellpose2 or Watershed output layouts of the Vizgen Post-processing Tool (VPT). It returns selected mosaics, transcript points, cell footprints and a linked expression table. Use the directory layout below to identify the export.

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

All four product flags default to `True`. A table-only read needs the count
and metadata CSVs. Boundary-only reads need just the Parquet file. Transcripts
also need a count header for their feature vocabulary. Mosaics use the transform
when available. Selecting no products returns an empty SpatialData object with
provenance.

`z_layers=None` reads every mosaic layer in ascending order. A sequence such
as `(2, 0)` selects layers in that order. Each selected layer must contain one
grayscale plane per stain, and all planes must share shape and dtype. Channels
are ordered by stain name.

Output names start with `<slide>_<region>`, taken from the region directory
and its parent. Use `slide_name` and `region_name` to override them. Invalid
characters in a derived name become underscores with a warning.

| Element | Contract |
| --- | --- |
| `images["<slide>_<region>_z<index>"]` | Lazy `(c, y, x)` mosaic with source pixel dtype, lexical stain channels and source spatial shape. |
| `points["<slide>_<region>_transcripts"]` | Lazy projected XY transcript events. |
| `shapes["<slide>_<region>_polygons"]` | Eager cell boundaries indexed by exact textual EntityID, with a categorical geometry representation. |
| `tables["table"]` | Eager owned sparse expression, sparse control measurements and aligned cell annotations. |

## VPT layouts and transcript provenance

Set `vpt_directory` to the replacement analysis directory. Relative paths resolve against the instrument region root; absolute paths may be outside that root. Discovered artifacts, including symlink targets, must remain within their owning instrument or analysis directory unless `path_policy="relaxed"` ([Dataset paths and symlinks](../user/paths.md)).

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

The [Vizgen Cellpose2 workflow](https://vizgen.github.io/vizgen-postprocessing/analysis_vignettes/segmentation_heart_dataset_cellpose2.html) uses `cellpose2_micron_space.parquet` in the same directory. VPT can also write mosaic-pixel boundaries for the same cells. The reader selects only the micron-space file and raises if several supported micron-space segmentations are present.

The Watershed layout uses the same instrument root and this analysis directory:

```text
region/
└── vpt/
    ├── cell_by_gene.csv
    ├── cell_metadata.csv
    ├── watershed_micron_space.parquet
    └── detected_transcripts.csv      # optional reassigned events
```

Choose one supported segmentation in the replacement directory. When it contains
its own transcript CSV, those are the reassigned events and their feature
vocabulary comes from the VPT count header. Otherwise the reader uses the
original instrument events and their original count header. In that case the
assignment column becomes `original_cell_id`, since its values describe the old
segmentation. Reassigned and instrument-only events use `cell_id`. The unassigned
sentinel is `-1`.

## Tables and identities

| Input | Required fields and values |
| --- | --- |
| `cell_by_gene.csv` | First field is the cell identifier. The remaining, uniquely named fields are nonnegative uint64 counts in source feature order. Each count record must fit in a 1 MiB text block. |
| `cell_metadata.csv` | Named `EntityID`, `center_x`, `center_y` columns in any order. Cell IDs must match the count population exactly. Physical fields use finite float64 values and volume must be nonnegative. Extra annotations remain text, including empty strings and literal `NA`. |

IDs retain their exact text across table rows, geometry indexes and transcript
assignments. Count row order is preserved. Integer fields accept hexadecimal
and surrounding spaces or tabs. They reject decimal fractions, exponents,
leading plus signs, separators and overflow. Floats round directly to float64
and may underflow to zero. Two optional metadata fields, `fov` and
`transcript_count`, accept empty entries or a nonnegative integer as decoded
in float64, including values VPT writes as `12.0`. They become nullable uint64,
so a large decimal value can round **before** conversion to integer.
[Vizgen metadata schema](https://vizgen.github.io/vizgen-postprocessing/output_data_formats/entity_metadata_format.html)

- `X` is a finite CSR array. Only names matching `Blank-[1-9][0-9]*` exactly
  are blank controls. Their measurements are in sparse `obsm["blank"]` with
  names in `uns["blank_names"]` in source order. Genes and blanks share one
  dtype. [Raw-count precision](../user/tables.md#raw-count-precision)
- `obsm["spatial"]` holds the vendor XY centroids in `global` micrometres.
  `uns["spatialdata_io_positions"]` describes the positions. Table rows also
  carry categorical `region`, `slide` and `dataset_id`.
- If footprints are included, `cells_region` and `EntityID` link table rows
  to their polygons. Otherwise the table is unlinked. Source fields that use
  these reserved names must agree with the returned values. Unsupported
  annotation names are sanitized with a warning.

## Cell boundaries and coordinates

Boundary Parquet files require `EntityID` (text or integer), `Geometry` (binary
WKB) and `ZIndex` (null or a nonnegative integral value representable as
uint64). An optional `Type` must contain `Cell` or `cell`. Each row supplies a
nonempty, finite, planar Polygon or MultiPolygon in micrometres.
[Vizgen geometry schema](https://vizgen.github.io/vizgen-postprocessing/output_data_formats/geometry_file_format.html)

Footprints project **all** slices onto XY, independently of the image layers
selected. A cell can be absent from z0. The result keeps holes and disconnected
components. Choose a policy for invalid topology:

| `invalid_geometries` | What happens to a cell with an invalid slice |
| --- | --- |
| `"keep"` (default) | Retain every source component in slice order with its vertices and holes. `geometry_representation` is `source_components` for this cell and `union_footprint` for a cell with valid slices. Overlapping source components can double-count area. |
| `"raise"` | Report the invalid cell IDs. |
| `"drop"` | Remove all slices for the affected cells and their table rows, including blank measurements. Surviving IDs keep their order. Transcripts retain their original assignments. |

Missing or empty slices, a failed union of valid slices, or dropping every
requested footprint raises an error. With boundaries disabled, the reader reads
no polygons and drops no cells.

Vectors and table centroids use `global` in micrometres. Mosaics map into
that frame using the **inverse** of the vendor's micron-to-mosaic affine. The
file contains three whitespace-separated rows of three finite coefficients
forming an invertible homogeneous matrix. For example:

```text
2  0.25  10
0  3     -7
0  0      1
```

This maps (2, 2) µm to mosaic pixel (14.5, −1). The image transform maps that
pixel back to (2, 2) µm. Pixel data retain their original values. Without the
transform, images stay in `mosaic_pixels` and the vectors remain in `global`.

## Transcripts

The transcript CSV requires `gene`, `global_x` and `global_y`. XY is float64 in
micrometres. The categorical `gene` column follows the full count header,
including controls and zero-count features.

| Optional field | Output |
| --- | --- |
| `global_z` | Float64 `z_raw` stack position. The format supplies no physical depth for this value. |
| `cell_id` | Current or original cell assignment, according to the source of the events. |
| `fov`, `barcode_id` | Nonnegative uint64 annotations. |
| `transcript_id` | Text codebook target, which can repeat across events. |

The reader omits the unnamed source index and local pixel XY. The returned
partition-local index can repeat. Each CSV record must occupy one physical line
and match the header width. Quoted commas and doubled quotes are supported.
Malformed data, unknown genes or invalid current assignments raise when the
affected partition is computed. Current assignments are checked against the
complete cell population when a table or boundaries supply one.
[Vizgen transcript schema](https://vizgen.github.io/vizgen-postprocessing/output_data_formats/detected_transcripts_format.html)

Tables and boundaries are eager. Keep the transcript CSV and TIFFs in place
until computing or writing the lazy points and mosaics. For image chunk sizes
and pyramids, see [raster options](../user/rasters.md).

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
python -m spatialdata_io merscope --input slide/region --output sample.zarr \
  --vpt-directory vpt --z-layers 2 --z-layers 0 \
  --scale-factors 2 --scale-factors 2

python -m spatialdata_io merscope --input slide/region --output counts.zarr \
  --no-include-images --no-include-transcripts --no-include-cell-boundaries
```
