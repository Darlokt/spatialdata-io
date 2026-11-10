# Xenium

{func}`spatialdata_io.xenium` reads a 10x Genomics Xenium Onboard Analysis (XOA) output bundle from XOA 1.0 to 4.x, or a Xenium Ranger output of the same layout. It returns the cell table, the segmentation masks, the cell and nucleus boundaries, lazy transcript points and the morphology images, all in one micrometre coordinate system. {func}`spatialdata_io.xenium_aligned_image` reads post-Xenium H&E or immunofluorescence images aligned in Xenium Explorer, and {func}`spatialdata_io.xenium_explorer_selection` reads Xenium Explorer selection exports.

## Bundle families

The `experiment.xenium` format version (`major_version`, `minor_version`, `patch_version`) selects one of three families. XOA 1.0 and 1.1 omit the version; their software version (`Xenium-1.0.x`, `xenium-1.1.x`) identifies them. Other versions fail.

| Family | Format version | XOA | Cell IDs | Mask label of a cell |
| --- | --- | --- | --- | --- |
| Integer IDs | absent or 1.x | 1.0–1.2 | integers | the cell ID |
| Hexadecimal IDs | 2.x–3.x | 1.3–1.9 | `aaaaaaab-1`, ... | `seg_mask_value` |
| Polygon sets | 4.x–6.x | 2.0–4.x | `aaaaaaab-1`, ... | polygon index + 1 |

The provenance attribute `spatialdata_io_format_version` records the format version; it is absent for XOA 1.0 and 1.1. Xenium Ranger outputs keep the format of their input bundle (`relabel`, `rename`) or of the Ranger release; the `xenium_ranger` block never selects the family.

Integer and hexadecimal families (XOA 1.0–1.9):

```text
outs/
├── experiment.xenium
├── cells.parquet                      # or cells.csv(.gz)
├── cell_boundaries.parquet            # or cell_boundaries.csv(.gz)
├── nucleus_boundaries.parquet         # or nucleus_boundaries.csv(.gz)
├── transcripts.parquet                # or transcripts.csv
├── cell_feature_matrix.h5             # or cell_feature_matrix/ or cell_feature_matrix.tar.gz
├── cells.zarr.zip                     # masks; required for labels
├── morphology_focus.ome.tif
├── morphology_mip.ome.tif             # optional product
├── metrics_summary.csv                # optional
├── gene_panel.json                    # optional
└── analysis/ or analysis.tar.gz       # optional product
```

Polygon-set family (XOA 2.0 onwards and Xenium Ranger):

```text
outs/
├── experiment.xenium
├── cells.parquet, cell_boundaries.parquet, nucleus_boundaries.parquet, transcripts.parquet
├── cell_feature_matrix.h5
├── cells.zarr.zip
├── morphology_focus/
│   ├── morphology_focus_0000.ome.tif  # XOA 2.0–3.x: one file per channel
│   └── ch0000_dapi.ome.tif, ...       # XOA 4.0: ch<NNNN>_<marker>.ome.tif
├── protein_panel.json                 # XOA 4.0 protein assay
├── metrics_summary.csv, gene_panel.json
└── analysis.tar.gz
```

Each tabular artifact is read from its parquet file when present, otherwise from exactly one CSV twin; two twins of one artifact are ambiguous. `transcripts.csv.gz` is a single gzip stream that cannot be split into partitions and is rejected; decompress it to `transcripts.csv`. `experiment.xenium` names the morphology focus entry, and the set is every `.ome.tif` file beside it. The 3D `morphology.ome.tif` stack is not read.

## Elements

| Element | Type | Content |
| --- | --- | --- |
| `morphology_focus` | image | `c, y, x`, OME channel names (for example `DAPI`, `ATP1A1/CD45/E-Cadherin`, `18S`, `alphaSMA/Vimentin`, protein markers), source dtype |
| `morphology_mip` | image | XOA 1.x maximum-intensity projection, `include_morphology_mip=True` |
| `he_image`, `if_image` | image | aligned post-Xenium images, see below |
| `cell_labels`, `nucleus_labels` | labels | uint32 masks of `cells.zarr.zip`, the exact 10x segmentation |
| `cell_boundaries` | shapes | cell polygons indexed by `cell_id` |
| `cell_circles` | shapes | centroid circles indexed by `cell_id`, column `radius` |
| `nucleus_boundaries` | shapes | nucleus polygons indexed by nucleus mask label `label_id`, column `cell_id` |
| `transcripts` | points | every decoded transcript |
| `table` | table | counts and cell annotations |

Labels are never derived from polygons. Circles have radius `sqrt(cell_area / pi)`, so each circle has the area 10x measured on the mask. In the integer and hexadecimal families every cell has one nucleus that carries the cell's label; polygon sets index nuclei separately, so a cell can have several nuclei or none.

## Coordinates

Every element is in the `global` coordinate system in micrometres, with its origin at the top-left corner of the morphology image.

| Element | Transformation to `global` |
| --- | --- |
| morphology images, labels | `Scale([pixel_size, pixel_size])` |
| `cell_boundaries`, `nucleus_boundaries` | `Translation([pixel_size / 2, pixel_size / 2])` |
| `cell_circles`, `transcripts` | `Identity()` |
| `he_image`, `if_image` | `Affine(diag(pixel_size, pixel_size, 1) @ M)`, or `Scale([pixel_size, pixel_size])` without an alignment matrix |
| Explorer selections | `Identity()` |

`pixel_size` is the `experiment.xenium` value (0.2125 µm in the public bundles). 10x writes polygon vertices through the centres of boundary pixels in pixel-index coordinates, half a pixel up and left of the corner frame shared by masks, centroids and transcripts. The half-pixel translation registers the unchanged vertices: in samples of ten public bundles from XOA 1.0.2 to 4.0, more than 99.5 % of registered vertices fall on their own cell's mask pixels, against at most 96 % without it. Transcript `z` is in micrometres too.

## Table

MatrixMarket archives must contain exactly one of each required `cell_feature_matrix/{features,barcodes}.tsv.gz` and `cell_feature_matrix/matrix.mtx.gz`. Transcript-only reads extract the two metadata files without writing the matrix to temporary storage; traversing the compressed archive may still be necessary.

`X` is an eager canonical CSR sparse array (float32 for official int32 HDF5 and MTX inputs) in the feature order of `cell_feature_matrix`. `feature_types=None` selects Gene Expression and, when present, Protein Expression; `"all"` keeps every feature type (negative controls, unassigned and deprecated codewords, genomic controls); a sequence selects exactly the named types, each of which must exist. Observation names are the matrix barcodes. Source feature names must be unique; duplicates raise rather than being renamed. `var` retains all selected source feature annotations, including `gene_ids`, `feature_types`, and available HDF5 fields such as `genome`. MatrixMarket sources supply only their own feature annotations. Feature annotation names that a SpatialData table does not accept are renamed with a warning, as described for `obs` below.

Selected protein features add `long_name`, `clone` and `codeword_index` from `protein_panel.json`. The panel codeword annotation is nullable uint64, with missing values for nonprotein features; present values are nonnegative integers. If any of those columns already exists in the source, panel enrichment raises rather than overwriting it. Without panel enrichment, those source columns are retained unchanged.

Float32 count storage retains its precision limits: integers above 2²⁴ may already have rounded in Scanpy. Duplicate HDF5 entries for one barcode and feature fail, as do retained negative or fractional counts. MatrixMarket inputs must declare integer coordinate records, but upstream parsing can truncate malformed tokens or sum invalid duplicates before returning; see [table numerical limits](table_contracts.md#persistence-ownership-and-numerical-limits). The Scanpy source decoder remains unchanged. Counts follow the [completed-count dtype policy](table_contracts.md#completed-raw-counts) before feature selection. Protein division follows this conversion and retains fractional measurement values.

`obs` holds the `cells` table columns in source order with documented types: integer IDs are positive uint64, counts are uint64, other IDs are text, `nucleus_area` is NaN for cells without a nucleus. A CSV twin is parsed by Arrow directly into these types (see [CSV twins](#csv-twins)); its undocumented columns stay text. With `cells.zarr.zip`, `label_id` (uint32), `nucleus_centroid_x` and `nucleus_centroid_y` are added; XOA 2.0 cells tables lack `nucleus_count` and `segmentation_method`, which then also come from the store. `obsm["spatial"]` holds the float64 µm centroids.

Additional `cells` columns whose names a SpatialData table does not accept are renamed once with a warning: invalid characters become `_` (`CD45 RO` becomes `CD45_RO`), `_index` becomes `index`, and a name equal to another column ignoring letter case gets a `_1`, `_2`, ... suffix. The documented columns and the reader's own `label_id`, nucleus centroids and `region` keep their names; a `cells` column named exactly like a reader column raises.

Table `uns["spatialdata_io_positions"]` describes unchanged canonical float64 XY as vendor cell centroids in `global` micrometres using the vendor cells schema. Nullable nucleus summaries are separate annotations.

Polygon-set cell and segmentation-method indexes must be nonnegative and below their respective population bounds, including unsigned source values. Native `cell_summary` storage must be float64, with one unique nonempty string name per column and the required nucleus centroid columns; unsupported storage types fail before conversion. NaN nucleus centroids mark cells without a nucleus; infinite centroids fail. Stored nucleus counts must be finite nonnegative integers below 2⁶⁴; they become uint64 after validating the float64 summary. Malformed values raise a contextual `ValueError` before conversion or indexing.

The table annotates one returned element, chosen with `table_target`:

| `table_target` | Instance column | Requires |
| --- | --- | --- |
| `"cell_labels"` (default) | `label_id` | `include_cell_labels=True` |
| `"cell_boundaries"` | `cell_id` | `include_cell_boundaries=True`, `cell_geometry` `"polygons"` or `"both"` |
| `"cell_circles"` | `cell_id` | `include_cell_boundaries=True`, `cell_geometry` `"circles"` or `"both"` |
| `None` | none | an unlinked table |

A target that is not returned fails before any file is read.

Protein Expression values (XOA 4.0) are the mean stain intensity per cell after the 100-count offset was removed. The HDF5 matrix stores them multiplied by the root attribute `protein_scaling_factor` (10 in the public protein bundle); the reader divides once in the selected floating dtype, using the accepted float32 divisor, and records `{"scaling_factor", "metric"}` in `attrs["xenium"]["protein"]`. `protein_metric` must be `MEAN_PER_CELL_STAIN`. The factor must be a positive integer. The MatrixMarket twins lack the factor, so protein features from them fail.

With `include_analysis=True`, 10x secondary analysis from `analysis/` or `analysis.tar.gz` is added: each clustering becomes a categorical `obs` column named by its method (`gene_expression_graphclust`, ...) with text categories in numeric order, and PCA and UMAP projections become float64 `obsm["pca_<method>"]` and `obsm["umap_<method>"]`. Cells that 10x filtered before clustering are missing values. Each `(kind, method)` must identify exactly one recognized CSV, including in archives with repeated filenames or alternate `analysis/` prefixes; methods that differ only in letter case count as the same method, because SpatialData compares names ignoring case. A clustering named exactly like a `cells` column raises; other unusable method names are renamed with a warning as above. Extracted files and archive members stream directly into Arrow without a separate complete encoded-input copy; source handles close after parsing, including on failure. Returned annotations and projection arrays are eager.

When labels are requested without a table or boundary polygons, the reader inspects the native mask metadata and calibration without loading cell IDs, summary vectors, segmentation methods or polygon membership arrays. It still checks the bundle family, named polygon-set structure, mask dtypes and matching grids, and the mask transform against `pixel_size`. Single-set transcript-assignment imports cannot return labels. Unused identity and summary arrays are neither loaded nor validated in this mode. Table and polygon reads retain their identity, label and polygon-membership cross-checks. Label pixels always remain lazy, and the native store closes before return.

## Transcripts

Points have NumPy `x`, `y`, `z` (float32 µm), a categorical `feature_name` whose known categories are the count-matrix features in source order, `cell_id` with the vendor sentinel for unassigned transcripts (`UNASSIGNED`, or `-1` with integer IDs), `transcript_id`, `overlaps_nucleus`, `qv` and, when present, `fov_name`, `nucleus_distance`, `codeword_index`, `codeword_category` and `is_gene`. These other columns are Arrow-backed (`uint64[pyarrow]` `transcript_id`, `uint8[pyarrow]` `overlaps_nucleus`, `float[pyarrow]` `qv`, `int64[pyarrow]` or `string[pyarrow]` `cell_id`, ...); the coordinates are NumPy because `PointsModel` requires it. Binary columns of early bundles are decoded strictly as UTF-8. Every transcript is kept: the count matrix contains assigned transcripts with `qv >= 20`, so `points[points.qv >= 20]` reproduces it.

Cell-ID indexes on cell polygons and circles, and owning `cell_id` annotations on nucleus polygons, also retain Arrow storage (`uint64[pyarrow]` or `string[pyarrow]`). Nucleus indexes derived from mask labels retain their NumPy integer type. Identifier values and order are unchanged.

Parquet partitions follow row groups and carry a global row index with known divisions. A plain CSV is split into newline-delimited blocks of 64 MiB whose indexes restart at zero. Construction reads only the parquet footer or the CSV header; unknown features or cells, missing values and nonfinite coordinates, `qv` or `nucleus_distance` fail when their partition is computed. A CSV has no spelling for a missing value: every numeric field must hold a number, so an empty or `NA` field is a malformed record, and `NaN` is a nonfinite value. Each block is parsed directly into the documented types, with the grammar described under [CSV twins](#csv-twins).

Integer domains are declared per artifact, so an unassigned transcript does not change a cells-table identifier type:

| Artifact / field | Parsed type and domain | Returned representation |
| --- | --- | --- |
| Integer-family cells and boundaries / `cell_id` | uint64, positive | Arrow-backed shapes; NumPy uint64 in final table annotations |
| Integer-family transcript / `cell_id` | int64, known cell ID or `-1` | Arrow-backed int64 |
| Cells / documented count columns | uint64, zero allowed; optional counts may be missing | NumPy uint64, nullable pandas UInt64 when missing |
| Boundaries / `label_id` | uint32, positive | NumPy uint32 mask labels |
| Transcript / `transcript_id`, `overlaps_nucleus` | uint64 and uint8 | Arrow-backed unsigned integers |
| Transcript / `codeword_index` | uint32, zero-based | Arrow-backed uint32; 2³²−1 is preserved verbatim |
| Secondary analysis / `Cluster` | uint64, positive | Categorical decimal text, sorted numerically |
| Protein panel / `codeword_index` | uint64, zero allowed | Nullable pandas UInt64 in `var` |

The [10x output descriptions](https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-understanding-outputs) define transcript codewords as panel indexes and boundary labels as nonzero mask values. The [native Zarr specification](https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/advanced/xoa-output-zarr) specifies uint32 masks and zero-based uint32 codeword indexes, with the maximum uint32 value representing an unknown codeword. The reader keeps that value without translating it to a negative sentinel. Integer-family native store IDs and mask-linked boundaries must fit positive uint32 labels; cells and circle linkage retain the full uint64 range. Matrix barcodes are compared as exact canonical decimal text, and signed transcript assignments are matched against the unsigned cell population without floating-point conversion. Checked Arrow parsing and casts reject negatives and overflow in unsigned source columns, also when written in hexadecimal; positive fields additionally reject zero. Arrow wraps hexadecimal in a signed type instead (`0xffffffffffffffff` would read as -1, the unassigned sentinel), so the signed integer-family transcript `cell_id` of a CSV is read as text and must be decimal digits or `-1` before its checked int64 cast.

## CSV twins

Arrow parses CSV twins, a plain `transcripts.csv`, the secondary-analysis CSVs, `metrics_summary.csv`, alignment matrices and Explorer selection exports directly into declared types. CSV twins and Parquet columns normalize to the same declared types; integer-family `cell_id` is positive uint64 in cells and boundaries, and signed int64 in transcripts to preserve `-1`; other families use text IDs. Arrow's CSV grammar is the contract:

- spaces and tabs around a number are trimmed, and integers may be hexadecimal (`0x10` reads as 16);
- an integer field rejects `2.0`, `2e0`, `+2` and values outside its type, except that the integer-family transcript `cell_id` accepts only decimal digits or `-1` (see above);
- floats are correctly rounded directly to their type (float32 for vertices and transcript coordinates), so `1e-400` reads as 0;
- every record has the header's width, also for columns that are not read; some malformed quoting is accepted, but every `transcripts.csv` record occupies one physical line, and a line with an odd number of double quotes fails when its partition is computed;
- in `cells.csv(.gz)` an empty field of a typed column is a missing value (nullable optional counts, NaN `nucleus_area`) and `NaN` reads as NaN; boundary, transcript, analysis, alignment and selection CSVs have no numeric missing values, so an empty numeric field is malformed;
- header names must be unique.

`metrics_summary.csv` has no fixed numeric schema because its metrics vary between releases, so Arrow infers the type of each metric: an empty field becomes `None`, integers become `int` (also hexadecimal such as `0x10`; beyond int64 they become floats), other numbers `float` (including `+2`, `NaN` and `inf`), `true` and `false` booleans, and other values stay text (`1_0`, `NA`). Dates and times keep their text. Quoted metric values may contain newlines.

Alignment matrices contain exactly three records and metrics exactly one data record; both small files are read whole.

## Invalid geometry

10x simplifies boundary polygons, and some are not valid geometries (`shapely.is_valid`). `invalid_geometries` applies to the returned polygons:

- `"keep"` (default) returns them unchanged;
- `"drop"` removes a cell whose polygon is invalid together with its circle, nucleus polygons and table row, and an invalid nucleus polygon alone;
- `"raise"` fails and names the invalid elements.

Without `cell_boundaries` no cell is dropped. Rings with fewer than four coordinates, which Xenium Ranger imports can write, cannot be represented and require `"drop"`. Labels and transcripts are never modified.

## Xenium Ranger and stripped deposits

`relabel`, `rename` and `resegment` outputs read like XOA bundles. Segmentation imports differ:

- a transcript-assignment import has one polygon set and no masks or nuclei: pass `include_cell_labels=False`, `include_nucleus_labels=False`, `include_nucleus_boundaries=False` and a shapes `table_target`;
- a cells-only import has an empty nucleus set: pass `include_nucleus_boundaries=False`.

Public deposits often keep only the CSV twins and MatrixMarket counts. Without `cells.zarr.zip`, labels are unavailable, so choose `include_cell_labels=False`, `include_nucleus_labels=False` and `table_target="cell_boundaries"` or `None`. Hexadecimal-ID nucleus boundaries carry their labels only in `cells.zarr.zip` and then need `include_nucleus_boundaries=False`. The remaining products equal the full bundle's.

Some deposits keep only `transcripts.zarr.zip`, the tiled transcript store that Xenium Explorer displays, and neither `transcripts.parquet` nor `transcripts.csv`. The reader does not read transcripts from that store: it records no `cell_id` or `overlaps_nucleus`, so its transcripts cannot be assigned to cells as the transcripts element requires. Pass `include_transcripts=False` to read the remaining products, or obtain the transcript table from the depositors.

## Aligned images and selections

At the bundle root, `*he_image.ome.tif` and `*if_image.ome.tif` become `he_image` and `if_image`, each with its optional `*imagealignment.csv` (`include_aligned_images`). For other names and locations use {func}`~spatialdata_io.xenium_aligned_image` with the bundle's pixel size:

```python
from spatialdata_io import xenium, xenium_aligned_image

sdata = xenium("outs")
sdata["he_image"] = xenium_aligned_image(
    "Xenium_V1_he_image.ome.tif",
    "Xenium_V1_he_imagealignment.csv",
    pixel_size=sdata.attrs["xenium"]["experiment"]["pixel_size"],
)
```

The alignment matrix `M` maps image pixels to morphology pixels; without it the image is assumed to share the morphology grid. Axes come from the TIFF metadata; `data_axes` overrides them with one string of distinct letters including `X` and `Y`, such as `"YXS"` (any letter case; `S` and `C` both become the channel axis, so at most one may appear), checked before the image is read. Declared axes that cannot become `c, y, x` raise a contextual `ReaderFormatError`; `data_axes` that do not fit the image raise `ValueError` naming it. Channel names follow what the file declares:

- interleaved samples of a TIFF that declares an RGB colour model (photometric RGB, or JPEG-compressed YCbCr as in typical H&E scans) become `r, g, b`, with `a` for declared alpha;
- other channels keep their OME channel names, or their OME channel IDs when names are missing or repeated, when the channel axis is the declared OME channel or samples axis; samples without a colour model are named `<name or ID>:sample:<index>`;
- without OME metadata, or when `data_axes` reads another declared axis (for example Z) as the channels, channels are numbered `0, 1, ...`; such a reinterpretation previously failed the OME channel count.

Three channels are never assumed to be RGB from their count: the Breast IDC H&E, stored as Hematoxylin, Eosin and Residual channels, keeps those names. An OME channel count that does not match a corresponding channel axis raises.

{func}`~spatialdata_io.xenium_explorer_selection` reads a Xenium Explorer coordinates export (`Selection,X,Y`, or `X,Y` with the name in the first comment line) as polygons in µm with `Identity()`, indexed by selection name in file order. Clicked cells are named `Cell <id>`. The result can be added to the `SpatialData` object or used with {func}`spatialdata.polygon_query`.

## Metadata

`attrs["xenium"]` holds JSON-serializable metadata: `experiment` (the `experiment.xenium` fields without image paths and Explorer files; the Xenium Ranger command line is never stored), `metrics_summary` with exact values, the `gene_panel` and `protein_panel` identities, and `protein`. Local paths are never recorded.

## Options and execution

`scale_factors` and `chunks` follow the shared [raster options](raster_options.md): by default a single scale, images in `(1, 1024, 1024)` chunks and labels in `(1024, 1024)` chunks; `xenium_aligned_image` takes the same options. Scale factors whose product exceeds the smaller side of an image or mask raise a `ValueError` before any pixel is read.

Format errors are `ReaderFormatError`s that name the artifact (`cells table`, `transcripts`, `morphology focus image`, ...), the detected `experiment.xenium` format version and the path. Each label task opens the zip store, reads its chunk and closes it, so graphs hold paths, not open files. Constructing the result executes no pixel or transcript tasks; the table and shapes are eager.

`path_policy="relaxed"` reads git-annex and DVC bundles whose files are symlinks to targets elsewhere, including a single-file XOA 1.x `morphology_focus.ome.tif`. A multi-file `morphology_focus/` set whose members link to separate targets is the exception: tifffile looks for the other channel files beside the real location of the opened file, so the read fails with `RasterFormatError` instead of returning zero-filled channels. See [Dataset paths and symlinks](dataset_paths.md#known-limitation-multi-file-ome-tiff-sets).

## Python and command line

```python
from spatialdata_io import xenium

sdata = xenium("outs", cell_geometry="both", invalid_geometries="drop")
sdata.write("outs.zarr")
```

```bash
python -m spatialdata_io xenium --input outs --output outs.zarr --feature-types all --no-include-transcripts
```

## Changes from earlier releases

| Earlier argument or behavior | Now |
| --- | --- |
| `cells_boundaries`, `nucleus_boundaries`, `cells_labels`, `nucleus_labels`, `transcripts`, `morphology_focus`, `aligned_images`, `cells_table` | `include_*` options |
| `cells_as_circles` | `cell_geometry` and `table_target` |
| `morphology_mip=True` | `include_morphology_mip=False` |
| `gex_only=True` | `feature_types=None`, which includes protein |
| `n_jobs`, `imread_kwargs`, `image_models_kwargs`, `labels_models_kwargs`, deprecated aliases | removed; use `scale_factors` and `chunks` |
| five-level pyramids and `(1, 4096, 4096)` chunks | single scale and `(1, 1024, 1024)` chunks |
| `global` in morphology pixels | `global` in micrometres, including `z` |
| instance column `cell_labels` | `label_id` or `cell_id` |
| `cell_id_str_from_prefix_suffix_uint32`, `prefix_suffix_uint32_from_cell_id_str` | removed; the string `cell_id` is in every table |
| `xenium_aligned_image(image_path, alignment_file, imread_kwargs, image_models_kwargs, dims, rgba, c_coords)` | `xenium_aligned_image(image_file, alignment_file=None, *, pixel_size, data_axes=None, scale_factors=(), chunks=(1, 1024, 1024))`; `data_axes` is a string such as `"YXS"` |
| aligned images: any three- or four-sample image named `r, g, b(, a)`; missing or repeated OME names gave `0, 1, ...` | `r, g, b(, a)` only for a declared RGB colour model; otherwise OME names or IDs; a channel-count mismatch raises |
| unusable `cells` column, HDF5 feature annotation and clustering method names failed when the table was linked | renamed with a warning |
| `xenium_explorer_selection(path, pixel_size, return_list)` | `xenium_explorer_selection(path, *, invalid_geometries="keep")` returning shapes in µm |
| CSV numbers parsed by a strict text parser (alignment, selection) or by Arrow with its default missing-value spellings (cells and boundary twins, analysis) | Arrow with declared types throughout; empty numeric fields are missing only in `cells.csv(.gz)`, `NA` and `null` no longer mean missing |
| `metrics_summary.csv` values typed by their spelling: `+2` an integer, `0x10`, `NaN` and `inf` text | types inferred by Arrow: `+2` is `2.0`, `0x10` is 16, `NaN` and `inf` are floats, `true`/`false` are booleans |
| integer-family transcript `cell_id` in `transcripts.csv` followed Arrow's integer grammar (`0x10` read as 16, `0xffffffffffffffff` as the unassigned -1) | decimal digits or `-1` only |
| `transcripts.csv` quoted fields could span lines, with a result depending on block boundaries; stray quotes were kept | a line with an odd number of double quotes fails when its partition is computed |
| transcript columns other than `x, y, z` and `feature_name` as NumPy (`uint64`, `float32`, ...) | Arrow-backed (`uint64[pyarrow]`, `float[pyarrow]`, ...) |
| shape cell-ID indexes and annotations as NumPy integers or pandas text | Arrow-backed integers or text; mask-label indexes retain their NumPy type |
| out-of-root symlinked artifacts accepted | rejected by default; `path_policy="relaxed"` accepts them, except a multi-file `morphology_focus/` set whose members link to separate targets ([Dataset paths and symlinks](dataset_paths.md#known-limitation-multi-file-ome-tiff-sets)) |
