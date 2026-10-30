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
| `he_image`, `if_image` | `Affine(diag(pixel_size, pixel_size, 1) @ M)` |

`pixel_size` is the `experiment.xenium` value (0.2125 µm in the public bundles). 10x writes polygon vertices through the centres of boundary pixels in pixel-index coordinates, half a pixel up and left of the corner frame shared by masks, centroids and transcripts. The half-pixel translation registers the unchanged vertices: in samples of ten public bundles from XOA 1.0.2 to 4.0, more than 99.5 % of registered vertices fall on their own cell's mask pixels, against at most 96 % without it. Transcript `z` is in micrometres too.

## Table

MatrixMarket archives must contain exactly one of each required `cell_feature_matrix/{features,barcodes}.tsv.gz` and `cell_feature_matrix/matrix.mtx.gz`. Transcript-only reads extract the two metadata files without writing the matrix to temporary storage; traversing the compressed archive may still be necessary.

`X` is an eager float32 canonical CSR sparse array in the feature order of `cell_feature_matrix`. `feature_types=None` selects Gene Expression and, when present, Protein Expression; `"all"` keeps every feature type (negative controls, unassigned and deprecated codewords, genomic controls); a sequence selects exactly the named types, each of which must exist. Observation names are the matrix barcodes. `var` holds `gene_ids` and `feature_types`; protein features add `long_name`, `clone` and `codeword_index` from `protein_panel.json`.

Float32 count storage is retained without a maximum-count or precision ceiling. Negative and fractional counts fail. MatrixMarket inputs must declare integer coordinate records.

`obs` holds the `cells` table columns in source order with documented types: integer IDs are int64, other IDs are text, `nucleus_area` is NaN for cells without a nucleus. With `cells.zarr.zip`, `label_id` (uint32), `nucleus_centroid_x` and `nucleus_centroid_y` are added; XOA 2.0 cells tables lack `nucleus_count` and `segmentation_method`, which then also come from the store. `obsm["spatial"]` holds the float64 µm centroids.

Polygon-set cell and segmentation-method indexes must be nonnegative and below their respective population bounds, including unsigned source values. Stored nucleus counts must be nonnegative integers representable as int64. Malformed values raise a contextual `ValueError` before conversion or indexing.

The table annotates one returned element, chosen with `table_target`:

| `table_target` | Instance column | Requires |
| --- | --- | --- |
| `"cell_labels"` (default) | `label_id` | `include_cell_labels=True` |
| `"cell_boundaries"` | `cell_id` | `include_cell_boundaries=True`, `cell_geometry` `"polygons"` or `"both"` |
| `"cell_circles"` | `cell_id` | `include_cell_boundaries=True`, `cell_geometry` `"circles"` or `"both"` |
| `None` | none | an unlinked table |

A target that is not returned fails before any file is read.

Protein Expression values (XOA 4.0) are the mean stain intensity per cell after the 100-count offset was removed. The HDF5 matrix stores them multiplied by the root attribute `protein_scaling_factor` (10 in the public protein bundle); the reader divides once in float32 and records `{"scaling_factor", "metric"}` in `attrs["xenium"]["protein"]`. `protein_metric` must be `MEAN_PER_CELL_STAIN`. The factor must be a positive integer with a finite float32 representation. The MatrixMarket twins lack the factor, so protein features from them fail.

With `include_analysis=True`, 10x secondary analysis from `analysis/` or `analysis.tar.gz` is added: each clustering becomes a categorical `obs` column named by its method (`gene_expression_graphclust`, ...) with text categories in numeric order, and PCA and UMAP projections become float64 `obsm["pca_<method>"]` and `obsm["umap_<method>"]`. Cells that 10x filtered before clustering are missing values. Each `(kind, method)` must identify exactly one recognized CSV, including in archives with repeated filenames or alternate `analysis/` prefixes. Extracted files and archive members stream directly into Arrow without a separate complete encoded-input copy; source handles close after parsing, including on failure. Returned annotations and projection arrays are eager.

## Transcripts

Points have `x`, `y`, `z` (float32 µm), a categorical `feature_name` whose known categories are the count-matrix features in source order, `cell_id` with the vendor sentinel for unassigned transcripts (`UNASSIGNED`, or `-1` with integer IDs), `transcript_id`, `overlaps_nucleus`, `qv` and, when present, `fov_name`, `nucleus_distance`, `codeword_index`, `codeword_category` and `is_gene`. Binary columns of early bundles are decoded strictly as UTF-8. Every transcript is kept: the count matrix contains assigned transcripts with `qv >= 20`, so `points[points.qv >= 20]` reproduces it.

Parquet partitions follow row groups and carry a global row index with known divisions. A plain CSV is split into newline-delimited blocks of 64 MiB whose indexes restart at zero. Construction reads only the parquet footer or the CSV header; unknown features or cells, missing values and nonfinite coordinates fail when their partition is computed.

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

The alignment matrix `M` maps image pixels to morphology pixels; without it the image is assumed to share the morphology grid. Axes come from the OME metadata (`data_axes` overrides them); RGB samples become channels `r, g, b`, and channel-first images keep their OME channel names.

{func}`~spatialdata_io.xenium_explorer_selection` reads a Xenium Explorer coordinates export (`Selection,X,Y`, or `X,Y` with the name in the first comment line) as polygons in µm with `Identity()`, indexed by selection name in file order. Clicked cells are named `Cell <id>`. The result can be added to the `SpatialData` object or used with {func}`spatialdata.polygon_query`.

## Metadata

`attrs["xenium"]` holds JSON-serializable metadata: `experiment` (the `experiment.xenium` fields without image paths and Explorer files; the Xenium Ranger command line is never stored), `metrics_summary` with exact values, the `gene_panel` and `protein_panel` identities, and `protein`. Local paths are never recorded.

## Options and execution

`scale_factors` and `chunks` follow the shared [raster options](raster_options.md): by default a single scale, images in `(1, 1024, 1024)` chunks and labels in `(1024, 1024)` chunks; `xenium_aligned_image` takes the same options. Each label task opens the zip store, reads its chunk and closes it, so graphs hold paths, not open files. Constructing the result executes no pixel or transcript tasks; the table and shapes are eager.

## Python and command line

```python
from spatialdata_io import xenium

sdata = xenium("outs", cell_geometry="both", invalid_geometries="drop")
sdata.write("outs.zarr")
```

```bash
spatialdata_io xenium --input outs --output outs.zarr --feature-types all --no-include-transcripts
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
| `xenium_aligned_image(image_path, alignment_file, imread_kwargs, image_models_kwargs, dims, rgba, c_coords)` | `xenium_aligned_image(image_file, alignment_file=None, *, pixel_size, data_axes=None, scale_factors=(), chunks=(1, 1024, 1024))` |
| `xenium_explorer_selection(path, pixel_size, return_list)` | `xenium_explorer_selection(path, *, invalid_geometries="keep")` returning shapes in µm |
