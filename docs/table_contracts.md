# Table inputs and outputs

Readers validate tabular inputs before constructing their SpatialData products.
The accepted columns, expression meaning, identifiers, coordinate units and
linkage remain specific to each format. Some products are unlinked tables or
channel metadata, and some create no table.

## Input fields

Named fields are selected by name. Required fields must be present; optional
fields are validated when present. Reordering named columns is accepted where
the format supports named fields. Additional columns are either preserved as
annotations or explicitly ignored. Ambiguous duplicate column names fail.
Headerless files retain their positional field counts and meanings.

Visium headered tissue positions accept the six documented fields in any
order and ignore additional fields. Headerless positions require exactly six
fields. Barcode order is aligned to the expression table, including spots
outside tissue. Grid coordinates must be nonnegative integers fitting int64.
Pixel coordinates must be finite, nonnegative float64 values; conversions
that lose large integer precision or underflow to zero fail.

Curio metric exports have a positional exception: their first two columns
contain the metric name and value. A subsequent named `Category` column selects
the metric group. These first two columns are not freely reorderable. CODEX
expression features are recognized by their cycle/channel names; additional
metadata columns are not automatically treated as expression. DBiT barcode
maps retain their two-column positional contract.

## Annotation storage

Reader-created tabular intermediates use explicit Arrow-backed columns.
Imported H5AD observation and feature annotations retain their source types
unless the reader documents a required normalization. Additional annotations
are preserved according to the reader's format contract. Leading zeros in
string identifiers, integer signedness and width, and observation and feature
order remain significant.

Final AnnData annotations use storage compatible with the supported AnnData
and SpatialData versions:

| Annotation | Final storage |
| --- | --- |
| Required integer fields | NumPy integers with the declared width and signedness |
| Nullable integer annotations | Matching pandas nullable integer types |
| Boolean annotations | NumPy bool or pandas nullable boolean |
| Floating annotations | NumPy float32 or float64 as declared |
| Text | pandas string types, including Arrow-backed strings |
| Regions and semantic categories | pandas categorical |

Floating nulls become NaN only where the reader explicitly defines them as
equivalent missing annotations. Otherwise an unsupported conversion fails.
The readers do not change process-global pandas or AnnData settings. Default
nullable-string persistence requires AnnData 0.13 or later and pandas 3.0 or
later, reflected in the package requirements.

This final storage conversion is temporary: general Arrow-backed pandas
numeric columns currently lack AnnData serialization support. It does not
alter expression values or reinterpret identifiers. Support for those columns
is tracked in [AnnData issue #948](https://github.com/scverse/anndata/issues/948).

## Expression and linkage

Expression is returned as eager, finite, canonical SciPy CSR sparse arrays.
Feature and observation order follow the reader's documented format contract;
expression is not densified to validate or link annotations.

Tables are linked only after their final identifiers, regions, and spatial
targets have been constructed. Instance identifiers are non-null, and each
region/instance pair is unique. When an independent target index is available,
table membership must match it exactly. Lazy label images are not computed
merely to enumerate identifiers.

Large transcript points use lazy Dask DataFrames and their Parquet persistence
path. Shapes use GeoDataFrames. These products do not pass through AnnData's
annotation conversion.

## Reader output inventory

The contracts below apply to the implemented layouts described by each reader's
API or linked guide. Layout differences that change table contents are stated
explicitly. Unless specified otherwise, readers create one table named `table`;
`obs` rows describe spots, beads or segmented cells, and `var` rows describe
measured features. Required generated fields are non-null. Annotation columns
retain source order unless the reader explicitly selects or derives them.
Additional fields follow the reader-specific preservation rules below.

Counts, marker intensities and imported analysis expression have different
scientific meanings. Readers do not normalize expression, synthesize universal
QC columns, add a `counts` layer, or replace imported `raw`. Vendor fields such
as `total_counts`, `transcript_count`, `expCount` and `geneCount` are source
summaries, possibly over different populations or features. They need not equal
returned `X` sums after selection, filtering or protein rescaling.

`uns["spatialdata_attrs"]` declares linkage on linked tables only. Reader and
package provenance belong to `SpatialData.attrs`, and are not automatically
copied into table `uns`. Generated shape indexes provide consistency with the
completed target element; when generated from the same table IDs they do not
constitute an independently encoded vendor population. Label dtype/range
checks do not prove presence in label pixels. Readers keep those pixels lazy.

Table embeddings are eager snapshots in the units stated below. A calibrated
transformation on an image, shape or points element does not automatically
transform `obsm["spatial"]`. Nor does the name `global` alone prove physical
units or register independent samples. Refer to [calibration](calibration.md)
for the spatial-element frames.

### Visium

Space Ranger headerless/headered position layouts and HDF5/plain or compressed
MTX counts share the same table contract. Matrix barcodes determine observation
names and order; every matrix row, including off-tissue spots, is retained.
All source feature classes and their order are kept. Feature symbols are made
unique unless disabled; stable IDs are in textual `var["gene_ids"]`. Modern inputs also supply
textual `feature_types` and available additional feature metadata; legacy
inputs do not fabricate them.

`X` contains source counts, with the dtype returned by Scanpy (int32 HDF5
usually becomes float32; MTX is float32). `obs` adds int64 `in_tissue`,
`array_row`, `array_col`, textual `spot_id` and categorical `region` in that
order. `spot_id` repeats the barcode and links to the generated spot circles
indexed by barcode, using `region` as the region key. Float64
`obsm["spatial"]` contains full-resolution image-pixel centres in XY order;
these pixel fields are not also `obs` columns. `uns["spatial"]` retains the
library's metadata and vendor scale factors without embedded images. No
expression layers or `raw` are created. See the
[API](generated/spatialdata_io.visium.rst) for image selection and overrides.

### Curio

The five-file Seeker export returns a bead table and circles named `cells`.
Rows, names, feature order, expression meaning and dtype come from the H5AD.
Imported observation and feature annotations retain their types and missingness.
Categorical `cluster`, categorical `region="cells"` and textual `instance_id`
are added. Cluster assignments align by exact bead name. The final circle index
matches these names; the target is generated from the same bead population.

Dynamic Moran statistics join into active `var` by feature ID and are missing
for unlisted features. Their source order follows existing feature annotations.
Metric categories become dictionaries in `uns`; `Top_cluster_defining_features`
is an eager DataFrame. Enrichment never changes `raw.var`. Source
`obsm["spatial"]` is retained, and an identical `X_spatial` is removed. Its
units remain the source coordinate units; bead radius does not establish a
physical calibration. Only `X` is converted to canonical CSR. Every inherited
layer, including a layer named `counts`, retains its source representation.
Other embeddings, aligned mappings, nested metadata and `raw` are preserved.
Inherited `spatialdata_attrs` is rejected. See the
[API](generated/spatialdata_io.curio.rst) for the export layout.

### CODEX

CSV and FCS inputs return events in source order with positional string names
`"0"`, `"1"`, … . Only complete `cycNNN_chNNN` names become features, in
source order. `X` contains marker intensities, including valid negative and
fractional values; CSV becomes float64, and FCS numerical widths follow the
source. Feature names are the cycle/channel IDs; FCS can add textual
`var["marker"]` aligned from its metadata.

All other source columns remain in `obs`, including integer `cell_id`
(int64 or uint64 as needed), categorical `region`, and numerical pixel `x`/`y`.
CSV coordinates are float64 and additional CSV annotations remain text; FCS
metadata retain their native primitive types. Optional annotations may be
missing. Float64 `obsm["spatial"]` repeats XY pixels. Circles are linked by
`(region, cell_id)` to their generated index; no independent segmentation
population is inferred. No layers, `raw` or additional embeddings are created.
See the [API](generated/spatialdata_io.codex.rst) for CSV/FCS selection.

### ISS

The explicit image/labels/H5AD layout retains imported row names/order,
feature order and expression dtype. Active feature names are made unique;
`raw` retains its own feature vocabulary and metadata. Imported annotations,
embeddings, aligned mappings and ordinary `uns` survive unchanged, apart from
validated reserved output fields. No position array is derived.

Categorical `region` names `<dataset_id>_labels_image`. The default integer
`instance_id` is parsed exactly from observation names, using int64 or uint64;
an explicit `instance_key` selects an existing integer column, promoting widths
below 16 bits. These IDs are checked against the label dtype range, without
reading label pixels. `X` and an optional documented `counts` layer become CSR;
other layers and `raw` retain their source storage. Inherited
`spatialdata_attrs` is rejected. The [API](generated/spatialdata_io.iss.rst)
describes the explicit artifact names and options.

### Steinbock

Deepcell and Ilastik segmentation selections share one table contract. Imported
H5AD row/feature order and expression dtype remain. Source `image`,
`centroid-0`, `centroid-1` and other annotations remain in `obs`; an equal
redundant `Image` column is removed. Generated integer `cell_id` (int64 or
uint64) and categorical `region` link to `<image_stem>_labels`. IDs derive
from compound observation names and are range-checked against the selected
label dtype, without checking pixel membership.

Active feature metadata column spaces become underscores, including
`Final Concentration / Dilution` becoming `Final_Concentration`; values retain
their source types. `obsm["spatial"]` becomes XY pixel centroids in
`(centroid-1, centroid-0)` order and retains the supported source numerical
width. Other mappings and metadata remain. `X` and optional `counts` become
CSR; other layers and `raw` retain their values and representation. Renaming
active feature columns does not rename `raw.var` columns. Inherited
`spatialdata_attrs` is rejected. See the
[API](generated/spatialdata_io.steinbock.rst) for sample layouts.

### DBiT

The H5AD plus barcode-map layout sorts observations by A then B, retaining
original names. Slicing aligns every observation mapping, both axes of `obsp`
and the rows of `raw`; raw features retain their independent order.
Imported expression dtype, annotations and missingness remain. `array_A` and
`array_B` are one-based int64 grid coordinates; textual `sample` and
categorical `region` contain the dataset name, and int64 `pixel_id` is the
zero-based sorted row position. Float64 is not inferred as a physical unit:
`obsm["spatial"]` contains integer `(A, B)` grid positions.

The generated grid index checks final membership in `(region, pixel_id)`;
source linkage is intentionally replaced. `X` and optional `counts` become
CSR; other layers, embeddings, feature mappings, ordinary `uns` and `raw`
survive. Integer duplicate accumulation that would overflow its source dtype
fails. The optional image determines the square-grid display scale but does
not rewrite table positions. See [DBiT](dbit.md).

### seqFISH

Every selected ROI returns `table_<roi>` in count-row and feature-header order.
`X` contains int64/uint64 counts; `var` has feature names and no columns.
Observation names are original positional strings, retaining gaps after polygon
drops. Required `obs` fields are float64 physical `area` (µm²) and integer
`instance_id` (int64/uint64). Categorical `region` is added only when labels
are returned, linking to `<roi>_Segmentation`; range checks do not enumerate
label pixels. Shapes-only reads remain unlinked.

Float64 `obsm["spatial"]` contains XY µm centroids in the ROI frame. No other
embeddings, expression layers or `raw` are generated. The source top-level
GeoJSON feature IDs establish boundary membership; properties, including an
unrelated `properties.id`, are annotations. The [seqFISH guide](seqfish.md)
describes the independent ROI frames, shape selection and whole-row drops.

### MERSCOPE

Instrument, VPT Cellpose and VPT Watershed layouts share the table contract;
selection without counts returns no table. `X` contains int64/uint64 counts in
exact textual cell-ID and source feature order. `obs_names` are those cell
IDs. Canonical `Blank-N` controls move to sparse `obsm["blank"]`, retaining
their order in `uns["blank_names"]`; other controls stay in `X`. `var` has
feature names and no columns.

Metadata retain source column order: `center_x`/`center_y` and optional
`min_x`, `max_x`, `min_y`, `max_y`, `volume`, `anisotropy`,
`perimeter_area_ratio`, `solidity` are float64; optional `transcript_count`
is int64/uint64 and `fov` is nullable int64. Other source annotations stay
text. Generated categorical `region`, `slide`, `dataset_id`, textual
`EntityID`, and (only with boundaries) categorical `cells_region` have distinct
roles: actual linkage uses `(cells_region, EntityID)`, not `region`.
Float64 `obsm["spatial"]` contains XY µm centres. No layers or `raw` are
created. All-z footprint membership is checked against the independently
encoded boundary IDs; a geometry drop removes the complete cell row and its
blank measurements. See [MERSCOPE](merscope.md) for modes and frames.

### MCMICRO

WSI and TMA workflows concatenate quantification by selected label order, then
source row order. They return no table without quantification. Positional
string observation names preserve that concatenation order. `X` is float64
mean marker intensity, with features in canonical MCQuant panel order. `var`
retains textual `marker_name` and every additional panel column; repeated
marker names receive MCQuant suffixes on every occurrence.

`obs` contains categorical `region` and int64 `CellID`, optional nullable
float64 morphology and additional text annotations in their declared source
order. `X_centroid`/`Y_centroid` leave `obs` and become float64 local pixel
`obsm["spatial"]`. Area is pixel², axis lengths and centroid annotations are
pixels, Orientation follows the source angle convention, and Eccentricity,
Solidity and Extent are dimensionless. Calibrated spatial-element transforms
do not rescale these annotations. `(region, CellID)` links to label images;
ID range checks do not prove label-pixel membership. No layers or `raw` are
created. See [MCMICRO](mcmicro.md) for the exact morphology names and angle
convention, independent core frames and parser bounds.

### CosMx

The supported flat-file layouts always return `table`, including when labels
are excluded. `X` contains int64 counts, including controls, in expression
feature order. Background `cell_ID=0` rows are removed; metadata align by
`(fov, cell_ID)`. Vendor `cell` values name observations when available,
otherwise positional string names are generated. `var` has no columns.

`obs` contains int64 `fov`/`cell_ID`, source annotations and categorical
`region` even in unlinked mode. Optional `Area`, `Area.um2`, `AspectRatio`,
`Width`, `Height`, and `Mean.*`/`Max.*` stain summaries are nullable float64;
other metadata stay text. Centre columns leave `obs`:
`obsm["spatial"]` is local XY pixels, and `obsm["global"]` is global XY
pixels, both float64. The word global does not convert pixels into µm.
Linkage, when requested, uses `(region, cell_ID)` with label dtype/range
validation, without enumerating pixels. No layers or `raw` are created.
See [CosMx](cosmx.md) for FOV transforms and chained gzip-member decoding,
including single-member input.

### MACSima

Both implemented OME schema versions and folder modes return `<stack>_table`
for each image stack, plus `<stack>_nuclei_table` when nuclei are separated.
These are channel metadata tables with shape `(0, n_channels)` and `X=None`.
They have no observations, generated positions, linkage, layers or `raw`.

`var_names` match the associated image channel names exactly and in order.
The ordered columns are textual `name`, integer `cycle`, textual
`imagetype`, textual `well`, nullable integer `ROI`, textual `fluorophore`,
textual `clone`, and float64 `exposure` in milliseconds; source-specific missing annotations
remain missing. See [MACSima](macsima.md) for channel naming and grouping.

### Stereo-seq

The supported compact-numbered, numbered and `outs` layouts use the same
product-specific contracts; GEF schema profiles determine source feature
fields independently of the producer release. See [Stereo-seq](stereoseq.md)
for supported profiles and product names.

Square-bin tables `<binN>_table` contain uint64 raw counts, source feature
order and first-occurrence coordinate order. Duplicate coordinate/feature
records sum deterministically. `obs` contains categorical `region` and
int64 `instance_id`; positional string observation names identify the bins.
`var` retains source `gene` or `geneID`/`geneName`, `offset` and record-span
`count`. Integer native-grid XY positions are in `obsm["spatial"]`;
optional uint64 `exon` is a CSR layer. `uns["stereoseq"]` describes origin
and resolution. Tables link to the returned bin points by exact IDs.

Cell and adjusted-cell tables contain uint64 raw counts in source cell/feature
order with decimal source-ID observation names. Source `obs` fields are
`id`, `offset`, `geneCount`, `expCount`, `dnbCount`, `area`, categorical
`cellTypeID`/`clusterID` and optional `cellExon`, followed by final linkage.
`instance_id` is uint64 and `region` is categorical.
`var` retains source gene identity/names, `offset`, `cellCount` or `cellcount`,
`expCount`, `maxMIDcount` and optional `geneExon`. Source integer widths remain
for summary annotations; summaries are not recomputed QC. Native-grid XY
centres, optional uint64 CSR `exon`, origin/resolution metadata and available
cell-type names are retained. Linkage uses `(region, instance_id)` with exact
returned point/polygon IDs; invalid-cell drops filter the complete cell row.

Analysis tables are separate products with their own point targets. Imported
expression values, dtype, names/order, annotations, embeddings, ordinary
metadata and `raw` remain. `obsm["spatial"]` is native-grid XY, and matching
GEF inputs establish source identity without replacing analysis values.
Only `X` and present `counts`, `log1p`, `exon` layers become CSR. Other
layers and aligned mappings retain their storage. Final `region` and
`instance_id` link to analysis points. Nonfinite expression fails during
bounded final normalization with the H5AD path attached to the error.

### Xenium

Integer-ID, hexadecimal-ID and polygon-set layouts share the table meaning;
source schema generations determine optional annotations. `include_table=False`
returns no table. Matrix barcodes determine row names/order and selected
features retain source order. Symbols become unique. Default selection keeps
Gene Expression and available Protein Expression; `"all"` includes source
controls, and a sequence selects explicit feature types.

Gene/control entries are transcript counts. Protein entries are rescaled
per-cell mean stain intensities and may legitimately be fractional after one
division. Official int32 HDF5 counts become float32 through Scanpy. Alternate
source dtypes follow Scanpy; protein division currently requires floating
storage. Required textual `var` fields are `gene_ids`, `feature_types`;
modern source feature metadata are retained. Protein annotation adds nullable
text `long_name`/`clone` and nullable int64 `codeword_index`.

Required `obs` fields are `cell_id` (int64 for early IDs, text otherwise),
float64 µm `x_centroid`/`y_centroid`, µm² `cell_area`/`nucleus_area`, and
int64 `transcript_counts`, `control_probe_counts`, `control_codeword_counts`,
`total_counts`. Nucleus area may be NaN for cells without a nucleus. Optional
nullable int64 summaries are `unassigned_codeword_counts`, `deprecated_codeword_counts`,
`genomic_control_counts`, `nucleus_count`; other source fields retain their
declared type. A cell store adds uint32 `label_id`, float64 µm nucleus
centroids, and available nucleus counts/segmentation methods. Cells/counts/store
IDs must agree exactly in order. Float64 µm `obsm["spatial"]` contains XY
source cell centres, not polygon-derived centres.

Optional secondary analysis adds categorical clustering columns and float64
PCA/UMAP arrays; cells omitted by that analysis retain missing values. No
expression layers or `raw` are created. `table_target` selects labels
`(region, label_id)`, boundaries/circles `(region, cell_id)`, or an unlinked
table without generated `region`. Store mappings and source boundary IDs
supply membership evidence. Protein/provenance metadata are attached to
`SpatialData.attrs`. See [Xenium](xenium.md) for feature classes, selections,
analysis names and format families.

### Visium HD

Space Ranger 3.x/4.x supported bin, cell and nucleus products have independent
tables: `square_XXXum_table`, `cell_table`, and requested `nucleus_table`.
No unselected table is fabricated. Bin/cell barcodes and features retain matrix
order. Feature metadata follow 10x (`gene_ids`, `feature_types`, available
`genome` and additional annotations); symbols become unique unless disabled.
Bin and cell dtypes follow Scanpy. Nucleus expression independently sums raw
2 µm bins in guarded int64, then converts to float32 with its precision limits.
It is not copied from the selected cell matrix.

Bin `obs` contains uint8 `in_tissue`, uint32 `array_row`/`array_col`, float64
`pxl_row_in_fullres`/`pxl_col_in_fullres`, uint32 `instance_id` and categorical
`region`. Cell/nucleus `obs` contain decoded integer `instance_id` and
categorical `region`. Bin float64 `obsm["spatial"]` gives physical XY µm
centres; cell/nucleus physical centres exist only when vendor centroid
properties are supplied. Tables link by `(region, instance_id)` to actual
returned shape indexes or derived grid-label IDs. No layers or `raw` are
created. See [Visium HD](visium_hd.md) for filtering, geometry-drop semantics,
source precision/duplicate handling and supported layouts.

### Products without AnnData

Generic images/labels/GeoJSON and ancillary Xenium aligned-image/selection
readers create no AnnData. Large transcript points remain lazy Dask DataFrames;
shape annotations remain GeoDataFrames. Neither requires a placeholder table.

## Persistence, ownership and numerical limits

Readers own freshly loaded tables and do not persist their mutations to source
files or change process-global settings. H5AD and SpatialData Zarr persistence
retain scientific values, names, annotation missingness, aligned mappings,
linkage and native coordinate arrays. Deserialization may return SciPy sparse
matrices instead of the reader-return sparse arrays. H5AD can also reconstruct
categorical text with pandas' NaN string sentinel rather than `pd.NA`; category
values, codes and ordering remain. Neither behavior changes the reader-return
contract.

Float32 cannot exactly represent every integer above 2²⁴. Scanpy's conversion
may round large counts before the reader sees them; a later cast cannot recover
them. HD nucleus output has the same final float32 limitation after guarded
int64 accumulation. Generic SciPy duplicate summation of noncanonical narrow
integer imported H5AD expression can overflow; DBiT, raw Stereo-seq and HD have
their documented guards. No unrestricted exact-count guarantee is made for
these imported or float32 paths.

Shared HDF5 validation checks nonnegative/integral entries retained by Scanpy
before local duplicate reduction; filtering may remove unselected records.
No extra source scan is performed. MTX has a separate upstream limitation:
the installed parser can truncate malformed integer-header tokens and reduce
duplicates before returning. Header checks and finite/integer result checks
cannot prove validity of every original MTX record. Valid integer records
retain the documented count behavior; strict malformed-record rejection needs
a separate shared parser decision.
