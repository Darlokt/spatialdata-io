# Stereo-seq

{func}`spatialdata_io.stereoseq` reads independently selected transcriptomic products from the documented SAW 7.0/7.1 numbered layouts, SAW 8.0/8.1/8.2 `outs` layouts, the compact numbered workflow, and explicit flat exports. It returns raw GEF counts, derived H5AD analysis and registered images as separate products.

The reader records its name and package version in SpatialData provenance. It does not record the producing SAW release: directory numbering does not identify it, and the GEF, H5AD and TIFF files the reader reads do not store it. As format version it records the GEF schema version `gef-<n>`, taken from the root `version` attribute of every square or cell GEF read in one call (for example `gef-2` or `gef-4`). This is the GEF file-format version, not a SAW release. No format version is recorded when a read GEF has no `version` attribute, when no count GEF is read (analysis or image products alone, even when a GEF supplies the image pitch), or when the read GEFs report different versions; the last case also warns. A `version` that is not one positive integer fails. Protein, H5MU/OMNI and microorganism products are outside this reader's supported contract.

When a GEF declares its `omics` attribute, it must be the UTF-8 text `Transcriptomics`. Historical GEFs without that attribute remain supported through their validated transcriptomic field contract.

## Select products explicitly

No products are selected by default. A call without a selection raises before dataset I/O.

```python
from spatialdata_io import stereoseq

# Read only a stored square bin. No cells, analysis or images are required.
squares = stereoseq("run", bin_sizes=(50,))

# Read raw cell counts and centroids, independently of square counts.
cells = stereoseq("run", cell_bins="unadjusted")

# Select both raw cell populations and matching registered microscopy.
combined = stereoseq(
    "run",
    cell_bins="both",
    image_roles=("registered",),
    scale_factors=(2, 2),
)
```

| Option | Selection |
| --- | --- |
| `bin_sizes=()` | No square counts. Pass distinct positive stored sizes, or `"all"` to read every stored bin in numeric order. |
| `cell_bins="none"` | No raw cells. Choose `"unadjusted"`, `"adjusted"` or `"both"`. |
| `cell_geometry="centroids"` | Return source cell centers as points. Choose `"polygons"` for literal stored contours under `invalid_geometries`. |
| `analysis_files=()` | No analysis. Pass exact H5AD paths; clustering resolutions are never all loaded implicitly. |
| `image_roles=()` | No images or masks. Select `"registered"`, `"mask"`, `"tissue_cut"`, `"distance_mask"` or `"expression"`; all matching stains are included. Mask roles return labels. |
| `sample_id=None` | Infer a unique source sample; supply its full identifier when several samples are present. Dots in identifiers are retained. It only selects files and does not name elements. |
| `scale_factors=()`, `chunks=(1, 1024, 1024)` | A single scale in one-channel 1024-pixel tiles. Both apply to every selected image and mask; see [raster options](../user/rasters.md). |

Only stored square bins are read. A tissue GEF containing bin1 cannot satisfy `bin_sizes=(50,)`; the reader reports the available bins instead of deriving a coarser bin or choosing another count population. Explicit `"all"` can require substantial memory.

## Supported layouts

The trees below show candidate products. Only selected products are required; a cell-only, analysis-only or image-only dataset does not need a tissue GEF. Discovery examines the recognized immediate directories, without recursively searching arbitrary folders. Conflicting layout families and ambiguous selected artifacts fail.

### Compact numbered workflow

```text
run/
├── 02.count/          <sample>.raw.gef
├── 03.register/       <sample>_regist.tif; <sample>_mask.tif
├── 04.tissuecut/      <sample>.tissue.gef or <sample>.gef
├── 041.cellcut/       <sample>.cellbin.gef
├── 05.spatialcluster/ <sample>.spatial.cluster.h5ad
└── 051.cellcluster/   <sample>.cell.cluster.h5ad
```

This tree follows the [pinned compact workflow](https://github.com/STOmics/SAW/blob/41e70aa12bb16da8c9ddae344863cef2ca12f735/script/stereoPipeline.sh#L99-L109), without assigning a release to an input from its directory names. That workflow establishes unadjusted cells and historical square analysis; it does not establish adjusted-cell or modern analysis outputs. Separately exported products can use the explicit source options.

### SAW 7.0 and 7.1 numbered layouts

```text
run/
├── 03.count/          <sample>.raw.gef
├── 04.register/       <stain>_<sample>_regist.tif; corresponding mask/cut TIFFs
├── 05.tissuecut/      <sample>.tissue.gef; <sample>.gef where exported
├── 051.cellcut/
│   ├── <sample>.cellbin.gef
│   ├── <sample>.adjusted.cellbin.gef
│   └── <stain>_<sample>_mask_edm_dis_<distance>.tif
├── 06.spatialcluster/ <sample>.bin<N>_<resolution>.spatial.cluster.h5ad
└── 061.cellcluster/
    ├── <sample>.cell.cluster.h5ad
    └── <sample>.adjusted.cell.cluster.h5ad
```

The [SAW 7.0 A8](https://github.com/STOmics/SAW/blob/0808e44619f84b67d44c063b2fd24762f6633051/Documents/UserManual/Stereo-seq_Analysis_Workflow_User_Manual_A8.pdf) and [SAW 7.1 A9](https://github.com/STOmics/SAW/blob/cebf5171105c7896539535a60ac8f293e2f62c4f/Documents/UserManual/Stereo-seq_Analysis_Workflow_User_Manual_A9.pdf) manuals agree on these paths. Expanded distance masks are cell-correction outputs under `051.cellcut`, while registered microscopy and its mask/cut images are under `04.register`.

Older numbered exports may instead use `<sample>.spatial.cluster.h5ad`. The reader retains that historical spelling, requiring native `uns["bin_type"]="bins"` and `uns["bin_size"]` metadata. Missing clustering resolution remains unknown; this is separate from the bin/resolution filename documented for SAW 7.0/7.1.

### SAW 8.0, 8.1 and 8.2 layouts

Pass either the task root or its direct `outs` directory:

```text
task/
└── outs/
    ├── feature_expression/
    │   ├── <sample>.gef
    │   ├── <sample>.raw.gef
    │   ├── <sample>.tissue.gef
    │   ├── <sample>.cellbin.gef
    │   └── <sample>.adjusted.cellbin.gef
    ├── analysis/
    │   ├── <sample>.bin<N>_<resolution>.h5ad
    │   ├── <sample>.cellbin_<resolution>.h5ad
    │   └── <sample>.cellbin_<resolution>.adjusted.h5ad
    └── image/
        ├── <sample>_<stain>_regist.tif
        ├── <sample>_<stain>_tissue_cut.tif
        ├── <sample>_<stain>_mask.tif
        ├── <sample>_<stain>_mask_edm_dis_<distance>.tif
        └── bin1_img_tissue_cut.tif
```

The [8.0](https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.0/analysis/outputs/count-outputs), [8.1](https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.1/analysis/outputs/count-outputs) and [8.2](https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.2/analysis/outputs/count-outputs) output tables share this directory and filename grammar. Their Matrices pages additionally list the plain `<sample>.gef`. The adjusted H5AD suffix follows the clustering resolution, unlike the numbered-layout spelling.

The documented no-image tissue mask is `bin1_img_tissue_cut.tif`. The reader also accepts `bin1_img.tif` as an additional expression-image export, without attributing that file to every SAW 8 final-output inventory. Expression exports are recognized in the image directory for `outs` layouts and in registration/tissue-cut directories for numbered layouts.

### Flat exports and source overrides

Individually downloaded GEF/TIFF files can remain in a flat directory. Count files then require explicit source paths; they do not need to be moved into a native SAW output tree:

```text
export/
├── <sample>.tissue.gef
├── <sample>.cellbin.gef
├── <sample>.adjusted.cellbin.gef
└── <sample>_HE_regist.tif
```

```python
sdata = stereoseq(
    "export",
    bin_sizes=(1,),
    cell_bins="unadjusted",
    square_counts_file="sample.tissue.gef",
    cell_counts_file="sample.cellbin.gef",
)
```

`adjusted_cell_counts_file` selects the source for adjusted cells. Overrides are also available in structured layouts. They resolve relative to the supplied dataset root; absolute caller paths are permitted. An override does not select a product by itself. Paths discovered from a dataset must remain inside that dataset root unless `path_policy="relaxed"` ([Dataset paths and symlinks](../user/paths.md)).

For qualified square exports such as `sample.HE.gef` or `sample.region.label.gef`, supply both `sample_id="sample"` and the explicit `square_counts_file`. This identifies the sample without truncating sample IDs that contain dots; conflicting GEF sample metadata still fails.

The default square source is the unique tissue/full GEF. Raw whole-chip counts require `square_counts_file="...raw.gef"`. Raw, tissue and full GEFs represent distinct populations. If multiple tissue/full candidates exist, choose one explicitly; the reader does not decide which population you intended.

## GEF fields and exact counts

Feature fields are interpreted by name, independently of their column order. Supported physical profiles are:

| Profile | Square feature fields | Cell feature fields | Feature identity |
| --- | --- | --- | --- |
| `name_only` | `gene`, `offset`, `count` | `geneName`, `offset`, `cellCount` or `cellcount`, `expCount`, `maxMIDcount` | Source `gene` or `geneName`. |
| `id_and_name` | `geneID`, `geneName`, `offset`, `count` | `geneID`, `geneName`, `offset`, `cellCount` or `cellcount`, `expCount`, `maxMIDcount` | Source `geneID`; `geneName` is a display annotation. |

These descriptions are storage profiles, not public selectors or release labels. Feature identities must be unique, nonempty UTF-8 text. Duplicate display names are retained when stable feature IDs distinguish them. Cell-expression feature indices may be uint16 or uint32 independently of the feature profile. Counts accept supported unsigned integer widths; XY coordinates are exact signed or unsigned 32-bit grid values, widened without changing their values. Compound integer fields and count/exon vectors accept either byte order, independently for each field or vector. Count destinations and categorical annotations use native byte order without changing integer width, signedness or values; expression never passes through floating storage before exact accumulation is established.

Raw `X` is an eager canonical `scipy.sparse.csr_array`, using float32 when the maximum completed count is at most 2²⁴, otherwise float64 through 2⁵³. Square records with the same coordinate and feature are summed. Completed values above 2⁵³ fail. Repeated feature records within a native cell span are malformed and fail. Square feature spans and native cell-expression spans must cover their source expression records contiguously, without gaps or overlaps.

Aligned square `exon` and cell `cellExpExon` payloads become a separate canonical `layers["exon"]` with the same selected float32/float64 dtype as X, with the same feature/observation order and permitted square aggregation. Per-cell `cellExon`, per-feature `geneExon`, source summary columns and cell-type names are retained when present. The alternate `geneExp` traversal and NNZ-sized internal exon payloads are not duplicated into `uns`.

## Output names, order and linkage

| Selected product | Target | Table |
| --- | --- | --- |
| Stored square bin N | `points["bin<N>_genes"]` | `tables["bin<N>_table"]` |
| Unadjusted cell centers | `points["cells_centroids"]` | `tables["cells_table"]` |
| Adjusted cell centers | `points["adjusted_cells_centroids"]` | `tables["adjusted_cells_table"]` |
| Unadjusted cell contours | `shapes["cells_polygons"]` | `tables["cells_table"]` |
| Adjusted cell contours | `shapes["adjusted_cells_polygons"]` | `tables["adjusted_cells_table"]` |
| Derived analysis | `points["<analysis>_points"]` | `tables["<analysis>_table"]` |

Raw feature order follows the GEF, including zero-occurrence features. Square observations follow the first occurrence of each XY pair in the expression records, including positions with zero totals. Their table `instance_id` columns contain consecutive int64 integers starting at zero, matching the points' `RangeIndex`. Cells follow source cell order and retain the source integer cell IDs, including gaps. Raw AnnData observation names are decimal strings of the instance IDs. Derived analysis keeps its source observation names as its point instance IDs. Physical coordinates are available as float64 XY micrometres in `obsm["spatial"]`; `obsm["spatial_grid"]` retains exact source grid values. Returned points contain native int64 `x` and `y` columns. Cell-type and cluster IDs remain categorical annotations.

Each GEF stores its groups and datasets in the selected file. HDF5 soft and external links, virtual datasets, and external dataset storage are rejected.
Square-bin names identify a unique positive numeric size. Two names for the
same size, such as `bin1` and `bin01`, are ambiguous in both count reading and
image-only calibration.

Square-bin `var` preserves the source gene identity fields, `offset` and
`count`. Cell and adjusted-cell tables preserve `id`, `offset`, `geneCount`,
`expCount`, `dnbCount`, `area`, categorical `cellTypeID` and `clusterID`, and
optional `cellExon` in `obs`. Their `var` includes gene identity and display
name, `offset`, `cellCount` or `cellcount`, `expCount`, `maxMIDcount` and
optional `geneExon`. Retained integer summaries keep their source widths.
Where supplied, `uns["stereoseq"]["cell_type_names"]` lists the original cell
type names.

To combine the uint64 cell instance IDs with int64 IDs from square bins or
other tables, choose a common integer dtype that represents both populations
exactly. NumPy promotes an implicit mix of uint64 and int64 to float64.

Every linked table has categorical `obs["region"]` and an `obs["instance_id"]` that exactly identifies its independent target. A selected table with no observations fails because each declared region must occur in the linked table; no empty region or partial dataset is invented.

## Derived analysis identity

Analysis preserves H5AD values, observation and feature order, additional annotations, embeddings, layers and imported `raw`. It does not replace raw GEF counts or rerun vendor analysis. Returned `X` is a canonical CSR sparse array retaining sufficient source dtype, with direct exact float32/f64 promotion when duplicate totals exceed source storage. Documented expression layers `counts`, `log1p` and `exon` are normalized to sparse arrays; other layers retain their imported representation. Imported `raw` remains the source analysis snapshot rather than the reader's canonical raw GEF product.

Imported sparse storage must be structurally valid, malformed storage raises a `ValueError` during expression normalization, noting the slot, reader and H5AD path. Duplicate entries in `X` and documented expression layers are combined, preserving sufficient source dtype and promoting integers losslessly only when needed. Float64 fallback requires exact completed values; absence of a supported exact result raises with file/slot context. After a successful read, one `UserWarning` per affected analysis file, `stereoseq <file name>: ...`, lists the combined entries and dtype changes of each slot. Summing repeated raw GEF records does not warn.

Analysis `obs` and `var` column names that a SpatialData table does not accept (for example `my note`, or `Region` beside the reader's `region`) are renamed with one warning per frame and file, as `my_note` and `Region_1`. The reader-owned and validated columns `x`, `y`, `id`, `region` and `instance_id` keep their names. Top-level analysis `uns` keys follow the same rule with one warning per file (`uns["my key"]` becomes `uns["my_key"]`); the keys the reader reads or adds, `bin_type`, `bin_size`, `resolution`, `sn`, `spatialdata_io_positions` and `spatialdata_attrs`, keep their spelling, so `Resolution` beside `resolution` becomes `Resolution_1`. Only the keys change: values, their order and nested keys stay as stored, because SpatialData checks only top-level `uns` keys. Raw GEF annotation columns have fixed, valid names.

```python
analysis = stereoseq(
    "task",
    analysis_files=("outs/analysis/sample.cellbin_0.5.h5ad",),
)

paired = stereoseq(
    "task",
    cell_bins="unadjusted",
    analysis_files=("outs/analysis/sample.cellbin_0.5.h5ad",),
)
```

Native observation names encode `(x << 32) | y` as decimal text; the supported reindexed form is `<prefix>:<x>_<y>`. `obsm["spatial"]` must contain exact nonnegative uint32-range grid coordinates, and redundant `obs["x"]`/`obs["y"]` must agree. Optional `obs["id"]` contains unique nonnegative integer source cell IDs. Native `uns["sn"]` identifies the sample when present and must agree with the filename; `obs["orig.ident"]` remains an arbitrary imported annotation. Filename population/bin identity and native metadata must agree when both are present.

Square analysis observations have unique XY coordinates. When corresponding raw GEF counts are selected, every analysis coordinate and feature must belong to that source. Coincident cell centroids require distinct source cell IDs to verify the matches. Analysis point IDs are its preserved observation names. A filtered or reordered analysis is valid and keeps its own order and point target; rows and features are never joined by position or silently intersected. Analysis-only input requires physical pitch metadata in `uns["resolution"]`; a matching selected GEF may provide it otherwise.

The analysis base name is `cells_analysis`, `adjusted_cells_analysis` or `bin<N>_analysis`. A filename clustering resolution appends its filename spelling as the suffix `_r<resolution>`, with `.` replaced by `_`, `+` dropped and the exponent marker lowercased: `0.5` gives `_r0_5`, `0.50` gives `_r0_50`, `10.0` gives `_r10_0` and `1E+3` gives `_r1e3`. Two selected files of one population whose suffixes coincide (`1e3` and `1E+3`) fail. Historical filenames without clustering resolution omit that suffix. This clustering value is separate from the physical pitch in `uns["resolution"]`.

## Coordinates and geometry

Every target and image stores native DNB-grid coordinates and maps to `global` in micrometres with `Scale(pitch / 1000)`, using the common source pitch in nanometres. `global` always means micrometres. Registered images are already resampled to the matching normalized expression grid. Source point/polygon coordinates and image pixels are unchanged, and `obsm["spatial_grid"]` retains exact native grid values, including the original dtype of imported analysis embeddings. Canonical `obsm["spatial"]` uses the common pitch to express those source reference positions in micrometres. `uns["spatialdata_io_positions"]` describes the units, position meaning, frame and calibration. These table arrays are snapshots; later element registration requires explicit coordinate refresh. Square coordinates are used as stored, without multiplication by bin size. Analysis-only inputs use their native `uns["resolution"]` calibration. Image-only reads inspect scalar pitch attributes from exact-sample GEF companions in supported producer directories, without reading count or coordinate arrays. Without such a companion (including flat image exports), images and masks have no physical calibration: they are placed in the `image_pixels` coordinate system by identity and have no `global` transformation. Ambiguous companions or conflicting declared pitches fail.

Each DNA nanoball forms a round capture spot. A square bin groups those spots by grid position. For example, bin50 sums counts across 50 rows × 50 columns. At a pitch of 500 nm between neighbouring spot centres, the bin spans 25 µm per side, including the gaps between spots. The reader returns a point at the bin’s stored upper-left grid position. [SAW glossary](https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.0/getting-started/glossary), [STOmics bin coordinates](https://github.com/STOmics/Stereopy/issues/366)

Raw tables retain `uns["stereoseq"]["origin"]` and `["resolution"]`. A common chip origin is not added to only one element. All selected count and analysis products must agree on physical pitch. Selected raw products with conflicting origins fail; corresponding analysis must agree with its GEF. Analysis-only input does not establish a chip origin and preserves its local coordinates. Separately cropped or recentered artifacts require an established common-frame contract rather than an assumed offset.

Centroid mode is the default and never reads unselected borders under any polygon policy. Polygon mode widens signed centroid-relative offsets before addition and removes only paired terminal padding. `invalid_geometries="keep"` preserves buildable invalid topology by default; `"raise"` reports invalid cell IDs, while `"drop"` removes each affected contour and its matching count-table row, layers and annotations. Adjusted and unadjusted populations are filtered independently. Analysis products are validated against the original raw population and retain their independent rows and point targets; masks remain unchanged. Missing/mismatched/interior padding and short contours fail in every policy. A requested all-dropped polygon element fails. No topology repair or extra area threshold is applied. Source area remains an independent source measurement.

## Images and labels

| Role | Element | Names |
| --- | --- | --- |
| `registered` | image | `<stain>_registered`, or `registered` without a stain. |
| `mask` | labels | `<stain>_mask`, or `mask` without a stain. |
| `tissue_cut` | labels | `<stain>_tissue_cut`, historical `tissue_cut`, and `expression_tissue_cut` for `bin1_img_tissue_cut.tif`. |
| `distance_mask` | labels | `<stain>_distance_mask_<distance>`. |
| `expression` | image | `expression` for `bin1_img.tif`. |

Stains come from file names. A stain that makes the element name invalid in SpatialData is sanitized with a warning (`sample_H&E_regist.tif` gives `H_E_registered`); names that then coincide, also ignoring letter case, fail. The expression-distribution TIFF and its derived tissue mask remain distinct products. Each TIFF must contain one supported image series; the reader uses its first full-resolution level and constructs no pyramid unless `scale_factors` requests one. Images have axes `(c, y, x)`, source dtype and source channel order. Grayscale gains one channel. Channels are named `r`, `g`, `b` (and `a`) only when the TIFF declares an RGB colour model for its samples (photometric RGB, with `ExtraSamples` alpha for a fourth sample, or JPEG-compressed YCbCr); other multi-sample or multi-channel images keep positional channel names `0, 1, ...`. Scale factors whose product exceeds the image extent raise `ValueError`. Masks are labels with axes `(y, x)` and their source boolean or integer dtype and values; multichannel or floating-point masks fail. Unsupported non-singleton dimensions fail with a contextual `ReaderFormatError` naming the TIFF.

`scale_factors` and `chunks` follow the shared [raster options](../user/rasters.md): by default a single scale, images in `(1, 1024, 1024)` chunks and labels in `(1024, 1024)` chunks. The same factors apply to every selected image and mask. Generated label levels are downsampled by nearest neighbour, so every level of a mask contains only source mask values.

Raster pixels remain lazy. The TIFF tiles or strips needed by the requested chunks are decoded when computed. A small region can require full-width strips. Keep TIFF files available for computation and writing.

Expression, metadata and coordinates are read eagerly from GEF and H5AD. Memory use grows with the products selected.

## Command line

```bash
python -m spatialdata_io stereoseq --input task --output result.zarr \
    --bin-sizes 50 --cell-bins unadjusted \
    --analysis-files outs/analysis/sample.cellbin_0.5.h5ad \
    --image-roles registered --scale-factors 2 --scale-factors 2
```

Repeat `--bin-sizes`, `--analysis-files`, `--image-roles` and `--scale-factors` for multiple values. Pass `--bin-sizes all` alone to select every stored bin. `--chunks` accepts one value or three CYX values. The CLI otherwise mirrors the Python names, choices and defaults.

Invalid public options raise `TypeError` for a wrong type and `ValueError` for a wrong value. Missing paths and wrong filesystem kinds retain their standard path exceptions. Malformed, ambiguous or scientifically inconsistent artifacts raise a contextual `ReaderFormatError`; selected artifacts are never silently omitted.
