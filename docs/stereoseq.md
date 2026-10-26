# Stereo-seq

{func}`spatialdata_io.stereoseq` reads independently selected transcriptomic products from the documented SAW 7.0/7.1 numbered layouts, SAW 8.0/8.1/8.2 `outs` layouts, the compact numbered workflow, and explicit flat exports. It returns raw GEF counts, derived H5AD analysis and registered images as separate products.

Directory numbering and GEF schema versions do not identify the producing SAW release. The reader records its name and package version in SpatialData provenance; it leaves the producer release unknown. Protein, H5MU/OMNI and microorganism products are outside this reader's supported contract.

When a GEF declares its `omics` attribute, it must be `Transcriptomics`. Historical GEFs without that attribute remain supported through their validated transcriptomic field contract.

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
    chunks=(1, 1024, 1024),
)
```

| Option | Selection |
| --- | --- |
| `bin_sizes=()` | No square counts. Pass distinct positive stored sizes, or `"all"` to read every stored bin in numeric order. |
| `cell_bins="none"` | No raw cells. Choose `"unadjusted"`, `"adjusted"` or `"both"`. |
| `cell_geometry="centroids"` | Return source cell centers as points. Choose `"polygons"` for strictly validated stored contours. |
| `analysis_files=()` | No analysis. Pass exact H5AD paths; clustering resolutions are never all loaded implicitly. |
| `image_roles=()` | No images. Select `"registered"`, `"mask"`, `"tissue_cut"`, `"distance_mask"` or `"expression"`; all matching stains are included. |
| `sample_id=None` | Infer a unique source sample; supply its full identifier when several samples are present. Dots in identifiers are retained. |
| `chunks=None` | Keep native TIFF chunks. A positive integer or three positive CYX sizes requests rechunking. |

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

`adjusted_cell_counts_file` selects the source for adjusted cells. Overrides are also available in structured layouts. They resolve relative to the supplied dataset root; absolute caller paths are permitted. An override does not select a product by itself. Paths discovered from a dataset must remain inside that dataset root.

For qualified square exports such as `sample.HE.gef` or `sample.region.label.gef`, supply both `sample_id="sample"` and the explicit `square_counts_file`. This identifies the sample without truncating sample IDs that contain dots; conflicting GEF sample metadata still fails.

The default square source is the unique tissue/full GEF. Raw whole-chip counts require `square_counts_file="...raw.gef"`. Raw, tissue and full GEFs represent distinct populations. If multiple tissue/full candidates exist, choose one explicitly; the reader does not decide which population you intended.

## GEF fields and exact counts

Feature fields are interpreted by name, independently of their column order. Supported physical profiles are:

| Profile | Square feature fields | Cell feature fields | Feature identity |
| --- | --- | --- | --- |
| `name_only` | `gene`, `offset`, `count` | `geneName`, `offset`, `cellCount` or `cellcount`, `expCount`, `maxMIDcount` | Source `gene` or `geneName`. |
| `id_and_name` | `geneID`, `geneName`, `offset`, `count` | `geneID`, `geneName`, `offset`, `cellCount` or `cellcount`, `expCount`, `maxMIDcount` | Source `geneID`; `geneName` is a display annotation. |

These descriptions are storage profiles, not public selectors or release labels. Feature identities must be unique, nonempty UTF-8 text. Duplicate display names are retained when stable feature IDs distinguish them. Cell-expression feature indices may be uint16 or uint32 independently of the feature profile. Counts accept supported unsigned integer widths; XY coordinates are exact signed or unsigned 32-bit grid values, widened without changing their values.

Raw `X` is an eager canonical `scipy.sparse.csr_array` of uint64 counts. Square records with the same coordinate and feature are summed exactly after promotion; an accumulation exceeding uint64 fails. Repeated feature records within a native cell span are malformed and fail. Square feature spans and native cell-expression spans must cover their source expression records contiguously, without gaps or overlaps.

Aligned square `exon` and cell `cellExpExon` payloads become a separate canonical uint64 `layers["exon"]`, with the same feature/observation order and permitted square aggregation. Per-cell `cellExon`, per-feature `geneExon`, source summary columns and cell-type names are retained when present. The alternate `geneExp` traversal and NNZ-sized internal exon payloads are not duplicated into `uns`.

## Output names, order and linkage

| Selected product | Target | Table |
| --- | --- | --- |
| Stored square bin N | `points["bin<N>_genes"]` | `tables["bin<N>_table"]` |
| Unadjusted cell centers | `points["cells_centroids"]` | `tables["cells_table"]` |
| Adjusted cell centers | `points["adjusted_cells_centroids"]` | `tables["adjusted_cells_table"]` |
| Unadjusted cell contours | `shapes["cells_polygons"]` | `tables["cells_table"]` |
| Adjusted cell contours | `shapes["adjusted_cells_polygons"]` | `tables["adjusted_cells_table"]` |
| Derived analysis | `points["<analysis>_points"]` | `tables["<analysis>_table"]` |

Raw feature order follows the GEF, including zero-occurrence features. Square observations follow the first occurrence of each XY pair in the expression records, including positions with zero totals; their target IDs are consecutive integers starting at zero. Cells follow source cell order and retain the source integer cell IDs, including gaps. AnnData observation names are the corresponding decimal strings. Coordinates are available in XY order in `obsm["spatial"]`; returned points contain native int64 `x` and `y` columns. Cell-type and cluster IDs remain categorical annotations.

Every linked table has categorical `obs["region"]` and an `obs["instance_id"]` that exactly identifies its independent target. A selected table with no observations fails because each declared region must occur in the linked table; no empty region or partial dataset is invented.

## Derived analysis identity

Analysis preserves H5AD values, observation and feature order, additional annotations, embeddings, layers and imported `raw`. It does not replace raw GEF counts or rerun vendor analysis. Returned `X` is a canonical CSR sparse array with its numerical dtype retained. Documented expression layers `counts`, `log1p` and `exon` are normalized to sparse arrays; other layers retain their imported representation. Imported `raw` remains the source analysis snapshot rather than the reader's canonical raw GEF product.

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

When corresponding raw GEF counts are selected, every analysis coordinate and feature must belong to that source. Optional cell IDs must identify the same source cells at those coordinates. Analysis point IDs are its preserved observation names. A filtered or reordered analysis is valid and keeps its own order and point target; rows and features are never joined by position or silently intersected. Analysis-only input requires physical pitch metadata in `uns["resolution"]`; a matching selected GEF may provide it otherwise.

The analysis base name is `cells_analysis`, `adjusted_cells_analysis` or `bin<N>_analysis`. A filename clustering resolution appends an exact normalized decimal suffix: `0.5` and `0.50` both produce `_r0_5`, so selecting both for one population is a collision and fails. Very large exponents use a bounded scientific spelling. Historical filenames without clustering resolution omit that suffix. This clustering value is separate from the physical pitch in `uns["resolution"]`.

## Coordinates and geometry

Every target and image maps by identity to the coordinate system `global`, in its matching normalized expression grid. Registered images are already resampled to that grid. Square coordinates are used as stored, without multiplication by bin size. Physical spot pitch is metadata in nanometers, not a square side length or an applied transformation.

Raw tables retain `uns["stereoseq"]["origin"]` and `["resolution"]`. A common chip origin is not added to only one element. All selected count and analysis products must agree on physical pitch. Selected raw products with conflicting origins fail; corresponding analysis must agree with its GEF. Analysis-only input does not establish a chip origin and preserves its local coordinates. Separately cropped or recentered artifacts require an established common-frame contract rather than an assumed offset.

Centroid mode is the default and never reads or validates unselected borders. Polygon mode adds signed centroid-relative offsets after widening, removes only paired terminal padding, and requires simple nondegenerate contours. Mismatched/interior padding, invalid rings and self-intersections fail without repair, omitted cells or fallback. Some vendor cell contours are nonsimple even with the documented decoding; choose centroid mode explicitly when their exact counts and centers are the desired product. Source area remains a measurement, independent of the polygon's geometric area.

## Images, execution and ownership

| Role | Image names |
| --- | --- |
| `registered` | `<stain>_registered`, or `registered` without a stain. |
| `mask` | `<stain>_mask`, or `mask` without a stain. |
| `tissue_cut` | `<stain>_tissue_cut`, historical `tissue_cut`, and `expression_tissue_cut` for `bin1_img_tissue_cut.tif`. |
| `distance_mask` | `<stain>_distance_mask_<distance>`. |
| `expression` | `expression` for `bin1_img.tif`. |

Masks are images, not instance labels. The expression-distribution TIFF and its derived tissue mask remain distinct products. Each TIFF must contain one supported image series; the reader uses its first full-resolution level and constructs no automatic pyramid. Images have axes `(c, y, x)`, source dtype and source channel order. Grayscale gains one channel; RGB(A) TIFF samples receive `r`, `g`, `b`, `a` channel names. Unsupported non-singleton dimensions fail.

Raster pixels remain lazy and constructing the reader result executes zero pixel tasks. Native TIFF tiles/strips determine initial chunks; requested `chunks` applies rechunking afterward. Partial computation reads the selected chunks, subject to the source codec's decoding granularity. Source paths must remain available for later image computation. Lazy graphs own no open HDF5/TIFF handles.

GEF expression and metadata are eager. Expression/exon input slices share an 8 MiB record budget, but eager output, numerical buffers and coordinate indexing use memory proportional to the selected records and targets. This is a bound on input slices, not on total process memory. Returned point graphs wrap already materialized count or analysis coordinates; they are not lazy GEF/H5AD scans. Files close on success and failure, and no caller-owned scientific container is mutated.

## Command line and migration

```bash
spatialdata_io stereoseq --input task --output result.zarr \
    --bin-sizes 50 --cell-bins unadjusted \
    --analysis-files outs/analysis/sample.cellbin_0.5.h5ad \
    --image-roles registered --chunks 1 --chunks 1024 --chunks 1024
```

Repeat `--bin-sizes`, `--analysis-files` and `--image-roles` for multiple values. Pass `--bin-sizes all` alone to select every stored bin. `--chunks` accepts one value or three CYX values. The CLI otherwise mirrors the Python names, choices and defaults.

The former `dataset_id`, `read_square_bin`, `optional_tif`, `imread_kwargs` and `image_models_kwargs` parameters have been removed. Use `sample_id`, explicit product selections and `chunks`. Raw GEF counts and analysis tables now remain separate; square features follow source order, raw counts use uint64 sparse arrays, and cell targets use source centroids or validated polygons instead of an additional circle representation.

Invalid public options raise `TypeError` or `ValueError`. Missing paths and wrong filesystem kinds retain their standard path exceptions. Malformed, ambiguous or scientifically inconsistent artifacts raise a contextual `ReaderFormatError`; selected artifacts are never silently omitted.

## Reproduce the real-data tests

The optional test-data manifest prioritizes original STOmics examples:

| Key | Producer attribution and exercised product |
| --- | --- |
| `stereoseq_mouse_eyeball` | Vendor SAW 8.1: C04687E314 unadjusted cells, adjusted cells, whole-chip raw square GEF and matching registered H&E image. The square GEF retains stable feature IDs and display names. |
| `stereoseq_mouse_kidney_saw7` | Vendor SAW 7.0: complete C02533C1 unadjusted cells and original tissue square GEF with name-only features. |
| `stereoseq_saw7_1_square` | Study SAW 7.1.2: A03994E6 Stereo-CITE RNA tissue GEF, CC BY 4.0. |
| `stereoseq_saw6_square` | Study SAW 6: complete A01785D6 tissue square GEF, CC BY 4.0. |

Vendor patch releases are unknown. These are original flat downloads, not
complete archived pipeline directory trees. Synthetic tests separately cover
the documented supported directory contracts. No release-verified SAW 8.0 or
8.2 original producer artifact is included. Public vendor downloads do not
establish an open redistribution or repacking grant; downloaded and temporary
reduced files stay outside Git.

From a repository checkout, download the registered group with checksum verification and run its data tests:

```bash
uv run python scripts/test_data_downloader --group stereoseq
uv run pytest tests/integration/readers/stereoseq -m data --no-cov
```

Use `--dataset <key>` to select individual sources. Downloads go below `data/`
by default. The group contains approximately 3.2 GB of original assets. Normal
synthetic tests run offline without these optional files. See the
[test-data downloader instructions](https://github.com/scverse/spatialdata-io/tree/main/scripts/test_data_downloader)
for an alternate data directory and replacement options.

Complete cell tests compare every count, exon, identifier, feature and coordinate
with the original source and verify target membership. The complete SAW 6 square
test requires several GiB of memory. Larger SAW 7.0, 7.1.2 RNA and 8.1 raw square
tests prepare deterministic temporary subsets containing complete leading
feature spans, at most 24 features and 50,000 expression records. These retain
native field dtypes, aligned exons, origin, pitch and source extents. Independent
source comparisons verify exact sparse counts/exons, first-occurrence coordinate
order and linkage. They establish behavior on those real-derived subsets,
without claiming complete-reader execution over the larger square sources.

Additional tests copy the downloaded company files into the documented SAW 7
numbered directories and SAW 8 `outs` directories, and compare those results
with explicit flat-input loading. They cover both task-root and direct-`outs`
input, exact sparse buffers, annotations, coordinates, linkage and lazy images.
These are reconstructed directories containing real data; the downloads do
not supply an intact archived pipeline tree or native analysis H5ADs.

Stereo-CITE tests cover RNA only. An offline metadata-derived synthetic test
also verifies that the observed protein header's one-element `Proteomics`
attribute fails before expression materialization; the complete protein file
is not downloaded. Historical files without `omics` can share the same physical
field contract across modalities, so explicitly select the known RNA source.
The reader does not pair RNA and protein tables.
