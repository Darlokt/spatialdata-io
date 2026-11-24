# Visium HD

{func}`spatialdata_io.visium_hd` reads Space Ranger HD outputs (tested with 3.0–4.1). All available
bin sizes are selected by default. Cell and nucleus boundaries are included
when present; nucleus counts are a separate opt-in. Expression remains sparse,
and image pixels are decoded only when computed or written.

## Supported layouts

Space Ranger 3.x produces binned outputs:

```text
outs/
├── hd_feature_slice.h5                 # or feature_slice.h5; required for CytAssist or no selected counts
├── binned_outputs/
│   ├── square_002um/
│   │   ├── filtered_feature_bc_matrix.h5
│   │   ├── raw_feature_bc_matrix.h5
│   │   └── spatial/
│   │       ├── tissue_positions.parquet
│   │       ├── scalefactors_json.json
│   │       ├── tissue_hires_image.png   # optional
│   │       ├── tissue_lowres_image.png  # optional
│   │       └── cytassist_image.tiff     # optional
│   ├── square_008um/                   # same structure
│   └── square_016um/                   # same structure
└── spatial/                           # optional; each image takes precedence over its 2 µm copy
```

Space Ranger 4.x adds segmentation:

```text
outs/
├── feature_slice.h5                    # required for CytAssist or no selected counts; otherwise optional
├── binned_outputs/                    # as above
├── barcode_mappings.parquet            # alternative: inside segmented_outputs/
├── segmented_outputs/
│   ├── filtered_feature_cell_matrix.h5
│   ├── raw_feature_cell_matrix.h5
│   ├── cell_segmentations.geojson
│   ├── nucleus_segmentations.geojson
│   └── spatial/scalefactors_json.json
└── spatial/                           # optional; each image takes precedence over its 2 µm copy
```

Stored custom even sizes such as `square_006um/` use the same bin structure.
Loose downloads may prefix `feature_slice.h5`, `hd_feature_slice.h5`,
`barcode_mappings.parquet`, and `tissue_image.{btf,tif,tiff}` with a dataset name.
Discovery uses the root and documented directories, with no recursive fallback
into neighbouring runs. Multiple candidates raise. Relative image overrides
resolve against the dataset root; absolute overrides are allowed. Discovered
files use the [dataset path policy](../user/paths.md).

The 2 µm positions and scale factors calibrate every read, even if only larger
bins or segmentations are returned. Grid dimensions come from feature-slice
metadata when present, otherwise from the position columns, followed by exact
coverage validation. Only selected count populations must be present.

Identity is reconciled during discovery, before calibration, from every
distinct selected bin and cell matrix, the raw 2 µm matrix needed for nucleus
counts, and any feature slice whose metadata is consumed. A source's
`software_version` attribute spelled `spaceranger-X.Y.Z` or `X.Y.Z` gives the
detected format version `X.Y.Z`, recorded in provenance and in every later
error. Other spellings (for example `spaceranger-4.0.1-rc1` or `4.1`) and a
missing attribute give no version, as in the classic Visium reader. The layout
is recognized from the artifacts present (`hd_feature_slice.h5`,
`segmented_outputs/`, centroid properties), not from the version, so any
release is read. The reader is tested with Space Ranger 3.0, 3.1, 4.0 and 4.1;
another release reads with one `UserWarning` that it is untested.

The detected **full versions must match exactly**; sources reporting different
versions come from different runs and raise. A source without a detected
version does not take part: the recorded version is the one the other sources
agree on, or none if no source has one. Multiple distinct known `library_ids`
also raise; the sole known ID is retained regardless of selection order.
Missing optional library IDs are allowed, and all-missing IDs yield no library
ID. Agreement of missing IDs does not establish a common export origin.
HDF5 text attributes must be nonempty UTF-8, whether stored as fixed- or
variable-length strings; invalid bytes raise.

With no count product selected, an identity-bearing feature slice is required,
including images-only and nucleus-boundary-only reads. Unselected raw or
filtered matrices are never used as identity fallbacks. A selected count
workflow can omit the feature slice unless CytAssist is requested.
Feature-slice metadata validates the outer JSON and the consumed grid and
transform fields (`nrows`, `ncols`, `spot_pitch`, `transform_matrices`); other
keys such as `sample_id` and `hd_layout_json` are ignored.

A null microscope matrix pair is supported by fitting registration directly
from 2 µm positions. CytAssist placement
requires the feature slice and its non-null CytAssist matrix pair. Analysis,
probe-level matrices, molecule information and 4.1 cell-type annotations are
outside this reader's contract.

## Products and counts

```python
from spatialdata_io import visium_hd

# Compact exact grid representation; default otherwise builds every bin polygon.
sdata = visium_hd("outs", bin_geometry="labels")

# Cells and derived nucleus counts, without bin elements or images.
sdata = visium_hd(
    "outs",
    bin_sizes=(),
    include_cell_boundaries=True,
    include_nucleus_boundaries=True,
    include_nucleus_table=True,
    cell_counts_kind="raw",
    include_images=False,
)
```

| Selection | Element | Table |
| --- | --- | --- |
| Each stored selected size | `square_XXXum` shapes or labels | `square_XXXum_table` |
| Cell boundaries | `cell_boundaries` | `cell_table` |
| Nucleus boundaries | `nucleus_boundaries` | `nucleus_table` only when requested |
| Available microscope images | `full_image`, `hires_image`, `lowres_image` | — |
| Explicit CytAssist image | `cytassist_image` | — |

`include_cell_boundaries` and `include_nucleus_boundaries` accept `None` (read
when present), `True` (require), or `False` (skip). Missing explicitly requested
sizes or products raise. Selecting no available product raises. Empty selected
populations cannot form a linked SpatialData table and raise a contextual error.

`bin_counts_kind` and `cell_counts_kind` independently select `"filtered"` or
`"raw"`. The original observation and feature order is preserved, including
spatially unsorted filtered barcodes. Feature annotations retain `gene_ids`,
`feature_types` and other available fields such as `genome`. Duplicate symbols are made unique consistently
unless `make_var_names_unique=False`.

Cell and nucleus polygon indexes and their linked table `instance_id` columns
use uint32 IDs from 1 through 999999999. These equal the positive nine-digit
ID in the canonical `cellid_<id:09d>-1` barcodes; source order is preserved.
Zero, alternate barcode spellings, duplicate polygon IDs and IDs outside that
range raise before conversion.

Bin table `obs` contains uint8 `in_tissue`, uint32 `array_row` and
`array_col`, float64 `pxl_row_in_fullres` and `pxl_col_in_fullres`, uint32
`instance_id` and categorical `region`. Cell and nucleus tables have uint32
`instance_id` and categorical `region`. Concatenation with tables using int64
instance identifiers retains an integer dtype.

10x supplies cell matrices but no nucleus matrix. The nucleus table sums **raw
2 µm counts** over bins with `in_nucleus=True`, including bins outside the tissue
mask, in the selected cell population's order. Returned counts follow the
[completed-count dtype policy](../user/tables.md#raw-count-precision).
Both matrices must provide `gene_ids`, `genome`, and `feature_types` in their
feature annotations. A missing column raises an error that identifies its
matrix. The feature identity values must match in source order.
Duplicate HDF5 entries for one barcode and feature fail. Counts must be
finite, nonnegative and integer-valued.
Nucleus reconstruction accumulates decoded bins without converting them to
integers. Before addition, the precision bound (maximum decoded count times
maximum contributing bins) selects float32 arithmetic when at most 2**24,
otherwise float64 when at most 2**53. This keeps partial sums exact. Exceeding the bound raises a precision error
even if the completed sums would fit.
Completed nucleus values use float32 through 2²⁴ and float64 above that threshold.
Linkage uses the returned shape indexes or derived grid-label IDs after filtering.

The raw 2 µm matrix and its barcode annotations stay in memory until aggregation
finishes. Every raw barcode must occur once in the mapping. Reordered mapping rows are
matched by barcode. Missing cell assignments are allowed. Unassigned bins and bins assigned to cells outside
the selected population contribute to no returned nucleus.
`invalid_geometries="keep"` preserves source topology. `"drop"` removes an invalid
returned cell polygon with its cell table row and associated nucleus; an invalid
nucleus removes only its nucleus polygon and nucleus table row. `"raise"` rejects
invalid geometry. Missing or malformed geometries, holes, MultiPolygons, duplicate
IDs and nonfinite coordinates raise in every mode. Returned geometries are eager.

## Physical coordinates

Every element is placed only in `global`; there is no pixel coordinate system.
The physical calibration is verifiable: it comes from the manufactured 2 µm
barcode grid, whose position in the image every 2 µm position row fixes, not
from Space Ranger's estimated `microns_per_pixel`.

`global` uses micrometres on the manufactured 2 µm grid, rigidly oriented like
the microscope image. Its origin is the fitted affine position of full-resolution pixel `(0, 0)`;
the exact projective position can differ by the reported registration residual.
The polar decomposition of the fitted image-to-slide affine determines the
orientation, including reflection. Bin squares, segmentation vertices and
`obsm["spatial"]` are stored in this frame; vectors carry identity transforms.
The bin source `pxl_col_in_fullres` and `pxl_row_in_fullres` columns remain in
`obs` for traceability.

Every selected positions file must agree with the 2 µm registration, including
rows absent from filtered counts. For bin factor `k = size_um / 2`, Space Ranger
writes the source pixel position as `H(k * col, k * row) + offset`, where
`offset = (H(k - 1, k - 1) - H(0, 0)) / 2` and `H` is the 2 µm homography.
The offset also applies to partial edge bins. This is the producer's image-space
convention, which differs from projecting each nominal centre when registration
is projective. The reader checks it with the same maximum 1e-9 pixel residual
used to validate the base registration; inconsistent selected-bin artifacts raise.
See the [Space Ranger bin-position implementation](https://github.com/10XGenomics/spaceranger/blob/0bc95c49e1c4db6247a97ed57998f7fe3719e2fa/lib/rust/spatial/src/stages/bin_count_matrix.rs#L99-L124).
The source pixel columns are preserved; physical geometry uses nominal grid
centres and extents.

Squares have the exact nominal physical side length, including the last partial
row or column. They can extend beyond the captured barcodes (for example 4 µm
at the 16 µm edge of a 3350-square grid). Labels encode the same nominal squares:
zero is background and table-order instance IDs start at one. Their uint32 grids
carry exact scale and rigid orientation transforms, without a half-bin shift.

Segmentation vertices and vendor centroids are reprojected through the inverse
homography fitted from 2 µm positions. The fit is checked against every position
at a maximum error of 1e-9 pixels. Pre-4.1 boundaries can lack vendor centroids.

Microscope pixels retain their source values and carry a least-squares affine
into the physical frame. Projective registration makes this approximate:
`attrs["visium_hd"]["image_affine_max_residual_um"]` and
`image_affine_rms_residual_um` report its error over the complete capture grid.
The residuals describe the capture grid. To retrieve the nominal size of a
full-resolution microscopy pixel, use `sdata.attrs["visium_hd"]["microns_per_pixel"]`.
For positions and measurements in the calibrated slide frame, use the image's
transformation to `global`.
Hires and lowres transforms compose the affine with the supplied inverse scale
factor; padded short dimensions do not change that scale.

Each final table describes float64 XY micrometres in `uns["spatialdata_io_positions"]`, distinguishing nominal bin centres, vendor cell centroids and vendor nucleus centroids in the calibrated `global` slide frame. Missing centroid properties yield no canonical array or declaration for that product. Nucleus tables never inherit cell positions; polygon centroids are not substituted.

### Query bins with a QuPath annotation

For polygons exported in full-resolution pixels of the same microscope image as `full_image`, apply that image's transform to the annotation. This example selects every returned 8 µm bin whose square intersects an annotation, including edge touches. Each selected bin contributes its complete expression row once, even where annotations overlap.

```python
from spatialdata import transform
from spatialdata.transformations import get_transformation, set_transformation

from spatialdata_io import geojson, visium_hd

sdata = visium_hd("outs", bin_sizes=(8,), include_cell_boundaries=False, include_nucleus_boundaries=False)
annotations = geojson("annotations.geojson", coordinate_system="microscope_pixels", invalid_geometries="raise")
image_to_slide = get_transformation(sdata.images["full_image"], to_coordinate_system="global")
set_transformation(annotations, image_to_slide, to_coordinate_system="global")
roi = transform(annotations, to_coordinate_system="global").geometry.union_all()

bins = sdata.shapes["square_008um"]
hits = bins.sindex.query(roi, predicate="intersects")
table = sdata.tables["square_008um_table"]
selected = table[table.obs["instance_id"].isin(bins.index.take(hits)), :]
max_residual_um = sdata.attrs["visium_hd"]["image_affine_max_residual_um"]
```

`selected.X` contains per-bin expression in the table's original row and feature order. For a hires or lowres annotation, use that image's transform. For a crop, include its offset and any change in pixel size. With projective registration, `max_residual_um` gives the maximum affine-fit error over the capture grid. Compare it with the precision you need near the annotation boundary.

## Images and memory

Full-resolution images are discovered among root `tissue_image` TIFF names and
supported files directly inside `microscope_image/`; more than one is ambiguous.
`fullres_image_file` overrides this selection. A file whose first bytes are a
TIFF or BigTIFF signature is read as TIFF whatever its suffix (including
`.btf`). For PNG and JPEG, the suffix of the discovered or supplied name
selects the decoder, including when that name is a symlink. A TIFF
must have one series unless `tiff_series` selects one; several series without
a selection raise a contextual `ReaderFormatError`. `data_axes` overrides the
TIFF's declared axes as one case-insensitive string such as `"CYX"`;
`data_axes` and `tiff_series` raise for a PNG or JPEG image. Declared axes
that cannot become `c, y, x` raise a contextual `ReaderFormatError`, and
`data_axes` that do not fit the file raise a `ValueError` naming the image. The
full-resolution/hires longest-side dimensions are checked against the vendor
scale factor.

Hires, lowres and requested CytAssist images are resolved independently. For
each artifact, `spatial/` at the dataset root takes precedence over
`binned_outputs/square_002um/spatial/`; an absent root copy falls back to its
2 µm copy. An empty or partially populated root directory therefore preserves
available bin-level images. A malformed higher-priority entry raises instead
of falling back. Hires and lowres are optional; requested CytAssist is required
in one of these two locations.

All images use `(c, y, x)`, preserve source channel order and dtype, and use the
shared [raster options](../user/rasters.md). `scale_factors` applies only to the
full-resolution image and must not reduce it below one pixel. `chunks` controls
image chunks and the spatial chunks of labels. TIFF tasks decode overlapping
tiles/strips; PNG/JPEG may decode the whole encoded frame. Returned lazy
sources must remain at their source paths. The reader never mutates caller
containers or process-global decoder settings.

Channel coordinates follow one rule for every image:

| Source | Channel coordinates |
| --- | --- |
| Samples the file declares RGB: TIFF `PhotometricInterpretation` RGB (alpha through `ExtraSamples`) or JPEG-compressed YCbCr, PNG colour types 2, 3 and 6, three-component JPEG | `r, g, b` or `r, g, b, a` |
| OME-TIFF whose channel axis is the OME channel (or samples) axis, or a single-channel OME-TIFF | OME channel names when all are nonempty and unique, else OME channel IDs; `<name or ID>:sample:<k>` for an OME channel with several samples |
| Anything else, including an axis reinterpreted as channels with `data_axes` | positions `0, 1, ...` |

The channel count and dtype never make an image RGB: a four-channel
fluorescence TIFF keeps its OME names (for example `DAPI, CD3, CD8, PanCK`) or
positions. The Space Ranger PNGs and the CytAssist TIFF are RGB files and keep
`r, g, b`.

`include_cytassist_image=True` uses a lazy projective warp into a regular physical
grid, with bounded source windows and bilinear SciPy interpolation. It preserves
the source dtype (integer interpolation therefore rounds to that dtype) and the
source channel coordinates. Each output tile reads only the source window that
can contribute to its projected footprint. Tiles outside the footprint are zero.
The registered grid must have finite positive pixel dimensions and pitch. No
image pixel tasks execute when the reader returns.

All-bin squares can require several GiB: the default includes every available
size, and a full 6.5 mm grid has 11.2 million 2 µm bins. Labels require about 45 MB
for that uint32 grid, plus model, graph, table and serialization overhead. The
11 mm grid has 30.5 million bins. Polygon construction batches limit temporary
vertex buffers. The returned geometries are eager.

## Command line

```bash
python -m spatialdata_io visium-hd --input outs --output slide.zarr \
  --bin-sizes 8 --bin-geometry labels --no-include-images
```

Repeat `--bin-sizes` for several sizes; `all` is the default and `none` skips
bins. Inclusion flags use `--include-…` / `--no-include-…`. `--data-axes` takes
one string such as `CYX`. Reader and CLI names, choices, defaults and option
order agree.

For multiple samples, use `spatialdata.concatenate({"sample_a": a, "sample_b": b})`
to suffix element names and update table regions and observation names together.
Separate coordinate systems are needed if samples are not physically aligned.

Malformed layouts, inconsistent identity or calibration, and invalid source data raise a contextual `ReaderFormatError` identifying the affected artifact.
