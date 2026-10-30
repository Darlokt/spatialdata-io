# Visium HD

`spatialdata_io.visium_hd()` reads Space Ranger 3.0–4.1 HD outputs. All available
bin sizes are selected by default. Cell and nucleus boundaries are included
when present; nucleus counts are a separate opt-in. Expression remains sparse,
and image pixels are decoded only when computed or written.

## Supported layouts

Space Ranger 3.x produces binned outputs:

```text
outs/
├── hd_feature_slice.h5                 # or feature_slice.h5; optional unless CytAssist requested
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
└── spatial/                           # optional primary image copies
```

Space Ranger 4.x adds segmentation:

```text
outs/
├── feature_slice.h5
├── binned_outputs/                    # as above
├── barcode_mappings.parquet            # alternative: inside segmented_outputs/
├── segmented_outputs/
│   ├── filtered_feature_cell_matrix.h5
│   ├── raw_feature_cell_matrix.h5
│   ├── cell_segmentations.geojson
│   ├── nucleus_segmentations.geojson
│   └── spatial/scalefactors_json.json
└── spatial/                           # optional image copies
```

Stored custom even sizes such as `square_006um/` use the same bin structure.
Loose downloads may prefix `feature_slice.h5`, `hd_feature_slice.h5`,
`barcode_mappings.parquet`, and `tissue_image.{btf,tif,tiff}` with a dataset name.
Discovery uses the root and documented directories, with no recursive fallback
into neighbouring runs. Multiple candidates raise. Relative image overrides
resolve against the dataset root; absolute overrides are allowed.

The 2 µm positions and scale factors calibrate every read, even if only larger
bins or segmentations are returned. Grid dimensions come from feature-slice
metadata when present, otherwise from the position columns, followed by exact
coverage validation. Only selected count populations must be present. Version
and optional library identity come from the selected HDF5 matrix.

A null microscope matrix pair does not prevent loading: the registration is
fitted directly from 2 µm positions. This is tested synthetically; no real
CytAssist-only producer output was available for validation. CytAssist placement
requires the feature slice and its non-null CytAssist matrix pair. Analysis,
probe-level matrices, molecule information and 4.1 cell-type annotations are
outside this reader's contract.

## Products and counts

```python
from spatialdata_io import visium_hd

# Compact exact grid representation; default otherwise builds every bin polygon.
sdata = visium_hd("outs", bin_geometry="labels")

# Cells and exact derived nucleus counts, without bin elements or images.
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
`feature_types`, and `genome`. Duplicate symbols are made unique consistently
unless `make_var_names_unique=False`.

10x supplies cell matrices but no nucleus matrix. The nucleus table sums **raw
2 µm counts** over bins with `in_nucleus=True`, including bins outside the tissue
mask, in the selected cell population's order. Bin and cell count dtypes follow
Scanpy: vendor int32 sources become float32, while alternate numeric source
dtypes are retained. Counts must be finite, nonnegative and integer-valued. Nucleus
aggregation uses int64, rejecting unsafe integer conversion or possible sum
overflow; final counts use float32, whose precision applies to large values.
Every returned expression matrix is a canonical SciPy CSR sparse **array**.
Nothing is densified. Linkage is performed once after all filtering, using
independent shape or label IDs.

`invalid_geometries="keep"` preserves source topology. `"drop"` removes an invalid
returned cell polygon with its cell table row and associated nucleus; an invalid
nucleus removes only its nucleus polygon and nucleus table row. `"raise"` rejects
invalid geometry. Missing or malformed geometries, holes, MultiPolygons, duplicate
IDs and nonfinite coordinates raise in every mode. GeoJSON properties are
projected during ingestion; the returned geometry vector is eager.

## Physical coordinates

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
at an unchanged maximum error of 1e-9 pixels. For pre-4.1 boundaries without
vendor centroids no centroid is invented.

Microscope pixels retain their source values and carry a least-squares affine
into the physical frame. Projective registration makes this approximate:
`attrs["visium_hd"]["image_affine_max_residual_um"]` and
`image_affine_rms_residual_um` report its error over the complete capture grid.
They do not bound extrapolation beyond that grid. `microns_per_pixel` is retained
as nominal vendor metadata, not used as a replacement for the physical ruler.
Hires and lowres transforms compose the affine with the supplied inverse scale
factor; padded short dimensions do not change that scale.

## Images, ownership and memory

Full-resolution images are discovered among root `tissue_image` TIFF names and
supported files directly inside `microscope_image/`; more than one is ambiguous.
`fullres_image_file` overrides this selection. TIFF content determines TIFF vs
BigTIFF, including `.btf`; `data_axes` and `tiff_series` resolve explicit source
selection. The full-resolution/hires longest-side dimensions are checked against
the vendor scale factor.

All images use `(c, y, x)`, preserve source channel order and dtype, and use the
shared [raster options](raster_options.md). `scale_factors` applies only to the
full-resolution image. `chunks` controls image chunks and the spatial chunks of
labels. TIFF tasks decode overlapping tiles/strips; PNG/JPEG may decode the
whole encoded frame. Returned lazy sources must remain at their source paths.
The reader never mutates caller containers or process-global decoder settings.

`include_cytassist_image=True` uses a lazy projective warp into a regular physical
grid, with bounded source windows and bilinear SciPy interpolation. It preserves
the source dtype (integer interpolation therefore rounds to that dtype). No image
pixel tasks execute when the reader returns.

All-bin squares can require several GiB: the default includes every available
size, and a full 6.5 mm grid has 11.2 million 2 µm bins. Labels require about 45 MB
for that uint32 grid, plus model, graph, table and serialization overhead. The
11 mm grid has 30.5 million bins. Polygon construction batches limit temporary
vertex buffers, but the returned geometries themselves are eager. Nucleus counts
use the shared Scanpy source and can require at least 4.4 GiB even on the tiny
example because the raw matrix includes every barcode. This cost is not hidden
behind a filtered-count approximation.

## CLI and migration

```bash
spatialdata-io visium-hd --input outs --output slide.zarr \
  --bin-sizes 8 --bin-geometry labels --no-include-images
```

Repeat `--bin-sizes` for several sizes; `all` is the default and `none` skips
bins. Inclusion flags use `--include-…` / `--no-include-…`. Reader and CLI names,
choices and defaults agree.

| Removed option | Replacement |
| --- | --- |
| `bin_size` | `bin_sizes` |
| `load_segmentations_only` | `bin_sizes=()` and explicit boundary selection |
| `load_nucleus_segmentations` | `include_nucleus_boundaries`, `include_nucleus_table` |
| `filtered_counts_file` | `bin_counts_kind`, `cell_counts_kind` |
| `bins_as_squares`, `annotate_table_by_labels` | `bin_geometry` |
| `load_all_images` | `include_images`, `include_cytassist_image` |
| `var_names_make_unique` | `make_var_names_unique` |
| Generic decoder/model/count mappings | Explicit raster and count options |
| `dataset_id` | Fixed element names; concatenate by sample |
| `gex_only` | All source features retained |

For multiple samples, use `spatialdata.concatenate({"sample_a": a, "sample_b": b})`
to suffix element names and update table regions and observation names together.
Separate coordinate systems are needed if samples are not physically aligned.

## Format errors

```{eval-rst}
.. autoclass:: spatialdata_io.readers._utils.errors.ReaderErrorContext

.. autoexception:: spatialdata_io.readers._utils.errors.ReaderFormatError
```
