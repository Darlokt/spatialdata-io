# Visium HD

`spatialdata_io.visium_hd()` reads Space Ranger HD outputs (tested with 3.0–4.1). All available
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
resolve against the dataset root; absolute overrides are allowed.

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
`feature_types`, and `genome`. Duplicate symbols are made unique consistently
unless `make_var_names_unique=False`.

10x supplies cell matrices but no nucleus matrix. The nucleus table sums **raw
2 µm counts** over bins with `in_nucleus=True`, including bins outside the tissue
mask, in the selected cell population's order. Returned counts use float32
when every completed value is exact, otherwise float64. Values that neither
type preserves raise. Counts must be finite, nonnegative and integer-valued.
Nucleus reconstruction accumulates decoded bins without converting them to
integers. Before addition, a Python-integer bound (maximum decoded count times
maximum contributing bins) selects float32 arithmetic when at most 2**24,
otherwise float64 when at most 2**53. The operands select that arithmetic
dtype before multiplication, proving every nonnegative integer partial sum
exact. Failure raises a precision error; this conservative bound can reject
exceptional inputs whose actual sums would still be safe.
Completed nucleus values return to float32 only when exact. Source integers
already rounded by Scanpy cannot be recovered.
Every returned expression matrix is a canonical SciPy CSR sparse **array**.
Nothing is densified. Linkage is performed once after all filtering, using
the actual returned shape indexes or derived grid-label IDs. Source count
validation inherits Scanpy conversion and duplicate handling; the reader does
not reject valid duplicate count entries or rescan counts before upstream
conversion. See [table numerical limits](table_contracts.md#persistence-ownership-and-numerical-limits).

`invalid_geometries="keep"` preserves source topology. `"drop"` removes an invalid
returned cell polygon with its cell table row and associated nucleus; an invalid
nucleus removes only its nucleus polygon and nucleus table row. `"raise"` rejects
invalid geometry. Missing or malformed geometries, holes, MultiPolygons, duplicate
IDs and nonfinite coordinates raise in every mode. GeoJSON properties are
projected during ingestion; the returned geometry vector is eager.

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



Each final table describes float64 XY micrometres in `uns["spatialdata_io_positions"]`, distinguishing nominal bin centres, vendor cell centroids and vendor nucleus centroids in the calibrated `global` slide frame. Missing centroid properties yield no canonical array or declaration for that product. Nucleus tables never inherit cell positions; polygon centroids are not substituted.

## Images, ownership and memory

Full-resolution images are discovered among root `tissue_image` TIFF names and
supported files directly inside `microscope_image/`; more than one is ambiguous.
`fullres_image_file` overrides this selection. A file whose first bytes are a
TIFF or BigTIFF signature is read as TIFF whatever its suffix (including
`.btf`); otherwise `.png`, `.jpg` and `.jpeg` select those decoders. A TIFF
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
shared [raster options](raster_options.md). `scale_factors` applies only to the
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
source channel coordinates. No image pixel tasks execute when the reader returns.

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
bins. Inclusion flags use `--include-…` / `--no-include-…`. `--data-axes` takes
one string such as `CYX`. Reader and CLI names, choices, defaults and option
order agree.

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

Changes to the reader's contract:

| Change | Migration |
| --- | --- |
| `data_axes` is one string such as `"CYX"` (`--data-axes CYX`); a sequence raises `TypeError` | Join the axis letters |
| Channels are `r, g, b(, a)` only for declared RGB, else OME names or positions; earlier releases labelled any 3- or 4-channel image RGB(A) | Select fluorescence channels by OME name or position |
| A multi-series TIFF without `tiff_series` raises `ReaderFormatError` (a `ValueError`) | Pass `tiff_series` |
| Wrongly typed options raise `TypeError`, invalid values `ValueError`, all before any file is read | — |
| `make_var_names_unique` precedes `invalid_geometries` in the signature and the CLI | Keyword arguments are unaffected |
| Invalid UTF-8 in HDF5 text attributes raises; feature-slice `sample_id` is no longer checked | — |
| Requested boundaries without `segmented_outputs/` raise `FileNotFoundError` for that directory | — |
| Any `[spaceranger-]X.Y.Z` release is read (a release other than 3.0, 3.1, 4.0 or 4.1 warns that it is untested); other spellings and a missing `software_version` record no format version instead of raising | — |

Discovered artifacts that are symlinks to targets outside the dataset directory are now rejected by default; earlier releases accepted them. Pass `path_policy="relaxed"` for git-annex or DVC datasets ([Dataset paths and symlinks](dataset_paths.md)).

For multiple samples, use `spatialdata.concatenate({"sample_a": a, "sample_b": b})`
to suffix element names and update table regions and observation names together.
Separate coordinate systems are needed if samples are not physically aligned.

## Format errors

```{eval-rst}
.. autoclass:: spatialdata_io.readers._utils.errors.ReaderErrorContext

.. autoexception:: spatialdata_io.readers._utils.errors.ReaderFormatError
```
