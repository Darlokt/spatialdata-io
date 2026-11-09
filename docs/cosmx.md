# CosMx

{func}`spatialdata_io.cosmx` reads a CosMx Spatial Molecular Imager flat-file export. It returns one count table, the FOV composite images and cell labels, and one lazy transcript points element. The 2021 flat-file release and AtoMx flat-file exports are recognized by their columns; no version label is inferred.

## Export layout

```text
export/
├── <prefix>_exprMat_file.csv
├── <prefix>_metadata_file.csv
├── <prefix>_fov_positions_file.csv      # optional
├── <prefix>_tx_file.csv                 # required with include_transcripts=True
├── CellComposite/<any>_F<fov>.jpg       # optional; .jpeg, .png, .tif or .tiff
└── CellLabels/<any>_F<fov>.tif          # optional; .tiff
```

Every flat file may instead be gzip-compressed (`.csv.gz`); one artifact cannot exist in both spellings. FOV numbers come from the `_F<number>` raster suffix and are ordered numerically. Polygon, overlay, compartment and morphology files are not read.

## File prefix and element names

`file_prefix` selects the flat files. It defaults to the prefix of the only `*_exprMat_file` file, so a directory holding several exports needs it. A prefix may contain any character except path separators and NUL, including spaces, as some vendor exports do.

`dataset_id` only names the transcripts element `<dataset_id>_transcripts`; image, label and table names do not use it. An explicit value must be a valid SpatialData element name and is checked before any file is read. By default it is the file prefix, with every character SpatialData does not accept replaced by `_` and one warning:

```python
from spatialdata_io import cosmx

# Files "Lung 5_exprMat_file.csv", ...: warns and names the points "Lung_5_transcripts".
sdata = cosmx("export")

# Select one of several exports and choose the name explicitly:
sdata = cosmx("export", file_prefix="Lung 5", dataset_id="lung5")  # "lung5_transcripts"
```

## Table

The count matrix needs `fov` and `cell_ID` columns. Every other column is an expression feature in source order, including negative probes and system controls, so source spellings such as `HLA-A` are kept. Counts become eager canonical CSR arrays, float32 when every completed value is exact and otherwise float64. Values that neither floating dtype preserves raise a precision error. Exact integer decoding and background-row validation precede conversion; identifiers retain their independent integer dtypes. Every integer field (`fov`, `cell_ID`, counts, transcript `z`) accepts any exact decimal spelling such as `2`, `2.0` or `2e0` and must fit int64; fractions and hexadecimal such as `0x10` fail. Coordinates and numeric annotations are correctly rounded float64 values; a nonzero value that rounds to zero, or an integral value of at least 2⁵³ that float64 cannot hold exactly, fails. The rows with `cell_ID` 0 are per-FOV background and are removed. The remaining `(fov, cell_ID)` population must equal the metadata population exactly; a mismatch fails rather than being intersected.

Table rows follow metadata order. Observation names are the vendor `cell` IDs when the metadata provides them (AtoMx) and row positions `"0"`, `"1"`, ... otherwise. `fov` and `cell_ID` are int64. `Area`, `Area.um2`, `AspectRatio`, `Width`, `Height` and every `Mean.*`/`Max.*` intensity are float64, with empty fields and `NA` as missing values. All other metadata columns stay exact text. AtoMx's `cell_id` column is an exact copy of `cell` and is dropped. Other column names that a SpatialData table does not accept are repaired with one warning listing every change: invalid characters become `_` (`Mean.CD45 RO` becomes `Mean.CD45_RO`), and a name equal to another ignoring letter case, including the reader's `fov`, `cell_ID`, `cell`, `region` and `spatial_pixel_frame`, gets a `_1`, `_2`, ... suffix (`Region` becomes `Region_1`). Exact source `region` and `spatial_pixel_frame` columns are rejected.

Cell positions:

| Key | Content | Unit | Frame |
| --- | --- | --- | --- |
| `obsm["spatial_pixels"]` | local cell centres | pixel | the cell's FOV frame `fov_<fov>_pixels`, per row in categorical `obs["spatial_pixel_frame"]` (categories in FOV order) |
| `obsm["spatial_global_pixels"]` | global cell centres | pixel | `global_pixels` |
| `obsm["spatial"]` (only with `pixel_size_um`) | global cell centres | micrometre | `global` |

`uns["spatialdata_io_positions"]` declares these units, positions and frames.

With a `CellLabels` directory, `region` links every cell to `<fov>_labels` with instance key `cell_ID`, and every FOV with cells must have labels. A present empty directory, including one containing only dotfiles, fails during discovery. Missing labels for any FOV with cells fail before count parsing or raster construction. Cell IDs must fit their own FOV's inspected integer label dtype; this check does not read label pixels or establish full pixel membership. Without that directory the table is returned unlinked. An empty optional `CellComposite` directory is allowed.

## Coordinates

`global` always means micrometres. Without `pixel_size_um` no element is placed in `global`; the export's global pixel frame is `global_pixels` instead.

| Element | Without `pixel_size_um` | With `pixel_size_um` |
| --- | --- | --- |
| `images["<fov>_image"]` | `fov_<fov>_pixels`, `global_pixels` | `fov_<fov>_pixels`, `global` |
| `labels["<fov>_labels"]` | `fov_<fov>_pixels`, `global_pixels` | `fov_<fov>_pixels`, `global` |
| `points["<dataset_id>_transcripts"]` | `global_pixels` (identity), every `fov_<fov>_pixels` | `global` (scale), every `fov_<fov>_pixels` |

The `fov_<fov>_pixels` systems (for example `fov_2_pixels`) are each FOV's local pixel frame. CosMx local coordinates are pixel indices with a reversed vertical axis: the raster pixel at row `r` and column `c` of an `H`-row FOV has local coordinates `(c, H - r)`. This was measured on the public 2021 Lung5_Rep2 release and on an AtoMx mouse-brain export, from label pixels against cell metadata and polygon extremes. Raster pixels are returned exactly as stored; the reflection is part of each raster's transformation.

Each FOV's local-to-global map is a translation per axis, possibly with a reversed axis: `global = ±local + offset`. Exports differ in the sign of the y axis. The sign and offset are derived from every metadata cell and accepted only when one offset lies within the printed rounding interval of all of them; no affine is fitted. A pixel FOV-position file (`fov,x_px,y_px` or `FOV,x_global_px,y_global_px,...`) must agree with the cells and places FOVs without cells; millimetre-only position files are accepted but not used. For a positive axis, the printed position's rounding interval is intersected with the cell-derived offset interval. The printed center is preserved when feasible; otherwise the midpoint of their intersection is used. This selects a feasible placement within source precision and cannot recover an unprinted exact origin. Reflected axes use cell evidence for their offsets. Stage positions alone do not establish stage-to-pixel calibration; no universal micrometre-per-pixel constant or scale fitted from a single FOV is applied. A FOV with rasters but no cells and no pixel position fails. A wholly cell-empty export is unsupported because it cannot establish the dataset's axis signs, even when pixel FOV positions are present.

The raster convention has only been evidenced for exports whose global axes run in the same direction as the local axes. Rasters in an export with a reversed global axis therefore fail, while its table and transcripts are read.

Printed rounding half-widths must remain finite and positive in float64, including for coordinates written as zero. Extreme decimal exponents that overflow or collapse that width to zero fail with the source artifact identified. Placement requires finite offset intervals and finite selected offsets; a nonfinite sign candidate cannot establish a placement. The final raster affine, including pixel-center placement and any caller calibration, must also remain finite. These checks do not enlarge the printed rounding intervals.

`pixel_size_um` supplies a positive finite micrometre-per-global-pixel scale from the caller's associated acquisition/export evidence. It places the table positions, images, labels and lazy point transformations consistently in `global` micrometres. The local FOV frames remain pixels. Scaling follows the complete signed placement and raster-centre mapping. It establishes a local metric, not stage registration. No expression or transcript coordinates are resampled, and morphology summaries retain their source values.

Calibrated cell positions must remain finite, and a nonzero native global coordinate must not become zero through multiplication. Genuine zero coordinates and representable subnormal results retain their signs.

No scale is inferred automatically from headers, stage-only positions, paired pixel/mm fields, image dimensions or instrument guesses. Releases documenting 0.18, 0.12 and 0.12028 µm/pixel remain distinct. The current AtoMx 3.10.5 pancreas cell coordinates violate the supported signed unit-translation contract; supplying a scale does not make that profile supported. Automatic paired-FOV and TIFF optical-metadata profiles require associated source evidence and their own coordinate contracts.

## Transcripts

Transcript points are placed at their global `x_global_px`/`y_global_px` positions; the local coordinates are kept as `x_local` and `y_local`. Targets are categorical with the count features as categories. Annotations are `fov`, `cell_ID` (0 marks unassigned transcripts) and, when present, `z_raw`, `CellComp` and `cell`. `z_raw` is the vendor's slice number in arbitrary units and can be negative; it is not a physical coordinate.

Construction reads the CSV header, with small gzip decoder read-ahead; it does not decode a complete first member to discover boundaries. Each record must occupy one physical line. Plain CSV sources are split into newline-delimited blocks of about 8 MiB.

Compressed `.csv.gz` sources support concatenated, independently compressed members as emitted by AtoMx. Each member must be nonempty and end with a complete newline-terminated record; the first may contain both the header and rows. A single-stream gzip file is also accepted. Lazy tasks read consecutive members from byte zero, using the exact end offset verified by the preceding task. Every member's deflate stream, CRC and size trailer is checked; corrupt members and trailing garbage fail instead of being skipped. Source decoding is sequential, and computing a later partition requires its predecessors. There is no preliminary full-file scan or temporary conversion.

One task owns members beginning in about 8 MiB of compressed data, including the complete final member crossing that range. Decoder buffers use 1 MiB chunks, but complete members accumulate before CSV parsing, and projected Arrow fields and the returned pandas partition require additional storage. The compressed range target does not bound decoded partition size. A single-stream gzip file forms one nonempty partition containing all its rows; use plain CSV for independently readable partitions of a large single stream. Unknown targets and malformed values also fail when their partition is computed.

## Options and execution

`scale_factors` and `chunks` follow the shared [raster options](raster_options.md): by default a single scale, images in `(1, 1024, 1024)` chunks and labels in `(1024, 1024)` chunks. Generated label pyramids require labels of at most 32 bits; 64-bit labels need the default single scale. Interleaved samples (TIFF `S`, PNG and JPEG components) become image channels. Channels are named `r, g, b` (or `r, g, b, a`) only when the file declares RGB(A) samples: TIFF `PhotometricInterpretation` RGB or JPEG-compressed YCbCr, PNG colour types 2, 3 and 6, and three-component JPEG. Other images, including three-sample grayscale TIFFs, get positional channels `0, 1, ...`. JPEG and PNG composites decode their whole frame for any selection. TIFF sources read the native tiles, or whole-row strips grouped into bands with a 16 MiB decoded-byte target, that intersect the selected output chunks. A native strip larger than that target remains a single source chunk. Requested output chunks may span several source tiles or bands, so even a small selection can decode all source chunks needed for its output chunk. Construction executes no raster or transcript tasks.

All options are validated in signature order before any file is read; `path_policy` (see [dataset paths](dataset_paths.md)) is checked last. Errors name the artifact by its role (`count matrix`, `cell metadata`, `FOV positions`, `transcripts`, `composite images`, `cell labels`) and give its path.

## Python and command line

```python
from spatialdata_io import cosmx

# Supply pixel_size_um only from associated export/acquisition evidence.
sdata = cosmx("Lung5_Rep2", chunks=(3, 1024, 1024), pixel_size_um=0.18)
sdata.write("Lung5_Rep2.zarr")
```

```bash
python -m spatialdata_io cosmx --input Lung5_Rep2 --output Lung5_Rep2.zarr --no-include-transcripts
python -m spatialdata_io cosmx --input export --output lung5.zarr --file-prefix "Lung 5" --dataset-id lung5
```

The former `transcripts`, `imread_kwargs` and `image_models_kwargs` arguments have been removed; use `include_transcripts`, `scale_factors` and `chunks`. Compared with earlier releases, the reader:

- defaults to `(1, 1024, 1024)` image and `(1024, 1024)` label chunks instead of storage-aligned chunks, and takes `scale_factors=()` instead of `None` for a single scale;
- places each FOV by an exact translation instead of an affine fitted to filtered cells;
- keeps raster pixels unflipped and applies the measured half-pixel-exact pixel convention;
- returns one global transcript element instead of one element per FOV, without writing temporary files or printing progress;
- rejects mismatched count and metadata populations;
- names the linkage column `region` and uses vendor or positional observation names;
- removes the redundant `global_only_image` and `global_only_labels` coordinate systems;
- selects files with `file_prefix`; `dataset_id` now only names the transcripts element, must be a valid element name, and defaults to the repaired file prefix, so exports whose prefix contains a space can be read;
- names the local FOV coordinate systems `fov_<fov>_pixels` (for example `fov_2_pixels`) instead of the bare FOV number `2`, also in `obs["spatial_pixel_frame"]`;
- places uncalibrated rasters and transcripts in `global_pixels` instead of `global`, and renames the global pixel cell centres from `obsm["global"]` to `obsm["spatial_global_pixels"]`;
- names channels `r, g, b(, a)` only for declared RGB(A) samples, not for any 8-bit image with three or four channels;
- repairs metadata column names SpatialData cannot store, with a warning, instead of failing on case variants;
- names artifacts by role, not file name, in errors;
- accepts exact decimal spellings such as `2.0` in integer fields, rejects hexadecimal such as `0x10` (formerly read as 16), and rejects integral float values of at least 2⁵³ that float64 would round (formerly rounded silently).
