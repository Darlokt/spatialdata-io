# CosMx

{func}`spatialdata_io.cosmx` reads a CosMx Spatial Molecular Imager flat-file export. It returns one count table, the FOV composite images and cell labels, and one lazy transcript points element. The 2021 flat-file release and AtoMx flat-file exports are recognized by their columns; no version label is inferred.

## Export layout

```text
export/
├── <dataset_id>_exprMat_file.csv
├── <dataset_id>_metadata_file.csv
├── <dataset_id>_fov_positions_file.csv      # optional
├── <dataset_id>_tx_file.csv                 # required with include_transcripts=True
├── CellComposite/<prefix>_F<fov>.jpg        # optional; .jpeg, .png, .tif or .tiff
└── CellLabels/<prefix>_F<fov>.tif           # optional; .tiff
```

Every flat file may instead be gzip-compressed (`.csv.gz`); one artifact cannot exist in both spellings. `dataset_id` defaults to the prefix of the single `*_exprMat_file` file. FOV numbers come from the `_F<number>` raster suffix and are ordered numerically. Polygon, overlay, compartment and morphology files are not read.

## Table

The count matrix needs `fov` and `cell_ID` columns. Every other column is an expression feature in source order, including negative probes and system controls, so source spellings such as `HLA-A` are kept. Counts become eager, canonical CSR int64 arrays. The rows with `cell_ID` 0 are per-FOV background and are removed. The remaining `(fov, cell_ID)` population must equal the metadata population exactly; a mismatch fails rather than being intersected.

Table rows follow metadata order. Observation names are the vendor `cell` IDs when the metadata provides them (AtoMx) and row positions `"0"`, `"1"`, ... otherwise. AtoMx's `cell_id` column is an exact copy of `cell` and a case variant of `cell_ID`, so it is dropped; any other case-variant column names fail. `fov` and `cell_ID` are int64. `Area`, `Area.um2`, `AspectRatio`, `Width`, `Height` and every `Mean.*`/`Max.*` intensity are float64, with empty fields and `NA` as missing values. All other metadata columns stay exact text. `obsm["spatial_pixels"]` and `obsm["global"]` preserve the local and global cell centers in pixels. Categorical string `obs["spatial_pixel_frame"]` identifies each local FOV frame. Without calibration, no canonical `obsm["spatial"]` is returned. With explicit `pixel_size_um`, canonical `spatial` is float64 global cell-center XY in micrometres. `uns["spatialdata_io_positions"]` describes the units, position meaning and frames; the preserved native `global` array has the descriptive frame `global_pixels`.

With a `CellLabels` directory, `region` links every cell to `<fov>_labels` with instance key `cell_ID`, and every FOV with cells must have labels. Cell IDs must fit their own FOV's inspected integer label dtype; this check does not read label pixels or establish full pixel membership. Without that directory the table is returned unlinked.

## Coordinates

| Element | Coordinate systems |
| --- | --- |
| `images["<fov>_image"]` | `<fov>` (local pixels) and `global` |
| `labels["<fov>_labels"]` | `<fov>` and `global` |
| `points["<dataset_id>_transcripts"]` | `global` and every `<fov>` |

CosMx local coordinates are pixel indices with a reversed vertical axis: the raster pixel at row `r` and column `c` of an `H`-row FOV has local coordinates `(c, H - r)`. This was measured on the public 2021 Lung5_Rep2 release and on an AtoMx mouse-brain export, from label pixels against cell metadata and polygon extremes. Raster pixels are returned exactly as stored; the reflection is part of each raster's transformation.

Each FOV's local-to-global map is a translation per axis, possibly with a reversed axis: `global = ±local + offset`. Exports differ in the sign of the y axis. The sign and offset are derived from every metadata cell and accepted only when one offset lies within the printed rounding interval of all of them; no affine is fitted. A pixel FOV-position file (`fov,x_px,y_px` or `FOV,x_global_px,y_global_px,...`) must agree with the cells and places FOVs without cells; millimetre-only position files are accepted but not used. Stage positions alone do not establish stage-to-pixel calibration; no universal micrometre-per-pixel constant or scale fitted from a single FOV is applied. A FOV with rasters but no cells and no pixel position fails.

The raster convention has only been evidenced for exports whose global axes run in the same direction as the local axes. Rasters in an export with a reversed global axis therefore fail, while its table and transcripts are read.

`pixel_size_um` supplies a positive finite micrometre-per-global-pixel scale from the caller's associated acquisition/export evidence. It changes the registered `global` frame to micrometres and scales canonical table positions, images, labels and lazy point transformations consistently. The existing local FOV frames remain pixels. Scaling follows the complete signed placement and raster-centre mapping. It establishes a local metric, not stage registration. No expression or transcript coordinates are resampled, and morphology summaries retain their source values.

No scale is inferred automatically from headers, stage-only positions, paired pixel/mm fields, image dimensions or instrument guesses. Releases documenting 0.18, 0.12 and 0.12028 µm/pixel remain distinct. The current AtoMx 3.10.5 pancreas cell coordinates violate the supported signed unit-translation contract; supplying a scale does not make that profile supported. Automatic paired-FOV and TIFF optical-metadata profiles require associated source evidence and their own coordinate contracts.

## Transcripts

Transcript points are placed at their global `x_global_px`/`y_global_px` positions; the local coordinates are kept as `x_local` and `y_local`. Targets are categorical with the count features as categories. Annotations are `fov`, `cell_ID` (0 marks unassigned transcripts) and, when present, `z_raw`, `CellComp` and `cell`. `z_raw` is the vendor's slice number in arbitrary units and can be negative; it is not a physical coordinate.

Construction reads the CSV header, with small gzip decoder read-ahead; it does not decode a complete first member to discover boundaries. Each record must occupy one physical line. Plain CSV sources are split into newline-delimited blocks of about 8 MiB.

Compressed `.csv.gz` sources support concatenated, independently compressed members as emitted by AtoMx. Each member must be nonempty and end with a complete newline-terminated record; the first may contain both the header and rows. A single-stream gzip file is also accepted. Lazy tasks read consecutive members from byte zero, using the exact end offset verified by the preceding task. Every member's deflate stream, CRC and size trailer is checked; corrupt members and trailing garbage fail instead of being skipped. Source decoding is sequential, and computing a later partition requires its predecessors. There is no preliminary full-file scan or temporary conversion.

One task owns members beginning in about 8 MiB of compressed data, including the complete final member crossing that range. Decoder buffers use 1 MiB chunks, but complete members accumulate before CSV parsing, and projected Arrow fields and the returned pandas partition require additional storage. The compressed range target does not bound decoded partition size. A single-stream gzip file forms one nonempty partition containing all its rows; use plain CSV for independently readable partitions of a large single stream. Unknown targets and malformed values also fail when their partition is computed.

## Options and execution

`scale_factors` and `chunks` follow the shared [raster options](raster_options.md): by default a single scale, images in `(1, 1024, 1024)` chunks and labels in `(1024, 1024)` chunks. Generated label pyramids require labels of at most 32 bits; 64-bit labels need the default single scale. JPEG and PNG composites decode their whole frame for any selection; TIFF sources read only the tiles, or strips grouped into bands of at most 16 MiB, that intersect the selected chunks. Construction executes no raster or transcript tasks.

## Python and command line

```python
from spatialdata_io import cosmx

# Supply pixel_size_um only from associated export/acquisition evidence.
sdata = cosmx("Lung5_Rep2", chunks=(3, 1024, 1024), pixel_size_um=0.18)
sdata.write("Lung5_Rep2.zarr")
```

```bash
spatialdata_io cosmx --input Lung5_Rep2 --output Lung5_Rep2.zarr --no-include-transcripts
```

The former `transcripts`, `imread_kwargs` and `image_models_kwargs` arguments have been removed; use `include_transcripts`, `scale_factors` and `chunks`. Compared with earlier releases, the reader:

- defaults to `(1, 1024, 1024)` image and `(1024, 1024)` label chunks instead of storage-aligned chunks, and takes `scale_factors=()` instead of `None` for a single scale;
- places each FOV by an exact translation instead of an affine fitted to filtered cells;
- keeps raster pixels unflipped and applies the measured half-pixel-exact pixel convention;
- returns one global transcript element instead of one element per FOV, without writing temporary files or printing progress;
- rejects mismatched count and metadata populations;
- names the linkage column `region` and uses vendor or positional observation names;
- removes the redundant `global_only_image` and `global_only_labels` coordinate systems.
