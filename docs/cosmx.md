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

Table rows follow metadata order. Observation names are the vendor `cell` IDs when the metadata provides them (AtoMx) and row positions `"0"`, `"1"`, ... otherwise. AtoMx's `cell_id` column is an exact copy of `cell` and a case variant of `cell_ID`, so it is dropped; any other case-variant column names fail. `fov` and `cell_ID` are int64. `Area`, `Area.um2`, `AspectRatio`, `Width`, `Height` and every `Mean.*`/`Max.*` intensity are float64, with empty fields and `NA` as missing values. All other metadata columns stay exact text. `obsm["spatial"]` and `obsm["global"]` hold the local and global cell centers in pixels.

With a `CellLabels` directory, `region` links every cell to `<fov>_labels` with instance key `cell_ID`, and every FOV with cells must have labels. Without that directory the table is returned unlinked.

## Coordinates

| Element | Coordinate systems |
| --- | --- |
| `images["<fov>_image"]` | `<fov>` (local pixels) and `global` |
| `labels["<fov>_labels"]` | `<fov>` and `global` |
| `points["<dataset_id>_transcripts"]` | `global` and every `<fov>` |

CosMx local coordinates are pixel indices with a reversed vertical axis: the raster pixel at row `r` and column `c` of an `H`-row FOV has local coordinates `(c, H - r)`. This was measured on the public 2021 Lung5_Rep2 release and on an AtoMx mouse-brain export, from label pixels against cell metadata and polygon extremes. Raster pixels are returned exactly as stored; the reflection is part of each raster's transformation.

Each FOV's local-to-global map is a translation per axis, possibly with a reversed axis: `global = ±local + offset`. Exports differ in the sign of the y axis. The sign and offset are derived from every metadata cell and accepted only when one offset lies within the printed rounding interval of all of them; no affine is fitted. A pixel FOV-position file (`fov,x_px,y_px` or `FOV,x_global_px,y_global_px,...`) must agree with the cells and places FOVs without cells; millimetre-only position files are accepted but not used. A FOV with rasters but no cells and no pixel position fails.

The raster convention has only been evidenced for exports whose global axes run in the same direction as the local axes. Rasters in an export with a reversed global axis therefore fail, while its table and transcripts are read.

## Transcripts

Transcript points are placed at their global `x_global_px`/`y_global_px` positions; the local coordinates are kept as `x_local` and `y_local`. Targets are categorical with the count features as categories. Annotations are `fov`, `cell_ID` (0 marks unassigned transcripts) and, when present, `z_raw`, `CellComp` and `cell`. `z_raw` is the vendor's slice number in arbitrary units and can be negative; it is not a physical coordinate.

Construction reads only the header. Plain CSV sources are split into newline-delimited blocks of about 8 MiB, so each record must occupy one physical line. Compressed `.csv.gz` sources must consist of independently compressed members that each end with a complete record, as AtoMx exports do; these members are split into lazy ranges of about 16 MiB of compressed data without decompressing the file first. A single-stream gzip file (for example from the `gzip` command) fails: decompress it, or recompress it in blocks. Unknown targets and malformed values fail when their partition is computed.

## Options and execution

`scale_factors` and `chunks` follow the shared [raster options](raster_options.md): by default a single scale, images in `(1, 1024, 1024)` chunks and labels in `(1024, 1024)` chunks. Generated label pyramids require labels of at most 32 bits; 64-bit labels need the default single scale. JPEG and PNG composites decode their whole frame for any selection; TIFF sources read only the tiles, or strips grouped into bands of at most 16 MiB, that intersect the selected chunks. Construction executes no raster or transcript tasks.

## Python and command line

```python
from spatialdata_io import cosmx

sdata = cosmx("Lung5_Rep2", chunks=(3, 1024, 1024))
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
