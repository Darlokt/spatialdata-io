# CODEX

{func}`spatialdata_io.codex` reads a single regional processor-style cell table and an optional stitched image.

```text
region/
├── cells.csv                 # or FCS 3.1 cells.fcs
├── stitched.tif              # optional TIFF-family image
└── experiment.json           # optional, only read when explicitly selected
```

The supported normalized table has named `cell_id`, `region`, `x`, `y` and complete `cyc<digits>_ch<digits>` expression fields. Arbitrary column order and additional observation metadata are accepted; ambiguous tables or images require `cell_table_file` or `image_file`. Relative overrides resolve against the dataset root, and absolute caller overrides may be external. Multiple regions in one table are rejected. The region names the shapes element, so a named region must be a valid SpatialData element name (letters, digits, `_`, `-`, `.`) other than `table` or `images` in any letter case. Complete Akoya CSV headers of the form `short_name:long_label` are also supported: every field must be qualified, the first colon separates name and label, short names must be unique, and nonempty expression labels populate `var['marker']` in source feature order. Metadata values and normalized short-name order are preserved; repeated marker labels such as Blank remain valid. Original Nolan export field names require separate normalization and are unsupported.

`tables['table']` preserves source row and feature order, exact identifiers and native numerical XY in `obs['x']`/`obs['y']`. Expression is eager, finite, canonical CSR sparse-array data; negative compensated intensities remain valid. CSV expression uses a common real dtype, retaining integer storage when possible and rejecting lossy promotion. Native CSV coordinates may likewise retain integer storage. Cell IDs normalize exactly to int64 or uint64, including integral decimal or exponent spellings. Regions retain their numeric-or-name semantics: positive integral numbers normalize to string names, and nonempty names remain names.

Additional CSV observation annotations remain exact strings, including numeric-looking values, leading zeros, `NA`, `NULL` and explicit empty strings. A physically missing field is rejected. Numerical interpretation applies only to expression, defined `x`/`y` fields, region normalization and cell IDs. FCS events are decoded eagerly by readfcs, retaining native numerical widths for expression and annotations; `PnN` supplies feature IDs and available `PnS` labels populate `var['marker']`.

CSV numerical batches use at most 10,000 rows and a width-aware 32 MiB target, allowing at least one row. A single sufficiently wide numerical row can exceed that target. Header, parser and string buffers, retained annotations and final sparse outputs have additional costs; this policy bounds neither total memory nor an arbitrarily large raw CSV record. FCS decoding is separate from this batch policy.

Integer cell IDs link the table to `shapes[str(region)]`. These are radius-one display circles in native pixels, not measured cell boundaries. Without calibration they use `table_pixels`, and the table has no canonical `obsm['spatial']`.

## Calibration and registration

A positive finite `pixel_size_um` explicitly declares micrometres per acquisition/table pixel. The caller establishes association with the selected source; there is no universal scale. Canonical `obsm['spatial']` becomes float64 source XY multiplied by this scale, preserving the native origin and orientation. Nonfinite physical products are rejected. A defensive numerical check also rejects a nonzero coordinate rounded to zero by an exceptionally small scale; representable nonzero subnormal results remain accepted. The circles use the same transformation into physical `global`; their stored geometry and radius remain native. No duplicate native coordinate array is added. Position meaning, units, frame and bounded calibration evidence are described in `uns['spatialdata_io_positions']`.

Alternatively, explicitly select `experiment_file` for an associated Akoya experiment declaring `xyResolution` in nm/pixel. Supported declarations are versions `1.5.0.38` and `1.7.0.6`. Strict JSON is read once; the declared positive finite value is divided by 1000, and the Akoya version is recorded as the format version in `attrs`. The caller asserts that this experiment calibrates the selected regional table. The reader does not infer association from directory proximity, filenames or image bounds. Manufacturer processor metadata can originate in generated microscope defaults; callers must verify the declaration for their acquisition. If `pixel_size_um` is also supplied, it must equal `xyResolution / 1000` exactly. No unexplained conflict is averaged or fitted away. Automatic matched table/experiment discovery and Nolan metadata are not supported.

An optional image retains a separate `image_pixels` frame, even when the table is calibrated. Table pixel size does not establish a selected montage's resize factor, crop origin or orientation. Supply `image_to_table` as an existing SpatialData XY `Affine` to assert that complete mapping. Its input is the SpatialData raster convention, with pixel centres at `(column + 0.5, row + 0.5)`; its output is native table XY. The matrix must be finite and invertible. The reader composes the physical scale after this affine, or registers to `table_pixels` without physical calibration. Neither dimensions nor acquisition-versus-processed tile-size ratios determine registration.

## Images and execution

`images['images']` is lazy CYX with source dtype and channel order. OME names are used collectively when unique and nonempty, otherwise OME IDs; an OME channel count that differs from the raster fails. Interleaved samples (the TIFF `S` axis) become channels. When the TIFF declares them RGB, through `PhotometricInterpretation` RGB or JPEG-compressed YCbCr, they are named `r, g, b`, or `r, g, b, a` when `ExtraSamples` declares the fourth sample alpha, so viewers composite the image as colour. Otherwise OME describes them as one channel record with several samples per pixel, so they are named `<name or ID>:sample:0`, `:sample:1`, … (for example `Channel:0:0:sample:0`); without OME metadata they are labelled by position. Three- or four-channel images are never treated as RGB because of their channel count or dtype. RGB TIFF and OME-TIFF images previously had positional or `<name or ID>:sample:<k>` channel coordinates; they now use `r, g, b(, a)`. Images are not matched to expression markers. `data_axes` is one string of distinct axis letters (for example `"CYX"`) that names otherwise ambiguous source axes, and `tiff_series` selects a source series, which is required when the file has several; level zero is selected without assuming that the file itself is at acquisition resolution. `scale_factors` and `chunks` follow the [raster options](raster_options.md); factors whose product exceeds the image extent are rejected. Construction executes zero pixel tasks; partial reads retain the shared TIFF tile/strip behavior. Files close before return or inside lazy tasks. Caller-owned arrays, transformations and options are not mutated.

```python
from spatialdata_io import codex

# Use the acquisition's associated scale; the selected image stays independent.
sdata = codex("region", pixel_size_um=0.37740384615384613)
# Explicit caller association with a supported Akoya declaration:
sdata = codex("region", experiment_file="experiment.json")
```

```bash
spatialdata_io codex --input region --output region.zarr --pixel-size-um 0.5 \
  --image-to-table '[[2,0,10],[0,2,20],[0,0,1]]'
```

The CLI matrix maps selected-image raster coordinates to native table XY; the example's scale and translation must be replaced with the actual associated mapping. Path-kind errors preserve the corresponding standard exception. Ambiguous or malformed artifacts raise a contextual format error; invalid public calibration and image options raise `TypeError` or `ValueError`.
