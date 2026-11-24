# CODEX

{func}`spatialdata_io.codex` reads one regional processor-style cell table and
an optional stitched image. Use `cell_table_file` to choose between compensated
and uncompensated tables, or `image_file` to choose among images.

```text
region/
├── cells.csv                 # or an FCS 3.1 table
├── stitched.tif              # optional
└── experiment.json           # optional, read only with experiment_file
```

File options can be relative to the region directory or absolute. The selected
filename determines its format. [Dataset paths and symlinks](../user/paths.md)

## Cell table

Supply one row per cell and one region per file. Named columns may be reordered.
The original Nolan export needs to be converted to the supported column names.

| Fields | Interpretation |
| --- | --- |
| `cell_id`, `region` | IDs are unique uint64 values, including zero. CSV region names are literal strings, so `001` keeps its spelling. An FCS region is a positive integer. The region must be a valid SpatialData element name distinct from `table` and `images`, ignoring case. |
| `x`, `y` | Native cell positions in table pixels. CSV uses float64. |
| `cyc<digits>_ch<digits>` | At least one cycle/channel field is required. Intensities retain source feature order and can be negative after compensation. |
| Other fields | Observation annotations. CSV keeps literal text, including leading zeros, `NA`, `NULL` and empty strings. Unsupported annotation names are sanitized with a warning, e.g. `tile num` becomes `tile_num`. |

Akoya CSVs may use `short_name:long_label` on **every** field. Short names must
be unique. Expression fields retain their short names as feature IDs and use
the long labels as markers. FCS uses `$PnN` field names and available `$PnS`
marker labels. Marker labels can repeat.

**CSV requirements.** The header occupies one physical line. Each record has
the full header width and must fit in a 1 MiB source block. Quoted commas and
multiline data fields are accepted. Empty metadata strings are valid. IDs,
region names, XY and expression are required, and NUL characters are invalid.
Integer IDs accept hexadecimal and surrounding spaces or tabs, and reject
fractions, exponents, a leading plus sign, negatives, digit separators and
overflow. Intensities and XY round directly to float64 (so `1e-400` becomes
zero) and must be finite.

## Returned elements

- `tables["table"]` contains eager CSR expression in source row and feature
  order. CSV intensities are float64, and FCS preserves native numeric widths
  where possible. Observation names are zero-based row positions. `obs["cell_id"]`
  and categorical `obs["region"]` specify linkage. A uint64 ID column cannot
  be combined directly with an int64 instance column in SpatialData.
- `shapes[region]` contains radius-one display circles at the cell positions,
  indexed by cell ID. Radius is in native table pixels.
- `images["images"]`, when present, is a lazy `c, y, x` image at full resolution
  with source dtype and channel order. Its channels have their own names
  independent of the expression markers.

For a TIFF with several series, set `tiff_series`. Use `data_axes`, for example
`"CYX"`, to describe otherwise ambiguous axes. Declared RGB(A) samples get
colour names. Other channels use unique OME names, OME IDs or positional
indexes. Non-RGB OME samples within one channel get a `:sample:<index>` suffix.
Reinterpreting another axis as channels gives positional indexes. Retain the
source while computing or writing the image. [Raster options](../user/rasters.md)

## Calibration and registration

With no calibration, circles use `table_pixels` and the table retains native XY
in `obs`. To produce `obsm["spatial"]` and a `global` frame in micrometres,
supply the positive finite `pixel_size_um` for the table, or select its
associated Akoya experiment JSON. Versions `1.5.0.38` and `1.7.0.6` declare
`xyResolution` in nm/pixel. If supplied together, `pixel_size_um` must equal
`xyResolution / 1000`. Selecting an experiment asserts that its calibration
belongs to this table.

```python
from spatialdata_io import codex

sdata = codex("region", cell_table_file="cells.csv", experiment_file="experiment.json")
```

The stitched image stays in `image_pixels` unless you supply `image_to_table`.
This finite, invertible XY affine maps image pixel centres
`(column + 0.5, row + 0.5)` to table XY, accounting for the mosaic's size,
crop and orientation. Calibration then maps table pixels to micrometres. For
example, replace the following scale and affine with your acquisition values:

```bash
python -m spatialdata_io codex --input region --output region.zarr --pixel-size-um 0.5 \
  --image-to-table '[[2,0,10],[0,2,20],[0,0,1]]'
```
