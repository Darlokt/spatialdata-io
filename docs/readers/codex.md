# CODEX

{func}`spatialdata_io.codex` reads processed CODEX images and regional cell
tables. For a flat export with a table and an optional stitched image, use
`cell_table_file` to choose an explicit table or `image_file` to choose among
images.

```text
region/
├── cells.csv                 # or an FCS 3.1 table
├── stitched.tif              # optional
└── experiment.json           # optional, read only with experiment_file
```

File options can be relative to the region directory or absolute. The selected
filename determines its format. [Dataset paths and symlinks](../user/paths.md)

## Akoya Processor images and tables

Akoya's Processor aligns and stitches the acquisition and can segment cells.
An export can include one stitched TIFF per marker and one cell table per
region. The example below shows a processed directory with two markers and
one segmented region. Use `regions=["reg001"]` to select a region and
`read_tables=False` for image-only workflows.

```text
processed_run/
├── stitched/
│   └── reg001/
│       ├── reg001_cyc001_ch001_DAPI.tif
│       └── reg001_cyc002_ch002_CD3.tif
└── segm/segm-1/fcs/compensated/
    └── reg001_compensated.csv
```

Cycle and channel are one-based. Both `ch001` and `cha001` occur in Akoya's
documentation. Planes are stacked in numeric cycle/channel order. The marker
suffix identifies the measured target. The tiles used to construct a stitched
image and segmentation masks are separate products. Supply a mask explicitly with `labels`.
The channel records in `sdata.attrs["akoya"]["regions"][region]["channels"]`
retain the original biomarker and one-based cycle and channel, in image order.
[Akoya Processor outputs](https://help.codex.bio/codex/processor/technical-notes/expected-output)

Processor 1.8 also supports processed CODEX QPTIFF output. For whole-slide
scans, see [PhenoCycler and Akoya QPTIFF](phenocycler.md).
[Akoya Processor installation and versions](https://help.codex.bio/codex/processor/installation/download-and-install)

## Legacy Nolan Lab montages

The Nolan Lab Uploader is a separate pipeline used by earlier datasets.
A regional `reg001_montage.tif` contains best-focus images, with acquisition
cycles stored as ImageJ frames. Its writer reduces each tile before stitching.
The default reduction factor is two and the producer can change it. Tile sizes
that are not divisible by the factor are rounded down individually. Use the
actual registration to align regional cell positions to a montage.
[Nolan montage writer](https://github.com/nolanlab/CODEX/blob/529ea4a2ebc796c7a51698f8f695ad2365021f4b/src/main/java/org/nolanlab/codex/upload/driffta/MakeMontage.java#L38-L156)

## Cell table

Set `compensation="compensated"` or `compensation="uncompensated"` to select the
same signal product across regions. The corresponding CLI option is
`--compensation compensated` or `--compensation uncompensated`. Selection uses
the filename suffix, such as `reg001_compensated.csv`. Every selected region
that supplies table candidates must have the requested variant. Regions that
supply only images still return their images. With the default `None`, both
variants are eligible and a region with both requires a selection. A recognized
filename suffix is recorded in `table.uns["spatialdata_io_compensation"]`.

When the eligible outputs are a CSV and an FCS with the same stem in the same
folder, CODEX reads the CSV. The Processor exports both formats for the same
measurements, and CSV can retain precision lost in FCS numeric storage. The
stem and extension match case-insensitively. Tables with distinct stems or from
different folders still require disambiguation. Select an explicit file for a
single region with `cell_table_file` or pass the table directly. [Akoya FCS format](https://help.codex.bio/codex/processor/technical-notes/fcs-format)

Supply one row per cell and one region per file. Named columns may be reordered.

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

Raw Nolan segmentation tables require conversion before use. Their cell IDs
restart in each tile. When preparing a regional table, preserve the source tile
and cell identifiers as annotations, assign a unique regional `cell_id`, and
use stitched-region XY after tile offsets have been applied. Renaming
`Cyc_1_ch_1` to a supported expression field is only one part of that conversion.
[Nolan cell table writer](https://github.com/nolanlab/CODEX/blob/529ea4a2ebc796c7a51698f8f695ad2365021f4b/src/main/java/org/nolanlab/codex/segm/Main.java#L445-L509)
and [regional coordinate calculation](https://github.com/nolanlab/CODEX/blob/529ea4a2ebc796c7a51698f8f695ad2365021f4b/src/main/java/org/nolanlab/codex/segm/ConcatenateResults.java#L96-L123).

**CSV requirements.** The header occupies one physical line. Each record has
the full header width and must fit in a 1 MiB source block. Quoted commas and
multiline data fields are accepted. Empty metadata strings are valid. IDs,
region names, XY and expression are required, and NUL characters are invalid.
Integer IDs accept hexadecimal and surrounding spaces or tabs, and reject
fractions, exponents, a leading plus sign, negatives, digit separators and
overflow. Intensities and XY round directly to float64 (so `1e-400` becomes
zero) and must be finite.

## Returned elements

- A flat export uses `tables["table"]` and `images["images"]`. A named region
  such as `reg001` uses `tables["reg001_table"]` and `images["reg001_image"]`.
  A table's `region` column names its mask or centroid shapes element.
  Each supplied mask also contributes a labels element.
- Each table contains eager CSR expression in source row and feature
  order. CSV intensities are float64, and FCS preserves native numeric widths
  where possible. Observation names are zero-based row positions. `obs["cell_id"]`
  and categorical `obs["region"]` specify linkage. A uint64 ID column cannot
  be combined directly with an int64 instance column in SpatialData.
- `shapes[region]` contains radius-one display circles at the cell positions,
  indexed by cell ID. Radius is in native table pixels. The Processor's
  `size` field counts voxels in the segmented cell. It stays in the
  observation annotations. [Akoya FCS specification](https://help.codex.bio/codex/processor/technical-notes/fcs-format)
- Each raster is a lazy `c, y, x` image at full resolution
  with source dtype and channel order. Its channels have their own names
  independent of the expression markers.

For a TIFF with several series, set `tiff_series`. Use `data_axes`, for example
`"CYX"`, to describe otherwise ambiguous axes. Declared RGB(A) samples get
colour names. Other channels use unique OME names, OME IDs or positional
indexes. Non-RGB OME samples within one channel get a `:sample:<index>` suffix.
Reinterpreting another axis as channels gives positional indexes. Retain the
source while computing or writing the image. [Raster options](../user/rasters.md)

## Link segmented cells to a mask

Pass `labels="cells.tif"` for one selected region or a mapping such as
`labels={"reg001": "segm/reg001_mask.tif"}` for named regions. The
reader returns `labels["reg001_labels"]`. A flat export uses `labels["labels"]`
and accepts a single path or the mapping key `"images"`.
Supplying a mask associates its instance values with the table's
`cell_id`. The table links to the labels and keeps the normalized region name in
`uns["spatialdata_io_source_region"]`. Cell IDs must be positive and fit
the mask dtype. The reader checks those constraints without scanning label
pixels, so the caller must verify that both exports use the same IDs.
Regions without a mask retain their centroid circles.

The mask belongs to the selected image. Matching grids require an explicit
association and equal YX extents. Supply `labels_to_image` for a cropped
or resampled mask. The optional `image_to_table` independently registers
image pixels to table coordinates. Supplying mask IDs alone establishes
cell identity and leaves the centroid calibration independent.
[Mask coordinates and CLI example](phenocycler.md#segmentation-masks)

## Calibration and registration

For a single region without calibration, circles use `table_pixels` and the
table retains native XY in `obs`. To produce `obsm["spatial"]` and a `global` frame in micrometres,
supply the positive finite `pixel_size_um` for the table, or select its
associated Akoya experiment JSON. Versions `1.5.0.38` and `1.7.0.6` declare
`xyResolution` in nm/pixel. If supplied together, `pixel_size_um` must equal
`xyResolution / 1000`. Selecting an experiment asserts that its calibration
belongs to this table.

```python
from spatialdata_io import codex

sdata = codex("region", cell_table_file="cells.csv", experiment_file="experiment.json")
```

A stitched image with its own OME XY calibration has a physical image frame.
For one region this is `global` when reading the image alone and `image_um`
when an unregistered cell table is present. Without image calibration it uses
`image_pixels`. Multiple independent regions use region-prefixed frames,
such as `reg001_image_pixels` and `reg001_table_um`.

To register an image to its cell table, supply `image_to_table` for one selected
region. This finite, invertible XY affine maps image pixel centres
`(column + 0.5, row + 0.5)` to table XY, accounting for the mosaic's size,
crop and orientation. Calibration then maps table pixels to micrometres. For
example, replace the following scale and affine with your acquisition values:

```bash
python -m spatialdata_io codex --input region --output region.zarr --pixel-size-um 0.5 \
  --image-to-table '[[2,0,10],[0,2,20],[0,0,1]]'
```
