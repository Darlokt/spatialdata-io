# Steinbock

{func}`spatialdata_io.steinbock` reads matched OME images, segmentation masks
and an AnnData table from a Steinbock project. The layout carries no format
version marker.

## Dataset layout

```text
project/
├── cells.h5ad               # steinbock export anndata
├── images.csv               # optional dimensions and acquisition extents
├── ome/<sample>.ome.tiff    # steinbock export ome
└── masks/<sample>.tiff      # steinbock segment
```

Images and masks pair by exact sample filename stem. Every file requires a
matching counterpart and at least one pair is required. Set `masks_dir` to
select a segmentation directory within the project. Use `table_file` to
choose another H5AD, with relative paths resolved from the project root.
See [dataset paths](../user/paths.md). Steinbock's other outputs, including raw
images, intensities, region properties, neighbours and `panel.csv`, are
outside this layout.

## Elements

- `images["<sample>_image"]` is a lazy `c, y, x` array in source dtype
  and channel order. One full-resolution TIFF series with OME metadata is
  required. OME channel names are used when unique and nonempty, with OME
  channel IDs as the fallback. RGB sample images are unsupported.
- `labels["<sample>_labels"]` is a lazy `y, x` mask in its source integer
  dtype. It shares the dimensions and coordinate frame of the image.
- `tables["table"]` preserves the H5AD row and feature order. The input
  `obs["image"]` holds exact `<sample>.tiff` names and observation names
  follow `Object <id> in <sample>.tiff`. IDs are positive mask values,
  unique within a sample; zero denotes background. The table links via
  categorical `obs["region"]` and uint64 `obs["cell_id"]`. Mask pixels
  remain lazy, so membership is not checked against pixel values.

Sample names are sanitized with a warning where needed. Element names must
stay unique ignoring case and must avoid collisions with derived frames.
Images and labels follow lexical image filename order. Their chunks and
optional pyramids use the [raster options](../user/rasters.md). Mask pyramids
require labels of at most 32 bits. Keep the TIFF files for later computation
or writing.

## Calibration and centroids

A calibrated sample maps its rasters to micrometres using the acquisition
extent divided by the pixel dimensions from `images.csv`, or complete
image OME X and Y physical sizes. Present OME sizes must agree. The physical
frame is `global` for a single sample and `<sample>` when reading several
samples. An uncalibrated sample uses `<sample>_pixels`. Samples are independent
and retain their own origins.

The optional `images.csv` requires named fields `image`, `width_px` and
`height_px`. It also accepts the pair `acquisition_width_um` and
`acquisition_height_um`. Headers are unique and every record has their full
width. Quoted multiline fields are accepted. Image identities must be
unique, nonempty and NUL-free. Dimensions parse as uint64, accepting
hexadecimal and surrounding spaces or tabs. Decimal fractions, exponent
spellings, a leading plus sign and digit separators fail. Extents parse as
float64. In selected rows dimensions are positive, and both extents must
be empty or give finite, positive micrometres per pixel.

The source table's `centroid-1` and `centroid-0` supply XY regionprops
centroids in `obsm["spatial_pixels"]` in source dtype. Their native frame
is `<sample>_regionprops_indices`. When every sample is calibrated,
`obsm["spatial"]` contains float64 micrometres, computed as
`(spatial_pixels + 0.5) * spacing`. With several samples,
`obs["spatial_frame"]` (when physical positions are available) and
`obs["spatial_pixel_frame"]` name each row's frames. The position arrays are
described in `uns["spatialdata_io_positions"]`. Any imported
`obsm["spatial"]` moves unchanged to `obsm["spatial_source"]`.

## Expression and annotations

`X` and an optional `layers["counts"]` become finite canonical CSR arrays.
Duplicate entries are summed with a warning. Integer overflow promotes to a
floating dtype only when all values remain exact. Other layers, mappings,
metadata and `raw` are retained. If present, `obs["Image"]` must equal
`obs["image"]` and is dropped. The feature column
`Final Concentration / Dilution` becomes `Final_Concentration`; invalid
annotation names are sanitized with a warning.

## Python and command line

```python
from spatialdata_io import steinbock

sdata = steinbock("project", masks_dir="masks_deepcell", table_file="cells_deepcell.h5ad")
sdata.write("sample.zarr")
```

```bash
python -m spatialdata_io steinbock --input project --output sample.zarr \
  --masks-dir masks_deepcell --table-file cells_deepcell.h5ad
```
