# In situ sequencing (ISS, experimental)

The experimental reader {func}`spatialdata_io.experimental.iss` imports an
image, its segmentation and an H5AD expression table from an in situ
sequencing workflow. You specify the three files because ISS pipelines have
no common layout or format version.

## Required files

```text
dataset/
├── raw.tiff     # raw_image_file
├── labels.tiff  # labels_file
└── cells.h5ad   # table_file
```

Rasters are TIFF-family files (`.tif`, `.tiff` or `.btf`), each with one
series read at full resolution. Images and masks must already share a pixel
grid. The reader checks their YX dimensions and relies on the caller to
establish their registration. Relative paths resolve from the dataset root
and absolute file paths can be external. See [dataset paths](../user/paths.md).

## Output and coordinates

| Element | Contents |
| --- | --- |
| `images["<dataset_id>_raw_image"]` | Lazy `c, y, x` image with source channel order and dtype. |
| `labels["<dataset_id>_labels_image"]` | Lazy `y, x` segmentation with source integer dtype. Zero denotes background. |
| `tables["table"]` | H5AD table in source observation and feature order, linked to the labels. |

The default `dataset_id` is `"region"`. Raster axes come from TIFF metadata.
For an ambiguous raw image, supply `data_axes` as one string such as `"CYX"`.
Grayscale images receive one channel and singleton nonspatial axes can be
removed. Declared RGB(A) samples use `r, g, b` (and `a` for declared alpha).
Other channels take OME names or IDs when available, otherwise positions.

Complete OME X and Y physical sizes on the raw image map both rasters to
`global` in micrometres without moving the origin. OME sizes that the mask
declares must agree. With incomplete calibration, or if `data_axes` moves
X or Y, both rasters use `image_pixels`.

The table links through categorical `obs["region"]` and a positive integer
instance field. The default derives uint64 `obs["instance_id"]` from
observation names containing only ASCII digits, allowing leading zeros.
Specify `instance_key` to use a unique, non-null integer column in `obs`.
IDs must fit the label dtype. The reader leaves label pixels lazy, so it
does not check whether every ID appears in the mask.

`X` and optional `layers["counts"]` are eager, finite CSR sparse arrays.
Duplicate sparse entries are summed and reported. Integer overflow promotes
to a supported floating dtype only when the result is exact. Feature names
are made unique in place in the imported table. Other annotations and layers
are retained, with a warning if a name needs sanitizing. An original
`obsm["spatial"]` is preserved as `obsm["spatial_source"]` with its units and
axes unspecified.

The default is one raster scale with `(1, 1024, 1024)` image chunks and
`(1024, 1024)` label chunks. Generated label levels require a mask of at most
32 bits. Keep TIFF files available until lazy computation or writing is
complete. See [raster options](../user/rasters.md).

## Python and command line

```python
from spatialdata_io.experimental import iss

sdata = iss("dataset", raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad")
sdata.write("sample.zarr")
```

```bash
python -m spatialdata_io iss --input dataset --output sample.zarr \
  --raw-image-file raw.tiff --labels-file labels.tiff --table-file cells.h5ad
```
