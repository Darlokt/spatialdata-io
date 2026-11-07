# DBiT-seq

{func}`spatialdata_io.dbit` reads an unversioned DBiT-seq H5AD export and its barcode map, with an optional registered tissue image. It returns a linked expression table and one polygon for every observed barcode-grid location.

## Dataset layout

The supported layout is:

```text
dataset/
├── sample.h5ad                 # required: exactly one *.h5ad
├── barcode_list.txt            # required: one filename containing barcode_list
└── tissue_lowres_image.png     # optional: filename ending in tissue_lowres_image.png
```

`table_file`, `barcode_positions_file`, and `image_file` override discovery. Relative overrides resolve against `path`; absolute overrides may point outside that directory. Missing or ambiguous required artifacts fail. An explicitly requested image must exist; when the default image is absent, the reader returns the table and shapes without an image. Image overrides support TIFF, OME-TIFF, BigTIFF/BTF, PNG, and JPEG.

Discovered artifacts, including symlink targets, must remain inside the dataset root unless `path_policy="relaxed"` ([Dataset paths and symlinks](dataset_paths.md)). The barcode map is a headerless, tab-separated file with exactly two columns and at most fifty A and fifty B positions:

```text
A1	AACCGGTT
A2	TTGGCCAA
B1	GGGGTTTT
B2	CCCCAAAA
```

The position identifiers must supply both `A1` through `AN` and `B1` through `BN`, with contiguous, matching positions and `1 <= N <= 50`. Each position appears once. Each sequence contains exactly eight DNA bases (`A`, `C`, `G`, or `T`), matched case-insensitively, and is unique within its axis. The same sequence may occur on both axes. Source row order is unrestricted.

H5AD observation names encode the B sequence in their first eight characters and the A sequence in the next eight. Any trailing suffix is preserved. For example, `GGGGTTTTAACCGGTT-1` maps to `(A1, B1)`. Observation names and mapped grid locations must be unique, and both component barcodes must occur in the map. The table must have observations, nonempty unique feature names, and expression values. It may contain a subset of the full square grid.

## Output and coordinates

`dataset_id` defaults to the H5AD filename stem (`sample` in the layout above). It must be a valid SpatialData element name: nonempty, only letters, digits, underscores, dots or hyphens, not `.` or `..`, and not beginning with two underscores. Because SpatialData compares element names case-insensitively, `table` is reserved in any letter case. A `dataset_id` argument is checked before any file is read: a non-string raises `TypeError` and an invalid name `ValueError`. A filename stem that breaks these rules is repaired instead, with one `UserWarning` naming the original and the repaired name: every other character becomes `_`, a leading run of underscores becomes one, an empty name, `.` or `..` becomes `unnamed`, and a reserved name gains a trailing `_`. For example `my run.h5ad` gives `my_run` and `Table.h5ad` gives `Table_`. Pass `dataset_id` to choose the name yourself. The returned elements are:

| Element | Contract |
| --- | --- |
| `shapes[dataset_id]` | Eager polygons indexed by consecutive integer `pixel_id` values, starting at zero. |
| `tables["table"]` | Eager owned AnnData, sorted by A position and then B position, with source feature order retained. |
| `images[dataset_id + "_image"]` | Optional lazy image in `(c, y, x)` order, with source pixel dtype and channel order retained. |

The reader owns the generated `obs` columns `array_A`, `array_B`, `sample`, `region`, and `pixel_id`. It deliberately overwrites these columns whether imported values are equal, conflicting, or null. Any inherited `uns["spatialdata_attrs"]` is discarded without interpretation, and fresh table linkage targets the final shape index. Other observation/feature metadata and aligned arrays are retained, including their source dtypes, missingness, and categorical order. Imported `obs` and `var` column names that a SpatialData table does not accept are sanitized with one `UserWarning` per frame listing every original and sanitized name: characters other than letters, digits, `_`, `-` and `.` become `_` (`my note` gives `my_note`), and a name that then equals another column ignoring letter case gets the first free `_1`, `_2`, … suffix; the five reader columns keep their spelling. Named annotation columns accept arbitrary order and additional fields; duplicate column names fail. Generated grid identifiers reach the output as NumPy int64, `sample` as pandas strings, and `region` as categorical strings. `X` and an existing `counts` layer become finite, canonical SciPy CSR sparse arrays; other layers retain their representation. Duplicate sparse entries are summed, promoting overflowing integer storage directly to exact float32/f64 only when necessary. A successful read emits one warning identifying combined entries and dtype changes. No supported exact result raises an error; source files remain unchanged. The reader owns its newly read table and does not modify the source H5AD.

A is the x axis and B is the y axis. The polygons are always stored in grid units, with or without an image: grid cell `(A, B)` occupies `[A - 1, A] × [B - 1, B]` before the border inset. `include_border=True` shrinks each edge inward by `0.125 * border_scale` grid units. `border_scale` must be finite and satisfy `0 <= border_scale < 4`, including when `include_border=False`.

With an image, a uniform scale `s = max(image_height, image_width) / N` maps the grid onto the full-resolution image of the selected level. This preserves the historical square-grid convention for rectangular images too: the grid spans the longer image dimension on both axes. The image must already be cropped, oriented, and registered appropriately; the reader does not infer a registration or physical units.

DBiT files carry no physical calibration, so no element is placed in `global`, which always means micrometres. The coordinate systems are:

| Coordinate system | Unit | `shapes[dataset_id]` | `images[dataset_id + "_image"]` |
| --- | --- | --- | --- |
| `grid_pixels` | one barcode-grid step (one DBiT pixel) | `Identity()` | not present |
| `image_pixels` (only with an image) | one full-resolution image pixel | `Scale([s, s], axes=("x", "y"))` | `Identity()` |

OME spacing or nominal channel width alone cannot establish the barcode pitch, origin, orientation and capture footprint, so neither triggers automatic conversion to micrometres.

`invalid_geometries="keep"`, `"drop"` and `"raise"` produce identical supported squares and linkage because this geometry is valid by construction. No topology scan is performed. The option is validated before I/O, and floating-point collapse of a positive square width fails in every mode.

`table.obsm["spatial_grid"]` retains the one-based integer `(array_A, array_B)` coordinates. These values are barcode positions, not polygon centers or image-pixel coordinates.

`uns["spatialdata_io_positions"]` describes `spatial_grid` as one-based grid ordinals in the `grid_pixels` frame. Existing `spatial_grid` or `spatialdata_io_positions` destinations raise. Imported `obsm["spatial"]` moves exactly to `spatial_source` before sorting, preserving dtype and shape; an occupied destination raises when this move is required. No physical `spatial` is produced. Channel widths and fitting display polygons to an image do not establish physical table calibration.

## Image options and lazy execution

TIFF metadata supplies source axes when they are unambiguous. Use `data_axes`, one case-insensitive string such as `"CYX"` or `"YXS"`, when an explicit interpretation is required; other types raise `TypeError`. By default the reader reads the TIFF's only series; a file with several series raises `ReaderFormatError` unless `tiff_series` selects one. `tiff_level=0` selects the full-resolution level of that series; other nonnegative indices select an existing stored pyramid level. PNG and JPEG use their encoded image layout and do not accept `data_axes`, `tiff_series` (not even `0`) or a nonzero `tiff_level`. Wrongly typed indexes raise `TypeError`, negative ones `ValueError`.

Image channels are named `r, g, b`, or `r, g, b, a` with a declared alpha sample, when the image's samples become its channels and the file itself declares them RGB: TIFF `PhotometricInterpretation` RGB (alpha through `ExtraSamples`) or JPEG-compressed YCbCr, PNG colour types 2, 3 and 6 (alpha through colour type 6 or `tRNS`), and three-component JPEG. Viewers composite such images as colour. Otherwise reliable scientific channel names from OME metadata are retained in source order; OME declares an interleaved image as one channel with several samples per pixel, so its samples are named `<name or ID>:sample:0`, `:sample:1`, … (for example `Channel:0:0:sample:0`). Images without either declaration, such as grayscale PNG or non-OME `MinIsBlack` TIFF samples, keep positional channel coordinates. The channel count or dtype never makes an image RGB.

`scale_factors` and `chunks` follow the shared [raster options](raster_options.md): by default a single scale and images in `(1, 1024, 1024)` chunks. A pyramid is generated only when `scale_factors` requests one, and factors whose product exceeds the smaller image side raise `ValueError`; the grid scale always uses the full-resolution extent of the selected TIFF level. Image options other than their defaults require an image. Metadata inspection is eager, while pixel decoding stays lazy until the returned image is computed or written. TIFF tasks decode the tiles or strips that overlap the computed chunks. PNG and JPEG can require decoding the entire encoded frame even for a small selection; rechunking does not change that codec limitation.

Missing or wrong-kind paths raise the corresponding filesystem exception. Malformed or ambiguous layouts, barcode schemas, and identifier relationships raise `ReaderFormatError`, a `ValueError` subclass, with reader and artifact context. Invalid option values raise `ValueError`, and unsupported option types raise `TypeError`. The reader does not return partial results after a validation failure.

## Python and command line

```python
from spatialdata_io import dbit

sdata = dbit(
    "dataset",
    table_file="sample.h5ad",
    barcode_positions_file="barcode_list.txt",
    image_file="registered.tif",
    include_border=True,
    border_scale=1.0,
    scale_factors=(2, 2),
    chunks=(1, 512, 512),
)
sdata.write("sample.zarr")
```

The CLI uses the same option names in kebab case. Repeat `--scale-factors` once per generated level, and `--chunks` once for a single size or three times in CYX order:

```bash
python -m spatialdata_io dbit --input dataset --output sample.zarr \
  --table-file sample.h5ad --barcode-positions-file barcode_list.txt \
  --image-file registered.tif --no-include-border \
  --scale-factors 2 --scale-factors 2 --chunks 1 --chunks 512 --chunks 512
```

`--include-border/--no-include-border` is a boolean flag pair. The previous `anndata_path`, `barcode_position`, `image_path`, and `border` arguments and their CLI options have been removed.

Images are now chunked `(1, 1024, 1024)` by default instead of keeping the storage-aligned chunks of the source, and `chunks=None` is no longer accepted. Pass `chunks` explicitly to choose another layout. The new `scale_factors` option builds pyramid levels.

Shapes and images are no longer placed in `global`. The polygons are stored in grid units in `grid_pixels`; with an image they reach the image's `image_pixels` frame through `Scale([s, s])`, where earlier versions stored the scaled vertices directly. The overlay in `image_pixels` is unchanged up to one floating-point rounding step. The position declaration names the `grid_pixels` frame instead of `barcode_grid`.

A filename stem that is not a valid element name is now repaired with a warning (for example `my run.h5ad` gives `my_run`) instead of raising.

`tiff_series` now defaults to `None` (and `--tiff-series` has no default): a TIFF with several series raises instead of silently reading the first one. Pass `tiff_series=0` to keep the previous selection. PNG and JPEG images reject any `tiff_series`. A non-string `dataset_id` raises `TypeError` instead of `ValueError`.

Images whose file declares RGB samples (RGB TIFF and OME-TIFF, RGB, RGBA and palette PNG, three-component JPEG) now have channel coordinates `r, g, b` or `r, g, b, a` instead of positional indexes or `<name or ID>:sample:<k>`.

## Complete synthetic example

This example creates a two-by-two barcode map, two observed locations, sparse expression, and a small RGB image. All generated files live in a temporary directory, and pixels are computed before that directory closes.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import numpy as np
import pandas as pd
import imagecodecs
from scipy import sparse

from spatialdata import transform
from spatialdata.transformations import Identity, Scale, get_transformation

from spatialdata_io import dbit

with TemporaryDirectory() as directory:
    root = Path(directory)
    ad.AnnData(
        X=sparse.csr_array([[3, 4], [1, 2]], dtype=np.int32),
        obs=pd.DataFrame(index=["CCCCAAAATTGGCCAA", "GGGGTTTTAACCGGTT"]),
        var=pd.DataFrame(index=["gene-b", "gene-a"]),
    ).write_h5ad(root / "sample.h5ad")
    (root / "barcode_list.txt").write_text(
        "A2\tTTGGCCAA\nB2\tCCCCAAAA\nA1\tAACCGGTT\nB1\tGGGGTTTT\n",
        encoding="utf-8",
    )
    pixels = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    (root / "tissue_lowres_image.png").write_bytes(imagecodecs.png_encode(pixels))

    sdata = dbit(root)
    table = sdata.tables["table"]
    assert isinstance(table.X, sparse.csr_array)
    assert table.obs_names.tolist() == ["GGGGTTTTAACCGGTT", "CCCCAAAATTGGCCAA"]
    assert table.var_names.tolist() == ["gene-b", "gene-a"]
    assert table.obs["pixel_id"].tolist() == sdata.shapes["sample"].index.tolist()
    np.testing.assert_array_equal(table.X.toarray(), [[1, 2], [3, 4]])
    np.testing.assert_allclose(sdata.shapes["sample"].total_bounds, [0.125, 0.125, 1.875, 1.875])
    assert get_transformation(sdata.shapes["sample"], get_all=True) == {
        "grid_pixels": Identity(),
        "image_pixels": Scale([5.0, 5.0], axes=("x", "y")),
    }
    overlay = transform(sdata.shapes["sample"], to_coordinate_system="image_pixels")
    np.testing.assert_allclose(overlay.total_bounds, [0.625, 0.625, 9.375, 9.375])
    image = sdata.images["sample_image"]
    assert image.dims == ("c", "y", "x")
    assert image.coords["c"].values.tolist() == ["r", "g", "b"]
    np.testing.assert_array_equal(image.data.compute(), pixels.transpose(2, 0, 1))
```

A returned lazy image remains backed by its source path: keep the source files available until computing or writing it. Reader provenance records the reader name and package version without source paths; this unversioned layout does not supply a format version.
