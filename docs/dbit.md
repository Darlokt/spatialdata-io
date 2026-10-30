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

Discovered artifacts, including symlink targets, must remain inside the dataset root. The barcode map is a headerless, tab-separated file with exactly two columns and at most 16 KiB of metadata:

```text
A1	AACCGGTT
A2	TTGGCCAA
B1	GGGGTTTT
B2	CCCCAAAA
```

The position identifiers must supply both `A1` through `AN` and `B1` through `BN`, with contiguous, matching positions and `1 <= N <= 50`. Each position appears once. Each sequence contains exactly eight DNA bases (`A`, `C`, `G`, or `T`), matched case-insensitively, and is unique within its axis. The same sequence may occur on both axes. Source row order is unrestricted.

H5AD observation names encode the B sequence in their first eight characters and the A sequence in the next eight. Any trailing suffix is preserved. For example, `GGGGTTTTAACCGGTT-1` maps to `(A1, B1)`. Observation names and mapped grid locations must be unique, and both component barcodes must occur in the map. The table must have observations, nonempty unique feature names, and expression values. It may contain a subset of the full square grid.

## Output and coordinates

`dataset_id` defaults to the H5AD filename stem (`sample` in the layout above). It must be nonempty, use only letters, digits, underscores, dots or hyphens, and must not begin with two underscores. The names `.`, `..`, and `table` (case-insensitive) are reserved. The returned elements are:

| Element | Contract |
| --- | --- |
| `shapes[dataset_id]` | Eager polygons indexed by consecutive integer `pixel_id` values, starting at zero. |
| `tables["table"]` | Eager owned AnnData, sorted by A position and then B position, with source feature order retained. |
| `images[dataset_id + "_image"]` | Optional lazy image in `(c, y, x)` order, with source pixel dtype and channel order retained. |

The table records `array_A`, `array_B`, `sample`, categorical `region`, and integer `pixel_id` columns. Its region and instance linkage targets the final shape index. Existing H5AD linkage metadata is replaced. Its `region` and `pixel_id` observation fields are replaced when that previous linkage explicitly declares both field names; malformed corresponding declarations fail. Otherwise reserved `array_A`, `array_B`, `sample`, `region`, and `pixel_id` fields must be absent or semantically equal to the generated values. Conflicting preserved metadata fails instead of being overwritten. Other observation/feature metadata and aligned arrays are retained, including their source dtypes, missingness, and categorical order. Named annotation columns accept arbitrary order and additional fields; duplicate column names fail. Generated grid identifiers reach the output as NumPy int64, `sample` as pandas strings, and `region` as categorical strings. `X` and an existing `counts` layer become finite, canonical SciPy CSR sparse arrays; other layers retain their representation. Duplicate sparse entries are summed while retaining the source dtype. If an integer sum would overflow that dtype, reading fails: write the source counts with a wider integer dtype before retrying. The reader owns its newly read table and does not modify the source H5AD.

A is the x axis and B is the y axis. Without an image, grid cell `(A, B)` occupies `[A - 1, A] × [B - 1, B]` before the border inset. `include_border=True` shrinks each edge inward by `0.125 * border_scale` grid units. `border_scale` must be finite and satisfy `0 <= border_scale < 4`, including when `include_border=False`.

With an image, both grid axes and the border inset are multiplied by `max(image_height, image_width) / N`. This uniform scaling preserves the historical square-grid convention for rectangular images too: the grid spans the longer image dimension on both axes. The image must already be cropped, oriented, and registered appropriately; the reader does not infer a registration or physical units. Shapes and images use identity transformations into the `global` coordinate system.

Without an image these are barcode-grid units; with an image they use the documented image-pixel convention. OME spacing or nominal channel width alone cannot establish the barcode pitch, origin, orientation and capture footprint, so neither triggers automatic conversion to micrometres.

`invalid_geometries="keep"`, `"drop"` and `"raise"` produce identical supported squares and linkage because this geometry is valid by construction. No topology scan is performed. The option is validated before I/O, and floating-point collapse of a positive square width fails in every mode.

`table.obsm["spatial"]` retains the one-based integer `(array_A, array_B)` coordinates, even when an image scales the polygons. These values are barcode positions, not polygon centers or image-pixel coordinates.

## Image options and lazy execution

TIFF metadata supplies source axes when they are unambiguous. Use `data_axes`, such as `"CYX"` or `"YXC"`, when an explicit interpretation is required. `tiff_series=0` and `tiff_level=0` select the first TIFF series and its first resolution level; other nonnegative indices select an existing series or pyramid level. PNG and JPEG use their encoded image layout and do not accept TIFF-specific selection or axis overrides.

Images preserve numeric channel coordinates for ordinary RGB input. Reliable scientific channel names from OME metadata are retained in source order.

`scale_factors` and `chunks` follow the shared [raster options](raster_options.md): by default a single scale and images in `(1, 1024, 1024)` chunks. A pyramid is generated only when `scale_factors` requests one; the grid scale always uses the full-resolution extent of the selected TIFF level. Image options other than their defaults require an image. Metadata inspection is eager, while pixel decoding stays lazy until the returned image is computed or written. TIFF tasks decode the tiles or strips that overlap the computed chunks. PNG and JPEG can require decoding the entire encoded frame even for a small selection; rechunking does not change that codec limitation.

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
    tiff_series=0,
    tiff_level=0,
    scale_factors=(2, 2),
    chunks=(1, 512, 512),
)
sdata.write("sample.zarr")
```

The CLI uses the same option names in kebab case. Repeat `--scale-factors` once per generated level, and `--chunks` once for a single size or three times in CYX order:

```bash
spatialdata_io dbit --input dataset --output sample.zarr \
  --table-file sample.h5ad --barcode-positions-file barcode_list.txt \
  --image-file registered.tif --no-include-border \
  --scale-factors 2 --scale-factors 2 --chunks 1 --chunks 512 --chunks 512
```

`--include-border/--no-include-border` is a boolean flag pair. The previous `anndata_path`, `barcode_position`, `image_path`, and `border` arguments and their CLI options have been removed.

Images are now chunked `(1, 1024, 1024)` by default instead of keeping the storage-aligned chunks of the source, and `chunks=None` is no longer accepted. Pass `chunks` explicitly to choose another layout. The new `scale_factors` option builds pyramid levels.

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
    np.testing.assert_allclose(sdata.shapes["sample"].total_bounds, [0.625, 0.625, 9.375, 9.375])
    image = sdata.images["sample_image"]
    assert image.dims == ("c", "y", "x")
    np.testing.assert_array_equal(image.data.compute(), pixels.transpose(2, 0, 1))
```

A returned lazy image remains backed by its source path: keep the source files available until computing or writing it. Reader provenance records the reader name and package version without source paths; this unversioned layout does not supply a format version.
