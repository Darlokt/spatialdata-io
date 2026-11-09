# Visium

{func}`spatialdata_io.visium` reads one single-library classic Visium output of 10x Genomics Space Ranger (`spaceranger count`, 55 µm capture spots). It returns one sparse count table, one circle per spot and the tissue images, all in image pixels. Visium HD output is read by {func}`spatialdata_io.visium_hd` ([Visium HD](visium_hd.md)).

## Supported layouts

Space Ranger 2.0 and later:

```text
outs/
├── filtered_feature_bc_matrix.h5       # preferred count source
├── filtered_feature_bc_matrix/         # MEX alternative
│   ├── matrix.mtx.gz                   # all three gzip, or all three plain
│   ├── barcodes.tsv.gz
│   └── features.tsv.gz
├── raw_feature_bc_matrix.h5            # read only with counts_source
└── spatial/
    ├── tissue_positions.csv            # headered
    ├── scalefactors_json.json
    ├── tissue_hires_image.png          # optional
    └── tissue_lowres_image.png         # optional
```

Space Ranger 1.x writes the same tree with a headerless `spatial/tissue_positions_list.csv`. A `tissue_positions_list.csv` with a header is also accepted, and `tissue_positions.csv` wins when both exist.

10x sample downloads name the matrix after the dataset and ship `spatial/` as a separate archive:

```text
V1_Adult_Mouse_Brain/
├── V1_Adult_Mouse_Brain_filtered_feature_bc_matrix.h5   # or ..._filtered_feature_bc_matrix/
└── spatial/                                            # extracted V1_Adult_Mouse_Brain_spatial.tar.gz
```

Discovery looks only at the dataset root and `spatial/`:

- The standard HDF5 matrix wins over the standard MEX directory, which wins over one prefixed source. A prefixed HDF5 matrix beside a prefixed MEX directory is ambiguous and raises.
- A MEX directory must hold exactly one complete `[<prefix>]matrix.mtx`, `barcodes.tsv`, `features.tsv` triplet, all gzip-compressed or all plain. A matching uncompressed `[<prefix>]genes.tsv` raises, because it would switch the 10x MEX reader to its legacy feature format.
- Other files in `spatial/` (QC images, CytAssist images, enrichment tables) are ignored.
- The full-resolution tissue image is an input of Space Ranger, not part of `outs/`; pass it as `fullres_image_file`.
- `counts_source`, `positions_file`, `scale_factors_file` and `fullres_image_file` replace the discovered artifacts. Relative paths resolve against the dataset root; absolute paths may point elsewhere. `counts_source` is read as HDF5 when its suffix is `.h5` or `.hdf5`, otherwise as a MEX directory, for example `counts_source="raw_feature_bc_matrix.h5"` to keep spots outside the tissue.
- Symlinks follow `path_policy` ([Dataset paths and symlinks](dataset_paths.md)), including the three MEX member files.

The HDF5 matrix's `software_version` attribute (`spaceranger-2.1.0`) gives the format version `2.1.0`, recorded as `attrs["spatialdata_io_format_version"]` and named in later format errors. Other spellings and MEX sources give no format version. The layout does not depend on the version.

## Spot positions and scale factors

The position file has one row per capture spot, under tissue or not:

```text
barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres
ACGCCTGACACGCGCT-1,0,0,0,1234.5,2345.6
```

| Field | Meaning | Rule |
| --- | --- | --- |
| `barcode` | spot barcode | nonempty, unique, no NUL; quoted values may contain commas and newlines |
| `in_tissue` | `1` under tissue, `0` otherwise | uint8, restricted to `0` or `1` |
| `array_row`, `array_col` | spot row and column on the capture array | uint64, including zero through `2**64 - 1` |
| `pxl_row_in_fullres`, `pxl_col_in_fullres` | spot centre `y` and `x` in full-resolution pixels | finite, nonnegative float64; may be fractional |

A header names the fields, so they may come in any order and extra columns are ignored; empty or duplicate names raise. A quoted extra header name may contain a newline. A headerless file has exactly these six fields. The first record is parsed with strict CSV quoting to recognize the header; later records must have the same number of fields, but a malformed quote there may be accepted and read as a different value. Every count barcode needs one position row; position rows without counts are not returned.

The six fields are parsed directly with their Arrow types. Barcode strings remain literal, including leading zeros, whitespace, `NA` and `NULL`; empty or NUL-containing barcodes fail. Numeric fields cannot be empty and have no implicit missing-value spellings. Spaces and tabs around numbers are accepted. Unsigned integers accept hexadecimal (`0x10`) and reject negative values, a leading plus sign, decimal/exponent spellings (`2.0`, `2e0`) and digit separators. Tissue values outside zero and one fail even when they fit uint8. Values outside the declared unsigned width fail conversion. Floats are rounded directly to float64: `9007199254740993` rounds to `9007199254740992` and `1e-400` to zero; negative zero is preserved. Parsed `nan`/`inf` and negative pixel coordinates fail semantic validation. Source number text is not decoded a second time to impose extra precision restrictions.

`scalefactors_json.json` must give finite positive `tissue_hires_scalef` and `tissue_lowres_scalef`, whose inverses must be finite, and `spot_diameter_fullres`, whose half must be a positive float64. The whole object, including `fiducial_diameter_fullres` and any other key, is kept in `uns["spatial"]`.

## Elements

| Element | Type | Content |
| --- | --- | --- |
| `<dataset_id>` | shapes | one circle per table row, indexed by barcode (`spot_id`), radius `spot_diameter_fullres / 2` |
| `<dataset_id>_full_image` | image | `fullres_image_file`, lazy `c, y, x`, source dtype; a pyramid only with `scale_factors` |
| `<dataset_id>_hires_image`, `<dataset_id>_lowres_image` | image | the Space Ranger PNGs that are present, lazy, single-scale |
| `table` | table | counts linked to the circles |

The circle radius is the display size Space Ranger draws, not a measured capture area.

The table has one row per count-matrix barcode in matrix order, including spots outside the tissue when raw counts are read, and every feature type (gene expression, antibody capture, ...) in matrix order:

- `X` is a canonical finite CSR sparse array, float32 when every count is exact in float32, else float64. Duplicate sparse entries are summed.
- `var` holds `gene_ids`, `feature_types` and, when present, `genome`; `var_names` are the feature names, made unique with `-1`, `-2`, ... unless `make_var_names_unique=False`.
- `obs` holds uint8 `in_tissue` and uint64 `array_row`/`array_col`, the barcode again as `spot_id` (string) and categorical `region` (`dataset_id`); `spot_id` and `region` link the table to the circles.
- `obsm["spatial_pixels"]` holds float64 spot centres `x, y` in full-resolution pixels, declared in `uns["spatialdata_io_positions"]`. There is no physical `obsm["spatial"]`.
- `uns["spatial"][dataset_id]` holds `scalefactors` and `metadata` (`chemistry_description`, `software_version` of an HDF5 matrix) as Scanpy stores them, without images.

See also the [table contract](table_contracts.md#visium).

## Images and channels

Every image is two-dimensional `c, y, x` with the source dtype and channel order. Image pixels are read only when computed: TIFF computations decode only the tiles or strips they overlap, PNG and JPEG decode the whole frame. `chunks` applies to every image, `scale_factors` only to the full-resolution image ([raster options](raster_options.md)).

The full-resolution image may be a TIFF, OME-TIFF, BigTIFF (`.tif`, `.tiff`, `.btf`), PNG or JPEG (`.jpg`, `.jpeg`); the suffix is checked before any file is read.

- A TIFF must have one series, or `tiff_series` selects one; its full-resolution level is read.
- Its axes come from the file. `data_axes` overrides them with one case-insensitive string of distinct letters including `X` and `Y`, for example `"CYX"` for a multi-page stack without axis metadata. `S` (samples) counts as a channel axis.
- Interleaved samples become channels, named singleton dimensions are dropped and any other extra axis raises: a `ReaderFormatError` for axes the file declares, a `ValueError` naming the image for `data_axes` that do not fit the file.
- `data_axes`, `tiff_series` and `scale_factors` raise without a suitable `fullres_image_file`, before any file is read.

Channel names follow the colour model the file declares, never the channel count or dtype:

| Source | Channels |
| --- | --- |
| TIFF `PhotometricInterpretation` RGB or JPEG-compressed YCbCr; PNG colour types 2, 3 and 6; three-component JPEG | `r, g, b`, or `r, g, b, a` with declared alpha |
| OME-TIFF whose channel axis is the OME channel axis, or with a single channel | OME channel names when all are nonempty and unique, else OME IDs; `<name or ID>:sample:<k>` for an OME channel with several non-RGB samples |
| anything else, including a `data_axes` reinterpretation of another axis | positions `0, 1, ...` |

A four-channel fluorescence image (for example DAPI, CD3, CD8, PanCK) therefore keeps its OME names or positions and is never labelled `r, g, b, a`. The Space Ranger PNGs are named by their colour type, `r, g, b` for RGB.

## Coordinate systems

| Coordinate system | Unit | Elements by identity | Other elements |
| --- | --- | --- | --- |
| `<dataset_id>_pixels` | full-resolution pixel | circles, full-resolution image | PNGs by `Scale(1 / tissue_*_scalef)` |
| `<dataset_id>_hires_pixels` | hires pixel | hires image | circles by `Scale(tissue_hires_scalef)` |
| `<dataset_id>_lowres_pixels` | lowres pixel | lowres image | circles by `Scale(tissue_lowres_scalef)` |

The PNGs are scaled copies of the full-resolution image with aligned corners, so a pure scale relates the frames. The three names differ in their suffix, so they never collide with each other or with an element name, also after `dataset_id` is repaired. `obsm["spatial_pixels"]` is declared in the `<dataset_id>_pixels` frame. No element uses `global`, which means micrometres in spatialdata-io. Space Ranger estimates `spot_diameter_fullres` from the fiducial registration and a nominal spot size, not from the image's pixel size, and classic Visium output declares no pixel size, so no verifiable micrometre calibration exists. A caller who trusts a pixel size can add a `Scale` into `global` with `spatialdata.transformations.set_transformation`:

```python
from spatialdata.transformations import Scale, get_transformation, set_transformation

microns_per_pixel = 0.7  # your calibration
for element in [*sdata.images.values(), *sdata.shapes.values()]:
    to_pixels = get_transformation(element, to_coordinate_system="sample_pixels")
    set_transformation(
        element,
        to_pixels.compose_with(Scale([microns_per_pixel, microns_per_pixel], axes=("x", "y"))),
        to_coordinate_system="global",
    )
```

## Names

`dataset_id` names the circles, prefixes the image names and prefixes the three coordinate systems and names the table's region. An explicit value must be a valid SpatialData element name (letters, digits, `_`, `-` and `.`; not `.`, `..` or starting with `__`; not `table` in any letter case) and is checked before any file is read. By default it is the HDF5 `library_ids` entry, else the sample-download filename prefix; a MEX directory without a prefix needs an explicit `dataset_id`. A default that is not a valid name is repaired with one warning: other characters become `_` (`V1 Mouse:Brain` gives `V1_Mouse_Brain`), a leading run of underscores becomes one, an empty name becomes `unnamed` and `table` becomes `table_`.

## Examples

```python
from spatialdata_io import visium

sdata = visium("outs")

# Raw counts, the full-resolution image with two pyramid levels, 512-pixel tiles.
sdata = visium(
    "outs",
    counts_source="raw_feature_bc_matrix.h5",
    fullres_image_file="/data/images/tissue.tif",
    scale_factors=(2, 2),
    chunks=(1, 512, 512),
)
```

```bash
spatialdata-io visium --input outs --output sample.zarr \
  --fullres-image-file /data/images/tissue.tif --scale-factors 2 --scale-factors 2
```

## Changes from earlier releases

| Change | Reason |
| --- | --- |
| Position CSV fields use declared Arrow types: uint8 tissue state, uint64 array coordinates and float64 pixels. Native integer grammar and direct float rounding replace text-to-number conversion and source-lexeme precision checks. | Preserve the complete unsigned grid domain and make scalar types explicit at ingestion |
| Coordinate systems are `<dataset_id>_pixels`, `<dataset_id>_hires_pixels` and `<dataset_id>_lowres_pixels` (were `<dataset_id>`, `<dataset_id>_downscaled_hires` and `<dataset_id>_downscaled_lowres`), and the positions declaration names `<dataset_id>_pixels`. Replace `coordinate_systems="<dataset_id>"` with `"<dataset_id>_pixels"` in rendering and query code. | Frame names state their pixel unit; `global` is reserved for micrometres |
| Image channels are `r, g, b(, a)` only when the file declares RGB; OME-TIFF channels use their OME names. Before, every three- or four-channel image was labelled RGB(A). | A fluorescence image was mislabelled RGBA |
| `data_axes` accepts only a string such as `"CYX"` | One form across readers; the CLI already took a string |
| `chunks` no longer accepts a mapping; use one integer or a `(c, y, x)` tuple | One form across readers and the CLI |
| A multi-series TIFF without `tiff_series` raises `ReaderFormatError`; a non-integer `tiff_series` raises `TypeError` | Contextual format errors; type errors for wrong types |
| `data_axes`, `tiff_series` and `scale_factors` without a suitable full-resolution image, and an unsupported image suffix, raise before any file is read | Early option validation |
| An explicit `dataset_id` must be a valid element name; a derived one is repaired with a warning | Invalid names failed late when written |
| `attrs["spatialdata_io_format_version"]` is `2.1.0` rather than `spaceranger-2.1.0`; `uns["spatial"]` keeps the raw text | Same normalization as Visium HD |
| Discovered artifacts that are symlinks to targets outside the dataset directory are rejected by default; pass `path_policy="relaxed"` for git-annex or DVC datasets | Safe default ([Dataset paths and symlinks](dataset_paths.md)) |
