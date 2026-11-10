# MCMICRO

{func}`spatialdata_io.mcmicro` reads the outputs of one [MCMICRO](https://mcmicro.org/) whole-slide or tissue-microarray (TMA) run. It returns the registered image, Coreograph cores and tissue masks, segmentation labels and one linked MCQuant cell table. Pipeline inputs (`raw/`, `illumination/`), Coreograph QC files and segmentation puncta tables are not read.

## Run layouts

MCMICRO writes `qc/params.yml` as a complete dump of the parameters used. Its YAML boolean `workflow.tma` selects the layout; any other value, a missing or duplicated consumed `workflow`/`tma` key, or malformed YAML fails. The entire YAML document is composed into an inert node graph, but only `workflow.tma` is interpreted; unrelated metadata, including duplicate keys, is not recursively validated. A whole-slide run contains:

```text
run/
├── markers.csv
├── qc/params.yml
├── registration/<image>.ome.tif
├── segmentation/<method>-<image>/<mask>.ome.tif
└── quantification/<image>--<method>_<mask>.csv
```

A TMA run adds numbered Coreograph cores, each with one tissue mask:

```text
run/
├── markers.csv
├── qc/params.yml
├── registration/<tma>.ome.tif
├── dearray/<core>.ome.tif
├── dearray/masks/<core>_mask.tif
├── segmentation/<method>-<core>/<mask>.ome.tif
└── quantification/<core>--<method>_<mask>.csv
```

Registration must contain exactly one TIFF-family image; its image ID drops the extension and an `.ome` suffix. Cores are positive integers and are ordered numerically. A mask stem is the label file name up to its first dot, as MCQuant names it, so `cell_mask_ring.ome.tif` has the stem `cell_mask_ring`. Every quantification table must name exactly one discovered label: `<image>--<method>_<mask>.csv` is compared with whole method and mask names, so underscores are never split. Labels without a quantification table are returned without table rows.

A historical `segmentation/<image>/` directory, without the method prefix, is accepted when exactly one quantification table identifies the method of each of its masks. Segmentation and quantification are optional. Dataset-derived artifact paths, including present optional output directories and selected files, must resolve inside the dataset root unless `path_policy="relaxed"` ([Dataset paths and symlinks](dataset_paths.md)). In-root symlinks are accepted; escaping symlinks under the default policy, dangling symlinks and wrong-kind artifacts fail. Image IDs, methods and mask names come from file names. One that is not a valid SpatialData element name (letters, digits, `_`, `-` and `.`) is sanitized with a `UserWarning`, for example `my slide.ome.tif` gives `my_slide_image`. All element names must be unique ignoring letter case, as SpatialData compares them; a layout whose names would collide, also after sanitizing, fails before any image is opened.

## Elements and coordinates

| Element | Contract |
| --- | --- |
| `images["<image>_image"]` | Whole-slide registered image. |
| `images["tma_map"]` | TMA registered image. |
| `images["core_<n>_image"]` | Coreograph core. |
| `labels["core_<n>_mask"]` | Coreograph tissue mask on its own grid. |
| `labels["<image>_<method>_<mask>"]` | Segmentation labels on the grid and in the frame of their image (`core_<n>_<method>_<mask>` for cores). |
| `tables["table"]` | One table linked to every label with quantified cells. |

Images are lazy `(c, y, x)` arrays in source dtype. Their channels are the canonical MCQuant marker IDs in `markers.csv` row order: every occurrence of a repeated marker is suffixed, so `DAPI, CD3, DAPI` becomes `DAPI_1, CD3, DAPI_2`, and generated collisions such as `X, X, X_1` fail. The registered image and every core must have exactly one channel per marker. Labels are lazy integer `(y, x)` arrays that must match the spatial shape of their source image; masks keep their own, possibly downsampled, grid.

Every raster keeps its native origin, the corner of pixel `(0, 0)`. Coreograph exports no authoritative crop origins and masks may be downsampled, so no registration between map, cores and tissue masks is inferred. A raster with complete base OME X/Y physical sizes in µm, nm or mm is scaled into a micrometre coordinate system; otherwise it stays in source pixels with an identity into a pixel coordinate system. A coordinate-system name therefore never means micrometres in one run and pixels in another:

| Raster | Calibrated (µm) | Uncalibrated (pixels) |
| --- | --- | --- |
| Whole-slide image | `global` | `image_pixels` |
| TMA map | `global` | `tma_map_pixels` |
| Core `<n>` | `core_<n>` | `core_<n>_pixels` |
| Tissue mask of core `<n>` | `core_<n>_mask` | `core_<n>_mask_pixels` |

Missing or incomplete sizes keep the raster in pixels. Present invalid sizes or unsupported units fail. Anisotropic spacing is preserved. Segmentations share the transform and coordinate system of their image: their own sizes may be absent, but present sizes that conflict with the image's fail, and they cannot complete a partial image declaration. Independent tissue masks use only their own complete OME metadata. ASHLAR may write its own fallback spacing when acquisition metadata was unavailable: declared OME sizes are used as source metadata, not certified acquisition measurements.

## Table

`markers.csv` needs a `marker_name` column. Every other column is optional module metadata and is kept as exact text in `var`, together with the original `marker_name`. `var_names` are the canonical marker IDs.

Quantification tables need `CellID`, one column per canonical marker and `X_centroid`/`Y_centroid`. Canonical marker IDs cannot also name the required `CellID`, `X_centroid` or `Y_centroid` fields; roles are checked after repeated-marker suffixing. An image-only panel may use those names. The reader-owned `region` column is also reserved. All tables of a run must have the same columns. Marker columns become eager, finite, canonical CSR float64 expression. Centroids and morphology are also parsed directly to float64 with Arrow. `CellID` holds exact positive uint64 label values, unique within each label and representable in that label's dtype; zero is background, and label pixels are never read to check membership. A table with int64 instance IDs, such as `spatialdata.aggregate` output or a store written by an earlier release, does not concatenate with this table: NumPy promotes uint64 and int64 IDs to float64, which SpatialData rejects as instance IDs. MCQuant morphology columns (`Area`, `MajorAxisLength`, `MinorAxisLength`, `Eccentricity`, `Solidity`, `Extent`, `Orientation`, `column_centroid`, `row_centroid`) are float64 with empty fields as missing values. All other columns, including suffixed intensity properties, are kept as exact text.

Column names of `obs` (from the quantification header) and `var` (from the `markers.csv` header) that a SpatialData table does not accept are sanitized with one `UserWarning` per table part, which lists every original and sanitized name: characters other than letters, digits, `_`, `-` and `.` become `_` (`CD45 RO_intensity_max` gives `CD45_RO_intensity_max`), a leading `__` becomes `_`, and `_index` becomes `index`. A name that then equals another name ignoring letter case, or a source name differing only in case from an earlier one (`area` beside `Area`), gets the first free `_1`, `_2`, ... suffix. Valid names, `region` and `CellID` keep their spelling.

Rows follow label order and then source row order. Observation names are row positions `"0"`, `"1"`, ...; `region` names the label image and `CellID` is the instance key. `obsm["spatial_pixels"]` preserves the unchanged float64 centroid XY values. MCQuant obtains these directly from regionprops zero-based pixel-index centres. Their native frames are named `image_regionprops_indices` (whole slide) or `core_<n>_regionprops_indices`, independent of calibration and distinct from the raster's pixel-corner frames `image_pixels` and `core_<n>_pixels`: a native coordinate `x` is the pixel-frame coordinate `x + 0.5`.

When every source image represented by measured rows has complete XY calibration, `obsm["spatial"]` contains float64 XY micrometres in the micrometre coordinate system of the corresponding raster (`global` or `core_<n>`), calculated as `(source_xy + 0.5) * spacing_xy_um`. Nonfinite physical results and nonzero shifted pixel coordinates that become zero during scaling fail. A shifted coordinate of zero remains valid. For example, the source single-pixel centroid `(2, 1)` at spacing `(2, 3)` µm/pixel becomes `(5, 4.5)` µm. Several segmentations of one image share that frame. Independent cores remain independent, with categorical `obs["spatial_frame"]`; native positions use categorical `obs["spatial_pixel_frame"]` when multiple frames occur. A single represented frame is declared directly without a constant observation column. Missing, partial or mixed calibration omits physical `spatial` for the whole aggregate, retaining every row and native position. Unmeasured images and unrelated tissue masks do not decide table calibration; segmentation metadata cannot complete a partially calibrated source image.

`uns["spatialdata_io_positions"]` records version-1 position descriptions, units, frames and calibration evidence. Existing columns that collide with generated frame columns fail. Positions are snapshots: later element registration and ordinary AnnData concatenation do not update or register these arrays. Re-export positions explicitly after registration.

Area retains pixel², MajorAxisLength/MinorAxisLength and column_centroid/row_centroid retain pixels, and Eccentricity/Solidity/Extent are dimensionless. Orientation retains radians from the row axis to the major axis, counter-clockwise in the range −π/2 to π/2, following [MCQuant’s direct regionprops mapping](https://github.com/labsyspharm/quantification/blob/61a1f539665864d7a2896c6754d90de404c16759/mcquant/SingleCellDataExtraction.py#L47-L76) and [scikit-image’s convention](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops); concatenating cores does not register their independent physical frames. A run without quantification returns no table.

## Options and execution

`scale_factors` and `chunks` follow the shared [raster options](raster_options.md): by default a single scale, images in `(1, 1024, 1024)` chunks and labels in `(1024, 1024)` chunks. Generated label pyramids require labels of at most 32 bits; 64-bit labels need the default single scale.

Construction reads TIFF metadata only and executes no pixel tasks. Raster decoding uses native tiles or source bands with a 16 MiB decoded-data target; one native strip can exceed that target. Output rechunking can fetch adjacent source bands. A crop's output size therefore does not bound bytes decoded, read or allocated. The source files must remain available for later raster computation.

Arrow parses quantification CSV with explicit types in 1 MiB source blocks, with internal parser threads disabled. Headers must fit Arrow's header parser block; the source header line is read eagerly. Data records should fit one block to be supported regardless of alignment; longer records may parse when aligned favorably, but records spanning two block boundaries fail with artifact context. Duplicate headers are rejected and every record must have the full header width. Its permissive quote handling is retained: for example, a quote inside an unquoted field is literal text, and text following a closing quote can be appended to that field. This is not strict validation of all CSV quoting syntax.

A required numeric marker needs at least a digit and a separator in CSV and eight bytes as float64. A native batch can include the preceding partial record and the current block, so the required marker value payload is bounded by eight times the 1 MiB source target. Arrow allocation alignment, column metadata, parser/string buffers, the row-major dense tensor, retained annotations and IDs, CSR blocks and final sparse assembly consume memory separately; the source block target is not a total-memory limit. Expression becomes sparse after each native batch. Final linkage rejects duplicate `(region, CellID)` pairs, including duplicates across parser batches. The same ID may occur in different regions.

Arrow's typed numeric CSV grammar is the contract: spaces and tabs around numbers are trimmed; uint64 integers accept hexadecimal such as `0x10`, reject `2.0`, `2e0`, `+1` and `1_0`, and are range-checked before label-dtype validation. Floats round directly to float64, so `1e-400` becomes zero and a large decimal may round to its nearest representable value. `nan` and `inf` parse but fail the finite-value checks. Only empty optional morphology fields become missing; `NA` and `NULL` are not numeric null spellings. IDs, marker intensities and both centroids must be nonempty. Every string field keeps its exact spelling, including empty strings and missing-value words. Negative finite marker intensities and signed morphology such as Orientation remain valid.

## Python and command line

```python
from spatialdata_io import mcmicro

sdata = mcmicro("run", scale_factors=(2, 2))
sdata.write("run.zarr")
```

The CLI mirrors the reader with repeatable pyramid and chunk options:

```bash
spatialdata_io mcmicro --input run --output run.zarr --scale-factors 2 --scale-factors 2
```

The former `imread_kwargs`, `image_models_kwargs` and `labels_models_kwargs` mappings have been removed; use `scale_factors` and `chunks`. The default chunks are `(1, 1024, 1024)` for images and `(1024, 1024)` for labels instead of storage-aligned chunks, and `scale_factors=()` replaces `None` for a single scale. Compared with earlier releases, the reader no longer returns raw or illumination images, includes the method in label names, keeps every core and mask in its exact local frame instead of placing it by centroids, names masks `core_<n>_mask`, returns one table named `table` with positional observation names, and stores expression as sparse arrays. Native pixel centroids now use `spatial_pixels`; canonical `spatial` is reserved for completely calibrated physical positions. Uncalibrated rasters are no longer placed in `global` (or `core_<n>`, `core_<n>_mask`) with an identity, which made the unit of those coordinate systems depend on the file: they now use the pixel coordinate systems above, and the whole-slide native centroid frame is `image_regionprops_indices` instead of `global_regionprops_indices`. Error contexts name artifacts by role (`markers`, `workflow parameters`, `quantification`) instead of by file name. `CellID` now uses uint64 through linkage and persistence, including IDs above `2**63 - 1` when the label dtype supports them. Numeric CSV fields follow Arrow's grammar above: hexadecimal integer tokens are accepted, decimal/exponent integer spellings are rejected, and floats use native float64 rounding and underflow. These replace the former reader-specific lexical numeric checks.
