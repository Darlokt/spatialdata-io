# MACSima

{func}`spatialdata_io.macsima` reads processed MACSima exports written by MACS iQ View (MACSiQView): one single-plane OME-TIFF per channel. It returns one lazy channel stack and one metadata-only channel table per processed folder, plus an optional separate stack of nuclear-stain channels. Raw instrument data is not supported.

## Folder layouts

With `parsing_style="processed_single_folder"` (the default), every TIFF below the folder forms one stack:

```text
run/
├── <channel>.tif
└── <subfolder>/<channel>.ome.tif
```

With `parsing_style="processed_multiple_folders"`, every immediate subfolder forms one stack; TIFFs directly in the root are not read:

```text
experiment/
├── <stack_a>/<channel>.tif
└── <stack_b>/<channel>.tiff
```

Discovery follows these rules:

- TIFFs are found recursively by a case-insensitive `.tif` or `.tiff` suffix and ordered by path. Directories with such a suffix are ignored; any other match must be a regular file, so a dangling symlink fails with `FileNotFoundError` instead of silently dropping a channel.
- Paths beginning with `.` and `__MACOSX` are ignored, so macOS archive metadata is never read. Hidden folders are not even entered, so an unreadable one (a cache, for example) does not fail the read.
- A TIFF without OME-XML is not a channel, and a subfolder without TIFFs is not a stack; both are skipped with a warning.
- `exclude_folder_names` omits subfolders by their exact names, each of which must exist; it is invalid in single-folder mode.
- File names are never interpreted, and renamed files remain readable: an OME `TiffData` UUID equal to the file's own UUID refers to the file itself.

Each stack is named after its folder. A folder name that is not a valid SpatialData element name is repaired with a warning (`my run` becomes `my_run`); rename the folder to choose the name. Folder names that collide after this repair, ignoring case, fail before any TIFF is read.

The public example exports keep one ROI per folder, with file names such as `C-001_S-000_S_APC_R-01_W-D01_ROI-01_A-CD3_C-REA1151.tif` that spell the cycle, acquisition type, fluorochrome, well, ROI, antigen and clone. The reader takes all of these from the OME metadata instead.

## Channel metadata

Every channel file must contain exactly one OME image with one plane (`SizeC`, `SizeZ` and `SizeT` of one), whose TIFF shape matches the OME `SizeY` and `SizeX`. The MACSiQView software version selects the annotation schema:

| Field | Software major 0 | Software major 1 |
| --- | --- | --- |
| Version | `Software version` | `SoftwareVersion` |
| Marker name | screen reagent name before `__` | `Biomarker`, else `Dye` |
| Clone | screen reagent name after `__` | `Clone` |
| Fluorophore | `Fluorochrome` | `Fluorochrome`, else `Dye` |
| Cycle | `Cycle`, else `MICS cycle ID` | `Cycle` |
| Acquisition type | `MICS cycle type`: `AntigenCycle`, `BleachCycle` | `ScanType`: `S`, `B`, `AF` |
| ROI | `ROI ID` | `RoiId` |
| Exposure | `Exposure time`, `Exposure time unit` | `ExposureTime`, `ExposureTimeUnit` |

The well is the external identifier of the OME plate well; a file may record at most one well and, for software major 0, at most one screen reagent. Acquisition types become `stain`, `bleach` or `autofluorescence`; bleach images append `_background` to the marker name. The marker name, cycle and acquisition type are required. Cycles and ROI IDs are non-negative decimal integers from 0 through 9,223,372,036,854,775,807 (the signed int64 maximum; `001` is ROI 1), and exposure times are finite, non-negative milliseconds. Nonzero exposure text that becomes zero in float64 is rejected, including negative values whose sign would otherwise disappear. A key the reader uses must not carry different values in different annotation groups; both version keys may be present when they name the same version, with or without a leading `v`. Unsupported software majors, units or values, including integers above this bound, fail with `ReaderFormatError` carrying the reader, artifact and file path, and the detected software version once it is known.

## Elements and coordinates

| Element | Contract |
| --- | --- |
| `images["<stack>_image"]` | Lazy `(c, y, x)` channel stack in source dtype. |
| `tables["<stack>_table"]` | Metadata of the channels of `<stack>_image`. |
| `images["<stack>_nuclei_image"]` | Optional stack of nuclear-stain channels. |
| `tables["<stack>_nuclei_table"]` | Metadata of the channels of `<stack>_nuclei_image`. |

Channels are ordered by cycle, then by the `name` column (the marker, with `_background` for bleach images), then by file path; `exclude_cycles` removes cycles first. A channel is named after its marker, or `R<cycle> <marker>` with `include_cycle_in_channel_names=True`. Repeated names are numbered in channel order, for example `DAPI (1)` and `DAPI (2)`; names are assigned once per stack, so a channel has the same name in both images, and generated names that collide with a real marker fail.

When a stack contains more stain channels named exactly `nuclei_channel_name` than `split_threshold_nuclei_channel`, all of them form the nuclei stack and the main stack keeps only the first and the last of them: the registration reference and the end-of-run image. `split_threshold_nuclei_channel=None` never splits. Bleach images such as `DAPI_background` are never nuclear channels.

The channels of a stack must share one dtype, one pixel size, one software major, and one well and ROI where these are recorded. Each channel is placed on a common pixel grid by its OME plane position, which must be an integral number of pixels; absent positions mean the origin. Position integrality is checked on the decoded OME float64 values; original XML precision beyond float64 is not recovered. The canvas starts at the smallest plane position and spans every channel; uncovered pixels are zero, and padding emits a warning. A position shared by all channels is therefore not represented in the transformation.

Every stack is calibrated in micrometres through its square OME pixel size (declared in micrometres or nanometres); a channel without a pixel size, with non-square pixels or with another unit fails. The converted micrometre size must remain finite and strictly positive; conversion underflow to zero fails. One selected stack uses `global`. Multiple stacks are not registered to each other, so each keeps its own micrometre coordinate system named `<stack>`. A split nuclei stack shares its parent frame. Excluding folders can leave one stack, which then uses `global`.

## Table

Each table has no observations, `X=None` and one `var` row per image channel, with `var_names` equal to the channel names. It annotates no element.

| Column | Type | Content |
| --- | --- | --- |
| `name` | string | Marker name; bleach images end in `_background`. |
| `cycle` | int64 | Acquisition cycle from 0 through the signed int64 maximum. |
| `imagetype` | string | `stain`, `bleach` or `autofluorescence`. |
| `well` | string | Plate well, or missing. |
| `ROI` | nullable Int64 | Exact ROI number from 0 through the signed int64 maximum, or missing; integers retain precision alongside missing values. |
| `fluorophore` | string | Fluorophore, or missing. |
| `clone` | string | Antibody clone, or missing. |
| `exposure` | float64 | Exposure time in milliseconds, or NaN. |

The reader name and, when all returned channels agree, the MACSiQView software version are recorded in `sdata.attrs`.

## Options and execution

`scale_factors` and `chunks` follow the shared [raster options](raster_options.md): by default a single scale and images in `(1, 1024, 1024)` chunks. Vendor-written pyramid levels are not used.

Each TIFF is inspected once; construction reads metadata only and executes no pixel tasks. The shared TIFF reader initially follows native tiles or whole-row bands targeting 16 MiB of decoded pixels. A native strip larger than that target is kept whole, so a source with one whole-plane strip can require whole-plane decoding. Output rechunking, including the default chunks, can make a small region depend on neighboring tiles or bands beyond those it directly intersects. The target is not a hard allocation cap. The source files must remain available for later raster computation.

## Python and command line

```python
from spatialdata_io import macsima

sdata = macsima("experiment", parsing_style="processed_multiple_folders", scale_factors=(2, 2, 2))
sdata.write("experiment.zarr")
```

The CLI mirrors the reader. Repeat list options, and pass `none` to never split nuclei channels:

```bash
python -m spatialdata_io macsima --input experiment --output experiment.zarr \
    --parsing-style processed_multiple_folders --exclude-cycles 15 --split-threshold-nuclei-channel none
```

## Changes from earlier releases

Renamed and removed options:

- `filter_folder_names` is now `exclude_folder_names` and matches exact folder names.
- `skip_rounds` is now `exclude_cycles`, which rejects negative cycles, and `include_cycle_in_channel_name` is now `include_cycle_in_channel_names`.
- The `transformations` and `include_transformations` switches are removed. Calibrated physical coordinates are always attached; source pixel arrays remain unchanged.
- `max_chunk_size` and `c_chunks_size` are replaced by `chunks`.
- `multiscale` and `default_scale_factor` are replaced by `scale_factors`, which no longer defaults to an automatic pyramid.
- `scale_factors` defaults to `()` instead of `None` for a single scale; `None` is no longer accepted.
- `subset`, `c_subset`, `imread_kwargs`, the unimplemented `raw` style and `MACSimaParsingStyle` have been removed; slice the returned lazy images instead.

Corrections and contract changes that change results:

- Channels with different plane positions are now aligned even when their shapes are equal.
- Nuclear channels are matched by exact marker name among stain images.
- `split_threshold_nuclei_channel=0` now splits whenever a nuclear stain channel exists; use `None` to never split.
- The `name` column of the channel table holds the marker; numbered or cycle-prefixed channel names remain in `var_names`.
- Multiple-folder results keep their provenance.
- Malformed metadata now fails instead of being skipped or silently becoming missing.
- A dangling `.tif` or `.tiff` symlink now fails instead of being skipped.
- A folder name that is not a valid element name is repaired as before, but now with a warning.
- Discovered artifacts that are symlinks to targets outside the dataset directory are now rejected by default; earlier releases accepted them. Pass `path_policy="relaxed"` for git-annex or DVC datasets ([Dataset paths and symlinks](dataset_paths.md)).
