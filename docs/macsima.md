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

- TIFFs are found recursively by a case-insensitive `.tif` or `.tiff` suffix and ordered by path.
- Paths beginning with `.` and `__MACOSX` are ignored, so macOS archive metadata is never read.
- A TIFF without OME-XML is not a channel, and a subfolder without TIFFs is not a stack; both are skipped with a warning.
- `exclude_folder_names` omits subfolders by their exact names, each of which must exist; it is invalid in single-folder mode.
- File names are never interpreted, and renamed files remain readable: an OME `TiffData` UUID equal to the file's own UUID refers to the file itself.

Each stack is named after its folder, made a valid SpatialData element name (`my run` becomes `my_run`). Folder names that collide after this conversion, ignoring case, fail before any TIFF is read.

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

The well is the external identifier of the OME plate well; a file may record at most one well and, for software major 0, at most one screen reagent. Acquisition types become `stain`, `bleach` or `autofluorescence`; bleach images append `_background` to the marker name. The marker name, cycle and acquisition type are required. Cycles and ROI IDs are non-negative decimal integers (`001` is ROI 1), and exposure times are finite, non-negative milliseconds. A key the reader uses must not carry different values in different annotation groups; both version keys may be present when they name the same version, with or without a leading `v`. Unsupported software majors, units or values fail with the reader, artifact and file path, and the detected software version once it is known.

## Elements and coordinates

| Element | Contract |
| --- | --- |
| `images["<stack>_image"]` | Lazy `(c, y, x)` channel stack in source dtype. |
| `tables["<stack>_table"]` | Metadata of the channels of `<stack>_image`. |
| `images["<stack>_nuclei_image"]` | Optional stack of nuclear-stain channels. |
| `tables["<stack>_nuclei_table"]` | Metadata of the channels of `<stack>_nuclei_image`. |

Channels are ordered by cycle, then by the `name` column (the marker, with `_background` for bleach images), then by file path; `exclude_cycles` removes cycles first. A channel is named after its marker, or `R<cycle> <marker>` with `include_cycle_in_channel_names=True`. Repeated names are numbered in channel order, for example `DAPI (1)` and `DAPI (2)`; names are assigned once per stack, so a channel has the same name in both images, and generated names that collide with a real marker fail.

When a stack contains more stain channels named exactly `nuclei_channel_name` than `split_threshold_nuclei_channel`, all of them form the nuclei stack and the main stack keeps only the first and the last of them: the registration reference and the end-of-run image. `split_threshold_nuclei_channel=None` never splits. Bleach images such as `DAPI_background` are never nuclear channels.

The channels of a stack must share one dtype, one pixel size, one software major, and one well and ROI where these are recorded. Each channel is placed on a common pixel grid by its OME plane position, which must be an integral number of pixels; absent positions mean the origin. The canvas starts at the smallest plane position and spans every channel; uncovered pixels are zero, and padding emits a warning. A position shared by all channels is therefore not represented in the transformation.

Each stack has its own coordinate system `<stack>`. With `include_transformations=True` it is in micrometers, through a scale by the isotropic OME pixel size (micrometers or nanometers). With `False` the transformation is the identity, so the same coordinate system is the pixel grid.

## Table

Each table has no observations, `X=None` and one `var` row per image channel, with `var_names` equal to the channel names. It annotates no element.

| Column | Type | Content |
| --- | --- | --- |
| `name` | string | Marker name; bleach images end in `_background`. |
| `cycle` | int64 | Acquisition cycle. |
| `imagetype` | string | `stain`, `bleach` or `autofluorescence`. |
| `well` | string | Plate well, or missing. |
| `ROI` | nullable integer | ROI number, or missing. |
| `fluorophore` | string | Fluorophore, or missing. |
| `clone` | string | Antibody clone, or missing. |
| `exposure` | float64 | Exposure time in milliseconds, or NaN. |

The reader name and, when all returned channels agree, the MACSiQView software version are recorded in `sdata.attrs`.

## Options and execution

`scale_factors` and `chunks` follow the shared [raster options](raster_options.md): by default a single scale and images in `(1, 1024, 1024)` chunks. Vendor-written pyramid levels are not used.

Each TIFF is inspected once; construction reads metadata only and executes no pixel tasks. MACSima exports store one row per TIFF strip, so whole strips are read in bands of at most 16 MiB, and computing a small region decodes only the bands it intersects. The source files must remain available for later raster computation.

## Python and command line

```python
from spatialdata_io import macsima

sdata = macsima("experiment", parsing_style="processed_multiple_folders", scale_factors=(2, 2, 2))
sdata.write("experiment.zarr")
```

The CLI mirrors the reader. Repeat list options, and pass `none` to never split nuclei channels:

```bash
spatialdata_io macsima --input experiment --output experiment.zarr \
    --parsing-style processed_multiple_folders --exclude-cycles 15 --split-threshold-nuclei-channel none
```

## Changes from earlier releases

Renamed and removed options:

- `filter_folder_names` is now `exclude_folder_names` and matches exact folder names.
- `skip_rounds` is now `exclude_cycles`, and `include_cycle_in_channel_name` is now `include_cycle_in_channel_names`.
- `transformations` is now `include_transformations`. Without transformations, the stack keeps its own coordinate system instead of `global`.
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
