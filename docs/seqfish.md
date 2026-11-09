# seqFISH

{func}`spatialdata_io.seqfish` reads the repeated Spatial Genomics ROI export layout below. Each ROI produces an expression table and the requested images, segmentation, transcript points and cell geometry. Support describes this executable layout and its column contracts; no vendor release number is inferred from a directory name.

## Dataset layout

```text
dataset/
├── Roi1_CellxGene.csv          # required expression counts, first header empty
├── Roi1_CellCoordinates.csv    # required label, area, center_x, center_y
├── Roi1_DAPI.tiff              # OME physical size metadata; lazy image
├── Roi1_Segmentation.tiff      # lazy integer segmentation labels
├── Roi1_TranscriptList.csv     # name, x, y; optional z and cell
├── Roi1_Boundaries.geojson     # Polygon features with explicit top-level id
├── Roi2_CellxGene.csv
├── Roi2_CellCoordinates.csv
└── ...
```

ROI names are the file-name prefixes before `_CellxGene.csv` or `_CellCoordinates.csv`; macOS AppleDouble files such as `._Roi1_CellxGene.csv` are ignored. By default every ROI is read, in lexical prefix order. A nonempty, unique `rois` sequence of exact prefixes selects ROIs in the caller's order; a prefix without files fails as an unknown ROI. Each selected ROI requires both the expression and cell-coordinate files. Requested products must exist; disabled products need not exist unless their bounded metadata calibrates another requested product. In particular, table-only loading and circles need no DAPI, segmentation, transcript or boundary file. Pixel-based images, labels and polygons require DAPI physical-size metadata. Under the default `path_policy="strict"`, discovered artifacts, including symlink targets, must remain within the dataset root; see [dataset paths](dataset_paths.md) for `"relaxed"`.

The ROI name starts every element name and, with several ROIs, names the ROI's coordinate system. A prefix that is not a valid SpatialData element name is repaired once with a warning: characters other than letters, digits, `_`, `-` and `.` become `_`, a leading `__` becomes `_`, and an empty prefix becomes `unnamed` (`Roi 1` is read as `Roi_1`). The files keep their names; select such a ROI with its prefix, `rois=["Roi 1"]`. Selected ROIs whose names are equal ignoring letter case after this repair, such as `Roi1` and `roi1`, fail before any table is read, because SpatialData compares element names ignoring case; select one of them with `rois`.

The first CellxGene field contains positive cell identifiers; the remaining nonempty, unique headers identify expression features in their source order. CellCoordinates provides unique matching `label` values, finite `center_x` and `center_y`, and finite nonnegative `area`. Its rows may be reordered: alignment follows exact cell IDs, preserving the expression row order. Zero area is accepted for the table and polygon output; centroid circles require strictly positive areas.

Arrow parses every CSV field directly into its declared type:

| File | Field | Type |
| --- | --- | --- |
| CellxGene | first field (cell ID) | uint64, positive |
| CellxGene | every feature | uint64 count |
| CellCoordinates | `label` | uint64, positive |
| CellCoordinates | `area`, `center_x`, `center_y` | float64, finite |
| TranscriptList | `name` | text, a CellxGene feature |
| TranscriptList | `x`, `y` | float64, finite |
| TranscriptList | `z` (optional) | uint64, nonnegative |
| TranscriptList | `cell` (optional) | uint64, `0` unassigned |

Arrow's CSV grammar is the contract. Spaces and tabs around a number are trimmed, and integers may be written in hexadecimal (`0x10` reads as 16). An integer field rejects `2.0`, `2e0`, `+2`, `1_0` and values outside its type, so positive cell IDs range from 1 through 2⁶⁴ − 1. Floats are correctly rounded: `1e-400` reads as 0 and `9007199254740993` as `9007199254740992.0`. Numeric fields read by the reader cannot be empty. Floating fields accept `nan` and `inf` during parsing and then fail the finiteness check; integer fields reject these spellings. Ignored annotation fields may be empty. Duplicate header names fail before parsing.

GeoJSON is an eagerly parsed UTF-8 FeatureCollection. It contains polygon features with explicit top-level Feature `id` values corresponding to cell identifiers; Arrow casts each ID, a JSON string or integer, to the uint64 cell-ID type (`"01"` and `"0x10"` are accepted, `" 1"` and `"1.0"` are not). Property fields cannot substitute for these source identifiers. Properties, including `properties.id`, are independent annotations and do not need to match the top-level ID ([GeoJSON feature semantics](https://www.rfc-editor.org/rfc/rfc7946#section-3.2)). Property columns retain JSON values as objects, avoiding numerical inference of annotation IDs. The reserved property name `geometry` is rejected so annotations cannot replace the source geometry. Other source property annotations are retained. Features may be reordered. Integer polygon vertices must be exactly representable in float64. Missing, duplicate or mismatched IDs and malformed geometry fail in every policy; geometry is neither repaired nor assigned an ID by row position. Invalid topology follows `invalid_geometries`.

## Output and coordinates

A single selected ROI uses `global` in micrometres. When multiple independent ROIs are selected, each retains its ROI name as its frame name, also in micrometres; matching units do not establish registration. Element names start with the ROI name in either case. Every element of a ROI is calibrated, so no pixel-only frame exists: the vendor centroids, areas and transcript coordinates are micrometres, and the pixel-based DAPI, segmentation and polygons require the DAPI OME physical pixel size. Without that calibration a read that requests them fails instead of returning pixels. The returned elements are:

| Element | Contract |
| --- | --- |
| `tables["table_<roi>"]` | Eager owned AnnData; canonical SciPy CSR sparse-array counts, source feature and expression-row order. |
| `images["<roi>_DAPI"]` | Lazy `(c, y, x)` image retaining pixel dtype and channel order. |
| `labels["<roi>_Segmentation"]` | Lazy `(y, x)` integer labels retaining source pixel dtype. |
| `points["<roi>_TranscriptList"]` | Lazy projected transcript events with physical `x`, `y`, categorical `name`, optional `cell` and source slice annotation `z_raw`. |
| `shapes["<roi>_Boundaries"]` | Eager polygons indexed by explicit source IDs, retaining GeoJSON source annotations. |
| `shapes["<roi>_CellCoordinates"]` | Eager centroid circles indexed by cell IDs, with radius `sqrt(area / pi)`. |

The table retains `area`, uint64 `instance_id`, and `(center_x, center_y)` in `obsm["spatial"]`. Observation names are original positional strings starting at `"0"`, retaining gaps after geometry drops; biological identity comes from `instance_id`. When labels are included, categorical `region` and SpatialData table metadata explicitly link the table to `<roi>_Segmentation`. With labels excluded, the table is normalized and unlinked. Including polygons or circles does not change its annotation target.

DAPI OME metadata calibrates pixel X and Y independently into the ROI physical coordinate frame. The transform assigns PhysicalSizeX to X and PhysicalSizeY to Y, including anisotropic pixels. The same pixel-to-physical transform applies to segmentation and GeoJSON polygons. Segmentation OME physical sizes are optional: each size it declares must equal the DAPI one, and a size it does not declare is inherited from DAPI (a partial segmentation declaration previously failed). Cell centroids and transcript X/Y already use the physical ROI frame and have identity transforms. Area supplies square physical units for circle radii. OME micrometers, nanometers and millimeters are converted to micrometers in the ROI frame; absent OME unit attributes use the OME micrometer default. Unsupported units, missing calibration and inconsistent metadata fail.

When polygons are requested with both images and labels excluded, DAPI is inspected only for calibration; no DAPI pixel graph is constructed. This mode still requires a valid TIFF with exactly one series and one unambiguous OME image with finite positive physical X/Y sizes in supported units. Raster CYX normalization and channel naming apply only when images or labels are requested, so unrelated nonspatial-axis or channel metadata does not restrict calibration-only loading.

Transcript `z` is retained as `uint64[pyarrow]` `z_raw`, preserving the nonnegative image-plane index exactly. The export does not establish a physical calibration for this slice index, so transcript points have two spatial coordinates, X and Y. Optional `cell` values preserve source assignments, including unassigned zero values, in `uint64[pyarrow]` storage in every partition. This stable dtype preserves exact assignments through `2**64 - 1` even when partitions contain different identifier ranges. The complete CellxGene feature header supplies the transcript feature vocabulary; unknown transcript names fail when their partition is computed. Transcript name categories follow the CellxGene feature header order, including zero-count features.

Table `uns["spatialdata_io_positions"]` describes unchanged float64 vendor cell-centroid XY micrometres: `global` for one ROI and each ROI name for multiple independent ROIs. These positions do not derive from polygon centroids.

## Options and execution

`include_images`, `include_labels`, `include_transcripts` and `include_shapes` default to `True`. `shape_geometry="polygons"` is the default; `"circles"` or `"both"` selects centroid circles or both geometry types. `include_shapes=False` excludes both.

`invalid_geometries="keep"` preserves invalid polygon topology by default. `"raise"` reports invalid cell IDs. `"drop"` removes the affected polygon, its requested centroid circle and its expression-table row, preserving surviving IDs, row order, sparse values and annotations. Complete boundary/table identity agreement is validated before dropping. Labels and lazy transcript assignments remain unchanged. A requested boundary element becoming empty fails the whole read. Circles-only and shapes-disabled modes never read boundaries or drop cells.

`scale_factors` and `chunks` follow the shared [raster options](raster_options.md): by default a single scale, images in `(1, 1024, 1024)` chunks and labels in `(1024, 1024)` chunks. Computing a region decodes the native TIFF tiles or strip bands needed by the output chunks it intersects. Strips are grouped toward a 16 MiB decoded-band target; an indivisible native strip can exceed that target. An output chunk can depend on neighboring native tiles or bands, so 16 MiB is not a universal limit on decoded bytes per request. A sequence such as `(2, 2)` builds image and label pyramids with the appropriate interpolation for each product. Generated label pyramids require segmentation dtypes of at most 32 bits, avoiding upstream integer rounding. Full-resolution source series and level remain fixed, so alternate scenes and uncalibrated source pyramid levels cannot be selected accidentally.

Expression parsing uses bounded batches and preserves sparse counts without allocating a full dense expression table. Metadata, tables and shapes are eager. Pixels and transcript rows remain lazy through construction of the returned SpatialData. TIFF tasks reopen their path-backed source when computed and close their resources; retain the input files until computation or writing completes.

Provide valid CSV inputs. Arrow checks complete record widths but does not enforce strict quoting, so some malformed quote forms may be accepted. CellxGene and CellCoordinates fields may contain quoted newlines. Counts are read in batches of about 1 MiB of text, so a single CellxGene record must be shorter than that.

Transcript CSVs require one physical header line and one physical line per record; quoted commas and quotes are supported, while embedded newlines in quoted fields are unsupported. Every record must contain exactly the header field count, including projected-out fields. Transcript graph construction reads only the CSV header. Validation of unread transcript rows is deferred: malformed records or values, nonfinite coordinates, negative `z` planes and unknown features raise with source context when the affected partition is computed. The Dask index is a storage index local to each partition and may repeat; it is not a biological event identifier. Source row order is preserved without a global shuffle. Unsupported compressed transcript variants are not advertised as a bounded source format.

The reader owns newly created tables and annotations and does not mutate caller containers or source files. Provenance records the reader and package version without absolute input paths; the export has no version marker, so no format version is recorded. Missing or wrong-kind paths raise the corresponding filesystem exception. Malformed schemas, ambiguous layouts and inconsistent scientific relationships raise `ReaderFormatError`, a `ValueError` subclass naming the artifact (`CellxGene`, `CellCoordinates`, `DAPI`, `Segmentation`, `TranscriptList`, `Boundaries`) and its path. All options are validated in signature order before the directory is read: invalid option values raise `ValueError`; invalid types raise `TypeError`. Validation errors do not return partial results.

## Python and command line

```python
from spatialdata_io import seqfish

sdata = seqfish(
    "dataset",
    rois=("Roi2", "Roi1"),
    shape_geometry="both",
    chunks=(1, 512, 512),
)
sdata.write("sample.zarr")
```

The CLI uses paired inclusion flags and repeatable ROI, chunk and pyramid options, in the order of the reader's parameters:

```bash
python -m spatialdata_io seqfish --input dataset --output sample.zarr \
  --shape-geometry both --rois Roi2 --rois Roi1 \
  --chunks 1 --chunks 512 --chunks 512

python -m spatialdata_io seqfish --input dataset --output counts.zarr \
  --no-include-images --no-include-labels \
  --no-include-transcripts --no-include-shapes
```

`rois` accepts any exact file prefix, including prefixes such as `Roi 1` that the reader repairs into element names; previously such prefixes produced element names that SpatialData rejected after all files had been read. The former `load_*`, `load_points`, `cells_as_circles`, `imread_kwargs` and `raster_models_scale_factors` options have been removed. Use the explicit inclusion flags, geometry choice, `chunks` and `scale_factors`. Automatic pyramids based on image size have been removed; request a pyramid explicitly when needed. The default chunks are `(1, 1024, 1024)` for images and `(1024, 1024)` for labels instead of storage-aligned chunks, and `scale_factors=()` replaces `None` for a single scale.

Completed counts use float32 when every value is exactly representable, otherwise
float64. Values that neither floating dtype can preserve raise a contextual
precision error. Completed bounded blocks are converted before sparse assembly;
expression is never densified.

CSV fields now follow Arrow's typed CSV grammar instead of exact decimal decoding:
integer fields reject `2.0` and `2e0`, accept hexadecimal and surrounding spaces,
and cell IDs (table, CellCoordinates and Boundaries) use fixed positive uint64
through 2⁶⁴ − 1. Coordinates no longer fail for underflow to
zero or for integral values above 2⁵³ that float64 rounds. Transcript `cell`
assignments keep the full uint64 range. Shape indexes are Arrow-backed
`uint64[pyarrow]`. Table `instance_id` becomes native uint64
at the AnnData storage boundary, preserving full-range IDs through linkage and
H5AD/Zarr writes. Shape indexes restored by SpatialData from GeoParquet use
native uint64 and preserve the same IDs.
