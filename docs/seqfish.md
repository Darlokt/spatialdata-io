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
├── Roi1_Boundaries.geojson     # Polygon features with explicit string id
├── Roi2_CellxGene.csv
├── Roi2_CellCoordinates.csv
└── ...
```

ROI names are the prefixes before `_CellCoordinates.csv`. By default they are discovered in lexical order. A nonempty, unique `rois` sequence selects ROIs in the caller's order. Each selected ROI requires both the expression and cell-coordinate files. Requested products must exist; disabled products need not exist unless their bounded metadata calibrates another requested product. In particular, table-only loading and circles need no DAPI, segmentation, transcript or boundary file. Pixel-based images, labels and polygons require DAPI physical-size metadata. Discovered artifacts, including symlink targets, must remain within the dataset root.

The first CellxGene field contains positive integer cell identifiers; the remaining nonempty, unique headers identify expression features in their source order. Counts must be finite nonnegative integers. CellCoordinates provides unique matching integer `label` values, finite `center_x` and `center_y`, and finite nonnegative `area`. Its rows may be reordered: alignment follows exact cell IDs, preserving the expression row order. IDs retain their integer values, including values above 65535. Zero area is accepted for the table and polygon output; centroid circles require strictly positive areas.

GeoJSON is an eagerly parsed UTF-8 FeatureCollection. It contains polygon features with explicit top-level Feature `id` values corresponding to cell identifiers. Property fields cannot substitute for these source identifiers. Properties, including `properties.id`, are independent annotations and do not need to match the top-level ID ([GeoJSON feature semantics](https://www.rfc-editor.org/rfc/rfc7946#section-3.2)). Property columns retain JSON values as objects, avoiding numerical inference of annotation IDs. The reserved property name `geometry` is rejected so annotations cannot replace the source geometry. Other source property annotations are retained. Features may be reordered. Integer polygon vertices must be exactly representable in float64. Missing, duplicate or mismatched IDs and malformed geometry fail in every policy; geometry is neither repaired nor assigned an ID by row position. Invalid topology follows `invalid_geometries`.

## Output and coordinates

A single selected ROI uses `global` in micrometres. When multiple independent ROIs are selected, each retains its prefix as its frame name; matching units do not establish registration. Element names retain their ROI prefix in either case. The returned elements are:

| Element | Contract |
| --- | --- |
| `tables["table_<roi>"]` | Eager owned AnnData; canonical SciPy CSR sparse-array counts, source feature and expression-row order. |
| `images["<roi>_DAPI"]` | Lazy `(c, y, x)` image retaining pixel dtype and channel order. |
| `labels["<roi>_Segmentation"]` | Lazy `(y, x)` integer labels retaining source pixel dtype. |
| `points["<roi>_TranscriptList"]` | Lazy projected transcript events with physical `x`, `y`, categorical `name`, optional `cell` and source slice annotation `z_raw`. |
| `shapes["<roi>_Boundaries"]` | Eager polygons indexed by explicit source IDs, retaining GeoJSON source annotations. |
| `shapes["<roi>_CellCoordinates"]` | Eager centroid circles indexed by cell IDs, with radius `sqrt(area / pi)`. |

The table retains `area`, integer `instance_id`, and `(center_x, center_y)` in `obsm["spatial"]`. Observation names are original positional strings starting at `"0"`, retaining gaps after geometry drops; biological identity comes from `instance_id`. When labels are included, categorical `region` and SpatialData table metadata explicitly link the table to `<roi>_Segmentation`. With labels excluded, the table is normalized and unlinked. Including polygons or circles does not change its annotation target.

DAPI OME metadata calibrates pixel X and Y independently into the ROI physical coordinate frame. The transform assigns PhysicalSizeX to X and PhysicalSizeY to Y, including anisotropic pixels. The same pixel-to-physical transform applies to segmentation and GeoJSON polygons. Cell centroids and transcript X/Y already use the physical ROI frame and have identity transforms. Area supplies square physical units for circle radii. OME micrometers, nanometers and millimeters are converted to micrometers in the ROI frame; absent OME unit attributes use the OME micrometer default. Unsupported units, missing calibration and inconsistent metadata fail.

Transcript `z` is retained as integer `z_raw`. The export does not establish a physical calibration for this slice index, so transcript points have two spatial coordinates, X and Y. Optional `cell` values preserve source assignments, including unassigned zero values. The complete CellxGene feature header supplies the transcript feature vocabulary; unknown transcript names fail when their partition is computed. Transcript name categories follow the CellxGene feature header order, including zero-count features.



Table `uns["spatialdata_io_positions"]` describes unchanged float64 vendor cell-centroid XY micrometres: `global` for one ROI and each ROI name for multiple independent ROIs. These positions do not derive from polygon centroids.

## Options and execution

`include_images`, `include_labels`, `include_transcripts` and `include_shapes` default to `True`. `shape_geometry="polygons"` is the default; `"circles"` or `"both"` selects centroid circles or both geometry types. `include_shapes=False` excludes both.

`invalid_geometries="keep"` preserves invalid polygon topology by default. `"raise"` reports invalid cell IDs. `"drop"` removes the affected polygon, its requested centroid circle and its expression-table row, preserving surviving IDs, row order, sparse values and annotations. Complete boundary/table identity agreement is validated before dropping. Labels and lazy transcript assignments remain unchanged. A requested boundary element becoming empty fails the whole read. Circles-only and shapes-disabled modes never read boundaries or drop cells.

`scale_factors` and `chunks` follow the shared [raster options](raster_options.md): by default a single scale, images in `(1, 1024, 1024)` chunks and labels in `(1024, 1024)` chunks. Computing a region decodes only the TIFF tiles, or strips grouped into bands of at most 16 MiB, that intersect its chunks. A sequence such as `(2, 2)` builds image and label pyramids with the appropriate interpolation for each product. Generated label pyramids require segmentation dtypes of at most 32 bits, avoiding upstream integer rounding. Full-resolution source series and level remain fixed, so alternate scenes and uncalibrated source pyramid levels cannot be selected accidentally.

Expression parsing uses bounded batches and preserves sparse counts without allocating a full dense expression table. Metadata, tables and shapes are eager. Pixels and transcript rows remain lazy through construction of the returned SpatialData. TIFF tasks reopen their path-backed source when computed and close their resources; retain the input files until computation or writing completes. Small spatial selections read only the intersecting TIFF tiles or row bands of at most 16 MiB.

Provide valid CSV inputs. Arrow checks complete record widths but does not enforce strict quoting, so some malformed quote forms may be accepted. Numeric precision and exact integer-identifier validation remain unchanged and apply to the values produced by the parser.

Transcript CSVs require one physical header line and one physical line per record; quoted commas and quotes are supported, while embedded newlines in quoted fields are unsupported. Every record must contain exactly the header field count, including projected-out fields. Transcript graph construction reads only the CSV header. Validation of unread transcript rows is deferred: malformed coordinates, unknown features and invalid assignments raise with source context when the affected partition is computed. The Dask index is a storage index local to each partition and may repeat; it is not a biological event identifier. Source row order is preserved without a global shuffle. Unsupported compressed transcript variants are not advertised as a bounded source format.

The reader owns newly created tables and annotations and does not mutate caller containers or source files. Provenance records the reader and package version without absolute input paths. Missing or wrong-kind paths raise the corresponding filesystem exception. Malformed schemas, ambiguous layouts and inconsistent scientific relationships raise `ReaderFormatError`, a `ValueError` subclass. Invalid option values raise `ValueError`; invalid types raise `TypeError`. Validation errors do not return partial results.

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

The CLI uses paired inclusion flags and repeatable ROI, chunk and pyramid options:

```bash
spatialdata_io seqfish --input dataset --output sample.zarr \
  --shape-geometry both --rois Roi2 --rois Roi1 \
  --chunks 1 --chunks 512 --chunks 512

spatialdata_io seqfish --input dataset --output counts.zarr \
  --no-include-images --no-include-labels \
  --no-include-transcripts --no-include-shapes
```

The former `load_*`, `load_points`, `cells_as_circles`, `imread_kwargs` and `raster_models_scale_factors` options have been removed. Use the explicit inclusion flags, geometry choice, `chunks` and `scale_factors`. Automatic pyramids based on image size have been removed; request a pyramid explicitly when needed. The default chunks are `(1, 1024, 1024)` for images and `(1024, 1024)` for labels instead of storage-aligned chunks, and `scale_factors=()` replaces `None` for a single scale.
