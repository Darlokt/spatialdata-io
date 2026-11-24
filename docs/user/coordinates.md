# Coordinate systems and calibration

An image pixel, a transcript location and a cell centroid can all be described by
a pair of numbers. To overlay them or measure a distance you also need their
**units**, the **coordinate frame**, and **what the position represents**. A
reader can often establish these from the experiment, but the available evidence
differs by technology.

## How a spatial element gets its units

A reader can retain an image in its native pixel array and attach a transformation
from its intrinsic coordinates to a named frame. That transformation can account
for pixel size, orientation or placement. Inspect the transformations on an
element before comparing it with positions from another element.

```python
from spatialdata.transformations import get_transformation

image = next(iter(sdata.images.values()))
transforms = get_transformation(image, get_all=True)
print(transforms)
```

A calibrated sample commonly uses a frame called `global` in micrometres.
The frame name alone establishes neither its units nor a common origin with
another dataset. Several samples can each start at (0, 0). Separate samples,
cores, ROIs or image stacks retain independent frames unless there is evidence
for their registration.

A SpatialData 0.8 Zarr store preserves the numeric image transformation.
Its multiscale axis metadata alone does not identify which frames use
micrometres. For physical measurements, use the calibration documented for
the reader and frame. [SpatialData unit support](https://github.com/scverse/spatialdata/issues/120)

| Source | Interpretation |
| --- | --- |
| [Classic Visium](../readers/visium.md) | Spots and images share the full-resolution pixel frame. A matching original OME-TIFF with equal X and Y pixel sizes adds a micrometre frame for the spots and images. |
| [Visium HD](../readers/visium_hd.md) | The manufactured 2 µm grid establishes the slide's physical frame. |
| [Xenium](../readers/xenium.md) | Vendor cell centroids and transcripts use micrometres. Morphology pixels map into the same frame. |
| [Curio Seeker](../readers/curio.md) | Supported primary exports use the manufacturer's approximate 0.647 µm/pixel scale. |
| [DBiT](../readers/dbit.md) | Barcode positions use the barcode grid. Image spacing alone cannot establish barcode pitch or orientation. |
| [Stereo-seq](../readers/stereoseq.md) | Chip pitch maps native grid references to micrometres. Image-only reads without a matching pitch use pixels. |
| [PhenoCycler](../readers/phenocycler.md) | Calibrated QPTIFF pixels map to micrometres. A shared slide frame also requires matching slide identity and stage positions. |
| [CODEX](../readers/codex.md) | The table's pixel scale and the image-to-table mapping are separate inputs. |
| [CosMx](../readers/cosmx.md) | FOV placement establishes global pixels. An explicit pixel size gives the assembled sample physical units. |
| [MERSCOPE](../readers/merscope.md) | Transcripts and boundaries use micrometres. Registering the mosaic also requires its micron-to-mosaic transform. |
| [seqFISH](../readers/seqfish.md) and [MACSima](../readers/macsima.md) | Native or calibrated positions belong to individual ROIs or stacks. Independent acquisitions retain separate frames. |
| [MCMICRO](../readers/mcmicro.md) and [Steinbock](../readers/steinbock.md) | OME spacing or supported acquisition metadata calibrates a local grid. Their zero-based regionprops centroids map as `(source_xy + 0.5) * spacing_xy_um`. |
| [ISS](../readers/iss.md) | The caller supplies registered raw and label rasters. Calibrating their grid alone cannot assign units to an imported table embedding. |
| [Generic images and GeoJSON](../readers/generic.md) | Complete OME spacing on the selected base series can calibrate an image. An uncalibrated image uses pixels and GeoJSON retains source units, even if the chosen frame is called `global`. |

Where a reader uses optional OME calibration, it needs positive spacing on every
retained spatial axis. For example, XY spacing alone leaves a generic 3D image
in pixels. The reader guide lists the supported units and what happens when
metadata is missing or conflicting. Morphology annotations such as area and
axis length keep their documented source units.

## What table coordinates mean

Reader-produced canonical `table.obsm["spatial"]` contains finite float64
`(x, y)` positions in micrometres. Where calibration or a known frame is missing,
use the documented native coordinates. These may be `spatial_pixels`,
`spatial_grid`, an original embedding under `spatial_source`, or columns in
`obs`. Some products contain no positions.

For a table with declared positions, inspect `table.uns["spatialdata_io_positions"]`.
Its `coordinates` mapping describes the axes, units and meaning of each declared
embedding and the calibration for physical positions. A scalar `frame` identifies
one frame. A `frame_key` names a categorical column in `table.obs` giving the
frame of each row. A linkage region names a target element. Several regions
can share a frame. An aggregate table containing calibrated and uncalibrated
samples retains native positions for all rows and omits canonical `spatial`.

Stored positions are snapshots. Changing a transformation on an image or shape
does not recalculate table arrays. Cell centres reported by a vendor can also
differ from centroids calculated from its polygons. Choose a definition that
fits the analysis. The [AnnData guide](legacy_anndata.md) shows how to select
one frame and export centroids after registration.
