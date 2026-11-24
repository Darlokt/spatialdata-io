# Work with AnnData tables

## Use or save a reader table

Choose a table from `sdata.tables`. You can use it directly with AnnData
tools. Make a copy if you want to edit it independently of the SpatialData
object.

```python
table = sdata.tables["table"]
independent = table.copy()
```

To save a standalone H5AD without references to external spatial elements,
remove the linkage declaration on a copy:

```python
standalone = sdata.tables["table"].copy()
standalone.uns.pop("spatialdata_attrs", None)
standalone.write_h5ad("table.h5ad")
```

Use a SpatialData Zarr store to save the target geometries, pixels and
transformations together with the table. See [What a reader table contains](tables.md).

## Choose a frame for spatial analysis

Inspect `table.uns["spatialdata_io_positions"]` before treating XY values as
physical distances. Canonical reader `obsm["spatial"]` is in micrometres. A
scalar `frame` describes a single frame. When the declaration uses `frame_key`,
select rows from one frame before computing distances or neighbourhoods:

```python
description = table.uns["spatialdata_io_positions"]["coordinates"]["spatial"]
frame_key = description["frame_key"]
frame = table.obs[frame_key].iloc[0]
one_frame = table[table.obs[frame_key] == frame].copy()
```

This selects the first observation's frame. Slicing keeps row-aligned
embeddings, although linkage metadata may retain surplus region names.
Concatenating tables in AnnData does not register samples with one another.
See [Coordinate systems and calibration](coordinates.md).

## Export stored positions or linked-element centroids

Use `to_legacy_anndata()` for an independent table with an explicit position
source. Select a table if the dataset contains several and pass
`coordinate_system` when observations occupy multiple frames. Rows in a
selected frame keep their relative order.

```python
from spatialdata_io.converters.legacy_anndata import to_legacy_anndata

exported = to_legacy_anndata(sdata, table_name="table")
```

The default keeps stored positions and works without reading spatial elements.
An incoming `spatial` of unknown meaning moves to `spatial_source`. The
exporter copies existing legacy image-library metadata with the table. Images
held as SpatialData elements stay in SpatialData.

To calculate positions from the current transformations of linked shapes,
labels or points, select a frame with known units:

```python
centroids = to_legacy_anndata(
    sdata,
    table_name="table",
    coordinate_system="global",
    position_source="centroids",
)
```

Centroids are matched by region and instance ID. In a physical frame they
become `spatial`, with previous physical positions saved as `spatial_original`.
In a native pixel or grid frame the result goes in `spatial_centroids`. An
occupied preservation key raises an error. Computing label centroids can
require reading the complete label image, and points can require a full scan.
Vendor cell centres and geometry centroids can describe different positions.

## Import an older spatial AnnData

Start with an in-memory AnnData object. For a single library:

```python
from spatialdata_io.converters.legacy_anndata import from_legacy_anndata

imported = from_legacy_anndata(adata)
```

Plain legacy `obsm["spatial"]` is interpreted as full-resolution pixel-index
XY and moves to `spatial_pixels`. The result links the `table` to `locations`
points using new `spatialdata_region` and `spatialdata_instance_id` columns.
Original observation names and identifiers survive. Empty or positionless
tables remain unlinked. Both converters preserve feature order, layers and
`raw` on the copy, with canonical sparse CSR `X` when expression is present.

To select one of several libraries, supply its ID and the observation column
that associates rows with libraries:

```python
imported = from_legacy_anndata(
    adata,
    library_id="sample_a",
    library_key="library_id",
)
```

The selected library's images keep their stored pixels and resolution. Input
images are YX or YXC and become CYX. Pass `rgb=False` for three or four
numerical channels that should be displayed independently. For an image named
`hires`, the importer uses the direct `scalefactors["hires"]` entry when present,
otherwise `scalefactors["tissue_hires_scalef"]`. With scale `s`, intrinsic raster
centres map to full-resolution pixel indices as `(intrinsic_xy - 0.5) / s`.
These factors register resolutions in pixels. Importing a table with declared
physical positions is supported for one frame without legacy images. Legacy
images alongside physical positions lack a physical image mapping and raise
an error. See the [converter reference](../api.md#conversion-functions).
