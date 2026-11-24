# What a reader table contains

Start with `list(sdata.tables)` to see which tables the reader returned. Each
table is an AnnData object. Rows in `obs` can represent spots, beads, bins,
cells or nuclei. Some tables contain only channel metadata, and an image-only
read can have no table. The [reader guides](../readers/index.md) identify each
table, its measured population, feature order, annotations and spatial targets.

## Where to find annotations and positions

| AnnData field | What it tells you |
| --- | --- |
| `obs` and `var` | One row per observation or feature, in the order of `X`. The guide for each reader identifies the source of the index, annotations and any additional expression layers. |
| `obsm["spatial"]` | Physical positions, when established for every row. This is an `(n_obs, 2)` float64 array in **x, y order and micrometres**. |
| `obsm["spatial_pixels"]`, `obsm["spatial_grid"]` | Native positions with known pixel or grid semantics, where supplied. Some readers keep native XY in `obs` instead. |
| `obsm["spatial_source"]` | An imported position array whose original meaning or units could not be verified. Its values, shape and dtype are preserved. |
| `uns["spatialdata_io_positions"]` | The meaning, units and frame of declared positions. If the entry has a `frame_key`, look up each row's frame in that `obs` column. |
| `uns["spatialdata_attrs"]` | For a linked table, the target element names and the columns identifying each row's region and instance. |
| `uns["spatial"]` | May hold legacy library metadata such as Visium scale factors. Use the position declaration to determine coordinate units. |

For example, a Xenium table stores vendor centroids in `obsm["spatial"]`.
Classic Visium stores spot centres in `obsm["spatial_pixels"]` because the
Space Ranger scale factors give relative image resolutions. Both tables have
positions that can be used with their own images. Only the Xenium coordinates
in this example directly support micrometre distances.
[Coordinate systems and calibration](coordinates.md)

## Expression and counts

When a reader returns expression, `table.X` is an eager SciPy CSR sparse
array with observations as rows and features as columns. The acquisition
determines whether the values are molecule counts, marker intensities or
imported analysis results. A vendor total in `obs` can cover a different
population or feature set from `X`, for example after feature selection.
Check the reader guide before using a source summary as a library size.

(completed-raw-counts)=
### Raw-count precision

For seqFISH, MERSCOPE, CosMx, raw Stereo-seq and the 10x-backed readers,
completed integer counts use float32 through 2²⁴ and float64 through 2⁵³.
Larger integer results raise an error to avoid rounding. Data already stored
in floating point are validated at their available precision. Related count
arrays, such as MERSCOPE genes and blanks or Stereo-seq expression and exon
counts, share a dtype.

Curio, DBiT, ISS, Steinbock and Stereo-seq analysis can import expression
matrices with repeated entries for a row and feature. Those contributions add.
Integer data retain their dtype when the sums fit, or promote to a floating
dtype that preserves all values exactly. A warning identifies affected inputs
and dtype changes. Processed expression keeps its scientific meaning even
when its values are signed or fractional.

## Which spatial element does a row describe?

For a linked table, `table.uns["spatialdata_attrs"]` identifies the region and
instance columns. The region names the target element, for example a label
image. The instance ID selects the object within it, for example a label
value. The **pair** identifies the object. Two samples can use the same numeric
label in different images. Keep source string identifiers as strings so that
values such as `"007"` remain distinct from `"7"`.

Inspect a linked table in Python:

```python
table = next(iter(sdata.tables.values()))
linkage = table.uns["spatialdata_attrs"]
print(table.obs[[linkage["region_key"], linkage["instance_key"]]].head())
```

A target name establishes linkage. Distance measurements also require
[coordinate units and frames](coordinates.md). Group rows by the declared
frame when analysing independent samples. Label pixels remain lazy, so reading
the table does not enumerate IDs by scanning a label image.

(persistence-ownership-and-numerical-limits)=
## Saving a table

Use SpatialData Zarr to retain a table, its target elements and their
transformations together. An H5AD file stores the table, annotations and
embeddings. Any retained region names still refer to the source targets.
Deserialization can return a SciPy sparse matrix instead of the sparse array
returned by the reader. Reading back categorical text can also change its
missing-value sentinel while preserving category membership and order.

For examples of saving an independent H5AD, importing an older image library
or exporting positions from linked shapes, see [Work with AnnData tables](legacy_anndata.md).
