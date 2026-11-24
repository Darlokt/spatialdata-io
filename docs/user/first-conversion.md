# Convert your first dataset

Convert a small Xenium experiment into a SpatialData Zarr store. You will
open its cell-by-gene table, find the segmentation it annotates, then reopen
the saved experiment and check the result.

Use a Python 3.12 or newer environment with the version of spatialdata-io
that matches these docs. For a released version, install it with
`python -m pip install spatialdata-io`. For this development checkout,
run `python -m pip install -e .` from the repository root. Then start
in an empty working directory.

## Get the experiment

Download the **mouse ileum, nuclear expansion** example from the Xenium
Onboard Analysis 3.0 collection. 10x Genomics distributes this 12 MB ZIP
under CC BY 4.0. It contains three small tissue patches sampled across two
fields of view. [10x Genomics example datasets](https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/resources/xenium-example-data)

```bash
curl --fail --location --output mouse-ileum.zip \
  https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Mouse_Ileum_tiny/Xenium_Prime_Mouse_Ileum_tiny_outs.zip
python -m zipfile -e mouse-ileum.zip mouse-ileum
```

You can also download the ZIP in a browser and extract it into `mouse-ileum/`.
That directory should contain `experiment.xenium`, `cell_feature_matrix.h5`,
`cells.zarr.zip`, and `morphology_focus/`. Leave the inner Zarr ZIPs in place.
For a checkout with the test dataset already installed, use
`data/Xenium_Prime_Mouse_Ileum_tiny_outs` as the reader path.
[Developer dataset guide](../development/datasets.md)

## Read the cells and find their segmentation

Run the Python examples in order. Pass the directory containing
`experiment.xenium` to the Xenium reader.

```python
from spatialdata_io import xenium

sdata = xenium("mouse-ileum")
cells = sdata.tables["table"]
print(cells.shape)
print(cells.obs["region"].unique().tolist())
print(sdata.images["morphology_focus"].dims)
```

You should see:

```text
(23, 5006)
['cell_labels']
('c', 'y', 'x')
```

`cells` is an AnnData table, the annotated matrix used across scverse
{cite:p}`Virshup_2023`. Its 23 rows are cells and its 5,006 columns are
features. `cells.X` contains sparse counts. The `region` column tells you
which segmentation the table annotates. A row's `label_id` is the integer
value of that cell in `sdata.labels["cell_labels"]`. See the first three
identifiers and their XY centroids in micrometres:

```python
print(cells.obs[["region", "label_id"]].head(3))
print(cells.obsm["spatial"][:3])
```

The morphology image and segmentation masks are lazy. Keep the extracted
dataset available while you write the store.

## Write and reopen the experiment

```python
from spatialdata import read_zarr

sdata.write("mouse-ileum.zarr")
restored = read_zarr("mouse-ileum.zarr")
reloaded_cells = restored.tables["table"]

print(reloaded_cells.shape)
print(sorted(restored.labels))
assert reloaded_cells.obs_names.equals(cells.obs_names)
assert (reloaded_cells.X != cells.X).nnz == 0
```

You now have a `mouse-ileum.zarr/` directory containing the cells, images,
labels, shapes and transcript points together. The reopened table has the
same 23 cells and count values, and the masks are `cell_labels` and
`nucleus_labels`. In a new Python session you can start directly with
`read_zarr("mouse-ileum.zarr")`.

The [Xenium reader guide](../readers/xenium.md) describes its other products
and options. For a different instrument, [find its reader](../readers/index.md).
For a self-contained example using a single TIFF, [convert your first image](first-image.md).
