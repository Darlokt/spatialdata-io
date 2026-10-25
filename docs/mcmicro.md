# MCMICRO

{func}`spatialdata_io.mcmicro` reads the outputs of one [MCMICRO](https://mcmicro.org/) whole-slide or tissue-microarray (TMA) run. It returns the registered image, Coreograph cores and tissue masks, segmentation labels and one linked MCQuant cell table. Pipeline inputs (`raw/`, `illumination/`), Coreograph QC files and segmentation puncta tables are not read.

## Run layouts

MCMICRO writes `qc/params.yml` as a complete dump of the parameters used. Its YAML boolean `workflow.tma` selects the layout; any other value, a duplicated key or malformed YAML fails. A whole-slide run contains:

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

A historical `segmentation/<image>/` directory, without the method prefix, is accepted when exactly one quantification table identifies the method of each of its masks. Segmentation and quantification are optional. All element names must be unique; a layout whose names would collide fails before any image is opened.

## Elements and coordinates

| Element | Contract |
| --- | --- |
| `images["<image>_image"]` | Whole-slide registered image in `global`. |
| `images["tma_map"]` | TMA registered image in `global`. |
| `images["core_<n>_image"]` | Coreograph core in its local pixel frame `core_<n>`. |
| `labels["core_<n>_mask"]` | Coreograph tissue mask in its own frame `core_<n>_mask`. |
| `labels["<image>_<method>_<mask>"]` | Segmentation labels in the frame of their image (`core_<n>` for cores). |
| `tables["table"]` | One table linked to every label with quantified cells. |

Images are lazy `(c, y, x)` arrays in source dtype. Their channels are the canonical MCQuant marker IDs in `markers.csv` row order: every occurrence of a repeated marker is suffixed, so `DAPI, CD3, DAPI` becomes `DAPI_1, CD3, DAPI_2`, and generated collisions such as `X, X, X_1` fail. The registered image and every core must have exactly one channel per marker. Labels are lazy integer `(y, x)` arrays that must match the spatial shape of their source image; masks keep their own, possibly downsampled, grid.

All transformations are identities. Coreograph exports no authoritative crop origins and its masks may be downsampled, so the reader does not invent a transformation between the map, a core and its mask.

## Table

`markers.csv` needs a `marker_name` column. Every other column is optional module metadata and is kept as exact text in `var`, together with the original `marker_name`. `var_names` are the canonical marker IDs.

Quantification tables need `CellID`, one column per canonical marker and `X_centroid`/`Y_centroid`. All tables of a run must have the same columns. Marker columns become eager, finite, canonical CSR float64 expression; exact decimal values are decoded with correct rounding, and nonfinite values or nonzero values that underflow to zero fail. `CellID` holds exact positive int64 label values, unique within each label and representable in that label's dtype; label pixels are never read to check them. MCQuant morphology columns (`Area`, `MajorAxisLength`, `MinorAxisLength`, `Eccentricity`, `Solidity`, `Extent`, `Orientation`, `column_centroid`, `row_centroid`) are float64 with empty fields as missing values. All other columns, including suffixed intensity properties, are kept as exact text.

Rows follow label order and then source row order. Observation names are row positions `"0"`, `"1"`, ...; `region` names the label image and `CellID` is the instance key. `obsm["spatial"]` holds the float64 centroid pixel coordinates. A run without quantification returns no table.

## Options and execution

`scale_factors=None` keeps a single full-resolution scale; a sequence of integer factors greater than one adds pyramid levels to images and labels. Generated label pyramids require labels of at most 32 bits. `chunks` is one positive size or three CYX sizes, and labels use the final two; `None` keeps native TIFF tiles, or whole strips grouped into bands of at most 16 MiB.

Construction reads TIFF metadata only and executes no pixel tasks. Computing a small region reads only the TIFF storage blocks that intersect it. Quantification tables are parsed in bounded batches of at most 10,000 rows and 8 MiB of dense marker values, and only CSR blocks are retained. The source files must remain available for later raster computation.

## Python and command line

```python
from spatialdata_io import mcmicro

sdata = mcmicro("run", chunks=(1, 1024, 1024))
sdata.write("run.zarr")
```

The CLI mirrors the reader with repeatable pyramid and chunk options:

```bash
spatialdata_io mcmicro --input run --output run.zarr --chunks 1 --chunks 1024 --chunks 1024
```

The former `imread_kwargs`, `image_models_kwargs` and `labels_models_kwargs` mappings have been removed; use `scale_factors` and `chunks`. Compared with earlier releases, the reader no longer returns raw or illumination images, includes the method in label names, keeps every core and mask in its exact local frame instead of placing it by centroids, names masks `core_<n>_mask`, returns one table named `table` with positional observation names, and stores expression as sparse arrays.
