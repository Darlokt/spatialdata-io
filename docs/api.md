# Python API reference

Use the [reader guides](readers/index.md) to identify a supported input layout,
then select a function below for its parameters and return values. Dataset
readers take a local input directory as `path` and return a SpatialData
object. CODEX and PhenoCycler also accept individual files and resolve
relative file options against the selected file's parent directory. See
[dataset paths](user/paths.md), [raster options](user/rasters.md) and
[AnnData table conventions](user/tables.md) for behavior shared across readers.

```{eval-rst}
.. module:: spatialdata_io
```

## Technology readers

```{eval-rst}
.. autosummary::
    :toctree: generated

    codex
    cosmx
    curio
    dbit
    macsima
    mcmicro
    merscope
    phenocycler
    seqfish
    steinbock
    stereoseq
    visium
    visium_hd
    xenium
```

## Single-file readers

```{eval-rst}
.. autosummary::
    :toctree: generated

    generic
    image
    geojson
```

## Experimental readers

The ISS reader imports a caller-selected image, segmentation and H5AD table
from an in situ sequencing workflow. Its entry point is
`spatialdata_io.experimental.iss`.

```{eval-rst}
.. autosummary::
    :toctree: generated

    experimental.iss
```

## Conversion functions

For direct table access, explicit table/centroid position selection and importing
legacy image libraries, see [AnnData access and legacy conversion](user/legacy_anndata.md).
The exporter returns one coordinate frame per call and does not export images.

```{eval-rst}
.. currentmodule:: spatialdata_io

.. autosummary::
    :toctree: generated

    converters.legacy_anndata.from_legacy_anndata
    converters.legacy_anndata.to_legacy_anndata
    generic_to_zarr
```

## Xenium images and selections

```{eval-rst}
.. currentmodule:: spatialdata_io

.. autosummary::
    :toctree: generated

    xenium_aligned_image
    xenium_explorer_selection
```

## Format errors

```{eval-rst}
.. currentmodule:: spatialdata_io.readers._utils.errors

.. py:exception:: ReaderFormatError

   A :exc:`ValueError` raised when a dataset violates a reader's format
   requirements. The message identifies the reader and artifact, what the
   reader expected and what it found. When available, it includes the file
   path and detected format version.
```
