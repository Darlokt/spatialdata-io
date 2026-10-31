# API

This section documents the Application Programming Interface (API) for the `spatialdata_io` package.

```{eval-rst}
.. module:: spatialdata_io
```

## spatialdata-io

I/O for the `spatialdata` project.

### Readers

#### Spatial technologies

```{eval-rst}
.. autosummary::
    :toctree: generated

    codex
    cosmx
    curio
    dbit
    iss
    macsima
    mcmicro
    merscope
    seqfish
    steinbock
    stereoseq
    visium
    visium_hd
    xenium
```

#### Data types

```{eval-rst}
.. autosummary::
    :toctree: generated

    generic
    image
    geojson
```

### Conversion functions

For direct table access, explicit table/centroid position selection and importing
legacy image libraries, see [AnnData access and legacy conversion](legacy_anndata.md).
The exporter returns one coordinate frame per call and does not export images.

```{eval-rst}
.. currentmodule:: spatialdata_io

.. autosummary::
    :toctree: generated

    experimental.from_legacy_anndata
    experimental.to_legacy_anndata
    generic_to_zarr
```

### Utility functions

```{eval-rst}
.. currentmodule:: spatialdata_io

.. autosummary::
    :toctree: generated

    xenium_aligned_image
    xenium_explorer_selection
```
