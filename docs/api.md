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

### Experimental readers

The ISS reader imports a caller-selected image, segmentation and H5AD table
from an in situ sequencing workflow. Its entry point is
`spatialdata_io.experimental.iss`.

```{eval-rst}
.. autosummary::
    :toctree: generated

    experimental.iss
```

### Conversion functions

For direct table access, explicit table/centroid position selection and importing
legacy image libraries, see [AnnData access and legacy conversion](legacy_anndata.md).
The exporter returns one coordinate frame per call and does not export images.

```{eval-rst}
.. currentmodule:: spatialdata_io

.. autosummary::
    :toctree: generated

    converters.legacy_anndata.from_legacy_anndata
    converters.legacy_anndata.to_legacy_anndata
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
