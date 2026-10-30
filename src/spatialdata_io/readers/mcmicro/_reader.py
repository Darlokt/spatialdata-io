"""Assemble one MCMICRO run into lazy rasters and one linked cell table."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata import SpatialData

from spatialdata_io.readers._utils.image import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS, make_raster_options
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers.mcmicro._layout import TABLE_KEY, WsiLayout, discover_layout
from spatialdata_io.readers.mcmicro._metadata import read_markers
from spatialdata_io.readers.mcmicro._rasters import ImageGrid, read_image, read_labels
from spatialdata_io.readers.mcmicro._tables import read_table

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np
    from xarray import DataArray, DataTree


def mcmicro(
    path: str | Path,
    *,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
) -> SpatialData:
    """Read the outputs of one MCMICRO whole-slide or tissue-microarray run.

    The reader converts pipeline results, not pipeline inputs: ``raw/``,
    ``illumination/``, Coreograph QC files and segmentation puncta tables are
    never read. ``qc/params.yml`` is MCMICRO's complete parameter dump; its
    boolean ``workflow.tma`` selects the layout. Whole-slide runs contain::

        run/
        ├── markers.csv
        ├── qc/params.yml
        ├── registration/<image>.ome.tif
        ├── segmentation/<method>-<image>/<mask>.ome.tif
        └── quantification/<image>--<method>_<mask>.csv

    Tissue-microarray runs add Coreograph cores, each with a tissue mask::

        run/
        ├── markers.csv
        ├── qc/params.yml
        ├── registration/<tma>.ome.tif
        ├── dearray/<core>.ome.tif
        ├── dearray/masks/<core>_mask.tif
        ├── segmentation/<method>-<core>/<mask>.ome.tif
        └── quantification/<core>--<method>_<mask>.csv

    A historical ``segmentation/<image>/`` directory is accepted when exactly
    one quantification table identifies the method of each of its masks.
    Segmentation and quantification are optional, but every quantification
    table must name exactly one discovered label image. Registration must
    contain exactly one TIFF-family image, and cores are positive integers.

    Parameters
    ----------
    path
        MCMICRO project directory.
    scale_factors
        Integer factors of at least two, one per generated pyramid level of the
        images and labels; empty for a single scale.
    chunks
        One positive size for every axis or three ``c, y, x`` sizes of the
        images; labels use the ``y`` and ``x`` sizes. By default one channel per
        chunk and square 1024-pixel tiles.

    Returns
    -------
    spatialdata.SpatialData
        Images are lazy ``(c, y, x)`` arrays in source dtype whose channels are
        the canonical MCQuant marker IDs in ``markers.csv`` row order. Every
        occurrence of a repeated marker is suffixed (``DAPI, CD3, DAPI``
        becomes ``DAPI_1, CD3, DAPI_2``); generated collisions fail. Labels are
        lazy integer ``(y, x)`` arrays on their source image grid.

        Whole-slide element names are ``<image>_image`` and
        ``<image>_<method>_<mask>`` in coordinate system ``global``.
        Tissue-microarray runs return the registered map ``tma_map`` in
        ``global``, ``core_<n>_image`` and ``core_<n>_<method>_<mask>`` in the
        local frame ``core_<n>``, and Coreograph tissue masks
        ``core_<n>_mask`` in their own frame ``core_<n>_mask``. Coreograph
        records no authoritative crop origins and masks may be downsampled, so
        no transform between these frames is invented. Each image uses its
        complete base OME XY spacing in µm, nm or mm when present, otherwise
        source pixels. Same-grid segmentations inherit that calibration;
        conflicting declarations fail. Independent tissue masks use only their
        own complete OME calibration. Anisotropic spacing is preserved and the
        native origin is unchanged. ASHLAR may itself write a default spacing;
        declared output metadata cannot verify acquisition accuracy.

        ``tables['table']`` concatenates every quantification table in label
        order. ``X`` is an eager finite canonical CSR float64 array of mean
        marker intensities with ``var_names`` equal to the canonical marker
        IDs; ``var`` keeps every ``markers.csv`` column, including the original
        ``marker_name``, as text. Observation names are row positions
        ``"0"``, ``"1"``, ...; ``region`` names the label image and ``CellID``
        holds its exact int64 label value. ``obsm['spatial']``
        holds the float64 ``X_centroid``/``Y_centroid`` pixel coordinates.
        MCQuant morphology columns are nullable float64 and all other columns
        remain exact text. No table is returned without quantification.

    Raises
    ------
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        Required artifacts are missing or have the wrong filesystem kind.
    ValueError
        Layouts, YAML, schemas, identifiers, values, shapes or options violate
        the supported contract, or element names would collide.
    TypeError
        Public options have invalid types.

    Notes
    -----
    Construction executes no pixel tasks and never reads label pixels; table
    IDs are checked against the label dtype range only. Quantification tables
    are parsed in bounded batches and never densified as a whole. Source files
    remain necessary for later raster computation. Caller options are not
    mutated.
    """
    options = make_raster_options(scale_factors, chunks)
    layout = discover_layout(path)
    panel = read_markers(layout.markers)

    images: dict[str, DataArray | DataTree] = {}
    labels: dict[str, DataArray | DataTree] = {}
    grids: dict[str, ImageGrid] = {}
    frames: dict[str, str] = {}

    if isinstance(layout, WsiLayout):
        frames[layout.image_id] = "global"
        images[f"{layout.image_id}_image"], grids[layout.image_id] = read_image(
            layout.image, channels=panel.features, frame="global", options=options
        )
    else:
        images["tma_map"], _ = read_image(layout.tma_map, channels=panel.features, frame="global", options=options)
        for core in layout.cores:
            frames[core.image_id] = core.image_id
            images[f"{core.image_id}_image"], grids[core.image_id] = read_image(
                core.image, channels=panel.features, frame=core.image_id, options=options
            )
            labels[f"{core.image_id}_mask"], _ = read_labels(
                core.mask, grid=None, frame=f"{core.image_id}_mask", options=options
            )

    label_dtypes: dict[str, np.dtype[np.generic]] = {}
    for segmentation in layout.segmentations:
        labels[segmentation.key], label_dtypes[segmentation.key] = read_labels(
            segmentation.path, grid=grids[segmentation.image_id], frame=frames[segmentation.image_id], options=options
        )

    tables = {}
    if layout.quantifications:
        tables[TABLE_KEY] = read_table(layout.quantifications, panel, label_dtypes)

    result = SpatialData(images=images, labels=labels, tables=tables)
    set_reader_provenance(result.attrs, reader="mcmicro")
    return result
