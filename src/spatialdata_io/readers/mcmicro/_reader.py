"""Assemble one MCMICRO run into lazy images and labels and one linked cell table.

The public function only orchestrates: validate the raster options, discover
the layout (``_layout``), read the marker panel (``_metadata``), build the
rasters and their grids (``_rasters``), stream and link the quantification
tables (``_tables``) and attach provenance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from spatialdata import SpatialData

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers.mcmicro._layout import TABLE_KEY, TMA_MAP_KEY, WsiLayout, discover_layout
from spatialdata_io.readers.mcmicro._metadata import read_markers
from spatialdata_io.readers.mcmicro._rasters import ImageGrid, read_image, read_labels, read_mask
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
    path_policy: Literal["strict", "relaxed"] = "strict",
) -> SpatialData:
    """Read the outputs of one MCMICRO whole-slide or tissue-microarray run.

    The reader converts pipeline results, not inputs: ``raw/``,
    ``illumination/``, Coreograph QC files and segmentation puncta tables are
    never read. The YAML boolean ``workflow.tma`` in ``qc/params.yml`` (MCMICRO's
    complete parameter dump) selects the layout. A whole-slide run::

        run/
        ├── markers.csv
        ├── qc/params.yml
        ├── registration/<image>.ome.tif
        ├── segmentation/<method>-<image>/<mask>.ome.tif
        └── quantification/<image>--<method>_<mask>.csv

    A tissue-microarray (TMA) run adds Coreograph cores, each with a tissue
    mask, and segments and quantifies the cores::

        run/
        ├── markers.csv
        ├── qc/params.yml
        ├── registration/<tma>.ome.tif
        ├── dearray/<n>.ome.tif
        ├── dearray/masks/<n>_mask.tif
        ├── segmentation/<method>-<n>/<mask>.ome.tif
        └── quantification/<n>--<method>_<mask>.csv

    - Registration must hold exactly one TIFF-family image; its ID drops the
      extension and an ``.ome`` suffix. Cores are positive integers.
    - Segmentation and quantification are optional. Every quantification table
      must name exactly one discovered label; ``<method>`` and ``<mask>`` are
      compared as whole names. A historical ``segmentation/<image>/``
      directory is accepted when exactly one quantification table names the
      method of each of its masks.
    - Image IDs, methods and mask names that are not valid SpatialData element
      names are sanitized with a warning; element names that collide, ignoring
      letter case, fail before any raster is opened.

    Parameters
    ----------
    path
        MCMICRO project directory.
    scale_factors
        Integer factors of at least two, one per generated pyramid level of the
        images and labels; empty for a single scale. Generated label levels
        require labels of at most 32 bits.
    chunks
        One positive size for every axis or three ``c, y, x`` sizes of the
        images; labels use the ``y`` and ``x`` sizes. By default one channel per
        chunk and square 1024-pixel tiles.
    path_policy
        How symlinks inside the dataset directory are treated. ``"strict"``
        requires every discovered artifact and relative file option to resolve
        inside the directory. ``"relaxed"`` also accepts symlinks to targets
        stored elsewhere, as in git-annex or DVC datasets. Relative paths may
        not leave the directory with ``..`` in either mode, and absolute file
        options may point anywhere. File options keep the name the caller
        gave, and relaxed reads keep each artifact's own name rather than that
        of its link target.

    Returns
    -------
    SpatialData
        Images, labels and, when quantification exists, one linked table:

        ``images["<image>_image"]`` (whole slide), ``images["tma_map"]`` and ``images["core_<n>_image"]`` (TMA)
            Lazy ``c, y, x`` arrays in the source dtype. Channels are the
            canonical MCQuant marker IDs in ``markers.csv`` row order: every
            occurrence of a repeated marker is suffixed (``DAPI, CD3, DAPI``
            becomes ``DAPI_1, CD3, DAPI_2``); generated collisions fail.
        ``labels["<image>_<method>_<mask>"]`` (``core_<n>_<method>_<mask>`` for a TMA)
            Lazy integer ``y, x`` segmentation on the grid of its image.
        ``labels["core_<n>_mask"]``
            Lazy Coreograph tissue mask on its own, possibly downsampled, grid.
        ``tables["table"]``
            Every quantification table in label order, then source row order,
            linked by categorical ``obs["region"]`` (the label element) and
            ``obs["CellID"]`` (its exact positive uint64 label value; unique within a
            region and representable in the label dtype; label pixels are
            never read to check membership).

            - ``X``: canonical CSR float64 mean marker intensities, finite,
              parsed directly with Arrow's correctly rounded float64 grammar.
              ``var_names`` are the canonical marker IDs and ``var`` keeps
              every ``markers.csv`` column as text.
            - ``obs_names`` are row positions ``"0"``, ``"1"``, ...; MCQuant
              morphology columns are nullable float64 (empty fields missing)
              and every other column is exact text.
            - ``obs`` and ``var`` column names that a SpatialData table does
              not accept (``CD45 RO_intensity_max``, or ``area`` beside
              ``Area``) are sanitized with a warning: invalid characters
              become ``_`` and a name that collides, ignoring letter case,
              gets a ``_1``, ``_2``, ... suffix. ``region``, ``CellID`` and
              valid names keep their spelling.
            - ``obsm["spatial_pixels"]``: the float64 ``X_centroid``,
              ``Y_centroid`` regionprops centroids, zero-based pixel indices
              in frame ``<grid>_regionprops_indices`` (``image``, ``core_<n>``).
            - ``obsm["spatial"]`` (only when every measured image is
              calibrated): float64 micrometres ``(spatial_pixels + 0.5) *
              spacing`` in the image's coordinate system. Nonfinite results,
              and nonzero coordinates that would become zero, fail.
            - With several frames (TMA cores), categorical
              ``obs["spatial_pixel_frame"]`` and ``obs["spatial_frame"]``
              name each row's frames; such source columns then fail.
            - ``uns["spatialdata_io_positions"]`` declares both arrays, their
              units, frames and calibration evidence. They are snapshots that
              later registration or concatenation do not update.

        Coordinate systems: every raster keeps its native origin, and no
        transform between the TMA map, cores and tissue masks is invented,
        because Coreograph records no authoritative crop origins. A raster
        with complete base OME ``PhysicalSizeX``/``PhysicalSizeY`` (µm, nm or
        mm) is scaled into a micrometre frame: ``global`` for the whole-slide
        image or the TMA map, ``core_<n>`` for a core and ``core_<n>_mask``
        for its tissue mask. Otherwise it uses the identity into a pixel frame:
        ``image_pixels``, ``tma_map_pixels``, ``core_<n>_pixels`` or
        ``core_<n>_mask_pixels``. So ``global`` always means micrometres.
        Segmentations share their image's frame; their own present sizes must
        agree with it. Tissue masks use only their own sizes.

        ``attrs`` record the reader and package version. Pixels are not read
        until computed, and a partial computation decodes only the TIFF tiles
        or strips it overlaps; the source files must stay available. The
        quantification tables are streamed in bounded blocks and expression is
        never densified as a whole. Files are closed on return.

        CSV integer fields accept hexadecimal and surrounding spaces or tabs,
        and reject decimal/exponent integer spellings, signs such as ``+1``
        and separators such as ``1_0``. Floating text rounds directly to
        float64; underflow may become zero. Only empty optional morphology
        fields become missing; IDs, intensities and centroids are required.
        Text annotations preserve empty strings and missing-value words.

    Warns
    -----
    UserWarning
        If an image ID, method or mask name was sanitized into an element name,
        or a quantification or ``markers.csv`` column name into a table
        column name.

    Raises
    ------
    TypeError
        If an option has the wrong type.
    ValueError
        If an option has an invalid value, or a requested pyramid level would
        downsample 64-bit labels or shrink a raster below one pixel.
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        If a required artifact is missing or has the wrong kind.
    ValueError
        A contextual ``ReaderFormatError`` naming the artifact and path if the
        layout, ``qc/params.yml``, ``markers.csv``, a raster or a
        quantification table is malformed or they disagree, for example an
        ambiguous registration, unpaired tables, colliding element names, a
        channel count other than the panel's, labels off their image grid,
        conflicting calibrations, nonfinite values or out-of-range IDs.
    """
    options = make_raster_options(scale_factors, chunks)
    layout = discover_layout(path, path_policy=path_policy)
    panel = read_markers(layout.markers)

    images: dict[str, DataArray | DataTree] = {}
    labels: dict[str, DataArray | DataTree] = {}
    grids: dict[str, ImageGrid] = {}

    if isinstance(layout, WsiLayout):
        images[layout.image_key], grids[layout.image_id] = read_image(
            layout.image, name="image", calibrated_frame="global", channels=panel.features, options=options
        )
    else:
        images[TMA_MAP_KEY], _ = read_image(
            layout.tma_map, name=TMA_MAP_KEY, calibrated_frame="global", channels=panel.features, options=options
        )
        for core in layout.cores:
            images[core.image_key], grids[core.image_id] = read_image(
                core.image, name=core.image_id, calibrated_frame=core.image_id, channels=panel.features, options=options
            )
            labels[core.mask_key] = read_mask(core.mask, name=core.mask_key, options=options)

    label_dtypes: dict[str, np.dtype[np.integer]] = {}
    for segmentation in layout.segmentations:
        labels[segmentation.key], label_dtypes[segmentation.key] = read_labels(
            segmentation.path, grid=grids[segmentation.image_id], options=options
        )

    tables = {}
    if layout.quantifications:
        tables[TABLE_KEY] = read_table(layout.quantifications, panel, label_dtypes, grids)

    result = SpatialData(images=images, labels=labels, tables=tables)
    set_reader_provenance(result.attrs, reader="mcmicro")
    return result
