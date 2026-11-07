"""Assemble one Steinbock dataset: paired sample rasters and the linked cell table."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, cast

from spatialdata import SpatialData

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table, prepare_anndata_dataframe
from spatialdata_io.readers.steinbock._images import read_rasters
from spatialdata_io.readers.steinbock._layout import discover_layout, validate_masks_dir, validate_table_file
from spatialdata_io.readers.steinbock._metadata import read_image_metadata
from spatialdata_io.readers.steinbock._tables import read_table, set_positions

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pandas as pd

    from spatialdata_io.readers._utils.table import SparseDuplicateSummary

_LINKAGE_CONTEXT = ReaderErrorContext("steinbock", "linked cell table")


def steinbock(  # noqa: PLR0913, RUF100 - explicit format choices keep the public boundary typed.
    path: str | Path,
    *,
    masks_dir: str = "masks",
    table_file: str | Path = "cells.h5ad",
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
    path_policy: Literal["strict", "relaxed"] = "strict",
) -> SpatialData:
    """Read a Steinbock project: OME images, one segmentation-mask directory and the AnnData table.

    Supported layout (Steinbock has no format version marker)::

        project/
        ├── cells.h5ad              # steinbock export anndata (table_file)
        ├── images.csv              # optional; dimensions and IMC acquisition extents
        ├── ome/<sample>.ome.tiff   # steinbock export ome
        └── masks/<sample>.tiff     # steinbock segment (masks_dir)

    Other Steinbock outputs (``img/``, ``intensities/``, ``regionprops/``,
    ``neighbors/``, ``panel.csv``) are not read.

    - Images and masks pair by sample name (the file name without
      ``.ome.tiff`` or ``.tiff``); every image needs exactly one mask and the
      reverse, and at least one pair is required. Unmatched files fail.
    - Sample names name elements and coordinate systems. A sample name that
      is not a valid SpatialData name (only letters, digits, ``_``, ``-`` and
      ``.``; not ``.``, ``..`` or starting with ``__``) is sanitized with a
      warning, for example ``sample 1`` becomes ``sample_1``; the table still
      refers to the file name. The sanitized names must be distinct ignoring
      case, and a sample may not be named like another sample's derived frame
      (``<other>_pixels`` or ``<other>_regionprops_indices``).
    - The table must describe exactly these samples and the selected masks: ``obs["image"]`` is
      ``<sample>.tiff`` and every observation is named
      ``Object <id> in <sample>.tiff``, where ``<id>`` is the object's
      positive mask label (zero is background). ``centroid-0`` and
      ``centroid-1`` are required.
    - Each image and mask has one TIFF series, read at full resolution; the
      image needs OME metadata and the mask the image's ``Y, X`` extent.

    Parameters
    ----------
    path
        Project directory.
    masks_dir
        Directory of the segmentation masks, relative to ``path``, for example
        ``"masks_deepcell"`` in a project that keeps several segmentations.
        It may not be absolute or contain ``..`` and follows ``path_policy``
        like every discovered artifact. Other mask directories are not read.
    table_file
        AnnData table exported by ``steinbock export anndata``. Relative
        paths resolve against ``path``; absolute paths may point elsewhere.
    scale_factors
        Integer factors of at least two, one per generated pyramid level of
        every image and mask; empty for a single scale. Generated mask levels
        require masks of at most 32 bits, because downsampling rounds 64-bit
        IDs above ``2**53``; 64-bit masks are exact at a single scale.
    chunks
        One positive size for every axis or three ``c, y, x`` sizes of the
        images; masks use the ``y`` and ``x`` sizes. By default one channel
        per chunk and square 1024-pixel tiles.
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
        One image and one labels element per sample, in the lexical order of
        the image file names, and one linked table:

        ``images["<sample>_image"]``
            Lazy ``c, y, x`` image with the source dtype and channel order.
            Channels are named by the OME channel names when all are nonempty
            and unique, else by the OME channel IDs; they are not matched to
            the table's features. Other source axes must be singleton (and are
            dropped); singleton ``y`` or ``x`` axes are kept. Steinbock images
            are scientific channel stacks: an RGB TIFF (a non-singleton
            samples axis ``S``) is rejected, and three or four channels are
            never read as colours.
        ``labels["<sample>_labels"]``
            Lazy ``y, x`` mask with its source integer dtype.
        ``tables["table"]``
            The imported table with its row and feature order, linked to the
            labels by categorical ``obs["region"]`` (``<sample>_labels``) and
            ``obs["cell_id"]`` (the mask label; int64, or uint64 when a label
            needs it). IDs may repeat across samples, not within one; mask
            pixels are never read to check them.

            - ``X`` and an optional ``counts`` layer: canonical CSR sparse
              arrays with their source values and dtypes; duplicate sparse
              entries are summed, promoting integers to an exact float dtype
              only if a sum would overflow, and failing when no float dtype
              holds the sums exactly.
            - ``obsm["spatial_pixels"]``: the regionprops centroids as
              ``x, y`` (``centroid-1``, ``centroid-0``) in their source dtype,
              in zero-based pixel-index coordinates, declared as frame
              ``<sample>_regionprops_indices``.
            - ``obsm["spatial"]`` (only when every sample is calibrated):
              float64 ``x, y`` micrometres ``(spatial_pixels + 0.5) *
              spacing``, in the coordinate system of the sample's rasters.
              With several samples, categorical ``obs["spatial_frame"]`` and
              ``obs["spatial_pixel_frame"]`` name each row's frames.
            - ``uns["spatialdata_io_positions"]`` declares both arrays; an
              imported ``obsm["spatial"]`` moves unchanged to
              ``obsm["spatial_source"]``.
            - The redundant ``obs["Image"]`` column is dropped (it must equal
              ``obs["image"]``). ``Final Concentration / Dilution`` becomes
              ``Final_Concentration``, and other imported ``obs`` and ``var``
              column names that a SpatialData table does not accept are
              sanitized with a warning (``Channel Name`` becomes
              ``Channel_Name``; a name that then collides, ignoring letter
              case, with another column or a reader-owned one such as
              ``region`` gets a ``_1``, ``_2``, ... suffix). Everything else,
              including other layers, mappings and ``raw``, is kept unchanged.

        Coordinate systems: a sample's image and mask share one frame with
        its native pixel origin. A calibrated sample is scaled to
        micrometres, from the ``images.csv`` acquisition extents divided by
        its pixel dimensions, or else from complete OME
        ``PhysicalSizeX``/``PhysicalSizeY`` of the image; every other present
        OME size must agree. Its micrometre frame is ``global`` when one
        sample is read and ``<sample>`` otherwise. An uncalibrated sample uses
        the identity into the pixel frame ``<sample>_pixels``, also when it is
        the only sample, so ``global`` and ``<sample>`` always mean
        micrometres. Samples are never registered to each other.

        ``attrs`` record the reader and package version. Pixels are not read
        until computed, and a partial computation decodes only the TIFF tiles
        or strips it overlaps. All files are closed on return; the source
        files are not modified.

    Warns
    -----
    UserWarning
        Once per call if duplicate sparse entries of ``X`` or ``counts`` were
        summed, naming the affected slots and dtype changes; once per
        sanitized sample name; once per frame if ``obs`` or ``var`` column
        names were sanitized.

    Raises
    ------
    TypeError
        If an option has the wrong type.
    ValueError
        If an option has an invalid value, or a requested pyramid level
        would downsample 64-bit masks or shrink a sample below one pixel.
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        If an artifact is missing or has the wrong kind.
    ValueError
        A contextual ``ReaderFormatError`` naming the artifact and path if the
        layout, a raster, ``images.csv`` or the table is malformed or they
        disagree, for example unmatched samples, an RGB or non-OME image,
        conflicting calibrations, malformed observation names, out-of-range
        IDs or centroids, nonfinite expression, or occupied output columns.

    Notes
    -----
    ``labels_kind`` was removed. It only chose between the directories
    ``masks_deepcell`` and ``masks_ilastik``, which are not Steinbock
    defaults. Pass ``masks_dir="masks_deepcell"`` (or ``"masks_ilastik"``)
    instead; a project with Steinbock's default ``masks/`` needs no option.
    """
    masks_dir = validate_masks_dir(masks_dir)
    table_file = validate_table_file(table_file)
    raster = make_raster_options(scale_factors, chunks)
    layout = discover_layout(path, masks_dir=masks_dir, table_file=table_file, path_policy=path_policy)

    metadata = (
        read_image_metadata(layout.image_metadata, {sample.image_name for sample in layout.samples})
        if layout.image_metadata is not None
        else {}
    )
    samples = {
        sample.name: read_rasters(
            sample,
            raster,
            metadata=metadata.get(sample.image_name),
            calibrated_frame="global" if len(layout.samples) == 1 else sample.name,
        )
        for sample in layout.samples
    }
    grids = {name: sample.grid for name, sample in samples.items()}
    table = read_table(layout.table, grids)
    set_positions(table, grids, layout.table)

    # read_table validated eager frames; AnnData also types backed Dataset2D.
    prepare_anndata_dataframe(cast("pd.DataFrame", table.obs), context=_LINKAGE_CONTEXT)
    prepare_anndata_dataframe(cast("pd.DataFrame", table.var), context=_LINKAGE_CONTEXT)
    duplicate_sums: list[SparseDuplicateSummary] = []
    table = parse_linked_table(
        table,
        TableLinkage(tuple(table.obs["region"].unique()), "region", "cell_id"),
        context=_LINKAGE_CONTEXT,
        path=layout.table,
        duplicate_sums=duplicate_sums,
        expression_layers=("counts",) if "counts" in table.layers else (),
    )

    result = SpatialData(
        images={f"{name}_image": sample.image for name, sample in samples.items()},
        labels={f"{name}_labels": sample.labels for name, sample in samples.items()},
        tables={"table": table},
    )
    set_reader_provenance(result.attrs, reader="steinbock")
    if duplicate_sums:
        warnings.warn(
            f"steinbock {layout.table.name}: " + "; ".join(map(str, duplicate_sums)),
            UserWarning,
            stacklevel=2,
        )
    return result
