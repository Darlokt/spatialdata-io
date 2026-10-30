"""Assemble DBiT expression, grid geometry, optional image, and final linkage."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
from spatialdata import SpatialData
from xarray import DataTree

from spatialdata_io.readers._utils.errors import ReaderErrorContext, add_reader_context
from spatialdata_io.readers._utils.geometry import validate_invalid_geometry_policy
from spatialdata_io.readers._utils.image import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS, make_raster_options
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table, prepare_anndata_dataframe
from spatialdata_io.readers.dbit._barcodes import _read_barcodes
from spatialdata_io.readers.dbit._geometry import _make_grid
from spatialdata_io.readers.dbit._images import _read_image, _validate_image_options
from spatialdata_io.readers.dbit._layout import _discover_layout
from spatialdata_io.readers.dbit._schemas import GRID_DTYPE, SAMPLE_DTYPE, _validate_output_column
from spatialdata_io.readers.dbit._tables import _read_table

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy.typing as npt
    from xarray import DataArray

_MAX_BORDER_SCALE = 4


def _validate_dataset_id(dataset_id: str | None) -> None:
    """Reject names that cannot identify the returned spatial grid."""
    if dataset_id is not None and (
        not isinstance(dataset_id, str)
        or not dataset_id
        or dataset_id.startswith("__")
        or dataset_id.casefold() in {".", "..", "table"}
        or not all(character.isalnum() or character in "_-." for character in dataset_id)
    ):
        message = (
            "dataset_id must be nonempty, contain only letters, digits, '_-.', not start with '__', "
            "and not be '.', '..', or 'table' (case-insensitive)."
        )
        raise ValueError(message)


def _validate_grid_options(dataset_id: str | None, *, include_border: bool, border_scale: float) -> None:
    """Reject invalid names and nonfinite, negative, or collapsed border widths."""
    _validate_dataset_id(dataset_id)

    if not isinstance(include_border, bool):
        message = "include_border must be a boolean."
        raise TypeError(message)

    if isinstance(border_scale, bool) or not isinstance(border_scale, (int, float)):
        message = "border_scale must be a real number."
        raise TypeError(message)
    if not isfinite(border_scale) or not 0 <= border_scale < _MAX_BORDER_SCALE:
        message = "border_scale must be finite and in [0, 4), leaving a positive square width."
        raise ValueError(message)


def dbit(  # noqa: PLR0913, RUF100 - explicit format choices keep the public boundary typed.
    path: str | Path,
    *,
    table_file: str | Path | None = None,
    barcode_positions_file: str | Path | None = None,
    image_file: str | Path | None = None,
    dataset_id: str | None = None,
    include_border: bool = True,
    border_scale: float = 1.0,
    invalid_geometries: Literal["keep", "drop", "raise"] = "keep",
    data_axes: str | None = None,
    tiff_series: int = 0,
    tiff_level: int = 0,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
) -> SpatialData:
    """Read an unversioned DBiT H5AD export and its barcode-position grid.

    The supported layout has one H5AD expression table, one headerless barcode
    map, and optionally a cropped, registered tissue image::

        sample.h5ad
        barcode_list.txt
        tissue_lowres_image.png  # optional

    Discovery matches the H5AD suffix, names containing ``barcode_list``, and
    names ending in ``tissue_lowres_image.png``. Multiple matches require an
    override, including for images. No pipeline version is inferred.
    The barcode map has two tab-separated columns: position and eight DNA
    bases. Positions must be A1..AN and B1..BN, with the same contiguous extent
    and 1 <= N <= 50. Sequences are case-insensitive and unique within each axis.
    Observation names begin with eight B bases followed by eight A bases.
    Optional suffixes are preserved. Each grid location occurs at most once.

    Parameters
    ----------
    path
        Required dataset directory. Discovered symlinks must remain inside it.
    table_file, barcode_positions_file
        Optional artifact overrides. Relative paths resolve against the dataset
        root. Absolute overrides may point outside it.
    image_file
        Optional TIFF, OME-TIFF, BigTIFF/BTF, PNG, or JPEG override, with the
        same path policy. The image must already be cropped and registered.
    dataset_id
        Grid name and sample label. Defaults to the H5AD filename stem.
        Names contain only letters, digits, underscores, dots or hyphens and
        cannot start with two underscores. The names ``.``, ``..``, and
        ``table`` (case-insensitive) are reserved.
    include_border
        Shrink each square inward on all sides by one eighth of its edge
        length multiplied by ``border_scale``.
    border_scale
        Finite inset multiplier in [0, 4). Zero gives touching squares.
        Values whose remaining width rounds to zero are rejected.
    invalid_geometries
        Polygon topology policy. Supported squares are valid by construction,
        so all three modes return identical geometry and table linkage. Loss
        of representable positive square width always fails.
    data_axes
        Optional TIFF source axes such as ``CYX`` for an unlabeled stack.
        Otherwise metadata determines axes. Only singleton extra axes may be
        removed. Singleton spatial and channel axes are retained.
    tiff_series, tiff_level
        Nonnegative TIFF series and stored pyramid-level indices. Both default
        to zero. Nonzero selections require a TIFF image.
    scale_factors
        Integer factors of at least two, one per generated pyramid level of the
        images; empty for a single scale.
    chunks
        One positive size for every axis or three ``c, y, x`` sizes; by default
        one channel per chunk and square 1024-pixel tiles.

        Image options other than their defaults require an image; axis and
        TIFF selections require a TIFF image.

    Returns
    -------
    SpatialData
        ``tables['table']`` contains eager canonical CSR sparse-array ``X``
        and, if present, ``layers['counts']``, preserving finite source values,
        dtype and feature order. Rows are ordered by A then B. Observation,
        feature and auxiliary metadata follow that order. Other layers retain
        their representation. Dense expression converts before row reordering
        so sorting does not make a second dense expression copy.
        Duplicate sparse entries are summed. Integer sums outside the source
        dtype range are rejected rather than wrapped.

        ``array_A`` and ``array_B`` are one-based int64 grid coordinates.
        ``obsm['spatial_grid']`` retains these columns in A/B order, described as
        grid ordinals in the ``barcode_grid`` frame. Imported ``spatial`` moves
        exactly to ``spatial_source`` before sorting; occupied destinations fail.
        Physical ``spatial`` is absent. ``pixel_id``
        is the zero-based sorted row index. ``sample`` and categorical ``region``
        contain the grid name. Previous source linkage is replaced with exact
        membership in the new grid. Additional H5AD annotation fields accept
        arbitrary order and retain source dtypes and missingness. Duplicate
        annotation names and conflicting reserved output columns fail. Equal
        reserved values are accepted. Region/pixel_id columns belonging to a
        declared previous linkage are replaced together with that linkage.

        ``shapes[dataset_id]`` contains clockwise squares, with A mapped to x
        and B to y. Without an image a grid step is one unit. With an image,
        the uniform step is ``max(height, width) / N`` for the selected raster
        level, preserving the square-grid convention even for rectangular
        images. Image channel count never affects scale. No physical units,
        anisotropic stretching, registration or origin shift are inferred.

        Optional ``images['<dataset_id>_image']`` is lazy CYX, preserving source
        dtype and channel order, positional RGB labels, and available OME
        channel names or IDs. It is a single scale unless ``scale_factors``
        requests pyramid levels; the grid step uses its full-resolution extent.
        Grid and image use identity transforms to ``global``. Construction
        executes zero pixel tasks. TIFF partial computations decode only the
        tiles or strips they overlap. PNG/JPEG computations decode a whole
        frame. Resources close after inspection or inside each
        lazy task. Caller-owned state and source files are never mutated.
        Provenance records the reader and package version.

    Raises
    ------
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        If an artifact is missing or has the wrong filesystem kind.
    ValueError
        If options, table values, barcodes, axes or linkage are invalid.
        Malformed or ambiguous layouts and schemas raise the
        ``ReaderFormatError`` subclass with artifact context.
    TypeError
        If option or expression types are unsupported.
    """
    validate_invalid_geometry_policy(invalid_geometries)
    _validate_grid_options(dataset_id, include_border=include_border, border_scale=border_scale)
    _validate_image_options(data_axes, tiff_series, tiff_level)
    raster = make_raster_options(scale_factors, chunks)

    layout = _discover_layout(
        path, table_file=table_file, barcode_positions_file=barcode_positions_file, image_file=image_file
    )
    if dataset_id is None:
        dataset_id = layout.table.stem
        _validate_dataset_id(dataset_id)

    image_options = data_axes is not None or tiff_series != 0 or tiff_level != 0 or raster != make_raster_options()
    if layout.image is None and image_options:
        message = "data_axes, tiff_series, tiff_level, and nondefault scale_factors or chunks require an image."
        raise ValueError(message)

    barcodes = _read_barcodes(layout.barcode_positions)
    table = _read_table(layout.table, barcodes)

    images: dict[str, DataArray | DataTree] = {}
    scale = 1.0
    if layout.image is not None:
        image = _read_image(
            layout.image, data_axes=data_axes, tiff_series=tiff_series, tiff_level=tiff_level, raster=raster
        )
        images[f"{dataset_id}_image"] = image

        # Preserve the square-grid convention for a pre-registered crop. Use
        # the selected level's full-resolution extent, independent of channel
        # count and of generated pyramid levels.
        scale0 = next(iter(image["scale0"].data_vars.values())) if isinstance(image, DataTree) else image
        scale = max(scale0.sizes["y"], scale0.sizes["x"]) / len(barcodes.a)

    # _read_table produces these exact int64 coordinates in final row order.
    coordinates = cast("npt.NDArray[np.int64]", table.obsm["spatial_grid"])
    grid = _make_grid(coordinates, scale=scale, include_border=include_border, border_scale=border_scale)

    # _read_table validates eager frames; AnnData also types backed Dataset2D.
    obs, var = cast("pd.DataFrame", table.obs), cast("pd.DataFrame", table.var)

    regions = pd.Series(pd.Categorical([dataset_id] * table.n_obs), index=obs.index)
    samples = pd.Series(dataset_id, index=obs.index, dtype=SAMPLE_DTYPE)
    identifiers = pd.Series(np.arange(table.n_obs, dtype=np.int64), index=obs.index, dtype=GRID_DTYPE)
    for key, values in (("region", regions), ("sample", samples), ("pixel_id", identifiers)):
        _validate_output_column(obs, key, values, path=layout.table)
        obs[key] = values

    # Link once, after sorting and constructing the independent geometry target.
    context = ReaderErrorContext("dbit", "linked table")
    prepare_anndata_dataframe(obs, context=context)
    prepare_anndata_dataframe(var, context=context)

    try:
        table = parse_linked_table(
            table,
            TableLinkage((dataset_id,), "region", "pixel_id"),
            context=context,
            expression_layers=("counts",) if "counts" in table.layers else (),
            expected_instance_ids={dataset_id: grid.index},
        )
    except (TypeError, ValueError) as error:
        add_reader_context(error, context=context, path=layout.table)
        raise

    result = SpatialData(images=images, shapes={dataset_id: grid}, tables={"table": table})
    set_reader_provenance(result.attrs, reader="dbit")
    return result
