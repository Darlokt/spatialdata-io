"""Assemble a DBiT grid, its expression table and an optional tissue image.

One call validates every option, discovers the layout, reads the barcode map
and the H5AD table, optionally reads the image to size the grid, builds the
squares, links the table to them and sorts its rows by A and then B.
"""

from __future__ import annotations

import warnings
from math import isfinite
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
from spatialdata import SpatialData
from spatialdata.transformations import Identity, Scale
from xarray import DataTree

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.geometry import validate_invalid_geometry_policy
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.names import sanitize_element_name, validate_element_name
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table, prepare_anndata_dataframe
from spatialdata_io.readers.dbit._barcodes import read_barcodes
from spatialdata_io.readers.dbit._geometry import make_grid
from spatialdata_io.readers.dbit._images import read_image, validate_image_options
from spatialdata_io.readers.dbit._layout import discover_layout
from spatialdata_io.readers.dbit._schemas import GRID_DTYPE, GRID_FRAME, IMAGE_FRAME, SAMPLE_DTYPE
from spatialdata_io.readers.dbit._tables import grid_order, read_table, sort_rows

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy.typing as npt
    from spatialdata.transformations import BaseTransformation
    from xarray import DataArray

    from spatialdata_io.readers._utils.table import SparseDuplicateSummary

_MAX_BORDER_SCALE = 4
_TABLE_CONTEXT = ReaderErrorContext("dbit", "expression table")
# The grid shares SpatialData's case-insensitive namespace with the table.
_RESERVED_NAMES = ("table",)


def _validate_options(dataset_id: str | None, *, include_border: bool, border_scale: float) -> None:
    """Validate the grid name and border options before any I/O."""
    if dataset_id is not None:
        validate_element_name(dataset_id, parameter="dataset_id", reserved=_RESERVED_NAMES)

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
    tiff_series: int | None = None,
    tiff_level: int = 0,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
    path_policy: Literal["strict", "relaxed"] = "strict",
) -> SpatialData:
    """Read a DBiT-seq H5AD export, its barcode map and an optional tissue image.

    The layout is unversioned. Only the top level of ``path`` is searched::

        sample.h5ad  # the only *.h5ad
        barcode_list.txt  # the only name containing "barcode_list"
        tissue_lowres_image.png  # optional: the only name ending in this

    The barcode map is a headerless two-column TSV of positions ``A1..AN`` and
    ``B1..BN`` (``1 <= N <= 50``, contiguous, each listed once) and their
    eight-base sequences (case-insensitive, unique within an axis), in any row
    order. Each H5AD observation name starts with the eight B bases followed by
    the eight A bases of its spot; a suffix is kept. Every observation must map
    to a distinct grid location; the table may cover any subset of the grid.

    Parameters
    ----------
    path
        Dataset directory.
    table_file, barcode_positions_file
        H5AD table or barcode map to read instead of the discovered one, for
        example when several match. Relative paths resolve against ``path``;
        absolute paths may point elsewhere.
    image_file
        TIFF, OME-TIFF, BigTIFF/BTF, PNG or JPEG image to read instead of the
        discovered one, resolved like ``table_file``. It must exist. The image
        must already be cropped to the grid and registered to it.
    dataset_id
        Name of the shapes element and of the table's ``sample`` and
        ``region`` values. Must be a valid SpatialData element name: nonempty,
        only letters, digits, ``_``, ``-`` and ``.``, not ``.`` or ``..``, not
        starting with ``__``, and not ``table`` in any letter case; it is
        checked before any I/O. Defaults to the H5AD filename stem, repaired
        with one warning if it breaks these rules: other characters become
        ``_``, a leading run of underscores becomes one, an empty name, ``.``
        or ``..`` becomes ``unnamed``, and a reserved name gains a trailing
        ``_`` (``my run.h5ad`` gives ``my_run``).
    include_border
        Shrink every square inward on all sides by one eighth of the grid step
        times ``border_scale``.
    border_scale
        Finite inset multiplier in ``[0, 4)``, checked even without a border.
        Zero gives touching squares. A width lost to floating-point rounding
        raises.
    invalid_geometries
        Polygon validity policy. The squares are valid by construction, so
        ``"keep"``, ``"drop"`` and ``"raise"`` return identical results.
    data_axes
        Source TIFF axes as one string of distinct letters including ``X`` and
        ``Y``, case-insensitive, for example ``"CYX"`` for a stack without axis
        metadata; ``S`` (samples) counts as a channel axis. Defaults to the
        axes the file declares. Requires a TIFF image.
    tiff_series
        Zero-based TIFF series. Required when the file has several series.
        Requires a TIFF image.
    tiff_level
        Zero-based stored pyramid level of the selected series. Nonzero values
        require a TIFF image.
    scale_factors
        Integer factors of at least two, one per generated pyramid level of the
        image; empty for a single scale. Nondefault values require an image.
    chunks
        One positive size for every axis or three ``c, y, x`` sizes; by default
        one channel per chunk and square 1024-pixel tiles. Nondefault values
        require an image.
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
        One linked table, one shapes element and an optional image:

        ``tables["table"]``
            One row per observation, sorted by A and then B; feature order and
            all other observation- and feature-aligned data follow the source.

            - ``X`` and, if present, ``layers["counts"]``: canonical finite CSR
              sparse arrays with the source dtype. Duplicate sparse entries
              are summed; an integer sum that overflows promotes to the
              smallest exact float32 or float64 storage. Other layers keep
              their representation.
            - ``obs["array_A"]``, ``obs["array_B"]``: one-based int64 grid
              positions; ``obsm["spatial_grid"]`` holds them as int64 ``x, y``
              (A, B), declared in ``uns["spatialdata_io_positions"]`` as
              ordinals of the ``grid_pixels`` frame, also with an image.
            - ``obs["pixel_id"]``: int64 row number, matching the shapes index;
              ``obs["sample"]`` (string) and categorical ``obs["region"]``:
              ``dataset_id``. These five columns replace any imported columns
              of the same name, and any imported ``uns["spatialdata_attrs"]``
              is replaced by the new linkage.
            - An imported ``obsm["spatial"]`` moves unchanged to
              ``obsm["spatial_source"]``; no physical ``spatial`` is produced.
              Other imported fields keep their dtypes and missing values.
              Imported ``obs`` and ``var`` column names and top-level ``uns``
              keys that a SpatialData table does not accept (``my note``) are
              sanitized with a warning (``my_note``; a name that then
              collides, ignoring letter case, with another one, one of the
              five reader columns or the reader's ``uns`` keys
              ``spatialdata_io_positions`` and ``spatialdata_attrs`` gets a
              ``_1``, ``_2``, ... suffix). ``uns`` values, their order and
              nested keys stay as stored; SpatialData checks only top-level
              keys.

        ``shapes[dataset_id]``
            One square per observation, indexed by ``pixel_id``, with vertices
            in grid steps whether or not an image is read. A is ``x`` and B is
            ``y``; location ``(A, B)`` covers ``[A - 1, A] x [B - 1, B]``
            before the border inset.
        ``images[dataset_id + "_image"]`` (only with an image)
            Lazy CYX image of the selected TIFF series and level, or of the
            PNG or JPEG file, with the source dtype and channel order. Samples
            the file declares RGB are named ``r, g, b`` (``r, g, b, a`` with
            declared alpha): TIFF ``PhotometricInterpretation`` RGB or
            JPEG-compressed YCbCr, PNG colour types 2, 3 and 6, and
            three-component JPEG. Otherwise OME channel names, or IDs, name the
            channels, and an OME channel with several samples names them
            ``<name or ID>:sample:<index>``; other images use positions. A
            pyramid is built only when ``scale_factors`` asks for one.

        Coordinate systems: DBiT files carry no physical calibration, so no
        element uses ``global`` (micrometres) and no registration is inferred.

        - ``grid_pixels``, one unit per barcode-grid step: the shapes, by
          identity;
        - ``image_pixels`` (only with an image), the full-resolution pixels of
          the image: the image by identity and the shapes by
          ``Scale([s, s], axes=("x", "y"))`` with grid step
          ``s = max(height, width) / N``, so the grid spans the longer side of
          the image on both axes.

        ``attrs`` record the reader and package version; the layout has no
        format version. Image pixels are not read until computed: TIFF
        computations decode only the tiles or strips they overlap, PNG and
        JPEG decode the whole frame. Files are closed on return and the source
        H5AD is not modified.

    Warns
    -----
    UserWarning
        Once per call if duplicate sparse entries of the imported expression
        were summed, naming the slots and any dtype change; once if the H5AD
        filename stem had to be repaired, naming both names; once each if
        ``obs`` or ``var`` column names or top-level ``uns`` keys were
        sanitized.

    Raises
    ------
    TypeError
        If an option has the wrong type, or expression storage is unsupported.
    ValueError
        If an option has an invalid value, an image option is given without a
        suitable image, expression values are not finite, or a square loses its width.
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        If a path is missing or has the wrong kind.
    ValueError
        A contextual ``ReaderFormatError`` naming the artifact and path if an
        artifact is ambiguous or malformed, for example several tables, a
        multi-series TIFF without ``tiff_series``, an invalid barcode map,
        observation names whose barcodes are malformed, duplicated or not in
        the map, or occupied coordinate destinations.
    """
    _validate_options(dataset_id, include_border=include_border, border_scale=border_scale)
    validate_invalid_geometry_policy(invalid_geometries)
    validate_image_options(data_axes, tiff_series, tiff_level)
    raster = make_raster_options(scale_factors, chunks)

    layout = discover_layout(
        path,
        table_file=table_file,
        barcode_positions_file=barcode_positions_file,
        image_file=image_file,
        path_policy=path_policy,
    )
    if dataset_id is None:
        dataset_id = sanitize_element_name(
            layout.table.stem,
            source="dbit: the H5AD filename stem",
            parameter="dataset_id",
            reserved=_RESERVED_NAMES,
        )
    image_options = (
        data_axes is not None
        or tiff_series is not None
        or tiff_level != 0
        or raster.scale_factors != DEFAULT_SCALE_FACTORS
        or raster.chunks != DEFAULT_CHUNKS
    )
    if layout.image is None and image_options:
        message = "data_axes, tiff_series, tiff_level, and nondefault scale_factors or chunks require an image."
        raise ValueError(message)

    barcodes = read_barcodes(layout.barcode_positions)
    table = read_table(layout.table, barcodes)

    images: dict[str, DataArray | DataTree] = {}
    transformations: dict[str, BaseTransformation] = {GRID_FRAME: Identity()}
    if layout.image is not None:
        image = read_image(
            layout.image, data_axes=data_axes, tiff_series=tiff_series, tiff_level=tiff_level, raster=raster
        )
        images[f"{dataset_id}_image"] = image
        # The grid spans the longer side of the full-resolution image on both
        # axes, independent of channel count and generated pyramid levels.
        scale0 = next(iter(image["scale0"].data_vars.values())) if isinstance(image, DataTree) else image
        step = max(scale0.sizes["y"], scale0.sizes["x"]) / len(barcodes.a)
        transformations[IMAGE_FRAME] = Scale([step, step], axes=("x", "y"))

    # read_table stores exact int64 (A, B) positions in source row order.
    coordinates = cast("npt.NDArray[np.int64]", table.obsm["spatial_grid"])
    order = grid_order(coordinates)
    grid = make_grid(
        coordinates[order], include_border=include_border, border_scale=border_scale, transformations=transformations
    )

    # Link in source order: linkage converts expression to CSR, so the row
    # sort below copies sparse storage, never a dense matrix. pixel_id is each
    # row's position after that sort.
    pixel_ids = np.empty(table.n_obs, dtype=np.int64)
    pixel_ids[order] = np.arange(table.n_obs, dtype=np.int64)
    obs, var = cast("pd.DataFrame", table.obs), cast("pd.DataFrame", table.var)
    obs["region"] = pd.Categorical([dataset_id] * table.n_obs)
    obs["sample"] = pd.Series(dataset_id, index=obs.index, dtype=SAMPLE_DTYPE)
    obs["pixel_id"] = pd.Series(pixel_ids, index=obs.index, dtype=GRID_DTYPE)
    table.uns.pop("spatialdata_attrs", None)
    prepare_anndata_dataframe(obs, context=_TABLE_CONTEXT)
    prepare_anndata_dataframe(var, context=_TABLE_CONTEXT)

    duplicate_sums: list[SparseDuplicateSummary] = []
    table = parse_linked_table(
        table,
        TableLinkage((dataset_id,), "region", "pixel_id"),
        context=_TABLE_CONTEXT,
        path=layout.table,
        expression_layers=("counts",) if "counts" in table.layers else (),
        expected_instance_ids={dataset_id: grid.index},
        duplicate_sums=duplicate_sums,
    )
    table = sort_rows(table, order)

    result = SpatialData(images=images, shapes={dataset_id: grid}, tables={"table": table})
    set_reader_provenance(result.attrs, reader="dbit")
    if duplicate_sums:
        warnings.warn(f"dbit {layout.table.name}: " + "; ".join(map(str, duplicate_sums)), UserWarning, stacklevel=2)
    return result
