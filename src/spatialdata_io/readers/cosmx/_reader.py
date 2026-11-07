"""Assemble one CosMx flat-file export into placed rasters, points and a table.

The public reader validates every option, discovers the export once, reads the
eager cell metadata first (so missing labels fail before counts are parsed),
derives every FOV placement from the cells, then builds lazy rasters, the
table and lazy points, and links the table to the labels once.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from spatialdata import SpatialData

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.names import sanitize_element_name, validate_element_name
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers.cosmx._layout import discover_layout, validate_file_prefix
from spatialdata_io.readers.cosmx._rasters import read_image, read_labels
from spatialdata_io.readers.cosmx._schemas import LABELS, validate_label_ids
from spatialdata_io.readers.cosmx._tables import build_table, read_counts, read_fov_positions, read_metadata
from spatialdata_io.readers.cosmx._transcripts import read_transcripts
from spatialdata_io.readers.cosmx._transforms import solve_placements

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from dask.dataframe import DataFrame
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import RasterOptions
    from spatialdata_io.readers.cosmx._layout import CellLabels, CosmxLayout
    from spatialdata_io.readers.cosmx._tables import CellMetadata
    from spatialdata_io.readers.cosmx._transforms import Placement


def cosmx(  # noqa: PLR0913, RUF100 - Explicit supported format options.
    path: str | Path,
    *,
    file_prefix: str | None = None,
    dataset_id: str | None = None,
    include_transcripts: bool = True,
    pixel_size_um: float | None = None,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
    path_policy: Literal["strict", "relaxed"] = "strict",
) -> SpatialData:
    """Read a CosMx Spatial Molecular Imager flat-file export.

    Supported exports are the 2021 flat-file release and AtoMx flat-file
    exports, recognized by their columns; no version label is read. Every flat
    file may be ``.csv`` or ``.csv.gz``::

        export/
        ├── <prefix>_exprMat_file.csv
        ├── <prefix>_metadata_file.csv
        ├── <prefix>_fov_positions_file.csv      # optional
        ├── <prefix>_tx_file.csv                 # with include_transcripts
        ├── CellComposite/<any>_F<fov>.jpg       # optional; .jpeg, .png or .tif
        └── CellLabels/<any>_F<fov>.tif          # optional; .tiff

    Polygon, overlay, compartment and morphology files are not read.

    Parameters
    ----------
    path
        Export directory.
    file_prefix
        Prefix of the flat files to read, for example ``"Lung5_Rep2"``; any
        characters except path separators and NUL. ``None`` uses the prefix of
        the only ``*_exprMat_file.csv[.gz]``.
    dataset_id
        Prefix of the transcripts element name ``<dataset_id>_transcripts``; a
        valid SpatialData element name. ``None`` uses the file prefix, with
        characters SpatialData does not accept replaced by ``_``.
    include_transcripts
        Return the lazy transcript points.
    pixel_size_um
        Positive finite micrometres per global pixel, from the caller's
        acquisition or export evidence. It places every element in ``global``
        micrometres. ``None`` keeps the global pixel frame ``global_pixels``.
    scale_factors
        Integer factors of at least two, one per generated pyramid level of the
        images and labels; empty for a single scale.
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
    spatialdata.SpatialData
        The linked table, optional rasters and optional transcript points:

        - ``tables["table"]``: one row per metadata cell, in metadata order.
          ``X`` is an eager canonical CSR count array, float32 when exact and
          otherwise float64. Features are every count column except ``fov`` and
          ``cell_ID``, in source order, including negative and system controls.
          The per-FOV background rows (``cell_ID`` 0) are removed, and the
          remaining count and metadata populations must be identical.
          Observation names are the vendor ``cell`` IDs when present, else row
          positions ``"0"``, ``"1"``, ...
        - ``obs``: int64 ``fov`` and ``cell_ID``; nullable float64 ``Area``,
          ``Area.um2``, ``AspectRatio``, ``Width``, ``Height`` and
          ``Mean.*``/``Max.*``; every other metadata column as text (an exact
          AtoMx ``cell_id`` copy of ``cell`` is dropped); categorical
          ``region`` (``<fov>_labels``) and ``spatial_pixel_frame`` (the FOV's
          local frame ``fov_<n>_pixels``, categories in FOV order). With labels, the table links each cell to
          ``<fov>_labels`` by ``cell_ID``; IDs must fit that FOV's label dtype.
          Without a ``CellLabels`` directory the table is unlinked.
        - ``obsm``: ``spatial_pixels`` local cell centres and
          ``spatial_global_pixels`` global cell centres, both float64 pixels;
          with ``pixel_size_um`` also ``spatial``, global centres in
          micrometres. ``uns["spatialdata_io_positions"]`` declares their units
          and frames.
        - ``images["<fov>_image"]``: lazy ``(c, y, x)`` composite with source
          pixels and dtype. Channels are ``r, g, b(, a)`` when the file declares
          RGB(A) samples, otherwise positional.
        - ``labels["<fov>_labels"]``: lazy integer ``(y, x)`` labels on the
          image grid.
        - ``points["<dataset_id>_transcripts"]``: lazy points at the global
          pixel position, with categorical ``target`` (categories are the count
          features), int64 ``fov`` and ``cell_ID`` (0 for unassigned), float64
          ``x_local`` and ``y_local``, and, when present, int64 ``z_raw`` (the
          vendor's slice number), text ``CellComp`` and ``cell``.

    Raises
    ------
    TypeError
        If an option has the wrong type.
    ValueError
        If an option has an invalid value, or ``pixel_size_um`` makes a cell
        coordinate nonfinite or underflow to zero.
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        If a required artifact is missing or has the wrong filesystem kind.
    spatialdata_io.readers._utils.errors.ReaderFormatError
        If the layout, a schema, a value, the cell populations, the label
        coverage or the FOV placement violates the supported contract. Values
        in transcript partitions fail when the partition is computed.

    Warns
    -----
    UserWarning
        If the default ``dataset_id`` is repaired from the file prefix, or
        cell metadata column names are sanitized.

    Notes
    -----
    Coordinate systems:

    - ``fov_<n>_pixels`` (for example ``"fov_2_pixels"``): each FOV's local
      pixel frame. Local
      coordinates have a reversed vertical axis: the pixel at row ``r`` and
      column ``c`` of an ``H``-row FOV is at ``(c, H - r)``, as measured on the
      2021 release and an AtoMx export. Raster pixels are returned unflipped.
    - ``global_pixels`` without ``pixel_size_um``, or ``global`` with it:
      the export's global frame in pixels or micrometres. ``global`` always
      means micrometres, so an uncalibrated read has no ``global``.

    Each FOV's local-to-global map is ``global = ±local + offset`` per axis,
    derived from every metadata cell and accepted only when one offset lies
    within all printed rounding bounds; no affine is fitted. Pixel FOV
    positions must agree with the cells and place FOVs without cells. Wholly
    cell-empty exports cannot determine the axis signs and are unsupported.
    Rasters in exports with a reversed global axis fail: their raster
    convention has not been evidenced. AtoMx 3.10.5 pancreas coordinates are
    not a translation and remain unsupported. No scale is inferred from
    stage positions, millimetre fields or TIFF metadata.

    Building the result reads the transcript header and no pixel or transcript
    task runs. Plain transcript CSV partitions hold about 8 MiB of records.
    Gzip transcripts are read sequentially as concatenated record-aligned
    members, with every member's CRC verified; a single-stream gzip file forms
    one partition. JPEG and PNG composites decode their whole frame for any
    selection.
    """
    options = _validate_options(
        file_prefix=file_prefix,
        dataset_id=dataset_id,
        include_transcripts=include_transcripts,
        pixel_size_um=pixel_size_um,
        scale_factors=scale_factors,
        chunks=chunks,
    )
    layout = discover_layout(
        path, file_prefix=file_prefix, include_transcripts=include_transcripts, path_policy=path_policy
    )
    transcripts = None
    if layout.transcripts is not None:
        transcripts = (_transcripts_name(dataset_id, layout.file_prefix), layout.transcripts)

    metadata = read_metadata(layout.metadata)
    if layout.labels is not None:
        _require_labels(metadata, layout.labels)
    counts = read_counts(layout.counts)
    placements = _placements(layout, metadata)
    images, labels = _rasters(layout, metadata, placements, options=options, pixel_size_um=pixel_size_um)
    table = build_table(
        counts, metadata, path=layout.metadata, linked=layout.labels is not None, pixel_size_um=pixel_size_um
    )
    points: dict[str, DataFrame] = {}
    if transcripts is not None:
        name, source = transcripts
        points[name] = read_transcripts(
            source, features=counts.features, placements=placements, pixel_size_um=pixel_size_um
        )

    result = SpatialData(images=images, labels=labels, points=points, tables={"table": table})
    set_reader_provenance(result.attrs, reader="cosmx")
    return result


def _validate_options(  # noqa: PLR0913, RUF100 - Mirrors the public reader options.
    *,
    file_prefix: object,
    dataset_id: object,
    include_transcripts: object,
    pixel_size_um: object,
    scale_factors: Sequence[int],
    chunks: int | tuple[int, int, int],
) -> RasterOptions:
    """Validate the public options in signature order, before any I/O; ``path_policy`` is checked by discovery."""
    validate_file_prefix(file_prefix)
    if dataset_id is not None:
        _element_name(validate_element_name(dataset_id, parameter="dataset_id"))
    if not isinstance(include_transcripts, bool):
        message = f"include_transcripts must be a boolean; found {type(include_transcripts).__name__}."
        raise TypeError(message)
    if pixel_size_um is not None:
        message = "pixel_size_um must be a positive finite number or None."
        if isinstance(pixel_size_um, bool) or not isinstance(pixel_size_um, int | float):
            raise TypeError(message)
        if not math.isfinite(pixel_size_um) or pixel_size_um <= 0:
            raise ValueError(message)
    return make_raster_options(scale_factors, chunks)


def _element_name(dataset_id: str) -> str:
    """Return the points element name ``<dataset_id>_transcripts`` for a valid ``dataset_id``."""
    # A valid dataset_id "_" still starts the name with "__", which SpatialData reserves.
    return validate_element_name(
        f"{dataset_id}_transcripts", parameter="the transcripts element name '<dataset_id>_transcripts'"
    )


def _transcripts_name(dataset_id: str | None, file_prefix: str) -> str:
    """Name the points from the caller's ``dataset_id`` or, by default, the repaired file prefix."""
    if dataset_id is None:
        dataset_id = sanitize_element_name(file_prefix, source="cosmx: the file prefix", parameter="dataset_id")
    return _element_name(dataset_id)


def _placements(layout: CosmxLayout, metadata: CellMetadata) -> dict[int, Placement]:
    """Place every FOV with cells or rasters, rejecting rasters without evidenced orientation."""
    positions = {} if layout.fov_positions is None else read_fov_positions(layout.fov_positions)
    label_files = () if layout.labels is None else layout.labels.files
    raster_fovs = frozenset(item.fov for item in (*layout.images, *label_files))
    placements = solve_placements(
        metadata.fovs, metadata.samples, required=raster_fovs, positions=positions, path=layout.metadata
    )
    if raster_fovs and any((p.sign_x, p.sign_y) != (1, 1) for p in placements.values()):
        raise ReaderFormatError(
            context=ReaderErrorContext("cosmx", "FOV rasters"),
            path=layout.metadata,
            found="rasters in an export whose global axes are reversed relative to local axes",
            expected="rasters only in exports with the evidenced local and global orientation",
        )
    return placements


def _rasters(
    layout: CosmxLayout,
    metadata: CellMetadata,
    placements: Mapping[int, Placement],
    *,
    options: RasterOptions,
    pixel_size_um: float | None,
) -> tuple[dict[str, DataArray | DataTree], dict[str, DataArray | DataTree]]:
    """Build the lazy FOV images and labels; linked cell IDs must fit each label dtype."""
    images: dict[str, DataArray | DataTree] = {}
    grids: dict[int, tuple[int, int]] = {}
    for item in layout.images:
        images[f"{item.fov}_image"], grids[item.fov] = read_image(
            item.path, fov=item.fov, placement=placements[item.fov], options=options, pixel_size_um=pixel_size_um
        )
    labels: dict[str, DataArray | DataTree] = {}
    for item in () if layout.labels is None else layout.labels.files:
        labels[f"{item.fov}_labels"], dtype = read_labels(
            item.path,
            fov=item.fov,
            grid=grids.get(item.fov),
            placement=placements[item.fov],
            options=options,
            pixel_size_um=pixel_size_um,
        )
        # The inspected dtype bounds the IDs without reading label pixels.
        validate_label_ids(metadata.cell_ids[metadata.fovs == item.fov], dtype, item.path)
    return images, labels


def _require_labels(metadata: CellMetadata, labels: CellLabels) -> None:
    """Require a label raster for every FOV with cells, before counts or rasters are read."""
    missing = sorted(set(metadata.fovs.tolist()) - {item.fov for item in labels.files})
    if missing:
        raise ReaderFormatError(
            context=ReaderErrorContext("cosmx", LABELS),
            path=labels.directory,
            found=f"no labels for FOVs {missing}",
            expected="labels for every FOV with cells",
        )
