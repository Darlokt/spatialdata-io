"""Public CODEX reader: discover one region, read its cell table and image, and link them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from spatialdata import SpatialData
from spatialdata.transformations import Affine, Identity, Scale
from spatialdata.transformations import Sequence as TransformSequence

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table, prepare_anndata_dataframe
from spatialdata_io.readers.codex._calibration import resolve_calibration, validate_calibration
from spatialdata_io.readers.codex._geometry import make_shapes
from spatialdata_io.readers.codex._images import read_image, validate_image_options
from spatialdata_io.readers.codex._layout import discover_layout
from spatialdata_io.readers.codex._tables import read_cell_table, set_positions

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pandas as pd


def codex(  # noqa: PLR0913, RUF100 - explicit format choices keep the public boundary typed.
    path: str | Path,
    *,
    cell_table_file: str | Path | None = None,
    image_file: str | Path | None = None,
    pixel_size_um: float | None = None,
    experiment_file: str | Path | None = None,
    image_to_table: Affine | None = None,
    data_axes: str | None = None,
    tiff_series: int | None = None,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
    path_policy: Literal["strict", "relaxed"] = "strict",
) -> SpatialData:
    """Read one processor-style CODEX region: a cell table and an optional stitched image.

    Supported layout (one region, unversioned processor output)::

        region/
        ├── reg001_compensated.csv  # or .fcs (FCS 3.1); exactly one table
        ├── stitched.tif            # optional TIFF, OME-TIFF, BigTIFF or BTF
        └── experiment.json         # optional Akoya experiment, only read when selected

    The cell table needs the named fields ``cell_id``, ``region``, ``x``, ``y``
    and at least one ``cyc<digits>_ch<digits>`` expression field, in any order;
    all other fields are kept as observation metadata. Akoya CSV headers of the
    form ``short_name:long_label`` are supported when every field is qualified.
    Original Nolan export field names are not supported.

    Parameters
    ----------
    path
        Region directory.
    cell_table_file
        CSV or FCS table to read when the directory holds several (for example
        compensated and uncompensated). Relative paths resolve against
        ``path``; absolute paths may point elsewhere.
    image_file
        Stitched image to read when the directory holds several. Resolved like
        ``cell_table_file``.
    pixel_size_um
        Positive finite micrometres per native table pixel. The caller asserts
        that it belongs to the selected table; there is no default.
    experiment_file
        Akoya experiment JSON (version ``1.5.0.38`` or ``1.7.0.6``) whose
        ``xyResolution`` in nm/pixel calibrates the table. It is never
        discovered automatically; selecting it asserts that it belongs to the
        table. Resolved like ``cell_table_file``. Together with
        ``pixel_size_um``, both must agree exactly (``xyResolution / 1000``).
    image_to_table
        Invertible finite XY ``Affine`` from image pixel coordinates (pixel
        centres at ``column + 0.5, row + 0.5``) to native table ``x, y``.
        Without it the image is not registered to the table. Requires an image.
    data_axes
        Source TIFF axes as one string of distinct letters, for example
        ``"CYX"`` for a stack without axis metadata. Defaults to the axes the
        file declares. Requires an image.
    tiff_series
        Zero-based TIFF series. Required when the file has several series.
        Requires an image.
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
            One row per cell in source order; observation names are the
            zero-based row positions.

            - ``X``: CSR sparse array of the cycle intensities, in source
              feature order. CSV uses one int64, uint64 or float64 dtype,
              keeping integers when every value is integral and rejecting
              lossy promotion; FCS keeps its native dtype. Negative
              compensated intensities are kept. CSV is parsed in bounded
              batches, so the dense expression of the whole table never
              exists at once.
            - ``var``: indexed by the cycle field names; ``var["marker"]``
              holds FCS ``$PnS`` labels or Akoya long labels when present.
            - ``obs``: every non-cycle field. ``cell_id`` is int64, or uint64
              when an ID needs it; ``region`` is categorical; ``x`` and ``y``
              stay in native pixels. Other CSV fields keep their exact text
              (including ``NA``, ``NULL``, leading zeros and empty strings);
              FCS fields keep their numeric dtypes.
            - ``obsm["spatial"]`` (only with a calibration): float64 ``x, y``
              in micrometres, described in ``uns["spatialdata_io_positions"]``.

        ``shapes[region]``
            Radius-one circles at the native cell positions, indexed by
            ``cell_id`` and linked to the table. The element is named after
            the region: positive integers as decimal strings, names as given if
            they are valid element names other than ``table`` and ``images``.
            The circles are display geometry, not cell boundaries.
        ``images["images"]`` (only with an image)
            Lazy CYX image of the selected series at full resolution, with the
            source dtype and channel order. Channels are named by OME channel
            names when all are nonempty and unique, otherwise by OME channel
            IDs, otherwise by position; they are not matched to table markers.
            Interleaved samples become channels: ``r, g, b`` (``r, g, b, a``
            with a declared alpha sample) when the TIFF declares them RGB
            (``PhotometricInterpretation`` RGB or JPEG-compressed YCbCr);
            otherwise an OME channel with several samples names them
            ``<name or ID>:sample:0``, ``:sample:1``, and so on. A pyramid is
            built only when ``scale_factors`` asks for one.

        Coordinate systems:

        - without calibration, the shapes use ``table_pixels`` (identity);
        - with ``pixel_size_um`` or ``experiment_file``, they use ``global``
          in micrometres (a scale of the pixel size);
        - the image uses its own ``image_pixels`` frame (identity), unless
          ``image_to_table`` registers it into the frame of the shapes,
          followed by the calibration scale if there is one.

        ``attrs`` record the reader and package version; with
        ``experiment_file``, also the Akoya version as the format version.
        Image pixels are not read until computed, and a partial computation
        decodes only the TIFF tiles or strips it overlaps. All files are
        closed on return; caller-owned objects are not modified.

    Raises
    ------
    TypeError
        If an option has the wrong type.
    ValueError
        If an option has an invalid value, an image option is given without an
        image, ``pixel_size_um`` conflicts with the experiment, or the
        calibration over- or underflows a cell position.
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        If a path is missing or has the wrong kind.
    ValueError
        A contextual ``ReaderFormatError`` naming the artifact and path if the
        table or image is ambiguous or malformed, for example a
        multi-series TIFF without ``tiff_series``, missing or duplicate fields,
        lossy numbers, several regions, invalid or duplicate cell IDs, or an
        unsupported experiment version.
    """
    validate_calibration(pixel_size_um, image_to_table)
    validate_image_options(data_axes, tiff_series)
    raster = make_raster_options(scale_factors, chunks)
    layout = discover_layout(
        path,
        cell_table_file=cell_table_file,
        image_file=image_file,
        experiment_file=experiment_file,
        path_policy=path_policy,
    )
    if layout.image is None and (
        image_to_table is not None
        or data_axes is not None
        or tiff_series is not None
        or raster != make_raster_options()
    ):
        message = "image_to_table, data_axes, tiff_series, and nondefault scale_factors or chunks require an image."
        raise ValueError(message)

    calibration = resolve_calibration(pixel_size_um, layout.experiment)
    if calibration is None:
        frame, to_frame = "table_pixels", Identity()
    else:
        size = calibration.pixel_size_um
        frame, to_frame = "global", Scale([size, size], axes=("x", "y"))

    images = {}
    if layout.image is not None:
        images["images"] = read_image(
            layout.image,
            data_axes=data_axes,
            tiff_series=tiff_series,
            raster=raster,
            frame="image_pixels" if image_to_table is None else frame,
            transformation=Identity() if image_to_table is None else TransformSequence([image_to_table, to_frame]),
        )

    cells = read_cell_table(layout.cell_table)
    table = cells.table
    shapes = make_shapes(cells.cell_ids, cells.coordinates, frame=frame, transformation=to_frame)
    if calibration is not None:
        set_positions(table, cells.coordinates, pixel_size_um=calibration.pixel_size_um, calibration=calibration.source)

    # The reader builds eager frames; AnnData also annotates backed Dataset2D.
    prepare_anndata_dataframe(
        cast("pd.DataFrame", table.obs),
        context=ReaderErrorContext("codex", "cell observations"),
        allow_float_nulls=True,
    )
    prepare_anndata_dataframe(cast("pd.DataFrame", table.var), context=ReaderErrorContext("codex", "cycle features"))
    table = parse_linked_table(
        table,
        TableLinkage((cells.region,), "region", "cell_id"),
        context=ReaderErrorContext("codex", "linked cell table"),
        path=layout.cell_table,
        expected_instance_ids={cells.region: shapes.index},
    )

    result = SpatialData(images=images, tables={"table": table}, shapes={cells.region: shapes})
    set_reader_provenance(
        result.attrs, reader="codex", format_version=None if calibration is None else calibration.version
    )
    return result
