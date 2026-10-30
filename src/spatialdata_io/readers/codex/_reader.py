"""Public CODEX reader transaction and final element linkage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from spatialdata import SpatialData
from spatialdata.transformations import Affine, Identity, Scale
from spatialdata.transformations import Sequence as TransformSequence

from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.image import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS, make_raster_options
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table, prepare_anndata_dataframe
from spatialdata_io.readers.codex._calibration import read_experiment_scale, validate_calibration
from spatialdata_io.readers.codex._geometry import _make_shapes
from spatialdata_io.readers.codex._images import _read_image, _validate_image_options
from spatialdata_io.readers.codex._layout import _discover_layout
from spatialdata_io.readers.codex._tables import _read_cell_table, _set_positions

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
) -> SpatialData:
    """Read an unversioned processor-style CODEX region export.

    Supported layouts contain one CSV or FCS 3.1 cell table and optionally one
    stitched TIFF, OME-TIFF, BigTIFF, or BTF image in the same directory::

        reg001_compensated.csv  # or reg001_compensated.fcs
        stitched.tif  # optional

    The table requires ``cell_id``, ``region``, ``x``, ``y``, and at least one
    ``cycNNN_chNNN`` channel. Multiple tables or images require file overrides.
    Akoya CSV may qualify every header as short-name:long-label; short names
    identify fields and expression long labels populate var['marker'].
    Named fields accept arbitrary column order and additional metadata. Only a
    complete ``cyc<digits>_ch<digits>`` name denotes expression. Empty, duplicate,
    or NUL-containing field names and NUL-containing CSV values are rejected.
    Compensated versus uncompensated sources are never inferred from names.
    No processor version is guessed from an FCS container header.

    Parameters
    ----------
    path
        Dataset directory. Discovered symlinks must stay inside this root.
    cell_table_file
        Optional CSV/FCS override. Relative paths resolve within the dataset
        root. Absolute paths may be outside it.
    image_file
        Optional TIFF-family override, with the same path policy.
    pixel_size_um
        Positive finite micrometres per native acquisition/table pixel. The
        caller establishes source association. No universal default is used.
    experiment_file
        Explicitly associated Akoya experiment JSON (versions 1.5.0.38 or
        1.7.0.6), declaring ``xyResolution`` in nm/pixel. Relative paths resolve
        against the dataset root; absolute paths may be external. Files are
        limited to 1 MiB and read once. Selection asserts association with the
        selected regional table; proximity is not used to infer association.
        Simultaneous ``pixel_size_um`` must equal the declared value / 1000
        exactly; no tolerance or calibration fitting is applied.
    image_to_table
        Optional invertible finite XY ``Affine`` from selected-image coordinates
        in SpatialData's raster convention (pixel centers at c+0.5, r+0.5)
        into the table's native XY. Requires an image. The caller establishes
        reduction, crop, orientation and origin; image dimensions cannot do so.
    data_axes
        Explicit source TIFF axes, for example ``CYX`` for an unlabeled stack.
        File-declared axes are used otherwise. Only singleton extra dimensions
        may be removed. TIFF samples are treated as channels.
    tiff_series
        Zero-based series. Required for images containing multiple series.
        Only level zero of the selected image is read; its table registration
        requires image_to_table.
    scale_factors
        Integer factors of at least two, one per generated pyramid level of the
        images; empty for a single scale.
    chunks
        One positive size for every axis or three ``c, y, x`` sizes; by default
        one channel per chunk and square 1024-pixel tiles.

        Like ``data_axes`` and ``tiff_series``, nondefault ``scale_factors`` and
        ``chunks`` require an image.

    Returns
    -------
    SpatialData
        ``tables['table']`` contains eager, finite canonical CSR sparse-array
        expression in source row and feature order. CSV blocks contain at most
        10,000 rows and at most 32 MiB of dense numerical expression, with parser
        text buffers and retained sparse/annotation outputs accounted separately.
        FCS decoding materializes the event population with readfcs; expression
        keeps native NumPy storage before sparse construction.
        Intensities retain real numeric dtype and negative compensated signals.
        FCS PnN supplies feature IDs. Available PnS labels become ``var['marker']``.
        All non-cycle columns remain observation metadata, and observation names
        are zero-based source positions.
        CSV features are normalized within each bounded block. Observation
        metadata remains lexical until all blocks are concatenated, so metadata
        dtypes and exact integer values are independent of batch boundaries.
        New numeric and string annotations use explicit Arrow storage during
        processing and one final AnnData compatibility conversion per frame.
        Returned numeric metadata uses NumPy storage, nullable integer metadata
        uses pandas nullable storage, and strings use pandas nullable strings.
        Empty numeric metadata fields become missing, represented by NaN for
        floats; missing required coordinates, IDs, or intensities are rejected.
        Literal text such as ``NA`` and ``NULL`` remains text. Fractional ID
        values, overflow, floating underflow, and lossy large-number promotion
        are rejected; integral decimal ID tokens such as ``2.0`` remain exact.

        Integer cell IDs index radius-one circles in ``shapes[str(region)]``.
        Numeric FCS identifiers normalize to signed or unsigned 64-bit integers.
        Numeric regions normalize to integer strings for categorical linkage.
        Exact native XY remain in ``obs['x']`` and ``obs['y']``. Without
        calibration there is no canonical ``obsm['spatial']``; circles use the
        native ``table_pixels`` frame. With calibration, ``spatial`` contains
        float64 XY micrometres at the source cell positions, with native origin
        and orientation, and circles transform to physical ``global``. Position
        meaning, units, frame and calibration are declared in
        ``uns['spatialdata_io_positions']``. Radius-one circles are display
        geometry in source pixels, not measured cell boundaries.
        Images use a separate ``image_pixels`` frame unless ``image_to_table``
        explicitly registers them; acquisition spacing alone cannot register
        an independently selected montage. No experiment is auto-discovered,
        no resize factor is inferred. Complete Akoya colon-qualified CSV
        headers normalize short names and retain expression long labels;
        duplicate short names fail. Original Nolan export names are unsupported.

        Optional ``images['images']`` is lazy CYX and retains source dtype and
        channel order. It is a single scale unless ``scale_factors`` requests
        pyramid levels; no pyramid is added automatically. Raster channels use
        OME names only when every name is nonempty and unique, and otherwise
        use OME IDs collectively. Channels without OME metadata use positional labels. They are not matched to table
        markers. Scientific three- and four-channel stacks are not guessed to be
        RGB. Raster metadata is eager, construction executes zero pixel tasks,
        and partial computations decode only the TIFF tiles or strips of the
        output chunks they overlap.

        All resources close before return or inside lazy tasks. Caller-owned
        state is not mutated.

    Raises
    ------
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        If a required path is missing or has the wrong filesystem kind.
    TypeError, ValueError
        If public image options, expression values, schemas, or identifiers are
        invalid, sources are ambiguous, or a region is missing or mixed.
    """
    # Resolve all choices before loading the eager cell population.
    validate_calibration(pixel_size_um, image_to_table)
    _validate_image_options(data_axes, tiff_series)
    raster = make_raster_options(scale_factors, chunks)
    layout = _discover_layout(
        path, cell_table_file=cell_table_file, image_file=image_file, experiment_file=experiment_file
    )
    if layout.image is None and (data_axes is not None or tiff_series is not None or raster != make_raster_options()):
        message = "data_axes, tiff_series, and nondefault scale_factors or chunks require an image."
        raise ValueError(message)

    if image_to_table is not None and layout.image is None:
        msg = "image_to_table requires an image."
        raise ValueError(msg)
    calibration = "Caller-declared micrometres per native acquisition/table pixel."
    if layout.experiment is not None:
        declared = read_experiment_scale(layout.experiment)
        if pixel_size_um is not None and pixel_size_um != declared:
            msg = "pixel_size_um conflicts with experiment xyResolution / 1000."
            raise ValueError(msg)
        pixel_size_um = declared
        calibration = "Caller-associated Akoya experiment xyResolution in nm/pixel, divided by 1000."
    frame = "table_pixels" if pixel_size_um is None else "global"
    transform = Identity() if pixel_size_um is None else Scale([pixel_size_um, pixel_size_um], axes=("x", "y"))
    images = {}
    if layout.image is not None:
        images["images"] = _read_image(
            layout.image,
            data_axes=data_axes,
            tiff_series=tiff_series,
            raster=raster,
            frame="image_pixels" if image_to_table is None else frame,
            transformation=Identity() if image_to_table is None else TransformSequence([image_to_table, transform]),
        )

    table, coordinates = _read_cell_table(layout.cell_table)
    shapes = _make_shapes(table, coordinates, frame=frame, transformation=transform)
    if pixel_size_um is not None:
        _set_positions(table, coordinates, pixel_size_um=pixel_size_um, calibration=calibration)
    region = str(table.obs["region"].iloc[0])

    # Only final owned frames cross storage compatibility; AnnData also annotates
    # backed Dataset2D, whereas this reader constructs eager DataFrames.
    prepare_anndata_dataframe(
        cast("pd.DataFrame", table.obs),
        context=ReaderErrorContext("codex", "cell observations"),
        allow_float_nulls=True,
    )
    prepare_anndata_dataframe(cast("pd.DataFrame", table.var), context=ReaderErrorContext("codex", "cycle features"))

    # Link only after final source IDs and the independent geometry target exist.
    table = parse_linked_table(
        table,
        TableLinkage((region,), "region", "cell_id"),
        context=ReaderErrorContext("codex", "linked cell table"),
        expected_instance_ids={region: shapes.index},
    )

    result = SpatialData(images=images, tables={"table": table}, shapes={region: shapes})
    set_reader_provenance(result.attrs, reader="codex")
    return result
