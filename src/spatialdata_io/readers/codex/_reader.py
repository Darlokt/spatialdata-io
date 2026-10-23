"""Public CODEX reader transaction and final element linkage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata import SpatialData

from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table
from spatialdata_io.readers.codex._geometry import _make_shapes
from spatialdata_io.readers.codex._images import _read_image, _validate_image_options
from spatialdata_io.readers.codex._layout import _discover_layout
from spatialdata_io.readers.codex._tables import _read_cell_table

if TYPE_CHECKING:
    from pathlib import Path


def codex(  # noqa: PLR0913, RUF100 - explicit format choices keep the public boundary typed.
    path: str | Path,
    *,
    cell_table_file: str | Path | None = None,
    image_file: str | Path | None = None,
    data_axes: str | None = None,
    tiff_series: int | None = None,
    chunks: int | tuple[int, int, int] | None = None,
) -> SpatialData:
    """Read an unversioned processor-style CODEX region export.

    Supported layouts contain one CSV or FCS 3.1 cell table and optionally one
    stitched TIFF, OME-TIFF, BigTIFF, or BTF image in the same directory::

        reg001_compensated.csv  # or reg001_compensated.fcs
        stitched.tif  # optional

    The table requires ``cell_id``, ``region``, ``x``, ``y``, and at least one
    ``cycNNN_chNNN`` channel. Multiple tables or images require file overrides.
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
    data_axes
        Explicit source TIFF axes, for example ``CYX`` for an unlabeled stack.
        File-declared axes are used otherwise. Only singleton extra dimensions
        may be removed. TIFF samples are treated as channels.
    tiff_series
        Zero-based series. Required for images containing multiple series.
        Only full-resolution level zero is read, in the table's pixel frame.
    chunks
        Positive size for all image axes, or three sizes in output CYX order.
        By default source storage chunks are retained. Requires an image, as do
        the other image options.

    Returns
    -------
    SpatialData
        ``tables['table']`` contains eager, finite canonical CSR sparse-array
        expression in source row and feature order. CSV blocks contain at most
        10,000 rows. FCS decoding materializes the event population with readfcs.
        Intensities retain real numeric dtype and negative compensated signals.
        FCS PnN supplies feature IDs. Available PnS labels become ``var['marker']``.
        All non-cycle columns remain observation metadata, and observation names
        are zero-based source positions.

        Integer cell IDs index radius-one circles in ``shapes[str(region)]``.
        Numeric FCS identifiers normalize to signed or unsigned 64-bit integers.
        Numeric regions normalize to integer strings for categorical linkage.
        ``obsm['spatial']`` retains x/y pixel coordinates. Circles and images use
        identity transforms to ``global``. No physical-unit conversion is inferred.

        Optional ``images['images']`` is lazy CYX and retains source dtype and
        channel order. Two factor-two pyramid levels are added when both spatial
        dimensions permit them. Raster channels use OME names or IDs when
        available and positional labels otherwise. They are not matched to table
        markers. Scientific three- and four-channel stacks are not guessed to be
        RGB. Raster metadata is eager, construction executes zero pixel tasks,
        and partial computations retain native TIFF storage selection.

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
    _validate_image_options(data_axes, tiff_series, chunks)
    layout = _discover_layout(path, cell_table_file=cell_table_file, image_file=image_file)
    if layout.image is None and any(option is not None for option in (data_axes, tiff_series, chunks)):
        message = "data_axes, tiff_series, and chunks require an image."
        raise ValueError(message)

    images = {}
    if layout.image is not None:
        images["images"] = _read_image(layout.image, data_axes=data_axes, tiff_series=tiff_series, chunks=chunks)

    table = _read_cell_table(layout.cell_table)
    shapes = _make_shapes(table)
    region = str(table.obs["region"].iloc[0])

    # Link only after final source IDs and the independent geometry target exist.
    table = parse_linked_table(
        table,
        TableLinkage((region,), "region", "cell_id"),
        context=ReaderErrorContext("codex", "linked cell table"),
        expected_instance_ids={region: shapes.index.copy()},
    )

    result = SpatialData(images=images, tables={"table": table}, shapes={region: shapes})
    set_reader_provenance(result.attrs, reader="codex")
    return result
