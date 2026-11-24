"""Read one CODEX regional cell table (CSV or FCS 3.1) into a sparse AnnData.

The processor writes one table per region, as CSV or as FCS 3.1 list-mode
events, with compensated and uncompensated variants (``reg001_compensated.csv``,
``reg001_uncompensated.fcs``, ...). Both containers carry the same named fields,
described in :mod:`spatialdata_io.readers.codex._schemas`.

CSV
    Arrow reads uint64 cell IDs, float64 XY and intensities, and literal
    region/annotation strings in batches with a 1 MiB source target. Each batch
    is checked for finite intensities and converted to CSR before reading the
    next batch. Each record must fit in one source block for reliable parsing.
FCS
    Decoded eagerly by readfcs. ``$PnN`` keywords are the field names and
    available ``$PnS`` keywords the marker labels of expression channels
    (https://help.codex.bio/codex/processor/technical-notes/fcs-format).

Observation names are zero-based source row positions. The cell IDs that link
the table to shapes come from the source ``cell_id`` field. The ``x`` and ``y``
columns stay in native region pixels. With a calibration, :func:`set_positions`
adds ``obsm["spatial"]`` in micrometres. Unsupported annotation names are
sanitized with a warning, for example ``tile num`` becomes ``tile_num``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import readfcs
from scipy import sparse

from spatialdata_io._positions import POSITIONS_KEY, PositionsMetadata
from spatialdata_io.readers._utils.csv import csv_errors, csv_header
from spatialdata_io.readers._utils.errors import add_reader_context
from spatialdata_io.readers._utils.names import sanitize_column_names
from spatialdata_io.readers._utils.table import as_csr_array
from spatialdata_io.readers.codex._schemas import (
    CELL_TABLE_CONTEXT,
    REQUIRED_COLUMNS,
    STRING_DTYPE,
    cell_columns,
    csv_column_names,
    csv_types,
    expression_dtype,
    fcs_cell_ids,
    region_name,
    schema_error,
    validate_csv_annotations,
    validate_fcs_metadata,
    validate_observations,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import BinaryIO

    import numpy.typing as npt

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.codex._layout import CellTableSource

# A native batch may join the previous partial record to one source block.
# Required float64 values have at least one digit plus a separator: their
# payload is at most eight times this source target, before the dense tensor
# copy, sparse output, Arrow alignment and parser/annotation allocations.
_CSV_BLOCK_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class CellTable:
    """One validated regional cell population before final linkage.

    Attributes
    ----------
    table
        Owned AnnData with CSR expression and source-order observations.
    cell_ids
        Final uint64 cell IDs in observation order, identical to
        ``table.obs["cell_id"]``.
    coordinates
        Finite native ``x, y`` positions of shape ``(n_obs, 2)``.
    region
        Region name, which also names the shapes element.
    """

    table: ad.AnnData
    cell_ids: npt.NDArray[np.uint64]
    coordinates: npt.NDArray[np.number]
    region: str


def _csr_expression(values: npt.NDArray[np.number], path: ArtifactFile) -> sparse.csr_array:
    """Convert a typed 2D buffer and report nonfinite intensities with the source context."""
    try:
        # NaN and infinity are retained by the dense-to-CSR conversion and
        # shared construction checks their finiteness on the stored values.
        return as_csr_array(values)
    except ValueError as error:
        raise schema_error(
            path, found="nonfinite cycle intensities", expected="finite real cycle intensities"
        ) from error


def _csv_expression(batch: pa.RecordBatch, features: list[str], path: ArtifactFile) -> sparse.csr_array:
    """Convert one bounded float64 expression batch directly to canonical CSR."""
    values = np.asarray(batch.select(features).to_tensor(row_major=True))
    return _csr_expression(values, path)


def _read_csv(path: ArtifactFile) -> ad.AnnData:
    """Read typed CSV batches, retaining expression sparsely and annotations in Arrow."""
    columns, labels = csv_column_names(list(csv_header(path, context=CELL_TABLE_CONTEXT)), path)
    features, metadata = cell_columns(columns, path)
    types = csv_types(columns, features)
    blocks: list[sparse.csr_array] = []
    annotations: list[pa.RecordBatch] = []
    try:
        with (
            path.open("rb") as source,
            csv_errors(context=CELL_TABLE_CONTEXT, path=path),
            pacsv.open_csv(
                source,
                read_options=pacsv.ReadOptions(
                    column_names=columns, skip_rows=1, block_size=_CSV_BLOCK_BYTES, use_threads=False
                ),
                parse_options=pacsv.ParseOptions(newlines_in_values=True),
                convert_options=pacsv.ConvertOptions(
                    column_types=types, include_columns=columns, null_values=[], strings_can_be_null=False
                ),
            ) as reader,
        ):
            for batch in reader:
                blocks.append(_csv_expression(batch, features, path))
                annotations.append(batch.select(metadata))
    except OSError as error:
        add_reader_context(error, context=CELL_TABLE_CONTEXT, path=path)
        raise
    if not blocks:
        raise schema_error(path, found="no cells", expected="a nonempty cell population")

    expression = blocks[0] if len(blocks) == 1 else sparse.vstack(blocks, format="csr")
    fields = pa.Table.from_batches(annotations)
    validate_csv_annotations(fields, path)
    obs = fields.to_pandas(types_mapper=pd.ArrowDtype)
    obs.index = pd.RangeIndex(len(obs)).astype(str)
    var = pd.DataFrame(index=features)
    if labels is not None:
        var["marker"] = pd.Series(labels, index=columns, dtype=STRING_DTYPE).reindex(features)
    return ad.AnnData(X=expression, obs=obs, var=var)


def _read_fcs(path: ArtifactFile) -> ad.AnnData:
    """Read FCS events and distinguish parameter IDs from marker labels."""
    # ReadFCS forwards file-like inputs to FlowIO despite its path-only annotation.
    # Own the handle so even unexpected low-level parser failures close it.
    read_fcs = cast("Callable[[BinaryIO], readfcs.ReadFCS]", readfcs.ReadFCS)
    try:
        with path.open("rb") as handle:
            source = read_fcs(handle)
    except Exception as error:
        # The third-party decoder raises arbitrary exception types for malformed bytes.
        add_reader_context(error, context=CELL_TABLE_CONTEXT, path=path)
        raise

    columns = source.channels["PnN"].tolist()
    features, metadata = cell_columns(columns, path)
    frame = source.data
    if frame.shape[1] != len(columns):
        raise schema_error(path, found="inconsistent FCS parameters", expected="one PnN name per event channel")
    if frame.empty:
        raise schema_error(path, found="no cells", expected="a nonempty cell population")

    frame.columns = columns
    values = frame[features].to_numpy(dtype=expression_dtype(frame, features, path), copy=False)
    expression = _csr_expression(values, path)
    obs = frame[metadata]
    validate_fcs_metadata(obs, path)
    table = ad.AnnData(X=expression, obs=obs, var=pd.DataFrame(index=features))

    if "PnS" in source.channels:
        markers = source.channels.set_index("PnN")["PnS"]
        table.var["marker"] = pd.Series(markers.reindex(features).to_numpy(), index=table.var_names, dtype=STRING_DTYPE)

    return table


def read_cell_table(source: CellTableSource, *, expected_region: str | None = None) -> CellTable:
    """Read one nonempty cell population and normalize its IDs, region and positions."""
    path = source.path
    fcs = source.kind == "fcs"
    table = _read_fcs(path) if fcs else _read_csv(path)
    # This reader constructs eager AnnData, never backed annotation frames.
    obs = cast("pd.DataFrame", table.obs)
    if fcs:
        obs["cell_id"] = fcs_cell_ids(obs["cell_id"], path)
    coordinates = validate_observations(obs, path)
    # ShapesModel requires a native instance index; obs keeps the Arrow column.
    identifiers = obs["cell_id"].to_numpy(dtype=np.uint64, copy=False)
    region = region_name(obs["region"], path, expected_region=expected_region)
    obs["region"] = pd.Categorical([region] * table.n_obs)
    obs.columns = sanitize_column_names(
        list(obs.columns), source="codex: the cell table header", reserved=REQUIRED_COLUMNS
    )
    return CellTable(table, identifiers, coordinates, region)


def set_positions(
    table: ad.AnnData,
    coordinates: npt.NDArray[np.number],
    *,
    pixel_size_um: float,
    calibration: str,
    frame: str = "global",
) -> None:
    """Add micrometre ``obsm["spatial"]`` and its description to the owned table."""
    physical = np.multiply(coordinates, pixel_size_um, dtype=np.float64)
    # Coordinates and scale are finite, so only extreme products can overflow.
    if not np.isfinite(physical).all():
        message = "pixel_size_um produces nonfinite physical cell coordinates."
        raise ValueError(message)

    table.obsm["spatial"] = physical
    descriptions: PositionsMetadata = {
        "schema_version": 1,
        "coordinates": {
            "spatial": {
                "axes": ["x", "y"],
                "units": "micrometer",
                "position": "source cell position",
                "frame": frame,
                "calibration": {frame: f"{calibration} Resolved pixel size: {pixel_size_um} micrometres."},
            }
        },
    }
    table.uns[POSITIONS_KEY] = descriptions
