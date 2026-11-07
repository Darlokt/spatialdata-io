"""Read one CODEX regional cell table (CSV or FCS 3.1) into a sparse AnnData.

The processor writes one table per region, as CSV or as FCS 3.1 list-mode
events, with compensated and uncompensated variants (``reg001_compensated.csv``,
``reg001_uncompensated.fcs``, ...). Both containers carry the same named fields,
described in :mod:`spatialdata_io.readers.codex._schemas`.

CSV
    Decoded as text in bounded batches of about 32 MiB of numerical values (at
    least one row). Each batch becomes a CSR block, so the dense
    expression of the whole population never exists at once. Defined fields
    (``region``, ``x``, ``y``) are normalized once over the complete population,
    so their dtypes do not depend on batch boundaries.
FCS
    Decoded eagerly by readfcs. ``$PnN`` keywords are the field names and
    available ``$PnS`` keywords the marker labels of expression channels
    (https://help.codex.bio/codex/processor/technical-notes/fcs-format).

Observation names are zero-based source row positions; ``x`` and ``y`` stay in
native region pixels. Metadata column names that a SpatialData table does not
accept (``tile num``) are sanitized with one warning (``tile_num``). With a calibration, :func:`set_positions` adds
``obsm["spatial"]`` in micrometres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import readfcs
from scipy import sparse

from spatialdata_io._positions import POSITIONS_KEY, PositionsMetadata
from spatialdata_io.readers._utils.csv import read_csv_header
from spatialdata_io.readers._utils.errors import add_reader_context
from spatialdata_io.readers._utils.names import sanitize_column_names
from spatialdata_io.readers._utils.table import as_csr_array
from spatialdata_io.readers.codex._schemas import (
    CELL_TABLE_CONTEXT,
    REQUIRED_COLUMNS,
    STRING_DTYPE,
    cell_columns,
    cell_ids,
    check_integer_promotion,
    csv_column_names,
    expression_dtype,
    normalize_csv_numbers,
    normalize_fcs_metadata,
    region_name,
    schema_error,
    validate_csv_fields,
    validate_observations,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import BinaryIO, TextIO

    import numpy.typing as npt

    from spatialdata_io.readers._utils.paths import ArtifactFile

_CSV_MAX_DENSE_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CellTable:
    """One validated regional cell population before final linkage.

    Attributes
    ----------
    table
        Owned AnnData with CSR expression and source-order observations.
    cell_ids
        Final integer cell IDs in observation order, identical to
        ``table.obs["cell_id"]``.
    coordinates
        Finite native ``x, y`` positions of shape ``(n_obs, 2)``.
    region
        Region name, which also names the shapes element.
    """

    table: ad.AnnData
    cell_ids: npt.NDArray[np.int64 | np.uint64]
    coordinates: npt.NDArray[np.number]
    region: str


def _split_cells(
    frame: pd.DataFrame, features: list[str], metadata: list[str], path: ArtifactFile
) -> tuple[sparse.csr_array, pd.DataFrame]:
    """Extract finite CSR intensities and metadata without changing the owned frame.

    Numeric promotion must preserve integer intensities exactly. Negative
    compensated measurements remain valid. Feature and observation order follow
    the source, and dense expression is limited to this caller-bounded block.
    """
    values = frame[features].to_numpy(dtype=expression_dtype(frame, features, path), copy=False)
    if values.dtype.kind == "f":
        for feature in features:
            check_integer_promotion(frame[feature].to_numpy(copy=False), path)

    return as_csr_array(values), frame[metadata]


def _has_record(source: TextIO) -> bool:
    """Return whether a record follows, leaving the stream position unchanged.

    pandas' Python engine skips whitespace-only lines but fails with an
    unrelated error on input without any record when ``index_col=[]``.
    """
    position = source.tell()
    while line := source.readline():
        if line.strip():
            source.seek(position)
            return True
    return False


def _read_csv(path: ArtifactFile) -> ad.AnnData:
    """Parse bounded CSV batches, then assemble sparse expression once."""
    header = read_csv_header(path, context=CELL_TABLE_CONTEXT)
    columns, labels = csv_column_names(header, path)
    features, metadata = cell_columns(columns, path)
    # Bound numerical materialization in bytes, rows times expression width. The
    # lexical parser buffers and retained sparse/annotation outputs are separate.
    batch_rows = max(1, _CSV_MAX_DENSE_BYTES // (len(features) * np.dtype(np.float64).itemsize))

    blocks: list[sparse.csr_array] = []
    observations: list[pd.DataFrame] = []
    try:
        with path.open(encoding="utf-8-sig", newline="") as source:
            # The strictly parsed header occupies the first physical line.
            source.readline()
            if not _has_record(source):
                raise schema_error(path, found="no cells", expected="a nonempty cell population")

            # The Python engine distinguishes missing fields from explicit
            # empties, and index_col=[] prevents implicit-index repair of
            # over-wide records. String decoding protects identifiers from
            # float inference.
            with pd.read_csv(
                source,
                names=columns,
                header=None,
                chunksize=batch_rows,
                engine="python",
                index_col=[],
                dtype=STRING_DTYPE,
                keep_default_na=False,
            ) as reader:
                for frame in reader:
                    validate_csv_fields(frame, path)
                    normalize_csv_numbers(frame, features, path)
                    expression, obs = _split_cells(frame, features, metadata, path)
                    blocks.append(expression)
                    observations.append(obs)
    except pd.errors.ParserError as error:
        raise schema_error(
            path, found=f"malformed CSV row width or quoting: {error}", expected="one field per header column"
        ) from error
    except UnicodeDecodeError as error:
        raise schema_error(path, found=f"invalid UTF-8 ({error.reason})", expected="UTF-8 CSV records") from error
    except OSError as error:
        add_reader_context(error, context=CELL_TABLE_CONTEXT, path=path)
        raise

    # A later fractional block can promote earlier integral blocks. Verify only
    # the values outside float64's exact-integer range before stacking once.
    if np.result_type(*(block.dtype for block in blocks)).kind == "f":
        for block in blocks:
            check_integer_promotion(block.data, path)

    expression = blocks[0] if len(blocks) == 1 else sparse.vstack(blocks, format="csr")

    # Normalize scientific fields across the population, independently of batch
    # boundaries. Unknown annotations retain their original lexical values.
    obs = observations[0] if len(observations) == 1 else pd.concat(observations)
    normalize_csv_numbers(obs, ["region", "x", "y"], path)

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
    # FCS expression already has native numerical buffers. Only annotations
    # need Arrow storage for subsequent identifier and relationship operations.
    expression, obs = _split_cells(frame, features, metadata, path)
    normalize_fcs_metadata(obs, path)
    table = ad.AnnData(X=expression, obs=obs, var=pd.DataFrame(index=features))

    if "PnS" in source.channels:
        markers = source.channels.set_index("PnN")["PnS"]
        table.var["marker"] = pd.Series(markers.reindex(features).to_numpy(), index=table.var_names, dtype=STRING_DTYPE)

    return table


def read_cell_table(path: ArtifactFile) -> CellTable:
    """Read one nonempty cell population and normalize its IDs, region and positions."""
    table = _read_fcs(path) if path.suffix.lower() == ".fcs" else _read_csv(path)

    # This reader constructs eager AnnData. Dataset2D is only a backed-mode alternative.
    obs = cast("pd.DataFrame", table.obs)
    identifiers = cell_ids(obs["cell_id"], path)
    obs["cell_id"] = pd.Series(
        identifiers, index=obs.index, dtype=pd.ArrowDtype(pa.from_numpy_dtype(identifiers.dtype))
    )
    coordinates = validate_observations(obs, path)
    region = region_name(obs["region"], path)
    obs["region"] = pd.Categorical([region] * table.n_obs)
    obs.columns = sanitize_column_names(
        list(obs.columns), source="codex: the cell table header", reserved=REQUIRED_COLUMNS
    )
    return CellTable(table, identifiers, coordinates, region)


def set_positions(
    table: ad.AnnData, coordinates: npt.NDArray[np.number], *, pixel_size_um: float, calibration: str
) -> None:
    """Add micrometre ``obsm["spatial"]`` and its description to the owned table."""
    physical = np.multiply(coordinates, pixel_size_um, dtype=np.float64)
    # Coordinates and scale are finite and the scale is positive, so only
    # extreme products can overflow or turn a nonzero position into zero.
    if not np.isfinite(physical).all():
        message = "pixel_size_um produces nonfinite physical cell coordinates."
        raise ValueError(message)
    if np.any((physical == 0) & (coordinates != 0)):
        message = "pixel_size_um underflows nonzero physical cell coordinates to zero."
        raise ValueError(message)

    table.obsm["spatial"] = physical
    descriptions: PositionsMetadata = {
        "schema_version": 1,
        "coordinates": {
            "spatial": {
                "axes": ["x", "y"],
                "units": "micrometer",
                "position": "source cell position",
                "frame": "global",
                "calibration": {"global": f"{calibration} Resolved pixel size: {pixel_size_um} micrometres."},
            }
        },
    }
    table.uns[POSITIONS_KEY] = descriptions
