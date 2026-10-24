"""Bounded CSV and owned FCS parsing into CODEX sparse cell tables."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import readfcs
from scipy import sparse

from spatialdata_io.readers._utils.errors import ReaderErrorContext, add_reader_context
from spatialdata_io.readers._utils.table import as_csr_array
from spatialdata_io.readers.codex._schemas import (
    STRING_DTYPE,
    cell_columns,
    cell_ids,
    check_integer_promotion,
    expression_dtype,
    normalize_csv_numbers,
    normalize_fcs_metadata,
    region_name,
    schema_error,
    validate_observations,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import BinaryIO

    from spatialdata_io.readers._utils.paths import ArtifactFile

_CSV_BATCH_ROWS = 10_000
_CSV_MAX_DENSE_BYTES = 32 * 1024 * 1024
_MAX_HEADER_CHARACTERS = 1_048_576


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

    expression = as_csr_array(values)
    return expression, frame[metadata]


def _read_csv(path: ArtifactFile) -> ad.AnnData:
    """Parse owned bounded CSV blocks, then assemble sparse expression once."""
    blocks: list[sparse.csr_array] = []
    observations: list[pd.DataFrame] = []
    with path.open(encoding="utf-8-sig", newline="") as source:
        header = source.readline(_MAX_HEADER_CHARACTERS + 1)
        if len(header) > _MAX_HEADER_CHARACTERS:
            raise schema_error(path, found="oversized CSV header", expected="a header of at most 1 Mi characters")

        columns = next(csv.reader([header], strict=True), [])
        features, metadata = cell_columns(columns, path)
        # Bound numerical materialization by width as well as row count. The
        # lexical parser buffers and retained sparse/annotation outputs are separate.
        batch_rows = min(
            _CSV_BATCH_ROWS, max(1, _CSV_MAX_DENSE_BYTES // (len(features) * np.dtype(np.float64).itemsize))
        )

        # pandas' Python reader mishandles header-only input with index_col=[].
        position = source.tell()
        if not source.read(1):
            raise schema_error(path, found="no cells", expected="a nonempty cell population")
        source.seek(position)

        # The Python engine distinguishes missing fields from explicit empties.
        # The index_col=[] option prevents implicit-index repair of over-wide records.
        # String decoding protects source identifiers from float inference.
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
                if frame.empty:
                    continue

                if frame.isna().any(axis=None):
                    raise schema_error(path, found="inconsistent CSV row width", expected="one field per header column")

                normalize_csv_numbers(frame, features, path)
                expression, obs = _split_cells(frame, features, metadata, path)

                blocks.append(expression)
                observations.append(obs)

    if not blocks:
        raise schema_error(path, found="no cells", expected="a nonempty cell population")

    # A later fractional block can promote earlier integral blocks. Verify only
    # the values outside float64's exact-integer range before stacking once.
    if np.result_type(*(block.dtype for block in blocks)).kind == "f":
        for block in blocks:
            check_integer_promotion(block.data, path)

    expression = sparse.vstack(blocks, format="csr")

    # Preserve lexical metadata until concatenation so integer promotion and
    # missingness cannot depend on arbitrary CSV block boundaries.
    obs = pd.concat(observations)
    normalize_csv_numbers(obs, metadata, path)

    return ad.AnnData(X=expression, obs=obs, var=pd.DataFrame(index=features))


def _read_fcs(path: ArtifactFile) -> ad.AnnData:
    """Read owned FCS events and distinguish parameter IDs from marker labels."""
    # ReadFCS forwards file-like inputs to FlowIO despite its path-only annotation.
    # Own the handle so even unexpected low-level parser failures close it.
    read_fcs = cast("Callable[[BinaryIO], readfcs.ReadFCS]", readfcs.ReadFCS)
    with path.open("rb") as handle:
        try:
            source = read_fcs(handle)
        except Exception as error:
            add_reader_context(error, context=ReaderErrorContext("codex", "FCS events"), path=path)
            raise

    columns = source.channels["PnN"].tolist()
    features, metadata = cell_columns(columns, path)
    frame = source.data
    if frame.shape[1] != len(columns):
        raise schema_error(path, found="inconsistent FCS parameters", expected="one PnN name per event channel")

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


def _read_cell_table(path: ArtifactFile) -> ad.AnnData:
    """Validate one cell population and normalize its final IDs and pixel frame."""
    try:
        table = _read_fcs(path) if path.suffix.lower() == ".fcs" else _read_csv(path)
    except pd.errors.ParserError as error:
        raise schema_error(
            path, found=f"malformed CSV row width or quoting: {error}", expected="one field per header column"
        ) from error
    except (OSError, ValueError, TypeError, csv.Error) as error:
        add_reader_context(error, context=ReaderErrorContext("codex", "cell table"), path=path)
        raise

    if not table.n_obs:
        raise schema_error(path, found="no cells", expected="a nonempty cell population")

    # This reader constructs eager AnnData. Dataset2D is only a backed-mode alternative.
    obs = cast("pd.DataFrame", table.obs)
    identifiers = cell_ids(obs["cell_id"], path)
    obs["cell_id"] = pd.Series(
        identifiers, index=obs.index, dtype=pd.ArrowDtype(pa.from_numpy_dtype(identifiers.dtype))
    )

    coordinates = validate_observations(obs, path)

    table.obs["region"] = pd.Categorical([region_name(obs["region"], path)] * table.n_obs)
    table.obsm["spatial"] = coordinates
    return table
