"""Bounded CSV and owned FCS parsing into CODEX sparse cell tables."""

from __future__ import annotations

import csv
import re
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd
import readfcs
from scipy import sparse

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.table import as_csr_array

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import BinaryIO

    import numpy.typing as npt

    from spatialdata_io.readers._utils.paths import ArtifactFile

_CSV_BATCH_ROWS = 10_000
_MAX_HEADER_CHARACTERS = 1_048_576
_CYCLE_COLUMN = re.compile(r"cyc\d+_ch\d+")


def _format_error(path: ArtifactFile, *, found: str, expected: str) -> ReaderFormatError:
    """Describe a source-schema violation with its CODEX artifact identity."""
    return ReaderFormatError(context=ReaderErrorContext("codex", path.name), path=path, found=found, expected=expected)


def _cell_columns(columns: list[str], path: ArtifactFile) -> tuple[list[str], list[str]]:
    """Validate source names and separate cycle features from metadata in source order."""
    if (
        not columns
        or any(not isinstance(column, str) or not column for column in columns)
        or len(set(columns)) != len(columns)
    ):
        raise _format_error(
            path, found="empty or duplicate column names", expected="nonempty unique source column names"
        )

    required = {"cell_id", "region", "x", "y"}
    if not required.issubset(columns):
        raise _format_error(path, found=f"columns {columns!r}", expected=f"required columns {sorted(required)!r}")

    features = [column for column in columns if _CYCLE_COLUMN.fullmatch(column)]
    if not features:
        raise _format_error(path, found="no cycle channels", expected="at least one cycNNN_chNNN expression column")
    feature_set = set(features)
    return features, [column for column in columns if column not in feature_set]


def _split_cells(
    frame: pd.DataFrame, features: list[str], metadata: list[str], path: ArtifactFile
) -> tuple[sparse.csr_array, pd.DataFrame]:
    """Extract finite CSR intensities and metadata without changing the owned frame.

    Numeric promotion must preserve integer intensities exactly. Negative
    compensated measurements remain valid. Feature and observation order follow
    the source, and dense expression is limited to this caller-bounded block.
    """
    values = frame[features].to_numpy(copy=False)
    if values.dtype.kind not in "iuf":
        raise _format_error(path, found="nonnumeric cycle values", expected="finite real integrated intensities")
    if values.dtype.kind == "f":
        for feature in features:
            _check_integer_promotion(frame[feature].to_numpy(copy=False), path)
    expression = as_csr_array(values)
    return expression, frame[metadata]


def _read_csv(path: ArtifactFile) -> ad.AnnData:
    """Parse owned bounded CSV blocks, then assemble sparse expression once."""
    blocks: list[sparse.csr_array] = []
    observations: list[pd.DataFrame] = []
    with path.open(encoding="utf-8-sig", newline="") as source:
        header = source.readline(_MAX_HEADER_CHARACTERS + 1)
        if len(header) > _MAX_HEADER_CHARACTERS:
            raise _format_error(path, found="oversized CSV header", expected="a header of at most 1 Mi characters")
        columns = next(csv.reader([header], strict=True), [])
        features, metadata = _cell_columns(columns, path)

        # pandas' Python reader mishandles header-only input with index_col=[].
        position = source.tell()
        if not source.read(1):
            raise _format_error(path, found="no cells", expected="a nonempty cell population")
        source.seek(position)

        # The Python engine distinguishes missing fields from explicit empties.
        # The index_col=[] option prevents implicit-index repair of over-wide records.
        # String decoding protects source identifiers from float inference.
        with pd.read_csv(
            source,
            names=columns,
            header=None,
            chunksize=_CSV_BATCH_ROWS,
            engine="python",
            index_col=[],
            dtype=str,
            keep_default_na=False,
        ) as reader:
            for frame in reader:
                if frame.empty:
                    continue
                if frame.isna().any(axis=None):
                    raise _format_error(
                        path, found="inconsistent CSV row width", expected="one field per header column"
                    )
                _convert_csv_numbers(frame, path)
                expression, obs = _split_cells(frame, features, metadata, path)
                blocks.append(expression)
                observations.append(obs)

    if not blocks:
        raise _format_error(path, found="no cells", expected="a nonempty cell population")

    # A later fractional block can promote earlier integral blocks. Verify only
    # the values outside float64's exact-integer range before stacking once.
    if np.result_type(*(block.dtype for block in blocks)).kind == "f":
        for block in blocks:
            _check_integer_promotion(block.data, path)

    expression = sparse.vstack(blocks, format="csr")
    return ad.AnnData(X=expression, obs=pd.concat(observations), var=pd.DataFrame(index=features))


def _convert_csv_numbers(frame: pd.DataFrame, path: ArtifactFile) -> None:
    """Normalize owned numeric columns while retaining lexical cell IDs."""
    for column in frame.columns:
        if column == "cell_id":
            continue
        try:
            numeric = pd.to_numeric(frame[column])
        except ValueError:
            continue
        if numeric.dtype.kind == "f":
            values = numeric.to_numpy(copy=False)
            large = np.isfinite(values) & (np.abs(values) >= 2**53)
            for raw, value in zip(frame[column][large], numeric[large], strict=True):
                if Decimal(raw) != Decimal(value):
                    raise _format_error(
                        path, found="CSV numeric precision loss", expected="exactly representable large numbers"
                    )
        frame[column] = numeric


def _check_integer_promotion(values: npt.NDArray[np.number], path: ArtifactFile) -> None:
    """Reject integers that float64 expression or geometry storage would change."""
    if values.dtype.kind in "iu":
        large = values > 2**53
        if values.dtype.kind == "i":
            large |= values < -(2**53)
        for value in values[large]:
            if int(value) != int(float(value)):
                raise _format_error(
                    path, found="integer-to-float precision loss", expected="exactly representable integer values"
                )


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
    features, metadata = _cell_columns(columns, path)
    frame = source.data
    if frame.shape[1] != len(columns):
        raise _format_error(path, found="inconsistent FCS parameters", expected="one PnN name per event channel")
    frame.columns = columns
    expression, obs = _split_cells(frame, features, metadata, path)
    table = ad.AnnData(X=expression, obs=obs, var=pd.DataFrame(index=features))

    if "PnS" in source.channels:
        markers = source.channels.set_index("PnN")["PnS"]
        table.var["marker"] = markers.reindex(features).to_numpy()
    return table


def _read_cell_table(path: ArtifactFile) -> ad.AnnData:
    """Validate one cell population and normalize its final IDs and pixel frame."""
    try:
        table = _read_fcs(path) if path.suffix.lower() == ".fcs" else _read_csv(path)
    except pd.errors.ParserError as error:
        raise _format_error(
            path, found=f"malformed CSV row width or quoting: {error}", expected="one field per header column"
        ) from error
    except (OSError, ValueError, TypeError, csv.Error) as error:
        add_reader_context(error, context=ReaderErrorContext("codex", "cell table"), path=path)
        raise

    if not table.n_obs:
        raise _format_error(path, found="no cells", expected="a nonempty cell population")

    # This reader constructs eager AnnData. Dataset2D is only a backed-mode alternative.
    obs = cast("pd.DataFrame", table.obs)
    obs["cell_id"] = _cell_ids(obs["cell_id"], path)
    if not table.obs["cell_id"].is_unique:
        raise _format_error(
            path, found="duplicate cell_id values", expected="unique cell identifiers within one region"
        )

    # Geometry stores float64 centers even when the input coordinates are ints.
    for axis in ("x", "y"):
        _check_integer_promotion(obs[axis].to_numpy(copy=False), path)
    coordinates = obs[["x", "y"]].to_numpy()
    if coordinates.dtype.kind not in "iuf" or not np.isfinite(coordinates).all():
        raise _format_error(path, found="invalid x/y coordinates", expected="finite real region pixel coordinates")

    table.obs["region"] = pd.Categorical([_region_name(obs["region"], path)] * table.n_obs)
    table.obsm["spatial"] = coordinates
    return table


def _cell_ids(values: pd.Series, path: ArtifactFile) -> npt.NDArray[np.int64 | np.uint64]:
    """Convert a nonempty population's integral IDs exactly without mutating it.

    Choose uint64 only when required by the largest identifier. Reject values
    that cannot share one signed or unsigned 64-bit representation.
    """
    # CSV identifiers remain lexical until this lossless conversion. For FCS,
    # Decimal(float) retains the exact stored binary value, unlike str(float).
    identifiers: list[int] = []
    for value in values:
        try:
            exact = Decimal(float(value)) if isinstance(value, (float, np.floating)) else Decimal(str(value))
        except InvalidOperation as error:
            raise _format_error(
                path, found=f"cell_id {value!r}", expected="finite integral cell identifiers"
            ) from error
        if not exact.is_finite() or exact != exact.to_integral_value() or exact < -(2**63) or exact >= 2**64:
            raise _format_error(path, found=f"cell_id {value!r}", expected="finite 64-bit integral cell identifiers")
        identifiers.append(int(exact))

    if min(identifiers) < 0 and max(identifiers) >= 2**63:
        raise _format_error(path, found="mixed signed/unsigned overflow", expected="one representable 64-bit ID dtype")
    return np.asarray(identifiers, dtype=np.uint64 if max(identifiers) >= 2**63 else np.int64)


def _region_name(regions: pd.Series, path: ArtifactFile) -> str:
    """Require one region and normalize positive integral values to string names."""
    if regions.isna().any() or regions.ne(regions.iloc[0]).any():
        raise _format_error(path, found="missing or multiple regions", expected="one non-null region per cell table")
    region = regions.iloc[0]
    if isinstance(region, (int, float, np.integer, np.floating)):
        if isinstance(region, bool) or not np.isfinite(region) or region <= 0 or region != int(region):
            raise _format_error(path, found=f"region {region!r}", expected="a positive integer region or nonempty name")
        return str(int(region))
    if isinstance(region, str) and region:
        return region
    raise _format_error(path, found=f"region {region!r}", expected="a positive integer region or nonempty name")
