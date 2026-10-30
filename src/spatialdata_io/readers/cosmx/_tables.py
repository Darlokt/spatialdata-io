"""Bounded CosMx counts and cell metadata aligned into one finite sparse table."""

from __future__ import annotations

import csv
import gzip
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
from scipy import sparse

from spatialdata_io._positions import POSITIONS_KEY, PositionsMetadata
from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.table import (
    TableLinkage,
    as_csr_array,
    normalize_owned_anndata,
    parse_linked_table,
    prepare_anndata_dataframe,
)
from spatialdata_io.readers.cosmx._schemas import (
    CELL,
    CELL_ID,
    CENTERS,
    FOV,
    INTEGER_DTYPE,
    REGION_KEY,
    STRING_DTYPE,
    count_block,
    count_features,
    float_values,
    fov_position_columns,
    half_units,
    integer_values,
    metadata_columns,
    optional_float_values,
    schema_error,
)
from spatialdata_io.readers.cosmx._transforms import AxisSamples, FovPosition

if TYPE_CHECKING:
    import io
    from collections.abc import Iterator

    import numpy.typing as npt

    from spatialdata_io.readers._utils.paths import ArtifactFile

_CSV_BLOCK_BYTES = 1024 * 1024
_BATCH_MEMORY_BYTES = 8 * 1024 * 1024
_MAX_BATCH_ROWS = 10_000


def open_source(path: ArtifactFile) -> io.BufferedReader | gzip.GzipFile:
    """Open one plain or gzip byte stream for the caller to own; gzip accepts concatenated members."""
    return gzip.GzipFile(path, "rb") if path.name.endswith(".gz") else path.open("rb")


def read_header(path: ArtifactFile) -> list[str]:
    """Read and parse the first CSV record before interpreting names."""
    try:
        with open_source(path) as handle:
            line = handle.readline()
    except (gzip.BadGzipFile, EOFError, zlib.error) as error:
        raise schema_error(path, found=f"invalid gzip data: {error}", expected="a readable gzip stream") from error
    try:
        return next(csv.reader([line.decode("utf-8-sig")], strict=True), [])
    except (UnicodeDecodeError, csv.Error) as error:
        raise schema_error(path, found="malformed CSV header", expected="one UTF-8 header record") from error


def _batches(path: ArtifactFile, columns: list[str]) -> Iterator[pa.RecordBatch]:
    """Stream lexical record batches; Arrow checks full record widths."""
    try:
        with (
            open_source(path) as handle,
            pacsv.open_csv(
                handle,
                read_options=pacsv.ReadOptions(
                    column_names=columns, skip_rows=1, block_size=_CSV_BLOCK_BYTES, use_threads=False
                ),
                parse_options=pacsv.ParseOptions(newlines_in_values=True),
                convert_options=pacsv.ConvertOptions(column_types=dict.fromkeys(columns, pa.string()), null_values=[]),
            ) as reader,
        ):
            yield from reader
    except pa.ArrowInvalid as error:
        raise schema_error(path, found=f"malformed CSV: {error}", expected="one field per header column") from error
    except (gzip.BadGzipFile, EOFError, zlib.error) as error:
        raise schema_error(path, found=f"invalid gzip data: {error}", expected="a readable gzip stream") from error


def _table(path: ArtifactFile, columns: list[str]) -> pa.Table:
    """Read a bounded-width eager source as exact text."""
    schema = pa.schema([(name, pa.string()) for name in columns])
    return pa.Table.from_batches(list(_batches(path, columns)), schema=schema)


@dataclass(frozen=True, slots=True)
class Counts:
    """Own sparse positive-cell counts, their exact keys, and the feature vocabulary."""

    expression: sparse.csr_array
    fovs: npt.NDArray[np.int64]
    cell_ids: npt.NDArray[np.int64]
    features: tuple[str, ...]


def read_counts(path: ArtifactFile) -> Counts:
    """Stream exact counts to CSR, removing the zero-ID background rows."""
    columns = read_header(path)
    features = count_features(columns, path)
    rows_per_batch = max(1, min(_MAX_BATCH_ROWS, _BATCH_MEMORY_BYTES // (8 * len(features))))
    blocks: list[sparse.csr_array] = []
    fovs: list[npt.NDArray[np.int64]] = []
    cell_ids: list[npt.NDArray[np.int64]] = []
    for batch in _batches(path, columns):
        for offset in range(0, batch.num_rows, rows_per_batch):
            block = batch.slice(offset, rows_per_batch)
            fov = integer_values(block.column(FOV), path, FOV, minimum=1)
            cell = integer_values(block.column(CELL_ID), path, CELL_ID, minimum=0)
            values = count_block([block.column(name) for name in features], features, path)
            # CosMx reports per-FOV background as cell 0; it is not a cell observation.
            keep = cell != 0
            blocks.append(as_csr_array(values[keep]))
            fovs.append(fov[keep])
            cell_ids.append(cell[keep])
    expression = sparse.vstack(blocks, format="csr") if blocks else sparse.csr_array((0, len(features)), dtype=np.int64)
    all_fovs = np.concatenate(fovs) if fovs else np.empty(0, np.int64)
    all_cells = np.concatenate(cell_ids) if cell_ids else np.empty(0, np.int64)
    if pd.MultiIndex.from_arrays([all_fovs, all_cells]).has_duplicates:
        raise schema_error(path, found="duplicate (fov, cell_ID) count rows", expected="one count row per cell")
    return Counts(expression, all_fovs, all_cells, features)


@dataclass(frozen=True, slots=True)
class CellMetadata:
    """Own typed cell annotations, exact keys, and coordinates with their rounding bounds."""

    obs: pd.DataFrame
    fovs: npt.NDArray[np.int64]
    cell_ids: npt.NDArray[np.int64]
    local: npt.NDArray[np.float64]
    global_: npt.NDArray[np.float64]
    samples: tuple[AxisSamples, AxisSamples]


def read_metadata(path: ArtifactFile) -> CellMetadata:
    """Read eager cell metadata with exact keys, coordinates, and typed annotations."""
    columns = metadata_columns(read_header(path), path)
    source = _table(path, list(columns.names))
    fovs = integer_values(source.column(FOV), path, FOV, minimum=1)
    cell_ids = integer_values(source.column(CELL_ID), path, CELL_ID, minimum=1)
    if pd.MultiIndex.from_arrays([fovs, cell_ids]).has_duplicates:
        raise schema_error(path, found="duplicate (fov, cell_ID) keys", expected="one row per cell")

    centers = {name: float_values(source.column(name), path, name) for name in CENTERS}
    halves = {name: half_units(source.column(name), path, name) for name in CENTERS}
    samples = tuple(
        AxisSamples(centers[local], centers[global_], halves[local], halves[global_])
        for local, global_ in (CENTERS[0::2], CENTERS[1::2])
    )

    index = pd.RangeIndex(source.num_rows)
    obs = pd.DataFrame(index=index)
    obs[FOV] = pd.Series(fovs, index=index, dtype=INTEGER_DTYPE)
    obs[CELL_ID] = pd.Series(cell_ids, index=index, dtype=INTEGER_DTYPE)
    numeric = set(columns.numeric)
    for name in (*columns.numeric, *columns.text):
        column = source.column(name)
        if name in numeric:
            obs[name] = optional_float_values(column, path, name).set_axis(index)
        else:
            obs[name] = pd.Series(column.to_pandas(types_mapper=pd.ArrowDtype), dtype=STRING_DTYPE).set_axis(index)

    if CELL in obs and (obs[CELL].isna().any() or obs[CELL].eq("").any() or not obs[CELL].is_unique):
        raise schema_error(path, found="missing or repeated vendor cell IDs", expected="unique nonempty 'cell' values")
    # AtoMx repeats the vendor 'cell' ID as 'cell_id', a case variant of 'cell_ID' that
    # SpatialData cannot store. Only an exact duplicate is dropped.
    if "cell_id" in obs and CELL in obs and obs["cell_id"].equals(obs[CELL]):
        obs = obs.drop(columns="cell_id")
    folded = pd.Index([*obs.columns, REGION_KEY]).str.lower()
    if folded.has_duplicates:
        raise schema_error(
            path,
            found=f"case-variant column names {sorted(set(folded[folded.duplicated()]))}",
            expected="annotation names that are unique ignoring case",
        )
    return CellMetadata(
        obs,
        fovs,
        cell_ids,
        np.column_stack([centers["CenterX_local_px"], centers["CenterY_local_px"]]),
        np.column_stack([centers["CenterX_global_px"], centers["CenterY_global_px"]]),
        (samples[0], samples[1]),
    )


def read_fov_positions(path: ArtifactFile) -> dict[int, FovPosition]:
    """Read pixel FOV positions; millimetre-only exports provide no pixel cross-check."""
    columns = read_header(path)
    selected = fov_position_columns(columns, path)
    if selected is None:
        return {}
    fov_name, x_name, y_name = selected
    source = _table(path, columns)
    fovs = integer_values(source.column(fov_name), path, fov_name, minimum=1)
    if len(np.unique(fovs)) != len(fovs):
        raise schema_error(path, found="repeated FOV positions", expected="one position per FOV")
    x, y = (float_values(source.column(name), path, name) for name in (x_name, y_name))
    x_half, y_half = (half_units(source.column(name), path, name) for name in (x_name, y_name))
    return {
        int(fov): FovPosition(float(x[i]), float(y[i]), float(x_half[i]), float(y_half[i]))
        for i, fov in enumerate(fovs.tolist())
    }


def build_table(
    counts: Counts, metadata: CellMetadata, *, path: ArtifactFile, linked: bool, pixel_size_um: float | None = None
) -> ad.AnnData:
    """Align counts to metadata order by exact keys and link once to per-FOV labels.

    ``path`` identifies the metadata artifact in population errors.
    """
    keys = pd.MultiIndex.from_arrays([counts.fovs, counts.cell_ids])
    order = keys.get_indexer(pd.MultiIndex.from_arrays([metadata.fovs, metadata.cell_ids]))
    if len(keys) != len(order) or (order < 0).any():
        raise schema_error(
            path,
            found=f"{len(keys)} positive count rows for {len(order)} metadata cells",
            expected="identical positive count and metadata populations",
        )
    expression = counts.expression if np.array_equal(order, np.arange(len(order))) else counts.expression[order]

    obs = metadata.obs
    if CELL in obs:
        obs.index = pd.Index(obs[CELL].to_numpy(), name=None)
    else:
        obs.index = pd.Index([str(position) for position in range(len(obs))])
    fovs = np.unique(metadata.fovs)
    regions = [f"{fov}_labels" for fov in fovs.tolist()]
    codes = np.searchsorted(fovs, metadata.fovs)
    obs.insert(0, REGION_KEY, pd.Series(pd.Categorical.from_codes(codes, pd.Index(regions)), index=obs.index))

    obs["spatial_pixel_frame"] = pd.Categorical(metadata.fovs.astype(str))
    context = ReaderErrorContext("cosmx", "cell table")
    table = ad.AnnData(
        X=expression,
        obs=prepare_anndata_dataframe(obs, context=context, allow_float_nulls=True),
        var=pd.DataFrame(index=pd.Index(counts.features)),
    )
    table.obsm["spatial_pixels"] = metadata.local
    table.obsm["global"] = metadata.global_
    descriptions: PositionsMetadata = {
        "schema_version": 1,
        "coordinates": {
            "spatial_pixels": {
                "axes": ["x", "y"],
                "units": "pixel",
                "position": "cell center",
                "frame_key": "spatial_pixel_frame",
            },
            "global": {
                "axes": ["x", "y"],
                "units": "pixel",
                "position": "cell center",
                "frame": "global_pixels",
            },
        },
    }
    if pixel_size_um is not None:
        physical = metadata.global_ * pixel_size_um
        if not np.isfinite(physical).all():
            msg = "pixel_size_um produces nonfinite physical cell coordinates."
            raise ValueError(msg)
        table.obsm["spatial"] = physical
        descriptions["coordinates"]["spatial"] = {
            "axes": ["x", "y"],
            "units": "micrometer",
            "position": "cell center",
            "frame": "global",
            "calibration": {
                "global": (
                    f"Caller-declared {pixel_size_um} micrometres per validated global pixel; no stage registration."
                )
            },
        }
    table.uns[POSITIONS_KEY] = descriptions
    if linked:
        return parse_linked_table(table, TableLinkage(tuple(regions), REGION_KEY, CELL_ID), context=context)
    normalize_owned_anndata(table)
    return table
