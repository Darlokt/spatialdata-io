"""Read CosMx counts, cell metadata and FOV positions into one aligned sparse table.

Plain CSV and gzip (one stream or concatenated members) use typed Arrow
ingestion under the contracts in ``_schemas``:

- the count matrix becomes canonical CSR in bounded blocks of 1 MiB of
  source text. Each block drops the per-FOV background rows
  (``cell_ID`` 0) and is converted to float32 at once, or float64 when
  float32 cannot hold its counts exactly;
- cell metadata keeps its row order. Keys stay Arrow-backed; placement
  coordinates become NumPy values with their printed rounding bounds, and
  typed annotations become ``obs`` columns;
- FOV positions are read only when the file has pixel positions.

The table follows metadata row order. Counts are aligned to it by the exact
``(fov, cell_ID)`` keys, and both populations must be identical. Coordinates::

    obsm["spatial_pixels"]          local cell centres, pixels, frame per FOV
    obsm["spatial_global_pixels"]   global cell centres, pixels, frame global_pixels
    obsm["spatial"]                 global cell centres, micrometres, frame global
                                    (only with pixel_size_um)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from scipy import sparse

from spatialdata_io._positions import POSITIONS_KEY, PositionsMetadata
from spatialdata_io.readers._utils.csv import csv_errors, csv_header, open_csv_stream
from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.names import sanitize_column_names
from spatialdata_io.readers._utils.table import (
    TableLinkage,
    count_batch_to_csr,
    parse_linked_table,
    prepare_anndata_dataframe,
)
from spatialdata_io.readers.cosmx._schemas import (
    CELL,
    CELL_ID,
    CENTERS,
    COUNTS,
    FOV,
    FRAME_KEY,
    METADATA,
    POSITIONS,
    REGION_KEY,
    count_types,
    fov_position_types,
    metadata_types,
    placement_values,
    require_finite,
    require_positive,
    schema_error,
)
from spatialdata_io.readers.cosmx._transforms import (
    GLOBAL,
    GLOBAL_PIXELS,
    AxisSamples,
    FovPosition,
    fov_frame,
    group_fov_rows,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.cosmx._layout import CsvFile
    from spatialdata_io.readers.cosmx._schemas import Artifact

# A parser batch can combine a preceding unfinished record with one source
# block. Required uint64 values use at most eight times this source target,
# excluding parser buffers, tensor copies and CSR storage. Records should fit
# one block; a record spanning two block boundaries fails.
_BATCH_BYTES = 1024 * 1024
_PARSE = pacsv.ParseOptions(newlines_in_values=True)
_TABLE_CONTEXT = ReaderErrorContext("cosmx", "cell table")
# Columns of obs that the reader generates or relies on by name; the vendor
# 'cell' IDs too, when present, because they name the observations.
_READER_OBS_COLUMNS = (FOV, CELL_ID, REGION_KEY, FRAME_KEY)


def read_header(source: CsvFile, artifact: Artifact) -> tuple[str, ...]:
    """Parse one header with Arrow under its plain or gzip stream owner."""
    context = ReaderErrorContext("cosmx", artifact)
    with open_csv_stream(source.path, context=context, compressed=source.compressed) as stream:
        return csv_header(source.path, context=context, source=stream)


@dataclass(frozen=True, slots=True)
class Counts:
    """Own sparse positive-cell counts, their exact keys, and the feature vocabulary."""

    expression: sparse.csr_array
    keys: pd.MultiIndex
    features: tuple[str, ...]


def read_counts(source: CsvFile) -> Counts:
    """Stream unsigned counts to CSR after validating and removing background rows."""
    path = source.path
    types = count_types(read_header(source, COUNTS), path)
    features = tuple(types.names[2:])
    context = ReaderErrorContext("cosmx", COUNTS)
    blocks: list[sparse.csr_array] = []
    fovs: list[pa.Array] = []
    cell_ids: list[pa.Array] = []
    with (
        open_csv_stream(path, context=context, compressed=source.compressed) as stream,
        csv_errors(context=context, path=path),
        pacsv.open_csv(
            stream,
            read_options=pacsv.ReadOptions(block_size=_BATCH_BYTES, use_threads=False),
            parse_options=_PARSE,
            convert_options=pacsv.ConvertOptions(
                column_types=types, include_columns=types.names, null_values=[], strings_can_be_null=False
            ),
        ) as reader,
    ):
        for batch in reader:
            require_positive(batch.column(FOV), path, COUNTS, FOV)
            # Typed ingestion validates every background count before filtering.
            keep = pc.call_function("not_equal", [batch.column(CELL_ID), pa.scalar(0, pa.uint64())])
            retained = batch if pc.call_function("all", [keep]).as_py() else batch.filter(keep)
            if not retained.num_rows:
                continue
            # Each row is complete: finalize now and retain only sparse float
            # storage. Stacking promotes blocks exactly if any needs float64.
            blocks.append(count_batch_to_csr(retained.select(features), context=context, path=path))
            fovs.append(retained.column(FOV))
            cell_ids.append(retained.column(CELL_ID))
    expression = (
        sparse.vstack(blocks, format="csr") if blocks else sparse.csr_array((0, len(features)), dtype=np.float32)
    )
    all_fovs = pd.Index(pd.arrays.ArrowExtensionArray(pa.chunked_array(fovs, pa.uint64())))
    all_cells = pd.Index(pd.arrays.ArrowExtensionArray(pa.chunked_array(cell_ids, pa.uint64())))
    keys = pd.MultiIndex.from_arrays([all_fovs, all_cells])
    if keys.has_duplicates:
        raise schema_error(path, COUNTS, found="duplicate (fov, cell_ID) count rows", expected="one count row per cell")
    return Counts(expression, keys, features)


@dataclass(frozen=True, slots=True)
class CellMetadata:
    """Own typed cell annotations, exact keys, and coordinates with their rounding bounds."""

    obs: pd.DataFrame
    keys: pd.MultiIndex
    fov_rows: dict[int, npt.NDArray[np.intp]]
    local: npt.NDArray[np.float64]
    global_: npt.NDArray[np.float64]
    samples: tuple[AxisSamples, AxisSamples]


def read_metadata(source: CsvFile) -> CellMetadata:
    """Read typed annotations and exact keys, retaining printed coordinate uncertainty."""
    path = source.path
    types = metadata_types(read_header(source, METADATA), path)
    context = ReaderErrorContext("cosmx", METADATA)
    with (
        open_csv_stream(path, context=context, compressed=source.compressed) as stream,
        csv_errors(context=context, path=path),
    ):
        records = pacsv.read_csv(
            stream,
            parse_options=_PARSE,
            convert_options=pacsv.ConvertOptions(
                column_types=types, include_columns=types.names, null_values=["", "NA"], strings_can_be_null=False
            ),
        )
    for name in (FOV, CELL_ID):
        require_positive(records[name], path, METADATA, name)
    numeric = [field.name for field in types if pa.types.is_floating(field.type)]
    text = [field.name for field in types if pa.types.is_string(field.type) and field.name not in CENTERS]
    for name in numeric:
        require_finite(records[name], path, METADATA, name, nullable=True)
    obs = records.select([FOV, CELL_ID, *numeric, *text]).to_pandas(types_mapper=pd.ArrowDtype)
    fovs, cell_ids = pd.Index(obs[FOV].array), pd.Index(obs[CELL_ID].array)
    keys = pd.MultiIndex.from_arrays([fovs, cell_ids])
    if keys.has_duplicates:
        raise schema_error(path, METADATA, found="duplicate (fov, cell_ID) keys", expected="one row per cell")

    local = np.empty((records.num_rows, 2), dtype=np.float64)
    global_ = np.empty_like(local)
    halves: dict[str, npt.NDArray[np.float64]] = {}
    for axis, (local_name, global_name) in enumerate((CENTERS[0::2], CENTERS[1::2])):
        local[:, axis], halves[local_name] = placement_values(records[local_name], path, METADATA, local_name)
        global_[:, axis], halves[global_name] = placement_values(records[global_name], path, METADATA, global_name)
    samples = (
        AxisSamples(local[:, 0], global_[:, 0], halves[CENTERS[0]], halves[CENTERS[2]]),
        AxisSamples(local[:, 1], global_[:, 1], halves[CENTERS[1]], halves[CENTERS[3]]),
    )
    if CELL in obs and (obs[CELL].eq("").any() or not obs[CELL].is_unique):
        raise schema_error(
            path, METADATA, found="missing or repeated vendor cell IDs", expected="unique nonempty 'cell' values"
        )
    # AtoMx's exact repeated text identity is redundant; other case variants
    # retain their independent source values through column-name sanitization.
    if "cell_id" in obs and CELL in obs and obs["cell_id"].equals(obs[CELL]):
        obs = obs.drop(columns="cell_id")
    reserved = (*_READER_OBS_COLUMNS, CELL) if CELL in obs else _READER_OBS_COLUMNS
    obs.columns = sanitize_column_names(
        list(obs.columns), source="cosmx: the cell metadata CSV header", reserved=reserved
    )
    return CellMetadata(obs, keys, group_fov_rows(fovs), local, global_, samples)


def read_fov_positions(source: CsvFile) -> dict[int, FovPosition]:
    """Read projected pixel positions; stage-only exports provide no pixel cross-check."""
    path = source.path
    types = fov_position_types(read_header(source, POSITIONS), path)
    if types is None:
        return {}
    fov_name, x_name, y_name = types.names
    context = ReaderErrorContext("cosmx", POSITIONS)
    with (
        open_csv_stream(path, context=context, compressed=source.compressed) as stream,
        csv_errors(context=context, path=path),
    ):
        records = pacsv.read_csv(
            stream,
            parse_options=_PARSE,
            convert_options=pacsv.ConvertOptions(
                column_types=types, include_columns=types.names, null_values=[], strings_can_be_null=False
            ),
        )
    require_positive(records[fov_name], path, POSITIONS, fov_name)
    fovs = pd.Index(pd.arrays.ArrowExtensionArray(records[fov_name]))
    if not fovs.is_unique:
        raise schema_error(path, POSITIONS, found="repeated FOV positions", expected="one position per FOV")
    x, x_half = placement_values(records[x_name], path, POSITIONS, x_name)
    y, y_half = placement_values(records[y_name], path, POSITIONS, y_name)
    return {
        int(fov): FovPosition(float(x[i]), float(y[i]), float(x_half[i]), float(y_half[i]))
        for i, fov in enumerate(fovs)
    }


def _positions(
    metadata: CellMetadata, pixel_size_um: float | None
) -> tuple[dict[str, npt.NDArray[np.float64]], PositionsMetadata]:
    """Return the observation coordinates and their declaration, scaling global pixels when calibrated."""
    arrays = {"spatial_pixels": metadata.local, "spatial_global_pixels": metadata.global_}
    descriptions: PositionsMetadata = {
        "schema_version": 1,
        "coordinates": {
            "spatial_pixels": {"axes": ["x", "y"], "units": "pixel", "position": "cell center", "frame_key": FRAME_KEY},
            "spatial_global_pixels": {
                "axes": ["x", "y"],
                "units": "pixel",
                "position": "cell center",
                "frame": GLOBAL_PIXELS,
            },
        },
    }
    if pixel_size_um is None:
        return arrays, descriptions

    with np.errstate(over="ignore"):
        physical = metadata.global_ * pixel_size_um
    if not np.isfinite(physical).all():
        msg = "pixel_size_um produces nonfinite physical cell coordinates."
        raise ValueError(msg)
    arrays["spatial"] = physical
    descriptions["coordinates"]["spatial"] = {
        "axes": ["x", "y"],
        "units": "micrometer",
        "position": "cell center",
        "frame": GLOBAL,
        "calibration": {
            GLOBAL: f"Caller-declared {pixel_size_um} micrometres per validated global pixel; no stage registration."
        },
    }
    return arrays, descriptions


def build_table(
    counts: Counts, metadata: CellMetadata, *, path: ArtifactFile, linked: bool, pixel_size_um: float | None = None
) -> ad.AnnData:
    """Align counts to metadata order by exact keys and link once to per-FOV labels.

    Finalize the reader-owned ``metadata.obs`` in place and reuse its native
    coordinate buffers. ``path`` identifies the metadata artifact in population errors.
    """
    order = counts.keys.get_indexer(metadata.keys)
    if len(counts.keys) != len(order) or (order < 0).any():
        raise schema_error(
            path,
            METADATA,
            found=f"{len(counts.keys)} positive count rows for {len(order)} metadata cells",
            expected="identical positive count and metadata populations",
        )
    expression = counts.expression if np.array_equal(order, np.arange(len(order))) else counts.expression[order]
    arrays, descriptions = _positions(metadata, pixel_size_um)

    obs = metadata.obs
    obs.index = pd.Index(obs[CELL].array if CELL in obs else pd.RangeIndex(len(obs)).astype(str), name=None)
    regions = [f"{fov}_labels" for fov in metadata.fov_rows]
    codes = np.empty(len(obs), dtype=np.intp)
    for code, rows in enumerate(metadata.fov_rows.values()):
        codes[rows] = code
    obs.insert(0, REGION_KEY, pd.Series(pd.Categorical.from_codes(codes, pd.Index(regions)), index=obs.index))
    obs[FRAME_KEY] = pd.Categorical.from_codes(codes, pd.Index([fov_frame(fov) for fov in metadata.fov_rows]))

    table = ad.AnnData(
        X=expression,
        obs=prepare_anndata_dataframe(obs, context=_TABLE_CONTEXT, allow_float_nulls=True),
        var=pd.DataFrame(index=pd.Index(counts.features)),
    )
    for key, array in arrays.items():
        table.obsm[key] = array
    table.uns[POSITIONS_KEY] = descriptions
    if linked:
        return parse_linked_table(table, TableLinkage(tuple(regions), REGION_KEY, CELL_ID), context=_TABLE_CONTEXT)
    return table
