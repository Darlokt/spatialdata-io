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

import gzip
import zlib
from contextlib import contextmanager
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
from spatialdata_io.readers._utils.csv import csv_errors, csv_header
from spatialdata_io.readers._utils.errors import ReaderErrorContext, add_reader_context
from spatialdata_io.readers._utils.names import sanitize_column_names
from spatialdata_io.readers._utils.table import (
    TableLinkage,
    as_csr_array,
    finalize_count_arrays,
    normalize_owned_anndata,
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
from spatialdata_io.readers.cosmx._transforms import GLOBAL, GLOBAL_PIXELS, AxisSamples, FovPosition, fov_frame

if TYPE_CHECKING:
    import io
    from collections.abc import Generator

    import numpy.typing as npt

    from spatialdata_io.readers._utils.paths import ArtifactFile
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


@contextmanager
def _source(path: ArtifactFile, artifact: Artifact) -> Generator[io.BufferedIOBase]:
    """Own one plain or gzip byte stream, translating corrupt gzip data read inside the block.

    Gzip sources may hold one stream or concatenated members.
    """
    try:
        with gzip.GzipFile(path, "rb") if path.name.endswith(".gz") else path.open("rb") as source:
            yield source
    except (gzip.BadGzipFile, EOFError, zlib.error) as error:
        raise schema_error(
            path, artifact, found=f"invalid gzip data: {error}", expected="a readable gzip stream"
        ) from error
    except OSError as error:
        add_reader_context(error, context=ReaderErrorContext("cosmx", artifact), path=path)
        raise


def read_header(path: ArtifactFile, artifact: Artifact) -> tuple[str, ...]:
    """Parse one header with Arrow under its plain or gzip stream owner."""
    with _source(path, artifact) as source:
        return csv_header(path, context=ReaderErrorContext("cosmx", artifact), source=source)


@dataclass(frozen=True, slots=True)
class Counts:
    """Own sparse positive-cell counts, their exact keys, and the feature vocabulary."""

    expression: sparse.csr_array
    fovs: pd.Index
    cell_ids: pd.Index
    features: tuple[str, ...]


def read_counts(path: ArtifactFile) -> Counts:
    """Stream unsigned counts to CSR after validating and removing background rows."""
    types = count_types(read_header(path, COUNTS), path)
    features = tuple(types.names[2:])
    context = ReaderErrorContext("cosmx", COUNTS)
    blocks: list[sparse.csr_array] = []
    fovs: list[pa.Array] = []
    cell_ids: list[pa.Array] = []
    with (
        _source(path, COUNTS) as source,
        csv_errors(context=context, path=path),
        pacsv.open_csv(
            source,
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
            values = np.asarray(retained.select(features).to_tensor(row_major=True))
            # Each row is complete: finalize now and retain only sparse float
            # storage. Stacking promotes blocks exactly if any needs float64.
            blocks.extend(finalize_count_arrays((as_csr_array(values),), context=context, path=path))
            fovs.append(retained.column(FOV))
            cell_ids.append(retained.column(CELL_ID))
    expression = (
        sparse.vstack(blocks, format="csr") if blocks else sparse.csr_array((0, len(features)), dtype=np.float32)
    )
    all_fovs = pd.Index(pd.arrays.ArrowExtensionArray(pa.chunked_array(fovs, pa.uint64())))
    all_cells = pd.Index(pd.arrays.ArrowExtensionArray(pa.chunked_array(cell_ids, pa.uint64())))
    if pd.MultiIndex.from_arrays([all_fovs, all_cells]).has_duplicates:
        raise schema_error(path, COUNTS, found="duplicate (fov, cell_ID) count rows", expected="one count row per cell")
    return Counts(expression, all_fovs, all_cells, features)


@dataclass(frozen=True, slots=True)
class CellMetadata:
    """Own typed cell annotations, exact keys, and coordinates with their rounding bounds."""

    obs: pd.DataFrame
    fovs: pd.Index
    cell_ids: pd.Index
    local: npt.NDArray[np.float64]
    global_: npt.NDArray[np.float64]
    samples: tuple[AxisSamples, AxisSamples]


def read_metadata(path: ArtifactFile) -> CellMetadata:
    """Read typed annotations and exact keys, retaining printed coordinate uncertainty."""
    types = metadata_types(read_header(path, METADATA), path)
    with _source(path, METADATA) as stream, csv_errors(context=ReaderErrorContext("cosmx", METADATA), path=path):
        source = pacsv.read_csv(
            stream,
            parse_options=_PARSE,
            convert_options=pacsv.ConvertOptions(
                column_types=types, include_columns=types.names, null_values=["", "NA"], strings_can_be_null=False
            ),
        )
    for name in (FOV, CELL_ID):
        require_positive(source[name], path, METADATA, name)
    numeric = [field.name for field in types if pa.types.is_floating(field.type)]
    text = [field.name for field in types if pa.types.is_string(field.type) and field.name not in CENTERS]
    for name in numeric:
        require_finite(source[name], path, METADATA, name, nullable=True)
    obs = source.select([FOV, CELL_ID, *numeric, *text]).to_pandas(types_mapper=pd.ArrowDtype)
    fovs, cell_ids = pd.Index(obs[FOV].array), pd.Index(obs[CELL_ID].array)
    if pd.MultiIndex.from_arrays([fovs, cell_ids]).has_duplicates:
        raise schema_error(path, METADATA, found="duplicate (fov, cell_ID) keys", expected="one row per cell")

    local = np.empty((source.num_rows, 2), dtype=np.float64)
    global_ = np.empty_like(local)
    halves: dict[str, npt.NDArray[np.float64]] = {}
    for axis, (local_name, global_name) in enumerate((CENTERS[0::2], CENTERS[1::2])):
        local[:, axis], halves[local_name] = placement_values(source[local_name], path, METADATA, local_name)
        global_[:, axis], halves[global_name] = placement_values(source[global_name], path, METADATA, global_name)
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
    return CellMetadata(obs, fovs, cell_ids, local, global_, samples)


def read_fov_positions(path: ArtifactFile) -> dict[int, FovPosition]:
    """Read projected pixel positions; stage-only exports provide no pixel cross-check."""
    types = fov_position_types(read_header(path, POSITIONS), path)
    if types is None:
        return {}
    fov_name, x_name, y_name = types.names
    with _source(path, POSITIONS) as stream, csv_errors(context=ReaderErrorContext("cosmx", POSITIONS), path=path):
        source = pacsv.read_csv(
            stream,
            parse_options=_PARSE,
            convert_options=pacsv.ConvertOptions(
                column_types=types, include_columns=types.names, null_values=[], strings_can_be_null=False
            ),
        )
    require_positive(source[fov_name], path, POSITIONS, fov_name)
    fovs = pd.Index(pd.arrays.ArrowExtensionArray(source[fov_name]))
    if not fovs.is_unique:
        raise schema_error(path, POSITIONS, found="repeated FOV positions", expected="one position per FOV")
    x, x_half = placement_values(source[x_name], path, POSITIONS, x_name)
    y, y_half = placement_values(source[y_name], path, POSITIONS, y_name)
    return {
        int(fov): FovPosition(float(x[i]), float(y[i]), float(x_half[i]), float(y_half[i]))
        for i, fov in enumerate(fovs)
    }


def _positions(metadata: CellMetadata, pixel_size_um: float | None) -> tuple[dict[str, np.ndarray], PositionsMetadata]:
    """Return the observation coordinates and their declaration, scaling global pixels when calibrated."""
    arrays: dict[str, np.ndarray] = {"spatial_pixels": metadata.local, "spatial_global_pixels": metadata.global_}
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
    keys = pd.MultiIndex.from_arrays([counts.fovs, counts.cell_ids])
    order = keys.get_indexer(pd.MultiIndex.from_arrays([metadata.fovs, metadata.cell_ids]))
    if len(keys) != len(order) or (order < 0).any():
        raise schema_error(
            path,
            METADATA,
            found=f"{len(keys)} positive count rows for {len(order)} metadata cells",
            expected="identical positive count and metadata populations",
        )
    expression = counts.expression if np.array_equal(order, np.arange(len(order))) else counts.expression[order]
    arrays, descriptions = _positions(metadata, pixel_size_um)

    obs = metadata.obs
    obs.index = pd.Index(obs[CELL].array if CELL in obs else pd.RangeIndex(len(obs)).astype(str), name=None)
    fovs = metadata.fovs.unique().sort_values()
    regions = [f"{fov}_labels" for fov in fovs.tolist()]
    codes = fovs.get_indexer(metadata.fovs)
    obs.insert(0, REGION_KEY, pd.Series(pd.Categorical.from_codes(codes, pd.Index(regions)), index=obs.index))
    obs[FRAME_KEY] = pd.Categorical.from_codes(codes, pd.Index([fov_frame(fov) for fov in fovs.tolist()]))

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
    normalize_owned_anndata(table)
    return table
