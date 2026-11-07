"""Stream MCQuant tables into one finite sparse table linked to its label images.

Each ``quantification/<image>--<method>_<mask>.csv`` (schema in ``_schemas``)
measures one label image. Tables are read in label order, all as exact text
with Arrow's streaming CSV reader, and must share one set of columns:

- marker columns are decoded in bounded rows-by-markers float64 blocks (about
  8 MiB, at least one row) and each block becomes CSR at once, so expression is
  never dense as a whole. Arrow checks record widths but accepts some
  malformed quoting (a quote inside an unquoted field is literal text);
- ``CellID`` becomes exact int64 IDs, the morphology columns nullable float64
  and every other column stays Arrow text; these annotations are eager.

The rows of all tables are concatenated once, linked by ``region`` (the label
element name) and ``CellID``, and validated by one final linkage, which also
rejects duplicate IDs within a region across parser blocks. A table without
rows adds no region. ``obs`` and ``var`` (``markers.csv``) column names that a
SpatialData table does not accept are sanitized with one warning each.

Positions. MCQuant's ``X_centroid``/``Y_centroid`` are scikit-image regionprops
centroids, zero-based pixel-index coordinates in which pixel ``(0, 0)`` has its
centre at ``(0, 0)``. They are kept unchanged in ``obsm["spatial_pixels"]``,
declared in frame ``<grid>_regionprops_indices``. The raster frames place the
pixel corner at the origin, so a calibrated position is
``(native + 0.5) * spacing_um`` in the image's micrometre frame
(``obsm["spatial"]``), computed only when every measured image is calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
from scipy import sparse

from spatialdata_io._positions import POSITIONS_KEY, FixedFrame, PositionsMetadata, RowFrames
from spatialdata_io.readers._utils.csv import read_csv_header
from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.names import sanitize_column_names
from spatialdata_io.readers._utils.table import (
    TableLinkage,
    as_csr_array,
    parse_linked_table,
    prepare_anndata_dataframe,
)
from spatialdata_io.readers.mcmicro._schemas import (
    CELL_ID,
    CENTROIDS,
    MARKER_NAME,
    MARKERS_CONTEXT,
    QUANTIFICATION_CONTEXT,
    REGION_KEY,
    QuantificationColumns,
    cell_ids,
    float_block,
    float_values,
    optional_float_values,
    quantification_columns,
    schema_error,
    validate_position_columns,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.mcmicro._layout import Quantification
    from spatialdata_io.readers.mcmicro._rasters import ImageGrid
    from spatialdata_io.readers.mcmicro._schemas import MarkerPanel

_CSV_BLOCK_BYTES = 1024 * 1024
_BATCH_MEMORY_BYTES = 8 * 1024 * 1024
_NATIVE_FRAME_SUFFIX = "_regionprops_indices"
_NATIVE_POSITION = "source_regionprops_centroid_zero_based_pixel_index_centres"
_PHYSICAL_POSITION = "source_regionprops_centroid_in_raster_corner_frame"
_TABLE_CONTEXT = ReaderErrorContext("mcmicro", "cell table")
# Columns whose names the reader relies on or generates; source columns that
# equal one ignoring case are renamed rather than these.
_READER_OBS_COLUMNS = (REGION_KEY, CELL_ID, "spatial_frame", "spatial_pixel_frame")


@dataclass(frozen=True, slots=True)
class _Measurements:
    """Own one table's sparse expression blocks, exact IDs, and lexical annotations."""

    blocks: list[sparse.csr_array]
    cell_ids: pd.Series
    annotations: pa.Table


def _read_measurements(
    path: ArtifactFile, columns: QuantificationColumns, label_dtype: np.dtype[np.integer]
) -> _Measurements:
    """Decode bounded dense marker blocks to CSR, retaining annotations as exact text."""
    features = columns.features
    rows_per_batch = max(1, _BATCH_MEMORY_BYTES // (8 * len(features)))
    annotation_names = [*columns.morphology, *columns.text]
    blocks: list[sparse.csr_array] = []
    identifiers: list[pa.Array] = []
    annotations: list[pa.RecordBatch] = []

    # Arrow checks full record widths but can accept malformed CSV quoting.
    try:
        with (
            path.open("rb") as handle,
            pacsv.open_csv(
                handle,
                read_options=pacsv.ReadOptions(
                    column_names=columns.names, skip_rows=1, block_size=_CSV_BLOCK_BYTES, use_threads=False
                ),
                parse_options=pacsv.ParseOptions(newlines_in_values=True),
                convert_options=pacsv.ConvertOptions(
                    column_types=dict.fromkeys(columns.names, pa.string()), null_values=[]
                ),
            ) as reader,
        ):
            for batch in reader:
                for offset in range(0, batch.num_rows, rows_per_batch):
                    block = batch.slice(offset, rows_per_batch)
                    dense = float_block([block.column(name) for name in features], features, path)
                    blocks.append(as_csr_array(dense))
                    identifiers.append(block.column(CELL_ID))
                    annotations.append(block.select(annotation_names))
    except pa.ArrowInvalid as error:
        raise schema_error(
            QUANTIFICATION_CONTEXT,
            path,
            found=f"malformed quantification CSV: {error}",
            expected="one field per header column",
        ) from error

    schema = pa.schema([(name, pa.string()) for name in annotation_names])
    return _Measurements(
        blocks,
        cell_ids(pa.chunked_array(identifiers, pa.string()), label_dtype, path),
        pa.Table.from_batches(annotations, schema=schema),
    )


def _observations(
    measurements: list[_Measurements], regions: list[str], columns: QuantificationColumns, paths: list[ArtifactFile]
) -> tuple[pd.DataFrame, np.ndarray[tuple[int, int], np.dtype[np.float64]]]:
    """Normalize concatenated lexical annotations once, independent of batch boundaries."""
    sizes = [len(measurement.cell_ids) for measurement in measurements]
    index = pd.Index([str(position) for position in range(sum(sizes))])
    sources = list(zip(measurements, paths, strict=True))

    # Every source file is validated with its own path before its values merge.
    spatial = np.empty((len(index), 2), dtype=np.float64)
    offset = 0
    for measurement, path in sources:
        end = offset + len(measurement.cell_ids)
        for axis, name in enumerate(CENTROIDS):
            spatial[offset:end, axis] = float_values(measurement.annotations.column(name), path, name)
        offset = end

    obs = pd.DataFrame(index=index)
    obs[REGION_KEY] = pd.Categorical.from_codes(np.repeat(np.arange(len(regions)), sizes), categories=pd.Index(regions))
    obs[CELL_ID] = pd.concat([measurement.cell_ids for measurement in measurements], ignore_index=True).set_axis(index)

    morphology = set(columns.morphology)
    for name in (*columns.morphology, *columns.text):
        if name in CENTROIDS:
            continue
        if name in morphology:
            parts = [
                optional_float_values(measurement.annotations.column(name), path, name) for measurement, path in sources
            ]
        else:
            parts = [
                measurement.annotations.column(name).to_pandas(types_mapper=pd.ArrowDtype)
                for measurement in measurements
            ]
        obs[name] = pd.concat(parts, ignore_index=True).set_axis(index)
    return obs, spatial


def read_table(
    quantifications: tuple[Quantification, ...],
    panel: MarkerPanel,
    label_dtypes: Mapping[str, np.dtype[np.integer]],
    grids: Mapping[str, ImageGrid],
) -> ad.AnnData:
    """Concatenate a nonempty sequence of MCQuant tables once and link them to their labels."""
    columns: QuantificationColumns | None = None
    measurements: list[_Measurements] = []
    regions: list[str] = []
    paths: list[ArtifactFile] = []
    source_grids: list[ImageGrid] = []
    for quantification in quantifications:
        path = quantification.path
        current = quantification_columns(read_csv_header(path, context=QUANTIFICATION_CONTEXT), panel.features, path)
        if columns is None:
            columns = current
        elif set(current.names) != set(columns.names):
            raise schema_error(
                QUANTIFICATION_CONTEXT,
                path,
                found=f"columns {list(current.names)!r}",
                expected=f"the shared MCQuant schema {columns.names!r}",
            )
        measurement = _read_measurements(path, current, label_dtypes[quantification.target.key])

        # A label without measured cells is not a table region; empty linked tables fail below.
        if len(measurement.cell_ids):
            measurements.append(measurement)
            regions.append(quantification.target.key)
            paths.append(path)
            source_grids.append(grids[quantification.target.image_id])

    if columns is None or not measurements:
        raise schema_error(
            QUANTIFICATION_CONTEXT,
            quantifications[0].path,
            found="no quantified cells",
            expected="at least one MCQuant row",
        )

    validate_position_columns(columns, multiple=len({grid.name for grid in source_grids}) > 1, path=paths[0])

    obs, spatial = _observations(measurements, regions, columns, paths)
    obs.columns = sanitize_column_names(
        list(obs.columns), source="mcmicro: the quantification CSV header", reserved=_READER_OBS_COLUMNS
    )
    var = panel.var.set_axis(
        sanitize_column_names(
            list(panel.var.columns), source="mcmicro: the markers.csv header", reserved=(MARKER_NAME,)
        ),
        axis="columns",
    )
    expression = sparse.vstack([block for measurement in measurements for block in measurement.blocks], format="csr")
    table = ad.AnnData(X=expression, obs=obs)
    table.var = prepare_anndata_dataframe(var, context=MARKERS_CONTEXT)
    _set_positions(table, spatial, source_grids, [len(measurement.cell_ids) for measurement in measurements], paths)
    # This reader constructed an eager DataFrame; AnnData also admits backed frames.
    prepare_anndata_dataframe(cast("pd.DataFrame", table.obs), context=_TABLE_CONTEXT, allow_float_nulls=True)
    return parse_linked_table(table, TableLinkage(tuple(regions), REGION_KEY, CELL_ID), context=_TABLE_CONTEXT)


def _physical_positions(
    native: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    grids: list[ImageGrid],
    sizes: list[int],
    paths: list[ArtifactFile],
    spacings: list[tuple[float, float]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Shift regionprops indices to the raster's pixel-corner frame and scale each table's rows."""
    physical = np.add(native, 0.5, dtype=np.float64)
    offset = 0
    for grid, size, path, spacing in zip(grids, sizes, paths, spacings, strict=True):
        rows = physical[offset : offset + size]
        nonzero = rows != 0
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            rows *= spacing
        if not np.isfinite(rows).all():
            raise schema_error(
                QUANTIFICATION_CONTEXT,
                path,
                found=f"nonfinite physical positions in {grid.frame!r}",
                expected="finite float64 physical XY",
            )
        if np.any(nonzero & (rows == 0)):
            raise schema_error(
                QUANTIFICATION_CONTEXT,
                path,
                found="nonzero shifted pixel coordinates becoming zero physical positions",
                expected="nonzero float64 physical XY for nonzero shifted pixel coordinates",
            )
        offset += size
    return physical


def _set_positions(
    table: ad.AnnData,
    native: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    grids: list[ImageGrid],
    sizes: list[int],
    paths: list[ArtifactFile],
) -> None:
    """Declare the native centroids, adding physical positions when every measured grid is calibrated.

    ``grids``, ``sizes`` and ``paths`` describe the contiguous row slice of each
    quantification table in order.
    """
    table.obsm["spatial_pixels"] = native
    names = pd.Index(list(dict.fromkeys(grid.name for grid in grids)))
    single = len(names) == 1
    codes = np.repeat(names.get_indexer(pd.Index([grid.name for grid in grids])), sizes)
    descriptions: dict[str, FixedFrame | RowFrames] = {}

    if single:
        descriptions["spatial_pixels"] = {
            "axes": ["x", "y"],
            "units": "pixel",
            "frame": f"{names[0]}{_NATIVE_FRAME_SUFFIX}",
            "position": _NATIVE_POSITION,
        }
    else:
        native_frames = pd.Index([f"{name}{_NATIVE_FRAME_SUFFIX}" for name in names])
        table.obs["spatial_pixel_frame"] = pd.Categorical.from_codes(codes, categories=native_frames)
        descriptions["spatial_pixels"] = {
            "axes": ["x", "y"],
            "units": "pixel",
            "frame_key": "spatial_pixel_frame",
            "position": _NATIVE_POSITION,
        }

    spacings = [grid.spacing_um for grid in grids]
    if all(spacing is not None for spacing in spacings):
        complete = [spacing for spacing in spacings if spacing is not None]
        table.obsm["spatial"] = _physical_positions(native, grids, sizes, paths, complete)
        # Calibrated grids use their micrometre frame, never a pixel frame.
        evidence = {grid.frame: grid.calibration for grid in grids}
        if single:
            descriptions["spatial"] = {
                "axes": ["x", "y"],
                "units": "micrometer",
                "frame": grids[0].frame,
                "position": _PHYSICAL_POSITION,
                "calibration": evidence,
            }
        else:
            frames = pd.Index(list(dict.fromkeys(grid.frame for grid in grids)))
            table.obs["spatial_frame"] = pd.Categorical.from_codes(codes, categories=frames)
            descriptions["spatial"] = {
                "axes": ["x", "y"],
                "units": "micrometer",
                "frame_key": "spatial_frame",
                "position": _PHYSICAL_POSITION,
                "calibration": evidence,
            }

    declaration: PositionsMetadata = {"schema_version": 1, "coordinates": descriptions}
    table.uns[POSITIONS_KEY] = declaration
