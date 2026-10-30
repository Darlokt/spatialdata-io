"""Stream MCQuant tables into one finite sparse table linked to its label images."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
from scipy import sparse

from spatialdata_io._positions import POSITIONS_KEY, FixedFrame, PositionsMetadata, RowFrames
from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.table import (
    TableLinkage,
    as_csr_array,
    parse_linked_table,
    prepare_anndata_dataframe,
)
from spatialdata_io.readers.mcmicro._schemas import (
    CELL_ID,
    CENTROIDS,
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
_MAX_BATCH_ROWS = 10_000


@dataclass(frozen=True, slots=True)
class _Measurements:
    """Own one table's sparse expression blocks, exact IDs, and lexical annotations."""

    blocks: list[sparse.csr_array]
    cell_ids: pd.Series
    annotations: pa.Table


def _read_header(path: ArtifactFile) -> list[str]:
    """Read the first physical UTF-8 CSV record before interpreting names."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
    try:
        return next(csv.reader([header], strict=True), [])
    except csv.Error as error:
        raise schema_error(path, found="malformed CSV header", expected="one consistently quoted header") from error


def _read_measurements(
    path: ArtifactFile, columns: QuantificationColumns, label_dtype: np.dtype[np.generic]
) -> _Measurements:
    """Decode bounded dense marker blocks to CSR, retaining annotations as exact text."""
    features = columns.features
    rows_per_batch = max(1, min(_MAX_BATCH_ROWS, _BATCH_MEMORY_BYTES // (8 * len(features))))
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
            path, found=f"malformed quantification CSV: {error}", expected="one field per header column"
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
    spatial = np.column_stack(
        [
            np.concatenate(
                [float_values(measurement.annotations.column(name), path, name) for measurement, path in sources]
            )
            for name in CENTROIDS
        ]
    )

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
    label_dtypes: Mapping[str, np.dtype[np.generic]],
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
        current = quantification_columns(_read_header(path), panel.features, path)
        if columns is None:
            columns = current
        elif set(current.names) != set(columns.names):
            raise schema_error(
                path, found=f"columns {list(current.names)!r}", expected=f"the shared MCQuant schema {columns.names!r}"
            )
        measurement = _read_measurements(path, current, label_dtypes[quantification.target.key])

        # A label without measured cells is not a table region; empty linked tables fail below.
        if len(measurement.cell_ids):
            measurements.append(measurement)
            regions.append(quantification.target.key)
            paths.append(path)
            source_grids.append(grids[quantification.target.image_id])

    if columns is None or not measurements:
        raise schema_error(quantifications[0].path, found="no quantified cells", expected="at least one MCQuant row")

    validate_position_columns(columns, multiple=len({grid.frame for grid in source_grids}) > 1, path=paths[0])
    context = ReaderErrorContext("mcmicro", "cell table")

    obs, spatial = _observations(measurements, regions, columns, paths)
    expression = sparse.vstack([block for measurement in measurements for block in measurement.blocks], format="csr")
    table = ad.AnnData(X=expression, obs=obs)
    table.var = prepare_anndata_dataframe(panel.var, context=ReaderErrorContext("mcmicro", "markers"))
    _set_positions(table, spatial, source_grids, [len(measurement.cell_ids) for measurement in measurements], paths[0])
    # This reader constructed an eager DataFrame; AnnData also admits backed frames.
    prepare_anndata_dataframe(cast("pd.DataFrame", table.obs), context=context, allow_float_nulls=True)
    return parse_linked_table(table, TableLinkage(tuple(regions), REGION_KEY, CELL_ID), context=context)


def _set_positions(
    table: ad.AnnData,
    native: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    grids: list[ImageGrid],
    sizes: list[int],
    path: ArtifactFile,
) -> None:
    """Use each quantification's contiguous row slice and authoritative source image grid."""
    table.obsm["spatial_pixels"] = native
    frames = pd.Index(list(dict.fromkeys(grid.frame for grid in grids)))
    frame_codes = (
        np.repeat(frames.get_indexer(pd.Index([grid.frame for grid in grids])), sizes) if len(frames) > 1 else None
    )
    descriptions: dict[str, FixedFrame | RowFrames] = {}
    if frame_codes is None:
        descriptions["spatial_pixels"] = {
            "axes": ["x", "y"],
            "units": "pixel",
            "frame": f"{frames[0]}_regionprops_indices",
            "position": "source_regionprops_centroid_zero_based_pixel_index_centres",
        }
    else:
        table.obs["spatial_pixel_frame"] = pd.Categorical.from_codes(
            frame_codes, categories=pd.Index([f"{frame}_regionprops_indices" for frame in frames])
        )
        descriptions["spatial_pixels"] = {
            "axes": ["x", "y"],
            "units": "pixel",
            "frame_key": "spatial_pixel_frame",
            "position": "source_regionprops_centroid_zero_based_pixel_index_centres",
        }
    if all(x is not None and y is not None for x, y in (grid.declared_sizes_um for grid in grids)):
        # MCQuant exports regionprops pixel-index centres. Match the unchanged
        # raster's half-pixel centre convention before applying its XY spacing.
        with np.errstate(over="ignore", invalid="ignore"):
            physical = np.add(native, 0.5, dtype=np.float64)
            offset = 0
            for grid, size in zip(grids, sizes, strict=True):
                physical[offset : offset + size] *= grid.declared_sizes_um
                offset += size
        if not np.isfinite(physical).all():
            raise schema_error(path, found="nonfinite physical positions", expected="finite float64 physical XY")
        table.obsm["spatial"] = physical
        evidence = {grid.frame: grid.calibration for grid in grids}
        if frame_codes is None:
            descriptions["spatial"] = {
                "axes": ["x", "y"],
                "units": "micrometer",
                "frame": frames[0],
                "position": "source_regionprops_centroid_in_raster_corner_frame",
                "calibration": evidence,
            }
        else:
            table.obs["spatial_frame"] = pd.Categorical.from_codes(frame_codes, categories=frames)
            descriptions["spatial"] = {
                "axes": ["x", "y"],
                "units": "micrometer",
                "frame_key": "spatial_frame",
                "position": "source_regionprops_centroid_in_raster_corner_frame",
                "calibration": evidence,
            }
    declaration: PositionsMetadata = {"schema_version": 1, "coordinates": descriptions}
    table.uns[POSITIONS_KEY] = declaration
