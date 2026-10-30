"""Bounded position ingestion, grid alignment and registration validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers.visium_hd._schemas import (
    POSITIONS_SCHEMA,
    parse_bin_barcodes,
    validate_position_batch,
    validate_schema,
)
from spatialdata_io.readers.visium_hd._transforms import SPOT_PITCH_UM, SlideFrame, apply_projective, fit_homography

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.paths import ArtifactFile

MAX_PROJECTIVE_ERROR_PX = 1e-9
POSITION_BATCH_SIZE = 131_072


def read_grid_shape(positions: ArtifactFile) -> tuple[int, int]:
    """Infer grid bounds from exact Parquet maxima, with a projected fallback.

    Footer statistics avoid a pixel-position scan when the optional feature
    slice is absent. The consuming iterator independently proves complete,
    nonnull, unique coverage inside these bounds.
    """
    try:
        source = pq.ParquetFile(positions)
    except pa.ArrowInvalid as error:
        raise ReaderFormatError(
            context=ReaderErrorContext("visium_hd", "tissue positions"),
            path=positions,
            found="malformed Parquet footer",
            expected="a readable positions Parquet artifact",
        ) from error

    with source:
        validate_schema(source.schema_arrow, POSITIONS_SCHEMA, positions, "tissue positions")
        if source.metadata.num_rows == 0:
            raise ReaderFormatError(
                context=ReaderErrorContext("visium_hd", "tissue positions"),
                path=positions,
                found="zero position rows",
                expected="a nonempty rectangular grid",
            )

        indices = [
            next(i for i in range(source.metadata.num_columns) if source.schema.column(i).path == name)
            for name in ("array_row", "array_col")
        ]
        maxima = [-1, -1]
        have_statistics = True
        for i in range(source.metadata.num_row_groups):
            group = source.metadata.row_group(i)
            for axis, column in enumerate(indices):
                statistics = group.column(column).statistics
                if statistics is None or not statistics.has_min_max:
                    have_statistics = False
                    break
                maxima[axis] = max(maxima[axis], int(statistics.max))

        if not have_statistics:
            maxima = [-1, -1]
            for batch in source.iter_batches(batch_size=POSITION_BATCH_SIZE, columns=["array_row", "array_col"]):
                for axis, name in enumerate(("array_row", "array_col")):
                    column = batch.column(name)
                    if column.null_count:
                        raise ReaderFormatError(
                            context=ReaderErrorContext("visium_hd", "tissue positions"),
                            path=positions,
                            found=f"null {name}",
                            expected="nonnull grid indices",
                        )
                    maxima[axis] = max(maxima[axis], int(column.to_numpy().max(initial=0)))

        return maxima[0] + 1, maxima[1] + 1


def iter_position_batches(positions: ArtifactFile, shape: tuple[int, int]) -> Iterator[pa.RecordBatch]:
    """Yield projected source batches, validating complete unique grid coverage.

    One boolean per grid site records coverage; decoded source columns have at
    most ``POSITION_BATCH_SIZE`` rows. No complete positions frame is built.
    """
    context = ReaderErrorContext("visium_hd", "tissue positions")
    try:
        source = pq.ParquetFile(positions)
    except pa.ArrowInvalid as error:
        raise ReaderFormatError(
            context=ReaderErrorContext("visium_hd", "tissue positions"),
            path=positions,
            found="malformed Parquet footer",
            expected="a readable positions Parquet artifact",
        ) from error

    with source:
        validate_schema(source.schema_arrow, POSITIONS_SCHEMA, positions, "tissue positions")
        if source.metadata.num_rows != shape[0] * shape[1]:
            raise ReaderFormatError(
                context=context,
                path=positions,
                found=f"{source.metadata.num_rows} rows",
                expected=f"complete grid {shape}",
            )

        seen = np.zeros(shape[0] * shape[1], dtype=np.bool_)
        for batch in source.iter_batches(batch_size=POSITION_BATCH_SIZE, columns=POSITIONS_SCHEMA.names):
            validate_position_batch(batch, positions)
            rows = batch.column("array_row").to_numpy()
            cols = batch.column("array_col").to_numpy()
            if np.any(rows >= shape[0]) or np.any(cols >= shape[1]):
                raise ReaderFormatError(
                    context=context, path=positions, found="out-of-grid indices", expected=f"indices within {shape}"
                )

            indices = rows.astype(np.int64) * shape[1] + cols
            if np.any(seen[indices]) or np.unique(indices).size != indices.size:
                raise ReaderFormatError(
                    context=context, path=positions, found="duplicate grid indices", expected="one row per grid site"
                )
            seen[indices] = True
            yield batch


def read_positions(
    positions: ArtifactFile,
    shape: tuple[int, int],
    names: pd.Index[str],
    size_um: int,
    frame: SlideFrame,
) -> pd.DataFrame:
    """Validate registration on every source bin and align selected matrix rows.

    Space Ranger offsets the transformed lower grid index by half the image
    displacement from spot (0, 0) to (factor - 1, factor - 1). This is neither
    the projected nominal centre under a homography nor a mean of captured
    spots; partial edge bins use the same full-bin offset.
    """
    rows, cols = parse_bin_barcodes(names, size_um, shape, positions)
    indices = rows.astype(np.int64) * shape[1] + cols
    # Search spatially sorted IDs, then scatter values back into matrix observation order.
    order = np.argsort(indices)
    sorted_indices = indices[order]

    tissue = np.empty(len(names), dtype=np.uint8)
    x = np.empty(len(names), dtype=np.float64)
    y = np.empty(len(names), dtype=np.float64)

    factor = size_um // 2
    corners = apply_projective(frame.spot_to_image, np.array([[0, 0], [factor - 1, factor - 1]], dtype=np.float64))
    offset = (corners[1] - corners[0]) / 2
    for batch in iter_position_batches(positions, shape):
        batch_rows = batch.column("array_row").to_numpy()
        batch_cols = batch.column("array_col").to_numpy()
        xy = np.column_stack((batch_cols, batch_rows)).astype(np.float64)
        pixels = np.column_stack(
            (batch.column("pxl_col_in_fullres").to_numpy(), batch.column("pxl_row_in_fullres").to_numpy())
        )
        expected = apply_projective(frame.spot_to_image, factor * xy) + offset
        error_px = np.max(np.linalg.norm(expected - pixels, axis=1), initial=0.0)
        if error_px > MAX_PROJECTIVE_ERROR_PX:
            raise ReaderFormatError(
                context=ReaderErrorContext("visium_hd", f"{size_um} micrometre positions"),
                path=positions,
                found=f"cross-bin registration residual {error_px:.6g} pixels",
                expected=f"the 2 micrometre registration with residual <= {MAX_PROJECTIVE_ERROR_PX:g} pixels",
            )

        grid = batch_rows.astype(np.int64) * shape[1] + batch_cols
        matches = np.searchsorted(sorted_indices, grid)
        valid = matches < len(indices)
        if not valid.any():
            continue

        source_rows = np.flatnonzero(valid)
        source_rows = source_rows[sorted_indices[matches[source_rows]] == grid[source_rows]]
        destination = order[matches[source_rows]]

        source_names = batch.column("barcode").take(pa.array(source_rows)).to_pylist()
        if source_names != names[destination].tolist():
            raise ReaderFormatError(
                context=ReaderErrorContext("visium_hd", "tissue positions"),
                path=positions,
                found="barcode differs at selected grid index",
                expected="exact matrix/positions barcode agreement",
            )

        tissue[destination] = batch.column("in_tissue").to_numpy()[source_rows]
        x[destination] = batch.column("pxl_col_in_fullres").to_numpy()[source_rows]
        y[destination] = batch.column("pxl_row_in_fullres").to_numpy()[source_rows]

    result = pa.table(
        {"in_tissue": tissue, "array_row": rows, "array_col": cols, "pxl_row_in_fullres": y, "pxl_col_in_fullres": x}
    ).to_pandas(types_mapper=pd.ArrowDtype)
    result.index = names
    return result


def fit_slide_frame(positions: ArtifactFile, shape: tuple[int, int]) -> SlideFrame:
    """Fit the physical image frame in two bounded scans of all 2 µm positions.

    A centered covariance reduction fits the affine using every source bin.
    Nine well-spread correspondences fit the exact homography; the second scan
    validates it on every bin and measures the affine's registration residual.
    No complete position array or string index is retained.
    """
    sample_rows = np.unique([0, shape[0] // 2, shape[0] - 1])
    sample_cols = np.unique([0, shape[1] // 2, shape[1] - 1])
    sample_source: list[NDArray[np.float64]] = []
    sample_target: list[NDArray[np.float64]] = []

    mean = np.zeros(4, dtype=np.float64)
    covariance = np.zeros((4, 4), dtype=np.float64)
    count = 0

    for batch in iter_position_batches(positions, shape):
        row = batch.column("array_row").to_numpy()
        col = batch.column("array_col").to_numpy()
        pixels = np.column_stack(
            (batch.column("pxl_col_in_fullres").to_numpy(), batch.column("pxl_row_in_fullres").to_numpy())
        )
        grid = np.column_stack((col, row)).astype(np.float64)

        selected = np.isin(row, sample_rows) & np.isin(col, sample_cols)
        sample_source.append(grid[selected])
        sample_target.append(pixels[selected])

        values = np.column_stack((pixels, SPOT_PITCH_UM * (grid + 0.5)))
        batch_count = len(values)
        batch_mean = values.mean(axis=0)
        centered = values - batch_mean
        delta = batch_mean - mean
        total = count + batch_count
        # Merge centered moments to avoid subtracting large, nearly equal raw sums.
        covariance += centered.T @ centered + np.outer(delta, delta) * (count * batch_count / total)
        mean += delta * (batch_count / total)
        count = total

    homography = fit_homography(np.concatenate(sample_source), np.concatenate(sample_target))
    linear = np.linalg.solve(covariance[:2, :2], covariance[:2, 2:]).T
    origin = mean[2:] - linear @ mean[:2]
    rigid, affine = _orient_frame(linear, origin)

    maximum = 0.0
    squared_sum = 0.0

    for batch in iter_position_batches(positions, shape):
        grid = np.column_stack((batch.column("array_col").to_numpy(), batch.column("array_row").to_numpy())).astype(
            np.float64
        )
        pixels = np.column_stack(
            (batch.column("pxl_col_in_fullres").to_numpy(), batch.column("pxl_row_in_fullres").to_numpy())
        )
        error_px = np.max(np.linalg.norm(apply_projective(homography, grid) - pixels, axis=1), initial=0.0)
        if error_px > MAX_PROJECTIVE_ERROR_PX:
            raise ReaderFormatError(
                context=ReaderErrorContext("visium_hd", "2 micrometre positions"),
                path=positions,
                found=f"projective registration residual {error_px:.6g} pixels",
                expected=f"one exact homography with residual <= {MAX_PROJECTIVE_ERROR_PX:g} pixels on every bin",
            )

        residual = pixels @ linear.T + origin - SPOT_PITCH_UM * (grid + 0.5)
        squared = np.einsum("ij,ij->i", residual, residual)
        maximum = max(maximum, float(np.sqrt(np.max(squared, initial=0.0))))
        squared_sum += float(squared.sum())

    for matrix in (homography, rigid, affine):
        matrix.flags.writeable = False
    return SlideFrame(homography, rigid, affine, maximum, float(np.sqrt(squared_sum / count)))


def _orient_frame(
    linear: NDArray[np.float64], origin: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Remove fitted rotation/reflection while keeping exact physical distances."""
    left, _, right = np.linalg.svd(linear)
    # Keep reflections: changing the determinant would mirror the tissue relative to its image.
    orthogonal = left @ right

    rigid = np.eye(3)
    rigid[:2, :2] = orthogonal.T
    rigid[:2, 2] = -orthogonal.T @ origin

    affine = np.eye(3)
    affine[:2, :2] = orthogonal.T @ linear
    return rigid, affine
