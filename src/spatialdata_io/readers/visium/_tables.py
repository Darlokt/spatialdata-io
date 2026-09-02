"""Sparse counts, position alignment, and final Visium table linkage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from spatialdata_io.readers._utils.counts import read_10x_h5, read_10x_mtx
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table
from spatialdata_io.readers.visium._layout import H5Counts

if TYPE_CHECKING:
    from pathlib import Path

    from anndata import AnnData

    from spatialdata_io.readers.visium._layout import CountsLayout, VisiumLayout
    from spatialdata_io.readers.visium._metadata import CountMetadata, ScaleFactors

POSITION_COLUMNS = (
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
)
OBS_POSITION_COLUMNS = ("in_tissue", "array_row", "array_col")
REGION_KEY = "region"
INSTANCE_KEY = "spot_id"


@dataclass(frozen=True, slots=True)
class PreparedTable:
    """Owned unlinked table and positions aligned in observation order."""

    adata: AnnData
    positions: pd.DataFrame


@dataclass(frozen=True, slots=True)
class TableOptions:
    """Validated reader state needed to construct the owned table."""

    dataset_id: str
    metadata: CountMetadata
    scale_factors: ScaleFactors
    make_var_names_unique: bool


def _context(artifact: str) -> ReaderErrorContext:
    return ReaderErrorContext(reader="visium", artifact=artifact)


def _format_error(path: Path, found: str, expected: str) -> ReaderFormatError:
    return ReaderFormatError(
        context=_context("tissue positions"),
        found=found,
        expected=expected,
        path=path,
    )


def _numeric_column(frame: pd.DataFrame, column: str, *, integer: bool, path: Path) -> np.ndarray:
    source = frame[column]
    converted = pd.to_numeric(source, errors="coerce").to_numpy(dtype=np.float64)
    if not bool(np.isfinite(converted).all()):
        raise _format_error(path, f"null, nonnumeric, or nonfinite values in {column!r}", "finite numeric values")
    if integer and not bool(np.equal(converted, np.floor(converted)).all()):
        raise _format_error(path, f"fractional values in {column!r}", "integer values")
    if integer and bool((converted < 0).any()):
        raise _format_error(path, f"negative values in {column!r}", "nonnegative integer values")
    return converted.astype(np.int64, copy=False) if integer else converted


def _validate_barcodes(frame: pd.DataFrame, *, path: Path) -> None:
    barcodes = frame["barcode"]
    if bool(barcodes.isna().any()) or bool((barcodes.str.len() == 0).any()):
        raise _format_error(path, "null or empty barcodes", "nonempty barcode strings")
    if not barcodes.is_unique:
        duplicates = tuple(barcodes[barcodes.duplicated()].drop_duplicates().head(5).tolist())
        raise _format_error(path, f"duplicate barcodes {duplicates!r}", "unique barcode strings")


def _convert_position_columns(frame: pd.DataFrame, *, path: Path) -> None:
    frame["in_tissue"] = _numeric_column(frame, "in_tissue", integer=True, path=path)
    if not set(frame["in_tissue"].unique()).issubset({0, 1}):
        raise _format_error(path, "in_tissue values outside {0, 1}", "binary in_tissue values")
    for column in ("array_row", "array_col"):
        frame[column] = _numeric_column(frame, column, integer=True, path=path)
    for column in ("pxl_row_in_fullres", "pxl_col_in_fullres"):
        values = _numeric_column(frame, column, integer=False, path=path)
        if bool((values < 0).any()):
            raise _format_error(path, f"negative values in {column!r}", "nonnegative pixel coordinates")
        frame[column] = values


def read_positions(layout: VisiumLayout) -> pd.DataFrame:
    """Read and validate the complete bounded Visium spot grid."""
    try:
        frame = (
            pd.read_csv(layout.positions, dtype="string", header=None, names=POSITION_COLUMNS)
            if layout.position_format == "headerless"
            else pd.read_csv(layout.positions, dtype="string", header=0)
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        add_reader_context(error, context=_context("tissue positions"), path=layout.positions)
        raise
    if tuple(frame.columns) != POSITION_COLUMNS:
        raise _format_error(
            layout.positions,
            f"columns {tuple(frame.columns)!r}",
            f"exact columns {POSITION_COLUMNS!r}",
        )
    if frame.empty:
        raise _format_error(layout.positions, "zero rows", "at least one spot row")
    _validate_barcodes(frame, path=layout.positions)
    _convert_position_columns(frame, path=layout.positions)
    return frame.set_index("barcode")


def _read_counts(counts: CountsLayout, *, make_var_names_unique: bool) -> AnnData:
    context = _context("count matrix")
    if isinstance(counts, H5Counts):
        adata = read_10x_h5(counts.path, context=context, gex_only=False)
        if make_var_names_unique:
            adata.var_names_make_unique()
        return adata
    return read_10x_mtx(
        counts.path,
        context=context,
        gex_only=False,
        make_unique=make_var_names_unique,
        prefix=counts.prefix,
        compressed=counts.compressed,
    )


def prepare_table(
    layout: VisiumLayout,
    positions: pd.DataFrame,
    *,
    options: TableOptions,
) -> PreparedTable:
    """Read sparse counts and align position metadata without filtering spots."""
    adata = _read_counts(layout.counts, make_var_names_unique=options.make_var_names_unique)
    if not adata.obs_names.is_unique:
        duplicates = tuple(adata.obs_names[adata.obs_names.duplicated()].unique()[:5])
        raise ReaderFormatError(
            context=_context("count matrix"),
            found=f"duplicate observation barcodes {duplicates!r}",
            expected="unique count-matrix barcodes",
            path=layout.counts.path,
        )
    missing = adata.obs_names.difference(positions.index, sort=False)
    if len(missing):
        raise ReaderFormatError(
            context=_context("table alignment"),
            found=f"count barcodes absent from tissue positions {tuple(missing[:5])!r}",
            expected="every count barcode exactly once in the position table",
            path=layout.positions,
        )
    aligned = positions.loc[adata.obs_names].copy()
    obs = cast("pd.DataFrame", adata.obs)
    for column in OBS_POSITION_COLUMNS:
        obs[column] = aligned[column].to_numpy(copy=False)
    obs[INSTANCE_KEY] = pd.array(adata.obs_names.to_numpy(copy=True), dtype="string")
    obs[REGION_KEY] = pd.Categorical(
        np.repeat(options.dataset_id, adata.n_obs),
        categories=[options.dataset_id],
    )
    adata.obsm["spatial"] = np.column_stack(
        (
            aligned["pxl_col_in_fullres"].to_numpy(dtype=np.float64, copy=False),
            aligned["pxl_row_in_fullres"].to_numpy(dtype=np.float64, copy=False),
        )
    )
    source_metadata = {
        key: value
        for key, value in (
            ("chemistry_description", options.metadata.chemistry_description),
            ("software_version", options.metadata.software_version),
        )
        if value is not None
    }
    adata.uns["spatial"] = {
        options.dataset_id: {
            "metadata": source_metadata,
            "scalefactors": dict(options.scale_factors.values),
        }
    }
    return PreparedTable(adata=adata, positions=aligned)


def link_table(prepared: PreparedTable, *, dataset_id: str, shape_index: pd.Index) -> AnnData:
    """Validate exact circle membership and parse the table exactly once."""
    return parse_linked_table(
        prepared.adata,
        TableLinkage(regions=(dataset_id,), region_key=REGION_KEY, instance_key=INSTANCE_KEY),
        context=_context("linked table"),
        expected_instance_ids={dataset_id: shape_index.copy()},
    )
