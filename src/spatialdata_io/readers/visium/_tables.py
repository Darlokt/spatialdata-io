"""Sparse counts, position alignment, and final Visium table linkage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from spatialdata_io._positions import POSITIONS_KEY, PositionsMetadata
from spatialdata_io.readers._utils.counts import read_10x_h5, read_10x_mtx
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table, prepare_anndata_dataframe
from spatialdata_io.readers.visium._layout import H5Counts
from spatialdata_io.readers.visium._schemas import (
    INSTANCE_KEY,
    OBS_POSITION_COLUMNS,
    POSITION_COLUMNS,
    POSITION_SOURCE_SCHEMA,
    REGION_KEY,
    SPOT_ID_DTYPE,
    normalize_positions,
    position_error,
)

if TYPE_CHECKING:
    from anndata import AnnData

    from spatialdata_io.readers.visium._layout import CountsLayout, VisiumLayout
    from spatialdata_io.readers.visium._metadata import CountMetadata, ScaleFactors


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
    """Identify a Visium table artifact for contextual errors."""
    return ReaderErrorContext(reader="visium", artifact=artifact)


def read_positions(layout: VisiumLayout) -> pd.DataFrame:
    """Read projected position strings with strict CSV record widths."""
    try:
        with layout.positions.open("rb") as source:
            table = pacsv.read_csv(
                source,
                read_options=pacsv.ReadOptions(
                    column_names=list(POSITION_COLUMNS) if layout.position_format == "headerless" else None,
                ),
                convert_options=pacsv.ConvertOptions(
                    column_types=POSITION_SOURCE_SCHEMA,
                    include_columns=list(POSITION_COLUMNS),
                    strings_can_be_null=False,
                ),
            )
    except (pa.ArrowInvalid, pa.ArrowKeyError) as error:
        raise position_error(
            layout.positions, str(error), "required columns and consistent CSV record widths"
        ) from error
    except OSError as error:
        add_reader_context(error, context=_context("tissue positions"), path=layout.positions)
        raise

    return normalize_positions(table.to_pandas(types_mapper=pd.ArrowDtype), path=layout.positions)


def _read_counts(counts: CountsLayout, *, make_var_names_unique: bool) -> AnnData:
    """Read one validated count source and apply feature-name policy."""
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

    # Count barcodes determine row order and membership, including non-tissue spots.
    aligned = positions.loc[adata.obs_names]
    obs = cast("pd.DataFrame", adata.obs)
    for column in OBS_POSITION_COLUMNS:
        obs[column] = aligned[column].array

    obs[INSTANCE_KEY] = pd.Series(adata.obs_names, dtype=SPOT_ID_DTYPE).array
    obs[REGION_KEY] = pd.Categorical(
        np.repeat(options.dataset_id, adata.n_obs),
        categories=[options.dataset_id],
    )

    # Vendor CSV coordinates are row/column, while SpatialData uses x/y.
    adata.obsm["spatial_pixels"] = np.column_stack(
        (
            aligned["pxl_col_in_fullres"].to_numpy(dtype=np.float64, copy=False),
            aligned["pxl_row_in_fullres"].to_numpy(dtype=np.float64, copy=False),
        )
    )

    description: PositionsMetadata = {
        "schema_version": 1,
        "coordinates": {
            "spatial_pixels": {
                "axes": ["x", "y"],
                "units": "pixel",
                "frame": options.dataset_id,
                "position": "full-resolution spot centre",
            }
        },
    }
    adata.uns[POSITIONS_KEY] = description

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
    # The shared count readers return in-memory frames; AnnData's annotations
    # additionally cover Dataset2D, used only for backed tables.
    prepare_anndata_dataframe(cast("pd.DataFrame", prepared.adata.obs), context=_context("observation annotations"))
    prepare_anndata_dataframe(cast("pd.DataFrame", prepared.adata.var), context=_context("feature annotations"))

    return parse_linked_table(
        prepared.adata,
        TableLinkage(regions=(dataset_id,), region_key=REGION_KEY, instance_key=INSTANCE_KEY),
        context=_context("linked table"),
        expected_instance_ids={dataset_id: shape_index},
    )
