"""Read the spot positions and the 10x counts and link them into one sparse Visium table.

The count matrix decides the observations: its barcodes give the row names
and order, and every barcode must have exactly one position row. Position rows
without counts (spots not in a filtered matrix) are dropped; nothing is
filtered by ``in_tissue``, so raw counts keep spots outside the tissue::

    counts barcodes   b  a            positions   a  b  c
                      |  |                        |  |  |
    obs               b  a  <-------------------- a  b     (c has no counts)

The table gains the grid fields as ``obs`` columns, the spot centre as
``obsm["spatial_pixels"]`` in ``x, y`` order (``pxl_col_in_fullres``,
``pxl_row_in_fullres``), the barcode again as the instance key ``spot_id``,
and ``region``. ``uns["spatial"][<dataset_id>]`` keeps the scale-factor JSON
and the HDF5 identity text as Scanpy's Visium reader stores them, without
images.

Positions are parsed by Arrow as text, projected to the six fields, with
quoted newlines allowed. Arrow checks every record's field count but not
strict quoting, so a malformed quote after the first record (which discovery
parses strictly) may be read as a different value.

Counts are decoded by the shared 10x readers (Scanpy) with every feature type
and the source feature order. Duplicate sparse entries are summed; the result
is float32 when every count is exact in float32, otherwise float64.
"""

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
from spatialdata_io.readers._utils.table import (
    TableLinkage,
    finalize_count_arrays,
    parse_linked_table,
    prepare_anndata_dataframe,
)
from spatialdata_io.readers.visium._layout import H5Counts
from spatialdata_io.readers.visium._schemas import (
    INSTANCE_KEY,
    OBS_POSITION_COLUMNS,
    POSITION_COLUMNS,
    POSITION_SOURCE_SCHEMA,
    REGION_KEY,
    SPOT_ID_DTYPE,
    normalize_positions,
)

if TYPE_CHECKING:
    from anndata import AnnData
    from scipy.sparse import csr_array

    from spatialdata_io.readers.visium._layout import VisiumLayout
    from spatialdata_io.readers.visium._metadata import ScaleFactors


@dataclass(frozen=True, slots=True)
class PreparedTable:
    """Owned unlinked table and its positions in observation order."""

    adata: AnnData
    positions: pd.DataFrame


def read_positions(layout: VisiumLayout) -> pd.DataFrame:
    """Read the six position fields as text with strict record widths and normalize them."""
    context = ReaderErrorContext("visium", "tissue positions", layout.format_version)
    try:
        with layout.positions.open("rb") as source:
            table = pacsv.read_csv(
                source,
                read_options=pacsv.ReadOptions(
                    column_names=list(POSITION_COLUMNS) if layout.position_format == "headerless" else None,
                ),
                # Quoted barcodes may span native Arrow CSV block boundaries.
                parse_options=pacsv.ParseOptions(newlines_in_values=True),
                convert_options=pacsv.ConvertOptions(
                    column_types=POSITION_SOURCE_SCHEMA,
                    include_columns=list(POSITION_COLUMNS),
                    strings_can_be_null=False,
                ),
            )
    except (pa.ArrowInvalid, pa.ArrowKeyError) as error:
        raise ReaderFormatError(
            context=context,
            path=layout.positions,
            found=str(error),
            expected="required columns and consistent CSV record widths",
        ) from error
    except OSError as error:
        add_reader_context(error, context=context, path=layout.positions)
        raise

    return normalize_positions(table.to_pandas(types_mapper=pd.ArrowDtype), path=layout.positions, context=context)


def _read_counts(layout: VisiumLayout, *, make_var_names_unique: bool) -> AnnData:
    """Read the selected count source and apply the feature-name policy."""
    context = ReaderErrorContext("visium", "count matrix", layout.format_version)
    counts = layout.counts
    if isinstance(counts, H5Counts):
        adata = read_10x_h5(counts.path, context=context)
        if make_var_names_unique:
            adata.var_names_make_unique()
        return adata

    return read_10x_mtx(
        counts.path,
        context=context,
        make_unique=make_var_names_unique,
        prefix=counts.prefix,
        compressed=counts.compressed,
    )


def _spatial_metadata(layout: VisiumLayout) -> dict[str, str]:
    """Return the HDF5 identity text that Scanpy keeps in ``uns["spatial"]``."""
    if not isinstance(layout.counts, H5Counts):
        return {}
    metadata = layout.counts.metadata
    return {
        key: value
        for key, value in (
            ("chemistry_description", metadata.chemistry_description),
            ("software_version", metadata.software_version),
        )
        if value is not None
    }


def prepare_table(  # noqa: PLR0913 - explicit reader-owned table construction inputs.
    layout: VisiumLayout,
    positions: pd.DataFrame,
    *,
    dataset_id: str,
    pixel_frame: str,
    scale_factors: ScaleFactors,
    make_var_names_unique: bool,
) -> PreparedTable:
    """Read sparse counts and align position metadata without filtering spots."""
    adata = _read_counts(layout, make_var_names_unique=make_var_names_unique)
    if not adata.obs_names.is_unique:
        duplicates = tuple(adata.obs_names[adata.obs_names.duplicated()].unique()[:5])
        raise ReaderFormatError(
            context=ReaderErrorContext("visium", "count matrix", layout.format_version),
            found=f"duplicate observation barcodes {duplicates!r}",
            expected="unique count-matrix barcodes",
            path=layout.counts.path,
        )

    missing = adata.obs_names.difference(positions.index, sort=False)
    if len(missing):
        raise ReaderFormatError(
            context=ReaderErrorContext("visium", "tissue positions", layout.format_version),
            found=f"count barcodes absent from tissue positions {tuple(missing[:5])!r}",
            expected="every count barcode exactly once in the position table",
            path=layout.positions,
        )

    aligned = positions.loc[adata.obs_names]
    obs = cast("pd.DataFrame", adata.obs)
    for column in OBS_POSITION_COLUMNS:
        obs[column] = aligned[column].array
    obs[INSTANCE_KEY] = pd.Series(adata.obs_names, dtype=SPOT_ID_DTYPE).array
    obs[REGION_KEY] = pd.Categorical(np.repeat(dataset_id, adata.n_obs), categories=[dataset_id])

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
                "frame": pixel_frame,
                "position": "full-resolution spot centre",
            }
        },
    }
    adata.uns[POSITIONS_KEY] = description
    adata.uns["spatial"] = {
        dataset_id: {
            "metadata": _spatial_metadata(layout),
            "scalefactors": scale_factors.values,  # noqa: PD011 - Owned JSON mapping, not a pandas object.
        }
    }

    return PreparedTable(adata=adata, positions=aligned)


def link_table(adata: AnnData, *, dataset_id: str, shape_index: pd.Index, layout: VisiumLayout) -> AnnData:
    """Finalize the counts and link the table to the spot circles exactly once."""
    context = ReaderErrorContext("visium", "count matrix", layout.format_version)
    # The shared 10x readers return CSR arrays, possibly with duplicate entries
    # or integer dtypes; alignment above changed annotations only.
    adata.X = finalize_count_arrays((cast("csr_array", adata.X),), context=context, path=layout.counts.path)[0]
    # The shared count readers return in-memory frames; AnnData's annotations
    # additionally cover Dataset2D, used only for backed tables.
    prepare_anndata_dataframe(cast("pd.DataFrame", adata.obs), context=context)
    prepare_anndata_dataframe(cast("pd.DataFrame", adata.var), context=context)

    return parse_linked_table(
        adata,
        TableLinkage(regions=(dataset_id,), region_key=REGION_KEY, instance_key=INSTANCE_KEY),
        context=context,
        path=layout.counts.path,
        expected_instance_ids={dataset_id: shape_index},
    )
