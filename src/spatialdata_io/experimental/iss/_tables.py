"""Read the ISS ``.h5ad`` table and add the columns that link it to the labels.

The table is read eagerly into a reader-owned AnnData; the source file and any
caller objects are never modified. Each row is linked to one label through two
observation columns:

- ``region``: categorical, the name of the labels element, on every row;
- the instance column: the caller's ``instance_key`` (integers kept, widened to
  16 bits when narrower), or ``instance_id`` parsed from observation names that
  consist only of ASCII digits. Parsed IDs are exact Python integers, stored as
  int64, or uint64 when an ID needs it, so values above ``2**53`` stay exact.

Imported ``obs`` and ``var`` column names and top-level ``uns`` keys that a
SpatialData table does not accept (``my note``) are sanitized with one warning
per frame or mapping before the linkage columns are added; ``region``, the
instance column and the ``uns`` keys ``spatialdata_io_positions`` and
``spatialdata_attrs`` keep their names. ``uns`` values, their order and nested
keys stay as stored.
Feature names are made unique (``GeneB``, ``GeneB-1``) in their source order,
and an imported ``obsm["spatial"]`` moves to ``spatial_source`` unchanged,
because its axes and units are unknown. Expression is normalized later by the
shared linkage step.
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
from spatialdata.models import TableModel

from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.experimental.iss._schemas import (
    DEFAULT_INSTANCE_KEY,
    MIN_INSTANCE_BITS,
    REGION_KEY,
    TABLE_CONTEXT,
    validate_index_ids,
    validate_instance_ids,
    validate_output_column,
    validate_table,
)
from spatialdata_io.readers._utils.errors import ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.names import sanitize_column_names, sanitize_uns_keys
from spatialdata_io.readers._utils.table import prepare_anndata_dataframe

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import ArtifactFile

# uns keys reserved for positions and linkage; imported case variants are renamed.
_RESERVED_UNS = (POSITIONS_KEY, TableModel.ATTRS_KEY)
_DUPLICATE_VARIABLES_WARNING = (
    re.escape("Variable names are not unique. To make them unique, call `.var_names_make_unique`.") + r"\Z"
)


def read_table(
    path: ArtifactFile, *, region: str, instance_key: str | None, label_dtype: np.dtype[np.integer]
) -> ad.AnnData:
    """Read the owned table and set its final feature names and linkage columns."""
    try:
        with warnings.catch_warnings():
            # The reader makes feature names unique below, so AnnData's advice to do so is noise.
            warnings.filterwarnings("ignore", message=_DUPLICATE_VARIABLES_WARNING, category=UserWarning)
            table = ad.read_h5ad(path)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=TABLE_CONTEXT, path=path)
        raise

    validate_table(table, path=path, instance_key=instance_key)
    # read_h5ad without backed mode loads eager DataFrames; AnnData also types backed Dataset2D.
    obs, var = cast("pd.DataFrame", table.obs), cast("pd.DataFrame", table.var)
    # The caller's instance_key names a column, so it keeps its spelling.
    obs.columns = sanitize_column_names(
        list(obs.columns), source="iss: the H5AD obs", reserved=(REGION_KEY, instance_key or DEFAULT_INSTANCE_KEY)
    )
    var.columns = sanitize_column_names(list(var.columns), source="iss: the H5AD var")
    sanitize_uns_keys(table, source="iss: the H5AD uns", reserved=_RESERVED_UNS)
    if "spatial" in table.obsm:
        table.obsm["spatial_source"] = table.obsm.pop("spatial")
    table.var_names_make_unique()

    _set_instance_ids(obs, instance_key=instance_key, label_dtype=label_dtype, path=path)
    regions = pd.Series(pd.Categorical([region] * len(obs)), index=obs.index)
    validate_output_column(obs, REGION_KEY, regions, path=path)
    obs[REGION_KEY] = regions

    prepare_anndata_dataframe(obs, context=TABLE_CONTEXT)
    prepare_anndata_dataframe(var, context=TABLE_CONTEXT)
    return table


def _set_instance_ids(
    obs: pd.DataFrame, *, instance_key: str | None, label_dtype: np.dtype[np.integer], path: ArtifactFile
) -> None:
    """Set exact positive integer IDs within the label dtype's range in an owned frame."""
    if instance_key is None:
        # int() also accepts signs, underscores, whitespace and non-ASCII digits.
        names = obs.index.tolist()
        if not all(isinstance(name, str) and name.isascii() and name.isdigit() for name in names):
            raise ReaderFormatError(
                context=TABLE_CONTEXT,
                path=path,
                found="observation names that are not ASCII decimal digits",
                expected="integer label IDs as observation names, or an explicit instance_key",
            )
        # Python integers keep IDs above 2**53 exact.
        values = [int(name) for name in names]
        validate_index_ids(values, label_dtype=label_dtype, path=path)

        index_dtype = np.uint64 if values and max(values) > np.iinfo(np.int64).max else np.int64
        column = pd.Series(values, index=obs.index, dtype=pd.ArrowDtype(pa.from_numpy_dtype(index_dtype)))
        validate_output_column(obs, DEFAULT_INSTANCE_KEY, column, path=path)
        obs[DEFAULT_INSTANCE_KEY] = column
        return

    column = obs[instance_key]
    validate_instance_ids(column, label_dtype=label_dtype, path=path)
    if isinstance(column.dtype, np.dtype) and column.dtype.itemsize * 8 >= MIN_INSTANCE_BITS:
        return

    # Narrow and nullable-extension IDs move to Arrow storage, keeping their signedness.
    arrow_type = (
        column.dtype.pyarrow_dtype
        if isinstance(column.dtype, pd.ArrowDtype)
        else pa.from_numpy_dtype(np.dtype(str(column.dtype).lower()))
    )
    if arrow_type.bit_width < MIN_INSTANCE_BITS:
        arrow_type = pa.uint16() if pa.types.is_unsigned_integer(arrow_type) else pa.int16()
    obs[instance_key] = column.astype(pd.ArrowDtype(arrow_type))
