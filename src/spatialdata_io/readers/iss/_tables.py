"""Read the owned ISS expression table and establish exact label identifiers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers.iss._schemas import (
    DEFAULT_INSTANCE_KEY,
    MIN_INSTANCE_BITS,
    _validate_index_ids,
    _validate_instance_ids,
    _validate_output_column,
    _validate_table,
)

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import ArtifactFile


def _read_table(path: ArtifactFile, *, instance_key: str | None, label_dtype: np.dtype[np.generic]) -> ad.AnnData:
    """Load an owned eager expression table and establish final feature and cell IDs."""
    context = ReaderErrorContext("iss", "counts")
    try:
        table = ad.read_h5ad(path)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=context, path=path)
        raise

    _validate_table(table, path=path, instance_key=instance_key)

    # AnnData cannot express the eager frame invariant established by the schema.
    _set_instance_ids(cast("pd.DataFrame", table.obs), instance_key=instance_key, label_dtype=label_dtype, path=path)
    table.var_names_make_unique()

    return table


def _set_instance_ids(
    obs: pd.DataFrame, *, instance_key: str | None, label_dtype: np.dtype[np.generic], path: ArtifactFile
) -> None:
    """Set exact positive integer IDs in an owned frame within the label dtype's range."""
    context = ReaderErrorContext("iss", "counts")

    # Parse decimal index strings as Python integers. Float64 loses IDs above 2**53.
    key = DEFAULT_INSTANCE_KEY if instance_key is None else instance_key
    if instance_key is None:
        try:
            values = [int(value) for value in obs.index]
        except ValueError as error:
            raise ReaderFormatError(
                context=context,
                path=path,
                found="nonnumeric observation names",
                expected="integer label IDs in the index, or an explicit instance_key",
            ) from error
        _validate_index_ids(values, label_dtype=label_dtype, path=path)

        index_dtype = np.uint64 if values and max(values) > np.iinfo(np.int64).max else np.int64
        column = pd.Series(values, index=obs.index, dtype=pd.ArrowDtype(pa.from_numpy_dtype(index_dtype)))
        _validate_output_column(obs, key, column, path=path)
    else:
        column = obs[key]
        _validate_instance_ids(column, label_dtype=label_dtype, path=path)

        # Linkage requires at least 16-bit integers; wider IDs keep signedness.
        arrow_type = (
            column.dtype.pyarrow_dtype
            if isinstance(column.dtype, pd.ArrowDtype)
            else pa.from_numpy_dtype(np.dtype(str(column.dtype).lower()))
        )
        if arrow_type.bit_width < MIN_INSTANCE_BITS:
            arrow_type = pa.uint16() if pa.types.is_unsigned_integer(arrow_type) else pa.int16()
        column = column.astype(pd.ArrowDtype(arrow_type))

    obs[key] = column
