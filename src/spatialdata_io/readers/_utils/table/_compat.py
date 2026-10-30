"""Prepare explicitly owned annotation frames for AnnData storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from pandas.api.extensions import ExtensionArray


def prepare_anndata_dataframe(
    frame: pd.DataFrame,
    /,
    *,
    context: ReaderErrorContext,
    allow_float_nulls: bool = False,
) -> pd.DataFrame:
    """Convert supported Arrow annotation storage in a reader-owned frame.

    Parameters
    ----------
    frame
        Owned eager annotation frame after final reader mutations. The caller
        must not pass a borrowed frame. Column order, labels, index names, and
        values are preserved; only general Arrow extension storage is replaced.
    context
        Reader and annotation artifact identity for unsupported representations.
    allow_float_nulls
        Whether the reader's semantic schema explicitly equates floating nulls
        with NaN. Defaults to rejecting Arrow floating columns containing nulls.
        When enabled, nulls become NaN and valid NaNs remain NaN.

    Returns
    -------
    pandas.DataFrame
        The same frame, with matching NumPy integer, boolean, and float32/64
        storage for nonnullable Arrow columns. Nullable integers and booleans
        use matching pandas extension types. General Arrow strings use pandas
        ArrowStringArray storage. Existing StringDtype, categorical, and non-Arrow
        arrays are retained. Arrow index storage follows the same rules.

    Raises
    ------
    ReaderFormatError
        If Arrow storage has no supported lossless mapping, or contains floating
        nulls without explicit authorization to equate them with NaN.

    Notes
    -----
    The function does not inspect AnnData, expression, or nested annotations,
    perform I/O, or change global settings. Compatible arrays are not converted
    or copied. Conversion is sequential; discard the owned frame after failure.
    The supported AnnData and pandas minimum versions persist nullable strings
    without writer opt-in under their default settings.
    """
    for position, dtype in enumerate(frame.dtypes):
        if isinstance(dtype, pd.ArrowDtype):
            values = _convert_arrow_values(
                frame.iloc[:, position],
                dtype,
                context=context,
                location=f"column {frame.columns[position]!r}",
                allow_float_nulls=allow_float_nulls,
            )
            frame.isetitem(position, values)

    index = frame.index
    if isinstance(index, pd.MultiIndex):
        levels = [_prepare_index(level, context=context, allow_float_nulls=allow_float_nulls) for level in index.levels]
        if any(new is not old for new, old in zip(levels, index.levels, strict=True)):
            # pandas accepts Index levels; pandas-stubs models them as nested sequences.
            frame.index = index.set_levels(levels)  # pyrefly: ignore[bad-argument-type]
    else:
        prepared = _prepare_index(index, context=context, allow_float_nulls=allow_float_nulls)
        if prepared is not index:
            frame.index = prepared

    return frame


def _prepare_index(
    index: pd.Index,
    *,
    context: ReaderErrorContext,
    allow_float_nulls: bool,
) -> pd.Index:
    """Replace general Arrow index storage while retaining its name and order."""
    dtype = index.dtype
    if not isinstance(dtype, pd.ArrowDtype):
        return index

    values = _convert_arrow_values(
        index,
        dtype,
        context=context,
        location=f"index {index.name!r}",
        allow_float_nulls=allow_float_nulls,
    )
    return pd.Index(values, name=index.name)


def _convert_arrow_values(
    values: pd.Series | pd.Index,
    dtype: pd.ArrowDtype,
    *,
    context: ReaderErrorContext,
    location: str,
    allow_float_nulls: bool,
) -> ExtensionArray | np.ndarray:
    """Map one Arrow column or index to supported storage with exact semantics."""
    arrow_type = dtype.pyarrow_dtype

    if pa.types.is_integer(arrow_type):
        if values.hasnans:
            prefix = "UInt" if pa.types.is_unsigned_integer(arrow_type) else "Int"
            return values.array.astype(f"{prefix}{arrow_type.bit_width}")
        return values.to_numpy(dtype=f"{arrow_type}", copy=False)

    if pa.types.is_boolean(arrow_type):
        if values.hasnans:
            return values.array.astype(pd.BooleanDtype())
        return values.to_numpy(dtype=np.bool_, copy=False)

    if pa.types.is_float32(arrow_type) or pa.types.is_float64(arrow_type):
        if not values.hasnans or allow_float_nulls:
            return values.to_numpy(dtype=f"float{arrow_type.bit_width}", na_value=np.nan, copy=False)
        expected = "floating storage without Arrow nulls, or explicit schema authorization to equate nulls with NaN"
    elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return values.array.astype(pd.StringDtype(storage="pyarrow"))
    else:
        expected = "Arrow integers, booleans, strings, or float32/64 with supported missingness semantics"

    raise ReaderFormatError(
        context=context,
        found=f"{location} with dtype {dtype}",
        expected=expected,
    )
