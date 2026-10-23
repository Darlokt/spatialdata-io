"""Read the owned ISS expression table and establish exact label identifiers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context

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

    if table.X is None:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="missing expression X",
            expected="an observation-by-feature expression array",
        )
    if not isinstance(table.obs, pd.DataFrame):
        message = "ISS requires an eager observation DataFrame from read_h5ad."
        raise TypeError(message)
    if (
        not table.obs_names.is_unique
        or bool(table.obs_names.isin([""]).any())
        or bool(table.var_names.isin([""]).any())
    ):
        raise ReaderFormatError(
            context=context,
            path=path,
            found="duplicate observations or empty observation/feature identifiers",
            expected="unique nonempty observation names and nonempty feature names",
        )

    _set_instance_ids(table.obs, instance_key=instance_key, label_dtype=label_dtype, path=path)
    table.var_names_make_unique()
    return table


def _set_instance_ids(
    obs: pd.DataFrame, *, instance_key: str | None, label_dtype: np.dtype[np.generic], path: ArtifactFile
) -> None:
    """Set exact positive integer IDs in an owned frame within the label dtype's range."""
    context = ReaderErrorContext("iss", "counts")

    # Parse decimal index strings as Python integers. Float64 loses IDs above 2**53.
    label_max = np.iinfo(label_dtype.str).max
    key = "instance_id" if instance_key is None else instance_key
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
        if any(value <= 0 or value > label_max for value in values):
            raise ReaderFormatError(
                context=context,
                path=path,
                found="nonpositive or out-of-range index identifiers",
                expected=f"positive label IDs representable by {label_dtype}. Zero is background",
            )

        index_dtype = np.uint64 if values and max(values) > np.iinfo(np.int64).max else np.int64
        obs[key] = np.asarray(values, dtype=index_dtype)
    else:
        if key not in obs:
            raise ReaderFormatError(
                context=context,
                path=path,
                found=f"missing observation column {key!r}",
                expected="the requested integer instance column",
            )

        column = obs[key]
        if (
            column.isna().any()
            or pd.api.types.is_bool_dtype(column.dtype)
            or not pd.api.types.is_integer_dtype(column.dtype)
        ):
            raise ReaderFormatError(
                context=context,
                path=path,
                found=f"null or noninteger identifiers with dtype {column.dtype}",
                expected="nonnull positive integer label IDs",
            )
        if len(column) and (int(column.min()) <= 0 or int(column.max()) > label_max):
            raise ReaderFormatError(
                context=context,
                path=path,
                found="nonpositive or out-of-range instance identifiers",
                expected=f"positive label IDs representable by {label_dtype}. Zero is background",
            )

        # AnnData's parser needs >=16-bit integer columns. Nullable integer masks
        # are removed only after rejecting nulls, without changing numeric values.
        dtype = np.dtype(str(column.dtype).lower())
        if dtype.itemsize == 1:
            dtype = np.dtype(np.uint16 if dtype.kind == "u" else np.int16)
        obs[key] = column.to_numpy(dtype=dtype, copy=False)
