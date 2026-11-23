"""Validated linkage of reader-owned tables to SpatialData elements."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from spatialdata.models import TableModel

from spatialdata_io.readers._utils.errors import (
    ReaderErrorContext,
    ReaderFormatError,
    add_reader_context,
)
from spatialdata_io.readers._utils.table._normalize import (
    _require_owned_in_memory_anndata,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from anndata import AnnData


type _IdentifierFamily = Literal["integer", "string"]
type _IntegerDType = np.dtype[np.signedinteger[Any] | np.unsignedinteger[Any]]

_OBSERVED_INTEGER_DTYPES = frozenset(
    np.dtype(dtype)
    for dtype in (
        np.int16,
        np.int32,
        np.int64,
        np.uint16,
        np.uint32,
        np.uint64,
    )
)
_MISMATCH_PREVIEW_SIZE = 5


type _ObservedIdentifiers = _IntegerDType | Literal["string"]


@dataclass(frozen=True, slots=True)
class TableLinkage:
    """Describe the observation columns linking a table to spatial regions.

    Parameters
    ----------
    regions
        Ordered nonempty region names. Every region must occur in the table.
    region_key
        Observation column containing region names.
    instance_key
        Observation column containing region-local instance identifiers.

    Raises
    ------
    ValueError
        If regions are empty or repeated, either key is empty, or the keys are
        identical.
    """

    regions: tuple[str, ...]
    region_key: str
    instance_key: str

    def __post_init__(self) -> None:
        """Validate that regions are nonempty and unique and that the two keys are distinct and nonempty."""
        if not self.regions:
            message = "TableLinkage.regions must contain at least one region."
            raise ValueError(message)
        if not all(self.regions):
            message = "TableLinkage.regions must not contain empty names."
            raise ValueError(message)
        duplicates = tuple(region for region, count in Counter(self.regions).items() if count > 1)
        if duplicates:
            message = f"TableLinkage.regions must not contain duplicates; found {duplicates!r}."
            raise ValueError(message)
        if not self.region_key or not self.instance_key:
            message = "TableLinkage observation keys must be nonempty."
            raise ValueError(message)
        if self.region_key == self.instance_key:
            message = "TableLinkage.region_key and instance_key must be distinct."
            raise ValueError(message)


def _categorical_values(values: pd.Series | pd.Index) -> pd.Index | pd.Series:
    """Return the categories of categorical ``values``, otherwise ``values`` itself."""
    dtype = values.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return dtype.categories
    return values


def _inferred_family(values: pd.Series | pd.Index) -> _IdentifierFamily | None:
    """Classify identifiers as ``"integer"`` or ``"string"``.

    Categorical data is classified by its categories. Booleans and any other or mixed
    content return ``None``.
    """
    inspected = _categorical_values(values)
    dtype = inspected.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return None
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if isinstance(dtype, pd.StringDtype):
        return "string"
    inferred = pd.api.types.infer_dtype(inspected, skipna=False)
    if inferred == "integer":
        return "integer"
    if inferred in {"string", "unicode"}:
        return "string"
    return None


def _observed_identifier_kind(
    values: pd.Series,
    *,
    context: ReaderErrorContext,
    path: Path | None,
) -> _ObservedIdentifiers:
    """Return the kind of the table instance identifiers.

    Only strings and parser-compatible 16-, 32- or 64-bit integers are accepted.
    """
    family = _inferred_family(values)
    inspected = _categorical_values(values)
    if family == "string":
        return "string"
    if family == "integer":
        dtype = inspected.dtype
        if dtype in _OBSERVED_INTEGER_DTYPES:
            return cast("_IntegerDType", dtype)
    raise ReaderFormatError(
        context=context,
        path=path,
        found=f"instance identifiers with dtype {values.dtype!s}",
        expected=(
            "homogeneous strings or parser-compatible signed or unsigned 16-, 32-, or 64-bit integer identifiers"
        ),
    )


def _validate_region_values(
    values: pd.Series,
    linkage: TableLinkage,
    *,
    context: ReaderErrorContext,
    path: Path | None,
) -> None:
    """Require non-null string regions that are exactly the declared regions."""
    if bool(values.isna().any()):
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"null values in observation column {linkage.region_key!r}",
            expected="non-null string region identifiers",
        )
    if len(values) == 0:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="no observed regions",
            expected=f"exactly the declared regions {linkage.regions!r}",
        )
    if _inferred_family(values) != "string":
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"region identifiers with dtype {values.dtype!s}",
            expected="homogeneous non-null string region identifiers",
        )
    observed = tuple(pd.unique(values).tolist())
    if set(observed) != set(linkage.regions):
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"observed regions {observed!r}",
            expected=f"exactly the declared regions {linkage.regions!r}",
        )


def _validate_instance_values(
    values: pd.Series,
    *,
    context: ReaderErrorContext,
    path: Path | None,
) -> _ObservedIdentifiers:
    """Reject null instance identifiers and return their validated kind."""
    if bool(values.isna().any()):
        raise ReaderFormatError(
            context=context,
            path=path,
            found="null instance identifiers",
            expected="non-null homogeneous string or integer instance identifiers",
        )
    return _observed_identifier_kind(values, context=context, path=path)


def _validate_expected_index(
    index: pd.Index,
    *,
    region: str,
    context: ReaderErrorContext,
    path: Path | None,
) -> _IdentifierFamily:
    """Require a unique, non-null, one-level integer or string target index and return its family."""
    if index.nlevels != 1:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"a {index.nlevels}-level target index for region {region!r}",
            expected="a one-level target instance index",
        )
    if index.hasnans:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"null target identifiers for region {region!r}",
            expected="non-null target instance identifiers",
        )
    if not index.is_unique:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"duplicate target identifiers for region {region!r}",
            expected="unique target instance identifiers",
        )
    family = _inferred_family(index)
    if family is None:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"target identifiers with dtype {index.dtype!s} for region {region!r}",
            expected="homogeneous non-boolean integer or string target identifiers",
        )
    return family


def _integer_comparison_values(
    index: pd.Index,
    *,
    dtype: _IntegerDType,
    region: str,
    context: ReaderErrorContext,
    path: Path | None,
) -> np.ndarray:
    """Return target identifiers cast to the observed integer dtype.

    A value outside the range of ``dtype`` is an error, so the cast is always lossless.
    """
    values = index.to_numpy(copy=False)
    if values.size == 0:
        return values.astype(dtype, copy=False)
    minimum = int(values.min())
    maximum = int(values.max())
    limits = np.iinfo(dtype)
    if minimum < limits.min or maximum > limits.max:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"target identifier range [{minimum}, {maximum}] for region {region!r}",
            expected=f"values losslessly representable by observed dtype {dtype!s}",
        )
    return values.astype(dtype, copy=False)


def _validate_membership(  # noqa: PLR0913 - compare one region with its diagnostic context.
    observed: pd.Index,
    target: pd.Index,
    kind: _ObservedIdentifiers,
    *,
    region: str,
    context: ReaderErrorContext,
    path: Path | None,
) -> None:
    """Compare unique IDs within one region, preserving integer precision."""
    family = _validate_expected_index(target, region=region, context=context, path=path)
    expected_family = "string" if isinstance(kind, str) else "integer"
    if family != expected_family:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"{family} target identifiers for region {region!r}",
            expected=f"{expected_family} identifiers matching the table instance column",
        )
    if not isinstance(kind, str):
        target = pd.Index(_integer_comparison_values(target, dtype=kind, region=region, context=context, path=path))
    if observed.equals(target):
        return
    missing = target.difference(observed, sort=False)
    unexpected = observed.difference(target, sort=False)
    if len(missing) or len(unexpected):
        raise ReaderFormatError(
            context=context,
            path=path,
            found=(
                f"{len(missing)} target pairs missing from the table and {len(unexpected)} unexpected table pairs "
                f"in region {region!r}; missing preview {missing[:_MISMATCH_PREVIEW_SIZE].tolist()!r}, "
                f"unexpected preview {unexpected[:_MISMATCH_PREVIEW_SIZE].tolist()!r}"
            ),
            expected="exact target-instance membership for every supplied region",
        )


def _validate_instances(
    adata: AnnData,
    linkage: TableLinkage,
    targets: Mapping[str, pd.Index],
    *,
    context: ReaderErrorContext,
    path: Path | None,
) -> None:
    """Validate the linkage columns and supplied target indexes region by region.

    Regions must be exactly the declared set and every pair must be unique.
    """
    missing_columns = tuple(key for key in (linkage.region_key, linkage.instance_key) if key not in adata.obs)
    if missing_columns:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"missing observation columns {missing_columns!r}",
            expected=f"linkage columns {linkage.region_key!r} and {linkage.instance_key!r}",
        )
    obs = cast("pd.DataFrame", adata.obs)
    region_values = obs[linkage.region_key]
    instance_values = obs[linkage.instance_key]
    _validate_region_values(region_values, linkage, context=context, path=path)
    identifiers = _validate_instance_values(instance_values, context=context, path=path)
    groups = (
        ((linkage.regions[0], instance_values),)
        if len(linkage.regions) == 1
        else instance_values.groupby(region_values, observed=True, sort=False)
    )
    for region, values in groups:
        index = pd.Index(values.array)
        if not index.is_unique:
            preview = index[index.duplicated(keep=False)][:_MISMATCH_PREVIEW_SIZE].tolist()
            raise ReaderFormatError(
                context=context,
                path=path,
                found=f"duplicate (region, instance) pairs in region {region!r}; preview {preview!r}",
                expected="unique instance identifiers within each region",
            )
        if region in targets:
            _validate_membership(index, targets[region], identifiers, region=region, context=context, path=path)


def parse_linked_table(
    adata: AnnData,
    linkage: TableLinkage,
    /,
    *,
    context: ReaderErrorContext,
    path: Path | None = None,
    expected_instance_ids: Mapping[str, pd.Index] | None = None,
) -> AnnData:
    """Link a reader-owned table after its expression and annotations are final.

    Parameters
    ----------
    adata
        Owned in-memory table. Its producer supplies finite canonical CSR expression.
    linkage
        Region and instance columns and the ordered target regions.
    context
        Reader and artifact identity for validation and parser failures.
    path
        Source artifact for diagnostics.
    expected_instance_ids
        Independently materialized target indexes for any subset of regions.
        Each supplied index must match the unique instances in its region.

    Returns
    -------
    anndata.AnnData
        The same table with SpatialData linkage metadata.
    """
    _require_owned_in_memory_anndata(adata)
    if TableModel.ATTRS_KEY in adata.uns:
        # Imported tables can carry stale linkage metadata from their source.
        error = ValueError(f"adata.uns[{TableModel.ATTRS_KEY!r}] must be absent before linkage.")
        add_reader_context(error, context=context, path=path)
        raise error
    if expected_instance_ids is not None:
        unexpected_regions = tuple(region for region in expected_instance_ids if region not in linkage.regions)
        if unexpected_regions:
            message = f"expected_instance_ids contains regions outside TableLinkage.regions: {unexpected_regions!r}."
            raise ValueError(message)

    _validate_instances(adata, linkage, expected_instance_ids or {}, context=context, path=path)

    region: str | list[str] = linkage.regions[0] if len(linkage.regions) == 1 else list(linkage.regions)
    try:
        return TableModel.parse(
            adata,
            region=region,
            region_key=linkage.region_key,
            instance_key=linkage.instance_key,
        )
    except Exception as error:
        add_reader_context(error, context=context, path=path)
        raise
