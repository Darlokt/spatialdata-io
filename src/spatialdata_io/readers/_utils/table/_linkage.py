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
    normalize_owned_anndata,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from anndata import AnnData

    from spatialdata_io.readers._utils.table._duplicates import SparseDuplicateSummary

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


@dataclass(frozen=True, slots=True)
class _IntegerIdentifiers:
    """Integer instance identifiers and their observed dtype."""

    dtype: _IntegerDType


@dataclass(frozen=True, slots=True)
class _StringIdentifiers:
    """String instance identifiers."""


type _ObservedIdentifiers = _IntegerIdentifiers | _StringIdentifiers


@dataclass(frozen=True, slots=True)
class _ValidatedIdentifiers:
    """Unique observed ``(region, instance)`` pairs and the kind of their instance identifiers."""

    pairs: pd.MultiIndex
    identifiers: _ObservedIdentifiers


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
        return _StringIdentifiers()
    if family == "integer":
        dtype = inspected.dtype
        if dtype in _OBSERVED_INTEGER_DTYPES:
            return _IntegerIdentifiers(cast("_IntegerDType", dtype))
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


def _expected_pairs(
    expected_instance_ids: Mapping[str, pd.Index],
    *,
    observed: _ObservedIdentifiers,
    context: ReaderErrorContext,
    path: Path | None,
) -> pd.MultiIndex:
    """Return the ``(region, instance)`` pairs of every supplied target index.

    Each index must have the same identifier family as the table instance column.
    """
    observed_family: _IdentifierFamily = "integer" if isinstance(observed, _IntegerIdentifiers) else "string"
    region_arrays: list[np.ndarray] = []
    instance_arrays: list[np.ndarray] = []
    for region, index in expected_instance_ids.items():
        family = _validate_expected_index(index, region=region, context=context, path=path)
        if family != observed_family:
            raise ReaderFormatError(
                context=context,
                path=path,
                found=f"{family} target identifiers for region {region!r}",
                expected=f"{observed_family} identifiers matching the table instance column",
            )
        if isinstance(observed, _IntegerIdentifiers):
            instances = _integer_comparison_values(
                index, dtype=observed.dtype, region=region, context=context, path=path
            )
        else:
            instances = index.to_numpy(dtype=object, copy=False)
        instance_arrays.append(instances)
        region_arrays.append(np.full(len(index), region, dtype=object))
    return pd.MultiIndex.from_arrays((np.concatenate(region_arrays), np.concatenate(instance_arrays)))


def _validate_expected_membership(
    observed: _ValidatedIdentifiers,
    expected_instance_ids: Mapping[str, pd.Index],
    *,
    context: ReaderErrorContext,
    path: Path | None,
) -> None:
    """Require exact table membership for every region with a supplied target index."""
    expected_pairs = _expected_pairs(expected_instance_ids, observed=observed.identifiers, context=context, path=path)
    region_values = observed.pairs.get_level_values(0)
    known_observed_pairs = observed.pairs[region_values.isin(list(expected_instance_ids))]
    missing_from_table = expected_pairs.difference(known_observed_pairs, sort=False)
    unexpected_in_table = known_observed_pairs.difference(expected_pairs, sort=False)
    if len(missing_from_table) == 0 and len(unexpected_in_table) == 0:
        return
    missing_preview = missing_from_table[:_MISMATCH_PREVIEW_SIZE].tolist()
    unexpected_preview = unexpected_in_table[:_MISMATCH_PREVIEW_SIZE].tolist()
    raise ReaderFormatError(
        context=context,
        path=path,
        found=(
            f"{len(missing_from_table)} target pairs missing from the table and "
            f"{len(unexpected_in_table)} unexpected table pairs; "
            f"missing preview {missing_preview!r}, unexpected preview {unexpected_preview!r}"
        ),
        expected="exact target-instance membership for every supplied region",
    )


def _validated_pairs(
    adata: AnnData,
    linkage: TableLinkage,
    *,
    context: ReaderErrorContext,
    path: Path | None,
) -> _ValidatedIdentifiers:
    """Validate the linkage columns of ``adata.obs`` and return their observed pairs.

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
    observed_pairs = pd.MultiIndex.from_arrays((region_values, instance_values))
    if not observed_pairs.is_unique:
        duplicate_preview = observed_pairs[observed_pairs.duplicated(keep=False)][:_MISMATCH_PREVIEW_SIZE].tolist()
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"duplicate (region, instance) pairs; preview {duplicate_preview!r}",
            expected="unique instance identifiers within each region",
        )
    return _ValidatedIdentifiers(observed_pairs, identifiers)


def parse_linked_table(  # noqa: PLR0913 - explicit representation and linkage options have separate owners
    adata: AnnData,
    linkage: TableLinkage,
    /,
    *,
    context: ReaderErrorContext,
    path: Path | None = None,
    expression_layers: tuple[str, ...] = (),
    expected_instance_ids: Mapping[str, pd.Index] | None = None,
    duplicate_sums: list[SparseDuplicateSummary] | None = None,
) -> AnnData:
    """Normalize and link a reader-owned table to SpatialData elements.

    The table must be an owned, eager, in-memory :class:`~anndata.AnnData`.
    Region, instance, and optional exact target membership are validated before
    expression mutation. ``X`` and declared expression layers are then
    normalized to canonical CSR sparse arrays before the public
    :meth:`spatialdata.models.TableModel.parse` boundary is called exactly once.

    Parameters
    ----------
    adata
        Exclusively reader-owned, unparsed, in-memory table.
    linkage
        Declared regions and observation linkage columns.
    context
        Reader and artifact identity for linkage, normalization and parser
        failures.
    path
        Optional source artifact of the table, added to the same failures.
    expression_layers
        Named layers whose documented meaning is expression or counts.
    expected_instance_ids
        Optional borrowed target indexes keyed by a subset of declared regions.
        Each supplied index must exactly match table instances in that region;
        index order is irrelevant.
    duplicate_sums
        Optional reader-owned summary destination. Integer duplicate reduction
        is lossless with or without this collector, as in :func:`normalize_owned_anndata`.
        The reader emits one summarized normalization warning after successfully
        assembling the complete result.

    Returns
    -------
    anndata.AnnData
        The identical normalized and linked table.

    Raises
    ------
    ValueError
        If the descriptor, ownership, stale metadata, or mapping keys violate
        the reader-authored call contract, or normalization finds invalid data.
    TypeError
        If the table is backed or expression storage is unsupported.
    KeyError
        If a declared expression layer is absent.
    ReaderFormatError
        If external linkage identifiers or target membership are malformed.

    Notes
    -----
    Linkage failures before normalization leave ``adata`` unchanged. A
    normalization or parser failure can leave the exclusively owned table
    partially mutated; callers must discard it. The function never
    copies the complete table, densifies expression data, or computes a spatial
    element. Its local validation uses linear identifier work and storage;
    SpatialData may perform additional public model validation.
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

    observed = _validated_pairs(adata, linkage, context=context, path=path)
    if expected_instance_ids:
        _validate_expected_membership(observed, expected_instance_ids, context=context, path=path)

    region: str | list[str] = linkage.regions[0] if len(linkage.regions) == 1 else list(linkage.regions)
    try:
        normalize_owned_anndata(adata, expression_layers=expression_layers, duplicate_sums=duplicate_sums)
        return TableModel.parse(
            adata,
            region=region,
            region_key=linkage.region_key,
            instance_key=linkage.instance_key,
        )
    except Exception as error:
        add_reader_context(error, context=context, path=path)
        raise
