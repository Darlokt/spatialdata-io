"""Validated linkage of reader-owned tables to SpatialData elements."""

from __future__ import annotations

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


@dataclass(frozen=True, slots=True)
class _IntegerIdentifiers:
    dtype: _IntegerDType


@dataclass(frozen=True, slots=True)
class _StringIdentifiers:
    pass


type _ObservedIdentifiers = _IntegerIdentifiers | _StringIdentifiers


@dataclass(frozen=True, slots=True)
class _ValidatedIdentifiers:
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
        if not self.regions:
            message = "TableLinkage.regions must contain at least one region."
            raise ValueError(message)
        empty_regions = tuple(region for region in self.regions if not region)
        if empty_regions:
            message = "TableLinkage.regions must not contain empty names."
            raise ValueError(message)
        seen: set[str] = set()
        duplicate_set: set[str] = set()
        duplicate_regions: list[str] = []
        for region in self.regions:
            if region in seen and region not in duplicate_set:
                duplicate_regions.append(region)
                duplicate_set.add(region)
            seen.add(region)
        if duplicate_regions:
            message = f"TableLinkage.regions must not contain duplicates; found {tuple(duplicate_regions)!r}."
            raise ValueError(message)
        if not self.region_key or not self.instance_key:
            message = "TableLinkage observation keys must be nonempty."
            raise ValueError(message)
        if self.region_key == self.instance_key:
            message = "TableLinkage.region_key and instance_key must be distinct."
            raise ValueError(message)


def _reader_format_error(
    *,
    context: ReaderErrorContext,
    found: str,
    expected: str,
) -> ReaderFormatError:
    return ReaderFormatError(context=context, found=found, expected=expected)


def _categorical_values(values: pd.Series | pd.Index) -> pd.Index | pd.Series:
    dtype = values.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return dtype.categories
    return values


def _inferred_family(values: pd.Series | pd.Index) -> _IdentifierFamily | None:
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
) -> _ObservedIdentifiers:
    family = _inferred_family(values)
    inspected = _categorical_values(values)
    if family == "string":
        return _StringIdentifiers()
    if family == "integer":
        dtype = inspected.dtype
        if dtype in _OBSERVED_INTEGER_DTYPES:
            return _IntegerIdentifiers(cast("_IntegerDType", dtype))
    raise _reader_format_error(
        context=context,
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
) -> None:
    if bool(values.isna().any()):
        raise _reader_format_error(
            context=context,
            found=f"null values in observation column {linkage.region_key!r}",
            expected="non-null string region identifiers",
        )
    if len(values) == 0:
        raise _reader_format_error(
            context=context,
            found="no observed regions",
            expected=f"exactly the declared regions {linkage.regions!r}",
        )
    if _inferred_family(values) != "string":
        raise _reader_format_error(
            context=context,
            found=f"region identifiers with dtype {values.dtype!s}",
            expected="homogeneous non-null string region identifiers",
        )
    observed = tuple(pd.unique(values).tolist())
    if set(observed) != set(linkage.regions):
        raise _reader_format_error(
            context=context,
            found=f"observed regions {observed!r}",
            expected=f"exactly the declared regions {linkage.regions!r}",
        )


def _validate_instance_values(
    values: pd.Series,
    *,
    context: ReaderErrorContext,
) -> _ObservedIdentifiers:
    if bool(values.isna().any()):
        raise _reader_format_error(
            context=context,
            found="null instance identifiers",
            expected="non-null homogeneous string or integer instance identifiers",
        )
    return _observed_identifier_kind(values, context=context)


def _validate_expected_index(
    index: pd.Index,
    *,
    region: str,
    context: ReaderErrorContext,
) -> _IdentifierFamily:
    if index.nlevels != 1:
        raise _reader_format_error(
            context=context,
            found=f"a {index.nlevels}-level target index for region {region!r}",
            expected="a one-level target instance index",
        )
    if index.hasnans:
        raise _reader_format_error(
            context=context,
            found=f"null target identifiers for region {region!r}",
            expected="non-null target instance identifiers",
        )
    if not index.is_unique:
        raise _reader_format_error(
            context=context,
            found=f"duplicate target identifiers for region {region!r}",
            expected="unique target instance identifiers",
        )
    family = _inferred_family(index)
    if family is None:
        raise _reader_format_error(
            context=context,
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
) -> np.ndarray:
    if len(index) == 0:
        return np.empty(0, dtype=dtype)
    values = index.to_numpy(copy=False)
    minimum = int(values.min())
    maximum = int(values.max())
    limits = np.iinfo(dtype)
    if minimum < limits.min or maximum > limits.max:
        raise _reader_format_error(
            context=context,
            found=f"target identifier range [{minimum}, {maximum}] for region {region!r}",
            expected=f"values losslessly representable by observed dtype {dtype!s}",
        )
    return values.astype(dtype, copy=True)


def _expected_pairs(
    expected_instance_ids: Mapping[str, pd.Index],
    linkage: TableLinkage,
    *,
    observed: _ObservedIdentifiers,
    context: ReaderErrorContext,
) -> tuple[pd.MultiIndex, tuple[str, ...]]:
    mapped_regions = tuple(region for region in linkage.regions if region in expected_instance_ids)
    region_arrays: list[np.ndarray] = []
    instance_arrays: list[np.ndarray] = []
    for region in mapped_regions:
        index = expected_instance_ids[region]
        family = _validate_expected_index(index, region=region, context=context)
        observed_family: _IdentifierFamily = "integer" if isinstance(observed, _IntegerIdentifiers) else "string"
        if family != observed_family:
            raise _reader_format_error(
                context=context,
                found=f"{family} target identifiers for region {region!r}",
                expected=f"{observed_family} identifiers matching the table instance column",
            )
        if isinstance(observed, _IntegerIdentifiers):
            instances = _integer_comparison_values(
                index,
                dtype=observed.dtype,
                region=region,
                context=context,
            )
        else:
            instances = index.to_numpy(dtype=object, copy=True)
        instance_arrays.append(instances)
        region_arrays.append(np.full(len(index), region, dtype=object))

    if instance_arrays:
        instances = np.concatenate(instance_arrays)
        regions = np.concatenate(region_arrays)
    else:
        instance_dtype = observed.dtype if isinstance(observed, _IntegerIdentifiers) else object
        instances = np.empty(0, dtype=instance_dtype)
        regions = np.empty(0, dtype=object)
    return pd.MultiIndex.from_arrays((regions, instances)), mapped_regions


def _validate_expected_membership(
    observed: _ValidatedIdentifiers,
    expected_instance_ids: Mapping[str, pd.Index],
    linkage: TableLinkage,
    *,
    context: ReaderErrorContext,
) -> None:
    expected_pairs, mapped_regions = _expected_pairs(
        expected_instance_ids,
        linkage,
        observed=observed.identifiers,
        context=context,
    )
    if not mapped_regions:
        return
    region_values = observed.pairs.get_level_values(0)
    known_observed_pairs = observed.pairs[region_values.isin(mapped_regions)]
    missing_from_table = expected_pairs.difference(known_observed_pairs, sort=False)
    unexpected_in_table = known_observed_pairs.difference(expected_pairs, sort=False)
    if len(missing_from_table) == 0 and len(unexpected_in_table) == 0:
        return
    missing_preview = missing_from_table[:_MISMATCH_PREVIEW_SIZE].tolist()
    unexpected_preview = unexpected_in_table[:_MISMATCH_PREVIEW_SIZE].tolist()
    raise _reader_format_error(
        context=context,
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
) -> _ValidatedIdentifiers:
    missing_columns = tuple(key for key in (linkage.region_key, linkage.instance_key) if key not in adata.obs)
    if missing_columns:
        raise _reader_format_error(
            context=context,
            found=f"missing observation columns {missing_columns!r}",
            expected=f"linkage columns {linkage.region_key!r} and {linkage.instance_key!r}",
        )
    obs = cast("pd.DataFrame", adata.obs)
    region_values = obs[linkage.region_key]
    instance_values = obs[linkage.instance_key]
    _validate_region_values(region_values, linkage, context=context)
    identifiers = _validate_instance_values(instance_values, context=context)
    observed_pairs = pd.MultiIndex.from_arrays((region_values, instance_values))
    if not observed_pairs.is_unique:
        duplicate_preview = observed_pairs[observed_pairs.duplicated(keep=False)][:_MISMATCH_PREVIEW_SIZE].tolist()
        raise _reader_format_error(
            context=context,
            found=f"duplicate (region, instance) pairs; preview {duplicate_preview!r}",
            expected="unique instance identifiers within each region",
        )
    return _ValidatedIdentifiers(observed_pairs, identifiers)


def parse_linked_table(
    adata: AnnData,
    linkage: TableLinkage,
    /,
    *,
    context: ReaderErrorContext,
    expression_layers: tuple[str, ...] = (),
    expected_instance_ids: Mapping[str, pd.Index] | None = None,
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
        Reader and artifact identity for external-data and parser failures.
    expression_layers
        Named layers whose documented meaning is expression or counts.
    expected_instance_ids
        Optional borrowed target indexes keyed by a subset of declared regions.
        Each supplied index must exactly match table instances in that region;
        index order is irrelevant.

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
        message = f"adata.uns[{TableModel.ATTRS_KEY!r}] must be absent before linkage."
        raise ValueError(message)
    if expected_instance_ids is not None:
        unexpected_regions = tuple(region for region in expected_instance_ids if region not in linkage.regions)
        if unexpected_regions:
            message = f"expected_instance_ids contains regions outside TableLinkage.regions: {unexpected_regions!r}."
            raise ValueError(message)

    observed = _validated_pairs(adata, linkage, context=context)
    if expected_instance_ids is not None:
        _validate_expected_membership(
            observed,
            expected_instance_ids,
            linkage,
            context=context,
        )

    normalize_owned_anndata(adata, expression_layers=expression_layers)
    region: str | list[str] = linkage.regions[0] if len(linkage.regions) == 1 else list(linkage.regions)
    try:
        return TableModel.parse(
            adata,
            region=region,
            region_key=linkage.region_key,
            instance_key=linkage.instance_key,
        )
    except Exception as error:
        add_reader_context(error, context=context)
        raise
