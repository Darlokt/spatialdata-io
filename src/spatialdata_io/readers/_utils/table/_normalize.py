"""Normalization of explicitly reader-owned in-memory AnnData objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from scipy import sparse

from spatialdata_io.readers._utils.table._arrays import as_csr_array

if TYPE_CHECKING:
    from anndata import AnnData

type _ExpressionData = np.ndarray[tuple[int, ...], np.dtype[np.generic]] | sparse.sparray | sparse.spmatrix


def _normalized_expression(data: object, *, location: str) -> sparse.csr_array:
    try:
        return as_csr_array(cast("_ExpressionData", data))
    except (TypeError, ValueError) as error:
        error.add_note(f"while normalizing {location}")
        raise


def _validate_expression_layers(adata: AnnData, expression_layers: tuple[str, ...]) -> None:
    seen: set[str] = set()
    duplicate_set: set[str] = set()
    duplicate_names: list[str] = []
    for name in expression_layers:
        if name in seen and name not in duplicate_set:
            duplicate_names.append(name)
            duplicate_set.add(name)
        seen.add(name)
    duplicates = tuple(duplicate_names)
    if duplicates:
        message = f"expression_layers must not contain duplicates; found {duplicates!r}."
        raise ValueError(message)

    missing = tuple(name for name in expression_layers if name not in adata.layers)
    if missing:
        message = f"Expression layers are missing from adata.layers: {missing!r}."
        raise KeyError(message)


def normalize_owned_anndata(
    adata: AnnData,
    /,
    *,
    expression_layers: tuple[str, ...] = (),
) -> AnnData:
    """Normalize expression storage in a reader-owned in-memory AnnData.

    ``adata`` must be a fully owned, in-memory object rather than a view or a
    backed table. Its ``X`` value, when present, and each explicitly named
    expression layer are replaced by canonical finite
    :class:`scipy.sparse.csr_array` values. Other aligned mappings, arbitrary
    layers, ``raw``, annotations, identifiers, and ordering remain unchanged.

    The function mutates and returns the same object. It validates all layer
    names before changing expression storage, then processes ``X`` and the
    requested layers sequentially to bound peak memory. Conversion is not
    transactional: if a later value is invalid, earlier values can already be
    normalized and the caller must discard the failed owned table.

    Parameters
    ----------
    adata
        A reader-created or freshly read, exclusively owned in-memory table.
    expression_layers
        Names of layers whose documented meaning is expression or counts.
        Unnamed layers and ``raw.X`` are deliberately not normalized.

    Returns
    -------
    anndata.AnnData
        The identical, mutated ``adata`` object.

    Raises
    ------
    ValueError
        If ``adata`` is a view, ``expression_layers`` contains duplicates, or
        selected expression values are malformed.
    TypeError
        If ``adata`` is backed or selected expression values use unsupported,
        lazy, or otherwise non-memory storage.
    KeyError
        If a requested expression layer does not exist.

    Notes
    -----
    This is an eager representation boundary. It never computes or loads lazy
    or backed storage and never copies the complete :class:`~anndata.AnnData`.
    """
    if adata.is_view:
        message = "normalize_owned_anndata() requires an owned AnnData, not a view."
        raise ValueError(message)
    if adata.isbacked:
        message = "normalize_owned_anndata() requires an in-memory AnnData, not a backed table."
        raise TypeError(message)

    _validate_expression_layers(adata, expression_layers)

    expression = adata.X
    if expression is not None:
        adata.X = _normalized_expression(expression, location="adata.X")
    for name in expression_layers:
        adata.layers[name] = _normalized_expression(
            adata.layers[name],
            location=f"adata.layers[{name!r}]",
        )
    return adata
