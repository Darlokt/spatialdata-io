"""Materialize ``cells.zarr.zip`` identities, label mappings and summary columns.

One ``ZipStore`` is opened, its small arrays are copied and validated, and it is
closed on success and failure. Masks are described, not read.

Store layout by family (cells ``N``, polygons ``P_k`` in set ``k``)::

    cells.zarr.zip/
    ├── cell_id              # (N,) uint32 integer IDs (XOA 1.0-1.2) or (N, 2) uint32 pairs
    ├── seg_mask_value       # (N,) uint32 label of each cell (XOA 1.3-1.9 only)
    ├── cell_summary         # (N, 7|8) float64 with attrs column_names
    ├── polygon_sets/{k}/    # XOA 2.0+: cell_index (P_k,) owning cell row, method (P_k,)
    └── masks/               # 0: nucleus, 1: cell labels (uint32); homogeneous_transform

A label is the integer ID itself (integer family), ``seg_mask_value``
(hexadecimal family; a nucleus carries its cell's label), or the polygon index
plus one within its named polygon set (polygon-set family, where a cell may have
any number of nuclei).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import zarr
from zarr.storage import ZipStore

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers.xenium._cell_ids import decode_cell_ids
from spatialdata_io.readers.xenium._labels import MaskSource

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray
    from zarr.core.metadata import ArrayMetadata

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.xenium._version import Family

type MaskRole = Literal["nucleus", "cell"]

_MASK_KEYS: dict[MaskRole, str] = {"nucleus": "masks/0", "cell": "masks/1"}
_SET_NAMES: tuple[MaskRole, ...] = ("nucleus", "cell")
_MASK_RANK = 2
_TRANSFORM_SHAPE = (4, 4)
# masks/homogeneous_transform stores 1 / pixel_size in float32; observed builds
# round it to either neighbouring float32 value.
_TRANSFORM_RTOL = 4 * float(np.finfo(np.float32).eps)


@dataclass(frozen=True, slots=True)
class Nuclei:
    """Nucleus mask labels and the row of the cell owning each nucleus."""

    labels: NDArray[np.uint32]
    cells: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class CellStore:
    """Owned, validated arrays of one closed ``cells.zarr.zip``.

    ``nuclei`` is ``None`` when the bundle has no nucleus polygon set (Xenium
    Ranger transcript-assignment imports). ``masks`` holds only usable masks;
    ``cell_methods`` and ``nucleus_counts`` exist only where the store has them.
    """

    cell_ids: pd.Index
    cell_labels: NDArray[np.uint32]
    nuclei: Nuclei | None
    cell_methods: NDArray[np.str_] | None
    nucleus_centroids: NDArray[np.float64]
    nucleus_counts: NDArray[np.int64] | None
    masks: Mapping[MaskRole, MaskSource]


@dataclass(frozen=True, slots=True)
class _Store:
    """Typed access to one open zarr group with contextual failures."""

    group: zarr.Group
    path: ArtifactFile
    context: ReaderErrorContext

    def error(self, *, found: str, expected: str) -> ReaderFormatError:
        return ReaderFormatError(context=self.context, path=self.path, found=found, expected=expected)

    def array(self, key: str) -> zarr.Array[ArrayMetadata]:
        node = self.group.get(key)
        if not isinstance(node, zarr.Array):
            raise self.error(found=f"no array {key!r}", expected=f"a {key!r} array")
        return node

    def read(self, key: str, *, ndim: int, kind: str) -> np.ndarray:
        """Read a small array of the given rank whose dtype kind is in ``kind``."""
        array = self.array(key)
        if array.ndim != ndim or np.dtype(array.dtype).kind not in kind:
            raise self.error(
                found=f"{key!r} with shape {array.shape} and dtype {array.dtype}", expected=f"a {ndim}-D {key!r} array"
            )
        return np.asarray(array[...])


def _labels(store: _Store, values: np.ndarray, name: str) -> NDArray[np.uint32]:
    """Require unique positive uint32 mask labels."""
    if len(np.unique(values)) != len(values) or (values <= 0).any() or values.max(initial=0) > np.iinfo(np.uint32).max:
        raise store.error(found=f"duplicate, zero or out-of-range {name}", expected=f"unique positive uint32 {name}")
    return values.astype(np.uint32)


def _hex_ids(store: _Store) -> pd.Index:
    pairs = store.read("cell_id", ndim=2, kind="u")
    try:
        ids = pd.Index(decode_cell_ids(pairs), dtype="str")
    except ValueError as error:
        raise store.error(
            found=f"cell_id of shape {pairs.shape} and dtype {pairs.dtype}",
            expected="(N, 2) uint32 prefix/suffix pairs",
        ) from error
    if not ids.is_unique:
        raise store.error(found="repeated decoded cell IDs", expected="unique cell IDs")
    return ids


def _polygon_set_roles(store: _Store) -> dict[MaskRole, int]:
    """Map each named polygon set to its index; names must match the stored sets."""
    names = store.group.attrs.get("polygon_set_names")
    sets = store.group.get("polygon_sets")
    stored = sorted(int(key) for key in sets.group_keys()) if isinstance(sets, zarr.Group) else []
    if (
        not isinstance(names, list)
        or "cell" not in names
        or len(set(names)) != len(names)
        or not set(names) <= set(_SET_NAMES)
    ):
        raise store.error(
            found=f"polygon_set_names {names!r}", expected="distinct names from ('nucleus', 'cell') including 'cell'"
        )
    if stored != list(range(len(names))):
        raise store.error(found=f"polygon sets {stored} named {names!r}", expected="one stored polygon set per name")
    return {role: names.index(role) for role in _SET_NAMES if role in names}


def _polygon_sets(store: _Store, n_cells: int) -> tuple[NDArray[np.uint32], Nuclei | None, NDArray[np.str_]]:
    """Return cell labels, nuclei and cell segmentation methods of the polygon-set family."""
    roles = _polygon_set_roles(store)
    cell_set = f"polygon_sets/{roles['cell']}"
    cell_index = store.read(f"{cell_set}/cell_index", ndim=1, kind="ui")
    if not np.array_equal(np.sort(cell_index), np.arange(n_cells)):
        raise store.error(found="cell polygons not one per cell", expected="one cell polygon per cell")
    cell_index = cell_index.astype(np.int64)
    # Label p + 1 belongs to cell cell_index[p].
    cell_labels = np.empty(n_cells, dtype=np.uint32)
    cell_labels[cell_index] = np.arange(1, n_cells + 1, dtype=np.uint32)

    methods = store.group.attrs.get("segmentation_methods")
    method_index = store.read(f"{cell_set}/method", ndim=1, kind="ui")
    if (
        not isinstance(methods, list)
        or len(method_index) != n_cells
        or (method_index < 0).any()
        or (method_index >= len(methods)).any()
    ):
        raise store.error(
            found="cell methods outside segmentation_methods",
            expected="indexes into the segmentation_methods attribute",
        )
    cell_methods = np.empty(n_cells, dtype=object)
    cell_methods[cell_index] = np.asarray(methods, dtype=object)[method_index.astype(np.int64)]

    nuclei = None
    if "nucleus" in roles:
        owners = store.read(f"polygon_sets/{roles['nucleus']}/cell_index", ndim=1, kind="ui")
        if (owners < 0).any() or (owners >= n_cells).any():
            raise store.error(found="nuclei owned by unknown cells", expected="nucleus cell_index values below N")
        nuclei = Nuclei(np.arange(1, len(owners) + 1, dtype=np.uint32), owners.astype(np.int64))
    return cell_labels, nuclei, cell_methods.astype(np.str_)


def _summary(store: _Store, n_cells: int) -> tuple[NDArray[np.float64], NDArray[np.int64] | None]:
    """Return nucleus centroids (NaN without a nucleus) and, when stored, nucleus counts."""
    summary = store.array("cell_summary")
    columns = summary.attrs.get("column_names")
    if not isinstance(columns, list) or summary.shape != (n_cells, len(columns)):
        raise store.error(
            found=f"cell_summary of shape {summary.shape} with columns {columns!r}",
            expected="one named column per value",
        )

    def column(name: str) -> NDArray[np.float64]:
        if name not in columns:
            raise store.error(found=f"cell_summary columns {columns!r}", expected=f"a {name!r} column")
        return np.asarray(summary[:, columns.index(name)], dtype=np.float64)

    centroids = np.column_stack([column("nucleus_centroid_x"), column("nucleus_centroid_y")])
    if np.isinf(centroids).any():
        raise store.error(found="infinite nucleus centroids", expected="finite nucleus centroids or NaN")
    if "nucleus_count" not in columns:
        return centroids, None
    counts = column("nucleus_count")
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.round(counts)).any():
        raise store.error(found="non-integral nucleus_count values", expected="nonnegative integral nucleus counts")
    # int64.max rounds up to 2**63 in float64, so the upper bound is exclusive.
    if (counts >= 2**63).any():
        raise store.error(
            found="nucleus_count outside int64 range", expected="nonnegative integral nucleus counts below 2**63"
        )
    return centroids, counts.astype(np.int64)


def _masks(store: _Store, pixel_size: float) -> dict[MaskRole, MaskSource]:
    """Describe the present masks and cross-check their transform against ``pixel_size``."""
    masks: dict[MaskRole, MaskSource] = {}
    for role, key in _MASK_KEYS.items():
        node = store.group.get(key)
        if node is None:
            continue
        # Every XOA version writes uint32 masks; this also keeps label pyramids exact.
        if not isinstance(node, zarr.Array) or node.ndim != _MASK_RANK or np.dtype(node.dtype) != np.uint32:
            raise store.error(found=f"{key} {node!r}", expected="a 2-D uint32 label mask")
        masks[role] = MaskSource(store.path, key, (node.shape[0], node.shape[1]), (node.chunks[0], node.chunks[1]))
    if not masks:
        return masks
    if len({mask.shape for mask in masks.values()}) != 1:
        raise store.error(
            found=f"mask shapes {[m.shape for m in masks.values()]}", expected="equal nucleus and cell masks"
        )

    transform = store.read("masks/homogeneous_transform", ndim=2, kind="f").astype(np.float64)
    expected = np.diag([1 / pixel_size, 1 / pixel_size, 1.0, 1.0])
    if transform.shape != _TRANSFORM_SHAPE or not np.allclose(transform, expected, rtol=_TRANSFORM_RTOL, atol=0):
        raise store.error(
            found=f"masks/homogeneous_transform {transform.tolist()}",
            expected=f"diag(1/pixel_size, 1/pixel_size, 1, 1) for pixel_size {pixel_size}",
        )
    return masks


def read_cells_store(
    path: ArtifactFile, family: Family, *, pixel_size: float, context: ReaderErrorContext
) -> CellStore:
    """Read and validate one ``cells.zarr.zip``, closing it before returning.

    Raises
    ------
    ReaderFormatError
        If the store structure contradicts the family, identities or labels repeat,
        polygon sets are unnamed or inconsistent, summary columns are missing or
        malformed, or the mask transform disagrees with ``pixel_size``.
    """
    with ZipStore(path, read_only=True) as zip_store:
        store = _Store(zarr.open_group(store=zip_store, mode="r", zarr_format=2), path, context)
        markers = ("seg_mask_value" in store.group, "polygon_sets" in store.group)
        if markers != (family == "hex_ids", family == "polygon_sets"):
            raise store.error(
                found=f"seg_mask_value/polygon_sets presence {markers}",
                expected=f"the {family} layout of experiment.xenium",
            )

        cell_methods = None
        nuclei: Nuclei | None = None
        if family == "integer_ids":
            raw = store.read("cell_id", ndim=1, kind="ui")
            cell_ids = pd.Index(raw.astype(np.int64))
            cell_labels = _labels(store, raw, "integer cell IDs")
        else:
            cell_ids = _hex_ids(store)
            if family == "hex_ids":
                values = store.read("seg_mask_value", ndim=1, kind="ui")
                if len(values) != len(cell_ids):
                    raise store.error(found=f"{len(values)} seg_mask_value entries", expected=f"{len(cell_ids)} labels")
                cell_labels = _labels(store, values, "seg_mask_value labels")
            else:
                cell_labels, nuclei, cell_methods = _polygon_sets(store, len(cell_ids))

        masks = _masks(store, pixel_size)
        if family != "polygon_sets":
            # Strict 1:1 nuclei labelled like their cell.
            nuclei = Nuclei(cell_labels, np.arange(len(cell_ids), dtype=np.int64))
        elif nuclei is None:
            # Transcript-assignment imports have one polygon set and masks that 10x
            # documents as absent or empty; they are never offered as labels.
            masks = dict[MaskRole, MaskSource]()
        nucleus_centroids, nucleus_counts = _summary(store, len(cell_ids))

    return CellStore(cell_ids, cell_labels, nuclei, cell_methods, nucleus_centroids, nucleus_counts, masks)
