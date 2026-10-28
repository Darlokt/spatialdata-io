"""Read cells.zarr.zip identities, label mappings and summaries, closing the store always."""

from __future__ import annotations

import zipfile
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.xenium import _cells
from spatialdata_io.readers.xenium._cells import read_cells_store
from tests.integration.readers.xenium import _fixtures as fx

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pytest_mock import MockerFixture

type Mutation = Callable[[zarr.Group], object]

CONTEXT = ReaderErrorContext("xenium", "cells.zarr.zip")


def _assign(key: str, index: int | tuple[int, int], value: float | list[int]) -> Mutation:
    """Return a mutation that assigns ``value`` at ``index`` of the stored array ``key``."""

    def mutate(group: zarr.Group) -> None:
        array = group[key]
        assert isinstance(array, zarr.Array)
        array[index] = value

    return mutate


# Each mutation breaks one store invariant of an otherwise valid bundle.
STORE_DEFECTS: dict[str, tuple[fx.Profile, Mutation, str]] = {
    "family-structure": (
        fx.XOA_1_9,
        lambda g: g.create_group("polygon_sets"),
        "seg_mask_value/polygon_sets",
    ),
    "set-count": (
        fx.XOA_2_0,
        lambda g: g.attrs.update({"polygon_set_names": ["cell"]}),
        "one stored polygon set per name",
    ),
    "set-names": (
        fx.XOA_2_0,
        lambda g: g.attrs.update({"polygon_set_names": ["nucleus", "nucleus"]}),
        "including 'cell'",
    ),
    "methods": (fx.XOA_2_0, lambda g: g.attrs.update({"segmentation_methods": ["only"]}), "segmentation_methods"),
    "labels": (fx.XOA_1_9, _assign("seg_mask_value", 1, 1), "unique positive uint32 seg_mask_value"),
    "integer-ids": (fx.XOA_1_0, _assign("cell_id", 0, 0), "unique positive uint32 integer cell IDs"),
    "hex-ids": (fx.XOA_1_9, _assign("cell_id", 1, [1, 1]), "repeated decoded cell IDs"),
    "cell-index": (fx.XOA_2_0, _assign("polygon_sets/1/cell_index", 0, 1), "one cell polygon per cell"),
    "nucleus-owner": (fx.XOA_2_0, _assign("polygon_sets/0/cell_index", 0, 9), "nucleus cell_index values below N"),
    "summary-columns": (
        fx.XOA_2_0,
        lambda g: g["cell_summary"].attrs.update({"column_names": ["a"]}),
        "one named column per value",
    ),
    "nucleus-count": (fx.XOA_2_0, _assign("cell_summary", (0, 7), 1.5), "integral nucleus counts"),
    "centroids": (fx.XOA_2_0, _assign("cell_summary", (0, 3), np.inf), "finite nucleus centroids"),
}


def _rewrite(path: Path, mutate: Mutation) -> None:
    """Unpack the stored zip, mutate its zarr v2 tree and pack it again."""
    tree = path.with_suffix(".tree")
    with zipfile.ZipFile(path) as archive:
        archive.extractall(tree)
    mutate(zarr.open_group(store=LocalStore(tree), mode="r+", zarr_format=2))
    fx.zip_tree(tree, path)


def _read(bundle: fx.XeniumFixture, pixel_size: float = fx.PIXEL_SIZE) -> _cells.CellStore:
    return read_cells_store(
        ArtifactFile(bundle.root / "cells.zarr.zip"), bundle.profile.family, pixel_size=pixel_size, context=CONTEXT
    )


class TestReadCellsStore:
    """The cells store yields IDs, mask labels, nuclei and masks consistent per family."""

    def test_integer_family(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_1_0)

        store = _read(bundle)

        assert store.cell_ids.tolist() == [1, 2, 3, 4]
        assert store.cell_ids.dtype == np.int64
        np.testing.assert_array_equal(store.cell_labels, [1, 2, 3, 4])
        assert store.cell_labels.dtype == np.uint32
        assert store.nuclei is not None
        np.testing.assert_array_equal(store.nuclei.labels, [1, 2, 3, 4])
        np.testing.assert_array_equal(store.nuclei.cells, [0, 1, 2, 3])
        assert store.cell_methods is None
        assert store.nucleus_counts is None
        assert set(store.masks) == {"nucleus", "cell"}
        assert store.masks["cell"].shape == fx.SHAPE
        assert store.masks["cell"].chunks == fx.MASK_CHUNKS
        assert store.masks["cell"].key == "masks/1"

    def test_hex_family_decodes_and_maps_seg_mask_value(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_1_9)

        store = _read(bundle)

        assert store.cell_ids.tolist() == list(fx.HEX_IDS)
        np.testing.assert_array_equal(store.cell_labels, [1, 2, 3, 4])
        assert store.nuclei is not None
        # Strict 1:1 nuclei carry their cell's label.
        np.testing.assert_array_equal(store.nuclei.labels, store.cell_labels)

    def test_polygon_sets_map_multiple_and_missing_nuclei(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_2_0)

        store = _read(bundle)

        np.testing.assert_array_equal(store.cell_labels, [1, 2, 3, 4])
        assert store.nuclei is not None
        np.testing.assert_array_equal(store.nuclei.labels, [1, 2, 3, 4])
        np.testing.assert_array_equal(store.nuclei.cells, [0, 1, 2, 2])
        np.testing.assert_array_equal(store.nucleus_counts, [1, 1, 2, 0])
        assert store.cell_methods is not None
        assert store.cell_methods.tolist() == [fx.SEGMENTATION_METHODS[m] for m in fx.CELL_METHODS]
        # Cell 3 has no nucleus: its nucleus centroid is NaN.
        assert np.isnan(store.nucleus_centroids[3]).all()
        # Corner-frame centre of mass of both nuclei of cell 2, from their pixels.
        rows, columns = np.nonzero(np.isin(bundle.nucleus_mask, [3, 4]))
        expected = [(columns.mean() + 0.5) * fx.PIXEL_SIZE, (rows.mean() + 0.5) * fx.PIXEL_SIZE]
        np.testing.assert_allclose(store.nucleus_centroids[2], expected, rtol=1e-15)

    def test_transcript_assignment_import_has_no_nuclei_or_labels(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(import_kind="transcripts"))

        store = _read(bundle)

        assert store.nuclei is None
        assert store.masks == {}

    def test_cells_only_import_keeps_an_empty_nucleus_set(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(import_kind="cells_only"))

        store = _read(bundle)

        assert store.nuclei is not None
        assert store.nuclei.labels.size == 0
        assert set(store.masks) == {"nucleus", "cell"}

    def test_polygon_set_order_follows_names(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)

        def swap(group: zarr.Group) -> None:
            group.attrs["polygon_set_names"] = ["cell", "nucleus"]
            tree = bundle.root / "cells.zarr.tree" / "polygon_sets"
            (tree / "0").rename(tree / "tmp")
            (tree / "1").rename(tree / "0")
            (tree / "tmp").rename(tree / "1")

        _rewrite(bundle.root / "cells.zarr.zip", swap)
        store = _read(bundle)

        assert store.nuclei is not None
        np.testing.assert_array_equal(store.nuclei.cells, [0, 1, 2, 2])

    def test_mask_transform_cross_checks_pixel_size(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)

        with pytest.raises(ReaderFormatError, match="diag\\(1/pixel_size"):
            _read(bundle, pixel_size=0.2)

    @pytest.mark.parametrize(("profile", "mutate", "match"), list(STORE_DEFECTS.values()), ids=list(STORE_DEFECTS))
    def test_rejects_inconsistent_stores(
        self, tmp_path: Path, profile: fx.Profile, mutate: Mutation, match: str
    ) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, profile)
        _rewrite(bundle.root / "cells.zarr.zip", mutate)

        with pytest.raises(ReaderFormatError, match=match):
            _read(bundle)

    def test_rejects_unequal_mask_shapes(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_2_0)

        def shrink(group: zarr.Group) -> None:
            del group["masks/0"]
            group.require_group("masks").create_array("0", data=np.zeros((8, 8), dtype=np.uint32), compressors=None)

        _rewrite(bundle.root / "cells.zarr.zip", shrink)
        with pytest.raises(ReaderFormatError, match="equal nucleus and cell masks"):
            _read(bundle)

    def test_rejects_masks_other_than_uint32(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_2_0)

        # 64-bit labels would lose exact IDs in generated label pyramids.
        def widen(group: zarr.Group) -> None:
            source = group["masks/1"]
            assert isinstance(source, zarr.Array)
            mask = np.asarray(source[...], dtype=np.uint64)
            del group["masks/1"]
            group.require_group("masks").create_array("1", data=mask, compressors=None)

        _rewrite(bundle.root / "cells.zarr.zip", widen)
        with pytest.raises(ReaderFormatError, match="2-D uint32 label mask"):
            _read(bundle)

    @pytest.mark.parametrize("fail", [False, True])
    def test_store_closes_on_success_and_failure(self, tmp_path: Path, mocker: MockerFixture, *, fail: bool) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_2_0)
        close = mocker.spy(_cells.ZipStore, "close")
        if fail:
            mocker.patch.object(_cells, "_summary", side_effect=RuntimeError("injected"))
            with pytest.raises(RuntimeError, match="injected"):
                _read(bundle)
        else:
            _read(bundle)

        assert close.call_count >= 1
