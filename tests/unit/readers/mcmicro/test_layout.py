"""Deterministic MCMICRO discovery, pairing, and element-name collision rules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.mcmicro._layout import TmaLayout, WsiLayout, discover_layout

if TYPE_CHECKING:
    from pathlib import Path


def _touch(root: Path, *relative: str) -> None:
    for name in relative:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def _wsi(root: Path, *extra: str) -> Path:
    (root / "qc").mkdir(parents=True)
    (root / "qc" / "params.yml").write_text("workflow:\n  tma: false\n", encoding="utf-8")
    _touch(root, "markers.csv", "registration/slide_1.ome.tif", *extra)
    return root


def _tma(root: Path, *extra: str) -> Path:
    (root / "qc").mkdir(parents=True)
    (root / "qc" / "params.yml").write_text("workflow:\n  tma: true\n", encoding="utf-8")
    _touch(
        root,
        "markers.csv",
        "registration/map.ome.tif",
        "dearray/10.ome.tif",
        "dearray/2.ome.tif",
        "dearray/masks/10_mask.tif",
        "dearray/masks/2_mask.tif",
        *extra,
    )
    return root


class TestWsiDiscovery:
    """Official and historical grammars resolve labels and their measurements."""

    def test_official_methods_compartments_and_order(self, tmp_path: Path) -> None:
        root = _wsi(
            tmp_path,
            "segmentation/unmicst-slide_1/nuclei.ome.tif",
            "segmentation/unmicst-slide_1/cell_mask_ring.ome.tif",
            "segmentation/unmicst-slide_1/puncta.csv",
            "segmentation/ilastik-slide_1/cell.ome.tif",
            "segmentation/ilastik-slide_1/.hidden.ome.tif",
            "quantification/slide_1--unmicst_cell_mask_ring.csv",
            "quantification/slide_1--ilastik_cell.csv",
            "quantification/notes.txt",
        )
        layout = discover_layout(root)
        assert isinstance(layout, WsiLayout)
        assert layout.mode == "wsi"
        assert layout.image_id == "slide_1"
        assert layout.image.name == "slide_1.ome.tif"
        assert [label.key for label in layout.segmentations] == [
            "slide_1_ilastik_cell",
            "slide_1_unmicst_cell_mask_ring",
            "slide_1_unmicst_nuclei",
        ]
        assert [(table.path.name, table.target.key) for table in layout.quantifications] == [
            ("slide_1--ilastik_cell.csv", "slide_1_ilastik_cell"),
            ("slide_1--unmicst_cell_mask_ring.csv", "slide_1_unmicst_cell_mask_ring"),
        ]

    def test_historical_directory_takes_method_from_mcquant(self, tmp_path: Path) -> None:
        root = _wsi(tmp_path, "segmentation/slide_1/cells.ome.tif", "quantification/slide_1--seg_x_cells.csv")
        layout = discover_layout(root)
        assert [(label.method, label.compartment) for label in layout.segmentations] == [("seg_x", "cells")]
        assert layout.quantifications[0].target is layout.segmentations[0]

    def test_outputs_are_optional(self, tmp_path: Path) -> None:
        layout = discover_layout(_wsi(tmp_path, "raw/cycle.czi", "illumination/cycle-ffp.tif"))
        assert layout.segmentations == ()
        assert layout.quantifications == ()

    @pytest.mark.parametrize("name", ["slide.ome.tiff", "slide.tif", "slide.btf"])
    def test_registered_image_id_strips_ome_suffix(self, tmp_path: Path, name: str) -> None:
        root = _wsi(tmp_path)
        (root / "registration" / "slide_1.ome.tif").rename(root / "registration" / name)
        layout = discover_layout(root)
        assert isinstance(layout, WsiLayout)
        assert layout.image_id == "slide"

    @pytest.mark.parametrize(
        ("extra", "match"),
        [
            (("registration/second.ome.tif",), "exactly one registered image"),
            (("segmentation/unmicst-other/cell.ome.tif",), "exactly one of"),
            (("segmentation/slide_1/cell.ome.tif",), "historical mask"),
            (
                (
                    "segmentation/slide_1/cell.ome.tif",
                    "quantification/slide_1--a_cell.csv",
                    "quantification/slide_1--b_cell.csv",
                ),
                "historical mask",
            ),
            (("segmentation/unmicst-slide_1/cell.ome.tif", "segmentation/unmicst-slide_1/cell.tif"), "duplicate"),
            (("quantification/other--unmicst_cell.csv",), "matching images"),
            (("segmentation/unmicst-slide_1/cell.ome.tif", "quantification/slide_1--unmicst_nuclei.csv"), "no label"),
            (
                (
                    "segmentation/a-slide_1/b_c.ome.tif",
                    "segmentation/a_b-slide_1/c.ome.tif",
                    "quantification/slide_1--a_b_c.csv",
                ),
                "colliding keys",
            ),
        ],
    )
    def test_rejects(self, tmp_path: Path, extra: tuple[str, ...], match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            discover_layout(_wsi(tmp_path, *extra))

    def test_empty_registration_fails(self, tmp_path: Path) -> None:
        root = _wsi(tmp_path)
        (root / "registration" / "slide_1.ome.tif").unlink()
        with pytest.raises(ReaderFormatError, match="0 TIFF images"):
            discover_layout(root)

    @pytest.mark.parametrize("missing", ["markers.csv", "qc/params.yml"])
    def test_required_metadata(self, tmp_path: Path, missing: str) -> None:
        root = _wsi(tmp_path)
        (root / missing).unlink()
        with pytest.raises(FileNotFoundError):
            discover_layout(root)

    def test_registration_directory_is_required(self, tmp_path: Path) -> None:
        root = _wsi(tmp_path)
        (root / "registration" / "slide_1.ome.tif").unlink()
        (root / "registration").rmdir()
        with pytest.raises(FileNotFoundError):
            discover_layout(root)


class TestTmaDiscovery:
    """Numbered cores pair with masks, labels, and tables in numeric order."""

    def test_numeric_core_order_and_keys(self, tmp_path: Path) -> None:
        root = _tma(
            tmp_path,
            "segmentation/unmicst-10/cell.ome.tif",
            "segmentation/unmicst-2/cell.ome.tif",
            "quantification/10--unmicst_cell.csv",
            "quantification/2--unmicst_cell.csv",
        )
        layout = discover_layout(root)
        assert isinstance(layout, TmaLayout)
        assert layout.mode == "tma"
        assert layout.tma_map.name == "map.ome.tif"
        assert [(core.number, core.image.name, core.mask.name) for core in layout.cores] == [
            (2, "2.ome.tif", "2_mask.tif"),
            (10, "10.ome.tif", "10_mask.tif"),
        ]
        assert [label.key for label in layout.segmentations] == ["core_2_unmicst_cell", "core_10_unmicst_cell"]
        assert [table.target.key for table in layout.quantifications] == ["core_2_unmicst_cell", "core_10_unmicst_cell"]

    @pytest.mark.parametrize(
        ("extra", "match"),
        [
            (("dearray/1.tif",), "cores"),
            (("dearray/03.ome.tif",), "cores"),
            (("dearray/masks/3_mask.tif",), "each with exactly one mask"),
            (("dearray/masks/notes.txt",), "masks"),
            (("segmentation/unmicst-map/cell.ome.tif",), "exactly one of"),
        ],
    )
    def test_rejects(self, tmp_path: Path, extra: tuple[str, ...], match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            discover_layout(_tma(tmp_path, *extra))

    def test_mask_directory_is_required(self, tmp_path: Path) -> None:
        root = _tma(tmp_path)
        for mask in (root / "dearray" / "masks").iterdir():
            mask.unlink()
        (root / "dearray" / "masks").rmdir()
        with pytest.raises(FileNotFoundError):
            discover_layout(root)
