"""Xenium Explorer selection exports as micrometre polygons under the invalid-geometry policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import shapely
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity, get_transformation

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.xenium import _selection, xenium_explorer_selection

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

# Comment lines as written by Xenium Explorer 1.x-3.x single-selection exports.
SINGLE = "#Selection name: Selection 1\n#Area (µm^2): 2500.00\nX,Y\n10.5,20.25\n60.5,20.25\n60.5,70.25\n10.5,70.25\n"
# Selections-window and Explorer 4.0 exports name each row's selection.
MULTI = (
    "#Selection coordinates exported from Xenium Explorer\n#Created 2025-01-01\nSelection,X,Y\n"
    "Selection 1,0,0\nSelection 1,4,0\nSelection 1,4,4\n"
    "Cell aaaaficg-1,10,10\nCell aaaaficg-1,12,10\nCell aaaaficg-1,11,12\n"
)
BOWTIE = "Selection 2,20,20\nSelection 2,24,24\nSelection 2,24,20\nSelection 2,20,24\n"


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class TestXeniumExplorerSelection:
    """Selections keep file order, names and micrometre vertices exactly."""

    def test_single_selection_named_by_comment(self, tmp_path: Path) -> None:
        shapes = xenium_explorer_selection(_write(tmp_path / "coordinates.csv", SINGLE))

        ShapesModel.validate(shapes)
        assert shapes.index.name == "selection"
        assert shapes.index.dtype == "str"
        assert shapes.index.tolist() == ["Selection 1"]
        assert shapely.equals_exact(
            shapes.geometry.iloc[0], shapely.Polygon([(10.5, 20.25), (60.5, 20.25), (60.5, 70.25), (10.5, 70.25)]), 0
        )
        assert isinstance(get_transformation(shapes, to_coordinate_system="global"), Identity)

    def test_multiple_selections_and_clicked_cells(self, tmp_path: Path) -> None:
        shapes = xenium_explorer_selection(_write(tmp_path / "coordinates.csv", MULTI))

        assert shapes.index.tolist() == ["Selection 1", "Cell aaaaficg-1"]
        np.testing.assert_array_equal(
            shapely.get_coordinates(shapes.geometry.iloc[1]), [(10, 10), (12, 10), (11, 12), (10, 10)]
        )

    def test_invalid_geometry_policy(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "coordinates.csv", MULTI + BOWTIE)

        assert xenium_explorer_selection(path).index.tolist() == ["Selection 1", "Cell aaaaficg-1", "Selection 2"]
        assert xenium_explorer_selection(path, invalid_geometries="drop").index.tolist() == [
            "Selection 1",
            "Cell aaaaficg-1",
        ]
        with pytest.raises(ReaderFormatError, match=r"1 invalid geometries .*\['Selection 2'\]"):
            xenium_explorer_selection(path, invalid_geometries="raise")

    def test_no_valid_selection(self, tmp_path: Path) -> None:
        bowtie = _write(tmp_path / "bowtie.csv", MULTI.split("Selection 1,0,0", maxsplit=1)[0] + BOWTIE)
        empty = _write(tmp_path / "empty.csv", MULTI.split("Selection 1,0,0", maxsplit=1)[0])

        with pytest.raises(ReaderFormatError, match="no valid selection"):
            xenium_explorer_selection(bowtie, invalid_geometries="drop")
        with pytest.raises(ReaderFormatError, match="a header without records"):
            xenium_explorer_selection(empty)

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("X,Y\n1,2\n", "without two leading comment lines"),
            ("#a\n#b\nA,B\n1,2\n", r"header \['A', 'B'\]"),
            ("#Area: 1\n#b\nX,Y\n1,2\n", "without a selection name"),
            ("#Selection name:  \n#b\nX,Y\n1,2\n", "without a selection name"),
            ("#a\n#b\nSelection,X,Y\nS,x,1\n", "non-numeric coordinates"),
            ("#a\n#b\nSelection,X,Y\nS,1\n", "records of the wrong width"),
            ("#a\n#b\nSelection,X,Y\nS,inf,1\n", "nonfinite coordinates"),
            ("#a\n#b\nSelection,X,Y\n,1,1\n", "unnamed selections"),
            ('#a\n#b\nSelection,X,Y\n"S,1,1\n', "malformed CSV text"),
            ("#a\n#b\nSelection,X,Y\nA,0,0\nA,1,0\nB,1,1\nA,0,1\n", "a selection name in several runs"),
            ("#a\n#b\nSelection,X,Y\nA,0,0\nA,1,0\nA,0,0\n", "fewer than three distinct vertices"),
        ],
        ids=[
            "comments",
            "header",
            "unnamed-single",
            "blank-name",
            "text",
            "width",
            "nonfinite",
            "unnamed",
            "quote",
            "runs",
            "distinct",
        ],
    )
    def test_rejects_malformed_files(self, tmp_path: Path, text: str, match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            xenium_explorer_selection(_write(tmp_path / "coordinates.csv", text))

    def test_rejects_non_utf8_and_oversized_files(self, tmp_path: Path, mocker: MockerFixture) -> None:
        latin = tmp_path / "latin.csv"
        latin.write_bytes(b"#Selection name: \xe9\n#b\nX,Y\n0,0\n1,0\n0,1\n")
        with pytest.raises(ReaderFormatError, match="malformed CSV text"):
            xenium_explorer_selection(latin)

        mocker.patch.object(_selection, "MAX_SELECTION_BYTES", 16)
        with pytest.raises(ReaderFormatError, match="larger than 256 MiB"):
            xenium_explorer_selection(_write(tmp_path / "coordinates.csv", SINGLE))

    def test_rejects_invalid_options_and_paths(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "coordinates.csv", SINGLE)

        with pytest.raises(ValueError, match="invalid_geometries"):
            xenium_explorer_selection(path, invalid_geometries="repair")  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.
        with pytest.raises(FileNotFoundError):
            xenium_explorer_selection(tmp_path / "missing.csv")
        with pytest.raises(IsADirectoryError):
            xenium_explorer_selection(tmp_path)
