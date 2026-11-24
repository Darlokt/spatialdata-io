"""Xenium Explorer selection exports as micrometre polygons under the invalid-geometry policy."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.csv as pacsv
import pytest
import shapely
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity, Scale, get_transformation

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.xenium import xenium_explorer_selection

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
ANNOTATED = (
    "#Selection names: Region A, Region B\n#Areas (µm^2): 2,2\nSelection,X,Y,Class,Color\n"
    'Region A,0,0,"A, B",#010203\nRegion A,2,0,"A, B",#010203\nRegion A,0,2,"A, B",#010203\n'
    "Region B,0,3,Unclassified,#c2c2c2\nRegion B,2,3,Unclassified,#c2c2c2\nRegion B,0,5,Unclassified,#c2c2c2\n"
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _feature(
    name: str = "Region A",
    *,
    color: object = (194, 194, 194),
    coordinates: object = (((0, 0), (1, 0), (0, 1), (0, 0)),),
    classification: dict[str, object] | None = None,
    plane: dict[str, object] | None = None,
) -> dict[str, object]:
    properties: dict[str, object] = {"name": name, "objectType": "annotation", "color": color}
    if classification is not None:
        properties["classification"] = classification
    geometry: dict[str, object] = {"type": "Polygon", "coordinates": coordinates}
    if plane is not None:
        geometry["plane"] = plane
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": geometry,
    }


def _geojson(path: Path, *features: object) -> Path:
    return _write(path, json.dumps({"type": "FeatureCollection", "features": features}))


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

    def test_annotated_csv(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "annotated.csv", ANNOTATED)
        shapes = xenium_explorer_selection(path)

        assert shapes.index.tolist() == ["Region A", "Region B"]
        assert shapes["annotation_class"].tolist() == ["A, B", "Unclassified"]
        assert shapes["color"].tolist() == ["#010203", "#c2c2c2"]
        assert shapes.geometry.area.tolist() == [2.0, 2.0]

    def test_csv_annotation_must_be_constant(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "annotated.csv", ANNOTATED.replace("Region B,2,3,Unclassified", "Region B,2,3,Changed")
        )
        with pytest.raises(ReaderFormatError, match=r"selection 'Region B' with inconsistent Class"):
            xenium_explorer_selection(path)

    def test_geojson_names_colors_and_classes(self, tmp_path: Path) -> None:
        file = _geojson(
            tmp_path / "annotations.GeoJSON",
            _feature(
                "ROI 2", color=[1, 8, 255], classification={"name": "Tissue, A", "color": [10, 20, 30]}, plane={"z": 3}
            ),
            _feature("ROI 1"),
        )
        shapes = xenium_explorer_selection(file, image_to_global=Scale([0.5, 0.5], axes=("x", "y")))

        assert shapes.index.tolist() == ["ROI 2", "ROI 1"]
        assert shapes["color"].tolist() == ["#0108ff", "#c2c2c2"]
        assert shapes["annotation_class"].iloc[0] == "Tissue, A"
        assert shapes["annotation_class_color"].iloc[0] == "#0a141e"
        assert shapes["annotation_class"].isna().tolist() == [False, True]
        assert shapes["annotation_class_color"].isna().tolist() == [False, True]
        assert shapes["z_plane"].iloc[0] == 3
        assert shapes["z_plane"].isna().tolist() == [False, True]

    def test_geojson_rejects_duplicate_selection_names(self, tmp_path: Path) -> None:
        path = _geojson(tmp_path / "duplicate.geojson", _feature("ROI"), _feature("ROI", color=[1, 2, 3]))
        with pytest.raises(ReaderFormatError, match="duplicate selection name 'ROI'"):
            xenium_explorer_selection(path, image_to_global=Identity())

    def test_geojson_drop_keeps_valid_shapes(self, tmp_path: Path) -> None:
        path = _geojson(tmp_path / "valid.geojson", _feature("ROI"))
        result = xenium_explorer_selection(path, image_to_global=Identity(), invalid_geometries="drop")
        assert result.index.tolist() == ["ROI"]

    def test_geojson_invalid_geometry_policy_preserves_annotations(self, tmp_path: Path) -> None:
        path = _geojson(
            tmp_path / "annotations.geojson",
            _feature("Bowtie", coordinates=[[(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)]]),
            _feature("Valid", color=[0, 1, 2]),
        )
        kept = xenium_explorer_selection(path, image_to_global=Identity())
        assert kept.geometry.is_valid.tolist() == [False, True]
        dropped = xenium_explorer_selection(path, image_to_global=Identity(), invalid_geometries="drop")
        assert dropped.index.tolist() == ["Valid"]
        assert dropped["color"].tolist() == ["#000102"]
        with pytest.raises(ReaderFormatError, match=r"1 invalid geometries .*Bowtie"):
            xenium_explorer_selection(path, image_to_global=Identity(), invalid_geometries="raise")

    @pytest.mark.parametrize(
        ("feature", "match"),
        [
            ([], "feature 0 without an object"),
            ({}, "feature 0 geometry without an object"),
            ({"geometry": {}}, "feature 0 properties without an object"),
            (_feature(""), "malformed feature 0"),
            (_feature("\ud800"), "an invalid Unicode name"),
            (_feature(color=[0, 0]), "invalid feature 0 RGB color"),
            (_feature(color=[0, False, 0]), "invalid feature 0 RGB color"),
            (_feature(color=[0, 0, 256]), "invalid feature 0 RGB color"),
            (_feature(classification={"name": ""}), "without a class name"),
            (_feature(classification={"name": "Class"}), "invalid feature 0 class RGB color"),
            (_feature(coordinates=[]), "empty or malformed coordinates"),
            (_feature(coordinates=[[(0, 0), (1, 0), (0, 1)]]), "unclosed, short or malformed ring"),
            (_feature(coordinates=[[(0, 0), (1, 0), (0, "1"), (0, 0)]]), "nonnumeric coordinate"),
            (_feature(coordinates=[[(0, 0), (1, 0), (0, 1), (1, 1)]]), "unclosed, short or malformed ring"),
            (_feature(coordinates=[[[0, 0]] * 4] * 2), "with interior rings"),
            (_feature(coordinates=[[(0, 0), (1, 0), (0, 10**400), (0, 0)]]), "outside the finite float range"),
            (_feature(plane={"z": -1}), "unsupported geometry plane"),
            (_feature(plane={"z": 0, "t": 1}), "unsupported geometry plane"),
        ],
    )
    def test_geojson_rejects_malformed_features(self, tmp_path: Path, feature: object, match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            xenium_explorer_selection(_geojson(tmp_path / "malformed.geojson", feature), image_to_global=Identity())

    def test_geojson_requires_features_and_registration(self, tmp_path: Path) -> None:
        path = _geojson(tmp_path / "empty.geojson")
        with pytest.raises(ValueError, match="image_to_global is required"):
            xenium_explorer_selection(path)
        with pytest.raises(ValueError, match="finite, invertible XY affine"):
            xenium_explorer_selection(path, image_to_global=Scale([0, 1], axes=("x", "y")))
        with pytest.raises(TypeError, match="SpatialData transformation"):
            xenium_explorer_selection(path, image_to_global="global")  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.
        with pytest.raises(ReaderFormatError, match="no nonempty FeatureCollection"):
            xenium_explorer_selection(path, image_to_global=Identity())

    def test_csv_rejects_reregistering_micrometre_coordinates(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="already in micrometres"):
            xenium_explorer_selection(_write(tmp_path / "selections.csv", MULTI), image_to_global=Identity())

    def test_invalid_geometry_policy(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "coordinates.csv", MULTI + BOWTIE)

        assert xenium_explorer_selection(path).index.tolist() == ["Selection 1", "Cell aaaaficg-1", "Selection 2"]
        assert xenium_explorer_selection(path, invalid_geometries="drop").index.tolist() == [
            "Selection 1",
            "Cell aaaaficg-1",
        ]
        with pytest.raises(ReaderFormatError, match=r"1 invalid geometries .*\['Selection 2'\]"):
            xenium_explorer_selection(path, invalid_geometries="raise")

    def test_keep_skips_topology_and_preserves_invalid_ring(self, tmp_path: Path, mocker: MockerFixture) -> None:
        topology = mocker.patch(
            "spatialdata_io.readers.xenium._selection.invalid_geometry_mask", side_effect=AssertionError
        )
        path = _write(tmp_path / "coordinates.csv", MULTI + BOWTIE)

        result = xenium_explorer_selection(path)

        assert result.index.tolist() == ["Selection 1", "Cell aaaaficg-1", "Selection 2"]
        assert not result.geometry.loc["Selection 2"].is_valid
        topology.assert_not_called()

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
            ("#a\n#b\nSelection,X,Y\nS,x,1\n", "CSV conversion error to double: invalid value 'x'"),
            ("#a\n#b\nSelection,X,Y\nS,1\n", "Expected 3 columns, got 2"),
            ("#a\n#b\nSelection,X,Y\nS,inf,1\n", "nonfinite coordinates"),
            ("#a\n#b\nSelection,X,Y\n,1,1\n", "unnamed selections"),
            ('#a\n#b\nSelection,X,Y\n"S,1,1\n', "malformed CSV"),
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

    def test_rejects_non_utf8(self, tmp_path: Path) -> None:
        latin = tmp_path / "latin.csv"
        latin.write_bytes(b"#Selection name: \xe9\n#b\nX,Y\n0,0\n1,0\n0,1\n")
        with pytest.raises(ReaderFormatError, match=r"malformed CSV .*'utf-8' codec"):
            xenium_explorer_selection(latin)

    @pytest.mark.parametrize("malformed", [False, True])
    def test_csv_is_parsed_directly_from_owned_stream(
        self, tmp_path: Path, mocker: MockerFixture, *, malformed: bool
    ) -> None:
        path = _write(tmp_path / "coordinates.csv", SINGLE)
        text = SINGLE + '"unfinished,1\n' if malformed else SINGLE
        handle = io.BytesIO(text.encode())
        parser = mocker.spy(pacsv, "read_csv")
        mocker.patch.object(type(path), "open", return_value=handle)
        if malformed:
            with pytest.raises(ReaderFormatError, match="malformed CSV"):
                xenium_explorer_selection(path)
        else:
            assert xenium_explorer_selection(path).index.tolist() == ["Selection 1"]
        assert parser.call_args_list[-1].args[0] is handle
        assert handle.closed

    @pytest.mark.parametrize("value", [" 10.5 ", "1.05e1"])
    def test_coordinates_follow_arrow_grammar(self, tmp_path: Path, value: str) -> None:
        shapes = xenium_explorer_selection(
            _write(tmp_path / "coordinates.csv", SINGLE.replace("\n10.5,", f"\n{value},"))
        )
        assert shapely.get_coordinates(shapes.geometry.iloc[0])[0, 0] == 10.5

    def test_rejects_invalid_options_and_paths(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "coordinates.csv", SINGLE)

        with pytest.raises(ValueError, match="invalid_geometries"):
            xenium_explorer_selection(path, invalid_geometries="repair")  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.
        with pytest.raises(FileNotFoundError):
            xenium_explorer_selection(tmp_path / "missing.csv")
        with pytest.raises(IsADirectoryError):
            xenium_explorer_selection(tmp_path)
