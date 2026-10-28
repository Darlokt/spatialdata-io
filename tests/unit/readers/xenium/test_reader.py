"""Public option validation, product selection and the shared pixel grid, before any I/O."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from spatialdata.models import Image2DModel, Labels2DModel
from xarray import DataArray, DataTree

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.xenium import _reader, xenium
from spatialdata_io.readers.xenium._layout import Selection
from spatialdata_io.readers.xenium._reader import _check_grid, _feature_types, _selection

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

DEFAULTS: dict[str, object] = {
    "include_table": True,
    "table_target": "cell_labels",
    "include_cell_labels": True,
    "include_nucleus_labels": True,
    "include_cell_boundaries": True,
    "cell_geometry": "polygons",
    "include_nucleus_boundaries": True,
    "include_transcripts": True,
    "include_morphology_focus": True,
    "include_morphology_mip": False,
    "include_aligned_images": True,
    "include_analysis": False,
}
NOTHING = {name: False for name, value in DEFAULTS.items() if isinstance(value, bool)}


def _select(**changes: object) -> Selection:
    return _selection(**(DEFAULTS | changes))  # pyrefly: ignore[bad-argument-type] - Untyped option combinations.


class TestFeatureTypes:
    """Feature types are None, 'all' or distinct nonempty names."""

    def test_valid_values(self) -> None:
        assert _feature_types(None) is None
        assert _feature_types("all") == "all"
        assert _feature_types(["Gene Expression", "Protein Expression"]) == ("Gene Expression", "Protein Expression")

    @pytest.mark.parametrize(
        ("value", "error"),
        [
            ("Gene Expression", TypeError),
            (3, TypeError),
            ([], ValueError),
            (["Gene Expression", "Gene Expression"], ValueError),
            ([""], ValueError),
            ([1], ValueError),
        ],
        ids=["string", "number", "empty", "repeated", "blank", "non-text"],
    )
    def test_invalid_values(self, value: object, error: type[Exception]) -> None:
        with pytest.raises(error, match="feature_types"):
            _feature_types(value)  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.


class TestSelection:
    """Inclusion flags resolve to returned products with one returned table target."""

    def test_defaults(self) -> None:
        assert _select() == Selection(
            table=True,
            table_target="cell_labels",
            cell_labels=True,
            nucleus_labels=True,
            cell_polygons=True,
            cell_circles=False,
            nucleus_polygons=True,
            transcripts=True,
            morphology_focus=True,
            morphology_mip=False,
            aligned_images=True,
            analysis=False,
        )

    @pytest.mark.parametrize(
        ("geometry", "polygons", "circles"),
        [("polygons", True, False), ("circles", False, True), ("both", True, True)],
    )
    def test_cell_geometry(self, geometry: str, *, polygons: bool, circles: bool) -> None:
        selection = _select(cell_geometry=geometry, table_target="cell_boundaries" if polygons else "cell_circles")

        assert (selection.cell_polygons, selection.cell_circles) == (polygons, circles)

    def test_target_is_ignored_without_a_table(self) -> None:
        selection = _select(include_table=False, include_cell_labels=False)

        assert selection.table_target is None
        assert _select(table_target=None).table_target is None

    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            ({"include_cell_labels": False}, "table_target='cell_labels' links the table to an element"),
            ({"table_target": "cell_circles"}, "table_target='cell_circles' links the table to an element"),
            ({"table_target": "cell_boundaries", "cell_geometry": "circles"}, "'cell_boundaries' links"),
            ({"table_target": "cell_boundaries", "include_cell_boundaries": False}, "'cell_boundaries' links"),
            ({"include_table": False, "include_analysis": True}, "requires include_table=True"),
            (NOTHING, "Select at least one product"),
            ({"cell_geometry": "squares"}, "cell_geometry must be one of"),
            ({"table_target": "nucleus_labels"}, "table_target must be one of"),
        ],
        ids=["labels", "circles", "geometry", "boundaries", "analysis", "nothing", "geometry-name", "target-name"],
    )
    def test_inconsistent_options(self, changes: dict[str, object], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _select(**changes)

    def test_aligned_images_alone_are_a_product(self) -> None:
        selection = _select(**(NOTHING | {"include_aligned_images": True}))

        assert selection.aligned_images
        assert selection.table_target is None

    def test_flags_must_be_booleans(self) -> None:
        with pytest.raises(TypeError, match="include_transcripts must be a boolean"):
            _select(include_transcripts=1)


class TestCheckGrid:
    """Masks and morphology images share one calibrated pixel grid."""

    def test_equal_grids_including_pyramids(self) -> None:
        image = Image2DModel.parse(np.zeros((2, 8, 6), dtype=np.float32), dims=("c", "y", "x"), scale_factors=[2])
        labels = Labels2DModel.parse(np.zeros((8, 6), dtype=np.uint32), dims=("y", "x"))
        assert isinstance(image, DataTree)
        assert isinstance(labels, DataArray)

        _check_grid({"morphology_focus": image, "cell_labels": labels}, Path("bundle"))

    def test_unequal_grids_raise(self) -> None:
        image = Image2DModel.parse(np.zeros((2, 8, 6), dtype=np.float32), dims=("c", "y", "x"))
        labels = Labels2DModel.parse(np.zeros((8, 5), dtype=np.uint32), dims=("y", "x"))

        with pytest.raises(ReaderFormatError, match="masks and morphology images on one pixel grid"):
            _check_grid({"morphology_focus": image, "cell_labels": labels}, Path("bundle"))


class TestXenium:
    """Every public option is validated before discovery touches the path."""

    @pytest.mark.parametrize(
        ("options", "error"),
        [
            ({"feature_types": "Gene Expression"}, TypeError),
            ({"invalid_geometries": "repair"}, ValueError),
            ({"scale_factors": [1]}, ValueError),
            ({"chunks": (1, 2)}, ValueError),
            ({"include_table": "yes"}, TypeError),
        ],
        ids=["feature-types", "invalid-geometries", "scale-factors", "chunks", "flag"],
    )
    def test_options_precede_io(
        self, tmp_path: Path, mocker: MockerFixture, options: dict[str, object], error: type[Exception]
    ) -> None:
        discover = mocker.spy(_reader, "discover_layout")

        with pytest.raises(error):
            xenium(tmp_path / "missing", **options)  # pyrefly: ignore[bad-argument-type] - Invalid runtime inputs.

        discover.assert_not_called()
