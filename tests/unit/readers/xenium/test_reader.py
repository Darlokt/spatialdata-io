"""Public option validation, product selection and the shared pixel grid, before any I/O."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from spatialdata.models import Image2DModel, Labels2DModel
from xarray import DataArray, DataTree

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers.xenium import _reader, xenium
from spatialdata_io.readers.xenium._layout import Selection
from spatialdata_io.readers.xenium._reader import _check_grid, _feature_types, _validate_options

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# Public defaults in signature order.
DEFAULTS: dict[str, object] = {
    "include_table": True,
    "feature_types": None,
    "table_target": "cell_labels",
    "include_cell_labels": True,
    "include_nucleus_labels": True,
    "include_cell_boundaries": True,
    "cell_geometry": "polygons",
    "include_nucleus_boundaries": True,
    "invalid_geometries": "keep",
    "include_transcripts": True,
    "include_morphology_focus": True,
    "include_morphology_mip": False,
    "include_aligned_images": True,
    "include_analysis": False,
    "scale_factors": (),
    "chunks": (1, 1024, 1024),
}
NOTHING = {name: False for name, value in DEFAULTS.items() if isinstance(value, bool)}
CONTEXT = ReaderErrorContext("xenium", "pixel grid", "6.0.0")


def _select(**changes: object) -> Selection:
    options = DEFAULTS | changes
    return _validate_options(**options).selection  # pyrefly: ignore[bad-argument-type] - Untyped option combinations.


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
            ([1], TypeError),
        ],
        ids=["string", "number", "empty", "repeated", "blank", "non-text"],
    )
    def test_invalid_values(self, value: object, error: type[Exception]) -> None:
        with pytest.raises(error, match="feature_types"):
            _feature_types(value)


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
            ({"cell_geometry": "squares"}, "cell_geometry must be one of .*found 'squares'"),
            ({"table_target": "nucleus_labels"}, "table_target must be one of .*found 'nucleus_labels'"),
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

    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            ({"include_transcripts": 1}, "include_transcripts must be a boolean; found int"),
            ({"table_target": 1}, "table_target must be one of .*found int"),
            ({"cell_geometry": None}, "cell_geometry must be one of .*found NoneType"),
            ({"invalid_geometries": 1}, "invalid_geometries"),
        ],
        ids=["flag", "target", "geometry", "policy"],
    )
    def test_wrong_types_raise_type_errors(self, changes: dict[str, object], match: str) -> None:
        with pytest.raises(TypeError, match=match):
            _select(**changes)

    def test_options_are_validated_in_signature_order(self) -> None:
        # Every value is invalid; each check reports the first remaining option.
        invalid: dict[str, object] = {
            "include_table": "yes",
            "feature_types": 3,
            "table_target": 3,
            "include_cell_labels": "yes",
            "include_nucleus_labels": "yes",
            "include_cell_boundaries": "yes",
            "cell_geometry": 3,
            "include_nucleus_boundaries": "yes",
            "invalid_geometries": 3,
            "include_transcripts": "yes",
            "include_morphology_focus": "yes",
            "include_morphology_mip": "yes",
            "include_aligned_images": "yes",
            "include_analysis": "yes",
            "scale_factors": "2",
        }
        for name in list(invalid):
            with pytest.raises(TypeError, match=name):
                _select(**invalid)
            invalid.pop(name)


class TestCheckGrid:
    """Masks and morphology images share one calibrated pixel grid."""

    def test_equal_grids_including_pyramids(self) -> None:
        image = Image2DModel.parse(np.zeros((2, 8, 6), dtype=np.float32), dims=("c", "y", "x"), scale_factors=[2])
        labels = Labels2DModel.parse(np.zeros((8, 6), dtype=np.uint32), dims=("y", "x"))
        assert isinstance(image, DataTree)
        assert isinstance(labels, DataArray)

        _check_grid({"morphology_focus": image, "cell_labels": labels}, Path("bundle"), CONTEXT)

    def test_unequal_grids_raise(self) -> None:
        image = Image2DModel.parse(np.zeros((2, 8, 6), dtype=np.float32), dims=("c", "y", "x"))
        labels = Labels2DModel.parse(np.zeros((8, 5), dtype=np.uint32), dims=("y", "x"))

        with pytest.raises(ReaderFormatError, match="masks and morphology images on one pixel grid") as caught:
            _check_grid({"morphology_focus": image, "cell_labels": labels}, Path("bundle"), CONTEXT)

        assert caught.value.context == CONTEXT


class TestXenium:
    """Every public option is validated before discovery touches the path."""

    @pytest.mark.parametrize(
        ("options", "error"),
        [
            ({"feature_types": "Gene Expression"}, TypeError),
            ({"table_target": "nucleus_labels"}, ValueError),
            ({"include_cell_labels": False}, ValueError),
            ({"invalid_geometries": "repair"}, ValueError),
            ({"scale_factors": [1]}, ValueError),
            ({"chunks": (1, 2)}, ValueError),
            ({"include_table": "yes"}, TypeError),
        ],
        ids=["feature-types", "target", "unreturned-target", "invalid-geometries", "scale-factors", "chunks", "flag"],
    )
    def test_options_precede_io(
        self, tmp_path: Path, mocker: MockerFixture, options: dict[str, object], error: type[Exception]
    ) -> None:
        discover = mocker.spy(_reader, "discover_layout")

        with pytest.raises(error):
            xenium(tmp_path / "missing", **options)  # pyrefly: ignore[bad-argument-type] - Invalid runtime inputs.

        discover.assert_not_called()
