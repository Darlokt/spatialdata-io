"""Public validation and final scientific composition of Visium HD products."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers.visium_hd import _reader, visium_hd
from spatialdata_io.readers.visium_hd._reader import _bin_sizes

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestBinSizes:
    """Selection is an owned deterministic sequence, without widening."""

    def test_all_empty_and_owned_order(self) -> None:
        values = [16, 2, 8]
        assert _bin_sizes("all") == "all"
        assert _bin_sizes(()) == ()
        result = _bin_sizes(values)
        values[0] = 32
        assert result == (2, 8, 16)

    @pytest.mark.parametrize(
        "values",
        ["8", 8, [True], [3], [0], [-2], [2, 2], [2.0]],
        ids=["text", "integer", "boolean", "odd", "zero", "negative", "duplicate", "float"],
    )
    def test_invalid(self, values: object) -> None:
        with pytest.raises((TypeError, ValueError), match="bin_sizes"):
            _bin_sizes(values)


class TestVisiumHD:
    """Inputs fail before I/O and geometry policy reaches every linked element."""

    @pytest.mark.parametrize(
        ("options", "error", "match"),
        [
            ({"bin_counts_kind": "unknown"}, ValueError, "bin_counts_kind"),
            ({"cell_counts_kind": "unknown"}, ValueError, "cell_counts_kind"),
            ({"bin_geometry": "circles"}, ValueError, "bin_geometry"),
            ({"include_cell_boundaries": 1}, TypeError, "Boundary inclusion"),
            ({"include_nucleus_boundaries": "yes"}, TypeError, "Boundary inclusion"),
            ({"include_nucleus_table": 1}, TypeError, "must be booleans"),
            ({"include_images": None}, TypeError, "must be booleans"),
            ({"include_cytassist_image": 0}, TypeError, "must be booleans"),
            ({"make_var_names_unique": "no"}, TypeError, "must be booleans"),
            ({"include_nucleus_table": True, "include_nucleus_boundaries": False}, ValueError, "requires nucleus"),
            ({"fullres_image_file": "x.tif", "include_images": False}, ValueError, "requires include_images"),
            ({"invalid_geometries": "repair"}, ValueError, "invalid_geometries"),
            ({"scale_factors": [1]}, ValueError, "at least 2"),
            ({"chunks": 0}, ValueError, "chunks"),
            ({"data_axes": [1]}, TypeError, "data_axes"),
            ({"tiff_series": -1}, ValueError, "tiff_series"),
        ],
    )
    def test_option_validation_before_discovery(
        self, tmp_path: Path, mocker: MockerFixture, options: dict[str, object], error: type[Exception], match: str
    ) -> None:
        discovery = mocker.spy(_reader, "discover_layout")
        with pytest.raises(error, match=match):
            visium_hd(tmp_path, **options)  # pyrefly: ignore[bad-argument-type] - Invalid runtime options.
        discovery.assert_not_called()

    @pytest.mark.parametrize("options", [{"data_axes": "CYX"}, {"tiff_series": 0}, {"scale_factors": [2]}])
    def test_microscopy_options_require_images(self, tmp_path: Path, options: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="include_images"):
            visium_hd(tmp_path, include_images=False, **options)  # pyrefly: ignore[bad-argument-type] - Runtime option combinations.
