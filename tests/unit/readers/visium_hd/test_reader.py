"""Public validation and final scientific composition of Visium HD products."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers.visium_hd import _reader, visium_hd
from spatialdata_io.readers.visium_hd._reader import _bin_sizes

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

# One invalid value per public option, in signature order, with the error it must raise.
_INVALID: list[tuple[str, object, type[Exception], str]] = [
    ("bin_sizes", [3], ValueError, "bin_sizes must contain positive even"),
    ("bin_geometry", "circles", ValueError, "bin_geometry must be one of"),
    ("bin_counts_kind", "unknown", ValueError, "bin_counts_kind must be one of"),
    ("include_cell_boundaries", 1, TypeError, "include_cell_boundaries must be a boolean or None"),
    ("include_nucleus_boundaries", "yes", TypeError, "include_nucleus_boundaries must be a boolean or None"),
    ("include_nucleus_table", 1, TypeError, "include_nucleus_table must be a boolean; found int"),
    ("cell_counts_kind", "unknown", ValueError, "cell_counts_kind must be one of"),
    ("make_var_names_unique", "no", TypeError, "make_var_names_unique must be a boolean; found str"),
    ("invalid_geometries", "repair", ValueError, "invalid_geometries"),
    ("include_images", None, TypeError, "include_images must be a boolean; found NoneType"),
    ("include_cytassist_image", 0, TypeError, "include_cytassist_image must be a boolean; found int"),
    ("fullres_image_file", 3, TypeError, "fullres_image_file must be a string, a path or None"),
    ("data_axes", ["C", "Y", "X"], TypeError, "data_axes must be a string"),
    ("tiff_series", -1, ValueError, "tiff_series must be nonnegative"),
    ("scale_factors", [1], ValueError, "at least 2"),
    ("chunks", 0, ValueError, "chunks"),
    ("path_policy", "loose", ValueError, "path_policy"),
]


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
        ("values", "error", "match"),
        [
            ("8", ValueError, "bin_sizes must be 'all' or a sequence"),
            (8, TypeError, "bin_sizes must be 'all' or a sequence of integers; found int"),
            ([True], TypeError, "bin_sizes must contain integers; found bool"),
            ([2.0], TypeError, "bin_sizes must contain integers; found float"),
            ([3], ValueError, "positive even micrometre sizes; found 3"),
            ([0], ValueError, "positive even micrometre sizes; found 0"),
            ([-2], ValueError, "positive even micrometre sizes; found -2"),
            ([2, 2], ValueError, "bin_sizes must not contain duplicates"),
        ],
        ids=["text", "integer", "boolean", "float", "odd", "zero", "negative", "duplicate"],
    )
    def test_invalid(self, values: object, error: type[Exception], match: str) -> None:
        with pytest.raises(error, match=match):
            _bin_sizes(values)


class TestVisiumHD:
    """Inputs fail before I/O, in signature order."""

    def test_signature(self) -> None:
        parameters = inspect.signature(visium_hd).parameters
        assert list(parameters)[1:] == [name for name, *_ in _INVALID]
        assert parameters["data_axes"].annotation == "str | None"
        assert parameters["tiff_series"].default is None

    @pytest.mark.parametrize("case", _INVALID, ids=[name for name, *_ in _INVALID])
    def test_option_validation_before_io(
        self, tmp_path: Path, mocker: MockerFixture, case: tuple[str, object, type[Exception], str]
    ) -> None:
        name, value, error, match = case
        # The root does not exist, so any filesystem access would raise FileNotFoundError instead.
        discovery = mocker.spy(_reader, "discover_layout")
        with pytest.raises(error, match=match):
            visium_hd(tmp_path / "missing", **{name: value})  # pyrefly: ignore[bad-argument-type] - Invalid runtime options.
        if name != "path_policy":
            discovery.assert_not_called()

    @pytest.mark.parametrize("index", range(len(_INVALID) - 1), ids=[name for name, *_ in _INVALID[:-1]])
    def test_options_are_validated_in_signature_order(self, tmp_path: Path, index: int) -> None:
        (first, first_value, error, match), (second, second_value, *_) = _INVALID[index], _INVALID[index + 1]
        options = {first: first_value, second: second_value}
        with pytest.raises(error, match=match):
            visium_hd(tmp_path, **options)  # pyrefly: ignore[bad-argument-type] - Invalid runtime options.

    @pytest.mark.parametrize("name", ["bin_geometry", "bin_counts_kind", "cell_counts_kind"])
    def test_choices_must_be_strings(self, tmp_path: Path, name: str) -> None:
        with pytest.raises(TypeError, match=f"{name} must be one of .*; found int"):
            visium_hd(tmp_path, **{name: 1})  # pyrefly: ignore[bad-argument-type] - Invalid runtime options.

    @pytest.mark.parametrize(
        "options",
        [{"fullres_image_file": "x.tif"}, {"data_axes": "CYX"}, {"tiff_series": 0}, {"scale_factors": [2]}],
    )
    def test_microscopy_options_require_images(self, tmp_path: Path, options: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="require include_images=True"):
            visium_hd(tmp_path, include_images=False, **options)  # pyrefly: ignore[bad-argument-type] - Runtime option combinations.

    def test_nucleus_table_requires_nucleus_boundaries(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="include_nucleus_table requires nucleus boundaries"):
            visium_hd(tmp_path, include_nucleus_table=True, include_nucleus_boundaries=False)

    @pytest.mark.parametrize("value", [False, None])
    def test_single_options_precede_combinations(self, tmp_path: Path, *, value: bool | None) -> None:
        # An invalid chunk size is reported before the nucleus/images combinations it accompanies.
        with pytest.raises(ValueError, match="chunks"):
            visium_hd(
                tmp_path,
                include_nucleus_table=True,
                include_nucleus_boundaries=value,
                include_images=False,
                tiff_series=0,
                chunks=0,
            )
