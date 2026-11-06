"""Tests for the shared SpatialData element-name validation."""

from __future__ import annotations

import re
import warnings

import pytest
from spatialdata._core.validation import check_valid_name

from spatialdata_io.readers._utils.names import sanitize_element_name, validate_element_name

_VALID = ["a", "sample", "Sample_1-2.3", "_x", "-", "...", "a..b", "_", "é-1", "échantillon", "样本", "x²", "tables"]
_INVALID = [
    "",
    ".",
    "..",
    "__",
    "__private",
    "has space",
    "a/b",
    "a\\b",
    "a:b",
    "a?",
    "a[1]",
    "a\x00",
    "a\tb",
    "a\nb",
    "a+b",
]


class TestValidateElementName:
    @pytest.mark.parametrize("name", _VALID)
    def test_returns_valid_name_unchanged(self, name: str) -> None:
        assert validate_element_name(name, parameter="dataset_id") is name

    @pytest.mark.parametrize("name", _INVALID)
    def test_rejects_invalid_name(self, name: str) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape("dataset_id must be a valid SpatialData element name: nonempty, only alphanumeric")
            + ".*"
            + re.escape(f"; found {name!r}."),
        ):
            validate_element_name(name, parameter="dataset_id")

    @pytest.mark.parametrize("name", [*_VALID, *_INVALID])
    def test_agrees_with_spatialdata(self, name: str) -> None:
        """The rule mirrors SpatialData's own unexported name check."""
        try:
            check_valid_name(name)
        except ValueError:
            accepted = False
        else:
            accepted = True

        assert accepted is (name in _VALID)

    @pytest.mark.parametrize("name", [None, 3, 1.5, b"sample", ["sample"], ("sample",)])
    def test_rejects_non_string(self, name: object) -> None:
        with pytest.raises(TypeError, match=re.escape(f"region must be a string; found {type(name).__name__}.")):
            validate_element_name(name, parameter="region")

    def test_type_is_checked_before_reservation(self) -> None:
        with pytest.raises(TypeError, match="sample name must be a string"):
            validate_element_name(b"table", parameter="sample name", reserved=("table",))

    @pytest.mark.parametrize("name", ["table", "Table", "TABLE", "tAbLe", "images", "IMAGES"])
    def test_rejects_reserved_name_in_any_letter_case(self, name: str) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "dataset_id must be a valid SpatialData element name other than ['images', 'table'] in any letter "
                f"case, because SpatialData compares element names case-insensitively; found {name!r}."
            ),
        ):
            validate_element_name(name, parameter="dataset_id", reserved=frozenset({"table", "images"}))

    @pytest.mark.parametrize("reserved", [("Table",), ("TABLE",), ["table"]])
    def test_reserved_names_compare_case_insensitively(self, reserved: tuple[str, ...] | list[str]) -> None:
        with pytest.raises(ValueError, match="other than"):
            validate_element_name("table", parameter="dataset_id", reserved=reserved)

    @pytest.mark.parametrize("name", ["tables", "table_1", "my_table", "image\u017f"])
    def test_accepts_name_differing_from_reserved_after_lowercasing(self, name: str) -> None:
        """SpatialData folds case with ``str.lower``, which keeps the long s (U+017F) distinct from ``s``."""
        assert validate_element_name(name, parameter="dataset_id", reserved=("table", "images")) == name

    def test_invalid_rule_is_checked_before_reservation(self) -> None:
        with pytest.raises(ValueError, match="not starting with '__'; found '__table'"):
            validate_element_name("__table", parameter="dataset_id", reserved=("__table",))

    def test_parameter_prefixes_message(self) -> None:
        parameter = "the H5AD filename stem, which names the grid unless dataset_id is given,"

        with pytest.raises(ValueError, match=re.escape(f"{parameter} must be a valid SpatialData element name")):
            validate_element_name("bad name", parameter=parameter)


def _sanitize(value: str, reserved: tuple[str, ...] = ()) -> str:
    return sanitize_element_name(value, source="test: the stem", parameter="name", reserved=reserved)


_REPAIRS = [
    ("has space", "has_space"),
    ("my run", "my_run"),
    ("a/b", "a_b"),
    ("a\\b", "a_b"),
    ("a:b", "a_b"),
    ("a?", "a_"),
    ("a[1]", "a_1_"),
    ("a\x00", "a_"),
    ("a\tb", "a_b"),
    ("a\nb", "a_b"),
    ("a+b", "a_b"),
    ("a  b", "a__b"),
    (" lead", "_lead"),
    ("  lead", "_lead"),
    ("__", "_"),
    ("___", "_"),
    ("__private", "_private"),
    ("__.x", "_.x"),
    ("/ x", "_x"),
    ("", "unnamed"),
    (".", "unnamed"),
    ("..", "unnamed"),
]


class TestSanitizeElementName:
    @pytest.mark.parametrize("name", _VALID)
    def test_valid_name_is_returned_unchanged_without_warning(self, name: str) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _sanitize(name) is name

    @pytest.mark.parametrize(("value", "expected"), _REPAIRS)
    def test_repairs_invalid_name(self, value: str, expected: str) -> None:
        with pytest.warns(UserWarning, match="is not a usable SpatialData element name") as caught:
            assert _sanitize(value) == expected

        assert len(caught) == 1
        assert str(caught[0].message) == (
            f"test: the stem {value!r} is not a usable SpatialData element name; using {expected!r}. "
            "Pass name to choose the name explicitly."
        )

    def test_every_invalid_fixture_name_is_repaired(self) -> None:
        assert {value for value, _ in _REPAIRS} >= set(_INVALID)

    @pytest.mark.parametrize(
        ("value", "reserved", "expected"),
        [
            ("table", ("table",), "table_"),
            ("Table", ("table",), "Table_"),
            ("TABLE", ("Table",), "TABLE_"),
            ("table", ("table", "table_"), "table__"),
            ("table", ("TABLE", "Table_", "table__"), "table___"),
            ("ta ble", ("ta_ble",), "ta_ble_"),
            ("__", ("_",), "unnamed"),
            ("_", ("_",), "unnamed"),
            ("_", ("_", "unnamed"), "unnamed_"),
            ("", ("unnamed",), "unnamed_"),
        ],
    )
    def test_reserved_names_get_an_underscore_suffix(
        self, value: str, reserved: tuple[str, ...], expected: str
    ) -> None:
        with pytest.warns(UserWarning, match=re.escape(f"using {expected!r}")):
            assert _sanitize(value, reserved) == expected

    @pytest.mark.parametrize("value", ["tables", "table_1", "image\u017f"])
    def test_near_reserved_names_are_unchanged(self, value: str) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _sanitize(value, ("table", "images")) is value

    @pytest.mark.parametrize("reserved", [(), ("table", "_", "unnamed", "a_b")])
    @pytest.mark.parametrize(
        "value",
        list(dict.fromkeys([*_VALID, *_INVALID, *(value for value, _ in _REPAIRS), "table", "TABLE", "a b", "ñ ñ"])),
    )
    def test_result_is_valid_deterministic_and_idempotent(self, value: str, reserved: tuple[str, ...]) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first, second = _sanitize(value, reserved), _sanitize(value, reserved)

        assert first == second
        assert validate_element_name(first, parameter="name", reserved=reserved) is first
        check_valid_name(first)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _sanitize(first, reserved) is first

    def test_warning_points_at_the_caller_outside_the_package(self) -> None:
        with pytest.warns(UserWarning, match="'bad name'") as caught:
            _sanitize("bad name")

        assert caught[0].filename == __file__
