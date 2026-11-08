"""Tests for the shared SpatialData element-name and table column-name rules."""

from __future__ import annotations

import re
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from spatialdata._core.validation import check_valid_dataframe_column_name, check_valid_name, validate_table_attr_keys

from spatialdata_io.readers._utils.names import (
    sanitize_column_names,
    sanitize_element_name,
    sanitize_uns_keys,
    validate_element_name,
)

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

    def test_without_parameter_the_warning_suggests_renaming_the_files(self) -> None:
        with pytest.warns(UserWarning, match="'Roi 1'") as caught:
            result = sanitize_element_name("Roi 1", source="test: the ROI file prefix", parameter=None)

        assert result == "Roi_1"
        assert len(caught) == 1
        assert str(caught[0].message) == (
            "test: the ROI file prefix 'Roi 1' is not a usable SpatialData element name; using 'Roi_1'. "
            "Rename the source files to choose the name explicitly."
        )

    def test_without_parameter_a_valid_name_is_silent(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert sanitize_element_name("Roi1", source="test", parameter=None) == "Roi1"


def _columns(columns: list[str], reserved: tuple[str, ...] = ()) -> list[str]:
    return sanitize_column_names(columns, source="test: the header", reserved=reserved)


def _message(pairs: str) -> str:
    return (
        f"test: the header has column names that a SpatialData table does not accept; using {pairs}. "
        "Rename the source columns to choose the names explicitly."
    )


def _assert_usable(result: list[str], reserved: tuple[str, ...]) -> None:
    """Check SpatialData's own column rules: valid names, unique ignoring case within the frame."""
    for name in result:
        check_valid_dataframe_column_name(name)
    folded = [name.lower() for name in result]
    assert len(set(folded)) == len(folded)
    assert not {name.lower() for name in reserved if name not in result} & set(folded)
    frame = pd.DataFrame({name: np.zeros(1) for name in result}, index=["0"])
    validate_table_attr_keys(ad.AnnData(obs=frame, var=frame.rename(index={"0": "f"})))


class TestSanitizeColumnNames:
    @pytest.mark.parametrize(
        "columns",
        [[], ["a"], ["CellID", "X_centroid", "Area", "tile_num", "MoransI_p.value", "_x", "-", "é", "_INDEX", "Index"]],
    )
    def test_valid_names_are_returned_unchanged_without_warning(self, columns: list[str]) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _columns(columns) == columns

    @pytest.mark.parametrize(("value", "expected"), _REPAIRS)
    def test_repairs_invalid_name_like_an_element_name(self, value: str, expected: str) -> None:
        with pytest.warns(UserWarning, match="column names") as caught:
            assert _columns(["keep", value]) == ["keep", expected]

        assert len(caught) == 1
        assert str(caught[0].message) == _message(f"{value!r} -> {expected!r}")

    @pytest.mark.parametrize(
        ("columns", "expected"),
        [
            (["_index"], ["index"]),
            (["_index", "index"], ["index_1", "index"]),
            (["index", "_index", "index_1"], ["index", "index_2", "index_1"]),
        ],
    )
    def test_index_becomes_index_as_in_spatialdata(self, columns: list[str], expected: list[str]) -> None:
        with pytest.warns(UserWarning, match="'_index' -> "):
            assert _columns(columns) == expected

    @pytest.mark.parametrize(
        ("columns", "expected"),
        [
            # A valid name keeps its spelling wherever it appears.
            (["CD45 RO", "CD45_RO"], ["CD45_RO_1", "CD45_RO"]),
            (["CD45_RO", "CD45 RO"], ["CD45_RO", "CD45_RO_1"]),
            # Repairs that collide are numbered in column order.
            (["a b", "a:b", "a/b"], ["a_b", "a_b_1", "a_b_2"]),
            # The suffix skips names already taken, also by later columns.
            (["a b", "a:b", "a_b_1"], ["a_b", "a_b_2", "a_b_1"]),
            # SpatialData compares column names ignoring letter case.
            (["Area", "area"], ["Area", "area_1"]),
            (["A b", "a_B"], ["A_b_1", "a_B"]),
            (["x", "X", "X_1"], ["x", "X_2", "X_1"]),
            # "__1" would be invalid.
            (["_", "__", "/", "_1"], ["_", "_2", "_3", "_1"]),
        ],
    )
    def test_collisions_get_the_first_free_numeric_suffix(self, columns: list[str], expected: list[str]) -> None:
        with pytest.warns(UserWarning, match="column names"):
            assert _columns(columns) == expected

    def test_one_warning_lists_every_change_in_column_order(self) -> None:
        with pytest.warns(UserWarning, match="column names") as caught:
            result = _columns(["CD45 RO", "ok", "area", "Area", "_index", "CD45_RO"])

        assert result == ["CD45_RO_1", "ok", "area", "Area_1", "index", "CD45_RO"]
        assert len(caught) == 1
        assert str(caught[0].message) == _message("'CD45 RO' -> 'CD45_RO_1', 'Area' -> 'Area_1', '_index' -> 'index'")

    @pytest.mark.parametrize(
        ("columns", "reserved", "expected"),
        [
            (["Region", "x"], ("region",), ["Region_1", "x"]),
            (["cell id"], ("cell_id",), ["cell_id_1"]),
            (["cell id", "CELL_ID_1"], ("cell_id",), ["cell_id_2", "CELL_ID_1"]),
            (["region", "Region"], ("region",), ["region", "Region_1"]),
            (["Region", "region"], ("region",), ["Region_1", "region"]),
            (["spatial_Frame"], ("spatial_frame", "region"), ["spatial_Frame_1"]),
        ],
    )
    def test_reserved_names_keep_their_spelling(
        self, columns: list[str], reserved: tuple[str, ...], expected: list[str]
    ) -> None:
        with pytest.warns(UserWarning, match="column names"):
            assert _columns(columns, reserved) == expected

    def test_exact_reserved_name_is_the_reserved_column(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _columns(["region", "CellID", "x"], ("region", "CellID")) == ["region", "CellID", "x"]

    def test_rejects_duplicate_columns(self) -> None:
        with pytest.raises(ValueError, match=re.escape("test: the header has duplicate column names ['a', 'b c'].")):
            _columns(["b c", "a", "b c", "a", "ok"])

    @pytest.mark.parametrize("reserved", [(), ("region", "cell_id", "a_b")])
    @pytest.mark.parametrize(
        "columns",
        [
            [*_VALID, *_INVALID],
            [value for value, _ in _REPAIRS],
            ["_index", "index", "INDEX", "Index_1", "a b", "A_B", "a_b_1", "region", "Region", "cell id"],
            ["ñ ñ", "Ñ_Ñ", "x²", "X²", "", "unnamed", "Unnamed_1"],
        ],
    )
    def test_result_is_usable_deterministic_and_idempotent(self, columns: list[str], reserved: tuple[str, ...]) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first, second = _columns(columns, reserved), _columns(columns, reserved)

        assert first == second
        assert len(first) == len(columns)
        _assert_usable(first, reserved)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _columns(first, reserved) == first

    def test_warning_points_at_the_caller_outside_the_package(self) -> None:
        with pytest.warns(UserWarning, match="'bad name'") as caught:
            _columns(["bad name"])

        assert caught[0].filename == __file__

    def test_uns_keys_follow_the_element_rule_only(self) -> None:
        with pytest.warns(UserWarning, match="uns keys") as caught:
            result = sanitize_column_names(
                ["Run Metrics", "_index", "QC", "ok"], source="test: the categories", reserved=("qc",), kind="uns"
            )

        assert result == ["Run_Metrics", "_index", "QC_1", "ok"]
        assert [str(warning.message) for warning in caught] == [
            (
                "test: the categories has names that a SpatialData table does not accept as uns keys; using "
                "'Run Metrics' -> 'Run_Metrics', 'QC' -> 'QC_1'. Rename them in the source to choose the names "
                "explicitly."
            )
        ]
        uns = dict.fromkeys(["qc", *result], 1)
        validate_table_attr_keys(ad.AnnData(uns=uns))

    def test_uns_duplicates_are_rejected(self) -> None:
        with pytest.raises(ValueError, match=re.escape("test: the categories has duplicate names ['a'].")):
            sanitize_column_names(["a", "a"], source="test: the categories", kind="uns")


class TestSanitizeUnsKeys:
    def test_renames_keys_in_place_keeping_value_identity_and_order(self) -> None:
        nested = {"inner key": 1}
        frame = pd.DataFrame({"sn": ["chip"]})
        table = ad.AnnData(uns={"z key": nested, "_index": frame, "Positions": 3, "a": "kept"})
        mapping = table.uns

        with pytest.warns(UserWarning, match="uns keys") as caught:
            sanitize_uns_keys(table, source="test: the H5AD uns", reserved=("positions",))

        assert list(table.uns) == ["z_key", "_index", "Positions_1", "a"]
        assert table.uns is mapping
        assert table.uns["z_key"] is nested
        assert table.uns["_index"] is frame
        assert (table.uns["Positions_1"], table.uns["a"]) == (3, "kept")
        assert [str(warning.message) for warning in caught] == [
            (
                "test: the H5AD uns has names that a SpatialData table does not accept as uns keys; using "
                "'z key' -> 'z_key', 'Positions' -> 'Positions_1'. Rename them in the source to choose the names "
                "explicitly."
            )
        ]
        validate_table_attr_keys(table)

    def test_exact_reserved_key_keeps_its_spelling(self) -> None:
        table = ad.AnnData(uns={"spatialdata_attrs": {"region": "stale"}})

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sanitize_uns_keys(table, source="test: the H5AD uns", reserved=("spatialdata_attrs",))

        assert table.uns == {"spatialdata_attrs": {"region": "stale"}}

    @pytest.mark.parametrize("uns", [{}, {"a": 1, "B.c-d": np.arange(2)}])
    def test_valid_keys_are_untouched_without_warning(self, uns: dict[str, object]) -> None:
        table = ad.AnnData(uns=uns)
        values = list(table.uns.values())

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sanitize_uns_keys(table, source="test: the H5AD uns")

        assert list(table.uns) == list(uns)
        assert all(new is old for new, old in zip(table.uns.values(), values, strict=True))

    def test_warning_points_at_the_caller_outside_the_package(self) -> None:
        table = ad.AnnData(uns={"bad key": 1})

        with pytest.warns(UserWarning, match="'bad key'") as caught:
            sanitize_uns_keys(table, source="test: the H5AD uns")

        assert caught[0].filename == __file__
