"""Named and positional Visium contracts and exact normalization limits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.visium._schemas import POSITION_COLUMNS, classify_position_row, normalize_positions

_PATH = Path("positions.csv")


def _frame() -> pd.DataFrame:
    """Construct lexical input with ordinary and decimal grid representations."""
    return pd.DataFrame(
        {
            "barcode": ["002-1", "001-1"],
            "in_tissue": ["1", "0"],
            "array_row": ["2", "1.0"],
            "array_col": ["3", "0"],
            "pxl_row_in_fullres": ["30.5", "10"],
            "pxl_col_in_fullres": ["40.5", "20"],
        },
        dtype="string[pyarrow]",
    )


class TestClassifyPositionRow:
    """Distinguish positional records from named headers before name repair."""

    @pytest.mark.parametrize("columns", [POSITION_COLUMNS, tuple(reversed(POSITION_COLUMNS))])
    def test_named_fields_accept_order_and_extensions(self, columns: tuple[str, ...]) -> None:
        """Named fields accept order and extensions."""
        assert classify_position_row(",".join(("quality", *columns)), path=_PATH) == "headered"

    def test_headerless_fields_have_positional_meaning(self) -> None:
        """Headerless fields have positional meaning."""
        assert classify_position_row("002-1,1,0,0,20,30", path=_PATH) == "headerless"

    @pytest.mark.parametrize(
        ("row", "message"),
        [
            ("barcode,x,y", "required columns"),
            (",".join((*POSITION_COLUMNS, "barcode")), "duplicate column"),
            (",".join((*POSITION_COLUMNS, "")), "empty or duplicate"),
            ("a,1,0,0,2", "six positional"),
            ("a,1,0,0,2,3,4", "six positional"),
            ('"barcode,in_tissue', "malformed first"),
        ],
    )
    def test_rejects_ambiguous_and_malformed_fields(self, row: str, message: str) -> None:
        """Rejects ambiguous and malformed fields."""
        with pytest.raises(ReaderFormatError, match=message):
            classify_position_row(row, path=_PATH)


class TestNormalizePositions:
    """Validate source data once and retain exact grid integers and coordinates."""

    def test_explicit_arrow_storage_and_order(self) -> None:
        """Explicit arrow storage and order."""
        result = normalize_positions(_frame(), path=_PATH)

        assert result.index.tolist() == ["002-1", "001-1"]
        assert result.index.name == "barcode"
        for column in ("in_tissue", "array_row", "array_col"):
            assert str(result[column].dtype) == "int64[pyarrow]"
        for column in ("pxl_row_in_fullres", "pxl_col_in_fullres"):
            assert str(result[column].dtype) == "double[pyarrow]"
        np.testing.assert_array_equal(result["array_row"], [2, 1])
        np.testing.assert_array_equal(result["pxl_row_in_fullres"], [30.5, 10.0])

    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("9007199254740993", 2**53 + 1),
            ("9007199254740993.0", 2**53 + 1),
            ("9223372036854775807", 2**63 - 1),
            ("9223372036854775807.0", 2**63 - 1),
            ("1e3", 1000),
        ],
    )
    def test_grid_integers_never_round_through_float(self, token: str, expected: int) -> None:
        """Grid integers never round through float."""
        frame = _frame()
        frame.loc[0, "array_row"] = token

        result = normalize_positions(frame, path=_PATH)

        assert int(result["array_row"].iloc[0]) == expected
        assert int(result["array_row"].iloc[1]) == 1

    @pytest.mark.parametrize(
        ("column", "value", "message"),
        [
            ("barcode", "", "empty"),
            ("barcode", "001-1", "duplicate barcodes"),
            ("barcode", "a\x00b", "NUL"),
            ("barcode", None, "null"),
            ("array_row", "nan", "finite numeric"),
            ("array_row", None, "finite numeric"),
            ("array_row", "not-a-number", "finite numeric"),
            ("array_row", "-1.0", "nonnegative integer"),
            ("array_col", "-1", "nonnegative integer"),
            ("pxl_row_in_fullres", None, "finite numeric"),
            ("array_row", "inf", "finite numeric"),
            ("array_row", "-1", "nonnegative integer"),
            ("array_row", "0.5", "integer values"),
            ("array_row", "9223372036854775808", "fitting int64"),
            ("array_col", "9223372036854775808", "fitting int64"),
            ("array_row", "9007199254740993.5", "integer values"),
            ("array_row", "9223372036854775808.0", "fitting int64"),
            ("array_row", "1.00000000000000001", "integer values"),
            ("array_row", "9007199254740990.5", "integer values"),
            ("array_row", "-1e-400", "integer values"),
            ("array_row", "1e-400", "integer values"),
            ("pxl_row_in_fullres", "1e-400", "precision loss"),
            ("pxl_row_in_fullres", "nonnumeric", "finite numeric"),
            ("pxl_row_in_fullres", "inf", "finite numeric"),
            ("pxl_row_in_fullres", "9007199254740991.5", "precision loss"),
            ("pxl_row_in_fullres", "-1e-400", "precision loss"),
            ("in_tissue", "2", "binary"),
            ("pxl_row_in_fullres", "-1", "nonnegative pixel"),
            ("pxl_row_in_fullres", "9007199254740993", "precision loss"),
            ("pxl_col_in_fullres", "9007199254740993.0", "precision loss"),
        ],
    )
    def test_rejects_invalid_values(self, column: str, value: str | None, message: str) -> None:
        """Rejects invalid values."""
        frame = _frame()
        frame.loc[0, column] = value

        with pytest.raises(ReaderFormatError, match=message):
            normalize_positions(frame, path=_PATH)

    def test_rejects_empty_positions(self) -> None:
        """Rejects empty positions."""
        with pytest.raises(ReaderFormatError, match="zero rows"):
            normalize_positions(_frame().iloc[:0], path=_PATH)

    def test_float_text_conversion_uses_correct_rounding(self) -> None:
        """Float text conversion uses correct rounding."""
        frame = _frame()
        frame.loc[0, "pxl_row_in_fullres"] = "0.10000000000000002"

        result = normalize_positions(frame, path=_PATH)

        assert result["pxl_row_in_fullres"].iloc[0] == float("0.10000000000000002")

    def test_exact_large_pixel_coordinate_is_allowed(self) -> None:
        """Exact large pixel coordinate is allowed."""
        frame = _frame()
        frame.loc[0, "pxl_row_in_fullres"] = str(2**54)

        result = normalize_positions(frame, path=_PATH)

        assert result["pxl_row_in_fullres"].iloc[0] == float(2**54)
