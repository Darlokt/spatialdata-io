"""Named and positional Visium contracts and typed semantic validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers.visium._schemas import (
    POSITION_COLUMNS,
    POSITION_SOURCE_SCHEMA,
    classify_position_row,
    normalize_positions,
)

_PATH = Path("positions.csv")
_CONTEXT = ReaderErrorContext("visium", "tissue positions", "2.1.0")


def _frame() -> pd.DataFrame:
    """Construct already typed input with integer grid fields and fractional pixels."""
    return pa.Table.from_pydict(
        {
            "barcode": ["002-1", "001-1"],
            "in_tissue": [1, 0],
            "array_row": [2, 1],
            "array_col": [3, 0],
            "pxl_row_in_fullres": [30.5, 10.0],
            "pxl_col_in_fullres": [40.5, 20.0],
        },
        schema=POSITION_SOURCE_SCHEMA,
    ).to_pandas(types_mapper=pd.ArrowDtype)


class TestPositionSourceSchema:
    """Keep the vendor scalar domains explicit at ingestion."""

    def test_fixed_native_types_and_field_order(self) -> None:
        assert (
            pa.schema(
                [
                    ("barcode", pa.string()),
                    ("in_tissue", pa.uint8()),
                    ("array_row", pa.uint64()),
                    ("array_col", pa.uint64()),
                    ("pxl_row_in_fullres", pa.float64()),
                    ("pxl_col_in_fullres", pa.float64()),
                ]
            )
            == POSITION_SOURCE_SCHEMA
        )


class TestClassifyPositionRow:
    """Distinguish positional records from named headers before name repair."""

    @pytest.mark.parametrize("columns", [POSITION_COLUMNS, tuple(reversed(POSITION_COLUMNS))])
    def test_named_fields_accept_order_and_extensions(self, columns: tuple[str, ...]) -> None:
        """Named fields accept order and extensions."""
        assert classify_position_row(("quality", *columns), path=_PATH, context=_CONTEXT) == "headered"

    def test_headerless_fields_have_positional_meaning(self) -> None:
        """Headerless fields have positional meaning."""
        assert classify_position_row(("002-1", "1", "0", "0", "20", "30"), path=_PATH, context=_CONTEXT) == "headerless"

    @pytest.mark.parametrize(
        ("columns", "message"),
        [
            (("barcode", "x", "y"), "required columns"),
            ((*POSITION_COLUMNS, "barcode"), "duplicate column"),
            ((*POSITION_COLUMNS, ""), "empty or duplicate"),
            (("a", "1", "0", "0", "2"), "six positional"),
            (("a", "1", "0", "0", "2", "3", "4"), "six positional"),
            ((), "six positional"),
        ],
    )
    def test_rejects_ambiguous_and_malformed_fields(self, columns: tuple[str, ...], message: str) -> None:
        """Rejects ambiguous and malformed fields."""
        with pytest.raises(ReaderFormatError, match=message) as caught:
            classify_position_row(columns, path=_PATH, context=_CONTEXT)

        assert caught.value.context is _CONTEXT
        assert caught.value.path == _PATH


class TestNormalizePositions:
    """Validate typed source semantics and retain source identity and order."""

    def test_explicit_arrow_storage_and_order(self) -> None:
        result = normalize_positions(_frame(), path=_PATH, context=_CONTEXT)

        assert result.index.tolist() == ["002-1", "001-1"]
        assert result.index.name == "barcode"
        assert result.index.dtype == pd.ArrowDtype(pa.string())
        assert str(result["in_tissue"].dtype) == "uint8[pyarrow]"
        for column in ("array_row", "array_col"):
            assert str(result[column].dtype) == "uint64[pyarrow]"
        for column in ("pxl_row_in_fullres", "pxl_col_in_fullres"):
            assert str(result[column].dtype) == "double[pyarrow]"
        np.testing.assert_array_equal(result["array_row"], [2, 1])
        np.testing.assert_array_equal(result["pxl_row_in_fullres"], [30.5, 10.0])

    @pytest.mark.parametrize("value", [0, 2**53 + 1, 2**63 - 1, 2**63, 2**64 - 1])
    def test_grid_integers_never_round_through_float(self, value: int) -> None:
        frame = _frame()
        frame.loc[0, "array_row"] = value

        result = normalize_positions(frame, path=_PATH, context=_CONTEXT)

        assert int(result["array_row"].iloc[0]) == value
        assert int(result["array_row"].iloc[1]) == 1

    @pytest.mark.parametrize(
        ("column", "value", "message"),
        [
            ("barcode", "", "empty"),
            ("barcode", "001-1", "duplicate barcodes"),
            ("barcode", "a\x00b", "NUL"),
            ("pxl_row_in_fullres", float("inf"), "finite numeric"),
            ("pxl_col_in_fullres", float("-inf"), "finite numeric"),
            ("in_tissue", 2, "binary"),
            ("in_tissue", 255, "binary"),
            ("in_tissue", None, "binary"),
            ("pxl_row_in_fullres", -1, "nonnegative pixel"),
            ("pxl_col_in_fullres", -0.25, "nonnegative pixel"),
        ],
    )
    def test_rejects_invalid_values(self, column: str, value: str | float | None, message: str) -> None:
        frame = _frame()
        frame.loc[0, column] = value

        with pytest.raises(ReaderFormatError, match=message) as caught:
            normalize_positions(frame, path=_PATH, context=_CONTEXT)

        assert caught.value.context is _CONTEXT
        assert "detected version '2.1.0'" in str(caught.value)

    def test_rejects_empty_positions(self) -> None:
        with pytest.raises(ReaderFormatError, match="zero rows"):
            normalize_positions(_frame().iloc[:0], path=_PATH, context=_CONTEXT)
