"""Named positions and mapping field contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pytest
from numpy.testing import assert_array_equal

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.visium_hd._schemas import (
    POSITIONS_SCHEMA,
    parse_bin_barcodes,
    validate_mapping_schema,
    validate_mappings,
    validate_position_batch,
    validate_schema,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestValidateSchema:
    """Named source fields, duplicate rejection and extra-column policy."""

    def test_reordered_extra_columns(self, tmp_path: Path) -> None:
        validate_schema(
            pa.schema([pa.field("extra", pa.string()), *reversed(POSITIONS_SCHEMA)]),
            POSITIONS_SCHEMA,
            tmp_path,
            "positions",
        )

    @pytest.mark.parametrize(
        "schema",
        [
            pa.schema([pa.field("barcode", pa.string())]),
            pa.schema([pa.field("barcode", pa.string()), pa.field("barcode", pa.string())]),
            pa.schema(
                [
                    pa.field(field.name, pa.int64() if field.name == "array_row" else field.type)
                    for field in POSITIONS_SCHEMA
                ]
            ),
        ],
    )
    def test_invalid(self, tmp_path: Path, schema: pa.Schema) -> None:
        with pytest.raises(ReaderFormatError):
            validate_schema(schema, POSITIONS_SCHEMA, tmp_path, "positions")


class TestParseBinBarcodes:
    """Canonical square identifiers and source observation order."""

    def test_preserves_order(self, tmp_path: Path) -> None:
        rows, cols = parse_bin_barcodes(
            pd.Index(["s_008um_00002_00001-1", "s_008um_00000_00003-1"]), 8, (4, 4), tmp_path
        )
        assert_array_equal(rows, [2, 0])
        assert_array_equal(cols, [1, 3])

    @pytest.mark.parametrize(
        "names", [["x"], ["s_008um_00004_00000-1"], ["s_008um_00000_00004-1"], ["s_008um_00000_00000-1"] * 2]
    )
    def test_invalid(self, tmp_path: Path, names: list[str]) -> None:
        with pytest.raises(ReaderFormatError):
            parse_bin_barcodes(pd.Index(names), 8, (4, 4), tmp_path)


class TestValidatePositionBatch:
    """Position nullability, finiteness and tissue flags."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("barcode", None),
            ("in_tissue", 2),
            ("pxl_col_in_fullres", float("nan")),
            ("pxl_row_in_fullres", float("inf")),
        ],
    )
    def test_invalid(self, tmp_path: Path, field: str, value: str | float | None) -> None:
        data = {
            "barcode": ["s_002um_00000_00000-1"],
            "in_tissue": [1],
            "array_row": [0],
            "array_col": [0],
            "pxl_row_in_fullres": [2.0],
            "pxl_col_in_fullres": [3.0],
        }
        arrays = [
            pa.array([value] if item.name == field else data[item.name], type=item.type) for item in POSITIONS_SCHEMA
        ]
        with pytest.raises(ReaderFormatError):
            validate_position_batch(pa.RecordBatch.from_arrays(arrays, schema=POSITIONS_SCHEMA), tmp_path)


class TestValidateMappingSchema:
    """Projected barcode-mapping physical field types."""

    def test_named_contract(self, tmp_path: Path) -> None:
        validate_mapping_schema(
            pa.schema(
                [
                    ("cell_id", pa.string()),
                    ("square_002um", pa.string()),
                    ("in_nucleus", pa.bool_()),
                    ("in_cell", pa.bool_()),
                ]
            ),
            tmp_path,
        )


class TestValidateMappings:
    """Mapping nullability and membership relationships."""

    def test_valid_nullable_cell(self, tmp_path: Path) -> None:
        validate_mappings(
            pa.table(
                {
                    "square_002um": ["a", "b"],
                    "cell_id": [None, "cell"],
                    "in_cell": [False, True],
                    "in_nucleus": [False, True],
                }
            ),
            tmp_path,
        )

    @pytest.mark.parametrize(
        ("column", "values"),
        [
            ("square_002um", [None]),
            ("in_cell", [None]),
            ("in_nucleus", [None]),
            ("cell_id", ["cell"]),
            ("in_nucleus", [True]),
        ],
    )
    def test_invalid(self, tmp_path: Path, column: str, values: list[str | bool | None]) -> None:
        data = {
            "square_002um": pa.array(["a"]),
            "cell_id": pa.array([None], type=pa.string()),
            "in_cell": pa.array([False]),
            "in_nucleus": pa.array([False]),
        }
        data[column] = pa.array(values, type=data[column].type)
        with pytest.raises(ReaderFormatError):
            validate_mappings(pa.table(data), tmp_path)
