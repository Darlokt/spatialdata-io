"""Named positions and mapping field contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from numpy.testing import assert_array_equal

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.visium_hd._schemas import (
    MAPPINGS_SCHEMA,
    POSITIONS_SCHEMA,
    bin_instance_ids,
    parse_bin_barcodes,
    parse_cell_barcodes,
    validate_mappings,
    validate_position_batch,
    validate_probe_annotations,
    validate_schema,
)
from tests.integration.readers.visium_hd._fixtures import VERSION

if TYPE_CHECKING:
    from pathlib import Path

    from pandas.api.extensions import ExtensionArray

_CONTEXT = ReaderErrorContext("visium_hd", "test artifact", VERSION)


class TestValidateProbeAnnotations:
    """Probe identity stays unique even when a gene has multiple probes."""

    @pytest.mark.parametrize(
        ("column", "values", "expected"),
        [
            ("filtered_barcodes", None, "filtered_barcodes flags"),
            ("filtered_barcodes", ["True", "False"], "filtered_barcodes flags"),
            ("filtered_barcodes", pd.array([True, None], dtype="boolean"), "filtered_barcodes flags"),
            ("filtered_probes", None, "filtered_probes flags"),
            ("filtered_probes", [0, 1], "filtered_probes flags"),
            ("filtered_probes", pd.array([True, None], dtype="boolean"), "filtered_probes flags"),
            ("probe_ids", None, "separate probe IDs and gene IDs"),
            ("gene_ids", None, "separate probe IDs and gene IDs"),
            ("probe_ids", ["probe", "probe"], "nonempty unique probe IDs"),
            ("probe_ids", ["probe", ""], "nonempty unique probe IDs"),
            ("probe_ids", ["probe", None], "nonempty unique probe IDs"),
            ("probe_name", ["other", "fields"], "conflicts with the stored probe names"),
        ],
    )
    def test_invalid(
        self, tmp_path: Path, column: str, values: list[str | int | None] | ExtensionArray | None, expected: str
    ) -> None:
        obs = pd.DataFrame({"filtered_barcodes": [True, False]})
        var = pd.DataFrame(
            {"probe_ids": ["probe-a", "probe-b"], "gene_ids": ["gene-a", "gene-a"], "filtered_probes": [False, True]}
        )
        frame = obs if column == "filtered_barcodes" else var
        if values is None:
            del frame[column]
        else:
            frame[column] = values

        with pytest.raises(ReaderFormatError, match=expected):
            validate_probe_annotations(obs, var, tmp_path, context=_CONTEXT)


class TestValidateSchema:
    """Named source fields, duplicate rejection and extra-column policy."""

    def test_reordered_extra_columns(self, tmp_path: Path) -> None:
        validate_schema(
            pa.schema([pa.field("extra", pa.string()), *reversed(POSITIONS_SCHEMA)]),
            POSITIONS_SCHEMA,
            tmp_path,
            context=_CONTEXT,
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
        with pytest.raises(ReaderFormatError) as error:
            validate_schema(schema, POSITIONS_SCHEMA, tmp_path, context=_CONTEXT)
        assert error.value.context == _CONTEXT


class TestParseBinBarcodes:
    """Canonical square identifiers and source observation order."""

    def test_preserves_order(self, tmp_path: Path) -> None:
        rows, cols = parse_bin_barcodes(
            pd.Index(["s_008um_00002_00001-1", "s_008um_00000_00003-1"]), 8, (4, 4), tmp_path, version=VERSION
        )
        assert_array_equal(rows, [2, 0])
        assert_array_equal(cols, [1, 3])

    @pytest.mark.parametrize("names", [["x"], ["s_008um_00004_00000-1"], ["s_008um_00000_00004-1"]])
    def test_invalid(self, tmp_path: Path, names: list[str]) -> None:
        with pytest.raises(ReaderFormatError):
            parse_bin_barcodes(pd.Index(names), 8, (4, 4), tmp_path, version=VERSION)


class TestParseCellBarcodes:
    """Accept only the documented fixed-width positive ID spelling."""

    def test_exact(self, tmp_path: Path) -> None:
        result = parse_cell_barcodes(
            pd.Index(["cellid_999999999-1", "cellid_000000001-1"]), ArtifactFile(tmp_path), version="4.1.0"
        )
        np.testing.assert_array_equal(result, [999999999, 1])
        assert result.dtype == np.uint32

    def test_empty(self, tmp_path: Path) -> None:
        result = parse_cell_barcodes(pd.Index([], dtype=str), ArtifactFile(tmp_path), version="4.1.0")
        assert result.shape == (0,)
        assert result.dtype == np.uint32

    @pytest.mark.parametrize(
        "barcode",
        [
            "cellid_0-1",
            "cellid_000000000-1",
            "cellid_000000001-2",
            "cellid_00000001a-1",
            "cellid_1000000000-1",
            "cellid_-00000001-1",
            "cellid_4294967297-1",
        ],
    )
    def test_invalid(self, tmp_path: Path, barcode: str) -> None:
        with pytest.raises(ReaderFormatError, match="cell"):
            parse_cell_barcodes(pd.Index([barcode]), ArtifactFile(tmp_path), version="4.1.0")


class TestBinInstanceIds:
    def test_uses_positive_consecutive_uint32_ids(self) -> None:
        result = bin_instance_ids(3)
        assert result.dtype == np.uint32
        assert_array_equal(result, [1, 2, 3])

    def test_rejects_unrepresentable_population(self) -> None:
        with pytest.raises(ValueError, match="uint32 label capacity"):
            bin_instance_ids(np.iinfo(np.uint32).max + 1)


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
            validate_position_batch(
                pa.RecordBatch.from_arrays(arrays, schema=POSITIONS_SCHEMA), tmp_path, context=_CONTEXT
            )


class TestMappingsSchema:
    """Projected barcode-mapping physical field types, in any order beside extra columns."""

    def test_named_contract(self, tmp_path: Path) -> None:
        assert [field.name for field in MAPPINGS_SCHEMA if field.nullable] == ["cell_id"]
        validate_schema(
            pa.schema(
                [
                    ("square_008um", pa.string()),
                    ("cell_id", pa.string()),
                    ("square_002um", pa.string()),
                    ("in_nucleus", pa.bool_()),
                    ("in_cell", pa.bool_()),
                ]
            ),
            MAPPINGS_SCHEMA,
            tmp_path,
            context=_CONTEXT,
        )


class TestValidateMappings:
    """Mapping nullability and membership relationships."""

    def test_valid_nullable_cell(self, tmp_path: Path) -> None:
        validate_mappings(
            pa.RecordBatch.from_pydict(
                {
                    "square_002um": ["a", "b"],
                    "cell_id": [None, "cell"],
                    "in_cell": [False, True],
                    "in_nucleus": [False, True],
                }
            ),
            tmp_path,
            context=_CONTEXT,
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
            validate_mappings(pa.RecordBatch.from_pydict(data), tmp_path, context=_CONTEXT)
