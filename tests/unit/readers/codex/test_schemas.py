"""CODEX named columns, dynamic expression features, and exact identifier schemas."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.codex._schemas import (
    STRING_DTYPE,
    cell_columns,
    check_integer_promotion,
    csv_column_names,
    csv_types,
    expression_dtype,
    fcs_cell_ids,
    normalize_fcs_metadata,
    region_name,
    validate_csv_annotations,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestCellColumns:
    """Keep vendor cycle channels in source order and metadata separate."""

    @pytest.mark.parametrize(
        "columns",
        [[], ["", "x"], ["x", "x"], ["null\x00column"], ["cell_id", "region"], ["cell_id", "region", "x", "y"]],
    )
    def test_invalid_headers(self, tmp_path: Path, columns: list[str]) -> None:
        """Reject missing required fields and ambiguous source parameter names."""
        with pytest.raises(ReaderFormatError):
            cell_columns(columns, ArtifactFile(tmp_path / "cells.csv"))

    def test_cycle_substring_in_metadata_is_not_expression(self, tmp_path: Path) -> None:
        """Require complete cycle-channel names when identifying expression fields."""
        features, metadata = cell_columns(
            ["cell_id", "region", "x", "y", "cyc002_ch003", "recycle_score"], ArtifactFile(tmp_path / "cells.csv")
        )

        assert features == ["cyc002_ch003"]
        assert metadata == ["cell_id", "region", "x", "y", "recycle_score"]


class TestCheckIntegerPromotion:
    @pytest.mark.parametrize("dtype", [np.int64, np.uint64])
    def test_large_exact_power_preserves_native_values(self, tmp_path: Path, dtype: type[np.integer]) -> None:
        check_integer_promotion(np.array([2**54], dtype=dtype), ArtifactFile(tmp_path / "cells.fcs"))


class TestFcsCellIds:
    """Preserve native integer buffers and the exact stored binary FCS float."""

    @pytest.mark.parametrize("values", [[float(2**63)], [float(2**63 + 2048)]])
    def test_stored_float_is_exact(self, tmp_path: Path, values: list[float]) -> None:
        result = fcs_cell_ids(pd.Series(values), ArtifactFile(tmp_path / "cells.fcs"))
        assert result.tolist() == [int(value) for value in values]
        assert result.dtype == np.dtype(np.uint64)

    @pytest.mark.parametrize("dtype", ["int32", "int64", "uint32", "uint64"])
    def test_native_integer_ids(self, tmp_path: Path, dtype: str) -> None:
        result = fcs_cell_ids(pd.Series([0, 2, 11], dtype=dtype), ArtifactFile(tmp_path / "cells.fcs"))
        assert result.tolist() == [0, 2, 11]
        assert result.dtype == np.dtype(np.int64)

    def test_native_uint64_maximum(self, tmp_path: Path) -> None:
        result = fcs_cell_ids(pd.Series([2**64 - 1], dtype="uint64"), ArtifactFile(tmp_path / "cells.fcs"))
        assert result.tolist() == [2**64 - 1]

    @pytest.mark.parametrize(
        "values", [[0.5], [float("inf")], [float("nan")], [float(2**64)], [-float(2**64)], [-1.0, float(2**63)]]
    )
    def test_invalid_native_ids(self, tmp_path: Path, values: list[float]) -> None:
        with pytest.raises(ReaderFormatError, match=r"cell_id|overflow"):
            fcs_cell_ids(pd.Series(values), ArtifactFile(tmp_path / "cells.fcs"))

    def test_signed_native_ids_are_preserved(self, tmp_path: Path) -> None:
        result = fcs_cell_ids(pd.Series([-2.0, 0.0, 1.0]), ArtifactFile(tmp_path / "cells.fcs"))
        assert result.tolist() == [-2, 0, 1]


class TestCsvTypes:
    def test_fixed_roles_and_order(self) -> None:
        columns = ["note", "x", "cyc002_ch001", "cell_id", "region", "y"]
        assert csv_types(columns, ["cyc002_ch001"]) == pa.schema(
            [
                ("note", pa.string()),
                ("x", pa.float64()),
                ("cyc002_ch001", pa.float64()),
                ("cell_id", pa.uint64()),
                ("region", pa.string()),
                ("y", pa.float64()),
            ]
        )


class TestValidateCsvAnnotations:
    @pytest.mark.parametrize("column", ["region", "note"])
    def test_nul_text_fails(self, tmp_path: Path, column: str) -> None:
        with pytest.raises(ReaderFormatError, match="NUL"):
            validate_csv_annotations(
                pd.DataFrame({column: ["1\x00suffix"]}, dtype=STRING_DTYPE), ArtifactFile(tmp_path / "cells.csv")
            )

    def test_literal_text_is_unchanged(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"note": ["0042", "", "NA", "NULL", "1e-400"]}, dtype=STRING_DTYPE)
        original = frame.copy()
        validate_csv_annotations(frame, ArtifactFile(tmp_path / "cells.csv"))
        pd.testing.assert_frame_equal(frame, original, check_exact=True)


class TestNormalizeFcsMetadata:
    """Preserve FCS annotation widths and reject nonnumeric metadata parameters."""

    def test_float32_parameters(self, tmp_path: Path) -> None:
        """Retain the source FCS floating width and compensated values."""
        values = np.asarray([0.125, -1.5], dtype=np.float32)
        frame = pd.DataFrame({"x": values})

        normalize_fcs_metadata(frame, ArtifactFile(tmp_path / "cells.fcs"))

        assert frame["x"].dtype == pd.ArrowDtype(pa.float32())
        np.testing.assert_array_equal(frame["x"].to_numpy(), values)

    def test_nonnumeric_parameter(self, tmp_path: Path) -> None:
        """Reject nonnumeric FCS event parameters before numerical assembly."""
        with pytest.raises(ReaderFormatError, match="FCS parameter"):
            normalize_fcs_metadata(pd.DataFrame({"cell_id": ["invalid"]}), ArtifactFile(tmp_path / "cells.fcs"))


class TestExpressionDtype:
    """Choose the declared real numerical dtype at the bounded sparse kernel boundary."""

    def test_mixed_integer_and_float_features(self, tmp_path: Path) -> None:
        """Choose a common NumPy kernel dtype for typed Arrow features."""
        frame = pd.DataFrame(
            {"first": pd.Series([1], dtype="int64[pyarrow]"), "second": pd.Series([0.5], dtype="float32[pyarrow]")}
        )

        assert expression_dtype(frame, ["first", "second"], ArtifactFile(tmp_path / "cells.csv")) == np.dtype(
            np.float64
        )


class TestRegionName:
    """Reject unsupported scalar region types explicitly."""

    @pytest.mark.parametrize("value", [True, False, np.bool_(1), "", complex(1)])
    def test_invalid_region_scalar(self, tmp_path: Path, *, value: bool | str | complex) -> None:
        """Reject unsupported region values before element naming and linkage."""
        with pytest.raises(ReaderFormatError, match="region"):
            region_name(pd.Series([value]), ArtifactFile(tmp_path / "cells.csv"))

    @pytest.mark.parametrize("value", [0, -1, 1.5, float("inf"), float("nan")])
    def test_invalid_native_numeric_regions(self, tmp_path: Path, value: float) -> None:
        with pytest.raises(ReaderFormatError, match="region"):
            region_name(pd.Series([value]), ArtifactFile(tmp_path / "cells.fcs"))

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(1, "1"), (2.0, "2"), (np.int64(3), "3"), ("reg001", "reg001"), ("é-1.a_b", "é-1.a_b"), ("tables", "tables")],
    )
    def test_valid_region(self, tmp_path: Path, value: object, expected: str) -> None:
        assert region_name(pd.Series([value, value]), ArtifactFile(tmp_path / "cells.csv")) == expected

    @pytest.mark.parametrize(
        "value", [".", "..", "__reg", "my region", "a:b", "reg/1", "table", "TABLE", "images", "Images", "iMaGeS"]
    )
    def test_region_must_be_an_element_name(self, tmp_path: Path, value: str) -> None:
        """The region names the shapes element, so it must not fail SpatialData construction later."""
        with pytest.raises(ReaderFormatError, match=r"artifact 'cell table'.*valid element name") as caught:
            region_name(pd.Series([value]), ArtifactFile(tmp_path / "cells.csv"))

        assert isinstance(caught.value.__cause__, ValueError)


class TestCsvColumnNames:
    """Separate Akoya source field identifiers from their long labels."""

    @pytest.mark.parametrize("columns", [["cell_id", "region", "x", "y", "note:extra"], [], ["cell_id:id", "x"]])
    def test_plain_or_incomplete_header(self, tmp_path: Path, columns: list[str]) -> None:
        assert csv_column_names(columns, ArtifactFile(tmp_path / "cells.csv")) == (columns, None)

    def test_first_colon_and_order(self, tmp_path: Path) -> None:
        names, labels = csv_column_names(
            ["y:y", "cyc001_ch001:marker:variant", "x:x"], ArtifactFile(tmp_path / "cells.csv")
        )
        assert names == ["y", "cyc001_ch001", "x"]
        assert labels == ["y", "marker:variant", "x"]

    @pytest.mark.parametrize("label", ["", "nul\x00"])
    def test_invalid_label(self, tmp_path: Path, label: str) -> None:
        with pytest.raises(ReaderFormatError, match="long label"):
            csv_column_names([f"x:{label}"], ArtifactFile(tmp_path / "cells.csv"))
