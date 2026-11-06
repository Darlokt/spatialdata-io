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
    cell_ids,
    csv_column_names,
    expression_dtype,
    normalize_csv_numbers,
    normalize_fcs_metadata,
    region_name,
    validate_csv_fields,
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


class TestCellIds:
    """Preserve exact floating source integers at the FCS boundary."""

    def test_large_stored_float_is_not_shortened_decimal_string(self, tmp_path: Path) -> None:
        """Preserve the exact binary FCS identifier without decimal-string shortening."""
        result = cell_ids(pd.Series([float(2**63)]), ArtifactFile(tmp_path / "cells.fcs"))

        assert result.tolist() == [2**63]
        assert result.dtype == np.dtype(np.uint64)

    @pytest.mark.parametrize("tail", ["2.0", "2e0"])
    def test_decimal_lexeme_does_not_round_another_identifier(self, tmp_path: Path, tail: str) -> None:
        """Normalize decimal ID tokens without rounding another large lexical identifier."""
        result = cell_ids(pd.Series([str(2**53 + 1), tail], dtype=STRING_DTYPE), ArtifactFile(tmp_path / "cells.csv"))

        np.testing.assert_array_equal(result, [2**53 + 1, 2])

    @pytest.mark.parametrize("value", ["1.00000000000000000001", "1e-1000", None])
    def test_fractional_or_missing_identifier(self, tmp_path: Path, value: str | None) -> None:
        """Reject fractional and missing IDs before canonical integer construction."""
        with pytest.raises(ReaderFormatError, match="cell_id"):
            cell_ids(pd.Series([value], dtype=STRING_DTYPE), ArtifactFile(tmp_path / "cells.csv"))


class TestNormalizeCsvNumbers:
    """Keep lexical source identifiers and exact numerical values in explicit Arrow fields."""

    def test_typed_primitive_columns(self, tmp_path: Path) -> None:
        """Preserve exact numerical values and missingness in declared Arrow primitive fields."""
        frame = pd.DataFrame(
            {
                "cell_id": ["002", str(2**53 + 1)],
                "integer": [str(2**53 + 1), "0"],
                "float": ["0.10000000000000002", "-0.00e1000"],
                "optional": ["0.25", ""],
                "text": ["NULL", "NA"],
            },
            dtype=STRING_DTYPE,
        )

        normalize_csv_numbers(frame, ["integer", "float", "optional"], ArtifactFile(tmp_path / "cells.csv"))

        assert frame["cell_id"].tolist() == ["002", str(2**53 + 1)]
        assert frame["integer"].tolist() == [2**53 + 1, 0]
        assert frame["integer"].dtype == pd.ArrowDtype(pa.int64())
        np.testing.assert_array_equal(frame["float"].to_numpy(), [float("0.10000000000000002"), -0.0])
        assert frame["float"].dtype == pd.ArrowDtype(pa.float64())
        assert frame["optional"].isna().tolist() == [False, True]
        assert frame["text"].tolist() == ["NULL", "NA"]

    @pytest.mark.parametrize("token", ["9007199254740993.0", "1e-1000", "1e999"])
    def test_numeric_precision_failure(self, tmp_path: Path, token: str) -> None:
        """Reject lossy large values, floating underflow, and finite-token overflow."""
        with pytest.raises(ReaderFormatError, match="precision"):
            normalize_csv_numbers(
                pd.DataFrame({"value": [token]}, dtype=STRING_DTYPE), ["value"], ArtifactFile(tmp_path / "cells.csv")
            )


class TestValidateCsvFields:
    """Validate every original lexical field before semantic conversion."""

    @pytest.mark.parametrize("column", ["cell_id", "region", "x", "y", "cyc001_ch001", "note"])
    def test_nul_cannot_truncate_source_text(self, tmp_path: Path, column: str) -> None:
        """Reject NULs in identifiers, scientific values, and unknown annotations."""
        frame = pd.DataFrame({column: ["1\x00suffix"]}, dtype=STRING_DTYPE)
        with pytest.raises(ReaderFormatError, match="NUL"):
            validate_csv_fields(frame, ArtifactFile(tmp_path / "cells.csv"))

    def test_physically_missing_field_is_rejected(self, tmp_path: Path) -> None:
        """A missing CSV field remains distinct from an explicitly empty string."""
        frame = pd.DataFrame({"note": ["", None]}, dtype=STRING_DTYPE)
        with pytest.raises(ReaderFormatError, match="row width"):
            validate_csv_fields(frame, ArtifactFile(tmp_path / "cells.csv"))

    def test_literal_text_and_empty_fields_are_unchanged(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"note": ["0042", "", "NA", "NULL", "1e-400"]}, dtype=STRING_DTYPE)
        original = frame.copy()

        validate_csv_fields(frame, ArtifactFile(tmp_path / "cells.csv"))

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
