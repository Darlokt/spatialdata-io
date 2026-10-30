"""MCMICRO marker naming, MCQuant column roles, and exact value decoding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.mcmicro._schemas import (
    CELL_ID_DTYPE,
    FLOAT_DTYPE,
    STRING_DTYPE,
    canonical_markers,
    cell_ids,
    float_block,
    float_values,
    normalize_markers,
    optional_float_values,
    quantification_columns,
    validate_columns,
    validate_position_columns,
)

if TYPE_CHECKING:
    from pathlib import Path


def _path(tmp_path: Path) -> ArtifactFile:
    return ArtifactFile(tmp_path / "source.csv")


def _markers(**columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(values, dtype=STRING_DTYPE) for name, values in columns.items()})


class TestValidateColumns:
    """Duplicate and empty names fail before any parser can repair them."""

    @pytest.mark.parametrize("columns", [[], ["a", "a"], ["a", ""], ["a\x00"]])
    def test_rejects(self, tmp_path: Path, columns: list[str]) -> None:
        with pytest.raises(ReaderFormatError, match="unique nonempty"):
            validate_columns(columns, _path(tmp_path))


class TestCanonicalMarkers:
    """Every occurrence of a repeated marker receives MCQuant's one-based suffix."""

    def test_mcquant_suffixes(self) -> None:
        assert canonical_markers(["DAPI", "CD3", "DAPI", "CD8"]) == ("DAPI_1", "CD3", "DAPI_2", "CD8")


class TestNormalizeMarkers:
    """Panel order is authoritative and optional columns remain exact text."""

    def test_marker_only_and_optional_text(self, tmp_path: Path) -> None:
        panel = normalize_markers(
            _markers(cycle=["1", "01", "2"], marker_name=["DAPI", "CD3", "DAPI"]), _path(tmp_path)
        )
        assert panel.features == ("DAPI_1", "CD3", "DAPI_2")
        assert panel.var.index.tolist() == ["DAPI_1", "CD3", "DAPI_2"]
        assert panel.var["marker_name"].tolist() == ["DAPI", "CD3", "DAPI"]
        assert panel.var["cycle"].tolist() == ["1", "01", "2"]
        assert panel.var["cycle"].dtype == STRING_DTYPE

    @pytest.mark.parametrize(
        ("frame", "match"),
        [
            (_markers(name=["A"]), "marker_name"),
            (_markers(marker_name=[]), "no markers"),
            (_markers(marker_name=["A", ""]), "empty or NUL"),
            (_markers(marker_name=["A\x00"]), "empty or NUL"),
            (_markers(marker_name=["X", "X", "X_1"]), "generated collisions"),
        ],
    )
    def test_rejects(self, tmp_path: Path, frame: pd.DataFrame, match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            normalize_markers(frame, _path(tmp_path))


class TestQuantificationColumns:
    """Expression is exactly the marker panel; morphology and text keep source order."""

    def test_roles(self, tmp_path: Path) -> None:
        columns = quantification_columns(
            ["CellID", "B", "note", "A", "X_centroid", "Y_centroid", "Area", "A_intensity_max"],
            ("A", "B"),
            _path(tmp_path),
        )
        assert columns.features == ("A", "B")
        assert columns.morphology == ("X_centroid", "Y_centroid", "Area")
        assert columns.text == ("note", "A_intensity_max")

    @pytest.mark.parametrize(
        ("header", "match"),
        [
            (["CellID", "A", "X_centroid"], "missing columns"),
            (["CellID", "X_centroid", "Y_centroid"], "missing columns"),
            (["CellID", "A", "X_centroid", "Y_centroid", "region"], "linkage column"),
            (["CellID", "A", "A", "X_centroid", "Y_centroid"], "unique nonempty"),
        ],
    )
    def test_rejects(self, tmp_path: Path, header: list[str], match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            quantification_columns(header, ("A",), _path(tmp_path))


class TestFloatValues:
    """Values are correctly rounded, finite, and never silently underflow."""

    def test_correctly_rounded_against_lexical_oracle(self, tmp_path: Path) -> None:
        tokens = ["0.1", "2.675", "1e-320", "9007199254740993", "0.30000000000000004", "-0", "1.7976931348623157e308"]
        np.testing.assert_array_equal(
            float_values(pa.array(tokens), _path(tmp_path), "A"), np.array([float(token) for token in tokens])
        )

    @pytest.mark.parametrize(
        ("token", "match"),
        [
            ("inf", "nonfinite"),
            ("nan", "nonfinite"),
            ("1e400", "nonfinite|nonnumeric"),
            ("abc", "nonnumeric"),
            ("", "nonnumeric"),
            ("1e-400", "underflow"),
        ],
    )
    def test_rejects(self, tmp_path: Path, token: str, match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            float_values(pa.array(["1", token]), _path(tmp_path), "A")

    def test_rejects_null(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="missing"):
            float_values(pa.array(["1", None]), _path(tmp_path), "A")

    @pytest.mark.parametrize("zero", ["0", "0.0", "-0.000", ".0", "0e5", "+0E-3"])
    def test_literal_zero_spellings(self, tmp_path: Path, zero: str) -> None:
        assert float_values(pa.array([zero]), _path(tmp_path), "A").tolist() == [0.0]


class TestFloatBlock:
    """A whole bounded block decodes in one pass and errors still name their marker."""

    def test_rows_by_markers(self, tmp_path: Path) -> None:
        block = float_block([pa.array(["1", "0"]), pa.array(["0.5", "2e-3"])], ("A", "B"), _path(tmp_path))
        np.testing.assert_array_equal(block, [[1, 0.5], [0, 0.002]])

    def test_error_names_the_marker(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="nonfinite 'B'"):
            float_block([pa.array(["1"]), pa.array(["inf"])], ("A", "B"), _path(tmp_path))


class TestOptionalFloatValues:
    """Only empty fields become missing; any other invalid value fails."""

    def test_empty_is_missing(self, tmp_path: Path) -> None:
        values = optional_float_values(pa.chunked_array([["1.5", ""], ["2"]]), _path(tmp_path), "Area")
        assert values.dtype == FLOAT_DTYPE
        assert values.isna().tolist() == [False, True, False]
        assert values.dropna().tolist() == [1.5, 2.0]

    def test_all_missing(self, tmp_path: Path) -> None:
        assert optional_float_values(pa.array(["", ""]), _path(tmp_path), "Area").isna().all()

    def test_invalid_fails(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="nonnumeric"):
            optional_float_values(pa.array(["1", "NA"]), _path(tmp_path), "Area")


class TestCellIds:
    """Label IDs are exact positive integers that fit their segmentation dtype."""

    def test_exact_values(self, tmp_path: Path) -> None:
        identifiers = cell_ids(pa.array(["3", "9007199254740993"]), np.dtype(np.uint64), _path(tmp_path))
        assert identifiers.dtype == CELL_ID_DTYPE
        assert identifiers.tolist() == [3, 9007199254740993]

    @pytest.mark.parametrize(
        ("tokens", "dtype", "match"),
        [
            (["1", None], np.uint32, "missing"),
            (["1", "2.0"], np.uint32, "non-integer"),
            (["1", "x"], np.uint32, "non-integer"),
            (["1", "99999999999999999999"], np.uint64, "non-integer"),
            (["0"], np.uint32, "outside"),
            (["-1"], np.int32, "outside"),
            (["1", "1"], np.uint32, "outside"),
            (["256"], np.uint8, "outside"),
        ],
    )
    def test_rejects(self, tmp_path: Path, tokens: list[str | None], dtype: type[np.integer], match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            cell_ids(pa.array(tokens, pa.string()), np.dtype(dtype), _path(tmp_path))


class TestValidatePositionColumns:
    @pytest.mark.parametrize("name", ["spatial_frame", "spatial_pixel_frame"])
    def test_generated_frame_collisions(self, tmp_path: Path, name: str) -> None:
        columns = quantification_columns(["CellID", "A", "X_centroid", "Y_centroid", name], ("A",), _path(tmp_path))
        with pytest.raises(ReaderFormatError, match="existing spatial_frame"):
            validate_position_columns(columns, multiple=True, path=_path(tmp_path))
        validate_position_columns(columns, multiple=False, path=_path(tmp_path))
