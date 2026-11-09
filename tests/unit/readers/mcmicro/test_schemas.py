"""MCMICRO marker naming, MCQuant column roles, and exact value decoding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.mcmicro._schemas import (
    CELL_ID_DTYPE,
    FLOAT_DTYPE,
    MARKERS_CONTEXT,
    QUANTIFICATION_CONTEXT,
    STRING_DTYPE,
    canonical_markers,
    cell_ids,
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
        with pytest.raises(ReaderFormatError, match="unique nonempty") as caught:
            validate_columns(columns, _path(tmp_path), context=MARKERS_CONTEXT)
        assert caught.value.context == MARKERS_CONTEXT

    def test_accepts_unique_names(self, tmp_path: Path) -> None:
        validate_columns(["a", "b"], _path(tmp_path), context=QUANTIFICATION_CONTEXT)


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
        with pytest.raises(ReaderFormatError, match=match) as caught:
            normalize_markers(frame, _path(tmp_path))
        assert caught.value.context == ReaderErrorContext("mcmicro", "markers")


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

    @pytest.mark.parametrize("feature", ["CellID", "X_centroid", "Y_centroid"])
    def test_required_roles_cannot_also_be_expression(self, tmp_path: Path, feature: str) -> None:
        path = _path(tmp_path)
        with pytest.raises(ReaderFormatError) as caught:
            quantification_columns(["CellID", "X_centroid", "Y_centroid"], (feature,), path)
        assert caught.value.context.reader == "mcmicro"
        assert caught.value.path == path
        assert feature in str(caught.value)

    @pytest.mark.parametrize("name", ["CellID", "X_centroid", "Y_centroid"])
    def test_suffixed_required_names_are_unambiguous(self, tmp_path: Path, name: str) -> None:
        features = canonical_markers([name, name])
        columns = quantification_columns(["CellID", *features, "X_centroid", "Y_centroid"], features, _path(tmp_path))
        assert columns.features == (f"{name}_1", f"{name}_2")
        assert columns.morphology == ("X_centroid", "Y_centroid")

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
        with pytest.raises(ReaderFormatError, match=match) as caught:
            quantification_columns(header, ("A",), _path(tmp_path))
        assert caught.value.context == ReaderErrorContext("mcmicro", "quantification")


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
        identifiers = cell_ids(pa.array(["3", "9007199254740993", "2.0", "4e0"]), np.dtype(np.uint64), _path(tmp_path))
        assert identifiers.dtype == CELL_ID_DTYPE
        assert identifiers.tolist() == [3, 9007199254740993, 2, 4]

    def test_duplicate_ids_are_preserved_for_final_table_validation(self, tmp_path: Path) -> None:
        identifiers = cell_ids(pa.array(["3", "3"]), np.dtype(np.uint32), _path(tmp_path))
        assert identifiers.tolist() == [3, 3]
        assert identifiers.dtype == CELL_ID_DTYPE

    @pytest.mark.parametrize(
        ("tokens", "dtype", "found"),
        [
            (["1", None], np.uint32, "missing 'CellID' values"),
            (["1", "2.5"], np.uint32, "'CellID' value '2.5'"),
            (["1", "x"], np.uint32, "'CellID' value 'x'"),
            (["1", "0x10"], np.uint32, "'CellID' value '0x10'"),
            (["1", "99999999999999999999"], np.uint64, "'CellID' value '99999999999999999999'"),
            (["0"], np.uint32, "'CellID' value '0'"),
            (["-1"], np.int32, "'CellID' value '-1'"),
            (["256"], np.uint8, "'CellID' value 256 above 255"),
            ([str(2**63)], np.uint64, f"'CellID' value {2**63} above {2**63 - 1}"),
        ],
    )
    def test_rejects(self, tmp_path: Path, tokens: list[str | None], dtype: type[np.integer], found: str) -> None:
        with pytest.raises(ReaderFormatError) as caught:
            cell_ids(pa.array(tokens, pa.string()), np.dtype(dtype), _path(tmp_path))
        assert caught.value.found == found
        assert caught.value.context == QUANTIFICATION_CONTEXT


class TestValidatePositionColumns:
    @pytest.mark.parametrize("name", ["spatial_frame", "spatial_pixel_frame"])
    def test_generated_frame_collisions(self, tmp_path: Path, name: str) -> None:
        columns = quantification_columns(["CellID", "A", "X_centroid", "Y_centroid", name], ("A",), _path(tmp_path))
        with pytest.raises(ReaderFormatError, match="existing spatial_frame"):
            validate_position_columns(columns, multiple=True, path=_path(tmp_path))
        validate_position_columns(columns, multiple=False, path=_path(tmp_path))
