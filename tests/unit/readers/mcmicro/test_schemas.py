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
    MARKERS_CONTEXT,
    QUANTIFICATION_CONTEXT,
    STRING_DTYPE,
    canonical_markers,
    normalize_markers,
    quantification_columns,
    validate_columns,
    validate_expression,
    validate_measurements,
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
        assert columns.types == pa.schema(
            [
                ("CellID", pa.uint64()),
                ("B", pa.float64()),
                ("note", pa.string()),
                ("A", pa.float64()),
                ("X_centroid", pa.float64()),
                ("Y_centroid", pa.float64()),
                ("Area", pa.float64()),
                ("A_intensity_max", pa.string()),
            ]
        )

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


class TestValidateMeasurements:
    """Typed values retain required fields, optional nulls and target-specific ID limits."""

    @pytest.mark.parametrize("identifier", [1, 2**53 + 1, 2**63, 2**64 - 1])
    def test_full_uint64_ids_and_optional_nulls(self, tmp_path: Path, identifier: int) -> None:
        path = _path(tmp_path)
        columns = quantification_columns(["CellID", "A", "X_centroid", "Y_centroid", "Area"], ("A",), path)
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array([identifier, identifier], pa.uint64()),
                *[pa.array([1.0, -0.25])] * 3,
                pa.array([None, None], pa.float64()),
            ],
            schema=columns.types,
        )
        validate_measurements(batch, columns, np.dtype(np.uint64), path)
        assert batch.column("CellID").to_pylist() == [identifier, identifier]

    @pytest.mark.parametrize("name", ["CellID", "A", "X_centroid", "Y_centroid"])
    def test_required_nulls(self, tmp_path: Path, name: str) -> None:
        path = _path(tmp_path)
        columns = quantification_columns(["CellID", "A", "X_centroid", "Y_centroid"], ("A",), path)
        batch = pa.RecordBatch.from_arrays(
            [pa.array([None if field.name == name else 1], field.type) for field in columns.types],
            schema=columns.types,
        )
        with pytest.raises(ReaderFormatError, match=f"missing '{name}'") as caught:
            validate_measurements(batch, columns, np.dtype(np.uint64), path)
        assert caught.value.context == QUANTIFICATION_CONTEXT
        assert caught.value.path == path

    @pytest.mark.parametrize(
        ("identifier", "dtype", "message"),
        [(0, np.uint64, "zero CellID"), (256, np.uint8, "above 255"), (2**63, np.int64, f"above {2**63 - 1}")],
    )
    def test_positive_target_range(
        self, tmp_path: Path, identifier: int, dtype: type[np.integer], message: str
    ) -> None:
        path = _path(tmp_path)
        columns = quantification_columns(["CellID", "A", "X_centroid", "Y_centroid"], ("A",), path)
        batch = pa.RecordBatch.from_arrays(
            [pa.array([identifier], pa.uint64()), *[pa.array([1.0])] * 3], schema=columns.types
        )
        with pytest.raises(ReaderFormatError, match=message):
            validate_measurements(batch, columns, np.dtype(dtype), path)

    @pytest.mark.parametrize("name", ["X_centroid", "Y_centroid", "Area"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_nonfinite_morphology(self, tmp_path: Path, name: str, value: float) -> None:
        path = _path(tmp_path)
        columns = quantification_columns(["CellID", "A", "X_centroid", "Y_centroid", "Area"], ("A",), path)
        batch = pa.RecordBatch.from_arrays(
            [pa.array([value if field.name == name else 1], field.type) for field in columns.types],
            schema=columns.types,
        )
        with pytest.raises(ReaderFormatError, match=f"nonfinite '{name}'"):
            validate_measurements(batch, columns, np.dtype(np.uint64), path)


class TestValidateExpression:
    """Signed finite intensities are valid; nonfinite values fail with source context."""

    def test_finite_signed_values(self, tmp_path: Path) -> None:
        validate_expression(np.array([[0, -0.25, 1e100]], dtype=np.float64), _path(tmp_path))

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_nonfinite_values(self, tmp_path: Path, value: float) -> None:
        with pytest.raises(ReaderFormatError, match="nonfinite marker") as caught:
            validate_expression(np.array([[value]], dtype=np.float64), _path(tmp_path))
        assert caught.value.context == QUANTIFICATION_CONTEXT
        assert caught.value.path == _path(tmp_path)


class TestValidatePositionColumns:
    @pytest.mark.parametrize("name", ["spatial_frame", "spatial_pixel_frame"])
    def test_generated_frame_collisions(self, tmp_path: Path, name: str) -> None:
        columns = quantification_columns(["CellID", "A", "X_centroid", "Y_centroid", name], ("A",), _path(tmp_path))
        with pytest.raises(ReaderFormatError, match="existing spatial_frame"):
            validate_position_columns(columns, multiple=True, path=_path(tmp_path))
        validate_position_columns(columns, multiple=False, path=_path(tmp_path))
