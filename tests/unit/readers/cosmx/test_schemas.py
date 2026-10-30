"""CosMx column roles, header variants, and exact lexical value decoding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.cosmx._schemas import (
    FLOAT_DTYPE,
    count_block,
    count_features,
    float_values,
    fov_position_columns,
    half_units,
    integer_values,
    metadata_columns,
    optional_float_values,
    transcript_columns,
    validate_columns,
    validate_label_ids,
)

if TYPE_CHECKING:
    from pathlib import Path

_CENTERS = ["CenterX_local_px", "CenterY_local_px", "CenterX_global_px", "CenterY_global_px"]


def _path(tmp_path: Path) -> ArtifactFile:
    return ArtifactFile(tmp_path / "source.csv")


class TestValidateColumns:
    """Duplicate and empty names fail before any parser can repair them."""

    @pytest.mark.parametrize("columns", [[], ["a", "a"], ["a", ""], ["a\x00"]])
    def test_rejects(self, tmp_path: Path, columns: list[str]) -> None:
        with pytest.raises(ReaderFormatError, match="unique nonempty"):
            validate_columns(columns, _path(tmp_path))


class TestValidateLabelIds:
    """The inspected label representation constrains linked positive cell IDs."""

    @pytest.mark.parametrize("dtype", ["uint8", "int8", "uint16", "int16", "uint32", "int32"])
    def test_exact_limit_and_overflow(self, tmp_path: Path, dtype: str) -> None:
        label_dtype = np.dtype(dtype)
        limit = np.iinfo(label_dtype).max
        validate_label_ids(np.array([1, limit], np.int64), label_dtype, _path(tmp_path))
        with pytest.raises(ReaderFormatError, match=r"outside.*label range"):
            validate_label_ids(np.array([limit + 1], np.int64), label_dtype, _path(tmp_path))

    @pytest.mark.parametrize("dtype", ["uint64", "int64"])
    def test_64_bit_and_empty(self, tmp_path: Path, dtype: str) -> None:
        validate_label_ids(np.array([2**63 - 1], np.int64), np.dtype(dtype), _path(tmp_path))
        validate_label_ids(np.array([], np.int64), np.dtype(dtype), _path(tmp_path))


class TestCountFeatures:
    """Every non-identifier field is expression, including controls, in source order."""

    def test_order_and_controls(self, tmp_path: Path) -> None:
        columns = ["fov", "HLA-A", "cell_ID", "NegPrb1", "SystemControl1"]
        assert count_features(columns, _path(tmp_path)) == ("HLA-A", "NegPrb1", "SystemControl1")

    @pytest.mark.parametrize(("columns", "match"), [(["fov", "A"], "missing"), (["fov", "cell_ID"], "no count")])
    def test_rejects(self, tmp_path: Path, columns: list[str], match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            count_features(columns, _path(tmp_path))


class TestMetadataColumns:
    """Vendor morphology and intensities are typed; other annotations stay text."""

    def test_roles(self, tmp_path: Path) -> None:
        columns = metadata_columns(
            ["cell", "fov", "cell_ID", *_CENTERS, "Area", "Area.um2", "Mean.DAPI", "Max.G", "Width", "version"],
            _path(tmp_path),
        )
        assert columns.numeric == ("Area", "Area.um2", "Mean.DAPI", "Max.G", "Width")
        assert columns.text == ("cell", "version")

    @pytest.mark.parametrize(
        ("columns", "match"),
        [(["fov", "cell_ID", *_CENTERS[:3]], "missing"), (["fov", "cell_ID", *_CENTERS, "region"], "linkage")],
    )
    def test_rejects(self, tmp_path: Path, columns: list[str], match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            metadata_columns(columns, _path(tmp_path))


class TestFovPositionColumns:
    """Both pixel conventions are recognized; millimetre-only files provide none."""

    @pytest.mark.parametrize(
        ("columns", "expected"),
        [
            (["fov", "x_px", "y_px"], ("fov", "x_px", "y_px")),
            (
                ["FOV", "x_global_px", "y_global_px", "x_global_mm", "y_global_mm"],
                ("FOV", "x_global_px", "y_global_px"),
            ),
            (["Slide", "X_mm", "Y_mm", "FOV"], None),
        ],
    )
    def test_conventions(self, tmp_path: Path, columns: list[str], expected: tuple[str, ...] | None) -> None:
        assert fov_position_columns(columns, _path(tmp_path)) == expected

    @pytest.mark.parametrize(
        "columns",
        [["x_px", "y_px"], ["fov", "FOV", "x_px", "y_px"], ["fov", "x_px", "y_px", "x_global_px", "y_global_px"]],
    )
    def test_rejects(self, tmp_path: Path, columns: list[str]) -> None:
        with pytest.raises(ReaderFormatError):
            fov_position_columns(columns, _path(tmp_path))


class TestTranscriptColumns:
    """Required event fields and supported annotations are projected."""

    def test_projection(self, tmp_path: Path) -> None:
        names = ["fov", "cell_ID", "cell", "x_local_px", "y_local_px", "x_global_px", "y_global_px", "z", "target", "x"]
        assert transcript_columns(names, _path(tmp_path)) == (
            "fov", "cell_ID", "x_local_px", "y_local_px", "x_global_px", "y_global_px", "target", "z", "cell"
        )  # fmt: skip

    def test_missing_fields(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="missing"):
            transcript_columns(["fov", "target"], _path(tmp_path))


class TestIntegerValues:
    """Identifiers are exact integers, optionally bounded below."""

    def test_exact(self, tmp_path: Path) -> None:
        values = integer_values(pa.array(["9007199254740993", "-1"]), _path(tmp_path), "z", minimum=None)
        assert values.tolist() == [9007199254740993, -1]

    @pytest.mark.parametrize(("token", "match"), [(None, "missing"), ("1.0", "non-integer"), ("0", "below 1")])
    def test_rejects(self, tmp_path: Path, token: str | None, match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            integer_values(pa.array(["1", token], pa.string()), _path(tmp_path), "fov", minimum=1)


class TestCountBlock:
    """Count blocks decode in one pass and errors name their feature."""

    def test_rows_by_features(self, tmp_path: Path) -> None:
        block = count_block([pa.array(["1", "0"]), pa.array(["0", "5"])], ("A", "B"), _path(tmp_path))
        np.testing.assert_array_equal(block, [[1, 0], [0, 5]])

    @pytest.mark.parametrize(("token", "match"), [("1.5", "non-integer 'B'"), ("-1", "'B' values below 0")])
    def test_error_names_the_feature(self, tmp_path: Path, token: str, match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            count_block([pa.array(["1"]), pa.array([token])], ("A", "B"), _path(tmp_path))


class TestFloatValues:
    """Coordinates are correctly rounded, finite, and never silently underflow."""

    def test_correctly_rounded_against_lexical_oracle(self, tmp_path: Path) -> None:
        tokens = ["162613.4", "159172.222222222", "0.1", "1e-320", "-0"]
        np.testing.assert_array_equal(
            float_values(pa.array(tokens), _path(tmp_path), "x"), [float(token) for token in tokens]
        )

    @pytest.mark.parametrize(
        ("token", "match"),
        [("inf", "nonfinite"), ("NA", "nonnumeric"), ("1e-400", "underflow")],
    )
    def test_rejects(self, tmp_path: Path, token: str, match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            float_values(pa.array(["1", token]), _path(tmp_path), "x")


class TestOptionalFloatValues:
    """Empty fields and the vendor NA spelling are missing; other text fails."""

    def test_missing_spellings(self, tmp_path: Path) -> None:
        values = optional_float_values(pa.array(["1.5", "", "NA"]), _path(tmp_path), "Area")
        assert values.dtype == FLOAT_DTYPE
        assert values.isna().tolist() == [False, True, True]

    def test_all_missing(self, tmp_path: Path) -> None:
        assert optional_float_values(pa.array(["", "NA"]), _path(tmp_path), "Area").isna().all()

    def test_invalid_text(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="nonnumeric"):
            optional_float_values(pa.array(["1", "x"]), _path(tmp_path), "Area")


class TestHalfUnits:
    """Each bound is half a unit in the last printed decimal place."""

    def test_bounds(self, tmp_path: Path) -> None:
        tokens = pa.array(["806", "159172.222222222", "5.", ".25", "1.5e3", "-2E-2", "+7"])
        np.testing.assert_allclose(
            half_units(tokens, _path(tmp_path), "x"), [0.5, 5e-10, 0.5, 5e-3, 50.0, 5e-3, 0.5], rtol=0, atol=0
        )

    def test_rejects_non_decimal(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="non-decimal"):
            half_units(pa.array(["1", "0x10"]), _path(tmp_path), "x")
