"""CosMx column roles, header variants, and exact lexical value decoding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.cosmx._schemas import (
    _validate_columns,
    count_features,
    count_types,
    fov_position_types,
    half_units,
    metadata_types,
    placement_values,
    require_finite,
    require_positive,
    transcript_types,
    validate_label_ids,
)

if TYPE_CHECKING:
    from pathlib import Path

_CENTERS = ["CenterX_local_px", "CenterY_local_px", "CenterX_global_px", "CenterY_global_px"]


def _path(tmp_path: Path) -> ArtifactFile:
    return ArtifactFile(tmp_path / "source.csv")


class TestValidateColumns:
    """Empty names fail; csv_header has already rejected repeated names."""

    @pytest.mark.parametrize("columns", [[], ["a", ""], ["a\x00"]])
    def test_rejects(self, tmp_path: Path, columns: list[str]) -> None:
        with pytest.raises(ReaderFormatError, match="nonempty field names"):
            _validate_columns(columns, _path(tmp_path), "transcripts")


class TestValidateLabelIds:
    """The inspected label representation constrains linked positive cell IDs."""

    @pytest.mark.parametrize("dtype", ["uint8", "int8", "uint16", "int16", "uint32", "int32"])
    def test_exact_limit_and_overflow(self, tmp_path: Path, dtype: str) -> None:
        label_dtype = np.dtype(dtype)
        limit = np.iinfo(label_dtype).max
        validate_label_ids(pd.Index([1, limit], dtype="uint64[pyarrow]"), label_dtype, _path(tmp_path))
        with pytest.raises(ReaderFormatError, match=r"outside.*label range") as error:
            validate_label_ids(pd.Index([limit + 1], dtype="uint64[pyarrow]"), label_dtype, _path(tmp_path))
        assert error.value.context == ReaderErrorContext("cosmx", "cell labels")

    @pytest.mark.parametrize("dtype", ["uint64", "int64"])
    def test_64_bit_and_empty(self, tmp_path: Path, dtype: str) -> None:
        validate_label_ids(pd.Index([2**63 - 1], dtype="uint64[pyarrow]"), np.dtype(dtype), _path(tmp_path))
        validate_label_ids(pd.Index([], dtype="uint64[pyarrow]"), np.dtype(dtype), _path(tmp_path))


class TestCountFeatures:
    """Every non-identifier field is expression, including controls, in source order."""

    def test_order_and_controls(self, tmp_path: Path) -> None:
        columns = ["fov", "HLA-A", "cell_ID", "NegPrb1", "SystemControl1"]
        assert count_features(columns, _path(tmp_path)) == ("HLA-A", "NegPrb1", "SystemControl1")

    @pytest.mark.parametrize(("columns", "match"), [(["fov", "A"], "missing"), (["fov", "cell_ID"], "no count")])
    def test_rejects(self, tmp_path: Path, columns: list[str], match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match) as error:
            count_features(columns, _path(tmp_path))
        assert error.value.context == ReaderErrorContext("cosmx", "count matrix")


class TestMetadataTypes:
    """Vendor morphology and intensities are typed; other annotations stay text."""

    def test_roles(self, tmp_path: Path) -> None:
        columns = metadata_types(
            ["cell", "fov", "cell_ID", *_CENTERS, "Area", "Area.um2", "Mean.DAPI", "Max.G", "Width", "version"],
            _path(tmp_path),
        )
        assert columns.field("fov").type == columns.field("cell_ID").type == pa.uint64()
        for name in _CENTERS:
            assert columns.field(name).type == pa.string()
        for name in ("Area", "Area.um2", "Mean.DAPI", "Max.G", "Width"):
            assert columns.field(name).type == pa.float64()
        for name in ("cell", "version"):
            assert columns.field(name).type == pa.string()

    @pytest.mark.parametrize(
        ("columns", "match"),
        [
            (["fov", "cell_ID", *_CENTERS[:3]], "missing"),
            (["fov", "cell_ID", *_CENTERS, "region"], "a source 'region' column"),
            (["fov", "cell_ID", *_CENTERS, "spatial_pixel_frame"], "a source 'spatial_pixel_frame' column"),
        ],
    )
    def test_rejects(self, tmp_path: Path, columns: list[str], match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match) as error:
            metadata_types(columns, _path(tmp_path))
        assert error.value.context == ReaderErrorContext("cosmx", "cell metadata")


class TestFovPositionTypes:
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
        result = fov_position_types(columns, _path(tmp_path))
        if expected is None:
            assert result is None
        else:
            assert result == pa.schema([(expected[0], pa.uint64()), *((name, pa.string()) for name in expected[1:])])

    @pytest.mark.parametrize(
        "columns",
        [["x_px", "y_px"], ["fov", "FOV", "x_px", "y_px"], ["fov", "x_px", "y_px", "x_global_px", "y_global_px"]],
    )
    def test_rejects(self, tmp_path: Path, columns: list[str]) -> None:
        with pytest.raises(ReaderFormatError) as error:
            fov_position_types(columns, _path(tmp_path))
        assert error.value.context == ReaderErrorContext("cosmx", "FOV positions")


class TestTranscriptTypes:
    """Required event fields and supported annotations are projected."""

    def test_projection(self, tmp_path: Path) -> None:
        names = ["fov", "cell_ID", "cell", "x_local_px", "y_local_px", "x_global_px", "y_global_px", "z", "target", "x"]
        types = transcript_types(names, _path(tmp_path))
        assert types.names == [
            "fov",
            "cell_ID",
            "x_local_px",
            "y_local_px",
            "x_global_px",
            "y_global_px",
            "target",
            "z",
            "cell",
        ]
        assert types.field("fov").type == types.field("cell_ID").type == pa.uint64()
        assert types.field("z").type == pa.int64()
        assert types.field("target").type == types.field("cell").type == pa.string()
        assert types.field("x_local_px").type == pa.float64()

    def test_missing_fields(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="missing") as error:
            transcript_types(["fov", "target"], _path(tmp_path))
        assert error.value.context == ReaderErrorContext("cosmx", "transcripts")


class TestCountTypes:
    def test_ordered_uint64_schema(self, tmp_path: Path) -> None:
        types = count_types(["B", "cell_ID", "A", "fov"], _path(tmp_path))
        assert types == pa.schema([(name, pa.uint64()) for name in ("fov", "cell_ID", "B", "A")])


class TestRequirePositive:
    @pytest.mark.parametrize("values", [[1, 2**64 - 1], []])
    def test_positive_or_empty(self, tmp_path: Path, values: list[int]) -> None:
        require_positive(pa.array(values, pa.uint64()), _path(tmp_path), "cell metadata", "fov")

    @pytest.mark.parametrize("values", [[0], [None]])
    def test_missing_or_zero(self, tmp_path: Path, values: list[int | None]) -> None:
        with pytest.raises(ReaderFormatError, match="missing or zero") as error:
            require_positive(pa.array(values, pa.uint64()), _path(tmp_path), "cell metadata", "fov")
        assert error.value.context == ReaderErrorContext("cosmx", "cell metadata")


class TestRequireFinite:
    @pytest.mark.parametrize("values", [[1.5, -1.0], [], [None, 1.0]])
    def test_finite_optional_values(self, tmp_path: Path, values: list[float | None]) -> None:
        require_finite(pa.array(values, pa.float64()), _path(tmp_path), "cell metadata", "Area", nullable=True)

    @pytest.mark.parametrize(
        ("value", "nullable"),
        [
            (None, False),
            (np.nan, False),
            (np.nan, True),
            (np.inf, False),
            (np.inf, True),
            (-np.inf, False),
            (-np.inf, True),
        ],
    )
    def test_required_or_nonfinite(self, tmp_path: Path, value: float | None, *, nullable: bool) -> None:
        with pytest.raises(ReaderFormatError, match="nonfinite"):
            require_finite(pa.array([value], pa.float64()), _path(tmp_path), "cell metadata", "Area", nullable=nullable)


class TestPlacementValues:
    def test_correct_rounding_and_printed_precision(self, tmp_path: Path) -> None:
        tokens = ["162613.4", "159172.222222222", "0.10000000000000002", "-0"]
        values, halves = placement_values(pa.array(tokens), _path(tmp_path), "cell metadata", "x")
        np.testing.assert_array_equal(values, [float(token) for token in tokens])
        np.testing.assert_array_equal(halves, [0.05, 5e-10, 5e-18, 0.5])
        assert np.signbit(values[-1])

    def test_nonfinite_coordinate(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="nonfinite"):
            placement_values(pa.array(["1" * 400]), _path(tmp_path), "cell metadata", "x")


class TestHalfUnits:
    """Each bound is half a unit in the last printed decimal place."""

    def test_bounds(self, tmp_path: Path) -> None:
        tokens = pa.array(["806", "159172.222222222", "5.", ".25", "1.5e3", "-2E-2", "+7"])
        np.testing.assert_allclose(
            half_units(tokens, _path(tmp_path), "cell metadata", "x"),
            [0.5, 5e-10, 0.5, 5e-3, 50.0, 5e-3, 0.5],
            rtol=0,
            atol=0,
        )

    def test_rejects_non_decimal(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="non-decimal"):
            half_units(pa.array(["1", "0x10"]), _path(tmp_path), "cell metadata", "x")

    def test_representable_extremes_and_exponent_cancellation(self, tmp_path: Path) -> None:
        tokens = ["0e-323", "-0.000e-320", "0e308", "0." + "0" * 400 + "e400"]
        np.testing.assert_array_equal(
            half_units(pa.array(tokens), _path(tmp_path), "cell metadata", "x"), [5e-324, 5e-324, 5e307, 0.5]
        )

    @pytest.mark.parametrize(
        "token",
        ["5e-324", "0e-400", "0e309", "0e9223372036854775808", "0.0e-9223372036854775808"],
    )
    def test_unrepresentable_width_has_artifact_context(self, tmp_path: Path, token: str) -> None:
        with np.errstate(all="raise"), pytest.raises(ReaderFormatError, match=r"rounding.*(width|exponent)") as caught:
            half_units(pa.array([token]), _path(tmp_path), "cell metadata", "x")
        assert caught.value.context == ReaderErrorContext("cosmx", "cell metadata")
        assert caught.value.path == tmp_path / "source.csv"
        assert "'x'" in str(caught.value)
