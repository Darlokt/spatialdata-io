"""Exact column contracts, identifier normalization and vocabulary lookups."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers.xenium._schemas import (
    CELL_FLOATS,
    TRANSCRIPT_REQUIRED,
    cast_column,
    cell_types,
    id_index,
    integer_ids,
    normalize_ids,
    normalize_text,
    require_finite,
    require_no_nulls,
    text_series,
    transcript_types,
    validate_analysis_header,
    validate_header,
    vocabulary_codes,
)

PATH = Path("bundle/cells.parquet")
CONTEXT = ReaderErrorContext("xenium", "cells table", "5.0.0")
ANALYSIS = ReaderErrorContext("xenium", "analysis", "5.0.0")


def _chunked(values: list[object], kind: pa.DataType) -> pa.ChunkedArray:
    return pa.chunked_array([pa.array(values, type=kind)])


class TestValidateHeader:
    """Named headers accept any order and extra columns but never miss or repeat names."""

    def test_any_order_and_extra_columns(self) -> None:
        validate_header(["b", "extra", "a"], ("a", "b"), path=PATH, context=CONTEXT)

    @pytest.mark.parametrize(
        ("names", "match"),
        [
            (["a", "a", "b"], "unique nonempty"),
            (["a", "", "b"], "unique nonempty"),
            (["a"], r"missing columns \['b'\]"),
        ],
    )
    def test_rejects(self, names: list[str], match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            validate_header(names, ("a", "b"), path=PATH, context=CONTEXT)


class TestValidateAnalysisHeader:
    """Analysis headers have exact field order and consecutive projection components."""

    def test_clustering_header(self) -> None:
        validate_analysis_header(["Barcode", "Cluster"], "clustering", path=PATH, context=ANALYSIS)

    @pytest.mark.parametrize(
        "names",
        [
            [],
            ["Barcode"],
            ["Cluster", "Barcode"],
            ["Barcode", "Cluster", "Extra"],
            ["Barcode", "Barcode"],
            ["Barcode", ""],
        ],
        ids=["empty", "missing", "reordered", "extra", "duplicate", "empty-name"],
    )
    def test_rejects_clustering_headers(self, names: list[str]) -> None:
        with pytest.raises(ReaderFormatError, match="Barcode,Cluster") as error:
            validate_analysis_header(names, "clustering", path=PATH, context=ANALYSIS)

        assert error.value.path == PATH
        assert error.value.context == ANALYSIS
        assert f"header {names!r}" in str(error.value)

    @pytest.mark.parametrize(("kind", "prefix"), [("pca", "PC"), ("umap", "UMAP")])
    @pytest.mark.parametrize("components", [1, 2, 3])
    def test_projection_headers(self, kind: str, prefix: str, components: int) -> None:
        validate_analysis_header(
            ["Barcode", *(f"{prefix}-{i}" for i in range(1, components + 1))], kind, path=PATH, context=ANALYSIS
        )

    @pytest.mark.parametrize(("kind", "prefix"), [("pca", "PC"), ("umap", "UMAP")])
    @pytest.mark.parametrize(
        "columns",
        [
            [],
            ["Barcode"],
            ["barcode", "{prefix}-1"],
            ["{prefix}-1", "Barcode"],
            ["Barcode", "{prefix}-2"],
            ["Barcode", "{prefix}-1", "{prefix}-3"],
            ["Barcode", "{prefix}-2", "{prefix}-1"],
            ["Barcode", "{prefix}-1", "{prefix}-1"],
            ["Barcode", "{prefix}-0"],
            ["Barcode", "{prefix}-01"],
            ["Barcode", "other-1"],
            ["Barcode", "{prefix}-1", "Extra"],
            ["Barcode", ""],
        ],
        ids=[
            "empty",
            "missing",
            "barcode-case",
            "barcode-order",
            "missing-first",
            "gap",
            "reordered",
            "duplicate",
            "zero",
            "leading-zero",
            "wrong-prefix",
            "extra",
            "empty-name",
        ],
    )
    def test_rejects_projection_headers(self, kind: str, prefix: str, columns: list[str]) -> None:
        names = [name.format(prefix=prefix) for name in columns]
        with pytest.raises(ReaderFormatError, match=rf"Barcode,{prefix}-1,\.\.\.") as error:
            validate_analysis_header(names, kind, path=PATH, context=ANALYSIS)

        assert error.value.path == PATH
        assert error.value.context == ANALYSIS
        assert f"header {names!r}" in str(error.value)


class TestCellTypes:
    """Cell tables declare explicit Arrow types per family."""

    def test_explicit_types_by_family(self) -> None:
        names = ["cell_id", *CELL_FLOATS, "transcript_counts", "nucleus_count", "segmentation_method", "extra"]

        text = cell_types(names, "hex_ids")
        integer = cell_types(names, "integer_ids")

        assert text["cell_id"] == pa.string()
        assert integer["cell_id"] == pa.int64()
        assert all(text[name] == pa.float64() for name in CELL_FLOATS)
        assert text["nucleus_count"] == pa.int64()
        assert text["segmentation_method"] == pa.string()
        assert text["extra"] == pa.string()
        assert "genomic_control_counts" not in text


class TestTranscriptTypes:
    """Transcript tables project the documented columns only."""

    def test_projects_documented_columns_only(self) -> None:
        names = [*TRANSCRIPT_REQUIRED, "codeword_index", "is_gene", "future_column"]

        types = transcript_types(names, path=PATH, context=CONTEXT)

        assert list(types) == [*TRANSCRIPT_REQUIRED, "codeword_index", "is_gene"]
        assert types["codeword_index"] == pa.int32()

    def test_requires_documented_columns(self) -> None:
        with pytest.raises(ReaderFormatError, match="missing columns"):
            transcript_types(["cell_id", "feature_name"], path=PATH, context=CONTEXT)


class TestCastColumn:
    """Column casts are lossless or raise."""

    def test_lossless_casts(self) -> None:
        result = cast_column(_chunked([1, 2], pa.int32()), pa.int64(), "count", path=PATH, context=CONTEXT)

        assert result.type == pa.int64()
        assert result.to_pylist() == [1, 2]

    def test_identity_returns_the_column(self) -> None:
        column = _chunked([1.5], pa.float32())

        assert cast_column(column, pa.float32(), "x", path=PATH, context=CONTEXT) is column

    @pytest.mark.parametrize(
        ("values", "source", "target"),
        [([1.5], pa.float64(), pa.int64()), (["a"], pa.string(), pa.int64()), ([2**40], pa.int64(), pa.int32())],
        ids=["fraction", "text", "overflow"],
    )
    def test_lossy_casts_raise(self, values: list[object], source: pa.DataType, target: pa.DataType) -> None:
        with pytest.raises(ReaderFormatError, match=f"x as {target}"):
            cast_column(_chunked(values, source), target, "x", path=PATH, context=CONTEXT)


class TestNormalizeText:
    """Text and binary columns decode strictly as UTF-8 without nulls."""

    @pytest.mark.parametrize("kind", [pa.string(), pa.large_string(), pa.binary(), pa.large_binary()])
    def test_text_and_binary(self, kind: pa.DataType) -> None:
        values = ["aaaaaaab-1", "UNASSIGNED"]
        column = _chunked(
            [
                value.encode() if pa.types.is_binary(kind) or pa.types.is_large_binary(kind) else value
                for value in values
            ],
            kind,
        )

        result = normalize_text(column, "cell_id", path=PATH, context=CONTEXT)

        assert result.type == pa.string()
        assert result.to_pylist() == values

    def test_invalid_utf8(self) -> None:
        with pytest.raises(ReaderFormatError, match="not UTF-8"):
            normalize_text(_chunked([b"\xff"], pa.binary()), "feature_name", path=PATH, context=CONTEXT)

    def test_rejects_numbers_and_nulls(self) -> None:
        with pytest.raises(ReaderFormatError, match="feature_name of type int64"):
            normalize_text(_chunked([1], pa.int64()), "feature_name", path=PATH, context=CONTEXT)
        with pytest.raises(ReaderFormatError, match="missing feature_name"):
            normalize_text(_chunked(["a", None], pa.string()), "feature_name", path=PATH, context=CONTEXT)


class TestNormalizeIds:
    """Cell IDs normalize to integers or text by family."""

    def test_integer_family(self) -> None:
        result = normalize_ids(_chunked([1, -1], pa.int32()), "integer_ids", path=PATH, context=CONTEXT)

        assert result.type == pa.int64()
        assert result.to_pylist() == [1, -1]

    @pytest.mark.parametrize(
        "column",
        [_chunked(["1"], pa.string()), _chunked([1, None], pa.int32()), _chunked([1.0], pa.float64())],
        ids=["text", "null", "float"],
    )
    def test_integer_family_rejects(self, column: pa.ChunkedArray) -> None:
        with pytest.raises(ReaderFormatError, match="non-null integer identifiers"):
            normalize_ids(column, "integer_ids", path=PATH, context=CONTEXT)

    def test_text_families(self) -> None:
        result = normalize_ids(_chunked([b"aaaaaaab-1"], pa.binary()), "hex_ids", path=PATH, context=CONTEXT)

        assert result.to_pylist() == ["aaaaaaab-1"]


class TestIdIndex:
    """ID columns become one typed pandas index."""

    def test_integer_and_text(self) -> None:
        integers = id_index(_chunked([3, 1], pa.int64()))
        text = id_index(pa.array(["b", "a"]))

        assert integers.dtype == np.int64
        assert integers.tolist() == [3, 1]
        assert isinstance(text.dtype, pd.StringDtype)
        assert text.tolist() == ["b", "a"]


class TestIntegerIds:
    """Integer barcodes round-trip as canonical decimals."""

    def test_canonical_decimal(self) -> None:
        result = integer_ids(pd.Index(["1", "20", "300"], dtype="str"), path=PATH, context=CONTEXT)

        assert result.dtype == np.int64
        assert result.tolist() == [1, 20, 300]

    @pytest.mark.parametrize(
        ("values", "match"),
        [(["1", "x"], "non-integer"), (["1", "02"], "non-canonical"), (["99999999999999999999"], "non-integer")],
    )
    def test_rejects(self, values: list[str], match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            integer_ids(pd.Index(values, dtype="str"), path=PATH, context=CONTEXT)


class TestVocabularyCodes:
    """Values map to vocabulary positions and unknown values raise."""

    def test_positions_in_vocabulary_order(self) -> None:
        vocabulary = pd.Index(["GeneB", "GeneA", "GeneC"], dtype="str")

        codes = vocabulary_codes(
            text_series(pa.array(["GeneA", "GeneC", "GeneA", "GeneB"])),
            vocabulary,
            "feature_name",
            path=PATH,
            context=CONTEXT,
        )

        np.testing.assert_array_equal(codes, [1, 2, 1, 0])

    def test_integer_values(self) -> None:
        codes = vocabulary_codes(pd.Series([-1, 3]), pd.Index([3, -1]), "cell_id", path=PATH, context=CONTEXT)

        np.testing.assert_array_equal(codes, [1, 0])

    def test_empty_values(self) -> None:
        codes = vocabulary_codes(
            text_series(pa.array([], pa.string())), pd.Index(["a"]), "x", path=PATH, context=CONTEXT
        )

        assert codes.shape == (0,)

    def test_unknown_values_raise_with_examples(self) -> None:
        with pytest.raises(ReaderFormatError, match=r"unknown feature_name values \['GeneZ'\]"):
            vocabulary_codes(
                text_series(pa.array(["GeneA", "GeneZ"])),
                pd.Index(["GeneA"]),
                "feature_name",
                path=PATH,
                context=CONTEXT,
            )


class TestRequireFinite:
    """Nonfinite values raise except documented NaN columns."""

    def test_nan_only_where_documented(self) -> None:
        require_finite(np.array([1.0, np.nan]), "nucleus_area", path=PATH, context=CONTEXT, allow_nan=True)

        with pytest.raises(ReaderFormatError, match="nonfinite nucleus_area"):
            require_finite(np.array([np.inf]), "nucleus_area", path=PATH, context=CONTEXT, allow_nan=True)
        with pytest.raises(ReaderFormatError, match="nonfinite x"):
            require_finite(np.array([np.nan]), "x", path=PATH, context=CONTEXT)


class TestRequireNoNulls:
    """Nulls raise and name their columns."""

    def test_reports_columns(self) -> None:
        require_no_nulls(pa.table({"a": [1]}), path=PATH, context=CONTEXT)

        with pytest.raises(ReaderFormatError, match=r"missing values in \['b'\]"):
            require_no_nulls(pa.table({"a": [1], "b": [None]}), path=PATH, context=CONTEXT)
