"""Exact decoding of CSV text into float64 and 64-bit integers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from spatialdata_io.readers._utils.csv import (
    LITERAL_ZERO,
    decode_float64,
    decode_float64_block,
    decode_integer_block,
    decode_integers,
)
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Sequence

_CONTEXT = ReaderErrorContext("test", "cells")
_PATH = Path("cells.csv")
_ZEROS = ["0", "00", "-0", "+0", "0.", "0.0", "-0.000", ".0", "0e5", "+0E-3", "0.0e-999", "-.00E+12"]


def _text(tokens: Sequence[str | None], *, chunked: bool) -> pa.Array | pa.ChunkedArray:
    """Return string values as one array or split into two chunks."""
    array = pa.array(tokens, pa.string())
    return pa.chunked_array([array[:1], array[1:]], pa.string()) if chunked else array


def _floats(tokens: Sequence[str | None], *, chunked: bool = False) -> np.ndarray:
    return decode_float64(_text(tokens, chunked=chunked), name="x", context=_CONTEXT, path=_PATH)


def _integers(tokens: Sequence[str | None], *, minimum: int | None = 0, chunked: bool = False) -> np.ndarray:
    return decode_integers(_text(tokens, chunked=chunked), name="n", minimum=minimum, context=_CONTEXT, path=_PATH)


class TestLiteralZero:
    @pytest.mark.parametrize("token", _ZEROS)
    def test_matches_every_zero_spelling(self, token: str) -> None:
        assert re.fullmatch(LITERAL_ZERO, token)
        assert float(token) == 0

    @pytest.mark.parametrize("token", ["", "+", ".", "e5", "0e", "1", "0.01", "1e-400", "0x0", "-", "0..0"])
    def test_rejects_other_text(self, token: str) -> None:
        assert re.fullmatch(LITERAL_ZERO, token) is None


@pytest.mark.parametrize("chunked", [False, True])
class TestDecodeFloat64:
    def test_correctly_rounded_against_python(self, *, chunked: bool) -> None:
        tokens = ["0.1", "2.675", "-1e-320", "4.9e-324", "2.5e-324", "1e22", "9007199254740992", "-0", "+.5E-3", "5."]
        result = _floats(tokens, chunked=chunked)
        assert result.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(result, [float(token) for token in tokens])
        assert np.signbit(result[7])

    @pytest.mark.parametrize("zero", _ZEROS)
    def test_literal_zeros(self, *, chunked: bool, zero: str) -> None:
        assert _floats(["1", zero], chunked=chunked).tolist() == [1.0, 0.0]

    @pytest.mark.parametrize("token", ["9007199254740994", "-9007199254740992", "9007199254740993.5", "-1e22"])
    def test_large_values_that_are_exact_or_not_integral(self, *, chunked: bool, token: str) -> None:
        assert _floats(["1", token], chunked=chunked).tolist() == [1.0, float(token)]

    @pytest.mark.parametrize(
        ("token", "found"),
        [
            (None, "missing 'x' values"),
            ("", "nonnumeric 'x' values"),
            ("abc", "nonnumeric 'x' values"),
            (" 1", "nonnumeric 'x' values"),
            ("1_0", "nonnumeric 'x' values"),
            ("0x10", "nonnumeric 'x' values"),
            ("inf", "nonfinite 'x' values"),
            ("-Infinity", "nonfinite 'x' values"),
            ("nan", "nonfinite 'x' values"),
            ("1e400", "nonfinite 'x' values"),
            ("1e-400", "'x' value '1e-400' underflowing to zero"),
            ("-2.4e-324", "'x' value '-2.4e-324' underflowing to zero"),
            ("9007199254740993", "'x' value '9007199254740993' beyond the exact float64 integers"),
            ("-9007199254740993.0", "'x' value '-9007199254740993.0' beyond the exact float64 integers"),
            ("1e23", "'x' value '1e23' beyond the exact float64 integers"),
            ("1.7976931348623157e308", "'x' value '1.7976931348623157e308' beyond the exact float64 integers"),
        ],
    )
    def test_rejects(self, *, chunked: bool, token: str | None, found: str) -> None:
        with pytest.raises(ReaderFormatError) as caught:
            _floats(["0", "2", token], chunked=chunked)
        assert caught.value.found == found
        assert caught.value.expected == "finite float64 'x' values without underflow or integer precision loss"
        assert caught.value.context == _CONTEXT
        assert caught.value.path == _PATH

    def test_empty(self, *, chunked: bool) -> None:
        result = decode_float64(
            pa.chunked_array([], pa.string()) if chunked else pa.array([], pa.string()),
            name="x",
            context=_CONTEXT,
            path=_PATH,
        )
        assert result.dtype == np.dtype(np.float64)
        assert result.shape == (0,)


@pytest.mark.parametrize("chunked", [False, True])
class TestDecodeIntegers:
    def test_plain_integers(self, *, chunked: bool) -> None:
        result = _integers(["7", "0", "007", "-0", str(2**63 - 1)], chunked=chunked)
        assert result.dtype == np.dtype(np.int64)
        assert result.tolist() == [7, 0, 7, 0, 2**63 - 1]

    def test_exact_decimal_spellings(self, *, chunked: bool) -> None:
        result = _integers(["+2", "2.0", "3e0", "1E+2", ".5e1", "5.", "20e-1", "-0.0", "0" * 30 + "1"], chunked=chunked)
        assert result.dtype == np.dtype(np.int64)
        assert result.tolist() == [2, 2, 3, 100, 5, 5, 2, 0, 1]

    @pytest.mark.parametrize(
        ("tokens", "expected"),
        [
            ([str(2**63)], [2**63]),
            (["1", str(2**64 - 1)], [1, 2**64 - 1]),
            (["1", f"{2**64 - 1}.0"], [1, 2**64 - 1]),
            (["2.0", f"{2**63}e0"], [2, 2**63]),
        ],
    )
    def test_unsigned_above_int64(self, *, chunked: bool, tokens: list[str], expected: list[int]) -> None:
        result = _integers(["0", *tokens], chunked=chunked)
        assert result.dtype == np.dtype(np.uint64)
        assert result.tolist() == [0, *expected]

    @pytest.mark.parametrize(
        ("tokens", "expected"),
        [
            (["-5", "3"], [-5, 3]),
            ([str(-(2**63)), "-1.0"], [-(2**63), -1]),
            (["-2e0", str(2**63 - 1)], [-2, 2**63 - 1]),
        ],
    )
    def test_without_minimum_accepts_int64(self, *, chunked: bool, tokens: list[str], expected: list[int]) -> None:
        result = _integers(["0", *tokens], minimum=None, chunked=chunked)
        assert result.dtype == np.dtype(np.int64)
        assert result.tolist() == [0, *expected]

    def test_negative_minimum(self, *, chunked: bool) -> None:
        assert _integers(["-2", "0"], minimum=-2, chunked=chunked).tolist() == [-2, 0]
        with pytest.raises(ReaderFormatError, match="'-3'"):
            _integers(["-2", "-3"], minimum=-2, chunked=chunked)
        with pytest.raises(ReaderFormatError, match="within int64"):
            _integers(["-2", str(2**63)], minimum=-2, chunked=chunked)

    @pytest.mark.parametrize(
        ("token", "minimum"),
        [
            ("0", 1),
            ("0.0", 1),
            ("-1", 0),
            ("-1e0", 0),
            (str(2**64), 0),
            (f"{2**64}.0", 0),
            (str(2**63), None),
            (str(-(2**63) - 1), None),
            ("1.5", 0),
            ("1e-1", 0),
            ("1.00000000000000001", 0),
            ("", 0),
            (" 1", 0),
            ("1 ", 0),
            ("1\n", 0),
            ("1_000", 0),
            ("0x10", 0),
            ("-0x10", None),
            ("0b1", 0),
            ("\u0661", 0),
            ("inf", 0),
            ("nan", 0),
            ("1e999999999", 0),
            ("1e-999999999", 0),
            ("1" * 129, 0),
        ],
    )
    def test_rejects_with_the_invalid_token(self, *, chunked: bool, token: str, minimum: int | None) -> None:
        with pytest.raises(ReaderFormatError) as caught:
            _integers(["1", "2", token], minimum=minimum, chunked=chunked)
        assert caught.value.found == f"'n' value {token!r}"
        assert caught.value.context == _CONTEXT
        assert caught.value.path == _PATH

    def test_rejects_missing_values(self, *, chunked: bool) -> None:
        with pytest.raises(ReaderFormatError) as caught:
            _integers(["1", None], chunked=chunked)
        assert caught.value.found == "missing 'n' values"

    @pytest.mark.parametrize(
        ("minimum", "expected"),
        [
            (1, "exact integer 'n' values of at least 1 below 2**64"),
            (None, "exact integer 'n' values within int64"),
            (-1, "exact integer 'n' values of at least -1 within int64"),
        ],
    )
    def test_states_the_contract(self, *, chunked: bool, minimum: int | None, expected: str) -> None:
        with pytest.raises(ReaderFormatError) as caught:
            _integers(["1", "x"], minimum=minimum, chunked=chunked)
        assert caught.value.expected == expected

    def test_hexadecimal_is_rejected_although_arrow_casts_it(self, *, chunked: bool) -> None:
        text = _text(["1", "0x10"], chunked=chunked)
        assert pc.cast(text, pa.int64()).to_pylist() == [1, 16]
        with pytest.raises(ReaderFormatError, match="'0x10'"):
            decode_integers(text, name="n", minimum=0, context=_CONTEXT, path=_PATH)

    @pytest.mark.parametrize("token", ["1-2", "--1", "-", "1-", "-1-", "0-0", "+-1"])
    def test_rejects_misplaced_signs(self, *, chunked: bool, token: str) -> None:
        with pytest.raises(ReaderFormatError) as caught:
            _integers(["1", token], minimum=None, chunked=chunked)
        assert caught.value.found == f"'n' value {token!r}"

    def test_sliced_text_is_checked_within_its_slice(self, *, chunked: bool) -> None:
        source = pa.array(["0x10", "5", "-7", "0x20"], pa.string())
        text = pa.chunked_array([source[1:2], source[2:3]]) if chunked else source[1:3]
        assert decode_integers(text, name="n", minimum=None, context=_CONTEXT, path=_PATH).tolist() == [5, -7]
        with pytest.raises(ReaderFormatError, match="'0x20'"):
            decode_integers(source[2:], name="n", minimum=None, context=_CONTEXT, path=_PATH)

    def test_other_string_types(self, *, chunked: bool) -> None:
        text = pa.array(["1", "2.0", str(2**63)], pa.large_string())
        result = decode_integers(
            pa.chunked_array([text]) if chunked else text, name="n", minimum=0, context=_CONTEXT, path=_PATH
        )
        assert result.dtype == np.dtype(np.uint64)
        assert result.tolist() == [1, 2, 2**63]

    def test_empty(self, *, chunked: bool) -> None:
        result = decode_integers(
            pa.chunked_array([], pa.string()) if chunked else pa.array([], pa.string()),
            name="n",
            minimum=1,
            context=_CONTEXT,
            path=_PATH,
        )
        assert result.dtype == np.dtype(np.int64)
        assert result.shape == (0,)


class TestDecodeIntegerBlock:
    def test_rows_by_columns(self) -> None:
        columns = [pa.array(["1", "0", "2.0"]), pa.array(["0", "5", "0"])]
        result = decode_integer_block(columns, names=("A", "B"), minimum=0, context=_CONTEXT, path=_PATH)
        assert result.dtype == np.dtype(np.int64)
        np.testing.assert_array_equal(result, [[1, 0], [0, 5], [2, 0]])

    def test_one_dtype_for_the_block(self) -> None:
        columns = [pa.array(["1", "0"]), pa.array(["0", str(2**64 - 1)])]
        result = decode_integer_block(columns, names=("A", "B"), minimum=0, context=_CONTEXT, path=_PATH)
        assert result.dtype == np.dtype(np.uint64)
        np.testing.assert_array_equal(result, np.array([[1, 0], [0, 2**64 - 1]], dtype=np.uint64))

    def test_no_rows(self) -> None:
        columns = [pa.array([], pa.string()), pa.array([], pa.string())]
        result = decode_integer_block(columns, names=("A", "B"), minimum=0, context=_CONTEXT, path=_PATH)
        assert result.shape == (0, 2)

    @pytest.mark.parametrize(
        ("second", "third", "column", "found"),
        [
            ("1.5", "x", "B", "'B' value '1.5'"),
            ("1", "-1", "C", "'C' value '-1'"),
            (None, "1", "B", "missing 'B' values"),
        ],
    )
    def test_error_names_the_first_invalid_column(
        self, second: str | None, third: str, column: str, found: str
    ) -> None:
        columns = [pa.array(["1"]), pa.array([second], pa.string()), pa.array([third])]
        with pytest.raises(ReaderFormatError) as caught:
            decode_integer_block(columns, names=("A", "B", "C"), minimum=0, context=_CONTEXT, path=_PATH)
        assert caught.value.found == found
        assert caught.value.expected == f"exact integer {column!r} values of at least 0 below 2**64"
        assert caught.value.context == _CONTEXT
        assert caught.value.path == _PATH


class TestDecodeFloat64Block:
    def test_rows_by_columns(self) -> None:
        columns = [pa.array(["1", "0"]), pa.array(["0.5", "2e-3"])]
        result = decode_float64_block(columns, names=("A", "B"), context=_CONTEXT, path=_PATH)
        assert result.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(result, [[1, 0.5], [0, 0.002]])

    @pytest.mark.parametrize(
        ("token", "found"),
        [
            ("inf", "nonfinite 'B' values"),
            ("1e-400", "'B' value '1e-400' underflowing to zero"),
            ("9007199254740993", "'B' value '9007199254740993' beyond the exact float64 integers"),
        ],
    )
    def test_error_names_the_column(self, token: str, found: str) -> None:
        with pytest.raises(ReaderFormatError) as caught:
            decode_float64_block(
                [pa.array(["1"]), pa.array([token]), pa.array(["x"])],
                names=("A", "B", "C"),
                context=_CONTEXT,
                path=_PATH,
            )
        assert caught.value.found == found
