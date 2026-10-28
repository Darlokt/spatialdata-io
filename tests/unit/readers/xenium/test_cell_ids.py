"""Decode cell-ID pairs exactly as the documented 10x codec specifies."""

from __future__ import annotations

import numpy as np
import pytest

from spatialdata_io.readers.xenium._cell_ids import decode_cell_ids


def _documented(prefix: int, suffix: int) -> str:
    """Independent oracle: eight hex digits of the prefix, each 0-f shifted to a-p."""
    shifted = "".join(chr(ord("a") + int(digit, 16)) for digit in f"{prefix:08x}")
    return f"{shifted}-{suffix}"


class TestDecodeCellIds:
    """Integer pairs decode with the documented 10x hexadecimal cell ID algorithm."""

    def test_documented_example(self) -> None:
        # 10x documentation: prefix 1437536272 (0x55af1010) with suffix 1 is ffkpbaba-1.
        pairs = np.array([[1437536272, 1]], dtype=np.uint32)

        np.testing.assert_array_equal(decode_cell_ids(pairs), ["ffkpbaba-1"])

    def test_limits_padding_and_suffixes(self) -> None:
        pairs = np.array([[0, 1], [1, 1], [2**32 - 1, 1], [0x3D51, 2], [7, 4294967295]], dtype=np.uint32)

        result = decode_cell_ids(pairs)

        np.testing.assert_array_equal(
            result, ["aaaaaaaa-1", "aaaaaaab-1", "pppppppp-1", "aaaadnfb-2", "aaaaaaah-4294967295"]
        )
        assert result.dtype.kind == "U"

    def test_randomized_against_documented_algorithm(self) -> None:
        rng = np.random.default_rng(20261026)
        pairs = np.column_stack(
            [
                rng.integers(0, 2**32, size=5000, dtype=np.uint32),
                rng.integers(1, 10, size=5000, dtype=np.uint32),
            ]
        )

        expected = [_documented(int(prefix), int(suffix)) for prefix, suffix in pairs]

        np.testing.assert_array_equal(decode_cell_ids(pairs), expected)

    def test_empty(self) -> None:
        assert decode_cell_ids(np.empty((0, 2), dtype=np.uint32)).shape == (0,)

    @pytest.mark.parametrize(
        "pairs",
        [
            np.zeros((3,), dtype=np.uint32),
            np.zeros((3, 3), dtype=np.uint32),
            np.zeros((3, 2), dtype=np.int64),
            np.zeros((3, 2), dtype=np.uint64),
        ],
        ids=["one-dimensional", "three-columns", "signed", "uint64"],
    )
    def test_rejects_other_shapes_and_dtypes(self, pairs: np.ndarray) -> None:
        with pytest.raises(ValueError, match=r"\(N, 2\) uint32"):
            decode_cell_ids(pairs)
