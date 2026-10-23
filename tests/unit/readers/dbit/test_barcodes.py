"""Verify the bounded two-column barcode schema and immutable lookups."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import require_file
from spatialdata_io.readers.dbit._barcodes import _read_barcodes

if TYPE_CHECKING:
    from pathlib import Path


class TestReadBarcodes:
    """Reject ambiguous grid positions and malformed bounded barcode maps."""

    def test_unordered_case_insensitive_sequences(self, tmp_path: Path) -> None:
        path = tmp_path / "barcodes.txt"
        path.write_text("B2\taaaaaaaa\nA1\tTTTTTTTT\nB1\tgggggggg\nA2\tAAAAAAAA\n", encoding="utf-8")

        result = _read_barcodes(require_file(path, context=ReaderErrorContext("dbit", "barcodes")))

        assert result.a == {"TTTTTTTT": 1, "AAAAAAAA": 2}
        assert result.b == {"GGGGGGGG": 1, "AAAAAAAA": 2}
        # A and B can use the same sequence: their position in the combined ID disambiguates it.
        assert isinstance(result.a, MappingProxyType)
        assert isinstance(result.b, MappingProxyType)

    @pytest.mark.parametrize(
        "contents",
        [
            "A1\nB1\n",
            "A1\tAAAAAAAA\textra\nB1\tCCCCCCCC\textra\n",
            "A0\tAAAAAAAA\nB1\tCCCCCCCC\n",
            "A51\tAAAAAAAA\nB1\tCCCCCCCC\n",
            "A01\tAAAAAAAA\nB1\tCCCCCCCC\n",
            "a1\tAAAAAAAA\nB1\tCCCCCCCC\n",
            "|1\tAAAAAAAA\nB1\tCCCCCCCC\n",
            "A1\t||||||||\nB1\tCCCCCCCC\n",
            "A1\tAAAAAAAN\nB1\tCCCCCCCC\n",
            "A1\tAAAAAAA\nB1\tCCCCCCCC\n",
            "A1\t\nB1\tCCCCCCCC\n",
            "A1\tAAAAAAAA\x00CORRUPT\nB1\tCCCCCCCC\n",
            "A1\x00CORRUPT\tAAAAAAAA\nB1\tCCCCCCCC\n",
            "A1\tAAAAAAAA\nA1\tGGGGGGGG\nB1\tCCCCCCCC\n",
            "A1\tAAAAAAAA\nA2\taaaaaaaa\nB1\tCCCCCCCC\nB2\tGGGGGGGG\n",
            "A1\tAAAAAAAA\nA3\tGGGGGGGG\nB1\tCCCCCCCC\nB2\tTTTTTTTT\n",
            "A1\tAAAAAAAA\n",
            "B1\tAAAAAAAA\n",
            "A1\tAAAAAAAA\nA2\tGGGGGGGG\nB1\tCCCCCCCC\n",
            "A1\tAAAAAAAA\n" * 101,
            "A1\t" + "A" * 16_384,
        ],
    )
    def test_invalid_schema(self, tmp_path: Path, contents: str) -> None:
        path = tmp_path / "barcodes.txt"
        path.write_text(contents, encoding="utf-8")

        with pytest.raises(ReaderFormatError) as raised:
            _read_barcodes(require_file(path, context=ReaderErrorContext("dbit", "barcodes")))

        assert raised.value.path == path
        assert raised.value.context.reader == "dbit"

    @pytest.mark.parametrize(
        ("contents", "error_type"),
        [
            (b"", pd.errors.EmptyDataError),
            (b"A1\t\xff\n", UnicodeDecodeError),
            (b'A1\t"unclosed', pd.errors.ParserError),
        ],
    )
    def test_parser_failures_close_source(self, tmp_path: Path, contents: bytes, error_type: type[Exception]) -> None:
        path = tmp_path / "barcodes.txt"
        path.write_bytes(contents)

        with pytest.raises(error_type) as raised:
            _read_barcodes(require_file(path, context=ReaderErrorContext("dbit", "barcodes")))

        assert any(str(path) in note for note in raised.value.__notes__)
        path.unlink()
