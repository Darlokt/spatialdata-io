"""Verify the bounded two-column barcode schema and immutable lookups."""

from __future__ import annotations

import io
from itertools import islice, product
from types import MappingProxyType
from typing import TYPE_CHECKING

import pyarrow.csv as pacsv
import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile, require_file
from spatialdata_io.readers.dbit._barcodes import MAX_BARCODE_BYTES, read_barcodes

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestReadBarcodes:
    """Reject ambiguous grid positions and malformed bounded barcode maps."""

    def test_unordered_case_insensitive_sequences(self, tmp_path: Path) -> None:
        path = tmp_path / "barcodes.txt"
        path.write_text("B2\taaaaaaaa\nA1\tTTTTTTTT\nB1\tgggggggg\nA2\tAAAAAAAA\n", encoding="utf-8")

        result = read_barcodes(require_file(path, context=ReaderErrorContext("dbit", "barcodes")))

        assert result.a == {"TTTTTTTT": 1, "AAAAAAAA": 2}
        assert result.b == {"GGGGGGGG": 1, "AAAAAAAA": 2}
        # A and B can use the same sequence: their position in the combined ID disambiguates it.
        assert isinstance(result.a, MappingProxyType)
        assert isinstance(result.b, MappingProxyType)

    @pytest.mark.parametrize("bom", ["", "\ufeff"])
    def test_quoted_positional_tokens(self, tmp_path: Path, bom: str) -> None:
        path = tmp_path / "barcodes.txt"
        path.write_text(bom + '\n"B1"\t"cccccccc"\n"A1"\t"aaaaaaaa"\n', encoding="utf-8")
        result = read_barcodes(require_file(path, context=ReaderErrorContext("dbit", "barcodes")))
        assert result.a == {"AAAAAAAA": 1}
        assert result.b == {"CCCCCCCC": 1}

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
            read_barcodes(require_file(path, context=ReaderErrorContext("dbit", "barcodes")))

        assert raised.value.path == path
        assert raised.value.context.reader == "dbit"

    @pytest.mark.parametrize("contents", [b"", b"A1\t\xff\n", b"A1\n", b"A1\tAAAAAAAA\textra\n"])
    def test_parser_failures_close_source(self, tmp_path: Path, contents: bytes, mocker: MockerFixture) -> None:
        path = tmp_path / "barcodes.txt"
        path.write_bytes(contents)
        opened = mocker.spy(type(path), "open")

        with pytest.raises(ReaderFormatError, match="malformed CSV") as raised:
            read_barcodes(require_file(path, context=ReaderErrorContext("dbit", "barcodes")))

        assert raised.value.path == path
        assert raised.value.context.artifact == "barcode positions"
        assert all(source.closed for source in opened.spy_return_list)

    @pytest.mark.parametrize("ending", [b"", b"\n"])
    def test_last_record_and_native_quoting(self, tmp_path: Path, ending: bytes) -> None:
        """Arrow accepts a last record without newline and text after a closing quote."""
        path = tmp_path / "barcodes.txt"
        path.write_bytes(b'A1\t"AAAA"AAAA\nB1\tCCCCCCCC' + ending)

        result = read_barcodes(ArtifactFile(path))

        assert result.a == {"AAAAAAAA": 1}
        assert result.b == {"CCCCCCCC": 1}

    def test_complete_one_hundred_record_domain(self, tmp_path: Path) -> None:
        sequences = ["".join(bases) for bases in islice(product("ACGT", repeat=8), 50)]
        path = tmp_path / "barcodes.txt"
        path.write_text(
            "".join(f"{axis}{i + 1}\t{sequence}\n" for axis in ("A", "B") for i, sequence in enumerate(sequences)),
            encoding="utf-8",
        )

        result = read_barcodes(ArtifactFile(path))

        assert len(result.a) == len(result.b) == 50
        assert result.a[sequences[-1]] == result.b[sequences[-1]] == 50

    def test_one_hundred_and_one_records_rejected_before_normalization(self, tmp_path: Path) -> None:
        path = tmp_path / "barcodes.txt"
        path.write_bytes(b"A1\tAAAAAAAA\n" * 101)

        with pytest.raises(ReaderFormatError, match="barcode map shape"):
            read_barcodes(ArtifactFile(path))

    @pytest.mark.parametrize("size", [MAX_BARCODE_BYTES, MAX_BARCODE_BYTES + 1, MAX_BARCODE_BYTES * 3])
    def test_single_bounded_read_and_size_limit(self, tmp_path: Path, size: int, mocker: MockerFixture) -> None:
        """Read one detection byte, close, and reject oversize before invoking Arrow."""
        path = ArtifactFile(tmp_path / "barcodes.txt")
        records = b"A1\tAAAAAAAA\nB1\tCCCCCCCC\n"
        source = io.BytesIO(records + b"\n" * (size - len(records)))
        read = mocker.spy(source, "read")
        opened = mocker.patch.object(type(path), "open", return_value=source)
        parse = mocker.spy(pacsv, "read_csv")

        if size > MAX_BARCODE_BYTES:
            with pytest.raises(ReaderFormatError, match="larger than 64 KiB") as raised:
                read_barcodes(path)
            assert raised.value.path == path
            parse.assert_not_called()
        else:
            result = read_barcodes(path)
            assert result.a == {"AAAAAAAA": 1}
            assert result.b == {"CCCCCCCC": 1}
        opened.assert_called_once_with("rb")
        read.assert_called_once_with(MAX_BARCODE_BYTES + 1)
        assert source.closed

    def test_read_failure_closes_and_retains_filesystem_error(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = ArtifactFile(tmp_path / "barcodes.txt")
        source = io.BytesIO()
        mocker.patch.object(type(path), "open", return_value=source)
        mocker.patch.object(source, "read", side_effect=PermissionError("denied"))

        with pytest.raises(PermissionError, match="denied") as raised:
            read_barcodes(path)

        assert source.closed
        assert any(str(path) in note and "barcode positions" in note for note in raised.value.__notes__)
