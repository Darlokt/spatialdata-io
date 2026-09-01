"""Tests for bounded JSON-object metadata reading."""

from __future__ import annotations

import inspect
import io
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast, get_type_hints, override

import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.metadata import _json, read_json_object
from spatialdata_io.readers._utils.paths import ArtifactFile, require_file

if TYPE_CHECKING:
    from collections.abc import Buffer
    from types import TracebackType

    from pytest_mock import MockerFixture

_CONTEXT = ReaderErrorContext(reader="test_reader", artifact="experiment metadata", detected_version="2.0")


def _artifact(tmp_path: Path, payload: bytes) -> ArtifactFile:
    path = tmp_path / "metadata.json"
    path.write_bytes(payload)
    return require_file(path, context=_CONTEXT)


class _ManagedStream:
    def __init__(
        self,
        payload: bytes,
        *,
        read_error: OSError | None = None,
        close_error: OSError | None = None,
    ) -> None:
        self.payload = payload
        self.read_error = read_error
        self.close_error = close_error
        self.read_sizes: list[int] = []
        self.close_attempted = False
        self.closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close_attempted = True
        if self.close_error is not None:
            raise self.close_error
        self.closed = True

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        if self.read_error is not None:
            raise self.read_error
        return self.payload[:size]


class _ShortRawStream(io.RawIOBase):
    def __init__(self, payload: bytes, *, chunk_size: int) -> None:
        super().__init__()
        self._payload = payload
        self._chunk_size = chunk_size
        self._position = 0

    @override
    def readable(self) -> bool:
        return True

    @override
    def readinto(self, buffer: Buffer, /) -> int:
        if self._position == len(self._payload):
            return 0
        view = memoryview(buffer)
        count = min(len(view), self._chunk_size, len(self._payload) - self._position)
        view[:count] = self._payload[self._position : self._position + count]
        self._position += count
        return count


class TestReadJsonObject:
    def test_returns_owned_plain_nested_objects(self, tmp_path: Path) -> None:
        path = _artifact(tmp_path, b'{"outer":{"items":[{"value":1}]},"flag":true,"missing":null}')

        first = read_json_object(path, max_bytes=128, context=_CONTEXT)
        second = read_json_object(path, max_bytes=128, context=_CONTEXT)

        assert type(first) is dict
        assert type(first["outer"]) is dict
        assert first == second
        assert first is not second
        first_outer = cast("dict[str, object]", first["outer"])
        second_outer = cast("dict[str, object]", second["outer"])
        assert first_outer is not second_outer
        first_outer["new"] = "owned"
        assert "new" not in second_outer

    def test_requires_positive_bound_before_artifact_access(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = cast("ArtifactFile", tmp_path / "missing.json")
        open_file = mocker.patch.object(Path, "open", autospec=True)

        for max_bytes in (0, -1):
            with pytest.raises(ValueError, match="max_bytes must be positive"):
                read_json_object(path, max_bytes=max_bytes, context=_CONTEXT)

        open_file.assert_not_called()

    def test_issues_one_bounded_binary_read_and_closes(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = cast("ArtifactFile", tmp_path / "metadata.json")
        stream = _ManagedStream(b"{}")
        open_file = mocker.patch.object(Path, "open", autospec=True, return_value=stream)
        resolve = mocker.patch.object(Path, "resolve", autospec=True)
        stat = mocker.patch.object(Path, "stat", autospec=True)
        is_file = mocker.patch.object(Path, "is_file", autospec=True)

        assert read_json_object(path, max_bytes=2, context=_CONTEXT) == {}

        open_file.assert_called_once_with(path, "rb")
        resolve.assert_not_called()
        stat.assert_not_called()
        is_file.assert_not_called()
        assert stream.read_sizes == [3]
        assert stream.close_attempted
        assert stream.closed

    @pytest.mark.parametrize(
        ("payload", "max_bytes"),
        [(b"{} ", 2), (b'{"value":"\xff"}', 32), (b"{", 1)],
    )
    def test_closes_before_post_read_content_failures(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        payload: bytes,
        max_bytes: int,
    ) -> None:
        path = cast("ArtifactFile", tmp_path / "metadata.json")
        stream = _ManagedStream(payload)
        mocker.patch.object(Path, "open", autospec=True, return_value=stream)

        with pytest.raises(ReaderFormatError):
            read_json_object(path, max_bytes=max_bytes, context=_CONTEXT)

        assert stream.close_attempted
        assert stream.closed

    def test_accepts_exact_limit_and_rejects_one_byte_over_before_decoding(self, tmp_path: Path) -> None:
        payload = b' {"value":1} '
        path = _artifact(tmp_path, payload)

        assert read_json_object(path, max_bytes=len(payload), context=_CONTEXT) == {"value": 1}

        with pytest.raises(ReaderFormatError, match="more than") as raised:
            read_json_object(path, max_bytes=len(payload) - 1, context=_CONTEXT)

        assert raised.value.__cause__ is None
        assert raised.value.path == path

    def test_counts_multibyte_utf8_as_encoded_bytes(self, tmp_path: Path) -> None:
        payload = '{"unit":"µm","symbol":"λ"}'.encode()
        path = _artifact(tmp_path, payload)

        assert read_json_object(path, max_bytes=len(payload), context=_CONTEXT) == {
            "unit": "µm",
            "symbol": "λ",
        }
        with pytest.raises(ReaderFormatError, match="encoded bytes"):
            read_json_object(path, max_bytes=len(payload) - 1, context=_CONTEXT)

    @pytest.mark.parametrize("payload", [b"", b"{", b"{} trailing", b"\xef\xbb\xbf{}"])
    def test_rejects_empty_malformed_and_bom_documents(self, tmp_path: Path, payload: bytes) -> None:
        path = _artifact(tmp_path, payload)

        with pytest.raises(ReaderFormatError, match="malformed JSON") as raised:
            read_json_object(path, max_bytes=max(1, len(payload)), context=_CONTEXT)

        assert isinstance(raised.value.__cause__, ValueError)

    def test_rejects_invalid_utf8_with_decoder_cause(self, tmp_path: Path) -> None:
        path = _artifact(tmp_path, b'{"value":"\xff"}')

        with pytest.raises(ReaderFormatError, match="not valid UTF-8") as raised:
            read_json_object(path, max_bytes=32, context=_CONTEXT)

        assert isinstance(raised.value.__cause__, UnicodeDecodeError)

    @pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
    def test_rejects_nonstandard_nonfinite_constants(self, tmp_path: Path, token: str) -> None:
        path = _artifact(tmp_path, f'{{"value":{token}}}'.encode())

        with pytest.raises(ReaderFormatError, match="nonstandard non-finite") as raised:
            read_json_object(path, max_bytes=64, context=_CONTEXT)

        assert isinstance(raised.value.__cause__, ValueError)

    @pytest.mark.parametrize("token", ["1e400", "-1e400"])
    def test_rejects_finite_token_overflow(self, tmp_path: Path, token: str) -> None:
        path = _artifact(tmp_path, f'{{"value":{token}}}'.encode())

        with pytest.raises(ReaderFormatError, match="finite float range") as raised:
            read_json_object(path, max_bytes=64, context=_CONTEXT)

        assert isinstance(raised.value.__cause__, ValueError)

    @pytest.mark.parametrize(
        "payload",
        [
            b'{"name":1,"name":2}',
            b'{"outer":{"name":1,"name":2}}',
            b'{"items":[{"name":1,"name":2}]}',
            b'{"name":1,"\\u006eame":2}',
        ],
    )
    def test_rejects_duplicate_decoded_names_at_every_nesting_position(self, tmp_path: Path, payload: bytes) -> None:
        path = _artifact(tmp_path, payload)

        with pytest.raises(ReaderFormatError, match="duplicate object name") as raised:
            read_json_object(path, max_bytes=128, context=_CONTEXT)

        assert isinstance(raised.value.__cause__, ValueError)

    @pytest.mark.parametrize(
        ("payload", "kind"),
        [
            (b"[]", "array"),
            (b'"value"', "string"),
            (b"1", "number"),
            (b"1.5", "number"),
            (b"true", "boolean"),
            (b"null", "null"),
        ],
    )
    def test_rejects_every_non_object_root_family(self, tmp_path: Path, payload: bytes, kind: str) -> None:
        path = _artifact(tmp_path, payload)

        with pytest.raises(ReaderFormatError, match=f"top-level JSON {kind}") as raised:
            read_json_object(path, max_bytes=16, context=_CONTEXT)

        assert raised.value.__cause__ is None

    @pytest.mark.parametrize(
        ("parser_error", "expected_message"),
        [(ValueError("integer limit"), "numeric value beyond"), (RecursionError("deep"), "nesting beyond")],
    )
    def test_converts_parser_limit_failures(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        parser_error: Exception,
        expected_message: str,
    ) -> None:
        path = _artifact(tmp_path, b"{}")
        mocker.patch.object(_json.json, "loads", side_effect=parser_error)

        with pytest.raises(ReaderFormatError, match=expected_message) as raised:
            read_json_object(path, max_bytes=2, context=_CONTEXT)

        assert raised.value.__cause__ is parser_error

    def test_does_not_hide_memory_error(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = _artifact(tmp_path, b"{}")
        error = MemoryError("allocation failed")
        mocker.patch.object(_json.json, "loads", side_effect=error)

        with pytest.raises(MemoryError) as raised:
            read_json_object(path, max_bytes=2, context=_CONTEXT)

        assert raised.value is error

    @pytest.mark.parametrize("stage", ["open", "read", "close"])
    def test_preserves_and_contextualizes_io_failures(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        stage: str,
    ) -> None:
        path = cast("ArtifactFile", tmp_path / "metadata.json")
        cause = RuntimeError("underlying cause")
        error = PermissionError(f"{stage} failed")
        error.__cause__ = cause
        stream = _ManagedStream(
            b"{}",
            read_error=error if stage == "read" else None,
            close_error=error if stage == "close" else None,
        )
        if stage == "open":
            mocker.patch.object(Path, "open", autospec=True, side_effect=error)
        else:
            mocker.patch.object(Path, "open", autospec=True, return_value=stream)

        with pytest.raises(PermissionError) as raised:
            read_json_object(path, max_bytes=2, context=_CONTEXT)

        assert raised.value is error
        assert raised.value.__cause__ is cause
        assert raised.value.__traceback__ is not None
        assert raised.value.__notes__ == [
            f"While handling {_CONTEXT.describe(path=path)}.",
        ]
        if stage != "open":
            assert stream.close_attempted
        if stage == "read":
            assert stream.closed

    @pytest.mark.parametrize(("payload", "max_bytes", "oversized"), [(b"{}", 2, False), (b"{} ", 2, True)])
    def test_buffered_short_raw_reads_cannot_truncate_or_bypass_bound(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        payload: bytes,
        max_bytes: int,
        *,
        oversized: bool,
    ) -> None:
        path = cast("ArtifactFile", tmp_path / "metadata.json")
        raw = _ShortRawStream(payload, chunk_size=1)
        buffered = io.BufferedReader(raw, buffer_size=2)
        mocker.patch.object(Path, "open", autospec=True, return_value=buffered)

        if oversized:
            with pytest.raises(ReaderFormatError, match="encoded bytes"):
                read_json_object(path, max_bytes=max_bytes, context=_CONTEXT)
        else:
            assert read_json_object(path, max_bytes=max_bytes, context=_CONTEXT) == {}

        assert buffered.closed
        assert raw.closed

    def test_signature_is_narrow(self) -> None:
        signature = inspect.signature(read_json_object)
        parameters = signature.parameters

        assert set(parameters) == {"path", "max_bytes", "context"}
        assert parameters["path"].kind is inspect.Parameter.POSITIONAL_ONLY
        assert parameters["max_bytes"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["context"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["max_bytes"].default is inspect.Parameter.empty
        assert parameters["context"].default is inspect.Parameter.empty
        assert get_type_hints(read_json_object, localns={"ArtifactFile": ArtifactFile}) == {
            "path": ArtifactFile,
            "max_bytes": int,
            "context": ReaderErrorContext,
            "return": dict[str, object],
        }
