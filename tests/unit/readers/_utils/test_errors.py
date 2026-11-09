"""Tests for shared private reader errors."""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from types import FunctionType
from typing import TYPE_CHECKING

import pytest

import spatialdata_io
from spatialdata_io.readers import _utils
from spatialdata_io.readers._utils import errors
from spatialdata_io.readers._utils.errors import (
    RasterFormatError,
    ReaderErrorContext,
    ReaderFormatError,
    add_reader_context,
    warn_user,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

_PACKAGE_DIRECTORY = Path(spatialdata_io.__file__).parent


class TestReaderErrorContext:
    def test_describes_fields(self, tmp_path: Path) -> None:
        context = ReaderErrorContext(reader="xenium", artifact="specification", detected_version="4.0")

        assert context.describe(path=tmp_path) == (
            f"reader 'xenium', artifact 'specification', detected version '4.0', path {str(tmp_path)!r}"
        )

    def test_omits_optional_fields(self) -> None:
        assert ReaderErrorContext(reader="visium", artifact="counts").describe() == (
            "reader 'visium', artifact 'counts'"
        )

    def test_describes_failure(self, tmp_path: Path) -> None:
        context = ReaderErrorContext(reader="curio", artifact="counts")

        assert context.describe_failure(found="two", expected="one", path=tmp_path) == (
            f"reader 'curio', artifact 'counts', path {str(tmp_path)!r}: expected one; found two."
        )
        assert context.describe_failure(found="two", expected="one") == (
            "reader 'curio', artifact 'counts': expected one; found two."
        )

    def test_is_frozen_and_slotted(self) -> None:
        context = ReaderErrorContext(reader="reader", artifact="artifact")

        dataclass_parameters = vars(ReaderErrorContext)["__dataclass_params__"]
        assert dataclass_parameters.frozen
        assert not hasattr(context, "__dict__")


class TestReaderFormatError:
    def test_records_structured_failure(self, tmp_path: Path) -> None:
        context = ReaderErrorContext(reader="visium_hd", artifact="layout", detected_version="3")
        error = ReaderFormatError(
            context=context,
            found="two candidates",
            expected="exactly one candidate",
            path=tmp_path,
        )

        assert isinstance(error, ValueError)
        assert error.context is context
        assert error.found == "two candidates"
        assert error.expected == "exactly one candidate"
        assert error.path == tmp_path
        assert "detected version '3'" in str(error)

    def test_preserves_explicit_cause(self) -> None:
        context = ReaderErrorContext(reader="reader", artifact="metadata")

        def parse() -> None:
            try:
                json.loads("{")
            except json.JSONDecodeError as error:
                raise ReaderFormatError(
                    context=context,
                    found="invalid JSON",
                    expected="a JSON object",
                ) from error

        with pytest.raises(ReaderFormatError) as raised:
            parse()

        assert isinstance(raised.value.__cause__, json.JSONDecodeError)

    def test_pickle_round_trip_preserves_structured_state_and_notes(self, tmp_path: Path) -> None:
        context = ReaderErrorContext(reader="reader", artifact="layout", detected_version="2")
        error = ReaderFormatError(
            context=context,
            found="two candidates",
            expected="exactly one candidate",
            path=tmp_path,
        )
        error.add_note("worker context")

        restored = pickle.loads(pickle.dumps(error))  # noqa: S301, RUF100 - controlled local round trip.

        assert isinstance(restored, ReaderFormatError)
        assert restored.context == context
        assert restored.found == error.found
        assert restored.expected == error.expected
        assert restored.path == error.path
        assert str(restored) == str(error)
        assert restored.__notes__ == ["worker context"]


class TestAddReaderContext:
    def test_preserves_exception_identity_and_traceback(self, tmp_path: Path) -> None:
        context = ReaderErrorContext(reader="reader", artifact="artifact", detected_version="1")
        original = ValueError("failure")
        traceback: TracebackType | None = None

        def annotate_and_reraise() -> None:
            nonlocal traceback
            try:
                raise original
            except ValueError as error:
                traceback = error.__traceback__
                add_reader_context(error, context=context, path=tmp_path / "artifact")
                raise

        with pytest.raises(ValueError, match="failure") as raised:
            annotate_and_reraise()

        assert raised.value is original
        reraised_traceback = raised.value.__traceback__
        while reraised_traceback is not None and reraised_traceback.tb_next is not None:
            reraised_traceback = reraised_traceback.tb_next
        assert reraised_traceback is traceback
        assert "reader 'reader'" in raised.value.__notes__[0]


def _line(offset: int) -> int:
    """Return the caller's current line number plus ``offset``."""
    return sys._getframe(1).f_lineno + offset


def _emit(message: str) -> None:
    warn_user(message)


def _relay(callback: Callable[[str], None], message: str) -> None:
    callback(message)


def _relocated(function: FunctionType, filename: Path) -> FunctionType:
    """Return ``function`` with its frames reported under ``filename``."""
    return FunctionType(function.__code__.replace(co_filename=str(filename)), function.__globals__)


class TestWarnUser:
    def test_emits_the_message_unchanged_as_a_user_warning_at_the_caller(self) -> None:
        message = "reader: {braces}, %s and 'quotes' are kept."

        line = _line(2)
        with pytest.warns(UserWarning, match="reader") as caught:
            warn_user(message)

        assert [(warning.category, str(warning.message)) for warning in caught] == [(UserWarning, message)]
        assert (caught[0].filename, caught[0].lineno) == (__file__, line)

    def test_nested_package_frames_are_skipped(self) -> None:
        emit = _relocated(_emit, _PACKAGE_DIRECTORY / "readers" / "_emit.py")
        relay = _relocated(_relay, _PACKAGE_DIRECTORY / "readers" / "reader" / "_relay.py")

        line = _line(2)
        with pytest.warns(UserWarning, match="nested") as caught:
            relay(emit, "reader: nested")

        assert (caught[0].filename, caught[0].lineno) == (__file__, line)

    def test_the_first_frame_outside_the_package_is_named(self) -> None:
        emit = _relocated(_emit, _PACKAGE_DIRECTORY / "_emit.py")
        relay = _relocated(_relay, _PACKAGE_DIRECTORY / "_relay.py")

        def callback(message: str) -> None:
            emit(message)

        with pytest.warns(UserWarning, match="callback") as caught:
            relay(callback, "reader: through a callback")

        assert (caught[0].filename, caught[0].lineno) == (__file__, callback.__code__.co_firstlineno + 1)

    def test_a_sibling_directory_sharing_the_package_name_is_outside(self) -> None:
        sibling = Path(f"{_PACKAGE_DIRECTORY}_extension") / "_emit.py"
        emit = _relocated(_emit, sibling)

        with pytest.warns(UserWarning, match="sibling") as caught:
            emit("reader: sibling")

        assert f"{_PACKAGE_DIRECTORY}{os.sep}" == errors._PACKAGE_PREFIX
        assert caught[0].filename == str(sibling)


class TestErrorPackage:
    def test_exports_shared_reader_errors(self) -> None:
        assert errors.__all__ == [
            "RasterFormatError",
            "ReaderErrorContext",
            "ReaderFormatError",
            "add_reader_context",
            "warn_user",
        ]
        assert issubclass(RasterFormatError, ValueError)
        assert not hasattr(_utils, "ReaderFormatError")
