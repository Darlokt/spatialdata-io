"""Tests for shared private reader errors."""

from __future__ import annotations

import json
import pickle
from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers import _utils
from spatialdata_io.readers._utils import errors
from spatialdata_io.readers._utils.errors import (
    RasterFormatError,
    ReaderErrorContext,
    ReaderFormatError,
    add_reader_context,
)

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType


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


class TestErrorPackage:
    def test_exports_shared_reader_errors(self) -> None:
        assert errors.__all__ == [
            "RasterFormatError",
            "ReaderErrorContext",
            "ReaderFormatError",
            "add_reader_context",
        ]
        assert issubclass(RasterFormatError, ValueError)
        assert not hasattr(_utils, "ReaderFormatError")
