"""Tests for strict UTF-8 decoding of HDF5 attribute values."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.metadata import decode_hdf5_text

_CONTEXT = ReaderErrorContext("test_reader", "count metadata", "2.0")
_PATH = Path("counts.h5")


def _decode(value: object) -> str:
    return decode_hdf5_text(value, attribute="software_version", context=_CONTEXT, path=_PATH)


class TestDecodeHdf5Text:
    """Accept both h5py string forms and reject everything that is not nonempty UTF-8."""

    @pytest.mark.parametrize(
        "value",
        [
            "spaceranger-2.1.0",
            np.str_("spaceranger-2.1.0"),
            b"spaceranger-2.1.0",
            np.bytes_(b"spaceranger-2.1.0"),
            np.asarray(b"spaceranger-2.1.0"),
            np.asarray("spaceranger-2.1.0"),
        ],
        ids=["str", "numpy-str", "bytes", "numpy-bytes", "0d-bytes", "0d-str"],
    )
    def test_decodes_text(self, value: object) -> None:
        result = _decode(value)

        assert result == "spaceranger-2.1.0"
        assert type(result) is str

    @pytest.mark.parametrize("value", ["Gewebe µm", "Gewebe µm".encode()], ids=["str", "bytes"])
    def test_decodes_non_ascii_utf8(self, value: object) -> None:
        assert _decode(value) == "Gewebe µm"

    def test_keeps_nul_characters(self) -> None:
        assert _decode(b"a\x00b") == "a\x00b"

    @pytest.mark.parametrize(
        "value",
        [b"\xff", np.bytes_(b"a\xffb"), np.asarray(b"\xc3"), "\udcff", "a\udcc3b"],
        ids=["bytes", "numpy-bytes", "0d-bytes", "surrogate-escaped", "surrogate-escaped-inner"],
    )
    def test_rejects_invalid_utf8(self, value: object) -> None:
        with pytest.raises(ReaderFormatError, match="non-UTF-8 bytes in attribute 'software_version'") as caught:
            _decode(value)

        assert caught.value.context is _CONTEXT
        assert caught.value.path == _PATH
        assert isinstance(caught.value.__cause__, UnicodeError)

    def test_rejects_a_surrogate_that_surrogateescape_cannot_produce(self) -> None:
        with pytest.raises(ReaderFormatError, match="non-UTF-8 bytes") as caught:
            _decode("\ud800")

        assert isinstance(caught.value.__cause__, UnicodeEncodeError)

    def test_rejects_h5py_variable_length_invalid_utf8(self, tmp_path: Path) -> None:
        path = tmp_path / "counts.h5"
        with h5py.File(path, "w") as source:
            source.attrs["software_version"] = b"\xff"
            stored = source.attrs["software_version"]

        # h5py stores bytes as a variable-length string and decodes it with surrogateescape.
        assert stored == "\udcff"
        with pytest.raises(ReaderFormatError, match="non-UTF-8 bytes"):
            _decode(stored)

    @pytest.mark.parametrize(
        ("value", "kind"),
        [
            (1, "int"),
            (np.int64(1), "int64"),
            (None, "NoneType"),
            (np.asarray([b"a"]), "ndarray"),
            (np.asarray(1), "int64"),
            ([b"a"], "list"),
        ],
        ids=["int", "numpy-int", "none", "1d-array", "0d-int", "list"],
    )
    def test_rejects_non_text_values(self, value: object, kind: str) -> None:
        with pytest.raises(ReaderFormatError, match=f"attribute 'software_version' with value type {kind}") as caught:
            _decode(value)

        assert caught.value.context is _CONTEXT
        assert caught.value.path == _PATH

    @pytest.mark.parametrize("value", [b"", "", np.bytes_(b""), np.asarray(b"")], ids=["bytes", "str", "numpy", "0d"])
    def test_rejects_empty_text(self, value: object) -> None:
        with pytest.raises(ReaderFormatError, match="empty attribute 'software_version'"):
            _decode(value)
