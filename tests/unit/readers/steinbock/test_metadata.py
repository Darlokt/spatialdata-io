"""Projected, strictly parsed IMC sidecar metadata and file ownership."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.steinbock._metadata import read_image_metadata

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

_HEADER = b"image,width_px,height_px,acquisition_width_um,acquisition_height_um\n"


class TestReadImageMetadata:
    """Read text once, close on every result and reject malformed source syntax."""

    @pytest.mark.parametrize("row", [b"s.tiff,12,8,6,20\n", b"s.tiff,12,8,6\n"])
    def test_one_owned_read_closes_on_success_and_failure(
        self, tmp_path: Path, mocker: MockerFixture, row: bytes
    ) -> None:
        owner = io.BytesIO(_HEADER + row)
        opened = mocker.patch("pathlib.Path.open", return_value=owner)
        path = ArtifactFile(tmp_path / "images.csv")
        if row.endswith(b"6\n"):
            with pytest.raises(ReaderFormatError, match="fields"):
                read_image_metadata(path, {"s.tiff"})
        else:
            metadata = read_image_metadata(path, {"s.tiff"})
            assert metadata["s.tiff"].shape == (8, 12)
            assert metadata["s.tiff"].spacing_um == (0.5, 2.5)
        assert owner.closed
        opened.assert_called_once_with("rb")

    @pytest.mark.parametrize(
        "content",
        [
            b"\xff",
            b"\x00",
            _HEADER + b"s.tiff,12,8,6,20\x00\n",
            _HEADER + b'"s.tiff,12,8,6,20\n',
            b"",
            b"image,width_px,height_px\ns.tiff,12,8,extra\n",
        ],
    )
    def test_invalid_source(self, tmp_path: Path, content: bytes) -> None:
        path = tmp_path / "images.csv"
        path.write_bytes(content)
        with pytest.raises(ReaderFormatError, match=r"images\.csv"):
            read_image_metadata(ArtifactFile(path), {"s.tiff"})

    def test_io_error_is_preserved_with_context(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch("pathlib.Path.open", side_effect=PermissionError("denied"))
        with pytest.raises(PermissionError) as caught:
            read_image_metadata(ArtifactFile(tmp_path / "images.csv"), set())
        assert any("steinbock" in note and "images.csv" in note for note in caught.value.__notes__)

    def test_empty_physical_fields_and_unselected_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "images.csv"
        path.write_bytes(_HEADER + b"s.tiff,12,8,,\nother.tiff,unused,unused,unused,unused\n")
        metadata = read_image_metadata(ArtifactFile(path), {"s.tiff", "absent.tiff"})
        assert set(metadata) == {"s.tiff"}
        assert metadata["s.tiff"].spacing_um is None

    def test_present_dimensions_without_physical_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "images.csv"
        path.write_text("height_px,extra,image,width_px\n8,ignored,s.tiff,12\n", encoding="utf-8")
        metadata = read_image_metadata(ArtifactFile(path), {"s.tiff"})
        assert metadata["s.tiff"].shape == (8, 12)
        assert metadata["s.tiff"].spacing_um is None
