"""Integration tests for shared whole-frame task execution and resources."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import imagecodecs
import numpy as np
import pytest

import spatialdata_io.readers._utils.image._encoded as encoded
from spatialdata_io.readers._utils.image import read_jpeg, read_png
from spatialdata_io.readers._utils.image._exceptions import RasterFormatError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Protocol

    from numpy.typing import NDArray
    from pytest_mock import MockerFixture

    from spatialdata_io.readers._utils.image import LazyRaster

    type EncodedReader = Callable[[str | Path], LazyRaster]

    class ClosedResource(Protocol):
        """Resource exposing closure state after task execution."""

        @property
        def closed(self) -> bool: ...


def _write_encoded(path: Path, codec: str) -> None:
    source = np.arange(99, dtype=np.uint8).reshape(9, 11)
    payload = imagecodecs.png_encode(source) if codec == "png" else imagecodecs.jpeg8_encode(source)
    path.write_bytes(payload)


class TestEncodedResult:
    """Own one task-local memory map and validate forced decoder output."""

    @pytest.mark.parametrize(
        "case",
        [
            ("png", read_png, "png_decode", ".png"),
            ("jpeg", read_jpeg, "jpeg8_decode", ".jpg"),
        ],
    )
    def test_construction_is_lazy_and_small_slice_decodes_whole_frame_once(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        case: tuple[str, EncodedReader, str, str],
    ) -> None:
        codec, reader, decoder_name, suffix = case
        path = tmp_path / f"image{suffix}"
        _write_encoded(path, codec)
        spy = mocker.spy(encoded.imagecodecs, decoder_name)

        result = reader(path)

        assert spy.call_count == 0
        pickle.dumps(result.data)
        result.data[:2, :3].compute(scheduler="synchronous")
        assert spy.call_count == 1

    def test_closes_memory_map_before_shape_validation_failure(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        path = tmp_path / "image.png"
        _write_encoded(path, "png")
        mapped_inputs: list[ClosedResource] = []

        def wrong_shape(mapped: ClosedResource) -> NDArray[np.uint8]:
            mapped_inputs.append(mapped)
            return np.zeros((1, 1), dtype=np.uint8)

        mocker.patch.object(encoded.imagecodecs, "png_decode", side_effect=wrong_shape)
        result = read_png(path)

        with pytest.raises(RasterFormatError, match="does not match"):
            result.data.compute(scheduler="synchronous")
        assert mapped_inputs
        assert mapped_inputs[0].closed is True

    def test_closes_memory_map_and_adds_context_after_decoder_failure(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        path = tmp_path / "image.jpg"
        _write_encoded(path, "jpeg")
        mapped_inputs: list[ClosedResource] = []

        def fail(mapped: ClosedResource) -> NDArray[np.uint8]:
            mapped_inputs.append(mapped)
            message = "decoder failed"
            raise RuntimeError(message)

        mocker.patch.object(encoded.imagecodecs, "jpeg8_decode", side_effect=fail)
        result = read_jpeg(path)

        with pytest.raises(RuntimeError, match="decoder failed") as caught:
            result.data.compute(scheduler="synchronous")
        assert any("JPEG source" in note for note in caught.value.__notes__)
        assert mapped_inputs
        assert mapped_inputs[0].closed is True
