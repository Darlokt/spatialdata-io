"""Test Xenium identifier, resource, and CLI contracts."""

from __future__ import annotations

import json
import zipfile
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import packaging.version
import pytest
import tifffile

from spatialdata_io._cli.commands.xenium import xenium_command
from spatialdata_io.readers.xenium import (
    _cell_id_str_from_prefix_suffix_uint32_reference,
    _get_labels,
    _get_morphology_focus,
    _XeniumCells,
    cell_id_str_from_prefix_suffix_uint32,
    prefix_suffix_uint32_from_cell_id_str,
    xenium,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Self

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


def test_cell_id_str_from_prefix_suffix_uint32() -> None:
    """Convert encoded integer cell identifiers to their string form."""
    cell_id_prefix = np.array([1, 1437536272, 1437536273], dtype=np.uint32)
    dataset_suffix = np.array([1, 1, 2])
    expected = np.array(["aaaaaaab-1", "ffkpbaba-1", "ffkpbabb-2"])

    result = cell_id_str_from_prefix_suffix_uint32(cell_id_prefix, dataset_suffix)
    reference = _cell_id_str_from_prefix_suffix_uint32_reference(cell_id_prefix, dataset_suffix)
    assert np.array_equal(result, expected)
    assert np.array_equal(reference, expected)


def test_cell_id_str_optimized_matches_reference() -> None:
    """Match the optimized cell-ID conversion to its reference."""
    rng = np.random.default_rng(42)
    cell_id_prefix = rng.integers(0, 2**32, size=10_000, dtype=np.uint32)
    dataset_suffix = rng.integers(0, 10, size=10_000)

    result = cell_id_str_from_prefix_suffix_uint32(cell_id_prefix, dataset_suffix)
    reference = _cell_id_str_from_prefix_suffix_uint32_reference(cell_id_prefix, dataset_suffix)
    assert np.array_equal(result, reference)


def test_prefix_suffix_uint32_from_cell_id_str() -> None:
    """Decode string cell identifiers to their integer components."""
    cell_id_str = np.array(["aaaaaaab-1", "ffkpbaba-1", "ffkpbabb-2"])

    cell_id_prefix, dataset_suffix = prefix_suffix_uint32_from_cell_id_str(cell_id_str)
    assert np.array_equal(cell_id_prefix, np.array([1, 1437536272, 1437536273], dtype=np.uint32))
    assert np.array_equal(dataset_suffix, np.array([1, 1, 2]))


def test_roundtrip_with_data_limits() -> None:
    """Round-trip the full uint32 identifier range."""
    # min and max values for uint32
    cell_id_prefix = np.array([0, 4294967295], dtype=np.uint32)
    dataset_suffix = np.array([1, 1])
    cell_id_str = np.array(["aaaaaaaa-1", "pppppppp-1"])
    f0 = cell_id_str_from_prefix_suffix_uint32
    f1 = prefix_suffix_uint32_from_cell_id_str
    assert np.array_equal(cell_id_prefix, f1(f0(cell_id_prefix, dataset_suffix))[0])
    assert np.array_equal(dataset_suffix, f1(f0(cell_id_prefix, dataset_suffix))[1])
    assert np.array_equal(cell_id_str, f0(*f1(cell_id_str)))


class _TrackingZipStore:
    def __init__(self) -> None:
        self.closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.closed = True


class _Group:
    def __init__(self, members: set[str] | None = None) -> None:
        self.members = members or set()

    def __contains__(self, member: object) -> bool:
        return member in self.members


def _write_labels_zip(path: Path, data: np.ndarray, chunks: tuple[int, int]) -> None:
    metadata = {
        "chunks": list(chunks),
        "compressor": None,
        "dtype": data.dtype.str,
        "fill_value": 0,
        "filters": None,
        "order": "C",
        "shape": list(data.shape),
        "zarr_format": 2,
    }
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr(".zgroup", json.dumps({"zarr_format": 2}))
        archive.writestr("masks/.zgroup", json.dumps({"zarr_format": 2}))
        archive.writestr("masks/0/.zarray", json.dumps(metadata))
        for row in range(0, data.shape[0], chunks[0]):
            for column in range(0, data.shape[1], chunks[1]):
                chunk = data[row : row + chunks[0], column : column + chunks[1]]
                archive.writestr(f"masks/0/{row // chunks[0]}.{column // chunks[1]}", chunk.tobytes())


class TestXeniumCells:
    """Test explicit ownership and lazy reopening of Xenium cell storage."""

    def test_store_closes_after_success(self, tmp_path: Path, mocker: MockerFixture) -> None:
        store = _TrackingZipStore()
        mocker.patch("spatialdata_io.readers.xenium.ZipStore", return_value=store)
        mocker.patch("spatialdata_io.readers.xenium.zarr.open_group", return_value=_Group())

        with _XeniumCells.open(tmp_path, version=None):
            assert not store.closed

        assert store.closed

    def test_store_closes_when_layout_validation_fails(self, tmp_path: Path, mocker: MockerFixture) -> None:
        store = _TrackingZipStore()
        mocker.patch("spatialdata_io.readers.xenium.ZipStore", return_value=store)
        mocker.patch(
            "spatialdata_io.readers.xenium.zarr.open_group",
            return_value=_Group({"polygon_sets", "seg_mask_value"}),
        )

        with pytest.raises(ValueError, match="mutually exclusive"):
            _XeniumCells.open(tmp_path, version=None)

        assert store.closed

    def test_lazy_labels_reopen_store_after_context_exit(self, tmp_path: Path) -> None:
        expected = np.arange(24, dtype=np.uint16).reshape(4, 6)
        store_path = tmp_path / "cells.zarr.zip"
        _write_labels_zip(store_path, expected, chunks=(2, 3))

        with _XeniumCells.open(tmp_path, version=None) as cells:
            labels = _get_labels(cells, mask_index=0)

        np.testing.assert_array_equal(labels.data.compute(), expected)


class TestXeniumOptions:
    def test_raster_model_options_are_not_mutated_on_layout_failure(self, tmp_path: Path) -> None:
        image_options = MappingProxyType({"chunks": (1, 8, 8), "scale_factors": [4, 4]})
        labels_options = MappingProxyType({"chunks": (8, 8), "scale_factors": [4, 4]})

        with pytest.raises(FileNotFoundError):
            xenium(
                tmp_path,
                image_models_kwargs=image_options,
                labels_models_kwargs=labels_options,
            )

        assert dict(image_options) == {"chunks": (1, 8, 8), "scale_factors": [4, 4]}
        assert dict(labels_options) == {"chunks": (8, 8), "scale_factors": [4, 4]}


class TestGetMorphologyFocus:
    """Test temporary tifffile logger-filter ownership."""

    @pytest.mark.parametrize("fail", [False, True])
    def test_tifffile_filter_is_removed(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        *,
        fail: bool,
    ) -> None:
        filename = "morphology_focus_0000.ome.tif"
        mocker.patch("spatialdata_io.readers.xenium.os.listdir", return_value=[filename])
        logger = tifffile.logger()
        before = list(logger.filters)
        result = mocker.sentinel.image

        def get_images(*_: object, **__: object) -> object:
            assert len(logger.filters) == len(before) + 1
            if fail:
                message = "injected"
                raise RuntimeError(message)
            return result

        mocker.patch("spatialdata_io.readers.xenium._get_images", side_effect=get_images)
        version = packaging.version.parse("2.0.0")
        if fail:
            with pytest.raises(RuntimeError, match="injected"):
                _get_morphology_focus(tmp_path, version)
        else:
            assert _get_morphology_focus(tmp_path, version) is result

        assert logger.filters == before


@pytest.mark.parametrize(
    "kwarg_name",
    ["--imread-kwargs", "--image-models-kwargs", "--labels-models-kwargs"],
)
def test_cli_xenium_invalid_json_rejected(runner: CliRunner, tmp_path: Path, kwarg_name: str) -> None:
    """Invalid JSON for any kwargs option must produce a non-zero exit and a clear error."""
    result = runner.invoke(
        xenium_command,
        [
            "--input",
            str(tmp_path),
            "--output",
            str(tmp_path / "out.zarr"),
            kwarg_name,
            "not-valid-json{",
        ],
    )
    assert result.exit_code != 0
    assert "Invalid JSON" in result.output


@pytest.mark.parametrize(
    ("kwarg_name", "kwarg_param"),
    [
        ("--imread-kwargs", "imread_kwargs"),
        ("--image-models-kwargs", "image_models_kwargs"),
        ("--labels-models-kwargs", "labels_models_kwargs"),
    ],
)
def test_cli_xenium_valid_json_forwarded(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture, kwarg_name: str, kwarg_param: str
) -> None:
    """Valid JSON kwargs must be parsed and forwarded to the xenium reader as a dict."""
    mock_xenium = mocker.patch("spatialdata_io.readers.xenium.xenium")
    mock_xenium.return_value = mocker.MagicMock()
    result = runner.invoke(
        xenium_command,
        [
            "--input",
            str(tmp_path),
            "--output",
            str(tmp_path / "out.zarr"),
            kwarg_name,
            '{"chunks": 512}',
        ],
    )
    assert result.exit_code == 0, result.output
    call_kwargs = mock_xenium.call_args.kwargs
    assert call_kwargs[kwarg_param] == {"chunks": 512}
