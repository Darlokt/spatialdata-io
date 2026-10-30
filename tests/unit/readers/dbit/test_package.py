"""DBiT package exports and explicit public signature."""

from __future__ import annotations

import importlib
import inspect

import pytest

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers.dbit import dbit


class TestPackage:
    """Preserve public imports while exposing only the supported reader contract."""

    def test_public_entry_point_and_root_identity(self) -> None:
        package = importlib.import_module("spatialdata_io.readers.dbit")
        implementation = importlib.import_module("spatialdata_io.readers.dbit._reader")
        root = importlib.import_module("spatialdata_io")

        assert package.__all__ == ["dbit"]
        assert package.dbit is implementation.dbit is root.dbit

    def test_required_path_and_explicit_keyword_only_options(self) -> None:
        parameters = inspect.signature(dbit).parameters
        assert list(parameters) == [
            "path",
            "table_file",
            "barcode_positions_file",
            "image_file",
            "dataset_id",
            "include_border",
            "border_scale",
            "invalid_geometries",
            "data_axes",
            "tiff_series",
            "tiff_level",
            "scale_factors",
            "chunks",
        ]
        assert parameters["path"].default is inspect.Parameter.empty
        assert parameters["path"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert all(
            parameter.kind == inspect.Parameter.KEYWORD_ONLY for name, parameter in parameters.items() if name != "path"
        )

        defaults = {name: parameter.default for name, parameter in parameters.items() if name != "path"}
        assert defaults == {
            "table_file": None,
            "barcode_positions_file": None,
            "image_file": None,
            "dataset_id": None,
            "include_border": True,
            "border_scale": 1.0,
            "invalid_geometries": "keep",
            "data_axes": None,
            "tiff_series": 0,
            "tiff_level": 0,
            "scale_factors": DEFAULT_SCALE_FACTORS,
            "chunks": DEFAULT_CHUNKS,
        }

    def test_path_is_required_at_runtime(self) -> None:
        with pytest.raises(TypeError, match="path"):
            dbit()  # type: ignore[missing-argument, call-arg]  # Exercise the required public argument.

    def test_options_cannot_be_positional(self) -> None:
        with pytest.raises(TypeError, match="positional"):
            dbit("dataset", "table.h5ad")  # type: ignore[bad-argument-count, call-arg]  # Exercise keyword-only options.
