"""Test MERSCOPE reader resource and warning-state ownership."""

from __future__ import annotations

import sys
import warnings
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers.merscope import _rioxarray_load_merscope

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class _NotGeoreferencedWarning(UserWarning):
    pass


class _RasterioErrors(ModuleType):
    NotGeoreferencedWarning = _NotGeoreferencedWarning


class TestRioxarrayLoadMerscope:
    """Test warning-filter scoping around the optional raster backend."""

    @pytest.mark.parametrize("fail", [False, True])
    def test_warning_filters_are_restored(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        *,
        fail: bool,
    ) -> None:
        opened = mocker.MagicMock()
        rioxarray = SimpleNamespace(open_rasterio=mocker.Mock(return_value=opened))
        rasterio = ModuleType("rasterio")
        rasterio_errors = _RasterioErrors("rasterio.errors")
        mocker.patch.dict(
            sys.modules,
            {"rioxarray": rioxarray, "rasterio": rasterio, "rasterio.errors": rasterio_errors},
        )
        before = list(warnings.filters)

        if fail:
            mocker.patch("spatialdata_io.readers.merscope.xarray.concat", side_effect=RuntimeError("injected"))
            with pytest.raises(RuntimeError, match="injected"):
                _rioxarray_load_merscope(tmp_path, ["DAPI"], 0, {"chunks": (1, 16, 16)})
        else:
            image = mocker.sentinel.image
            mocker.patch("spatialdata_io.readers.merscope.xarray.concat", return_value=image)
            parse = mocker.patch("spatialdata_io.readers.merscope.Image2DModel.parse", return_value=image)
            assert _rioxarray_load_merscope(tmp_path, ["DAPI"], 0, {"chunks": (1, 16, 16)}) is image
            parse.assert_called_once()

        assert warnings.filters == before
