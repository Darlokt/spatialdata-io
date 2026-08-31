"""Synthetic raster fixtures for centralized reader integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pytest
import tifffile

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class TiffOptions:
    """Storage and metadata options for one synthetic TIFF."""

    name: str = "image.tif"
    compression: str | None = None
    tile: tuple[int, int] | None = None
    rowsperstrip: int | None = None
    axes: str = "CYX"
    bigtiff: bool = False
    ome: bool = False


_DEFAULT_TIFF_OPTIONS = TiffOptions()


class TiffFactory(Protocol):
    """Write one small TIFF with explicit storage and axes metadata."""

    def __call__(
        self,
        data: NDArray[np.generic],
        options: TiffOptions = _DEFAULT_TIFF_OPTIONS,
    ) -> Path: ...


@pytest.fixture
def cyx_data() -> NDArray[np.uint16]:
    return np.arange(3 * 19 * 23, dtype=np.uint16).reshape(3, 19, 23)


@pytest.fixture
def write_tiff(tmp_path: Path) -> TiffFactory:
    def factory(
        data: NDArray[np.generic],
        options: TiffOptions = _DEFAULT_TIFF_OPTIONS,
    ) -> Path:
        path = tmp_path / options.name
        tifffile.imwrite(
            path,
            data,
            compression=options.compression,
            tile=options.tile,
            rowsperstrip=options.rowsperstrip,
            photometric="minisblack",
            metadata={"axes": options.axes},
            bigtiff=options.bigtiff,
            ome=options.ome,
        )
        return path

    return factory


@pytest.fixture
def linked_ome_tiff(tmp_path: Path) -> tuple[Path, NDArray[np.uint16]]:
    """Write a deterministic two-file, two-plane linked OME-TIFF."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    first_path = dataset / "first.ome.tif"
    second_path = dataset / "second.ome.tif"
    first_uuid = "11111111-1111-4111-8111-111111111111"
    second_uuid = "22222222-2222-4222-8222-222222222222"
    ome_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
  UUID="urn:uuid:{first_uuid}"><Image ID="Image:0"><Pixels ID="Pixels:0"
  DimensionOrder="XYZCT" Type="uint16" SizeX="5" SizeY="4" SizeZ="1"
  SizeC="1" SizeT="2"><Channel ID="Channel:0:0" SamplesPerPixel="1"/>
  <TiffData IFD="0" FirstT="0" PlaneCount="1"><UUID FileName="first.ome.tif">
  urn:uuid:{first_uuid}</UUID></TiffData>
  <TiffData IFD="0" FirstT="1" PlaneCount="1"><UUID FileName="second.ome.tif">
  urn:uuid:{second_uuid}</UUID></TiffData></Pixels></Image></OME>"""
    data = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    tifffile.imwrite(first_path, data[0], description=ome_xml, metadata=None)
    second_xml = ome_xml.replace(first_uuid, second_uuid, 1)
    tifffile.imwrite(second_path, data[1], description=second_xml, metadata=None)
    return first_path, data
