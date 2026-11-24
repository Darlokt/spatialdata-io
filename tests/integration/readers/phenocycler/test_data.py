"""Check real QPI scans against the TIFFs distributed by their producers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

import dask.array as da
import numpy as np
import pytest
import tifffile
from spatialdata import read_zarr
from spatialdata.models import get_channel_names
from spatialdata.transformations import BaseTransformation, get_transformation
from xarray import DataArray

from spatialdata_io import codex, phenocycler
from spatialdata_io._cli import cli

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from click.testing import CliRunner
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class _PublicScan:
    filename: str
    region: str
    generation: str
    version: str
    shape: tuple[int, int, int]
    first_channels: tuple[str, ...]
    pixels_per_cm: tuple[int, int]
    stage_position_cm: tuple[float, float]
    window_origin: tuple[int, int]
    reagent: tuple[int, dict[str, str | int | float]] | None = None


class TestPhenoCyclerPublicScans:
    @pytest.mark.parametrize(
        ("dataset", "scan"),
        [
            (
                "phenocycler_fusion1_aml",
                _PublicScan(
                    "AML2_183_Scan1.qptiff",
                    "AML2_183",
                    "fusion1",
                    "4",
                    (54, 10800, 7680),
                    ("DAPI", "NPM1C"),
                    (4294967295, 217731),
                    (4294967295 / 2404900755, 4294967295 / 885749082),
                    (5120, 3584),
                    (
                        1,
                        {
                            "profile_cycle": 2,
                            "profile_channel": 2,
                            "profile_exposure_ms": 200.0,
                            "profile_clone": "x",
                            "profile_reporter": "RX002",
                            "profile_reagent_barcode": "BX002",
                            "profile_dye": "ATTO550",
                        },
                    ),
                ),
            ),
            (
                "phenocycler_fusion2_opal",
                _PublicScan(
                    "Tonsil-ctrl_Scan1.qptiff",
                    "Tonsil-ctrl",
                    "fusion2",
                    "6",
                    (6, 11520, 19200),
                    ("DAPI", "Opal 690", "Opal 620", "Opal 780", "Opal 520", "Sample AF"),
                    (4294967295, 214482),
                    (3778371071 / 4294967295, 4294967295 / 1040289779),
                    (5632, 6656),
                ),
            ),
            (
                "phenocycler_cim_tonsil",
                _PublicScan(
                    "Arlene_Tonsil_9x9.qptiff",
                    "Arlene_Tonsil_9x9",
                    "cim_qptiff",
                    "3",
                    (43, 9508, 12672),
                    ("DAPI-01", "HLA-A"),
                    (4294967295, 162093),
                    (0.0, 0.0),
                    (4608, 6144),
                    (
                        0,
                        {
                            "profile_cycle": 1,
                            "profile_channel": 1,
                            "profile_exposure_ms": 10.0,
                            "profile_wavelength_nm": 358.0,
                            "profile_dye": "CH1 Fluor DAPI",
                            "profile_marker_class": "Nuclear",
                        },
                    ),
                ),
            ),
            (
                "phenocycler_vectra_brightfield",
                _PublicScan(
                    "HandEcompressed_Scan1.qptiff",
                    "HandEcompressed",
                    "qpi_other",
                    "2",
                    (3, 26640, 30720),
                    ("r", "g", "b"),
                    (4294967295, 214273),
                    (2291731967 / 4294967295, 4294967295 / 883248388),
                    (13312, 15360),
                ),
            ),
        ],
        ids=["fusion1", "fusion2-opal", "cim", "brightfield"],
    )
    def test_reads_baseline_channels(
        self, require_test_dataset: Callable[[str], Path], dataset: str, scan: _PublicScan
    ) -> None:
        source = require_test_dataset(dataset) / scan.filename
        sdata = phenocycler(source, scale_factors=(), chunks=(1, 512, 512))
        element = f"{scan.region}_image"
        assert list(sdata.images) == [element]
        image = sdata.images[element]
        assert isinstance(image, DataArray)
        assert isinstance(image.data, da.Array)
        assert image.dims == ("c", "y", "x")
        assert image.shape == scan.shape
        assert image.dtype == np.uint8
        names = get_channel_names(image)
        assert tuple(names[: len(scan.first_channels)]) == scan.first_channels
        assert len(set(names)) == scan.shape[0]
        assert sdata.attrs["spatialdata_io_reader"] == "phenocycler"
        assert sdata.attrs["spatialdata_io_format_version"] == scan.version
        metadata = sdata.attrs["akoya"]["regions"][scan.region]
        assert metadata["generation"] == scan.generation
        assert metadata["description_version"] == scan.version
        if scan.reagent is not None:
            position, reagent = scan.reagent
            channel = metadata["channels"][position]
            assert {field: channel[field] for field in reagent} == reagent

        numerator, denominator = scan.pixels_per_cm
        pixel_um = 10000 * denominator / numerator
        stage_x, stage_y = (position * 10000 for position in scan.stage_position_cm)
        assert metadata["pixel_size_um"] == pytest.approx([pixel_um, pixel_um])
        assert metadata["stage_position_um"] == pytest.approx([stage_x, stage_y])
        transform = get_transformation(image, to_coordinate_system="global")
        assert isinstance(transform, BaseTransformation)
        np.testing.assert_allclose(
            transform.to_affine_matrix(("x", "y"), ("x", "y")),
            [[pixel_um, 0, stage_x], [0, pixel_um, stage_y], [0, 0, 1]],
        )

        # Sample an 8x8 window at an interior tile boundary of every measured
        # band. Reading a full RGB page here would materialize about 2.5 GB.
        y, x = scan.window_origin
        expected, expected_channels = _reference_window(source, y, x)
        assert [channel["name"] for channel in metadata["channels"]] == names
        assert [
            (channel["page_index"], channel.get("biomarker"), channel.get("band_name"))
            for channel in metadata["channels"]
        ] == expected_channels
        np.testing.assert_array_equal(image.data[:, y : y + 8, x : x + 8].compute(scheduler="synchronous"), expected)

    @pytest.mark.codex
    def test_cim_public_readers_agree(self, require_test_dataset: Callable[[str], Path]) -> None:
        source = require_test_dataset("phenocycler_cim_tonsil") / "Arlene_Tonsil_9x9.qptiff"
        pheno = phenocycler(source, scale_factors=(), chunks=(1, 512, 512))
        codex_data = codex(source, scale_factors=(), chunks=(1, 512, 512))
        assert codex_data.attrs["spatialdata_io_reader"] == "codex"
        assert codex_data.attrs["spatialdata_io_format_version"] == "3"
        assert codex_data.attrs["akoya"] == pheno.attrs["akoya"]
        element = "Arlene_Tonsil_9x9_image"
        assert list(codex_data.images) == [element]
        pheno_image = pheno.images[element]
        codex_image = codex_data.images[element]
        assert isinstance(pheno_image, DataArray)
        assert isinstance(codex_image, DataArray)
        assert codex_image.dims == pheno_image.dims
        assert codex_image.shape == pheno_image.shape
        assert codex_image.dtype == pheno_image.dtype
        assert get_channel_names(codex_image) == get_channel_names(pheno_image)
        assert get_transformation(codex_image, get_all=True) == get_transformation(pheno_image, get_all=True)
        selection = np.s_[:, 4608:4616, 6144:6152]
        np.testing.assert_array_equal(
            codex_image.data[selection].compute(scheduler="synchronous"),
            pheno_image.data[selection].compute(scheduler="synchronous"),
        )


def _reference_window(
    path: Path, y: int, x: int
) -> tuple[NDArray[np.generic], list[tuple[int, str | None, str | None]]]:
    """Find measured IFDs from QPI XML and independently decode one tile of each."""
    windows = []
    channels = []
    with tifffile.TiffFile(path) as tiff:
        for index, entry in enumerate(tiff.pages):
            page = entry.aspage() if isinstance(entry, tifffile.TiffFrame) else entry
            description = page.description[page.description.index("<PerkinElmer-") :]
            xml = ET.fromstring(description)  # noqa: S314 - starts at the root, after any declarations.
            if xml.findtext("ImageType") != "FullResolution":
                continue
            assert page.is_tiled
            assert page.tilelength == page.tilewidth == 512
            tiles_per_row = (page.imagewidth + page.tilewidth - 1) // page.tilewidth
            tile_index = y // page.tilelength * tiles_per_row + x // page.tilewidth
            tiff.filehandle.seek(page.dataoffsets[tile_index])
            raw = tiff.filehandle.read(page.databytecounts[tile_index])
            tile, _, _ = page.decode(raw, tile_index, jpegtables=page.jpegtables)
            assert tile is not None
            windows.append(tile[0, :8, :8, :])
            channels.extend(
                (index, xml.findtext("Biomarker") or None, xml.findtext("Name") or None)
                for _ in range(page.samplesperpixel)
            )
    return np.concatenate(windows, axis=-1).transpose(2, 0, 1), channels


@pytest.mark.cli
class TestPhenoCyclerDataset:
    def test_akoya_inform_component_image(
        self, require_test_dataset: Callable[[str], Path], tmp_path: Path, runner: CliRunner
    ) -> None:
        root = require_test_dataset("phenocycler_inform")
        source = root / "FIHC4__0929309_HP_IM3_2_component_data.tif"
        output = tmp_path / "components.zarr"
        result = runner.invoke(cli, ["phenocycler", "-i", str(source), "-o", str(output)])
        assert result.exit_code == 0, (result.output, result.exception)

        stored = read_zarr(output)
        assert stored.attrs["spatialdata_io_reader"] == "phenocycler"
        assert len(stored.images) == 1
        image = next(iter(stored.images.values()))
        assert image.dims == ("c", "y", "x")
        assert image.shape == (5, 260, 348)
        assert image.dtype == np.float32
        assert image.coords["c"].values.tolist() == ["Alexa 594", "Cy3", "Cy5", "DAPI", "FITC"]
        with tifffile.TiffFile(source) as tiff:
            for channel in range(5):
                expected = tiff.pages[channel].asarray()
                np.testing.assert_array_equal(image.data[channel].compute(), expected)
        # The fixture's known resolution is 4294967295/213523 pixels/cm on both axes.
        pixel_um = 10000 * 213523 / 4294967295
        transform = get_transformation(image, to_coordinate_system="global")
        assert isinstance(transform, BaseTransformation)
        np.testing.assert_allclose(transform.to_affine_matrix(("x", "y"), ("x", "y")), np.diag([pixel_um, pixel_um, 1]))
