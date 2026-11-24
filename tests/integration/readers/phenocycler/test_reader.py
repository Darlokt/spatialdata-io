"""Synthetic public workflows for every supported QPI family."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import pytest
from cloudpickle import dumps, loads
from spatialdata import read_zarr
from spatialdata.models import get_channel_names
from spatialdata.transformations import Identity, Scale, Sequence, Translation, get_transformation
from xarray import DataArray, DataTree

from spatialdata_io import phenocycler
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.image.tiff import _dask as tiff_dask
from tests.integration.readers.phenocycler._fixtures import (
    CIM_PROFILE,
    FUSION_PROFILE,
    OPAL_PROFILE,
    QpiPage,
    write_pages,
    write_qptiff,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestPhenocycler:
    @pytest.mark.parametrize(
        ("software", "profile", "generation", "dtype", "rgb", "filename"),
        [
            ("Fusion 1.0.5", FUSION_PROFILE, "fusion1", "uint8", False, "Tonsil_Scan1.qptiff"),
            ("Fusion 2.3.1", FUSION_PROFILE, "fusion2", "uint16", False, "Tonsil_Scan1.qptiff"),
            ("Fusion 2.3.1", OPAL_PROFILE, "fusion2", "uint8", False, "Tonsil_Scan1.qptiff"),
            ("CODEX Instrument Manager", CIM_PROFILE, "cim_qptiff", "uint16", False, "Tonsil_Scan1.qptiff"),
            ("VectraPolaris 1.1", None, "qpi_other", "uint8", True, "Tonsil_Scan1.qptiff"),
            ("PhenoImager HT 2.2", OPAL_PROFILE, "qpi_other", "uint16", False, "Tonsil_Scan1.qptiff"),
            ("inForm 2.2.1", None, "qpi_other", "float32", False, "Tonsil_Scan1.tif"),
        ],
        ids=["fusion1", "fusion2", "fusion-opal", "cim", "vectra-rgb", "phenoimager-fl", "inform-components"],
    )
    def test_full_resolution_workflow(  # noqa: PLR0913 - each parameter names a source format fact.
        self,
        tmp_path: Path,
        *,
        software: str,
        profile: str | None,
        generation: str,
        dtype: str,
        rgb: bool,
        filename: str,
    ) -> None:
        source = tmp_path / filename
        expected = write_qptiff(source, software=software, scan_profile=profile, dtype=dtype, rgb=rgb)
        result = phenocycler(source, chunks=(1, 16, 16))
        assert list(result.images) == ["Tonsil_image"]
        assert not result.labels
        assert not result.shapes
        assert not result.tables
        assert not result.points
        image = result.images["Tonsil_image"]
        assert isinstance(image, DataArray)
        assert image.dims == ("c", "y", "x")
        assert image.dtype == expected.dtype
        assert get_channel_names(image) == (["r", "g", "b"] if rgb else ["DAPI", "CD3"])
        np.testing.assert_array_equal(image.data.compute(), expected)
        assert get_transformation(image, get_all=True) == {"global": Scale([0.5, 0.5], axes=("x", "y"))}
        scan = result.attrs["akoya"]["regions"]["Tonsil"]
        assert scan["generation"] == generation
        assert scan["description_version"] == result.attrs["spatialdata_io_format_version"] == "5"
        assert scan["pixel_size_um"] == [0.5, 0.5]
        assert [channel["page_index"] for channel in scan["channels"]] == ([0, 0, 0] if rgb else [0, 3])
        assert result.attrs["spatialdata_io_reader"] == "phenocycler"

    @pytest.mark.parametrize(("compressed", "tiled"), [("deflate", True), (None, False)])
    def test_lazy_selection_and_graph_serialization(
        self, tmp_path: Path, mocker: MockerFixture, *, compressed: str | None, tiled: bool
    ) -> None:
        path = tmp_path / "tiled.qptiff"
        expected = write_qptiff(path, shape=(64, 96), compression=compressed, tiled=tiled)
        mocker.patch.object(tiff_dask, "STRIP_BAND_BYTES", 16 * 96 * expected.dtype.itemsize)
        reads = mocker.spy(tiff_dask, "_read_tiff_selection")
        image = phenocycler(path, chunks=(1, 16, 16)).images["tiled_image"]
        assert isinstance(image, DataArray)
        assert isinstance(image.data, da.Array)
        assert image.data.chunks == ((1, 1), (16,) * 4, (16,) * 6)
        assert reads.call_count == 0
        graph = loads(dumps(image.data))
        np.testing.assert_array_equal(graph[1:, :2, :3].compute(), expected[1:, :2, :3])
        assert reads.call_count == 1

    def test_hubmap_sidecars_and_round_trip(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw" / "images"
        (raw / "phenocycler").mkdir(parents=True)
        (raw / "phenocycler" / "intermediate.qptiff").touch()
        (raw / "throwaway.raw.qptiff").touch()
        (tmp_path / ".hidden").mkdir()
        (tmp_path / ".hidden" / "hidden.qptiff").touch()
        (tmp_path / "external.zarr").mkdir()
        (tmp_path / "external.zarr" / "stray.qptiff").touch()
        processed = tmp_path / "lab_processed" / "images"
        processed.mkdir(parents=True)
        (processed / "Slide_Scan1.qptiff.CHANNELS.CSV").write_text("Marker,Note\nDAPI,foo\nCD3,/private/token\n")
        (raw / "Slide_Scan1.xpd").write_text(
            json.dumps(
                {
                    "formatVersion": 2,
                    "wells": [{"items": [{"markerName": "DAPI"}, {"markerName": "CD3"}]}],
                    "token": "/private/key",
                }
            ),
            encoding="utf-8-sig",
        )
        expected = write_qptiff(raw / "Slide_Scan1.qptiff", scan_profile=FUSION_PROFILE)
        result = phenocycler(tmp_path, scale_factors=(2,), channel_names=("Nuclei", "T cells"))
        image = result.images["Slide_image"]
        assert isinstance(image, DataTree)
        assert list(image.children) == ["scale0", "scale1"]
        assert list(image["scale0"].data_vars) == ["image"]
        assert image["scale1"]["image"].shape == (2, 16, 24)
        attrs = result.attrs["akoya"]["regions"]["Slide"]
        assert attrs["channels_csv"] == {"marker_count": 2, "matching_marker_count": 2}
        assert attrs["xpd"] == {"marker_count": 2, "matching_marker_count": 2, "cycle_count": 1, "format_version": 2}
        assert attrs["channels"][0]["id"] == "DAPI"
        assert attrs["channels"][0]["name"] == "Nuclei"
        assert attrs["channels"][0]["profile_cycle"] == 1
        assert "/private/" not in json.dumps(result.attrs)
        assert "private-host" not in json.dumps(result.attrs)
        assert str(tmp_path) not in json.dumps(result.attrs)

        dest = tmp_path / "image.zarr"
        result.write(dest)
        reread = read_zarr(dest)
        assert reread.attrs == result.attrs
        tree = reread.images["Slide_image"]
        assert isinstance(tree, DataTree)
        np.testing.assert_array_equal(tree["scale0"]["image"].data.compute(), expected)

        explicit = phenocycler(raw / "Slide_Scan1.qptiff")
        assert "channels_csv" not in explicit.attrs["akoya"]["regions"]["Slide"]

    def test_extended_range_and_physical_region_placement(self, tmp_path: Path) -> None:
        first = tmp_path / "A_Scan1.qptiff"
        second = tmp_path / "B_Scan1.qptiff"
        extended = tmp_path / "A_Scan1.er.qptiff"
        normal = write_qptiff(
            first, dtype="uint8", stage_position=(1.0, 2.0), scan_id="00000000-0000-0000-0000-000000000001"
        )
        high = write_qptiff(
            extended, dtype="uint16", stage_position=(1.0, 2.0), scan_id="00000000-0000-0000-0000-000000000002"
        )
        write_qptiff(second, stage_position=(1.02, 2.01), resolution=(40000, 40000))
        result = phenocycler(tmp_path, regions=["B", "A"], read_all_bit_depths=True)
        assert list(result.images) == ["B_image", "A_image", "A_8bit_image"]
        ordinary = result.images["A_8bit_image"]
        preferred = result.images["A_image"]
        assert isinstance(ordinary, DataArray)
        assert isinstance(preferred, DataArray)
        np.testing.assert_array_equal(ordinary.data.compute(), normal)
        np.testing.assert_array_equal(preferred.data.compute(), high)
        first_t = get_transformation(preferred, get_all=True)
        assert first_t == get_transformation(ordinary, get_all=True)
        transform = get_transformation(result.images["B_image"], "global")
        assert isinstance(transform, Sequence)
        scale, offset = transform.transformations
        assert scale == Scale([0.25, 0.25], axes=("x", "y"))
        assert isinstance(offset, Translation)
        np.testing.assert_allclose(offset.translation, [10_200, 20_100])
        subset = phenocycler(tmp_path, regions=["B"])
        assert get_transformation(subset.images["B_image"], "global") == transform
        assert list(phenocycler(extended).images) == ["A_image"]
        assert list(phenocycler(first, read_all_bit_depths=True).images) == ["A_image", "A_8bit_image"]
        assert phenocycler(tmp_path, regions=["A"]).images["A_image"].dtype == np.dtype("uint16")

    def test_unrelated_and_uncalibrated_frames(self, tmp_path: Path) -> None:
        write_qptiff(tmp_path / "A.qptiff", slide_id="A")
        write_qptiff(
            tmp_path / "B.qptiff", slide_id="B", resolution_unit="INCH", resolution=(96.0, 96.0), stage_position=(1, 2)
        )
        data = phenocycler(tmp_path)
        assert get_transformation(data.images["A_image"], get_all=True) == {"A_um": Scale([0.5, 0.5], axes=("x", "y"))}
        assert get_transformation(data.images["B_image"], get_all=True) == {"B_pixels": Identity()}
        single = phenocycler(tmp_path, regions=["B"])
        assert get_transformation(single.images["B_image"], get_all=True) == {"image_pixels": Identity()}

    def test_nominal_calibration_and_quantitative_fields(self, tmp_path: Path) -> None:
        values = np.full((16, 16), -0.4, dtype=np.float32)
        source = tmp_path / "component.tif"
        write_pages(
            source,
            [
                QpiPage(
                    values,
                    name="Alexa 594",
                    scan_profile=OPAL_PROFILE,
                    resolution=(96.0, 96.0),
                    resolution_unit="INCH",
                    minimum=-0.4,
                    maximum=10.1,
                    extra_xml=(
                        "<ExposureTime>2000</ExposureTime><ExposureTimeArray>1, 2</ExposureTimeArray>"
                        "<SignalUnits>33</SignalUnits><ScaleFactor>4.2</ScaleFactor>"
                        "<NormalizedUnitsFactor>0.75</NormalizedUnitsFactor><ImagingCycle>0</ImagingCycle>"
                        "<IsUnmixedComponent>True</IsUnmixedComponent><Color>255, 128, 0</Color>"
                        "<AutofluorescenceSubtracted>false</AutofluorescenceSubtracted>"
                        "<CameraSettings><Gain>2.5</Gain><Binning>2</Binning><BitDepth>14</BitDepth>"
                        "<OffsetCounts>0</OffsetCounts></CameraSettings>"
                    ),
                ),
            ],
            software="inForm",
        )
        result = phenocycler(source)
        im = result.images["component_image"]
        assert isinstance(im, DataArray)
        np.testing.assert_array_equal(im.data.compute(), values[None])
        scan = result.attrs["akoya"]["regions"]["component"]
        assert scan["pixel_size_um"] is None
        assert scan["nominal_pixel_size_um"] == 0.5
        channel = scan["channels"][0]
        assert channel["sample_min"] == pytest.approx(-0.4)
        assert channel["sample_max"] == pytest.approx(10.1)
        assert channel["exposure_us"] == 2000
        assert channel["exposure_times_us"] == [1, 2]
        assert channel["signal_units_code"] == 33
        assert channel["scale_factor"] == 4.2
        assert channel["normalized_units_factor"] == 0.75
        assert channel["imaging_cycle_zero_based"] == 0
        assert channel["autofluorescence_subtracted"] is False
        assert channel["is_unmixed_component"] is True
        assert channel["color_rgb"] == [255, 128, 0]
        assert channel["camera_gain"] == 2.5
        assert channel["camera_binning"] == 2
        assert channel["camera_bit_depth"] == 14
        assert channel["camera_offset_counts"] == 0

    def test_invalid_companion_is_not_registered(self, tmp_path: Path) -> None:
        write_qptiff(tmp_path / "sample.qptiff", dtype="uint8", stage_position=(1.0, 0.0))
        write_qptiff(tmp_path / "sample.er.qptiff", stage_position=(1.1, 0.0))
        with pytest.raises(ReaderFormatError, match="different channel or spatial grid"):
            phenocycler(tmp_path, read_all_bit_depths=True)
