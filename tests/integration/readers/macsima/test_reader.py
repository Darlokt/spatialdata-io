"""Public MACSima workflows for v0 and v1 exports in both folder layouts."""

from __future__ import annotations

import shutil
from dataclasses import replace
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import tifffile
from dask.callbacks import Callback
from spatialdata import read_zarr
from spatialdata.models import get_channel_names
from spatialdata.transformations import Scale, get_transformation
from xarray import DataTree

from spatialdata_io._cli.commands.macsima import macsima_command
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.macsima import macsima
from tests.integration.readers.macsima._fixtures import (
    V0_CHANNELS,
    V1_CHANNELS,
    ChannelSpec,
    SchemaVersion,
    full_resolution,
    write_channel,
    write_stack,
)

if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import CliRunner
    from spatialdata import SpatialData
    from xarray import DataArray


def _image(result: SpatialData, name: str) -> DataArray:
    return full_resolution(result.images[name])


def _frames(result: SpatialData, name: str) -> dict[str, object]:
    frames = get_transformation(result.images[name], get_all=True)
    assert isinstance(frames, dict)
    return dict(frames)


class TestMacsima:
    """Discover, inspect, plan and assemble complete MACSima exports."""

    @pytest.mark.parametrize(
        ("channels", "version", "software"), [(V0_CHANNELS, 0, "0.15.3"), (V1_CHANNELS, 1, "1.1.1")]
    )
    def test_single_folder_per_schema_version(
        self, tmp_path: Path, channels: tuple[ChannelSpec, ...], version: SchemaVersion, software: str
    ) -> None:
        pixels = write_stack(tmp_path / "run", channels, version=version)
        executed: list[object] = []

        with Callback(posttask=lambda *args: executed.append(args)):
            result = macsima(tmp_path / "run")

        assert not executed
        assert list(result.images) == ["run_image"]
        assert list(result.tables) == ["run_table"]
        assert result.coordinate_systems == ["global"]
        image = _image(result, "run_image")
        assert image.dims == ("c", "y", "x")
        assert image.data.chunks == ((1,) * len(channels), (12,), (15,))
        # Both fixtures list their channels in cycle-then-name order.
        np.testing.assert_array_equal(image.data.compute(), np.stack(pixels))
        assert result.attrs["spatialdata_io_reader"] == "macsima"
        assert result.attrs["spatialdata_io_format_version"] == software

    def test_multiple_folders_keep_provenance_and_separate_frames(self, tmp_path: Path) -> None:
        write_stack(tmp_path / "root" / "first run", V0_CHANNELS, version=0)
        write_stack(tmp_path / "root" / "second", V1_CHANNELS, version=1)

        result = macsima(tmp_path / "root", parsing_style="processed_multiple_folders")

        assert list(result.images) == ["first_run_image", "second_image"]
        assert list(result.tables) == ["first_run_table", "second_table"]
        assert sorted(result.coordinate_systems) == ["first_run", "second"]
        assert result.attrs["spatialdata_io_reader"] == "macsima"
        # Mixed software versions have no single format version.
        assert "spatialdata_io_format_version" not in result.attrs

    def test_excluded_folder_is_omitted(self, tmp_path: Path) -> None:
        write_stack(tmp_path / "keep", V1_CHANNELS[:1], version=1)
        write_stack(tmp_path / "keep_old", V1_CHANNELS[:1], version=1)

        result = macsima(tmp_path, parsing_style="processed_multiple_folders", exclude_folder_names=["keep_old"])

        assert list(result.images) == ["keep_image"]
        assert result.coordinate_systems == ["global"]

    def test_nuclei_stack_shares_channel_names(self, tmp_path: Path) -> None:
        channels = (
            ChannelSpec("CD3", 1, "APC"),
            *(ChannelSpec("DAPI", cycle, "DAPI") for cycle in (1, 3, 5)),
            ChannelSpec("DAPI", 15, "DAPI", scan="B"),
        )
        write_stack(tmp_path / "run", channels, version=1)

        result = macsima(tmp_path / "run")

        assert get_channel_names(result.images["run_image"]) == ["CD3", "DAPI (1)", "DAPI (3)", "DAPI_background"]
        assert get_channel_names(result.images["run_nuclei_image"]) == ["DAPI (1)", "DAPI (2)", "DAPI (3)"]
        assert result.tables["run_nuclei_table"].var["cycle"].tolist() == [1, 3, 5]
        # The name column keeps the marker; var_names hold the numbered channel names.
        assert result.tables["run_table"].var["name"].tolist() == ["CD3", "DAPI", "DAPI", "DAPI_background"]
        assert sorted(result.coordinate_systems) == ["global"]

    def test_equal_shapes_are_aligned_by_pixel_positions(self, tmp_path: Path) -> None:
        channels = (
            ChannelSpec("CD3", 1, "APC", shape=(4, 5), offset=(0, 0)),
            ChannelSpec("CD4", 2, "PE", shape=(4, 5), offset=(1, 2)),
        )
        pixels = write_stack(tmp_path / "run", channels, version=1)

        with pytest.warns(UserWarning, match=r"common canvas \(5, 7\)"):
            result = macsima(tmp_path / "run")

        expected = np.zeros((2, 5, 7), dtype=np.uint16)
        expected[0, :4, :5] = pixels[0]
        expected[1, 1:, 2:] = pixels[1]
        np.testing.assert_array_equal(_image(result, "run_image").data.compute(), expected)

    def test_single_stack_is_global_micrometers(self, tmp_path: Path) -> None:
        write_stack(tmp_path / "run", V1_CHANNELS[:2], version=1, physical_size=("170", "nm"))
        result = macsima(tmp_path / "run")
        assert list(_frames(result, "run_image")) == ["global"]
        scale = _frames(result, "run_image")["global"]
        assert isinstance(scale, Scale)
        np.testing.assert_array_equal(scale.scale, [0.17, 0.17])

    def test_pyramid_and_chunks(self, tmp_path: Path) -> None:
        write_stack(tmp_path / "run", (ChannelSpec("CD3", 1, "APC", shape=(16, 20)),), version=1)

        result = macsima(tmp_path / "run", scale_factors=(2,), chunks=(1, 8, 8))

        element = result.images["run_image"]
        assert isinstance(element, DataTree)
        assert list(element.children) == ["scale0", "scale1"]
        assert _image(result, "run_image").data.chunks == ((1,), (8, 8), (8, 8, 4))

    def test_renamed_release_files_are_read_by_their_own_uuid(self, tmp_path: Path) -> None:
        pixels = write_channel(tmp_path / "run", V1_CHANNELS[0], version=1, index=0, stored_name="OMAP_CD3.ome.tif")

        result = macsima(tmp_path / "run")

        np.testing.assert_array_equal(_image(result, "run_image").data.compute(), pixels[np.newaxis])

    def test_skips_tiffs_without_ome_metadata(self, tmp_path: Path) -> None:
        write_stack(tmp_path / "run", V1_CHANNELS[:1], version=1)
        tifffile.imwrite(tmp_path / "run" / "preview.tif", np.zeros((3, 3), dtype=np.uint16), metadata=None)

        with pytest.warns(UserWarning, match=r"without OME metadata in stack 'run': \['preview.tif'\]"):
            result = macsima(tmp_path / "run")

        assert get_channel_names(result.images["run_image"]) == ["CD3"]

    @pytest.mark.parametrize("storage", ["h5ad", "zarr"])
    def test_writes_and_restores_the_table(self, tmp_path: Path, storage: str) -> None:
        write_stack(tmp_path / "run", V1_CHANNELS, version=1)
        result = macsima(tmp_path / "run")

        table = result.tables["run_table"]
        if storage == "h5ad":
            table.write_h5ad(tmp_path / "out.h5ad", convert_strings_to_categoricals=False)
            restored = ad.read_h5ad(tmp_path / "out.h5ad")
        else:
            result.write(tmp_path / "out.zarr")
            restored = read_zarr(tmp_path / "out.zarr").tables["run_table"]

        for actual in (table, restored):
            assert actual.shape == (0, len(V1_CHANNELS))
            assert actual.X is None
            assert not actual.obsm
            assert not actual.layers
            assert actual.raw is None
            assert not actual.uns
        original_var, restored_var = table.var, restored.var
        assert isinstance(original_var, pd.DataFrame)
        assert isinstance(restored_var, pd.DataFrame)
        pd.testing.assert_frame_equal(restored_var, original_var, check_index_type=False)
        if storage == "zarr":
            np.testing.assert_array_equal(
                _image(read_zarr(tmp_path / "out.zarr"), "run_image").data.compute(),
                _image(result, "run_image").data.compute(),
            )

    def test_cli_writes_a_store(self, tmp_path: Path, runner: CliRunner) -> None:
        write_stack(tmp_path / "run", V0_CHANNELS, version=0)

        outcome = runner.invoke(
            macsima_command,
            ["--input", str(tmp_path / "run"), "--output", str(tmp_path / "out.zarr"), "--scale-factors", "2"],
        )

        assert outcome.exit_code == 0, outcome.output
        assert list(read_zarr(tmp_path / "out.zarr").images) == ["run_image"]

    def test_pixels_require_source_files_after_construction(self, tmp_path: Path) -> None:
        write_stack(tmp_path / "run", V1_CHANNELS[:1], version=1)
        result = macsima(tmp_path / "run")
        shutil.rmtree(tmp_path / "run")

        with pytest.raises(FileNotFoundError):
            _image(result, "run_image").data.compute()


class TestMacsimaRejects:
    """Scientifically inconsistent or ambiguous inputs fail before pixel work."""

    def test_stacks_mixing_wells(self, tmp_path: Path) -> None:
        write_stack(tmp_path / "run" / "a", V1_CHANNELS[:2], version=1)
        other_well = tuple(replace(spec, well="D02") for spec in V1_CHANNELS[2:])
        write_stack(tmp_path / "run" / "b", other_well, version=1, first_index=10)

        with pytest.raises(ReaderFormatError, match="one well per stack"):
            macsima(tmp_path / "run")

    def test_generated_element_names_that_collide(self, tmp_path: Path) -> None:
        nuclei = tuple(ChannelSpec("DAPI", cycle, "DAPI") for cycle in (1, 2, 3))
        write_stack(tmp_path / "a", nuclei, version=1)
        write_stack(tmp_path / "a_nuclei", V1_CHANNELS[:1], version=1)

        with pytest.raises(ReaderFormatError, match="element name 'a_nuclei_image' generated twice"):
            macsima(tmp_path, parsing_style="processed_multiple_folders")

    def test_generated_element_names_that_differ_only_in_case(self, tmp_path: Path) -> None:
        nuclei = tuple(ChannelSpec("DAPI", cycle, "DAPI") for cycle in (1, 2, 3))
        write_stack(tmp_path / "a", nuclei, version=1)
        write_stack(tmp_path / "A_nuclei", V1_CHANNELS[:1], version=1)

        with pytest.raises(ReaderFormatError, match="'a_nuclei_image' generated twice, ignoring case"):
            macsima(tmp_path, parsing_style="processed_multiple_folders")

    def test_folder_without_ome_channels(self, tmp_path: Path) -> None:
        (tmp_path / "run").mkdir()
        tifffile.imwrite(tmp_path / "run" / "preview.tif", np.zeros((3, 3), dtype=np.uint16), metadata=None)

        with (
            pytest.warns(UserWarning, match="without OME metadata"),
            pytest.raises(ReaderFormatError, match="without OME-TIFF channels"),
        ):
            macsima(tmp_path / "run")

    def test_folder_exclusion_in_single_folder_mode(self, tmp_path: Path) -> None:
        write_stack(tmp_path / "run", V1_CHANNELS[:1], version=1)

        with pytest.raises(ValueError, match="requires parsing_style='processed_multiple_folders'"):
            macsima(tmp_path / "run", exclude_folder_names=["x"])

    def test_options_are_validated_before_discovery(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="scale_factors"):
            macsima(tmp_path / "missing", scale_factors=(1,))
        with pytest.raises(TypeError, match="include_transformations"):
            macsima(tmp_path / "missing", include_transformations=False)  # pyrefly: ignore[unexpected-keyword] - Invalid runtime input.

    def test_removed_raw_style(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="parsing_style must be one of"):
            macsima(tmp_path, parsing_style="raw")  # pyrefly: ignore[bad-argument-type] - Removed legacy value.
