"""Public MCMICRO workflows: provenance, laziness, bounded reads, and round trips."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import pytest
import tifffile
from click.testing import CliRunner
from dask.callbacks import Callback
from scipy import sparse
from spatialdata import read_zarr
from spatialdata.models import get_table_keys
from spatialdata.testing import assert_spatial_data_objects_are_identical
from spatialdata.transformations import Identity, Scale, get_transformation
from xarray import DataTree

from spatialdata_io import __version__
from spatialdata_io._cli.commands.mcmicro import mcmicro_command
from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.mcmicro import mcmicro
from tests.integration.readers.mcmicro._fixtures import write_mcmicro_dataset

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from anndata import AnnData
    from pytest_mock import MockerFixture


def _write_historical_layout(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Write the smallest historical sample-directory run with independent values."""
    (path / "qc").mkdir()
    (path / "qc" / "params.yml").write_text("workflow:\n  tma: false\n", encoding="utf-8")
    pd.DataFrame({"marker_name": ["CD3", "DAPI"], "channel_number": [1, 2]}).to_csv(path / "markers.csv", index=False)

    for directory in ("registration", "segmentation/sample", "quantification"):
        (path / directory).mkdir(parents=True)
    image = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    labels = np.asarray([[0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 2, 2], [0, 0, 0, 2, 2]], dtype=np.uint16)
    tifffile.imwrite(path / "registration" / "sample.ome.tif", image, metadata={"axes": "CYX"})
    tifffile.imwrite(path / "segmentation" / "sample" / "cells.ome.tif", labels, metadata={"axes": "YX"})

    pd.DataFrame(
        {
            "CellID": [1, 2],
            "CD3": [10.0, 0.0],
            "DAPI": [30.0, 40.0],
            "X_centroid": [1.5, 3.5],
            "Y_centroid": [0.5, 2.5],
            "Area": [4.0, 4.0],
        }
    ).to_csv(path / "quantification" / "sample--seg_cells.csv", index=False)
    return image, labels


def _expression(table: AnnData) -> sparse.csr_array:
    """Narrow AnnData's broad expression union to the reader's sparse contract."""
    assert isinstance(table.X, sparse.csr_array)
    return table.X


def _calibrate(path: Path, spacing: tuple[float, float]) -> None:
    """Attach independently declared XY spacing without changing fixture pixels."""
    with tifffile.TiffFile(path) as source:
        values = source.asarray()
        axes = source.series[0].axes
    tifffile.imwrite(
        path,
        values,
        ome=True,
        photometric="minisblack",
        metadata={"axes": axes, "PhysicalSizeX": spacing[0], "PhysicalSizeY": spacing[1]},
    )


class TestMcmicro:
    """Assemble supported layouts through the public reader."""

    @pytest.mark.parametrize("mode", ["wsi", "tma"])
    def test_declared_base_scale_and_independent_masks(self, tmp_path: Path, mode: Literal["wsi", "tma"]) -> None:
        fixture = write_mcmicro_dataset(tmp_path, mode=mode)
        _calibrate(tmp_path / "registration" / "sample_part_7.ome.tif", (0.3, 0.7))
        if mode == "tma":
            _calibrate(tmp_path / "dearray" / "1.ome.tif", (0.4, 0.8))
        tasks: list[object] = []
        with Callback(posttask=lambda *args: tasks.append(args)):
            result = mcmicro(fixture.root)
        assert not tasks
        if mode == "wsi":
            selected = [
                (result.images["sample_part_7_image"], "global", [0.3, 0.7]),
                (result.labels["sample_part_7_method_a_cells"], "global", [0.3, 0.7]),
            ]
        else:
            selected = [
                (result.images["tma_map"], "global", [0.3, 0.7]),
                (result.images["core_1_image"], "core_1", [0.4, 0.8]),
                (result.labels["core_1_method_a_cells"], "core_1", [0.4, 0.8]),
            ]
            # Uncalibrated grids keep their pixel frames; the micrometre names are never reused for pixels.
            assert get_transformation(result.images["core_2_image"], get_all=True) == {"core_2_pixels": Identity()}
            assert get_transformation(result.labels["core_1_mask"], get_all=True) == {"core_1_mask_pixels": Identity()}
            assert sorted(result.coordinate_systems) == [
                "core_1",
                "core_1_mask_pixels",
                "core_2_mask_pixels",
                "core_2_pixels",
                "global",
            ]
        for element, frame, expected in selected:
            transform = get_transformation(element, to_coordinate_system=frame)
            assert isinstance(transform, Scale)
            np.testing.assert_array_equal(transform.scale, expected)
        np.testing.assert_array_equal(
            result.tables["table"].obsm["spatial_pixels"][:2],
            [[3.5, 3], [35.5, 26]] if mode == "tma" else [[3.5, 3], [75.5, 60]],
        )

    @pytest.mark.parametrize(
        ("mode", "calibrated", "systems", "spatial_frames"),
        [
            ("wsi", False, ["image_pixels"], None),
            ("wsi", True, ["global"], ["global"]),
            (
                "tma",
                False,
                ["core_1_mask_pixels", "core_1_pixels", "core_2_mask_pixels", "core_2_pixels", "tma_map_pixels"],
                None,
            ),
            (
                "tma",
                True,
                ["core_1", "core_1_mask_pixels", "core_2", "core_2_mask_pixels", "global"],
                ["core_1", "core_2"],
            ),
        ],
    )
    def test_frames_follow_calibration_and_table_names_the_same_frames(
        self,
        tmp_path: Path,
        mode: Literal["wsi", "tma"],
        *,
        calibrated: bool,
        systems: list[str],
        spatial_frames: list[str] | None,
    ) -> None:
        write_mcmicro_dataset(tmp_path, mode=mode)
        if calibrated:
            images = ["registration/sample_part_7.ome.tif"]
            if mode == "tma":
                images += ["dearray/1.ome.tif", "dearray/2.ome.tif"]
            for image in images:
                _calibrate(tmp_path / image, (0.5, 0.25))
        result = mcmicro(tmp_path)
        assert sorted(result.coordinate_systems) == systems

        table = result.tables["table"]
        coordinates = table.uns[POSITIONS_KEY]["coordinates"]
        if mode == "wsi":
            assert coordinates["spatial_pixels"]["frame"] == "image_regionprops_indices"
        else:
            assert table.obs["spatial_pixel_frame"].cat.categories.tolist() == [
                "core_1_regionprops_indices",
                "core_2_regionprops_indices",
            ]
        if spatial_frames is None:
            assert "spatial" not in coordinates
            return
        # Physical positions name the micrometre frames the rasters use.
        frames = [coordinates["spatial"]["frame"]] if mode == "wsi" else table.obs["spatial_frame"].cat.categories
        assert list(frames) == spatial_frames
        assert set(coordinates["spatial"]["calibration"]) == set(spatial_frames)
        for region in table.obs["region"].cat.categories:
            transform = get_transformation(result.labels[str(region)], get_all=True)
            assert isinstance(transform, dict)
            assert set(transform) <= set(spatial_frames)

    def test_file_derived_names_are_sanitized(self, tmp_path: Path) -> None:
        write_mcmicro_dataset(tmp_path / "input")
        root = tmp_path / "input"
        (root / "registration" / "sample_part_7.ome.tif").rename(root / "registration" / "my slide.ome.tif")
        (root / "segmentation" / "method_a-sample_part_7").rename(root / "segmentation" / "method_a-my slide")
        (root / "quantification" / "sample_part_7--method_a_cells.csv").rename(
            root / "quantification" / "my slide--method_a_cells.csv"
        )
        with pytest.warns(UserWarning, match=r"registered image ID 'my slide' .* using 'my_slide'"):
            result = mcmicro(root)
        assert list(result.images) == ["my_slide_image"]
        assert list(result.labels) == ["my_slide_method_a_cells"]
        assert get_table_keys(result.tables["table"]) == ("my_slide_method_a_cells", "region", "CellID")
        result.write(tmp_path / "output.zarr")
        assert list(read_zarr(tmp_path / "output.zarr").labels) == ["my_slide_method_a_cells"]

    def test_header_column_names_are_sanitized(self, tmp_path: Path) -> None:
        fixture = write_mcmicro_dataset(tmp_path / "input")
        table_path = fixture.root / "quantification" / "sample_part_7--method_a_cells.csv"
        source = pd.read_csv(table_path, dtype=str)
        source["CD45 RO_intensity_max"] = ["5", "6"]
        source["Area"] = ["12", "13"]
        source.to_csv(table_path, index=False)
        markers = pd.read_csv(fixture.root / "markers.csv", dtype=str)
        markers["cy cle note"] = ["a", "b"]
        markers.to_csv(fixture.root / "markers.csv", index=False)

        with pytest.warns(UserWarning, match="column names") as caught:
            result = mcmicro(fixture.root)

        assert [str(warning.message) for warning in caught] == [
            (
                "mcmicro: the quantification CSV header has column names that a SpatialData table does not accept; "
                "using 'area' -> 'area_1', 'CD45 RO_intensity_max' -> 'CD45_RO_intensity_max'. Rename the source "
                "columns to choose the names explicitly."
            ),
            (
                "mcmicro: the markers.csv header has column names that a SpatialData table does not accept; using "
                "'cy cle note' -> 'cy_cle_note'. Rename the source columns to choose the names explicitly."
            ),
        ]
        table = result.tables["table"]
        assert table.obs.columns.tolist() == [
            "region",
            "CellID",
            "Area",
            "area_1",
            "custom_note",
            "CD3_intensity_max",
            "CD45_RO_intensity_max",
        ]
        assert table.obs["Area"].tolist() == [12.0, 13.0]
        assert table.obs["area_1"].tolist() == ["12.0", "12.0"]
        assert table.obs["CD45_RO_intensity_max"].tolist() == ["5", "6"]
        assert table.var.columns.tolist() == ["marker_name", "cycle", "channel_number", "cy_cle_note"]
        assert table.var["cy_cle_note"].tolist() == ["a", "b"]
        result.write(tmp_path / "output.zarr")
        restored = read_zarr(tmp_path / "output.zarr").tables["table"]
        assert restored.obs.columns.tolist() == table.obs.columns.tolist()
        assert restored.var.columns.tolist() == table.var.columns.tolist()

    def test_case_colliding_labels_fail_before_rasters(self, tmp_path: Path, mocker: MockerFixture) -> None:
        write_mcmicro_dataset(tmp_path)
        directory = tmp_path / "segmentation" / "method_a-sample_part_7"
        (directory / "Cells.ome.tif").write_bytes((directory / "cells.ome.tif").read_bytes())
        inspect = mocker.spy(tifffile.TiffFile, "__init__")
        with pytest.raises(ReaderFormatError, match="colliding keys") as caught:
            mcmicro(tmp_path)
        assert "sample_part_7_method_a_Cells" in caught.value.found
        inspect.assert_not_called()

    def test_conflicting_same_grid_labels_fail(self, tmp_path: Path) -> None:
        write_mcmicro_dataset(tmp_path)
        _calibrate(tmp_path / "registration" / "sample_part_7.ome.tif", (0.3, 0.7))
        _calibrate(tmp_path / "segmentation" / "method_a-sample_part_7" / "cells.ome.tif", (0.4, 0.7))
        with pytest.raises(ReaderFormatError, match="calibration"):
            mcmicro(tmp_path)

    def test_historical_layout_public_contract(self, tmp_path: Path) -> None:
        image, labels = _write_historical_layout(tmp_path)
        result = mcmicro(tmp_path)

        assert list(result.images) == ["sample_image"]
        assert list(result.labels) == ["sample_seg_cells"]
        assert list(result.tables) == ["table"]
        assert not result.points
        assert not result.shapes
        assert result.attrs == {"spatialdata_io_reader": "mcmicro", "spatialdata_io_software_version": __version__}
        np.testing.assert_array_equal(result.images["sample_image"].data.compute(), image)
        np.testing.assert_array_equal(result.labels["sample_seg_cells"].data.compute(), labels)

        table = result.tables["table"]
        assert _expression(table).nnz == 3
        np.testing.assert_array_equal(_expression(table).toarray(), [[10.0, 30.0], [0.0, 40.0]])
        assert table.var["channel_number"].tolist() == ["1", "2"]
        assert table.obs["Area"].tolist() == [4.0, 4.0]
        assert get_table_keys(table) == ("sample_seg_cells", "region", "CellID")

    def test_run_without_quantification_returns_no_table(self, tmp_path: Path) -> None:
        _write_historical_layout(tmp_path)
        (tmp_path / "quantification" / "sample--seg_cells.csv").unlink()
        (tmp_path / "segmentation" / "sample" / "cells.ome.tif").unlink()
        result = mcmicro(tmp_path)
        assert list(result.images) == ["sample_image"]
        assert not result.labels
        assert not result.tables

    def test_cell_id_cannot_be_used_as_expression(self, tmp_path: Path) -> None:
        _write_historical_layout(tmp_path)
        (tmp_path / "markers.csv").write_text("marker_name\nCellID\nDAPI\n", encoding="utf-8")
        source = tmp_path / "quantification" / "sample--seg_cells.csv"
        source.write_text("CellID,DAPI,X_centroid,Y_centroid\n1,3,1.5,0.5\n", encoding="utf-8")
        with pytest.raises(ReaderFormatError) as caught:
            mcmicro(tmp_path)
        assert caught.value.context.reader == "mcmicro"
        assert caught.value.path == source
        assert "CellID" in str(caught.value)

    def test_image_only_panel_can_name_cell_id(self, tmp_path: Path) -> None:
        _write_historical_layout(tmp_path)
        (tmp_path / "markers.csv").write_text("marker_name\nCellID\nDAPI\n", encoding="utf-8")
        (tmp_path / "quantification" / "sample--seg_cells.csv").unlink()
        (tmp_path / "segmentation" / "sample" / "cells.ome.tif").unlink()
        result = mcmicro(tmp_path)
        assert result.images["sample_image"].coords["c"].to_numpy().tolist() == ["CellID", "DAPI"]
        assert not result.tables

    def test_duplicate_ids_fail_with_final_table_context(self, tmp_path: Path) -> None:
        _write_historical_layout(tmp_path)
        source = tmp_path / "quantification" / "sample--seg_cells.csv"
        source.write_text("CellID,CD3,DAPI,X_centroid,Y_centroid\n1,3,4,1.5,0.5\n1,5,6,2,3\n", encoding="utf-8")
        with pytest.raises(ReaderFormatError, match=r"duplicate \(region, instance\) pairs") as caught:
            mcmicro(tmp_path)
        assert caught.value.context.reader == "mcmicro"
        assert caught.value.context.artifact == "cell table"
        assert "sample_seg_cells" in caught.value.found
        assert "1" in caught.value.found

    @pytest.mark.parametrize(
        ("options", "error", "match"),
        [
            ({"scale_factors": "2", "chunks": 0, "path_policy": 1}, TypeError, "scale_factors"),
            ({"scale_factors": (1,), "chunks": 0, "path_policy": 1}, ValueError, "scale_factors"),
            ({"chunks": "a", "path_policy": 1}, TypeError, "chunks"),
            ({"chunks": (1, 2), "path_policy": 1}, ValueError, "chunks"),
            ({"path_policy": 1}, TypeError, "path_policy"),
            ({"path_policy": "lenient"}, ValueError, "path_policy"),
        ],
    )
    def test_options_fail_in_signature_order_before_io(
        self, tmp_path: Path, options: dict[str, object], error: type[Exception], match: str
    ) -> None:
        with pytest.raises(error, match=match):
            mcmicro(tmp_path / "missing", **options)  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.

    def test_construction_reads_no_pixels_and_partial_reads_open_one_source(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        fixture = write_mcmicro_dataset(tmp_path, mode="tma")
        decoder = mocker.spy(tifffile.TiffPage, "asarray")
        tasks: list[object] = []
        with Callback(posttask=lambda *args: tasks.append(args)):
            result = mcmicro(fixture.root)
        assert not tasks
        decoder.assert_not_called()

        # Core 2 is deflate-compressed with native five-row strip chunks; one pixel opens one source once.
        reads = mocker.spy(tifffile.TiffFile, "__init__")
        value = result.images["core_2_image"].data[1, 0, 0].compute(scheduler="single-threaded")
        assert value == 2651
        assert reads.call_count == 1

    def test_graphs_serialize_and_pyramids_stay_lazy(self, tmp_path: Path) -> None:
        fixture = write_mcmicro_dataset(tmp_path, mode="tma")
        result = mcmicro(fixture.root, scale_factors=(2,), chunks=(1, 8, 8))
        image = result.images["core_1_image"]
        assert isinstance(image, DataTree)
        data = image["scale0"]["image"].data
        assert data.chunks[1:] == ((8, 8, 8, 6), (8, 8, 8, 8, 8))
        restored = pickle.loads(pickle.dumps(data))  # noqa: S301, RUF100 - Only deserialize this test's own graph.
        np.testing.assert_array_equal(restored.compute(), data.compute())

    def test_zarr_round_trip_preserves_values_and_linkage(self, tmp_path: Path) -> None:
        fixture = write_mcmicro_dataset(tmp_path / "input", mode="tma")
        result = mcmicro(fixture.root)
        result.write(tmp_path / "output.zarr")
        restored = read_zarr(tmp_path / "output.zarr")
        table = restored.tables["table"]
        # Storage round trips return AnnData's own sparse type; compare exact values only.
        stored = table.X
        assert isinstance(stored, sparse.csr_array | sparse.csr_matrix)
        np.testing.assert_array_equal(stored.toarray(), _expression(result.tables["table"]).toarray())
        assert table.obs["CellID"].tolist() == [1, 2, 1, 2]
        assert get_table_keys(table) == (["core_1_method_a_cells", "core_2_method_a_cells"], "region", "CellID")
        np.testing.assert_array_equal(
            restored.labels["core_2_method_a_cells"].data.compute(),
            result.labels["core_2_method_a_cells"].data.compute(),
        )

    def test_cli_converts_a_run(self, tmp_path: Path) -> None:
        fixture = write_mcmicro_dataset(tmp_path / "input")
        output = tmp_path / "output.zarr"
        result = CliRunner().invoke(
            mcmicro_command, ["--input", str(fixture.root), "--output", str(output), "--chunks", "16"]
        )
        assert result.exit_code == 0, result.output
        restored = read_zarr(output)
        assert set(restored.labels) == {"sample_part_7_method_a_cells"}
        assert restored.tables["table"].shape == (2, 2)


class TestMcmicroPathPolicy:
    """Discovered artifacts follow the dataset root's symlink policy."""

    def test_out_of_root_symlinks_require_the_relaxed_policy(
        self, tmp_path: Path, link_outside_root: Callable[[Path], Path]
    ) -> None:
        fixture = write_mcmicro_dataset(tmp_path)
        expected = mcmicro(fixture.root)
        link_outside_root(fixture.root / "markers.csv")
        link_outside_root(fixture.root / "registration" / "sample_part_7.ome.tif")

        with pytest.raises(ReaderFormatError, match="path_policy='relaxed'"):
            mcmicro(fixture.root)

        assert_spatial_data_objects_are_identical(mcmicro(fixture.root, path_policy="relaxed"), expected)
