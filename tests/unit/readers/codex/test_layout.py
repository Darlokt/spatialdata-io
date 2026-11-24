"""CODEX discovery of the cell table, stitched image and selected experiment file."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.codex._layout import MarkerSet, Montage, discover_layout, validate_region_options
from tests.integration.readers.codex._fixtures import write_csv, write_image

if TYPE_CHECKING:
    from pathlib import Path


class TestDiscoverLayout:
    """Reject missing, ambiguous, wrong-kind, and escaping artifacts."""

    def test_immutable_canonical_layout_and_ignored_backups(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "cells.CSV")
        (tmp_path / "cells.csv.bak").touch()

        layout = discover_layout(tmp_path, cell_table_file=None, image_file=None)

        (region,) = layout.regions
        assert region.cell_table is not None
        assert region.cell_table.path == (tmp_path / "cells.CSV").resolve()
        assert region.cell_table.kind == "csv"
        assert region.image is None
        assert not hasattr(layout, "__dict__")

    def test_missing_table(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="cell table"):
            discover_layout(tmp_path, cell_table_file=None, image_file=None)

    def test_wrong_kind(self, tmp_path: Path) -> None:
        (tmp_path / "cells.csv").mkdir()

        with pytest.raises(IsADirectoryError):
            discover_layout(tmp_path, cell_table_file=None, image_file=None)

    def test_ambiguous_image_and_relative_override(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "cells.csv")
        write_image(tmp_path / "a.tif")
        write_image(tmp_path / "z.tif")

        with pytest.raises(ReaderFormatError, match="2 candidates"):
            discover_layout(tmp_path, cell_table_file=None, image_file=None)

        layout = discover_layout(tmp_path, cell_table_file="cells.csv", image_file="z.tif")

        assert layout.regions[0].image == (tmp_path / "z.tif").resolve()

    @pytest.mark.parametrize("artifact", ["cells.csv", "image.tif"])
    def test_discovered_symlink_must_stay_in_root(self, tmp_path: Path, artifact: str) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        if artifact != "cells.csv":
            write_csv(root / "cells.csv")
        (tmp_path / artifact).touch()
        (root / artifact).symlink_to(tmp_path / artifact)

        with pytest.raises(ReaderFormatError, match="outside"):
            discover_layout(root, cell_table_file=None, image_file=None)

    @pytest.mark.parametrize("artifact", ["table", "image"])
    def test_override_suffix_validation(self, tmp_path: Path, artifact: str) -> None:
        write_csv(tmp_path / "cells.csv")
        (tmp_path / "bad.txt").touch()

        with pytest.raises(ReaderFormatError, match=r"bad\.txt"):
            discover_layout(
                tmp_path,
                cell_table_file="bad.txt" if artifact == "table" else None,
                image_file="bad.txt" if artifact == "image" else None,
            )

    @pytest.mark.parametrize("second", ["reg001_cyc011_ch001_second.tif", "reg001_cyc011_cha001_second.tiff"])
    def test_duplicate_marker_cycle_and_channel(self, tmp_path: Path, second: str) -> None:
        (tmp_path / "reg001_cyc011_cha001_first.tif").touch()
        (tmp_path / second).touch()

        with pytest.raises(ReaderFormatError, match="duplicate cycle 11, channel 1"):
            discover_layout(tmp_path, cell_table_file=None, image_file=None)

    @pytest.mark.parametrize("filename", ["reg001_cyc000_cha001_A.tif", "reg001_cyc001_ch000_A.tif"])
    def test_marker_numbers_must_be_positive(self, tmp_path: Path, filename: str) -> None:
        (tmp_path / filename).touch()

        with pytest.raises(ReaderFormatError, match="positive cycle and channel"):
            discover_layout(tmp_path, cell_table_file=None, image_file=None)

    def test_marker_must_match_parent_region(self, tmp_path: Path) -> None:
        region = tmp_path / "stitched" / "reg001"
        region.mkdir(parents=True)
        (region / "reg002_cyc001_ch001_A.tif").touch()

        with pytest.raises(ReaderFormatError, match="belonging to the parent region 'reg001'"):
            discover_layout(tmp_path, cell_table_file=None, image_file=None)

    def test_marker_sets_from_several_directories_are_ambiguous(self, tmp_path: Path) -> None:
        region = tmp_path / "stitched" / "reg001"
        region.mkdir(parents=True)
        (region / "reg001_cyc001_ch001_A.tif").touch()
        (tmp_path / "reg001_cyc001_ch002_B.tif").touch()

        with pytest.raises(ReaderFormatError, match="one directory of marker planes per region"):
            discover_layout(tmp_path, cell_table_file=None, image_file=None)

    def test_montage_and_markers_for_same_region_require_selection(self, tmp_path: Path) -> None:
        (tmp_path / "reg001_cyc001_ch001_A.tif").touch()
        (tmp_path / "reg001_montage.tif").touch()

        with pytest.raises(ReaderFormatError, match="multiple image layouts"):
            discover_layout(tmp_path, cell_table_file=None, image_file=None)

        layout = discover_layout(tmp_path, cell_table_file=None, image_file="reg001_montage.tif")
        assert isinstance(layout.regions[0].image, Montage)

    def test_no_unrelated_search_and_deterministic_region_order(self, tmp_path: Path) -> None:
        for name in ["reg007", "reg002", "reg009"]:
            (tmp_path / f"{name}_cyc001_ch001_A.tif").touch()
        raw = tmp_path / "cyc001_reg001"
        raw.mkdir()
        (raw / "reg001_cyc002_ch002_raw.tif").touch()

        default = discover_layout(tmp_path, cell_table_file=None, image_file=None)
        explicit = discover_layout(tmp_path, cell_table_file=None, image_file=None, regions=("reg009", "reg002"))

        assert [region.name for region in default.regions] == ["reg002", "reg007", "reg009"]
        assert [region.name for region in explicit.regions] == ["reg009", "reg002"]
        assert all(isinstance(region.image, MarkerSet) for region in explicit.regions)
        with pytest.raises(ReaderFormatError, match="available regions"):
            discover_layout(tmp_path, cell_table_file=None, image_file=None, regions=("reg001",))

    def test_named_override_requires_selected_region_match(self, tmp_path: Path) -> None:
        (tmp_path / "reg002_compensated.csv").touch()
        (tmp_path / "reg001_montage.tif").touch()

        with pytest.raises(ValueError, match="exactly one selected"):
            discover_layout(tmp_path, cell_table_file=None, image_file="reg001_montage.tif")
        with pytest.raises(ReaderFormatError, match="an image for selected region 'reg002'"):
            discover_layout(tmp_path, cell_table_file=None, image_file="reg001_montage.tif", regions=("reg002",))

    def test_mask_is_neither_a_default_nor_explicit_stitched_image(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "cells.csv")
        (tmp_path / "reg001_mask.ome.tiff").touch()

        layout = discover_layout(tmp_path, cell_table_file=None, image_file=None)
        assert layout.regions[0].image is None
        with pytest.raises(ReaderFormatError, match="a stitched TIFF"):
            discover_layout(tmp_path, cell_table_file=None, image_file="reg001_mask.ome.tiff")

    def test_direct_csv_selects_only_that_file(self, tmp_path: Path) -> None:
        selected = tmp_path / "selected.csv"
        write_csv(selected)
        write_csv(tmp_path / "other.csv")
        write_image(tmp_path / "image.tif")

        layout = discover_layout(selected, cell_table_file=None, image_file=None)
        table = layout.regions[0].cell_table
        assert table is not None
        assert table.path == selected
        assert layout.regions[0].image is None
        with pytest.raises(ValueError, match="read_tables=True"):
            discover_layout(selected, cell_table_file=None, image_file=None, read_tables=False)

    def test_direct_relative_file_uses_its_parents_path_policy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        payload = tmp_path / "outside"
        write_csv(payload)
        (root / "cells.csv").symlink_to(payload)
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ReaderFormatError, match="outside"):
            discover_layout("dataset/cells.csv", cell_table_file=None, image_file=None)
        relaxed = discover_layout("dataset/cells.csv", cell_table_file=None, image_file=None, path_policy="relaxed")
        assert relaxed.regions == discover_layout(root / "cells.csv", cell_table_file=None, image_file=None).regions

    def test_native_symlink_directory_cannot_escape_root(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        markers = tmp_path / "elsewhere"
        markers.mkdir()
        (markers / "reg001_cyc001_ch001_A.tif").touch()
        (root / "stitched").symlink_to(markers, target_is_directory=True)

        with pytest.raises(ReaderFormatError, match="outside"):
            discover_layout(root, cell_table_file=None, image_file=None)
        layout = discover_layout(root, cell_table_file=None, image_file=None, path_policy="relaxed")
        assert layout.regions[0].name == "reg001"


class TestValidateRegionOptions:
    def test_caller_order_is_retained(self) -> None:
        assert validate_region_options(["reg002", "reg001"], read_tables=True, table_selected=False) == (
            "reg002",
            "reg001",
        )

    @pytest.mark.parametrize("regions", [(), ("reg001", "reg001")])
    def test_regions_are_nonempty_and_unique(self, regions: tuple[str, ...]) -> None:
        with pytest.raises(ValueError, match="nonempty and contain distinct"):
            validate_region_options(regions, read_tables=True, table_selected=False)

    def test_regions_cannot_be_a_bare_string(self) -> None:
        with pytest.raises(TypeError, match="sequence of distinct region names"):
            validate_region_options("reg001", read_tables=True, table_selected=False)

    def test_table_option_requires_table_discovery(self) -> None:
        with pytest.raises(ValueError, match="requires read_tables"):
            validate_region_options(None, read_tables=False, table_selected=True)
