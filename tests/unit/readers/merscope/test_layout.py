"""Instrument/VPT artifact ownership, product selection and deterministic image grids."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.merscope._layout import (
    MerscopeLayout,
    Selection,
    discover_layout,
    make_selection,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _selection() -> Selection:
    """Keep valid public defaults together for isolated path tests."""
    return make_selection(
        include_images=True, include_cell_boundaries=True, include_table=True, include_transcripts=True, z_layers=None
    )


def _files(root: Path, *names: str) -> None:
    """Write discovery placeholders without reading storage contents."""
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _standard(root: Path) -> None:
    """Write the complete supported artifact vocabulary in unsorted source order."""
    _files(
        root,
        "cell_by_gene.csv",
        "cell_metadata.csv",
        "cell_boundaries.parquet",
        "detected_transcripts.csv",
        "images/micron_to_mosaic_pixel_transform.csv",
        "images/mosaic_DAPI_z2.tif",
        "images/mosaic_Cyto_z2.tif",
        "images/mosaic_DAPI_z0.tif",
        "images/mosaic_Cyto_z0.tif",
    )


class TestMakeSelection:
    """Require real booleans and an ordered, owned nonnegative plane sequence."""

    def test_owns_order(self) -> None:
        supplied = [2, 0]
        selection = make_selection(
            include_images=True,
            include_cell_boundaries=False,
            include_table=False,
            include_transcripts=False,
            z_layers=supplied,
        )
        supplied.reverse()
        assert selection.z_layers == (2, 0)
        assert _selection().z_layers is None

    @pytest.mark.parametrize(
        "flags", [(1, True, True, True), (True, 0, True, True), (True, True, "yes", True), (True, True, True, None)]
    )
    def test_invalid_inclusion_types(self, flags: tuple[bool, bool, bool, bool]) -> None:
        with pytest.raises(TypeError, match="boolean"):
            make_selection(
                include_images=flags[0],
                include_cell_boundaries=flags[1],
                include_table=flags[2],
                include_transcripts=flags[3],
                z_layers=None,
            )

    @pytest.mark.parametrize(
        "indexes", [pytest.param(1, id="scalar-int"), "1", pytest.param(b"1", id="bytes"), {1, 2}, {1: 2}, iter([1, 2])]
    )
    def test_nonsequence_plane_inputs(
        self, indexes: int | str | bytes | set[int] | dict[int, int] | Iterator[int]
    ) -> None:
        with pytest.raises(TypeError, match="ordered sequence"):
            make_selection(
                include_images=True,
                include_cell_boundaries=False,
                include_table=False,
                include_transcripts=False,
                z_layers=indexes,  # pyrefly: ignore[bad-argument-type] - Validate malformed untyped caller input.
            )

    @pytest.mark.parametrize("indexes", [[], [-1], [True], [1.0], [1, 1]])
    def test_invalid_plane_values(self, indexes: list[float]) -> None:
        with pytest.raises(ValueError, match="z_layers"):
            make_selection(
                include_images=True,
                include_cell_boundaries=False,
                include_table=False,
                include_transcripts=False,
                z_layers=indexes,  # pyrefly: ignore[bad-argument-type] - Runtime integer and duplicate validation.
            )


class TestDiscoverLayout:
    """Resolve selected files once without inventing optional products or assignments."""

    def test_standard_sorted_grid_and_explicit_layer_order(self, tmp_path: Path) -> None:
        _standard(tmp_path)
        inferred = discover_layout(tmp_path, _selection())
        assert isinstance(inferred, MerscopeLayout)
        assert inferred.boundaries == tmp_path / "cell_boundaries.parquet"
        assert inferred.root == tmp_path.resolve()
        assert inferred.names.dataset_id == f"{tmp_path.parent.name}_{tmp_path.name}"
        assert inferred.images is not None
        assert [layer.z for layer in inferred.images.layers] == [0, 2]
        assert [[channel.stain for channel in layer.channels] for layer in inferred.images.layers] == [
            ["Cyto", "DAPI"]
        ] * 2
        assert inferred.tables is not None
        assert inferred.transcripts is not None
        assert inferred.transcripts.assignment == "current"
        selected = discover_layout(
            tmp_path, replace(_selection(), z_layers=(2, 0)), region_name="ROI", slide_name="slide"
        )
        assert selected.images is not None
        assert [layer.z for layer in selected.images.layers] == [2, 0]
        assert selected.names.dataset_id == "slide_ROI"

    @pytest.mark.parametrize("profile", ["cellpose", "watershed"])
    @pytest.mark.parametrize("current", [False, True])
    def test_vpt_mode_selects_analysis_tables_and_explicit_assignment_population(
        self, tmp_path: Path, profile: Literal["cellpose", "watershed"], *, current: bool
    ) -> None:
        _standard(tmp_path)
        analysis = tmp_path / "vpt"
        _files(analysis, "cell_by_gene.csv", "cell_metadata.csv", f"{profile}_micron_space.parquet")
        if current:
            _files(analysis, "detected_transcripts.csv")
        layout = discover_layout(tmp_path, _selection(), vpt_directory="vpt")
        assert isinstance(layout, MerscopeLayout)
        assert layout.tables is not None
        assert layout.tables.counts.parent == analysis
        boundaries, transcripts = layout.boundaries, layout.transcripts
        assert boundaries is not None
        assert boundaries == analysis / f"{profile}_micron_space.parquet"
        assert transcripts is not None
        assert transcripts.assignment == ("current" if current else "original")
        assert transcripts.path.parent == (analysis if current else tmp_path)
        assert transcripts.vocabulary.parent == transcripts.path.parent

    def test_absolute_external_vpt_override(self, tmp_path: Path) -> None:
        root, analysis = tmp_path / "region", tmp_path / "analysis"
        _standard(root)
        _files(analysis, "cell_by_gene.csv", "cell_metadata.csv", "cellpose_micron_space.parquet")
        assert isinstance(discover_layout(root, _selection(), vpt_directory=analysis), MerscopeLayout)

    @pytest.mark.parametrize(
        "artifact",
        [
            "cell_by_gene.csv",
            "cell_metadata.csv",
            "cell_boundaries.parquet",
            "detected_transcripts.csv",
        ],
    )
    @pytest.mark.parametrize("wrong_kind", [False, True])
    def test_required_products_missing_or_wrong_kind(self, tmp_path: Path, artifact: str, *, wrong_kind: bool) -> None:
        _standard(tmp_path)
        path = tmp_path / artifact
        path.unlink()
        if wrong_kind:
            path.mkdir()
        with pytest.raises(IsADirectoryError if wrong_kind else FileNotFoundError):
            discover_layout(tmp_path, _selection())

    def test_all_excluded_products_need_no_artifacts(self, tmp_path: Path) -> None:
        selected = replace(_selection(), images=False, boundaries=False, table=False, transcripts=False)
        layout = discover_layout(tmp_path, selected)
        assert layout.images is None
        assert layout.boundaries is None
        assert layout.transcripts is None
        assert layout.tables is None

    def test_images_only_do_not_require_physical_calibration(self, tmp_path: Path) -> None:
        _files(tmp_path, "images/mosaic_DAPI_z0.tif")
        layout = discover_layout(tmp_path, replace(_selection(), boundaries=False, table=False, transcripts=False))
        assert layout.images is not None
        assert len(layout.images.layers) == 1
        assert layout.images.transform is None
        assert layout.boundaries is None
        assert layout.transcripts is None

    def test_table_only_needs_no_images(self, tmp_path: Path) -> None:
        _files(tmp_path, "cell_by_gene.csv", "cell_metadata.csv")
        layout = discover_layout(tmp_path, replace(_selection(), images=False, boundaries=False, transcripts=False))
        assert layout.tables is not None
        assert layout.boundaries is None
        assert layout.transcripts is None
        assert layout.images is None

    @pytest.mark.parametrize("bad", ["missing", "ambiguous", "wrong_kind", "missing_raw_vocabulary"])
    def test_vpt_boundary_and_fallback_failures(self, tmp_path: Path, bad: str) -> None:
        _standard(tmp_path)
        analysis = tmp_path / "vpt"
        _files(analysis, "cell_by_gene.csv", "cell_metadata.csv")
        if bad != "missing":
            _files(analysis, "cellpose_micron_space.parquet")
        if bad == "ambiguous":
            _files(analysis, "watershed_micron_space.parquet")
        elif bad == "wrong_kind":
            (analysis / "detected_transcripts.csv").mkdir()
        elif bad == "missing_raw_vocabulary":
            (tmp_path / "cell_by_gene.csv").unlink()
        with pytest.raises((ReaderFormatError, FileNotFoundError, IsADirectoryError)):
            discover_layout(tmp_path, _selection(), vpt_directory="vpt")

    def test_image_transform_wrong_kind(self, tmp_path: Path) -> None:
        _files(tmp_path, "images/mosaic_DAPI_z0.tif")
        (tmp_path / "images" / "micron_to_mosaic_pixel_transform.csv").mkdir()
        with pytest.raises(IsADirectoryError):
            discover_layout(tmp_path, replace(_selection(), boundaries=False, table=False, transcripts=False))

    @pytest.mark.parametrize("bad", ["empty", "incomplete", "unknown", "alias", "wrong_kind"])
    def test_invalid_mosaic_grid(self, tmp_path: Path, bad: str) -> None:
        directory = tmp_path / "images"
        directory.mkdir()
        if bad != "empty":
            _files(tmp_path, "images/mosaic_DAPI_z0.tif")
        if bad == "incomplete":
            _files(tmp_path, "images/mosaic_Cyto_z0.tif", "images/mosaic_DAPI_z1.tif")
        elif bad == "alias":
            _files(tmp_path, "images/mosaic_DAPI_z00.tif")
        elif bad == "wrong_kind":
            (directory / "mosaic_DAPI_z0.tif").unlink()
            (directory / "mosaic_DAPI_z0.tif").mkdir()
        selected = replace(
            _selection(), boundaries=False, table=False, transcripts=False, z_layers=(3,) if bad == "unknown" else None
        )
        with pytest.raises((ReaderFormatError, IsADirectoryError)):
            discover_layout(tmp_path, selected)

    @pytest.mark.parametrize("name", ["", " ", ".", "..", "a/b", "a\\b", "a\x00b", 1])
    def test_invalid_name_overrides(self, tmp_path: Path, name: str | int) -> None:
        with pytest.raises((ValueError, TypeError), match="region_name"):
            discover_layout(tmp_path, _selection(), region_name=name)  # pyrefly: ignore[bad-argument-type] - Validate untyped public names before I/O.
        with pytest.raises((ValueError, TypeError), match="slide_name"):
            discover_layout(tmp_path, _selection(), slide_name=name)  # pyrefly: ignore[bad-argument-type] - Both names share the same declared boundary.

    @pytest.mark.parametrize("escape", ["relative_vpt", "artifact"])
    def test_containment(self, tmp_path: Path, escape: str) -> None:
        root = tmp_path / "region"
        _standard(root)
        outside = tmp_path / "outside"
        outside.mkdir()
        if escape == "relative_vpt":
            with pytest.raises(ReaderFormatError, match="root"):
                discover_layout(root, _selection(), vpt_directory="../outside")
        else:
            _files(outside, "cell_by_gene.csv")
            (root / "cell_by_gene.csv").unlink()
            (root / "cell_by_gene.csv").symlink_to(outside / "cell_by_gene.csv")
            with pytest.raises(ReaderFormatError, match="root"):
                discover_layout(root, _selection())
