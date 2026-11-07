"""Instrument/VPT artifact ownership, product selection and deterministic image grids."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers.merscope._layout import (
    MerscopeLayout,
    Names,
    Selection,
    discover_layout,
    validate_options,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _selection() -> Selection:
    """Keep valid public defaults together for isolated path tests."""
    return Selection(images=True, boundaries=True, table=True, transcripts=True, z_layers=None)


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


def _options(**overrides: object) -> Selection:
    """Validate the public defaults with selected overrides."""
    options: dict[str, object] = {
        "vpt_directory": None,
        "z_layers": None,
        "region_name": None,
        "slide_name": None,
        "include_transcripts": True,
        "include_cell_boundaries": True,
        "include_table": True,
        "include_images": True,
    }
    options.update(overrides)
    return validate_options(**options)  # pyrefly: ignore[bad-argument-type] - Runtime validation of untyped options.


class TestValidateOptions:
    """Validate layout options in signature order before any filesystem access."""

    def test_owns_order(self) -> None:
        supplied = [2, 0]
        selection = _options(
            z_layers=supplied, include_cell_boundaries=False, include_table=False, include_transcripts=False
        )
        supplied.reverse()
        assert selection == Selection(images=True, boundaries=False, table=False, transcripts=False, z_layers=(2, 0))
        assert _selection().z_layers is None

    @pytest.mark.parametrize("vpt_directory", ["vpt", Path("vpt"), None])
    def test_accepts_path_values(self, vpt_directory: str | Path | None) -> None:
        assert _options(vpt_directory=vpt_directory) == _selection()

    @pytest.mark.parametrize("vpt_directory", [1, b"vpt", ["vpt"]])
    def test_invalid_vpt_directory_type(self, vpt_directory: object) -> None:
        with pytest.raises(TypeError, match="vpt_directory must be a path or None"):
            _options(vpt_directory=vpt_directory)

    @pytest.mark.parametrize(
        "flag", ["include_transcripts", "include_cell_boundaries", "include_table", "include_images"]
    )
    @pytest.mark.parametrize("value", [1, 0, "yes", None])
    def test_invalid_inclusion_types(self, flag: str, value: object) -> None:
        with pytest.raises(TypeError, match=f"{flag} must be a boolean"):
            _options(**{flag: value})

    @pytest.mark.parametrize(
        "indexes", [pytest.param(1, id="scalar-int"), "1", pytest.param(b"1", id="bytes"), {1, 2}, {1: 2}, iter([1, 2])]
    )
    def test_nonsequence_plane_inputs(self, indexes: object) -> None:
        with pytest.raises(TypeError, match="ordered sequence"):
            _options(z_layers=indexes)

    @pytest.mark.parametrize("indexes", [[True], [1.0], [1, "2"]])
    def test_noninteger_plane_values(self, indexes: list[object]) -> None:
        with pytest.raises(TypeError, match="z_layers must contain integer indexes"):
            _options(z_layers=indexes)

    @pytest.mark.parametrize("indexes", [[], [-1], [1, 1]])
    def test_invalid_plane_values(self, indexes: list[int]) -> None:
        with pytest.raises(ValueError, match="z_layers"):
            _options(z_layers=indexes)

    @pytest.mark.parametrize("parameter", ["region_name", "slide_name"])
    @pytest.mark.parametrize("name", ["", " ", ".", "..", "a/b", "a\\b", "a\x00b", "a b", "__a", "a:b"])
    def test_invalid_names(self, parameter: str, name: str) -> None:
        with pytest.raises(ValueError, match=f"{parameter} must be a valid SpatialData element name"):
            _options(**{parameter: name})

    @pytest.mark.parametrize("parameter", ["region_name", "slide_name"])
    def test_noncharacter_names(self, parameter: str) -> None:
        with pytest.raises(TypeError, match=f"{parameter} must be a string"):
            _options(**{parameter: 1})

    def test_signature_order(self) -> None:
        # Each invalid option is reported before every later invalid option.
        invalid: list[tuple[str, object, type[Exception]]] = [
            ("vpt_directory", 1, TypeError),
            ("z_layers", [-1], ValueError),
            ("region_name", "a b", ValueError),
            ("slide_name", "a b", ValueError),
            ("include_transcripts", 1, TypeError),
            ("include_cell_boundaries", 1, TypeError),
            ("include_table", 1, TypeError),
            ("include_images", 1, TypeError),
        ]
        for index, (parameter, _, error) in enumerate(invalid):
            with pytest.raises(error, match=parameter):
                _options(**{name: value for name, value, _ in invalid[index:]})


class TestDiscoverLayout:
    """Resolve selected files once without inventing optional products or assignments."""

    def test_standard_sorted_grid_and_explicit_layer_order(self, tmp_path: Path) -> None:
        _standard(tmp_path)
        inferred = discover_layout(tmp_path, _selection())
        assert isinstance(inferred, MerscopeLayout)
        assert inferred.boundaries == tmp_path / "cell_boundaries.parquet"
        assert inferred.names.dataset_id == f"{tmp_path.parent.name}_{tmp_path.name}"
        assert inferred.images is not None
        assert [layer.z for layer in inferred.images.layers] == [0, 2]
        assert [[channel.stain for channel in layer.channels] for layer in inferred.images.layers] == [
            ["Cyto", "DAPI"]
        ] * 2
        assert inferred.tables is not None
        assert inferred.tables.counts == tmp_path.resolve() / "cell_by_gene.csv"
        assert inferred.tables.metadata == tmp_path.resolve() / "cell_metadata.csv"
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
        ("artifact", "name"),
        [
            ("cell_by_gene.csv", "count matrix"),
            ("cell_metadata.csv", "cell metadata"),
            ("cell_boundaries.parquet", "cell boundaries"),
            ("detected_transcripts.csv", "transcripts"),
            ("images/mosaic_DAPI_z0.tif", "mosaic channel"),
        ],
    )
    def test_required_products_wrong_kind_name_their_artifact(self, tmp_path: Path, artifact: str, name: str) -> None:
        _standard(tmp_path)
        path = tmp_path / artifact
        path.unlink()
        path.mkdir()
        with pytest.raises(IsADirectoryError) as caught:
            discover_layout(tmp_path, _selection())
        assert f"reader 'merscope', artifact {name!r}" in str(caught.value)

    @pytest.mark.parametrize(
        "artifact", ["cell_by_gene.csv", "cell_metadata.csv", "cell_boundaries.parquet", "detected_transcripts.csv"]
    )
    def test_required_products_missing(self, tmp_path: Path, artifact: str) -> None:
        _standard(tmp_path)
        (tmp_path / artifact).unlink()
        with pytest.raises(FileNotFoundError):
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

    def test_default_names_come_from_directories(self, tmp_path: Path) -> None:
        root = tmp_path / "slide.1" / "region-A"
        root.mkdir(parents=True)
        layout = discover_layout(
            root, replace(_selection(), images=False, boundaries=False, table=False, transcripts=False)
        )
        assert layout.names == Names(region="region-A", slide="slide.1")
        assert layout.names.dataset_id == "slide.1_region-A"

    def test_invalid_directory_names_are_repaired_with_a_warning(self, tmp_path: Path) -> None:
        root = tmp_path / "my slide" / "region:1"
        root.mkdir(parents=True)
        nothing = replace(_selection(), images=False, boundaries=False, table=False, transcripts=False)
        with pytest.warns(UserWarning, match="is not a usable SpatialData element name") as warnings:
            layout = discover_layout(root, nothing)
        region_warning = (
            "merscope: the region directory name 'region:1' is not a usable SpatialData element name; "
            "using 'region_1'. Pass region_name to choose the name explicitly."
        )
        slide_warning = (
            "merscope: the slide (parent) directory name 'my slide' is not a usable SpatialData element name; "
            "using 'my_slide'. Pass slide_name to choose the name explicitly."
        )
        assert [str(warning.message) for warning in warnings] == [region_warning, slide_warning]
        assert layout.names == Names(region="region_1", slide="my_slide")

    def test_explicit_names_replace_directory_names_without_warning(self, tmp_path: Path) -> None:
        root = tmp_path / "my slide" / "region:1"
        root.mkdir(parents=True)
        nothing = replace(_selection(), images=False, boundaries=False, table=False, transcripts=False)
        layout = discover_layout(root, nothing, region_name="ROI", slide_name="S1")
        assert layout.names.dataset_id == "S1_ROI"

    @pytest.mark.parametrize("explicit", [False, True])
    def test_underscore_slide_cannot_start_the_prefix_with_two_underscores(
        self, tmp_path: Path, *, explicit: bool
    ) -> None:
        root = tmp_path / "_" / "region"
        root.mkdir(parents=True)
        nothing = replace(_selection(), images=False, boundaries=False, table=False, transcripts=False)
        with pytest.raises(ValueError, match=r"element-name prefix.*found '__region'"):
            discover_layout(root, nothing, slide_name="_" if explicit else None)

    def test_vpt_boundary_wrong_kind(self, tmp_path: Path) -> None:
        _standard(tmp_path)
        analysis = tmp_path / "vpt"
        _files(analysis, "cell_by_gene.csv", "cell_metadata.csv")
        (analysis / "watershed_micron_space.parquet").mkdir()
        with pytest.raises(IsADirectoryError):
            discover_layout(tmp_path, _selection(), vpt_directory="vpt")

    def test_vpt_boundary_ambiguity_names_both_candidates(self, tmp_path: Path) -> None:
        _standard(tmp_path)
        analysis = tmp_path / "vpt"
        _files(
            analysis,
            "cell_by_gene.csv",
            "cell_metadata.csv",
            "cellpose_micron_space.parquet",
            "watershed_micron_space.parquet",
        )
        with pytest.raises(ReaderFormatError, match="2 candidates") as caught:
            discover_layout(tmp_path, _selection(), vpt_directory="vpt")
        assert caught.value.context == ReaderErrorContext("merscope", "cell boundaries")

    def test_mosaic_listing_failure_propagates(self, tmp_path: Path, mocker: MockerFixture) -> None:
        _files(tmp_path, "images/mosaic_DAPI_z0.tif")
        mocker.patch.object(Path, "iterdir", side_effect=PermissionError("injected"))
        with pytest.raises(PermissionError, match="injected") as caught:
            discover_layout(tmp_path, replace(_selection(), boundaries=False, table=False, transcripts=False))
        assert caught.value.__notes__ == [
            f"While handling reader 'merscope', artifact 'mosaic image grid', path '{tmp_path / 'images'}'."
        ]

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
