"""Documented source paths select independent products without recursive search."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactDirectory, ArtifactFile, DatasetRoot
from spatialdata_io.readers.stereoseq import _layout
from spatialdata_io.readers.stereoseq._layout import (
    AnalysisSource,
    DistanceMask,
    GridImage,
    StainedImage,
    _analysis_source,
    _count_sample,
    _image_identity,
    _image_parts,
    _inventory,
    discover_layout,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from spatialdata_io.readers.stereoseq._layout import Family, ImageRole, StereoseqLayout


def _file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _discover(
    root: Path,
    *,
    sample_id: str | None = None,
    bins: tuple[int, ...] | Literal["all"] = (),
    cell_bins: Literal["none", "unadjusted", "adjusted", "both"] = "none",
    square_counts_file: str | Path | None = None,
    cell_counts_file: str | Path | None = None,
    adjusted_cell_counts_file: str | Path | None = None,
    analysis_files: tuple[str | Path, ...] = (),
    image_roles: tuple[ImageRole, ...] = (),
) -> StereoseqLayout:
    return discover_layout(
        root,
        sample_id=sample_id,
        bins=bins,
        cell_bins=cell_bins,
        square_counts_file=square_counts_file,
        cell_counts_file=cell_counts_file,
        adjusted_cell_counts_file=adjusted_cell_counts_file,
        analysis_files=analysis_files,
        image_roles=image_roles,
    )


class TestAnalysisSource:
    """Filename clues preserve dotted sample IDs and exact decimal identity."""

    @pytest.mark.parametrize(
        ("filename", "population", "resolution"),
        [
            ("chip.v2.bin50_0.50.h5ad", 50, Decimal("0.5")),
            ("chip.v2.bin1_1e-2.spatial.cluster.h5ad", 1, Decimal("0.01")),
            ("chip.v2.cellbin_1.50.h5ad", "cells", Decimal("1.5")),
            ("chip.v2.cellbin_1.50.adjusted.h5ad", "adjusted_cells", Decimal("1.5")),
            ("chip.v2.cell.cluster.h5ad", "cells", None),
            ("chip.v2.adjusted.cell.cluster.h5ad", "adjusted_cells", None),
            ("chip.v2.spatial.cluster.h5ad", None, None),
        ],
    )
    def test_parses_only_documented_forms(
        self, filename: str, population: Literal["cells", "adjusted_cells"] | int | None, resolution: Decimal | None
    ) -> None:
        source = _analysis_source(ArtifactFile(Path(filename)))
        assert source.sample_id == "chip.v2"
        assert source.population == population
        assert source.resolution == resolution

    @pytest.mark.parametrize(
        "filename", ["arbitrary.h5ad", "sample.cellbin_1.adjusted.backup.h5ad", "sample.bin0_1.h5ad"]
    )
    def test_rejects_unknown_names(self, filename: str) -> None:
        with pytest.raises(ReaderFormatError, match="analysis filename"):
            _analysis_source(ArtifactFile(Path(filename)))

    @pytest.mark.parametrize("value", ["0", "0.00", "1e999999999999999999999999"])
    def test_rejects_zero_and_unrepresentable_decimal_resolution(self, value: str) -> None:
        with pytest.raises(ReaderFormatError, match="clustering resolution"):
            _analysis_source(ArtifactFile(Path(f"sample.cellbin_{value}.h5ad")))


class TestCountSample:
    """Compound count suffixes are removed as complete vendor suffixes."""

    @pytest.mark.parametrize("suffix", [".adjusted.cellbin.gef", ".cellbin.gef", ".tissue.gef", ".raw.gef", ".gef"])
    def test_preserves_dots(self, suffix: str) -> None:
        assert _count_sample(Path(f"chip.v2{suffix}")) == "chip.v2"

    def test_ignores_other_files(self) -> None:
        assert _count_sample(Path("readme.txt")) is None

    @pytest.mark.parametrize("qualifier", ["HE", "region.label", "region.raw.label"])
    def test_explicit_sample_disambiguates_square_population_qualifiers(self, qualifier: str) -> None:
        path = Path(f"chip.v2.{qualifier}.gef")
        assert _count_sample(path) == f"chip.v2.{qualifier}"
        assert _count_sample(path, sample_id="chip.v2") == "chip.v2"

    @pytest.mark.parametrize("suffix", [".adjusted.cellbin.gef", ".cellbin.gef", ".tissue.gef", ".raw.gef"])
    def test_explicit_selector_does_not_weaken_exact_product_suffix_identity(self, suffix: str) -> None:
        assert _count_sample(Path(f"chip.v2.other{suffix}"), sample_id="chip.v2") == "chip.v2.other"

    @pytest.mark.parametrize("name", ["chip.v20.HE.gef", "chip.v2..gef", "chip.v2..HE.gef"])
    def test_requires_complete_sample_prefix_and_nonempty_qualifiers(self, name: str) -> None:
        assert _count_sample(Path(name), sample_id="chip.v2") != "chip.v2"


class TestImageParts:
    """Image roles distinguish expression intensity, cut masks, and expansion distances."""

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("bin1_img.tif", ("", "expression", 0)),
            ("bin1_img_tissue_cut.tif", ("", "tissue_cut", 0)),
            ("chip_HE_regist.tif", ("chip_HE", "registered", 0)),
            ("chip_HE_mask.tif", ("chip_HE", "mask", 0)),
            ("chip_HE_tissue_cut.tif", ("chip_HE", "tissue_cut", 0)),
            ("chip_HE_mask_edm_dis_10.tif", ("chip_HE", "distance_mask", 10)),
            ("chip_HE_mask_edm_dis_0.tif", None),
            ("chip_HE_fov_stitched.tif", None),
        ],
    )
    def test_recognizes_exact_roles(self, filename: str, expected: tuple[str, str, int] | None) -> None:
        assert _image_parts(Path(filename)) == expected


class TestImageIdentity:
    """Family conventions retain complete sample/stain components when a selector is available."""

    @pytest.mark.parametrize(
        ("stem", "family", "sample", "expected"),
        [
            ("chip_v2", "compact_numbered", None, ("chip_v2", "")),
            ("chip_v2", "compact_numbered", "chip_v2", ("chip_v2", "")),
            ("HE_chip_v2", "numbered", "chip_v2", ("chip_v2", "HE")),
            ("HE_chip_v2", "numbered", None, ("chip_v2", "HE")),
            ("chip_v2_HE", "outs", "chip_v2", ("chip_v2", "HE")),
            ("chip_v2_HE", "outs", None, ("chip_v2", "HE")),
            ("chip", "flat", None, ("chip", "")),
        ],
    )
    def test_interprets_family_order(
        self, stem: str, family: Family, sample: str | None, expected: tuple[str, str]
    ) -> None:
        assert _image_identity(stem, family, sample) == expected


class TestInventory:
    """One bounded inventory identifies structural layout independently from release."""

    @pytest.mark.parametrize("directory", ["02.count", "03.count", "feature_expression"])
    def test_scans_known_directory_once(self, tmp_path: Path, mocker: MockerFixture, directory: str) -> None:
        _file(tmp_path / directory / "sample.gef")
        _file(tmp_path / "unrelated" / "nested" / "different.gef")

        spy = mocker.spy(Path, "iterdir")
        _, files = _inventory(DatasetRoot(ArtifactDirectory(tmp_path)))

        assert len(files) == 1
        assert [call.args[0] for call in spy.call_args_list] == [tmp_path, tmp_path / directory]

    @pytest.mark.parametrize("directory", ["03.count", "outs"])
    def test_rejects_wrong_kind_structural_artifact(self, tmp_path: Path, directory: str) -> None:
        _file(tmp_path / directory)
        with pytest.raises(NotADirectoryError):
            _discover(tmp_path)

    def test_rejects_mixed_families(self, tmp_path: Path) -> None:
        (tmp_path / "02.count").mkdir()
        (tmp_path / "03.count").mkdir()

        with pytest.raises(ReaderFormatError, match="conflicting families"):
            _discover(tmp_path)

    def test_rejects_external_structural_symlink(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (root / "03.count").symlink_to(outside, target_is_directory=True)
        with pytest.raises(ReaderFormatError, match="contain"):
            _discover(root)

    def test_rejects_task_and_direct_outs_combination(self, tmp_path: Path) -> None:
        (tmp_path / "outs").mkdir()
        (tmp_path / "image").mkdir()
        with pytest.raises(ReaderFormatError, match="conflicting families"):
            _discover(tmp_path)


class TestDiscoverLayout:
    """Each selected product has one required artifact; other products stay optional."""

    def test_compact_paths_from_pinned_workflow(self, tmp_path: Path) -> None:
        # https://github.com/STOmics/SAW/blob/41e70aa12bb16da8c9ddae344863cef2ca12f735/script/stereoPipeline.sh#L285-L422
        paths = (
            "02.count/sample.raw.gef",
            "04.tissuecut/sample.tissue.gef",
            "041.cellcut/sample.cellbin.gef",
            "05.spatialcluster/sample.spatial.cluster.h5ad",
            "051.cellcluster/sample.cell.cluster.h5ad",
            "03.register/sample_regist.tif",
            "03.register/sample_mask.tif",
        )
        for relative_path in paths:
            _file(tmp_path / relative_path)

        result = _discover(
            tmp_path,
            bins=(1,),
            cell_bins="unadjusted",
            analysis_files=paths[3:5],
            image_roles=("registered", "mask"),
        )

        assert result.family == "compact_numbered"
        assert result.sample_id == "sample"
        assert [source.file for source in result.squares] == [tmp_path / paths[1]]
        assert [source.file for source in result.cells] == [tmp_path / paths[2]]
        assert [source.file for source in result.analyses] == [tmp_path / name for name in paths[3:5]]
        assert [source.population for source in result.analyses] == [None, "cells"]
        assert {source.file for source in result.images} == {tmp_path / name for name in paths[5:]}

    def test_numbered_paths_from_saw7_user_manuals(self, tmp_path: Path) -> None:
        # Literal A8/A9 output paths, independent of the shared fixture writer:
        # https://github.com/STOmics/SAW/blob/0808e44619f84b67d44c063b2fd24762f6633051/Documents/UserManual/Stereo-seq_Analysis_Workflow_User_Manual_A8.pdf
        # https://github.com/STOmics/SAW/blob/cebf5171105c7896539535a60ac8f293e2f62c4f/Documents/UserManual/Stereo-seq_Analysis_Workflow_User_Manual_A9.pdf
        count_paths = (
            "03.count/sample.raw.gef",
            "05.tissuecut/sample.tissue.gef",
            "051.cellcut/sample.cellbin.gef",
            "051.cellcut/sample.adjusted.cellbin.gef",
        )
        analysis_paths = (
            "06.spatialcluster/sample.bin200_1.0.spatial.cluster.h5ad",
            "061.cellcluster/sample.cell.cluster.h5ad",
            "061.cellcluster/sample.adjusted.cell.cluster.h5ad",
        )
        image_paths = (
            "04.register/HE_sample_regist.tif",
            "04.register/HE_sample_mask.tif",
            "04.register/HE_sample_tissue_cut.tif",
            "051.cellcut/HE_sample_mask_edm_dis_10.tif",
        )
        for relative_path in (*count_paths, *analysis_paths, *image_paths):
            _file(tmp_path / relative_path)

        result = _discover(
            tmp_path,
            bins=(200,),
            cell_bins="both",
            analysis_files=analysis_paths,
            image_roles=("registered", "mask", "tissue_cut", "distance_mask"),
        )

        assert result.family == "numbered"
        assert result.sample_id == "sample"
        assert [source.file for source in result.squares] == [tmp_path / count_paths[1]]
        assert [source.file for source in result.cells] == [tmp_path / name for name in count_paths[2:]]
        assert [source.file for source in result.analyses] == [tmp_path / name for name in analysis_paths]
        assert [source.population for source in result.analyses] == [200, "cells", "adjusted_cells"]
        assert {source.file for source in result.images} == {tmp_path / name for name in image_paths}
        assert [source.name for source in result.images] == [
            "HE_distance_mask_10",
            "HE_mask",
            "HE_registered",
            "HE_tissue_cut",
        ]

    @pytest.mark.parametrize("direct_outs", [False, True], ids=["task-root", "outs-root"])
    @pytest.mark.parametrize("square_name", ["sample.tissue.gef", "sample.gef"])
    def test_shared_paths_from_saw8_output_tables(self, tmp_path: Path, square_name: str, *, direct_outs: bool) -> None:
        # These three manuals document the same output-path grammar:
        # https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.0/analysis/outputs/count-outputs
        # https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.1/analysis/outputs/count-outputs
        # https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.2/analysis/outputs/count-outputs
        outs = tmp_path / "outs"
        count_paths = (
            "feature_expression/sample.raw.gef",
            f"feature_expression/{square_name}",
            "feature_expression/sample.cellbin.gef",
            "feature_expression/sample.adjusted.cellbin.gef",
        )
        analysis_paths = (
            "analysis/sample.bin20_1.0.h5ad",
            "analysis/sample.cellbin_0.5.h5ad",
            "analysis/sample.cellbin_0.5.adjusted.h5ad",
        )
        image_paths = (
            "image/sample_HE_regist.tif",
            "image/sample_HE_mask.tif",
            "image/sample_HE_tissue_cut.tif",
            "image/sample_HE_mask_edm_dis_10.tif",
            "image/bin1_img_tissue_cut.tif",
        )
        for relative_path in (*count_paths, *analysis_paths, *image_paths):
            _file(outs / relative_path)
        root = outs if direct_outs else tmp_path

        result = _discover(
            root,
            bins=(20,),
            cell_bins="both",
            analysis_files=tuple((outs / name).relative_to(root) for name in analysis_paths),
            image_roles=("registered", "mask", "tissue_cut", "distance_mask"),
        )

        assert result.family == "outs"
        assert result.sample_id == "sample"
        assert [source.file for source in result.squares] == [outs / count_paths[1]]
        assert [source.file for source in result.cells] == [outs / name for name in count_paths[2:]]
        assert [source.file for source in result.analyses] == [outs / name for name in analysis_paths]
        assert [source.population for source in result.analyses] == [20, "cells", "adjusted_cells"]
        assert {source.file for source in result.images} == {outs / name for name in image_paths}
        assert [source.name for source in result.images] == [
            "HE_distance_mask_10",
            "HE_mask",
            "HE_registered",
            "HE_tissue_cut",
            "expression_tissue_cut",
        ]

    @pytest.mark.parametrize(
        ("directory", "family", "filename"),
        [
            ("04.tissuecut", "compact_numbered", "chip.v2.gef"),
            ("02.count", "compact_numbered", "chip.v2.gef"),
            ("05.tissuecut", "numbered", "chip.v2.tissue.gef"),
            ("03.count", "numbered", "chip.v2.gef"),
            ("outs/feature_expression", "outs", "chip.v2.tissue.gef"),
            ("feature_expression", "outs", "chip.v2.tissue.gef"),
        ],
    )
    def test_reads_only_square_selection(self, tmp_path: Path, directory: str, family: Family, filename: str) -> None:
        path = _file(tmp_path / directory / filename)

        result = _discover(tmp_path, bins=(1, 50))

        assert result.family == family
        assert result.sample_id == "chip.v2"
        assert result.squares[0].file == path
        assert result.squares[0].bins == (1, 50)
        assert result.cells == result.analyses == result.images == ()

    @pytest.mark.parametrize("directory", ["041.cellcut", "051.cellcut", "outs/feature_expression"])
    def test_selects_both_independent_cell_populations(self, tmp_path: Path, directory: str) -> None:
        raw = _file(tmp_path / directory / "sample.cellbin.gef")
        adjusted = _file(tmp_path / directory / "sample.adjusted.cellbin.gef")

        result = _discover(tmp_path, cell_bins="both")

        assert [(source.file, source.kind) for source in result.cells] == [(raw, "unadjusted"), (adjusted, "adjusted")]
        assert result.squares == result.analyses == result.images == ()

    def test_requires_sample_selector_for_multiple_samples(self, tmp_path: Path) -> None:
        _file(tmp_path / "03.count" / "sample.gef")
        chosen = _file(tmp_path / "03.count" / "other.gef")

        with pytest.raises(ReaderFormatError, match="sample_id"):
            _discover(tmp_path, bins=(1,))

        assert _discover(tmp_path, sample_id="other", bins="all").squares[0].file == chosen

    def test_missing_selected_artifact_uses_precise_path_error(self, tmp_path: Path) -> None:
        _file(tmp_path / "03.count" / "sample.raw.gef")
        with pytest.raises(FileNotFoundError):
            _discover(tmp_path, bins=(1,))
        with pytest.raises(FileNotFoundError):
            _discover(tmp_path, cell_bins="unadjusted")

    def test_rejects_full_and_tissue_ambiguity(self, tmp_path: Path) -> None:
        _file(tmp_path / "03.count" / "sample.gef")
        _file(tmp_path / "05.tissuecut" / "sample.tissue.gef")
        with pytest.raises(ReaderFormatError, match="exactly one candidate"):
            _discover(tmp_path, bins=(1,))

    def test_full_gef_in_two_documented_locations_is_ambiguous(self, tmp_path: Path) -> None:
        _file(tmp_path / "03.count" / "sample.gef")
        _file(tmp_path / "05.tissuecut" / "sample.gef")
        with pytest.raises(ReaderFormatError, match="exactly one candidate"):
            _discover(tmp_path, bins=(1,))

    @pytest.mark.parametrize("relative", [True, False])
    def test_caller_overrides_resolve_once_and_allow_absolute_external_paths(
        self, tmp_path: Path, mocker: MockerFixture, *, relative: bool
    ) -> None:
        root = tmp_path / "root"
        root.mkdir()
        file = _file((root if relative else tmp_path) / "sample.raw.gef")

        spy = mocker.spy(_layout, "require_caller_file")
        result = _discover(root, bins=(1,), square_counts_file=file.name if relative else file)

        assert result.family == "flat"
        assert result.squares[0].file == file
        spy.assert_called_once()

    def test_flat_automatic_count_discovery_requires_explicit_source(self, tmp_path: Path) -> None:
        _file(tmp_path / "sample.gef")
        with pytest.raises(ReaderFormatError, match="explicit"):
            _discover(tmp_path, bins=(1,))

    def test_wrong_kind_selected_count_raises_precise_error(self, tmp_path: Path) -> None:
        (tmp_path / "03.count" / "sample.gef").mkdir(parents=True)
        with pytest.raises(IsADirectoryError):
            _discover(tmp_path, bins=(1,))

    def test_missing_override_raises_precise_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _discover(tmp_path, bins=(1,), square_counts_file="sample.gef")

    def test_selected_metadata_file_cannot_escape_dataset(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        (root / "03.count").mkdir(parents=True)
        target = _file(tmp_path / "sample.gef")
        (root / "03.count" / target.name).symlink_to(target)
        with pytest.raises(ReaderFormatError, match="contain"):
            _discover(root, bins=(1,))

    def test_rejects_selected_override_sample_disagreement(self, tmp_path: Path) -> None:
        override = _file(tmp_path / "other.cellbin.gef")
        with pytest.raises(ReaderFormatError, match="override/sample disagreement"):
            _discover(tmp_path, sample_id="sample", cell_bins="unadjusted", cell_counts_file=override)

    @pytest.mark.parametrize("qualifier", ["HE", "region.label", "region.raw.label"])
    def test_explicit_square_override_accepts_documented_population_qualifiers(
        self, tmp_path: Path, qualifier: str
    ) -> None:
        override = _file(tmp_path / f"chip.v2.{qualifier}.gef")
        result = _discover(tmp_path, sample_id="chip.v2", bins=(1,), square_counts_file=override)
        assert result.sample_id == "chip.v2"
        assert result.squares[0].file == override

    @pytest.mark.parametrize("suffix", [".cellbin.gef", ".tissue.gef", ".raw.gef"])
    def test_explicit_square_override_keeps_strong_suffix_sample_checks(self, tmp_path: Path, suffix: str) -> None:
        override = _file(tmp_path / f"chip.v2.other{suffix}")
        with pytest.raises(ReaderFormatError, match="override/sample disagreement"):
            _discover(tmp_path, sample_id="chip.v2", bins=(1,), square_counts_file=override)

    def test_ignores_count_named_file_in_wrong_vendor_directory(self, tmp_path: Path) -> None:
        _file(tmp_path / "04.register" / "sample.cellbin.gef")
        with pytest.raises(FileNotFoundError):
            _discover(tmp_path, cell_bins="unadjusted")

    def test_analysis_only_needs_no_count_products(self, tmp_path: Path) -> None:
        path = _file(tmp_path / "outs" / "analysis" / "chip.v2.cellbin_0.5.adjusted.h5ad")

        result = _discover(tmp_path, analysis_files=(path.relative_to(tmp_path),))

        assert result.sample_id == "chip.v2"
        assert result.analyses == (AnalysisSource(ArtifactFile(path), "adjusted_cells", Decimal("0.5"), "chip.v2"),)
        assert result.cells == result.squares == result.images == ()

    def test_rejects_selected_analysis_sample_disagreement(self, tmp_path: Path) -> None:
        path = _file(tmp_path / "other.cell.cluster.h5ad")
        with pytest.raises(ReaderFormatError, match="analysis/sample disagreement"):
            _discover(tmp_path, sample_id="sample", analysis_files=(path,))

    @pytest.mark.parametrize("duplicate", ["normalized_resolution", "repeated_source", "historical_cells"])
    def test_rejects_duplicate_analysis_identity_before_loading(self, tmp_path: Path, duplicate: str) -> None:
        first = _file(tmp_path / "sample.cellbin_0.5.h5ad")
        if duplicate == "normalized_resolution":
            second = _file(tmp_path / "sample.cellbin_0.50.h5ad")
        elif duplicate == "repeated_source":
            first = second = _file(tmp_path / "sample.spatial.cluster.h5ad")
        else:
            first = _file(tmp_path / "a" / "sample.cell.cluster.h5ad")
            second = _file(tmp_path / "b" / "sample.cell.cluster.h5ad")
        with pytest.raises(ReaderFormatError, match="duplicate analysis"):
            _discover(tmp_path, analysis_files=(first, second))

    def test_does_not_require_unselected_missing_products(self, tmp_path: Path) -> None:
        (tmp_path / "051.cellcut").mkdir()
        result = _discover(tmp_path)
        assert result.sample_id is None
        assert result.squares == result.cells == result.analyses == result.images == ()


class TestImages:
    """All stains for selected roles load deterministically with distinct scientific names."""

    @pytest.mark.parametrize(
        ("directory", "prefix", "name"),
        [
            ("03.register", "chip_v2", "registered"),
            ("04.register", "HE_chip_v2", "HE_registered"),
            ("outs/image", "chip_v2_HE", "HE_registered"),
            (".", "chip_v2_HE", "HE_registered"),
        ],
    )
    def test_image_only_supports_family_sample_order(
        self, tmp_path: Path, directory: str, prefix: str, name: str
    ) -> None:
        path = _file(tmp_path / directory / f"{prefix}_regist.tif")

        result = _discover(tmp_path, image_roles=("registered",))

        assert result.sample_id == "chip_v2"
        assert result.images[0].file == path
        assert result.images[0].name == name

    def test_selected_stains_are_sorted_and_other_samples_ignored(self, tmp_path: Path) -> None:
        for name in ("HE_sample_regist.tif", "DAPI_sample_regist.tif", "HE_other_regist.tif", "readme.txt"):
            _file(tmp_path / "04.register" / name)

        result = _discover(tmp_path, sample_id="sample", image_roles=("registered",))

        assert [source.name for source in result.images] == ["DAPI_registered", "HE_registered"]

    def test_grid_expression_and_tissue_mask_remain_distinct(self, tmp_path: Path) -> None:
        expression = _file(tmp_path / "05.tissuecut" / "bin1_img.tif")
        mask = _file(tmp_path / "05.tissuecut" / "bin1_img_tissue_cut.tif")
        result = _discover(tmp_path, image_roles=("tissue_cut", "expression"))
        assert result.sample_id is None
        assert [(source.file, source.name) for source in result.images] == [
            (expression, "expression"),
            (mask, "expression_tissue_cut"),
        ]

    @pytest.mark.parametrize(
        ("directory", "filename"),
        [
            ("051.cellcut", "HE_sample_mask_edm_dis_10.tif"),
            ("outs/image", "sample_HE_mask_edm_dis_10.tif"),
        ],
    )
    def test_distance_mask_preserves_stain_and_expansion(self, tmp_path: Path, directory: str, filename: str) -> None:
        path = _file(tmp_path / directory / filename)

        source = _discover(tmp_path, image_roles=("distance_mask",)).images[0]

        assert isinstance(source, DistanceMask)
        assert source.file == path
        assert source.distance == 10
        assert source.name == "HE_distance_mask_10"

    def test_distance_mask_without_stain_is_unsupported(self, tmp_path: Path) -> None:
        _file(tmp_path / "03.register" / "sample_mask_edm_dis_10.tif")
        with pytest.raises(ReaderFormatError, match="without stain"):
            _discover(tmp_path, image_roles=("distance_mask",))

    def test_missing_selected_image_role_fails(self, tmp_path: Path) -> None:
        _file(tmp_path / "04.register" / "HE_sample_regist.tif")
        with pytest.raises(ReaderFormatError, match="missing image roles"):
            _discover(tmp_path, image_roles=("mask",))

    def test_duplicate_unstained_role_is_ambiguous(self, tmp_path: Path) -> None:
        _file(tmp_path / "03.register" / "sample_regist.tif")
        _file(tmp_path / "04.tissuecut" / "sample_regist.tif")
        with pytest.raises(ReaderFormatError, match="duplicate image"):
            _discover(tmp_path, image_roles=("registered",))

    def test_wrong_kind_and_external_symlink_images_fail(self, tmp_path: Path) -> None:
        directory = tmp_path / "04.register"
        directory.mkdir()
        (directory / "HE_sample_mask.tif").mkdir()
        with pytest.raises(IsADirectoryError):
            _discover(tmp_path, image_roles=("mask",))
        target = _file(tmp_path.parent / f"{tmp_path.name}_outside_regist.tif")
        (directory / "HE_sample_regist.tif").symlink_to(target)
        with pytest.raises(ReaderFormatError, match="contain"):
            _discover(tmp_path, image_roles=("registered",))

    def test_images_in_unrelated_vendor_directory_are_not_selected(self, tmp_path: Path) -> None:
        _file(tmp_path / "03.count" / "HE_sample_regist.tif")
        with pytest.raises(ReaderFormatError, match="missing image roles"):
            _discover(tmp_path, image_roles=("registered",))

    def test_cellcut_accepts_only_the_documented_distance_mask_role(self, tmp_path: Path) -> None:
        _file(tmp_path / "051.cellcut" / "HE_sample_regist.tif")

        with pytest.raises(ReaderFormatError, match="missing image roles"):
            _discover(tmp_path, image_roles=("registered",))


class TestImageNames:
    """Normalized descriptor names retain role and stain without sample prefixes."""

    def test_stained_grid_and_distance_names(self) -> None:
        file = ArtifactFile(Path("sample_HE_mask.tif"))
        assert StainedImage(file, "mask", "HE").name == "HE_mask"
        assert GridImage(file, "mask").name == "mask"
        assert DistanceMask(file, "HE", 25).name == "HE_distance_mask_25"
