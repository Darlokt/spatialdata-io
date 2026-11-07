"""Validate ordered product selection, ROI names and deterministic seqFISH artifact ownership."""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.seqfish._layout import Selection, discover_layout, make_selection

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_mock import MockerFixture


def _selection(*, rois: tuple[str, ...] | None = None) -> Selection:
    """Select only required tabular products for isolated discovery tests."""
    return make_selection(
        include_images=False,
        include_labels=False,
        include_transcripts=False,
        include_shapes=False,
        shape_geometry="polygons",
        invalid_geometries="keep",
        rois=rois,
    )


def _write_pairs(root: Path, names: tuple[str, ...] = ("Roi2", "Roi1")) -> None:
    """Write artifact placeholders; discovery owns paths rather than parsing."""
    for name in names:
        for suffix in ("CellxGene.csv", "CellCoordinates.csv"):
            (root / f"{name}_{suffix}").touch()


class TestMakeSelection:
    """Reject invalid public values in signature order and own the caller's ROI order."""

    def test_owns_borrowed_roi_order_and_shape_selection(self) -> None:
        rois = ["Roi2", "Roi1"]
        selection = make_selection(
            include_images=True,
            include_labels=False,
            include_transcripts=True,
            include_shapes=True,
            shape_geometry="both",
            invalid_geometries="drop",
            rois=rois,
        )
        rois.reverse()
        assert selection == Selection(
            images=True,
            labels=False,
            transcripts=True,
            shapes="both",
            invalid_geometries="drop",
            rois=("Roi2", "Roi1"),
        )
        assert _selection().shapes == "none"

    @pytest.mark.parametrize(
        "flags",
        [(1, False, False, False), (False, 0, False, False), (False, False, "yes", False), (False, False, False, None)],
    )
    def test_inclusion_flags_require_booleans(self, flags: tuple[bool, bool, bool, bool]) -> None:
        with pytest.raises(TypeError, match="must be a boolean"):
            make_selection(
                include_images=flags[0],
                include_labels=flags[1],
                include_transcripts=flags[2],
                include_shapes=flags[3],
                shape_geometry="polygons",
                invalid_geometries="keep",
                rois=None,
            )

    @pytest.mark.parametrize(
        ("geometry", "error"),
        [("polygon", ValueError), ("", ValueError), ("POLYGONS", ValueError), (1, TypeError), (None, TypeError)],
    )
    def test_geometry_must_be_supported(self, geometry: str | int | None, error: type[Exception]) -> None:
        with pytest.raises(error, match="shape_geometry"):
            make_selection(
                include_images=False,
                include_labels=False,
                include_transcripts=False,
                include_shapes=False,
                shape_geometry=geometry,  # pyrefly: ignore[bad-argument-type] - Validate untyped public choices even for excluded products.
                invalid_geometries="keep",
                rois=None,
            )

    @pytest.mark.parametrize(("policy", "error"), [("repair", ValueError), (None, TypeError)])
    def test_invalid_geometry_policy(self, policy: str | None, error: type[Exception]) -> None:
        with pytest.raises(error, match="invalid_geometries"):
            make_selection(
                include_images=False,
                include_labels=False,
                include_transcripts=False,
                include_shapes=False,
                shape_geometry="polygons",
                invalid_geometries=policy,  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.
                rois=None,
            )

    def test_options_are_validated_in_signature_order(self) -> None:
        with pytest.raises(ValueError, match="shape_geometry"):
            make_selection(
                include_images=False,
                include_labels=False,
                include_transcripts=False,
                include_shapes=False,
                shape_geometry="invalid",  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.
                invalid_geometries="invalid",  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.
                rois=[],
            )
        with pytest.raises(ValueError, match="invalid_geometries"):
            make_selection(
                include_images=False,
                include_labels=False,
                include_transcripts=False,
                include_shapes=False,
                shape_geometry="polygons",
                invalid_geometries="invalid",  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.
                rois=[],
            )

    @pytest.mark.parametrize(
        ("rois", "message"),
        [
            ("Roi1", "rois must be a sequence"),
            pytest.param(b"Roi1", "rois must be a sequence", id="bytes"),
            ({"Roi1"}, "rois must be a sequence"),
            ({"Roi1": 1}, "rois must be a sequence"),
            (1, "rois must be a sequence"),
            ([1], "only strings"),
            (["Roi1", None], "only strings"),
        ],
    )
    def test_roi_types(self, rois: object, message: str) -> None:
        with pytest.raises(TypeError, match=message):
            make_selection(
                include_images=False,
                include_labels=False,
                include_transcripts=False,
                include_shapes=False,
                shape_geometry="polygons",
                invalid_geometries="keep",
                rois=rois,  # pyrefly: ignore[bad-argument-type] - Exercise malformed caller-owned selections.
            )

    @pytest.mark.parametrize(("rois", "message"), [((), "at least one ROI"), (("Roi1", "Roi1"), "duplicate")])
    def test_empty_or_duplicate_rois(self, rois: tuple[str, ...], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            _selection(rois=rois)

    @pytest.mark.parametrize("prefix", ["", ".", "Roi 1", "a\\b"])
    def test_any_string_is_an_exact_prefix(self, prefix: str) -> None:
        assert _selection(rois=(prefix,)).rois == (prefix,)


class TestDiscoverLayout:
    """Resolve only selected products while enforcing paired artifacts and containment."""

    def test_sorted_inference_explicit_order_and_normalized_root(self, tmp_path: Path) -> None:
        _write_pairs(tmp_path)
        inferred = discover_layout(tmp_path / ".", _selection())
        assert isinstance(inferred, tuple)
        assert [roi.name for roi in inferred] == ["Roi1", "Roi2"]
        assert all(roi.pixels is None and roi.transcripts is None for roi in inferred)
        for roi in inferred:
            assert roi.counts == (tmp_path / f"{roi.name}_CellxGene.csv").resolve()
            assert roi.cells == (tmp_path / f"{roi.name}_CellCoordinates.csv").resolve()
        explicit = discover_layout(str(tmp_path), _selection(rois=("Roi2", "Roi1")))
        assert [roi.name for roi in explicit] == ["Roi2", "Roi1"]

    def test_lexical_prefix_order_is_independent_of_file_order(self, tmp_path: Path) -> None:
        # File order puts "a-b_..." before "a_..." because '-' sorts before '_'.
        _write_pairs(tmp_path, ("a-b", "a"))
        assert [roi.name for roi in discover_layout(tmp_path, _selection())] == ["a", "a-b"]

    @pytest.mark.parametrize("kind", ["missing", "file"])
    def test_root_missing_or_wrong_kind(self, tmp_path: Path, kind: str) -> None:
        path = tmp_path / "dataset"
        if kind == "file":
            path.touch()
        with pytest.raises(FileNotFoundError if kind == "missing" else NotADirectoryError):
            discover_layout(path, _selection())

    def test_empty_layout(self, tmp_path: Path) -> None:
        (tmp_path / "Roi1_DAPI.tiff").touch()
        with pytest.raises(ReaderFormatError, match="no ROI files") as caught:
            discover_layout(tmp_path, _selection())
        assert caught.value.context.artifact == "ROI layout"

    def test_listing_failure_keeps_error_identity_and_context(self, tmp_path: Path, mocker: MockerFixture) -> None:
        error = PermissionError("injected listing failure")

        def entries() -> Iterator[Path]:
            yield tmp_path / "Roi1_CellxGene.csv"
            raise error

        mocker.patch.object(Path, "iterdir", return_value=entries())
        with pytest.raises(PermissionError) as caught:
            discover_layout(tmp_path, _selection())
        assert caught.value is error
        assert any("seqfish" in note and "ROI layout" in note for note in error.__notes__)

    @pytest.mark.parametrize("rois", [("Roi3",), ("Roi1", "a/b")])
    def test_unknown_requested_roi(self, tmp_path: Path, rois: tuple[str, ...]) -> None:
        _write_pairs(tmp_path)
        with pytest.raises(ReaderFormatError, match=r"unknown ROIs \['(Roi3|a/b)'\]"):
            discover_layout(tmp_path, _selection(rois=rois))

    @pytest.mark.parametrize("suffix", ["CellCoordinates.csv", "CellxGene.csv"])
    @pytest.mark.parametrize("wrong_kind", [False, True])
    def test_missing_or_wrong_kind_required_pair(self, tmp_path: Path, suffix: str, *, wrong_kind: bool) -> None:
        _write_pairs(tmp_path, ("Roi1",))
        path = tmp_path / f"Roi1_{suffix}"
        path.unlink()
        if wrong_kind:
            path.mkdir()
        with pytest.raises(IsADirectoryError if wrong_kind else FileNotFoundError) as caught:
            discover_layout(tmp_path, _selection())
        assert f"artifact '{suffix.removesuffix('.csv')}'" in str(caught.value)

    @pytest.mark.parametrize("product", ["images", "labels", "transcripts", "polygons", "both", "circles"])
    def test_only_selected_artifacts_are_required(
        self, tmp_path: Path, product: Literal["images", "labels", "transcripts", "polygons", "both", "circles"]
    ) -> None:
        _write_pairs(tmp_path, ("Roi1",))
        selection = _selection()
        if product == "images":
            selection = replace(selection, images=True)
        elif product == "labels":
            selection = replace(selection, labels=True)
        elif product == "transcripts":
            selection = replace(selection, transcripts=True)
        else:
            selection = replace(selection, shapes=product)
        expected = []
        if product in {"images", "labels", "polygons", "both"}:
            expected.append("DAPI.tiff")
        if product == "labels":
            expected.append("Segmentation.tiff")
        if product in {"polygons", "both"}:
            expected.append("Boundaries.geojson")
        if product == "transcripts":
            expected.append("TranscriptList.csv")
        for suffix in expected:
            with pytest.raises(FileNotFoundError, match=suffix):
                discover_layout(tmp_path, selection)
            (tmp_path / f"Roi1_{suffix}").touch()
        layout = discover_layout(tmp_path, selection)
        roi = layout[0]
        assert (roi.pixels is None) == (product in {"circles", "transcripts"})
        assert (roi.transcripts is not None) == (product == "transcripts")

    def test_explicit_subset_ignores_an_unselected_incomplete_roi(self, tmp_path: Path) -> None:
        _write_pairs(tmp_path, ("Roi1",))
        (tmp_path / "Roi2_CellxGene.csv").touch()
        assert [roi.name for roi in discover_layout(tmp_path, _selection(rois=("Roi1",)))] == ["Roi1"]

    def test_metadata_derived_symlink_cannot_escape_root(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _write_pairs(root, ("Roi1",))
        outside = tmp_path / "outside.csv"
        outside.touch()
        counts = root / "Roi1_CellxGene.csv"
        counts.unlink()
        counts.symlink_to(outside)
        with pytest.raises(ReaderFormatError, match="root"):
            discover_layout(root, _selection())


class TestElementNames:
    """Repair file-derived ROI names once and reject names that collide ignoring case."""

    @pytest.mark.parametrize(("prefix", "name"), [("Roi 1", "Roi_1"), ("a:b", "a_b"), ("__x", "_x"), ("", "unnamed")])
    def test_invalid_prefix_is_sanitized_with_one_warning(self, tmp_path: Path, prefix: str, name: str) -> None:
        _write_pairs(tmp_path, (prefix, "Roi2"))
        with pytest.warns(UserWarning, match="ROI file prefix") as caught:
            layout = discover_layout(tmp_path, _selection())
        assert [str(warning.message) for warning in caught] == [
            (
                f"seqfish: the ROI file prefix {prefix!r} is not a usable SpatialData element name; using {name!r}. "
                "Rename the source files to choose the name explicitly."
            )
        ]
        assert caught[0].filename == __file__
        names = {roi.name: roi for roi in layout}
        assert set(names) == {name, "Roi2"}
        assert names[name].counts.name == f"{prefix}_CellxGene.csv"

    def test_explicit_prefix_selects_and_repairs_its_roi(self, tmp_path: Path) -> None:
        _write_pairs(tmp_path, ("Roi 1", "Roi:1"))
        with pytest.warns(UserWarning, match="'Roi 1'"):
            layout = discover_layout(tmp_path, _selection(rois=("Roi 1",)))
        assert [roi.name for roi in layout] == ["Roi_1"]
        assert layout[0].cells.name == "Roi 1_CellCoordinates.csv"

    def test_valid_prefixes_do_not_warn(self, tmp_path: Path) -> None:
        _write_pairs(tmp_path, ("Roi-1.a", "Roi1"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert [roi.name for roi in discover_layout(tmp_path, _selection())] == ["Roi-1.a", "Roi1"]

    @pytest.mark.parametrize("prefixes", [("Roi 1", "Roi:1"), ("Roi1", "roi1"), ("Roi 1", "roi_1")])
    def test_names_equal_ignoring_case_fail(self, tmp_path: Path, prefixes: tuple[str, str]) -> None:
        _write_pairs(tmp_path, prefixes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(ReaderFormatError, match="equal ignoring case") as caught:
                discover_layout(tmp_path, _selection())
        assert caught.value.found == f"ROIs {sorted(prefixes)!r} whose element names are equal ignoring case"
        assert caught.value.context.artifact == "ROI layout"
