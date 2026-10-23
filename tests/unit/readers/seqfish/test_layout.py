"""Validate ordered product selection and deterministic seqFISH artifact ownership."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.seqfish._layout import Selection, discover_layout, make_selection

if TYPE_CHECKING:
    from pathlib import Path


def _selection(*, rois: tuple[str, ...] | None = None) -> Selection:
    """Select only required tabular products for isolated discovery tests."""
    return make_selection(
        include_images=False,
        include_labels=False,
        include_transcripts=False,
        include_shapes=False,
        shape_geometry="polygons",
        rois=rois,
    )


def _write_pairs(root: Path, names: tuple[str, ...] = ("Roi2", "Roi1")) -> None:
    """Write artifact placeholders; discovery owns paths rather than parsing."""
    for name in names:
        for suffix in ("CellxGene.csv", "CellCoordinates.csv"):
            (root / f"{name}_{suffix}").touch()


class TestMakeSelection:
    """Reject invalid public values and retain the caller's explicit ROI order."""

    def test_owns_borrowed_roi_order_and_shape_selection(self) -> None:
        rois = ["Roi2", "Roi1"]
        selection = make_selection(
            include_images=True,
            include_labels=False,
            include_transcripts=True,
            include_shapes=True,
            shape_geometry="both",
            rois=rois,
        )
        rois.reverse()
        assert selection == Selection(images=True, labels=False, transcripts=True, shapes="both", rois=("Roi2", "Roi1"))
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
                rois=None,
            )

    @pytest.mark.parametrize("geometry", ["polygon", "", "POLYGONS", 1, None, []])
    def test_geometry_must_be_supported(self, geometry: str | int | list[str] | None) -> None:
        with pytest.raises((ValueError, TypeError), match="shape_geometry"):
            make_selection(
                include_images=False,
                include_labels=False,
                include_transcripts=False,
                include_shapes=False,
                shape_geometry=geometry,  # type: ignore[bad-argument-type, arg-type]  # Validate untyped public choices even for excluded products.
                rois=None,
            )

    @pytest.mark.parametrize(
        "rois", ["Roi1", pytest.param(b"Roi1", id="bytes"), [], [""], ["."], [".."], ["a/b"], ["a\\b"], ["a\x00b"], [1]]
    )
    def test_invalid_roi_names_or_scalar(self, rois: str | bytes | list[str] | list[int]) -> None:
        with pytest.raises((TypeError, ValueError), match="rois"):
            make_selection(
                include_images=False,
                include_labels=False,
                include_transcripts=False,
                include_shapes=False,
                shape_geometry="polygons",
                rois=rois,  # type: ignore[bad-argument-type, arg-type]  # Exercise malformed caller-owned selections.
            )

    def test_duplicate_rois_fail(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            _selection(rois=("Roi1", "Roi1"))

    @pytest.mark.parametrize("rois", [{"Roi1"}, {"Roi1": 1}, 1])
    def test_unordered_or_nonsequence_roi_values_fail(self, rois: set[str] | dict[str, int] | int) -> None:
        with pytest.raises(TypeError, match="rois must be a sequence"):
            make_selection(
                include_images=False,
                include_labels=False,
                include_transcripts=False,
                include_shapes=False,
                shape_geometry="polygons",
                rois=rois,  # type: ignore[bad-argument-type, arg-type]  # Reject unordered values from untyped callers.
            )


class TestDiscoverLayout:
    """Resolve only selected products while enforcing paired artifacts and containment."""

    def test_sorted_inference_explicit_order_and_normalized_root(self, tmp_path: Path) -> None:
        _write_pairs(tmp_path)
        inferred = discover_layout(tmp_path / ".", _selection())
        assert inferred.root == tmp_path.resolve()
        assert [roi.name for roi in inferred.rois] == ["Roi1", "Roi2"]
        assert all(roi.pixels is None and roi.transcripts is None for roi in inferred.rois)
        explicit = discover_layout(str(tmp_path), _selection(rois=("Roi2", "Roi1")))
        assert [roi.name for roi in explicit.rois] == ["Roi2", "Roi1"]

    @pytest.mark.parametrize("kind", ["missing", "file"])
    def test_root_missing_or_wrong_kind(self, tmp_path: Path, kind: str) -> None:
        path = tmp_path / "dataset"
        if kind == "file":
            path.touch()
        with pytest.raises(FileNotFoundError if kind == "missing" else NotADirectoryError):
            discover_layout(path, _selection())

    @pytest.mark.parametrize("filename", [None, "_CellCoordinates.csv"])
    def test_empty_or_nameless_layout(self, tmp_path: Path, filename: str | None) -> None:
        if filename is not None:
            (tmp_path / filename).touch()
        with pytest.raises(ReaderFormatError, match="paired"):
            discover_layout(tmp_path, _selection())

    def test_unknown_requested_roi(self, tmp_path: Path) -> None:
        _write_pairs(tmp_path)
        with pytest.raises(ReaderFormatError, match="unknown ROIs"):
            discover_layout(tmp_path, _selection(rois=("Roi3",)))

    @pytest.mark.parametrize("suffix", ["CellCoordinates.csv", "CellxGene.csv"])
    @pytest.mark.parametrize("wrong_kind", [False, True])
    def test_missing_or_wrong_kind_required_pair(self, tmp_path: Path, suffix: str, *, wrong_kind: bool) -> None:
        _write_pairs(tmp_path, ("Roi1",))
        path = tmp_path / f"Roi1_{suffix}"
        path.unlink()
        if wrong_kind:
            path.mkdir()
        with pytest.raises(IsADirectoryError if wrong_kind else FileNotFoundError):
            discover_layout(tmp_path, _selection())

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
        expected = ["TranscriptList.csv"] if product == "transcripts" else []
        if product in {"images", "labels", "polygons", "both"}:
            expected.append("DAPI.tiff")
        if product == "labels":
            expected.append("Segmentation.tiff")
        if product in {"polygons", "both"}:
            expected.append("Boundaries.geojson")
        for suffix in expected:
            with pytest.raises(FileNotFoundError):
                discover_layout(tmp_path, selection)
            (tmp_path / f"Roi1_{suffix}").touch()
        layout = discover_layout(tmp_path, selection)
        roi = layout.rois[0]
        assert (roi.pixels is None) == (product in {"circles", "transcripts"})
        assert (roi.transcripts is not None) == (product == "transcripts")

    def test_explicit_subset_ignores_an_unselected_incomplete_roi(self, tmp_path: Path) -> None:
        _write_pairs(tmp_path, ("Roi1",))
        (tmp_path / "Roi2_CellxGene.csv").touch()
        assert [roi.name for roi in discover_layout(tmp_path, _selection(rois=("Roi1",))).rois] == ["Roi1"]

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
