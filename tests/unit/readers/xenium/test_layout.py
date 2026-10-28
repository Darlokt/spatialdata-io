"""One discovery resolves only the artifacts that the selected products need."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.xenium._layout import (
    AnalysisArchive,
    AnalysisDirectory,
    H5Counts,
    MtxArchive,
    MtxDirectory,
    Selection,
    discover_layout,
)
from tests.integration.readers.xenium import _fixtures as fx

if TYPE_CHECKING:
    from pathlib import Path

NOTHING = Selection(
    table=False,
    table_target=None,
    cell_labels=False,
    nucleus_labels=False,
    cell_polygons=False,
    cell_circles=False,
    nucleus_polygons=False,
    transcripts=False,
    morphology_focus=False,
    morphology_mip=False,
    aligned_images=False,
    analysis=False,
)
EVERYTHING = Selection(
    table=True,
    table_target="cell_labels",
    cell_labels=True,
    nucleus_labels=True,
    cell_polygons=True,
    cell_circles=True,
    nucleus_polygons=True,
    transcripts=True,
    morphology_focus=True,
    morphology_mip=False,
    aligned_images=True,
    analysis=False,
)


def _experiment(root: Path, **changes: object) -> None:
    path = root / "experiment.xenium"
    path.write_text(json.dumps(json.loads(path.read_text()) | changes))


class TestDiscoverLayout:
    """One discovery resolves every selected artifact of a family deterministically."""

    @pytest.mark.parametrize(
        ("profile", "family", "version"),
        [
            (fx.XOA_1_0, "integer_ids", None),
            (fx.XOA_1_3, "hex_ids", "2.0"),
            (fx.XOA_1_9, "hex_ids", "3.0.0"),
            (fx.XOA_2_0, "polygon_sets", "4.0.0"),
            (fx.FORMAT_6_1, "polygon_sets", "6.1.0"),
        ],
    )
    def test_families_and_complete_selection(
        self, tmp_path: Path, profile: fx.Profile, family: str, version: str | None
    ) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, profile)

        layout = discover_layout(tmp_path, replace(EVERYTHING, morphology_mip=profile.images == "single"))

        assert layout.experiment.family == family
        assert (None if layout.experiment.version is None else str(layout.experiment.version)) == version
        assert layout.cells_store == tmp_path / "cells.zarr.zip"
        assert layout.cells is not None
        assert layout.cells.kind == "parquet"
        assert layout.transcripts is not None
        assert layout.transcripts.file.name == "transcripts.parquet"
        assert isinstance(layout.counts, H5Counts)
        assert layout.focus is not None
        assert layout.focus.entry == tmp_path / bundle.focus_entry
        assert layout.metrics == tmp_path / "metrics_summary.csv"
        assert layout.gene_panel == tmp_path / "gene_panel.json"
        assert layout.protein_panel is None
        assert layout.aligned == ()

    def test_unselected_products_are_never_touched(self, tmp_path: Path) -> None:
        fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(cells_store=False, images=False, transcripts=None))
        for name in (
            "cells.parquet",
            "cell_boundaries.parquet",
            "nucleus_boundaries.parquet",
            "cell_feature_matrix.h5",
        ):
            (tmp_path / name).unlink()

        layout = discover_layout(tmp_path, NOTHING)

        assert (layout.cells_store, layout.cells, layout.counts, layout.transcripts, layout.focus) == (
            None,
            None,
            None,
            None,
            None,
        )

    def test_store_required_for_labels_and_optional_otherwise(self, tmp_path: Path) -> None:
        fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(cells_store=False))
        tables_only = replace(NOTHING, table=True, cell_polygons=True)

        assert discover_layout(tmp_path, tables_only).cells_store is None
        with pytest.raises(FileNotFoundError, match=r"cells.zarr.zip \(labels"):
            discover_layout(tmp_path, replace(NOTHING, nucleus_labels=True))

    def test_csv_twins(self, tmp_path: Path) -> None:
        fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(tables="csv_gz"))
        selection = replace(NOTHING, table=True, cell_polygons=True, nucleus_polygons=True)

        layout = discover_layout(tmp_path, selection)

        assert layout.cells is not None
        assert (layout.cells.file.name, layout.cells.kind) == ("cells.csv.gz", "csv_gz")
        assert layout.nucleus_boundaries is not None
        assert layout.nucleus_boundaries.kind == "csv_gz"

        # A parquet file always wins over its twins.
        fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)
        cells = discover_layout(tmp_path, selection).cells
        assert cells is not None
        assert cells.kind == "parquet"

    def test_ambiguous_and_missing_twins(self, tmp_path: Path) -> None:
        fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(tables="csv_gz"))
        (tmp_path / "cells.csv").write_bytes(b"cell_id\n")

        with pytest.raises(ReaderFormatError, match=r"both cells.csv and cells.csv.gz"):
            discover_layout(tmp_path, replace(NOTHING, table=True))

        (tmp_path / "cells.csv").unlink()
        (tmp_path / "cells.csv.gz").unlink()
        with pytest.raises(FileNotFoundError, match=r"cells\.parquet"):
            discover_layout(tmp_path, replace(NOTHING, table=True))

    def test_transcript_sources(self, tmp_path: Path) -> None:
        fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(transcripts="csv"))
        selection = replace(NOTHING, transcripts=True)

        layout = discover_layout(tmp_path, selection)
        assert layout.transcripts is not None
        assert layout.transcripts.kind == "csv"

        (tmp_path / "transcripts.csv").rename(tmp_path / "transcripts.csv.gz")
        with pytest.raises(ReaderFormatError, match="single-stream gzip transcript CSV"):
            discover_layout(tmp_path, selection)

        (tmp_path / "transcripts.csv.gz").unlink()
        with pytest.raises(FileNotFoundError, match=r"transcripts\.parquet"):
            discover_layout(tmp_path, selection)

    def test_count_sources(self, tmp_path: Path) -> None:
        fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(counts=("h5", "mtx", "mtx_archive")))
        selection = replace(NOTHING, table=True)

        assert isinstance(discover_layout(tmp_path, selection).counts, H5Counts)
        (tmp_path / "cell_feature_matrix.h5").unlink()
        assert isinstance(discover_layout(tmp_path, selection).counts, MtxDirectory)
        (tmp_path / "cell_feature_matrix" / "matrix.mtx.gz").unlink()
        with pytest.raises(FileNotFoundError, match=r"matrix\.mtx\.gz"):
            discover_layout(tmp_path, selection)
        for path in (tmp_path / "cell_feature_matrix").iterdir():
            path.unlink()
        (tmp_path / "cell_feature_matrix").rmdir()
        assert isinstance(discover_layout(tmp_path, selection).counts, MtxArchive)
        (tmp_path / "cell_feature_matrix.tar.gz").unlink()
        with pytest.raises(FileNotFoundError, match=r"cell_feature_matrix\.h5"):
            discover_layout(tmp_path, selection)

    def test_focus_set_lists_every_ome_tiff_beside_the_entry(self, tmp_path: Path) -> None:
        fx.write_xenium_bundle(tmp_path, fx.XOA_2_0)
        (tmp_path / "morphology_focus" / "._morphology_focus_0000.ome.tif").write_bytes(b"AppleDouble")
        (tmp_path / "morphology_focus" / "notes.txt").write_text("not an image")

        layout = discover_layout(tmp_path, replace(NOTHING, morphology_focus=True))

        assert layout.focus is not None
        assert [path.name for path in layout.focus.files] == [f"morphology_focus_{i:04d}.ome.tif" for i in range(4)]

    def test_focus_set_members_must_be_files(self, tmp_path: Path) -> None:
        fx.write_xenium_bundle(tmp_path, fx.XOA_2_0)
        (tmp_path / "morphology_focus" / "morphology_focus_0004.ome.tif").mkdir()

        with pytest.raises(IsADirectoryError, match=r"morphology_focus_0004\.ome\.tif"):
            discover_layout(tmp_path, replace(NOTHING, morphology_focus=True))

    def test_image_entries(self, tmp_path: Path) -> None:
        fx.write_xenium_bundle(tmp_path, fx.XOA_2_0)

        with pytest.raises(ReaderFormatError, match=r"XOA 2\.0\+ bundles have no maximum-intensity projection"):
            discover_layout(tmp_path, replace(NOTHING, morphology_mip=True))

        _experiment(tmp_path, images={})
        with pytest.raises(ReaderFormatError, match=r"images\.morphology_focus_filepath"):
            discover_layout(tmp_path, replace(NOTHING, morphology_focus=True))

        _experiment(tmp_path, images={"morphology_focus_filepath": "../outside.ome.tif"})
        with pytest.raises(ReaderFormatError, match="contained by dataset root"):
            discover_layout(tmp_path, replace(NOTHING, morphology_focus=True))

    def test_mip_of_single_file_bundles(self, tmp_path: Path) -> None:
        fx.write_xenium_bundle(tmp_path, fx.XOA_1_9)

        layout = discover_layout(tmp_path, replace(NOTHING, morphology_mip=True))

        assert layout.mip == tmp_path / "morphology_mip.ome.tif"

    def test_aligned_images(self, tmp_path: Path) -> None:
        fx.write_xenium_bundle(tmp_path, fx.XOA_2_0, fx.Options(aligned=("he", "if")))
        selection = replace(NOTHING, aligned_images=True)

        layout = discover_layout(tmp_path, selection)

        assert [(image.name, image.file.name) for image in layout.aligned] == [
            ("he_image", "synthetic_he_image.ome.tif"),
            ("if_image", "synthetic_if_image.ome.tif"),
        ]
        assert layout.aligned[0].alignment == tmp_path / "synthetic_he_imagealignment.csv"
        assert layout.aligned[1].alignment is None

        (tmp_path / "other_he_image.ome.tif").write_bytes((tmp_path / "synthetic_he_image.ome.tif").read_bytes())
        with pytest.raises(ReaderFormatError, match="2 HE images"):
            discover_layout(tmp_path, selection)

    def test_analysis_sources(self, tmp_path: Path) -> None:
        fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(analysis="archive"))
        selection = replace(NOTHING, table=True, analysis=True)

        assert isinstance(discover_layout(tmp_path, selection).analysis, AnalysisArchive)
        fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(analysis="directory"))
        assert isinstance(discover_layout(tmp_path, selection).analysis, AnalysisDirectory)

    def test_missing_root_and_experiment(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discover_layout(tmp_path / "missing", NOTHING)
        with pytest.raises(FileNotFoundError, match=r"experiment\.xenium"):
            discover_layout(tmp_path, NOTHING)
        (tmp_path / "file").write_text("")
        with pytest.raises(NotADirectoryError):
            discover_layout(tmp_path / "file", NOTHING)
