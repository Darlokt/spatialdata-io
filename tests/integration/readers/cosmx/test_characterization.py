"""Preserve valid legacy CosMx values while testing explicit scientific corrections."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import sparse
from spatialdata.transformations import BaseTransformation, get_transformation

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.cosmx import cosmx
from tests.integration.readers.cosmx._fixtures import write_cosmx_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from xarray import DataArray, DataTree

_AXES = ("x", "y")


def _frames(element: DataArray | DataTree) -> dict[str, np.ndarray]:
    frames = get_transformation(element, get_all=True)
    assert isinstance(frames, dict)
    return {
        name: transform.to_affine_matrix(input_axes=_AXES, output_axes=_AXES)
        for name, transform in frames.items()
        if isinstance(transform, BaseTransformation)
    }


class TestCosmxCharacterization:
    """Values the legacy reader produced correctly on this fixture remain exactly unchanged."""

    def test_counts_order_and_centers(self, tmp_path: Path) -> None:
        result = cosmx(write_cosmx_dataset(tmp_path).root)
        table = result.tables["table"]
        assert isinstance(table.X, sparse.csr_array)
        np.testing.assert_array_equal(table.X.toarray(), [[1, 0, 2, 0], [0, 3, 0, 0], [4, 0, 0, 0], [0, 0, 7, 1]])
        assert table.var_names.tolist() == ["GeneB", "HLA-A", "GeneA", "NegPrb1"]
        np.testing.assert_array_equal(table.obsm["spatial"], [[2, 5], [6, 2], [2, 5], [6, 2]])
        np.testing.assert_array_equal(
            table.obsm["global"], [[1002, 5005.5], [1006, 5002.5], [3002.25, 7005], [3006.25, 7002]]
        )

    def test_transcript_values(self, tmp_path: Path) -> None:
        points = cosmx(write_cosmx_dataset(tmp_path).root).points["Run_1_transcripts"].compute()
        fov = points[points["fov"] == 2]
        np.testing.assert_array_equal(fov[["x_local", "y_local"]].to_numpy(), [[2, 5.125], [2.25, 5.125], [6.5, 2.125]])
        np.testing.assert_array_equal(
            fov[["x", "y"]].to_numpy(), [[1002, 5005.625], [1002.25, 5005.625], [1006.5, 5002.625]]
        )
        assert fov["target"].tolist() == ["GeneA", "HLA-A", "NegPrb1"]
        assert fov["CellComp"].tolist() == ["Nuclear", "0", "Nuclear"]


class TestCosmxCorrections:
    """Each legacy defect is replaced by an independently specified contract."""

    def test_translation_replaces_a_rank_deficient_affine_fit(self, tmp_path: Path) -> None:
        # Two cells per FOV made the legacy six-parameter fit return a distorted affine.
        result = cosmx(write_cosmx_dataset(tmp_path).root)
        frames = _frames(result.images["2_image"])
        np.testing.assert_array_equal(frames["global"], [[1, 0, 999.5], [0, -1, 5007.0], [0, 0, 1]])

    def test_source_pixels_are_unchanged_and_redundant_frames_are_gone(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path)
        result = cosmx(fixture.root)
        labels = result.labels["2_labels"].data.compute()
        assert np.argwhere(labels).tolist() == [[1, 2], [4, 6]]
        assert set(_frames(result.images["2_image"])) == {"2", "global"}
        assert set(_frames(result.labels["2_labels"])) == {"2", "global"}

    def test_one_global_points_element_with_reader_owned_names(self, tmp_path: Path) -> None:
        result = cosmx(write_cosmx_dataset(tmp_path).root)
        assert list(result.points) == ["Run_1_transcripts"]
        table = result.tables["table"]
        assert table.obs_names.tolist() == ["0", "1", "2", "3"]
        assert "region" in table.obs
        assert "fov_labels" not in table.obs

    def test_population_mismatch_fails_instead_of_silent_intersection(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path)
        counts = fixture.root / "Run_1_exprMat_file.csv"
        counts.write_text(counts.read_text().replace("\n2,2,", "\n2,9,"), encoding="utf-8")
        with pytest.raises(ReaderFormatError, match="identical positive count and metadata populations"):
            cosmx(fixture.root)

    def test_reading_writes_nothing_to_standard_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cosmx(write_cosmx_dataset(tmp_path).root).points["Run_1_transcripts"].compute()
        assert capsys.readouterr().out == ""
