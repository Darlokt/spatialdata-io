"""Explicit square population sources follow SAW 7 format Table 2-3 filenames."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_array
from spatialdata.testing import assert_spatial_data_objects_are_identical

from spatialdata_io import stereoseq
from spatialdata_io.readers._utils.errors import ReaderFormatError
from tests.integration.readers.stereoseq._fixtures import write_dataset, write_square_gef

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class TestStereoseq:
    """Disambiguate selected population qualifiers without weakening source identity."""

    @pytest.mark.parametrize(
        ("sample", "qualifier"),
        [("sample", "HE"), ("sample", "region.label"), ("sample", "region.raw.label"), ("chip.v2", "HE")],
    )
    def test_explicit_stain_and_label_square_sources(self, tmp_path: Path, sample: str, qualifier: str) -> None:
        path = write_square_gef(tmp_path / f"{sample}.{qualifier}.gef", bins=(1,))
        with h5py.File(path, "a") as handle:
            handle.attrs["sn"] = sample
        result = stereoseq(tmp_path, sample_id=sample, bin_sizes=(1,), square_counts_file=path.name)
        table = result.tables["bin1_table"]
        assert isinstance(table.X, csr_array)
        assert_array_equal(table.X.data, [6, 3])
        assert table.var_names.tolist() == ["z", "a"]
        assert table.obs["instance_id"].tolist() == [0, 1]

    def test_qualified_filename_cannot_override_conflicting_source_sample(self, tmp_path: Path) -> None:
        path = write_square_gef(tmp_path / "sample.HE.gef", bins=(1,))
        with h5py.File(path, "a") as handle:
            handle.attrs["sn"] = "other"
        with pytest.raises(ReaderFormatError, match="inconsistent source sample"):
            stereoseq(tmp_path, sample_id="sample", bin_sizes=(1,), square_counts_file=path)


class TestStereoseqPathPolicy:
    """Discovered products follow the dataset root's symlink policy."""

    def test_out_of_root_symlinks_require_the_relaxed_policy(
        self, tmp_path: Path, link_outside_root: Callable[[Path], Path]
    ) -> None:
        fixture = write_dataset(tmp_path, family="outs", products=("square", "cells"))
        expected = stereoseq(fixture.root, bin_sizes=(1,), cell_bins="unadjusted")
        link_outside_root(fixture.square)
        link_outside_root(fixture.cells)

        with pytest.raises(ReaderFormatError, match="path_policy='relaxed'"):
            stereoseq(fixture.root, bin_sizes=(1,), cell_bins="unadjusted")

        relaxed = stereoseq(fixture.root, bin_sizes=(1,), cell_bins="unadjusted", path_policy="relaxed")
        assert_spatial_data_objects_are_identical(relaxed, expected)
