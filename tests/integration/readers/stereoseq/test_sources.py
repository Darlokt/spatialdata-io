"""Explicit square population sources follow SAW 7 format Table 2-3 filenames."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_array

from spatialdata_io import stereoseq
from spatialdata_io.readers._utils.errors import ReaderFormatError
from tests.integration.readers.stereoseq._fixtures import write_square_gef

if TYPE_CHECKING:
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
