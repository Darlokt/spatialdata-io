"""Observed Stereo-CITE protein metadata fails before any expression materializes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest

from spatialdata_io import stereoseq
from spatialdata_io.readers._utils.errors import ReaderFormatError

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestStereoCiteProtein:
    """An offline metadata-derived fixture does not imply paired modality support."""

    def test_observed_protein_header_is_rejected_before_dataset_reads(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        # Observed header/schema: https://zenodo.org/records/12595241/files/A03994E6.protein.tissue.gef
        # Unallocated datasets reproduce metadata, not downloaded protein values.
        source = tmp_path / "A03994E6.protein.tissue.gef"
        with h5py.File(source, "w") as handle:
            handle.attrs["version"] = np.array([2], dtype=np.uint32)
            handle.attrs["bin_type"] = np.array([b"Bin"], dtype="S32")
            handle.attrs["omics"] = np.array([b"Proteomics"], dtype="S32")
            group = handle.create_group("geneExp/bin1")
            group.create_dataset("gene", shape=(31,), dtype=[("gene", "S64"), ("offset", "<u4"), ("count", "<u4")])
            expression = group.create_dataset(
                "expression", shape=(356544927,), dtype=[("x", "<i4"), ("y", "<i4"), ("count", "u1")]
            )
            for name, value in {"minX": 0, "minY": 0, "maxX": 20579, "maxY": 23521}.items():
                expression.attrs[name] = np.array([value], dtype=np.int32)
            expression.attrs["resolution"] = np.array([500], dtype=np.uint32)

        reads = mocker.patch.object(h5py.Dataset, "__getitem__", side_effect=AssertionError("Unexpected dataset read"))
        with pytest.raises(ReaderFormatError, match=r"omics.*Proteomics"):
            stereoseq(tmp_path, bin_sizes=(1,), square_counts_file=source.name)
        reads.assert_not_called()
