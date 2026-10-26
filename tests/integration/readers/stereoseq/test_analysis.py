"""Exact clustering identities survive complete SpatialData assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from scipy.sparse import sparray

from spatialdata_io import stereoseq
from tests.integration.readers.stereoseq._fixtures import write_analysis

if TYPE_CHECKING:
    from pathlib import Path


class TestStereoseqAnalysisResolution:
    """Large and precise decimals remain exact and valid element names."""

    @pytest.mark.parametrize("resolution", ["1e70", "1e-80", "0.123456789012345678901234567890123"])
    def test_extreme_exact_resolution_names_compose_in_public_spatialdata(
        self, tmp_path: Path, resolution: str
    ) -> None:
        path = write_analysis(tmp_path / f"sample.cellbin_{resolution}.h5ad")
        result = stereoseq(tmp_path, analysis_files=(path,))
        assert len(result.tables) == len(result.points) == 1
        table = next(iter(result.tables.values()))
        assert table.obs_names.tolist() == [str((10 << 32) | 20), str((30 << 32) | 40)]
        assert isinstance(table.X, sparray)
