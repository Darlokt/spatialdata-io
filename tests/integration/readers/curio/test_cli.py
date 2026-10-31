"""Convert an actual five-artifact export through the public command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import get_table_keys

from spatialdata_io._cli import cli
from tests.integration.readers.curio._fixtures import write_curio

if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import CliRunner


@pytest.mark.cli
class TestCurioCommand:
    def test_writes_complete_export(self, tmp_path: Path, runner: CliRunner) -> None:
        source = tmp_path / "input"
        source.mkdir()
        write_curio(source)
        output = tmp_path / "output.zarr"

        result = runner.invoke(cli, ["curio", "--input", str(source), "--output", str(output)])

        assert result.exit_code == 0, result.output
        restored = SpatialData.read(output)
        assert list(restored.tables) == ["table"]
        assert list(restored.shapes) == ["cells"]
        table = restored.tables["table"]
        assert isinstance(table.X, sparse.csr_matrix)
        np.testing.assert_array_equal(table.X.data, [1, 2, 3, 4, 5])
        assert table.obs["cluster"].tolist() == ["1", "0", "1"]
        assert table.obs["cluster"].cat.categories.tolist() == ["1", "0"]
        assert table.uns["Sequencing"] == {"Reads, total": "100"}
        assert get_table_keys(table) == ("cells", "region", "instance_id")
        assert table.obs_names.equals(restored.shapes["cells"].index)
