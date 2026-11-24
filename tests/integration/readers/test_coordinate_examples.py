"""Execute the coordinate-query examples against independent vendor landmarks and counts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_array

from spatialdata_io import xenium
from tests.integration.readers.visium_hd._fixtures import write_bundle
from tests.integration.readers.xenium import _fixtures as fx


def _run_example(reader: str, heading: str, namespace: dict[str, object]) -> None:
    guide = Path(__file__).parents[3] / "docs" / "readers" / f"{reader}.md"
    section = guide.read_text(encoding="utf-8").split(f"### {heading}\n", 1)[1]
    snippet = section.split("```python\n", 1)[1].split("```", 1)[0]
    exec(compile(snippet, str(guide), "exec"), namespace)  # noqa: S102 - Repository-owned executable example.


def test_xenium_he_centroid_example(tmp_path: Path) -> None:
    bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_2_0, fx.Options(aligned=("he",)))
    namespace: dict[str, object] = {"sdata": xenium(bundle.root)}
    _run_example("xenium", "Locate a cell in the H&E image", namespace)

    # The first cell is centred at morphology (6, 5). The fixture alignment
    # rotates and scales H&E (6, 8) to (-4, 3), then translates by (10, 2).
    assert namespace["cell_id"] == bundle.barcodes[0]
    assert namespace["column"] == pytest.approx(6.0)
    assert namespace["row"] == pytest.approx(8.0)


def test_visium_hd_qupath_bin_query_example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_bundle(tmp_path / "outs", sizes=(2, 8), segmented=False, fullres="root")
    # These two annotations overlap at the edge of the first 8 µm bin.
    # Its centre in the source image is near (10, 74), outside both polygons.
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[[6, y], [8, y], [8, 81], [6, 81], [6, y]]]},
            "properties": {"objectType": "annotation"},
        }
        for y in (78, 79)
    ]
    (tmp_path / "annotations.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {}
    _run_example("visium_hd", "Query bins with a QuPath annotation", namespace)

    table = namespace["selected"]
    assert isinstance(table, AnnData)
    assert table.obs_names.tolist() == ["s_008um_00000_00000-1"]
    assert isinstance(table.X, csr_array)
    np.testing.assert_array_equal(table.X.toarray(), [[3, 5, 5]])
    error = namespace["max_residual_um"]
    assert isinstance(error, float)
    assert error > 0
