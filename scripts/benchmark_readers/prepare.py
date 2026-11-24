"""Prepare the shared synthetic and DBiT inputs for the reader comparison."""

from __future__ import annotations

import argparse
import shutil
from inspect import unwrap
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from tests.integration.experimental.iss.conftest import write_iss_dataset
from tests.integration.readers.codex._fixtures import cell_frame, write_csv
from tests.integration.readers.cosmx._fixtures import write_cosmx_dataset
from tests.integration.readers.curio._fixtures import write_curio
from tests.integration.readers.dbit.test_dataset import _prepare_layout
from tests.integration.readers.mcmicro._fixtures import write_mcmicro_dataset
from tests.integration.readers.merscope._fixtures import write_merscope_dataset
from tests.integration.readers.steinbock.conftest import steinbock_dataset
from tests.integration.readers.stereoseq._fixtures import write_dataset


def prepare(data: Path, directory: Path) -> None:
    """Create fixtures once in a new directory. Both source revisions read the same files."""
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("generic", "codex", "curio", "steinbock", "iss"):
        (directory / name).mkdir()

    # Measure the cost of the optional panel using otherwise identical vendor files.
    shutil.copytree(
        data / "Xenium_Prime_Mouse_Ileum_tiny_outs",
        directory / "xenium-no-panel",
        ignore=shutil.ignore_patterns("gene_panel.json"),
    )
    raster = np.random.default_rng(730).integers(0, 4096, (4, 2048, 2048), dtype=np.uint16)
    tifffile.imwrite(
        directory / "generic" / "tiles.tif",
        raster,
        photometric="minisblack",
        metadata={"axes": "CYX"},
        compression="deflate",
        tile=(256, 256),
    )

    cells = pd.concat([cell_frame()] * 10_000, ignore_index=True)
    cells["cell_id"] = np.arange(len(cells)) + 1
    write_csv(directory / "codex" / "cells.csv", cells)
    write_curio(directory / "curio")
    write_cosmx_dataset(directory / "cosmx")
    mcmicro_root = directory / "mcmicro"
    write_mcmicro_dataset(mcmicro_root)
    # Only upstream returns raw and illumination images. Keep the common WSI products.
    for extra in ("raw", "illumination"):
        shutil.rmtree(mcmicro_root / extra)
    write_iss_dataset(directory / "iss")
    unwrap(steinbock_dataset)(directory / "steinbock")
    write_dataset(directory / "stereoseq", bins=(1,))
    _prepare_layout(data / "dbit_mouse_e10", directory / "dbit")

    write_merscope_dataset(directory / "merscope")
    region = directory / "merscope" / "slide" / "region"
    rows, features = 8192, 128
    cell_ids = np.arange(1_000_001, 1_000_001 + rows)
    expression = np.random.default_rng(730).binomial(1, 0.03, (rows, features)).astype(np.uint8)
    pd.DataFrame(
        expression, index=pd.Index(cell_ids, name="cell"), columns=[f"Gene{i:03}" for i in range(features)]
    ).to_csv(region / "cell_by_gene.csv")
    pd.DataFrame(
        {"center_x": cell_ids % 128 + 0.5, "center_y": np.arange(rows) // 128 + 0.5, "volume": np.full(rows, 3.0)},
        index=pd.Index(cell_ids, name="EntityID"),
    ).to_csv(region / "cell_metadata.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    prepare(args.data, args.directory)
