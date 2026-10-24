"""Deterministic MCMICRO WSI/TMA artifacts with explicit scientific oracles."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import tifffile

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class McmicroFixture:
    """Identify one format-faithful input tree and its intended pairing state."""

    root: Path
    mode: Literal["wsi", "tma"]
    samples: tuple[str, ...]
    methods: tuple[str, ...]
    compartment: str
    marker_names: tuple[str, ...]


def _write_image(path: Path, shape: tuple[int, int, int], offset: int, *, compressed: bool = False) -> None:
    """Write independent CYX integer pixel values and native strip metadata."""
    image = np.arange(prod(shape), dtype=np.uint16).reshape(shape) + offset
    tifffile.imwrite(
        path,
        image,
        metadata={"axes": "CYX"},
        photometric="minisblack",
        rowsperstrip=5,
        compression="deflate" if compressed else None,
    )


def _write_sample(  # noqa: PLR0913, RUF100 - Explicit sample, segmentation and channel roles define the fixture.
    root: Path,
    sample: str,
    shape: tuple[int, int, int],
    *,
    methods: tuple[str, ...],
    compartment: str,
    marker_names: tuple[str, ...],
    legacy_directory: bool,
) -> None:
    """Write each segmentation and its independently identified quantification table."""
    for index, method in enumerate(methods):
        directory = root / "segmentation" / (sample if legacy_directory else f"{method}-{sample}")
        directory.mkdir(parents=True)
        labels = np.zeros((1, shape[1], shape[2]), dtype=np.uint32)
        labels[:, 2:5, 2:6] = 1 + index * 10
        labels[:, -5:-2, -6:-2] = 2 + index * 10
        tifffile.imwrite(
            directory / f"{compartment}.ome.tif",
            labels,
            metadata={"axes": "CYX"},
            photometric="minisblack",
            rowsperstrip=5,
            compression="deflate" if index else None,
        )
        canonical = tuple(
            f"{name}_{marker_names[:i].count(name) + 1}" if marker_names.count(name) > 1 else name
            for i, name in enumerate(marker_names)
        )
        data = {name: [10.125 + i * 20 + index, 20.5 + i * 20 + index] for i, name in enumerate(canonical)}
        frame = pd.DataFrame(
            {
                "CellID": [1 + index * 10, 2 + index * 10],
                **data,
                "X_centroid": [3.5, shape[2] - 4.5],
                "Y_centroid": [3.0, shape[1] - 4.0],
                "area": [12.0, 12.0],
                "custom_note": ["first", "second"],
                f"{canonical[0]}_intensity_max": [100.0, 200.0],
            }
        )
        frame.to_csv(root / "quantification" / f"{sample}--{method}_{compartment}.csv", index=False)


def write_mcmicro_dataset(  # noqa: PLR0913, RUF100 - Independent format variants exercise existing cohort boundaries.
    path: Path,
    *,
    mode: Literal["wsi", "tma"] = "wsi",
    multiple_methods: bool = False,
    underscored_compartment: bool = False,
    duplicate_markers: bool = False,
    marker_only: bool = False,
    legacy_directory: bool = False,
) -> McmicroFixture:
    """Write registered, raw, illumination, segmentation, mask and table roles."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "qc").mkdir()
    (path / "qc" / "params.yml").write_text(f"workflow:\n  tma: {str(mode == 'tma').lower()}\n", encoding="utf-8")
    markers = ("DAPI", "CD3", "DAPI") if duplicate_markers else ("CD3", "DAPI")
    marker_frame = pd.DataFrame({"marker_name": markers, "cycle": [1] * len(markers)})
    if not marker_only:
        marker_frame["channel_number"] = np.arange(1, len(markers) + 1)
    marker_frame.to_csv(path / "markers.csv", index=False)
    (path / "registration").mkdir()
    (path / "quantification").mkdir()
    (path / "raw").mkdir()
    (path / "illumination").mkdir()
    _write_image(path / "registration" / "sample_part_7.ome.tif", (len(markers), 64, 80), 0)
    _write_image(path / "raw" / "cycle_01.ome.tif", (2, 16, 20), 3000)
    for suffix, value in (("ffp", 0.25), ("dfp", 0.5)):
        tifffile.imwrite(
            path / "illumination" / f"cycle_01-{suffix}.tif",
            np.full((2, 16, 20), value, dtype=np.float32),
            metadata={"axes": "CYX"},
            photometric="minisblack",
        )
    methods = ("method_a", "method_b") if multiple_methods else ("method_a",)
    compartment = "cell_mask_ring" if underscored_compartment else "cells"
    if mode == "wsi":
        _write_sample(
            path,
            "sample_part_7",
            (len(markers), 64, 80),
            methods=methods,
            compartment=compartment,
            marker_names=markers,
            legacy_directory=legacy_directory,
        )
        return McmicroFixture(path, mode, ("sample_part_7",), methods, compartment, markers)
    cores = path / "dearray"
    (cores / "masks").mkdir(parents=True)
    qc = path / "qc" / "coreograph"
    qc.mkdir()
    (qc / "centroidsY-X.txt").write_text("50 60\n2 4\n", encoding="utf-8")
    # This QC thumbnail intentionally differs from registered fluorescence.
    tifffile.imwrite(qc / "TMA_MAP.tif", np.full((6, 8, 3), 255, dtype=np.uint8), photometric="rgb")
    for sample, height, width in (("1", 30, 40), ("2", 21, 31)):
        shape = (len(markers), height, width)
        _write_image(cores / f"{sample}.ome.tif", shape, int(sample) * 1000, compressed=sample == "2")
        _write_sample(
            path,
            sample,
            shape,
            methods=methods,
            compartment=compartment,
            marker_names=markers,
            legacy_directory=legacy_directory,
        )
        mask = np.ones((height // 10, width // 10), dtype=np.uint8)
        mask[0, 0] = 0
        tifffile.imwrite(
            cores / "masks" / f"{sample}_mask.tif", mask, metadata={"axes": "YX"}, photometric="minisblack"
        )
    return McmicroFixture(path, mode, ("1", "2"), methods, compartment, markers)
