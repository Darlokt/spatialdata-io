"""Measure one public reader call in the source tree selected through PYTHONPATH."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import resource
import sys
import warnings
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

CASES = (
    "visium",
    "visium-write",
    "visium_hd",
    "dbit",
    "generic-crop",
    "generic-full",
    "xenium",
    "xenium-breast",
    "xenium-no-panel",
    "xenium-no-masks-write",
    "xenium-write",
    "seqfish",
    "macsima",
    "merscope",
    "codex",
    "cosmx",
    "mcmicro",
    "steinbock",
    "curio",
    "iss",
    "stereoseq",
)


def workload(case: str, revision: str, data: Path, fixtures: Path) -> tuple[Path, dict[str, object]]:
    """Select equivalent public options on the two APIs wherever both readers support the product."""
    reader = case.split("-", maxsplit=1)[0]
    main = revision == "main"
    source = fixtures / reader
    if reader == "generic":
        return source / "tiles.tif", {
            "data_axes": "cyx" if main else "CYX",
            "coordinate_system": "source_pixels",
            "chunks": (1, 256, 256),
        }
    public: dict[str, tuple[Path, dict[str, object]]] = {
        "visium": (data / "V1_Adult_Mouse_Brain", {"dataset_id": "sample"}),
        "macsima": (data / "OMAP10_small", {"multiscale": False} if main else {}),
    }
    if reader in public:
        return public[reader]
    if reader == "xenium":
        directory, options = _xenium_workload(case, main=main, data=data)
        if case == "xenium-no-panel":
            directory = fixtures / case
        return directory, options
    if reader == "visium_hd":
        return data / "Visium_HD_Tiny_3prime_Dataset_outs", (
            {
                "dataset_id": "sample",
                "load_segmentations_only": False,
                "bin_size": [8],
                "image_models_kwargs": {"scale_factors": None},
            }
            if main
            else {"bin_sizes": (8,), "include_cell_boundaries": True, "include_nucleus_boundaries": False}
        )
    if reader == "seqfish":
        flags = (
            ("load_images", "load_labels", "load_points", "load_shapes")
            if main
            else ("include_images", "include_labels", "include_transcripts", "include_shapes")
        )
        return data / "seqfish-2-test-dataset" / "instrument 2 official", dict.fromkeys(flags, False)
    return _fixture_workload(reader, source, main=main)


def _fixture_workload(reader: str, source: Path, *, main: bool) -> tuple[Path, dict[str, object]]:
    if reader == "merscope":
        flags = (
            ("cells_boundaries", "transcripts", "mosaic_images")
            if main
            else ("include_cell_boundaries", "include_transcripts", "include_images")
        )
        return source / "slide" / "region", dict.fromkeys(flags, False)
    if reader == "iss":
        return source, (
            {
                "raw_relative_path": "raw.tiff",
                "labels_relative_path": "labels.tiff",
                "h5ad_relative_path": "cells.h5ad",
                "instance_key": "segmentation_id",
                "multiscale_image": False,
                "multiscale_labels": False,
            }
            if main
            else {
                "raw_image_file": "raw.tiff",
                "labels_file": "labels.tiff",
                "table_file": "cells.h5ad",
                "instance_key": "segmentation_id",
            }
        )
    options: dict[str, dict[str, object]] = {
        "codex": {"fcs": False} if main else {},
        "cosmx": {"dataset_id": "Run_1", "transcripts" if main else "include_transcripts": False},
        "steinbock": {"labels_kind": "ilastik"} if main else {"masks_dir": "masks_ilastik"},
        "stereoseq": {"dataset_id": "sample"} if main else {"sample_id": "sample", "bin_sizes": (1,)},
    }
    return source, options.get(reader, {})


def _xenium_workload(case: str, *, main: bool, data: Path) -> tuple[Path, dict[str, object]]:
    source = data / (
        "Xenium_V1_human_Breast_2fov_outs" if case == "xenium-breast" else "Xenium_Prime_Mouse_Ileum_tiny_outs"
    )
    if case.endswith("-write"):
        options: dict[str, object] = (
            {
                "morphology_mip": False,
                "image_models_kwargs": {"scale_factors": None, "chunks": (1, 1024, 1024)},
                "labels_models_kwargs": {"scale_factors": None, "chunks": (1024, 1024)},
                "n_jobs": 1,
            }
            if main
            else {}
        )
        if case == "xenium-no-masks-write":
            options.update(
                {"cells_labels": False, "nucleus_labels": False}
                if main
                else {"include_cell_labels": False, "include_nucleus_labels": False, "table_target": "cell_boundaries"}
            )
        return source, options
    if main:
        return source, dict.fromkeys(
            (
                "cells_boundaries",
                "nucleus_boundaries",
                "cells_labels",
                "nucleus_labels",
                "transcripts",
                "morphology_focus",
                "morphology_mip",
                "aligned_images",
            ),
            False,
        )
    return source, {
        "table_target": None,
        **dict.fromkeys(
            (
                "include_cell_labels",
                "include_nucleus_labels",
                "include_cell_boundaries",
                "include_nucleus_boundaries",
                "include_transcripts",
                "include_morphology_focus",
                "include_aligned_images",
            ),
            False,
        ),
    }


def digest(value: object) -> str:
    """Hash a contiguous representation after the measured operation."""
    import numpy as np  # noqa: PLC0415 -- keep imports out of the startup baseline

    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def name_digest(names: Iterable[object]) -> str:
    """Include the spelling and order of row and feature names."""
    return hashlib.sha256(json.dumps(list(map(str, names)), ensure_ascii=False).encode()).hexdigest()


def measure(
    case: str,
    source: Path,
    options: dict[str, object],
    *,
    profile_path: Path | None = None,
    first_import_start: float | None = None,
) -> dict[str, object]:
    """Return wall time, peak process RSS on Linux and fingerprints of measured products."""
    import dask  # noqa: PLC0415 -- keep the worker importable without scientific dependencies
    from scipy.sparse import csr_array  # noqa: PLC0415 -- same startup measurement

    reader = case.split("-", maxsplit=1)[0]
    module = import_module("spatialdata_io.experimental" if reader == "iss" else "spatialdata_io")
    read = getattr(module, "image" if reader == "generic" else reader)
    if reader == "codex":
        # Main opens the discovered CSV relative to the working directory.
        os.chdir(source)
    with TemporaryDirectory() as temp, dask.config.set(scheduler="synchronous"):
        profile = cProfile.Profile() if profile_path is not None else None
        if profile is not None:
            profile.enable()
        start = perf_counter()
        result = read(source, **options)
        read_seconds = perf_counter() - start
        pixels = None
        channels = None
        if reader == "generic":
            selection = result.data[0, 768:1024, 768:1024] if case == "generic-crop" else result.data
            pixels = selection.compute()
        if case.endswith("-write"):
            result.write(Path(temp) / "out.zarr")
        if reader == "macsima":
            image = next(iter(result.images.values()))
            pixels = image.data.compute()
            channels = image.coords["c"]
        finished = perf_counter()
        elapsed = finished - start
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

        if profile is not None and profile_path is not None:
            profile.disable()
            profile.dump_stats(str(profile_path))

        products: list[dict[str, object]] | dict[str, object]
        if pixels is not None:
            products = {"pixels": digest(pixels), "shape": list(pixels.shape), "dtype": str(pixels.dtype)}
            if channels is not None:
                products["channels"] = list(map(str, channels.values))
        else:
            products = []
            for name, table in sorted(result.tables.items()):
                matrix = csr_array(table.X, copy=True)
                matrix.sum_duplicates()
                matrix.eliminate_zeros()
                matrix.sort_indices()
                products.append(
                    {
                        "name": name,
                        "shape": list(matrix.shape),
                        "nnz": matrix.nnz,
                        "values": digest(matrix.data.astype("float64")),
                        "indices": digest(matrix.indices.astype("int64")),
                        "indptr": digest(matrix.indptr.astype("int64")),
                        "obs": name_digest(table.obs_names),
                        "var": name_digest(table.var_names),
                    }
                )
        elements = (
            {}
            if reader == "generic"
            else {kind: list(getattr(result, kind)) for kind in ("tables", "labels", "points", "images", "shapes")}
        )
    startup = {} if first_import_start is None else {"first_call_s": finished - first_import_start}
    return {
        **startup,
        "read_s": read_seconds,
        "total_s": elapsed,
        "rss_mib": peak,
        "products": products,
        "elements": elements,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", choices=("main", "branch"))
    parser.add_argument("case", choices=CASES)
    parser.add_argument("data", type=Path)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--first-call", action="store_true")
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    expected_source = Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]).resolve()
    # Start before loading any scientific library in the first-call measurement.
    public_import = None
    first_import_start = perf_counter() if args.first_call else None
    if first_import_start is not None:
        reader_name = args.case.split("-", maxsplit=1)[0]
        namespace = import_module("spatialdata_io.experimental" if reader_name == "iss" else "spatialdata_io")
        getattr(namespace, "image" if reader_name == "generic" else reader_name)
        public_import = perf_counter() - first_import_start
    else:
        for dependency in ("dask", "numpy", "pyarrow", "scipy.sparse"):
            import_module(dependency)

    import pyarrow as pa

    import spatialdata_io

    if not Path(spatialdata_io.__file__).resolve().is_relative_to(expected_source):
        parser.error("The imported package is outside the first PYTHONPATH directory")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    path, options = workload(args.case, args.revision, args.data.resolve(), args.fixtures.resolve())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if args.warmup:
            measure(args.case, path, options)
        record = measure(args.case, path, options, profile_path=args.profile, first_import_start=first_import_start)
        if public_import is not None:
            record["public_import_s"] = public_import
    sys.stdout.write("METRIC " + json.dumps(record) + "\n")
