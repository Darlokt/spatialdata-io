"""Alternate upstream and branch workers in fresh processes using the same Python environment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.benchmark_readers.measure import CASES

# Optional controls and baselines that cannot return the same products.
DIAGNOSTIC_CASES = {"curio", "iss", "stereoseq", "xenium-no-panel", "xenium-no-masks-write"}
DEFAULT_CASES = tuple(case for case in CASES if case not in DIAGNOSTIC_CASES)


def run(args: argparse.Namespace) -> int:
    """Save each worker result, including errors, without modifying either checkout."""
    args.output.mkdir(parents=True, exist_ok=False)
    worker = Path(__file__).resolve().with_name("measure.py")
    branch = worker.parents[2]
    roots = {"branch": branch / "src", "main": args.main.resolve() / "src"}
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MALLOC_ARENA_MAX": "2",
        "PYTHONHASHSEED": "0",
    }
    failures = 0
    for trial in range(args.repeats):
        for index, case in enumerate(args.cases):
            revisions = ("main", "branch") if (trial + index) % 2 == 0 else ("branch", "main")
            for revision in revisions:
                search_path = [str(roots[revision])]
                if extra := os.environ.get("PYTHONPATH"):
                    search_path.append(extra)
                switches = ["--warmup"] if args.warmup else []
                if args.first_call:
                    switches.append("--first-call")
                if args.profile:
                    profile_file = args.output / f"{case}-{revision}-{trial + 1}.pstats"
                    switches.extend(("--profile", str(profile_file.resolve())))
                completed = subprocess.run(  # noqa: S603 -- fixed local Python executable, no shell
                    [sys.executable, str(worker), revision, case, str(args.data), str(args.fixtures), *switches],
                    env={**env, "PYTHONPATH": os.pathsep.join(search_path)},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                record = {"case": case, "revision": revision, "trial": trial + 1}
                if completed.returncode == 0:
                    line = next(line for line in completed.stdout.splitlines() if line.startswith("METRIC "))
                    record.update(json.loads(line.removeprefix("METRIC ")))
                else:
                    record["error"] = completed.stderr
                    failures += 1
                destination = args.output / f"{case}-{revision}-{trial + 1}.json"
                destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                sys.stdout.write(f"{destination.name} ({completed.returncode})\n")
                sys.stdout.flush()
    return int(bool(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main", type=Path, help="A separate source checkout of the baseline")
    parser.add_argument("data", type=Path, help="The downloaded dataset directory")
    parser.add_argument("fixtures", type=Path, help="The directory created by prepare.py")
    parser.add_argument("output", type=Path, help="A new directory for per-run JSON")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=DEFAULT_CASES)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--first-call", action="store_true", help="Also time from the first scientific import through completion"
    )
    mode.add_argument(
        "--warmup", action="store_true", help="Run the complete workflow once before timing it in each worker"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Save a cProfile for each timed region in this diagnostic run"
    )
    cli_args = parser.parse_args()
    if cli_args.repeats < 1:
        parser.error("--repeats must be positive")
    sys.exit(run(cli_args))
