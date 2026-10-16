"""Measure first-open index creation separately from metadata scans.

Accepts a TDMS file or a folder of TDMS files. Uses temporary copies beside
the source (or under --workspace), preserving all original files and indexes.
Copying, correctness checks and cleanup are excluded from timings. Repeated
opens are warm-cache measurements, not cold-storage benchmarks.
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import polars_tdms as pt
from nptdms import TdmsFile


def timed_read(path: Path, **options) -> tuple[pt.TdmsMetadata, float]:
    start = time.perf_counter()
    metadata = pt.read_metadata(path, **options)
    return metadata, time.perf_counter() - start


def benchmark_file(original: Path, workspace: Path, repeat: int, warmup: int) -> None:
    with tempfile.TemporaryDirectory(prefix="polars_tdms_index_bench_", dir=workspace) as directory:
        path = Path(directory) / original.name
        shutil.copyfile(original, path)
        index = Path(f"{path}_index")
        timings: dict[str, list[float]] = {
            "metadata scan, no index": [],
            "first open + index creation": [],
            "existing index open": [],
        }
        expected = pt.read_metadata(path, use_index_file=False)
        with TdmsFile.open(path) as reference:
            if expected.group_names != tuple(g.name for g in reference.groups()):
                raise AssertionError("group names differ from nptdms")
            for group in expected.groups:
                ref = reference[group.name]
                if group.channel_names != tuple(c.name for c in ref.channels()):
                    raise AssertionError(f"channel names differ from nptdms: {group.name}")
                for channel in group.channels:
                    if channel.length != len(ref[channel.name]):
                        raise AssertionError(f"channel length differs from nptdms: {group.name}/{channel.name}")
        index_size = 0
        for iteration in range(warmup + repeat):
            if index.exists():
                raise AssertionError("index must be absent before measuring creation")
            scanned, scan_time = timed_read(path, use_index_file=False)
            generated, creation_time = timed_read(path)
            if not index.is_file():
                raise RuntimeError(f"index was not created for {original.name}")
            with index.open("rb") as stream:
                if stream.read(4) != b"TDSh":
                    raise AssertionError("index signature is not TDSh")
            index_size = index.stat().st_size
            indexed, indexed_time = timed_read(path, create_index_if_missing=False)
            verified = pt.read_metadata(
                path, create_index_if_missing=False, verify_index=True
            )
            if not (scanned == generated == indexed == verified == expected):
                raise AssertionError("metadata differs between scan/generated/indexed/verified opens")
            if iteration >= warmup:
                for label, elapsed in zip(
                    timings, (scan_time, creation_time, indexed_time), strict=True
                ):
                    timings[label].append(elapsed)
            index.unlink()
        print(f"\nfile: {original.name}", flush=True)
        print(f"data MB: {original.stat().st_size / 1e6:.3f}; index MB: {index_size / 1e6:.3f}")
        for label, values in timings.items():
            print(
                f"{label:<32} best={min(values) * 1000:9.3f} ms "
                f"median={statistics.median(values) * 1000:9.3f} ms",
                flush=True,
            )
        print("correctness: nptdms structure=True, metadata equality=True, verify_index=True")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="TDMS file or folder containing .tdms files")
    parser.add_argument("--workspace", type=Path, help="existing folder for temporary copies")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0:
        parser.error("repeat must be positive and warmup nonnegative")
    if args.path.is_file():
        files = [args.path]
        workspace = args.workspace if args.workspace is not None else args.path.parent
    elif args.path.is_dir():
        files = sorted(args.path.glob("*.tdms"))
        workspace = args.workspace if args.workspace is not None else args.path
    else:
        parser.error("input file or folder does not exist")
    if not files:
        parser.error("no TDMS files found")
    if not workspace.is_dir():
        parser.error("workspace must be an existing directory")
    print("first-index creation benchmark", flush=True)
    print(f"files: {len(files)}; repeat: {args.repeat}; warmup: {args.warmup}")
    print("temporary copies only; original files and indexes preserved")
    print("no channel data read; copying and verification excluded; warm-cache timings")
    for path in files:
        benchmark_file(path, workspace, args.repeat, args.warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
