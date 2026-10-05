"""Benchmark metadata-only scanning with and without `.tdms_index` companions.

Writes ``--n-files`` TDMS files of identical structure but mixed random content,
each split across ``--segments`` segments (defaults sized to ~500 MB in total),
then times building a vertical lazy-frame union over all of them (``scan_tdms``
+ ``pl.concat`` + ``collect_schema``) --- no channel data is ever read. The
union build is timed first without any ``.tdms_index`` sidecars, then again
after generating the sidecars, reporting the per-configuration time and the
speedup. Multi-segment files spread the metadata through the data file, which is
where the compact companion index speeds up the scan.

Usage:
    uv run --all-extras python benchmarks/bench_tdms_index.py
    uv run --all-extras python benchmarks/bench_tdms_index.py --n-files=100 --samples=200000 --segments=200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import polars as pl
import polars_tdms as pt

GROUP = "G"

#: Channels shared by every generated file. Approximate on-disk bytes per row:
#: Float64 (8) + Int32 (4) + String (4-byte length prefix + ~10 char payload).
CHANNELS = ("Ch0", "Ch1", "Label")
_BYTES_PER_ROW = 8 + 4 + 4 + 10


def make_file(path: str, samples: int, segments: int, seed: int) -> None:
    """Write one TDMS file with the shared structure split across segments."""
    from nptdms import TdmsWriter, GroupObject, ChannelObject

    rng = np.random.default_rng(seed)
    per_segment = max(1, samples // segments)
    with TdmsWriter(path) as w:
        grp = GroupObject(GROUP)
        for _ in range(segments):
            f64 = rng.standard_normal(per_segment)
            i32 = rng.integers(0, 100_000, per_segment, dtype=np.int32)
            labels = [f"v{v:010d}" for v in rng.integers(0, 1_000_000, per_segment)]
            w.write_segment(
                [
                    grp,
                    ChannelObject(GROUP, "Ch0", f64),
                    ChannelObject(GROUP, "Ch1", i32),
                    ChannelObject(GROUP, "Label", labels),
                ]
            )


def timeit(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def build_lazy_union(files: list[str], use_index_file: bool) -> pl.LazyFrame:
    """Scan all files lazily and build a vertical union without reading data."""
    lfs = [
        pt.scan_tdms(
            p,
            group=GROUP,
            use_index_file=use_index_file,
            create_index_if_missing=False,
        )
        for p in files
    ]
    return pl.concat(lfs, how="vertical")


def union_time(files: list[str], use_index_file: bool) -> float:
    return timeit(lambda: build_lazy_union(files, use_index_file).collect_schema())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-files", type=int, default=100)
    ap.add_argument("--samples", type=int, default=200_000)
    ap.add_argument("--segments", type=int, default=200)
    ap.add_argument("--path", default="/tmp/opencode/bench_index")
    ap.add_argument("--skip-write", action="store_true", help="reuse existing files")
    args = ap.parse_args()

    out = Path(args.path)
    out.mkdir(parents=True, exist_ok=True)
    files = [str(out / f"data_{i:03d}.tdms") for i in range(args.n_files)]
    index_files = [Path(p).with_suffix(".tdms_index") for p in files]

    est_mib = args.n_files * args.samples * _BYTES_PER_ROW / 1e6
    print(
        f"n_files={args.n_files} samples/channel={args.samples} "
        f"segments={args.segments} "
        f"(~{est_mib:.0f} MiB total when written) · polars {pl.__version__}"
    )

    # 1. write files (no .tdms_index) ----------------------------------------
    if not args.skip_write:
        t_write = timeit(
            lambda: [
                make_file(files[i], args.samples, args.segments, i)
                for i in range(args.n_files)
            ],
            repeat=1,
        )
        print(f"{'write files no index':<52}{t_write:>12.3f}s")
    total_bytes = sum(Path(p).stat().st_size for p in files)
    print(f"{'files on disk':<52}{total_bytes / 1e6:>11.1f} MiB")
    print("no channel data read during timing: 'collect()' is never called")
    print("-" * 76)

    # 2. union build without .tdms_index -------------------------------------
    for idx in index_files:
        idx.unlink(missing_ok=True)
    t_without = union_time(files, use_index_file=False)
    print(f"{'build vertical lazy union (scan + schema), without index':<52}{t_without:>12.3f}s")

    # 3. generate the .tdms_index sidecars, then re-time ----------------------
    t_gen = timeit(lambda: [pt.read_metadata(p) for p in files], repeat=1)
    n_sidecars = 0
    for idx in index_files:
        if idx.exists() and idx.read_bytes()[:4] == b"TDSh":
            n_sidecars += 1
    print(f"{'index generation (one open per file)':<52}{t_gen:>12.3f}s")
    print(f"{'index files present and valid (TDSh)':<52}{n_sidecars}/{len(index_files):>5}")

    t_with = union_time(files, use_index_file=True)
    speedup = t_without / max(t_with, 1e-9)
    print(f"{'build vertical lazy union (scan + schema), with index':<52}{t_with:>12.3f}s")
    print(f"{'speedup (no index / with index)':<52}{speedup:>11.1f}x")

    # 4. correctness (metadata only, no raw reads) ----------------------------
    per_segment = max(1, args.samples // args.segments)
    per_file = per_segment * args.segments
    meta = pt.read_metadata(files[0])
    g = meta.group(GROUP)
    ok_len = g.channel("Ch0").length == per_file
    ok_cols = g.channel_names == CHANNELS
    schema = build_lazy_union(files, use_index_file=False).collect_schema()
    ok_schema = all(schema[col] == dt for col, dt in
                    (("Ch0", pl.Float64), ("Ch1", pl.Int32), ("Label", pl.Utf8)))
    print(f"correctness: rows/channel={ok_len}, columns={ok_cols}, union schema={ok_schema}")


if __name__ == "__main__":
    main()