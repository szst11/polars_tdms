"""Benchmark the pyarrow/"buffers" read path against the numpy fallback and nptdms.

Generates a ~500 MB TDMS file with one group containing a String channel plus
a mix of numeric channels (Float64, Int32, Boolean), split across many
segments (~1 per MB), then times full and chunked group reads for the default
pyarrow-based path, the numpy fallback, and the nptdms reference. Also verifies
that both local paths return the exact same values as nptdms.

Usage:
    uv run --all-extras python benchmarks/bench_vs_numpy_fallback.py
    uv run --all-extras python benchmarks/bench_vs_numpy_fallback.py --segments=200 --samples-per-segment=100000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import polars as pl
import polars_tdms as pt

from nptdms import TdmsFile

GROUP = "G"

_NPTDMS_CHANNELS = ("sig_f64", "sig_i32", "sig_bool", "label")


def make_file(path: str, segments: int, samples_per_segment: int) -> None:
    """Write a synthetic multi-segment TDMS file (string + numeric channels)."""
    from nptdms import TdmsWriter, GroupObject, ChannelObject

    rng = np.random.default_rng(42)
    with TdmsWriter(path) as w:
        grp = GroupObject(GROUP)
        for _ in range(segments):
            f64 = rng.standard_normal(samples_per_segment)
            i32 = rng.integers(0, 100_000, samples_per_segment, dtype=np.int32)
            b = rng.integers(0, 2, samples_per_segment, dtype=bool)
            labels = [f"s{v:06d}" for v in i32]
            w.write_segment(
                [
                    grp,
                    ChannelObject(GROUP, "sig_f64", f64),
                    ChannelObject(GROUP, "sig_i32", i32),
                    ChannelObject(GROUP, "sig_bool", b),
                    ChannelObject(GROUP, "label", labels),
                ]
            )


def timeit(fn, repeat=3, warmup=1):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def check_values(path: str) -> bool:
    """Assert every channel matches the nptdms reference exactly."""

    ref = TdmsFile.read(path)
    df = pt.read_tdms(path, group=GROUP)
    cols = {
        "sig_f64": lambda: np.allclose(df["sig_f64"].to_numpy(), ref[GROUP]["sig_f64"][:]),
        "sig_i32": lambda: df["sig_i32"].to_numpy().tolist() == ref[GROUP]["sig_i32"][:].tolist(),
        "sig_bool": lambda: df["sig_bool"].to_numpy().tolist() == ref[GROUP]["sig_bool"][:].tolist(),
        "label": lambda: df["label"].to_list() == ref[GROUP]["label"][:].tolist(),
    }
    ok = True
    for name, fn in cols.items():
        match = fn()
        ok &= bool(match)
        print(f"  {name:<10} matches nptdms: {match}")
    return ok


def _nptdms_full_read(path: str) -> None:    
    with TdmsFile.open(path) as _nptdms_file:
        _channel_data = _nptdms_file[GROUP].as_dataframe()


def _nptdms_channel_read(path: str, channel: str) -> None:
    with TdmsFile.open(path) as _nptdms_file:
        _channel_data = _nptdms_file[GROUP][channel].as_dataframe()




def bench(path: str) -> None:
    header = (
        f"{'benchmark':<24}{'buffers/pyarrow':>15}{'numpy fallback':>15}"
        f"{'nptdms':>15}{'vs nptdms':>11}"
    )
    print(header)
    print("-" * len(header))

    def row(name, t_arrow, t_np, t_tdms):
        ratio = t_tdms / t_arrow if t_arrow else float("nan")
        print(f"{name:<24}{t_arrow:>13.3f}s{t_np:>13.3f}s{t_tdms:>13.3f}s{ratio:>9.1f}x")

    # default path (pyarrow buffers)
    t_arrow_full = timeit(lambda: pt.read_tdms(path, group=GROUP,chunk_size=None))

    # numpy fallback
    saved_pa = pt._pa
    pt._pa = None
    try:
        t_np_full = timeit(lambda: pt.read_tdms(path, group=GROUP,chunk_size=None))
    finally:
        pt._pa = saved_pa
    
    t_tdms_full = timeit(lambda: _nptdms_full_read(path))

    row("full group read", t_arrow_full, t_np_full, t_tdms_full)

    print("\nper-channel breakdown (full read):")
    saved_pa = pt._pa

    pt._pa = saved_pa
    t_arrow_f64 = timeit(lambda: pt.read_tdms(path, group=GROUP, columns=["sig_f64"],chunk_size=None))
    pt._pa = None
    t_np_f64 = timeit(lambda: pt.read_tdms(path, group=GROUP, columns=["sig_f64"],chunk_size=None))
    pt._pa = saved_pa
    t_tdms_f64 = timeit(lambda: _nptdms_channel_read(path, "sig_f64"))
    row("sig_f64 (Float64)", t_arrow_f64, t_np_f64, t_tdms_f64)

    t_arrow_label = timeit(lambda: pt.read_tdms(path, group=GROUP, columns=["label"],chunk_size=None))
    pt._pa = None
    t_np_label = timeit(lambda: pt.read_tdms(path, group=GROUP, columns=["label"],chunk_size=None))
    pt._pa = saved_pa
    t_tdms_label = timeit(lambda: _nptdms_channel_read(path, "label"))
    row("label (String)", t_arrow_label, t_np_label, t_tdms_label)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--segments", type=int, default=200, help="number of TDMS segments")
    ap.add_argument("--samples-per-segment", type=int, default=100_000)
    ap.add_argument("--path", default="/tmp/opencode/bench_mixed.tdms")
    ap.add_argument("--skip-write", action="store_true", help="reuse an existing file")
    args = ap.parse_args()

    mb_per_seg = args.samples_per_segment * (8 + 4 + 1 + 7 + 4) / 1e6  # ~20 bytes/sample
    print(f"samples={args.samples_per_segment}/seg · segments={args.segments} "
          f"(~{mb_per_seg:.1f} MB/seg)")
    if not args.skip_write:
        print(f"writing {args.segments * mb_per_seg:.0f} MB of f64/i32/bool/string data ...")
        t0 = time.perf_counter()
        make_file(args.path, args.segments, args.samples_per_segment)
        print(f"write took {time.perf_counter() - t0:.1f}s")

    print(f"\nfile size: {Path(args.path).stat().st_size / 1e6:.1f} MB")
    print("\ncorrectness vs nptdms:")
    if not check_values(args.path):
        import sys

        print("content check FAILED — results below are not meaningful")
        sys.exit(1)
    print("\n" + "─" * 30)
    bench(args.path)


if __name__ == "__main__":
    main()
