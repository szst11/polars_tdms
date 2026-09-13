"""Benchmark polars_tdms (Rust/tdms-rs backend) against nptdms.

Generates a synthetic multi-channel TDMS file, then times metadata parsing
and full / partial channel reads, measuring wall time and peak RSS.

Usage:
    uv run --extra bench python benchmarks/bench_vs_nptdms.py --samples=2000000 --channels=8
"""

from __future__ import annotations

import argparse
import importlib.metadata
import resource
import sys
import time
from pathlib import Path

import numpy as np

import polars as pl
import polars_tdms as pt

GROUP = "DAQ"


def max_rss_kb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def timeit(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


_RUNNER = str(Path(__file__).resolve().parent / "_rss_runner.py")


def peak_rss_mib(op: str, path: str, group: str, cols) -> int:
    """Peak RSS of the op in a fresh subprocess (VmRSS, KiB -> MiB)."""
    import subprocess

    out = subprocess.run(
        [sys.executable, _RUNNER, op, path, group, ",".join(cols)],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"{op} failed:\n{out.stderr}")
    return int(out.stdout.strip()) / 1024  # KiB -> MiB


def make_file(path: str, samples: int, channels: int) -> None:
    """Write a synthetic TDMS file with nptdms (single segment, f64 channels)."""
    from nptdms import TdmsWriter, GroupObject, ChannelObject

    rng = np.random.default_rng(0)
    with TdmsWriter(path) as w:
        grp = GroupObject(GROUP, properties={"sampling_rate": float(50_000)})
        seg = [grp]
        for i in range(channels):
            data = np.sin(np.arange(samples) / 1000.0 + i) + 0.001 * rng.standard_normal(samples)
            seg.append(ChannelObject(GROUP, f"Ch{i}", data))
        w.write_segment(seg)


def bench(path: str, group: str, channels: int) -> None:
    header = f"{'benchmark':<52}{'nptdms':>14}{'polars_tdms':>14}{'ratio':>10}"
    print(header)
    print("-" * len(header))

    def row(name, nptdms_t, ours_t):
        ratio = nptdms_t / ours_t if ours_t else float("nan")
        print(f"{name:<52}{nptdms_t:>12.3f}s{ours_t:>12.3f}s{ratio:>9.1f}x")

    # 1. metadata only  -----------------------------------------------------
    t_np_meta = timeit(lambda: _np_metadata(path))
    t_rs_meta = timeit(lambda: pt.read_metadata(path))
    row("metadata (groups/channels/properties)", t_np_meta, t_rs_meta)

    # 2. full read ----------------------------------------------------------
    def np_full():
        from nptdms import TdmsFile

        return TdmsFile.read(path).as_dataframe()

    t_np_full = timeit(np_full, repeat=2)
    rss_np_full = peak_rss_mib("nptdms_full", path, group, [])

    def rs_full():
        pt.read_tdms(path, group=group)

    t_rs_full = timeit(rs_full, repeat=2)
    rss_rs_full = peak_rss_mib("polars_full", path, group, [])
    row("full group read", t_np_full, t_rs_full)

    # 2b. full read, lazy entry point ---------------------------------------
    t_rs_lazy = timeit(lambda: pt.scan_tdms(path, group=group).collect(), repeat=2)
    print(f"{'lazy scan_tdms(...).collect()':<52}{'-':>14}{t_rs_lazy:>12.3f}s{'':>10}")

    # 3. partial read (2 of N channels) -------------------------------------
    keep = [f"Ch{i}" for i in range(min(2, channels))]

    def np_partial():
        from nptdms import TdmsFile

        return TdmsFile.read(path).as_dataframe().iloc[:, : len(keep)]

    t_np_partial = timeit(np_partial, repeat=2)
    rss_np_partial = peak_rss_mib("nptdms_partial", path, group, keep)

    def rs_partial():
        pt.read_tdms(path, group=group, columns=keep)

    t_rs_partial = timeit(rs_partial, repeat=2)
    rss_rs_partial = peak_rss_mib("polars_partial", path, group, keep)
    row("partial read (2 channels)", t_np_partial, t_rs_partial)

    # 4. peak memory during full read ---------------------------------------
    print(f"\npeak RSS during full read: nptdms={rss_np_full:.0f} MiB, "
          f"polars_tdms={rss_rs_full:.0f} MiB, "
          f"ratio={rss_np_full/max(rss_rs_full,1):.1f}x")
    print(f"peak RSS during partial read: nptdms={rss_np_partial:.0f} MiB, "
          f"polars_tdms={rss_rs_partial:.0f} MiB, "
          f"ratio={rss_np_partial/max(rss_rs_partial,1):.1f}x")
    print("(RSS = resident set after the read in a fresh subprocess)")

    # 5. correctness check ---------------------------------------------------
    ours = pt.read_tdms(path, group=group, columns=keep)
    from nptdms import TdmsFile

    ref = TdmsFile.read(path)
    ok = all(np.allclose(ref[group][c][:], ours[c].to_numpy()) for c in keep)
    print(f"\nvalues match nptdms: {ok}")


def _np_metadata(path):
    from nptdms import TdmsFile

    return TdmsFile.read(path).groups()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=2_000_000)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--path", default="/tmp/opencode/bench.tdms")
    ap.add_argument("--skip-write", action="store_true", help="reuse an existing file")
    args = ap.parse_args()

    nptdms_ver = importlib.metadata.version("nptdms")
    print(f"nptdms {nptdms_ver} · polars {pl.__version__} · samples={args.samples} "
          f"channels={args.channels}")
    if not args.skip_write:
        print(f"writing {args.samples*args.channels*8/1e6:.0f} MiB of f64 channel data ...")
        make_file(args.path, args.samples, args.channels)

    bench(args.path, GROUP, args.channels)


if __name__ == "__main__":
    sys.exit(main())