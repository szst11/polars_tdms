"""Benchmark the polars_tdms pyarrow read path against nptdms on an existing file.

Takes a TDMS file path and a group name, verifies that polars_tdms returns the
same values as the nptdms reference for every readable channel, then times the
default pyarrow/"buffers" implementation against nptdms for full group reads
and per-channel reads. Channels whose samples are metadata-only (TimeStamp)
are skipped, matching polars_tdms behaviour.

Usage:
    uv run --all-extras python benchmarks/bench_file_vs_nptdms.py /path/to/file.tdms DAQ
    uv run --all-extras python benchmarks/bench_file_vs_nptdms.py file.tdms Sensors \
        --columns Voltage Current --chunk-size=0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

import polars_tdms as pt


class _Req(NamedTuple):
    name: str
    channel: str
    dtype: str
    length: int


def timeit(fn, repeat=3, warmup=1):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _nptdms_read_group(path: str, group: str, channels: list[str]) -> None:
    from nptdms import TdmsFile

    f = TdmsFile.read(path)
    for c in channels:
        f[group][c][:]


def _nptdms_read_channel(path: str, group: str, channel: str) -> None:
    _nptdms_read_group(path, group, [channel])


def _nptdms_read_group_chunked(
    path: str, group: str, channels: list[str], chunk: int, n: int
) -> None:
    """Read every channel in `chunk`-sized slices and rebuild the full arrays."""
    from nptdms import TdmsFile

    if n <= 0:
        return
    f = TdmsFile.read(path)
    pieces = {c: [] for c in channels}
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        for c in channels:
            pieces[c].append(f[group][c][start:end])
    for c in channels:
        np.concatenate(pieces[c])


def check_values(path: str, group: str, requests: list[_Req]) -> bool:
    """Assert every readable channel matches the nptdms reference exactly."""
    from nptdms import TdmsFile

    df = pt.read_tdms(path, group=group)
    f = TdmsFile.read(path)
    ok = True
    for r in requests:
        ref = f[group][r.channel][:]
        if r.dtype == "String":
            match = df[r.name].to_list() == ref.tolist()
        elif r.dtype in ("Float", "Double"):
            match = bool(np.allclose(df[r.name].to_numpy(), ref, equal_nan=True))
        else:
            match = bool(np.array_equal(df[r.name].to_numpy(), ref))
        ok &= match
        print(f"  {r.name:<24} matches nptdms: {match}")

    if len({r.length for r in requests}) == 1:
        chunked = pt.read_tdms(path, group=group, chunk_size=10_000)
        ok &= bool(chunked.equals(df))
        print(f"  chunked==whole: {chunked.equals(df)}")
    else:
        print("  chunked==whole: skipped (channels differ in length)")
    return ok


def bench(
    path: str,
    group: str,
    requests: list[_Req],
    chunk_size: int | None,
    repeat: int,
    warmup: int,
) -> None:
    equal_length = len({r.length for r in requests}) == 1

    header = f"{'benchmark':<30}{'polars_tdms':>16}{'nptdms':>16}{'speedup':>10}"
    print(header)
    print("-" * len(header))

    def row(label, t_ours, t_ref):
        ratio = t_ref / t_ours if t_ours else float("nan")
        print(f"{label:<30}{t_ours:>14.3f}s{t_ref:>14.3f}s{ratio:>9.1f}x")

    if equal_length:
        channels = [r.channel for r in requests]
        t_ours_full = timeit(lambda: pt.read_tdms(path, group=group), repeat, warmup)
        t_ref_full = timeit(lambda: _nptdms_read_group(path, group, channels), repeat, warmup)
        row("full group read", t_ours_full, t_ref_full)

        if chunk_size:
            n = requests[0].length
            t_ours_chunk = timeit(
                lambda: pt.read_tdms(path, group=group, chunk_size=chunk_size),
                repeat, warmup,
            )
            t_ref_chunk = timeit(
                lambda: _nptdms_read_group_chunked(path, group, channels, chunk_size, n),
                repeat, warmup,
            )
            row(f"chunked read ({chunk_size})", t_ours_chunk, t_ref_chunk)
    else:
        print("channels differ in length, benchmarking per channel only ...")

    print("\nper-channel breakdown:")
    for r in requests:
        t_ours_c = timeit(
            lambda r=r: pt.read_tdms(path, group=group, columns=[r.name]), repeat, warmup
        )
        t_ref_c = timeit(
            lambda r=r: _nptdms_read_channel(path, group, r.channel), repeat, warmup
        )
        row(f"{r.name} ({r.dtype})", t_ours_c, t_ref_c)


def _resolve_requests(meta: pt.TdmsMetadata, group: str, columns: list[str] | None) -> list[_Req]:
    grp = meta.group(group)
    if grp is None:
        raise ValueError(
            f"group {group!r} not found in {meta.path!r}; available: {list(meta.group_names)}"
        )
    keep = [c for c in grp.channels if c.readable]
    if columns:
        missing = set(columns) - {c.name for c in keep}
        if missing:
            raise ValueError(f"columns not found (or not readable): {sorted(missing)}")
        keep = [c for c in keep if c.name in columns]
    return [_Req(name=c.name, channel=c.name, dtype=c.dtype, length=c.length) for c in keep]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file", help="path to an existing .tdms file")
    ap.add_argument("group", help="group name to benchmark")
    ap.add_argument("--columns", nargs="*", help="restrict timing to these channels")
    ap.add_argument("--chunk-size", type=int, default=100_000,
                    help="chunk size for the chunked read row (0 disables it)")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    path = Path(args.file)
    if not path.is_file():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 1

    meta = pt.read_metadata(path)
    try:
        requests = _resolve_requests(meta, args.group, args.columns)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not requests:
        print(f"error: group {args.group!r} has no readable channel data", file=sys.stderr)
        return 1

    print(f"file: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"group: {args.group} · channels: "
          f"{', '.join(f'{r.channel}({r.dtype})' for r in requests)}")
    print("pyarrow path: " + ("enabled" if pt._pa is not None else "FALLBACK (numpy)"))
    print(f"samples: {requests[0].length}")

    print("\ncorrectness vs nptdms:")
    if not check_values(path, args.group, requests):
        print("content check FAILED — results below are not meaningful")
        return 1

    print("\n" + "─" * 30)
    bench(path, args.group, requests, args.chunk_size or None, args.repeat, args.warmup)
    return 0


if __name__ == "__main__":
    sys.exit(main())