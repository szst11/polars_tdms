"""Compare Rust scalar, vectored and coalesced reads using a TDMS fragment plan.

For the experimental rows, nptdms private metadata builds the plan, outside raw-read
timings. Only complete, little-endian, non-interleaved, non-DAQmx segments are
supported. Strings/Boolean remain on the normal reader in full-group timings.
No companion index is created or modified.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import polars_tdms as pt
from nptdms import TdmsFile
from nptdms.common import toc_properties
from polars_tdms import _core, _series_from_numeric_buffers

STRATEGIES = ("scalar", "vectored", "coalesced")


@dataclass
class Plan:
    names: list[str]
    dtypes: list[str]
    lengths: list[int]
    sizes: list[int]
    fragments: list[tuple[int, int, int, int]]
    segments: int


def build_plan(path: str, group: str, columns: list[str] | None = None) -> Plan:
    with TdmsFile.open(path) as source:
        channels = source[group].channels()
        if columns is not None:
            missing = set(columns) - {c.name for c in channels}
            if missing:
                raise ValueError(f"columns not found: {sorted(missing)}")
            channels = [c for c in channels if c.name in columns]
        meta = pt.read_metadata(path, create_index_if_missing=False)
        grp = meta.group(group)
        if grp is None:
            raise ValueError(f"group {group!r} not found")
        dtype_names = {c.name: c.dtype for c in grp.channels}
        channels = [
            c for c in channels
            if dtype_names[c.name] not in ("String", "Boolean", "TimeStamp", "Void")
        ]
        if not channels:
            raise ValueError("no fixed-width numeric channels selected")
        indices = {c.path: i for i, c in enumerate(channels)}
        sizes = [len(c) * c.data_type.size for c in channels]
        cursors = [0] * len(channels)
        fragments = []
        # Private nptdms layout access is confined to this benchmark, not the API.
        segments = source._reader._segments
        for segment in segments:
            if segment.num_chunks == 0:
                continue
            unsupported = ("kTocInterleavedData", "kTocBigEndian", "kTocDAQmxRawData")
            if (
                any(segment.toc_mask & toc_properties[flag] for flag in unsupported)
                or segment.segment_incomplete
                or segment.final_chunk_lengths_override is not None
            ):
                raise ValueError("unsupported layout: requires complete LE non-interleaved non-DAQmx segments")
            for chunk in range(segment.num_chunks):
                offset = segment.data_position + chunk * segment._get_chunk_size()
                for obj in segment.ordered_objects:
                    if not obj.has_data:
                        continue
                    size = int(obj.data_size)
                    index = indices.get(obj.path)
                    if index is not None and size:
                        expected = obj.number_values * channels[index].data_type.size
                        if size != expected or obj.data_type != channels[index].data_type:
                            raise ValueError("channel type or size changed across segments")
                        fragments.append((offset, index, cursors[index], size))
                        cursors[index] += size
                    offset += size
        if cursors != sizes:
            raise ValueError("fragment plan does not match channel lengths")
        return Plan(
            [c.name for c in channels], [dtype_names[c.name] for c in channels],
            [len(c) for c in channels], sizes, fragments, len(segments),
        )


def measure(fn, repeat: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times), statistics.median(times)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("group")
    parser.add_argument("--columns", nargs="+")
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0:
        parser.error("repeat must be positive and warmup nonnegative")
    path = str(Path(args.file))
    if not Path(path).is_file():
        parser.error("file does not exist")
    start = time.perf_counter()
    try:
        plan = build_plan(path, args.group, args.columns)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    planning = time.perf_counter() - start
    print("scatter-gather experiment")
    print(f"platform: {platform.system()}; file: {Path(path).stat().st_size / 1e6:.2f} MB")
    print(f"segments: {plan.segments}; numeric channels: {len(plan.names)}; fragments: {len(plan.fragments)}")
    print(f"numeric payload: {sum(plan.sizes) / 1e6:.2f} MB; plan construction: {planning * 1000:.2f} ms")
    print("warm-cache timings; plan construction excluded except in explicit planned row")
    print("read/seek counters are Rust API calls, not traced OS syscalls")

    def read_raw(strategy):
        return _core._read_fragment_buffers(path, plan.sizes, plan.fragments, strategy)

    with TdmsFile.open(path) as reference:
        for strategy in STRATEGIES:
            buffers, seeks, reads, transferred = read_raw(strategy)
            for name, dtype, length, data in zip(
                plan.names, plan.dtypes, plan.lengths, buffers, strict=True
            ):
                actual = _series_from_numeric_buffers(name, dtype, data, length).to_numpy()
                expected = reference[args.group][name][:]
                np.testing.assert_array_equal(actual, expected, err_msg=f"{strategy}: {name}")
            print(f"{strategy}: matches nptdms: True; seeks={seeks}; reads={reads}; bytes={transferred}")

    def row(label, fn):
        best, median = measure(fn, args.repeat, args.warmup)
        print(f"{label:<38} best={best * 1000:9.3f} ms median={median * 1000:9.3f} ms")

    with pt.open_tdms(path, create_index_if_missing=False) as source:
        handle = source._handle
        if handle is None:
            raise RuntimeError("TDMS source is closed")

        def current_raw():
            return [
                handle.read_channel_range_buffers(args.group, name, 0, length)
                for name, length in zip(plan.names, plan.lengths, strict=True)
            ]

        row("current raw (reused metadata)", current_raw)
        row(
            "legacy raw (per-channel scalar)",
            lambda: [
                handle._read_channel_range_buffers_scalar(args.group, name, 0, length)
                for name, length in zip(plan.names, plan.lengths, strict=True)
            ],
        )
        if len(set(plan.lengths)) == 1:
            row(
                "production batched raw (Rust index)",
                lambda: handle.read_channels_range_buffers(
                    [(args.group, name) for name in plan.names], 0, plan.lengths[0]
                ),
            )
        for strategy in STRATEGIES:
            row(f"{strategy} raw (reused plan)", lambda s=strategy: read_raw(s))

    def current_group():
        return pt.read_tdms(
            path, group=args.group, columns=args.columns,
            chunk_size=None, create_index_if_missing=False,
        )

    class LegacyHandle:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def read_channel_range_buffers(self, *args):
            return self.handle._read_channel_range_buffers_scalar(*args)

        def read_channels_range_buffers(self, channels, start, end):
            return [
                self.handle._read_channel_range_buffers_scalar(group, channel, start, end)
                for group, channel in channels
            ]

    def legacy_group():
        with pt.open_tdms(path, create_index_if_missing=False) as source:
            if source._handle is None:
                raise RuntimeError("TDMS source is closed")
            source._handle = LegacyHandle(source._handle)
            return source.read(group=args.group, columns=args.columns, chunk_size=None)

    meta = pt.read_metadata(path, create_index_if_missing=False)
    grp = meta.group(args.group)
    if grp is None:
        raise ValueError(f"group {args.group!r} not found")
    selected = [
        c.name for c in grp.channels
        if c.readable and (args.columns is None or c.name in args.columns)
    ]
    other = [name for name in selected if name not in plan.names]
    lengths = {c.length for c in grp.channels if c.name in selected and c.length > 0}
    if len(lengths) > 1:
        print("full group comparison: skipped (channels differ in length)")
        return 0

    def experimental_group(strategy, active_plan=plan):
        buffers, _, _, _ = _core._read_fragment_buffers(
            path, active_plan.sizes, active_plan.fragments, strategy
        )
        numeric = {
            name: _series_from_numeric_buffers(name, dtype, data, length)
            for name, dtype, length, data in zip(
                active_plan.names, active_plan.dtypes, active_plan.lengths, buffers, strict=True
            )
        }
        if other:
            remainder = pt.read_tdms(
                path, group=args.group, columns=other,
                chunk_size=None, create_index_if_missing=False,
            )
            numeric.update({name: remainder[name] for name in other})
        target_length = next(iter(lengths), 0)
        return pl.DataFrame([
            numeric[name] if len(numeric[name]) or not target_length
            else pl.Series(name, [None] * target_length, dtype=numeric[name].dtype)
            for name in selected
        ])

    baseline = current_group()
    if not legacy_group().equals(baseline):
        raise AssertionError("production full group differs from legacy reader")
    print("production==legacy full group: True")
    for strategy in STRATEGIES:
        if not experimental_group(strategy).equals(baseline):
            raise AssertionError(f"{strategy} full group differs from current reader")
    print("full group equality (including normal string/boolean reads): True")
    row("current full group (includes open)", current_group)
    row("legacy full group (includes open)", legacy_group)
    for strategy in STRATEGIES:
        row(f"{strategy} full group (reused plan)", lambda s=strategy: experimental_group(s))
    row(
        "coalesced full group + plan",
        lambda: experimental_group("coalesced", build_plan(path, args.group, args.columns)),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
