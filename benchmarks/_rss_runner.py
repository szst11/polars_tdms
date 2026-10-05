"""Subprocess helper that runs a single read and prints its current RSS (KiB).

Used by ``bench_vs_nptdms.py`` to measure memory in a fresh interpreter so the
result is not polluted by the benchmark process or earlier measurements.

``ru_maxrss`` is inherited through fork+exec (it is a peak high-water mark
carried into the child), so we read ``VmRSS`` from ``/proc/self/status``
(current resident set) instead; for a short-lived measurement process the
current RSS right after the read IS the footprint during the read.
"""

from __future__ import annotations

import sys


def current_rss_kb() -> int:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    raise RuntimeError("VmRSS not found in /proc/self/status")


def main() -> None:
    op, path, group = sys.argv[1:4]
    cols_raw = sys.argv[4] if len(sys.argv) > 4 else ""
    cols = [c for c in cols_raw.split(",") if c]

    import polars_tdms as pt
    from nptdms import TdmsFile

    if op == "polars_full":
        result = pt.read_tdms(path, group=group)
    elif op == "polars_partial":
        result = pt.read_tdms(path, group=group, columns=cols)
    elif op == "nptdms_full":
        result = TdmsFile.read(path).as_dataframe()
    elif op == "nptdms_partial":
        result = TdmsFile.read(path).as_dataframe().iloc[:, : len(cols)]
    else:
        raise SystemExit(f"unknown op {op!r}")

    # keep the result alive while measuring so its memory is still resident
    assert result is not None
    print(current_rss_kb())


if __name__ == "__main__":
    main()