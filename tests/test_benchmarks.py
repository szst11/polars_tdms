from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import polars_tdms as pt
import pytest
from nptdms import TdmsWriter, ChannelObject, GroupObject

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"

NEEDS_PYARROW = pytest.mark.skipif(pt._pa is None, reason="pyarrow fast path not available")


def _run_script(name: str, args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(BENCH_DIR / name), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def mixed_file(tmp_path_factory):
    """A small multi-segment group with numeric, boolean and string channels."""
    path = tmp_path_factory.mktemp("bench") / "mixed.tdms"
    rng = np.random.default_rng(3)
    with TdmsWriter(path) as w:
        grp = GroupObject("DAQ")
        for _ in range(2):
            w.write_segment(
                [
                    grp,
                    ChannelObject("DAQ", "Volts", rng.standard_normal(2000)),
                    ChannelObject("DAQ", "Counts", rng.integers(0, 100, 2000, dtype=np.int32)),
                    ChannelObject("DAQ", "On", rng.integers(0, 2, 2000, dtype=bool)),
                    ChannelObject("DAQ", "Label", [f"v{i}" for i in range(2000)]),
                ]
            )
    return path


@NEEDS_PYARROW
def test_bench_vs_numpy_fallback_runs(tmp_path):
    path = tmp_path / "bench_mixed.tdms"
    res = _run_script(
        "bench_vs_numpy_fallback.py",
        ["--segments=3", "--samples-per-segment=2000", f"--path={path}"],
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "chunked==whole: True" in out
    for marker in (
        "buffers/pyarrow",
        "numpy fallback",
        "nptdms",
        "vs nptdms",
        "full group read",
        "chunked read (10k)",
        "sig_f64 (Float64)",
        "label (String)",
    ):
        assert marker in out, f"missing {marker!r} in output:\n{out}"


def test_bench_file_vs_nptdms_runs(mixed_file, tmp_path):
    res = _run_script("bench_file_vs_nptdms.py", [str(mixed_file), "DAQ"])
    assert res.returncode == 0, res.stderr
    out = res.stdout
    for marker in (
        "Volts(Double)",
        "Counts(I32)",
        "Label(String)",
        "matches nptdms: True",
        "full group read",
        "chunked read (100000)",
        "per-channel breakdown",
        "polars_tdms",
        "nptdms",
        "speedup",
    ):
        assert marker in out, f"missing {marker!r} in output:\n{out}"
    assert "chunked==whole: True" in out


def test_bench_file_vs_nptdms_columns(tmp_path):
    from nptdms import TdmsWriter, ChannelObject, GroupObject

    path = tmp_path / "cols.tdms"
    with TdmsWriter(path) as w:
        w.write_segment(
            [
                GroupObject("G"),
                ChannelObject("G", "A", np.arange(100.0)),
                ChannelObject("G", "B", np.arange(100, 200.0)),
            ]
        )
    res = _run_script(
        "bench_file_vs_nptdms.py",
        [str(path), "G", "--columns", "B", "--chunk-size", "0"],
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "B (Double)" in out
    assert "A (Double)" not in out
    assert "chunked read" not in out


def test_bench_file_vs_nptdms_unknown_group(mixed_file):
    res = _run_script("bench_file_vs_nptdms.py", [str(mixed_file), "Missing"])
    assert res.returncode != 0
    assert "not found" in res.stderr


def test_bench_file_vs_nptdms_missing_file(tmp_path):
    res = _run_script("bench_file_vs_nptdms.py", [str(tmp_path / "nope.tdms"), "G"])
    assert res.returncode != 0
    assert "does not exist" in res.stderr