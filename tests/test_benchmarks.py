from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from nptdms import TdmsWriter, ChannelObject, GroupObject

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"


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
        [str(path), "G", "--columns", "B"],
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "B (Double)" in out
    assert "A (Double)" not in out


def test_bench_file_vs_nptdms_empty_channel(tmp_path):
    path = tmp_path / "empty.tdms"
    with TdmsWriter(path) as w:
        w.write_segment(
            [
                GroupObject("G"),
                ChannelObject("G", "Signal", np.arange(3, dtype=np.float64)),
                ChannelObject("G", "Empty", np.array([], dtype=np.float64)),
            ]
        )

    res = _run_script("bench_file_vs_nptdms.py", [str(path), "G"])
    assert res.returncode == 0, res.stderr
    assert "Empty (Double)" in res.stdout
    assert "Empty                    matches nptdms: True" in res.stdout


def test_bench_file_vs_nptdms_unknown_group(mixed_file):
    res = _run_script("bench_file_vs_nptdms.py", [str(mixed_file), "Missing"])
    assert res.returncode != 0
    assert "not found" in res.stderr


def test_bench_file_vs_nptdms_missing_file(tmp_path):
    res = _run_script("bench_file_vs_nptdms.py", [str(tmp_path / "nope.tdms"), "G"])
    assert res.returncode != 0
    assert "does not exist" in res.stderr


def test_bench_tdms_index_runs(tmp_path):
    res = _run_script(
        "bench_tdms_index.py",
        ["--n-files=3", "--samples=200", f"--path={tmp_path}"],
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout
    for marker in (
        "without index",
        "with index",
        "index generation",
        "index files present and valid",
        "speedup",
        "no channel data read",
        "build vertical lazy union",
        "correctness: rows/channel=True, columns=True, union schema=True",
    ):
        assert marker in out, f"missing {marker!r} in output:\n{out}"
    assert (tmp_path / "data_000.tdms_index").exists()


def test_bench_scatter_gather_runs(mixed_file):
    res = _run_script(
        "bench_scatter_gather.py",
        [str(mixed_file), "DAQ", "--repeat", "1", "--warmup", "0"],
    )
    assert res.returncode == 0, res.stderr
    for marker in (
        "scatter-gather experiment",
        "vectored: matches nptdms: True",
        "coalesced: matches nptdms: True",
        "fragments:",
        "current raw (reused metadata)",
        "production batched raw (Rust index)",
        "legacy full group (includes open)",
        "production==legacy full group: True",
        "vectored raw (reused plan)",
        "coalesced full group + plan",
        "full group equality (including normal string/boolean reads): True",
    ):
        assert marker in res.stdout, f"missing {marker!r} in output:\n{res.stdout}"


def test_bench_scatter_gather_selection(mixed_file):
    res = _run_script(
        "bench_scatter_gather.py",
        [str(mixed_file), "DAQ", "--columns", "Counts", "--repeat", "1", "--warmup", "0"],
    )
    assert res.returncode == 0, res.stderr
    assert "numeric channels: 1; fragments: 2" in res.stdout
    assert "full group equality (including normal string/boolean reads): True" in res.stdout


def test_bench_scatter_gather_unequal_lengths(tmp_path):
    path = tmp_path / "unequal.tdms"
    with TdmsWriter(path) as writer:
        writer.write_segment([
            ChannelObject("G", "A", np.arange(3, dtype=np.int32)),
            ChannelObject("G", "B", np.arange(5, dtype=np.int32)),
        ])
    res = _run_script(
        "bench_scatter_gather.py",
        [str(path), "G", "--repeat", "1", "--warmup", "0"],
    )
    assert res.returncode == 0, res.stderr
    assert "coalesced: matches nptdms: True" in res.stdout
    assert "full group comparison: skipped (channels differ in length)" in res.stdout


def test_bench_index_creation_preserves_originals(mixed_file, tmp_path):
    path = tmp_path / mixed_file.name
    path.write_bytes(mixed_file.read_bytes())
    index = Path(f"{path}_index")
    index.write_bytes(b"existing index must not be changed")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    res = _run_script(
        "bench_index_creation.py",
        [str(tmp_path), "--repeat", "2", "--warmup", "1"],
    )
    assert res.returncode == 0, res.stderr
    for marker in (
        "first-index creation benchmark",
        "files: 1",
        "temporary copies only; original files and indexes preserved",
        "metadata scan, no index",
        "first open + index creation",
        "existing index open",
        "correctness: nptdms structure=True, metadata equality=True, verify_index=True",
    ):
        assert marker in res.stdout, f"missing {marker!r} in output:\n{res.stdout}"
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_bench_index_creation_file_and_workspace(mixed_file, tmp_path):
    res = _run_script(
        "bench_index_creation.py",
        [str(mixed_file), "--workspace", str(tmp_path), "--repeat", "1", "--warmup", "0"],
    )
    assert res.returncode == 0, res.stderr
    assert "first open + index creation" in res.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("extra,message", [
    (["--repeat", "0"], "repeat must be positive"),
    (["--warmup", "-1"], "warmup nonnegative"),
])
def test_bench_index_creation_invalid_counts(mixed_file, extra, message):
    res = _run_script("bench_index_creation.py", [str(mixed_file), *extra])
    assert res.returncode != 0
    assert message in res.stderr


def test_bench_index_creation_empty_folder(tmp_path):
    res = _run_script("bench_index_creation.py", [str(tmp_path)])
    assert res.returncode != 0
    assert "no TDMS files found" in res.stderr