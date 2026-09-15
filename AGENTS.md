# polars-tdms

Hybrid Rust/Python project: an NI TDMS reader for polars. A PyO3 extension (Rust, `tdms-rs`) does the parsing; `src/polars_tdms/__init__.py` layers a polars-native API on top. No CI, lint, or typecheck config exists.

## Build & test

```bash
uv sync --all-extras        # build extension + install dev deps (Rust ≥ 1.75 required)
uv run --all-extras python -m pytest -q    # 23 tests (tests/test_tdms.py, tests/test_benchmarks.py)
```

- Use `python -m pytest`, not the bare `pytest` binary — the direct spawn fails under `uv run` in this repo.
- All test TDMS files are generated at runtime by nptdms; nothing is checked in. `tests/test_benchmarks.py` additionally shells out to the benchmark scripts with tiny inputs (asserts markers in their stdout).
- After editing `src/lib.rs`, re-run `uv sync` to rebuild the compiled module (`polars_tdms._core`). `Cargo.lock`, `/target/`, and the built `*.so` are `.gitignore`d.

## Benchmarks (`benchmarks/`)

Three scripts; all verify polars_tdms values against nptdms before reporting timings. `--skip-write` exists on the two synthetic scripts to reuse a generated file.

```bash
uv run --extra bench python benchmarks/bench_vs_nptdms.py --samples=4000000 --channels=8 --path=/tmp/bench.tdms
uv run --all-extras python benchmarks/bench_vs_numpy_fallback.py --segments=200 --samples-per-segment=100000
uv run --all-extras python benchmarks/bench_file_vs_nptdms.py /path/to/file.tdms DAQ
```

- `bench_vs_nptdms.py` — synthetic f64 file; metadata/full/partial reads + peak RSS (RSS measured in a fresh subprocess via `benchmarks/_rss_runner.py`). Its metadata row is printed in **ms**.
- `bench_vs_numpy_fallback.py` — synthetic mixed-type file (f64/i32/bool/string, many segments); tables the pyarrow/"buffers" fast path against the numpy fallback and nptdms. Needs pyarrow (`--all-extras`).
- `bench_file_vs_nptdms.py` — positional `FILE GROUP` args; pyarrow path vs nptdms per channel. `--columns` restricts channels, `--chunk-size 0` disables the chunked row, exits 1 on a missing file/group.

## Architecture & constraints worth knowing

- Rust core (`src/lib.rs`, crate `polars-tdms-core`) exposes `TdmsHandle`; all channel reads cross the boundary as numpy arrays via `read_channel_range(group, channel, start, end)` (end-exclusive). Numeric channels default to `read_channel_range_buffers` → raw little-endian bytes (Boolean as a packed LSB bitmap) → pyarrow `Array.from_buffers` → `pl.Series`, no numpy; falls back to the numpy path when pyarrow is unavailable.
- String/TimeStamp channel *data*: TimeStamp is metadata-only — tdms-rs 2.x can't decode it, gated by `_READABLE_DTYPES` in `__init__.py`. Reading one from Rust raises `NotImplementedError`; reading only these → `ValueError: no readable channel data`. String channels *are* readable via `read_channel_strings` (returns `Vec<String>` → Python list → `pl.Series`, no numpy). A second path uses `read_channel_strings_buffers` → `(i64-LE offsets, UTF-8 bytes)` consumed by pyarrow `Array.from_buffers` → `pl.Series` (zero-copy beyond the PyBytes copy in `_core`); falls back to the list path when pyarrow is unavailable, requires pyarrow ≥ 15 in deps. The tdms-rs upstream API behind it is `read_string_buffers(range, &mut Vec<u32>, &mut Vec<u8>)` (Cursor: cumulative offsets + one contiguous UTF-8 block, Arrow string layout).
- TimeStamp *properties* arrive from Rust as `(seconds, fraction)` tuples (fraction in 2^64ths); `_convert_property` turns them into datetimes (TDMS epoch 1904-01-01). Keep that conversion on the Python side.
- Lazy reads use `placeholder.lazy().map_batches(...)` with `projection_pushdown=True`, `predicate_pushdown=False`, `slice_pushdown=False`, `streamable=False`, `validate_output_schema=False`. Projection pushdown is behavior-tested (`test_projection_pushdown_only_loads_selected` uses a spy handle) — preserve it.
- `validate_output_schema` requires polars ≥ 1.33 (pinned via `polars-lts-cpu>=1.33.1`).
- `group=None` merges all groups and prefixes columns as `GroupName/Name`; a single group keeps unprefixed names. Multi-group frames must pass the prefix to `columns`.
- Channels of differing sample counts raise `ValueError` (polars columns must be equal length); `chunk_size=None` reads a whole channel in one allocation, chunked reads concat with `rechunk=False` and must equal the whole read.
- Python is locked to 3.13 (`.python-version`, `requires-python = ">=3.13"`).