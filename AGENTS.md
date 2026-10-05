# polars-tdms

Hybrid Rust/Python project: an NI TDMS reader for polars. A PyO3 extension (Rust, `tdms-rs`) does the parsing; `src/polars_tdms/__init__.py` layers a polars-native API on top. No CI, lint, or typecheck config exists.

## Build & test

```bash
uv sync --all-extras        # build extension + install dev deps (Rust ≥ 1.75 required)
uv run --all-extras python -m pytest -q    # 34 tests (tests/test_tdms.py, tests/test_benchmarks.py)
```

- Use `python -m pytest`, not the bare `pytest` binary — the direct spawn fails under `uv run` in this repo.
- All test TDMS files are generated at runtime by nptdms; nothing is checked in. `tests/test_benchmarks.py` additionally shells out to the benchmark scripts with tiny inputs (asserts markers in their stdout).
- Every benchmark script must keep at least one smoke test asserting on markers unique to *it*. One script was once silently replaced by a byte-for-byte copy of another benchmark, which nothing caught; each script's markers are now asserted individually.
- After editing `src/lib.rs`, re-run `uv sync` to rebuild the compiled module (`polars_tdms._core`). Plain `uv sync` no-ops when only Rust changed — use `uv sync --all-extras --reinstall-package polars-tdms`. `Cargo.lock`, `/target/`, and the built `*.so` are `.gitignore`d.

## Benchmarks (`benchmarks/`)

Two scripts; all verify polars_tdms values against nptdms before reporting timings. `bench_tdms_index.py` is the synthetic one and has `--skip-write` to reuse a generated file.

```bash
uv run --all-extras python benchmarks/bench_file_vs_nptdms.py /path/to/file.tdms DAQ
uv run --all-extras python benchmarks/bench_tdms_index.py --n-files=100 --samples=200000 --segments=200
```

- `bench_file_vs_nptdms.py` — positional `FILE GROUP` args; pyarrow path vs nptdms per channel. `--columns` restricts channels, exits 1 on a missing file/group; the chunked-vs-whole comparison row uses a fixed `chunk_size=10_000` and is skipped when channels differ in length.
- `bench_tdms_index.py` — N synthetic files of identical structure but mixed content, each split across `--segments` segments (~500 MB total at defaults); times building a vertical lazy union (`scan_tdms` + `pl.concat` + `collect_schema`, no raw reads) without `.tdms_index` sidecars, then after generating them, and reports the speedup. `--path` holds the generated files (default `/tmp/opencode/bench_index`).


## Architecture & constraints worth knowing

- Rust core (`src/lib.rs`, crate `polars-tdms-core`) exposes `TdmsHandle`; channel reads cross the boundary as raw little-endian bytes via `read_channel_range_buffers(group, channel, start, end)` (end-exclusive, Boolean as a packed LSB bitmap) → pyarrow `Array.from_buffers` → `pl.Series`, with no intermediate NumPy array. `read_channel_strings_buffers` is the String equivalent. pyarrow is a hard dependency (`pyarrow>=15.0`) and there is no fallback path.
- String/TimeStamp channel *data*: TimeStamp is metadata-only — tdms-rs 2.x can't decode it, gated by `_READABLE_DTYPES` in `__init__.py`. Reading one from Rust raises `NotImplementedError`; reading only these → `ValueError: no readable channel data`. String channels are read via `read_channel_strings_buffers` → `(i64-LE offsets, UTF-8 bytes)` consumed by pyarrow `Array.from_buffers` → `pl.Series` (zero-copy beyond the PyBytes copy in `_core`). The tdms-rs upstream API behind it is `read_string_buffers(range, &mut Vec<u64>, &mut Vec<u8>)` (cumulative 64-bit offsets + one contiguous UTF-8 block, Arrow string layout).
- TimeStamp *properties* arrive from Rust as `(seconds, fraction)` tuples (fraction in 2^64ths); `_convert_property` turns them into datetimes (TDMS epoch 1904-01-01). Keep that conversion on the Python side.
- Lazy reads use `placeholder.lazy().map_batches(...)` with `projection_pushdown=True`, `predicate_pushdown=False`, `slice_pushdown=False`, `streamable=False`, `validate_output_schema=False`. Projection pushdown is behavior-tested (`test_projection_pushdown_only_loads_selected` uses a spy handle) — preserve it.
- `validate_output_schema` requires polars ≥ 1.33 at runtime. `pyproject.toml` depends on unpinned `polars` (the `polars-lts-cpu` line is commented out), so the resolver does not enforce this.
- `group=None` merges all groups and prefixes columns as `GroupName/Name`; a single group keeps unprefixed names. Multi-group frames must pass the prefix to `columns`.
- Channels of differing sample counts raise `ValueError` (polars columns must be equal length); `chunk_size=None` reads a whole channel in one allocation, chunked reads concat with `rechunk=False` and must equal the whole read.
- Python is locked to 3.13 (`.python-version`, `requires-python = ">=3.13"`).
