# polars-tdms

Read NI TDMS files into [polars](https://pola.rs/) DataFrames and LazyFrames.
Built on the [tdms-rs](https://crates.io/crates/tdms-rs) Rust crate via a
PyO3 extension, with the Python layer adding a polars-native API.

## Install

```bash
uv sync --all-extras          # development + benchmark deps
uv sync                       # library only (requires Rust toolchain)
```

Rust ≥ 1.75 required at build time.

## Quick start

```python
import polars_tdms as pt

# read full file metadata (cached on first open)
meta = pt.read_metadata("acquisition.tdms")
print(meta.groups)          # [Group(name='...', channels=[...], ...), ...]

# eager read → pl.DataFrame
df = pt.read_tdms("acquisition.tdms", group="DAQ")
df.head()

# lazy read → pl.LazyFrame  (projection / filter / slice pushed down)
lf = pt.scan_tdms("acquisition.tdms", group="DAQ", columns=["Ch0", "Ch1"])
df = lf.filter(pl.col("Ch0") > 0.5).collect()

# read specific channels only
df = pt.read_tdms("acquisition.tdms", group="DAQ", columns=["Ch0", "Ch1"])

# scan with lazy selection that merges multiple groups
lf = pt.scan_tdms("acquisition.tdms")
df = lf.select("DAQ/Ch0", "DAQ/Ch1").collect()
```

### Supported channel types

Sample data: Float32, Float64, Int8–Int64, UInt8–UInt64, Boolean, String.
TimeStamp channels are exposed in the metadata (properties converted to
`datetime`) but tdms-rs 2.x does not decode their sample data, so they are
skipped when building frames.

## API

| Function | Returns | Description |
|---|---|---|
| `read_metadata(path)` | `TdmsMetadata` | Groups, channels and properties. |
| `scan_tdms(path, group, columns, chunk_size)` | `pl.LazyFrame` | Lazy read; supports projection pushdown. |
| `read_tdms(path, group, columns, chunk_size)` | `pl.DataFrame` | Eager read in chunks (default 1 M samples/chunk). |
| `open_tdms(path)` | `TdmsSource` | Context manager for manual access to group/channel data. |

`group=None` merges all groups (channels prefixed with `GroupName/`).

### `.tdms_index` companion files

Every entry point also accepts `use_index_file=True`, `create_index_if_missing=True`,
`verify_index=False` (matching `tdms-rs` `OpenOptions`) and `copy_to_temp=False`
(read a temporary copy of the file, optionally with its index). When a sibling
`<file>.tdms_index` exists it is used to build the metadata index quickly, so
opening a large file scans the small index instead of the raw data. A missing,
empty, stale, or corrupt index falls back to the data file and — unless
`create_index_if_missing=False` — is regenerated best-effort. `verify_index=True`
re-parses both files and raises `ValueError` on a mismatch.

## Benchmarks

All scripts live in `benchmarks/`. The synthetic script accepts `--skip-write`
to reuse a previously generated file instead of rewriting it.

| Script | Input | Measures |
|---|---|---|
| `bench_file_vs_nptdms.py` | an existing file + group (positional args) | pyarrow path vs nptdms, full/chunked/per-channel |
| `bench_tdms_index.py` | N synthetic files of identical structure, mixed content, many segments (~500 MB total) | vertical lazy-union build (scan + schema, no raw reads) with vs without `.tdms_index` |

```bash
uv run --all-extras python benchmarks/bench_file_vs_nptdms.py /path/to/file.tdms DAQ

uv run --all-extras python benchmarks/bench_tdms_index.py \
    --n-files=100 --samples=200000 --segments=200
```

Each script first verifies polars_tdms values against nptdms and only reports
timings once the read is known to be correct.

## Architecture

| Layer | Location | Purpose |
|---|---|---|
| Rust core (`_core`) | `src/lib.rs` | PyO3 extension: opens `TdmsFile`, reads channels as raw Arrow-layout byte buffers. |
| Python API | `src/polars_tdms/__init__.py` | Lazy nodes via `map_batches`, metadata dataclasses, chunked reads. |


Numeric/Boolean and String reads use zero-copy raw-byte (<code>read_channel_range_buffers</code> / <code>read_channel_strings_buffers</code>) paths built into `pl.Series` via pyarrow (`Array.from_buffers`). pyarrow is a hard dependency, so there is no fallback path.

### Zero-copy read path

Channel samples cross into polars without an intermediate NumPy array being
built:

- The Rust core hands back buffers that already match Arrow's memory layout:
  little-endian scalars for numeric channels (`read_channel_range_buffers`), a
  packed LSB-first bitmap for Boolean, and for String channels (`read_channel_strings_buffers`)
  a pair of buffers — cumulative `i64` offsets plus one contiguous UTF-8 block —
  which is exactly the payload of an Arrow `LargeUtf8` array.
- pyarrow wraps those buffers with `Array.from_buffers` and polars adopts the
  resulting arrays directly; the only copies are the `PyBytes` marshalling
  inside `_core` and pyarrow's import.
- Chunked reads concatenate the per-chunk series with `rechunk=False`, so a
  multi-GB channel streams into the frame without ever allocating a full-size
  intermediate array.

LazyFrame nodes use `validate_output_schema=False` (polars 1.33+); projection
pushdown is verified (only requested columns are read from the file).

## License

MIT
