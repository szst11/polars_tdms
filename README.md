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

Float32, Float64, Int8–Int64, UInt8–UInt64, Boolean, String, Timestamp (converted to
`datetime`).

## API

| Function | Returns | Description |
|---|---|---|
| `read_metadata(path)` | `TdmsMetadata` | Groups, channels and properties. |
| `scan_tdms(path, group, columns, chunk_size)` | `pl.LazyFrame` | Lazy read; supports projection pushdown. |
| `read_tdms(path, group, columns, chunk_size)` | `pl.DataFrame` | Eager read in chunks (default 1 M samples/chunk). |
| `open_tdms(path)` | `TdmsSource` | Context manager for manual access to group/channel data. |

`group=None` merges all groups (channels prefixed with `GroupName/`).

## Benchmark

Synthetic 64 MiB and 256 MiB TDMS files (`f64`, 8 channels), machine
timed on the host. RSS measured as resident set of a fresh subprocess
(after the read, result kept alive).

### 1 M samples × 8 channels  (64 MiB data)

```
benchmark                                     nptdms  polars_tdms  ratio
---------------------------------------------------------------
metadata (full file parse)                   0.051s      0.000s   873x
full group read                              0.089s      0.034s   2.6x
lazy scan_tdms().collect()                       —       0.032s      —
partial read (2 channels)                    0.090s      0.009s   9.8x
peak RSS (full read)                         150 MiB     130 MiB  1.2x
peak RSS (partial read)                      150 MiB      84 MiB  1.8x
```

### 4 M samples × 8 channels  (256 MiB data)

```
benchmark                                     nptdms  polars_tdms  ratio
---------------------------------------------------------------
metadata (full file parse)                   0.161s      0.000s   2623x
full group read                              0.339s      0.125s   2.7x
lazy scan_tdms().collect()                       —       0.123s      —
partial read (2 channels)                    0.299s      0.032s   9.3x
peak RSS (full read)                         333 MiB     313 MiB  1.1x
peak RSS (partial read)                      333 MiB     130 MiB  2.6x
```

Run the benchmark yourself:

```bash
uv run --extra bench python benchmarks/bench_vs_nptdms.py \
    --samples=4000000 --channels=8 --path=/tmp/bench.tdms
```

## Architecture

| Layer | Location | Purpose |
|---|---|---|
| Rust core (`_core`) | `src/lib.rs` | PyO3 extension: opens `TdmsFile`, reads channels into numpy arrays. |
| Python API | `src/polars_tdms/__init__.py` | Lazy nodes via `map_batches`, metadata dataclasses, chunked reads. |

LazyFrame nodes use `validate_output_schema=False` (polars 1.33+); projection
pushdown is verified (only requested columns are read from the file).

## License

MIT
