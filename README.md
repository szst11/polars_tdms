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

## Benchmarks

All scripts live in `benchmarks/`. The two synthetic scripts accept
`--skip-write` to reuse a previously generated file instead of rewriting it.

| Script | Input | Measures |
|---|---|---|
| `bench_vs_nptdms.py` | synthetic f64 file | metadata, full/partial reads and peak RSS vs nptdms |
| `bench_vs_nptdms_mixed.py` | synthetic mixed-type file (Float64/Int32/Boolean/String, many segments) | pyarrow read path vs nptdms, full group and per-channel |
| `bench_file_vs_nptdms.py` | an existing file + group (positional args) | pyarrow path vs nptdms, full/chunked/per-channel |

```bash
uv run --extra bench python benchmarks/bench_vs_nptdms.py \
    --samples=4000000 --channels=8 --path=/tmp/bench.tdms

uv run --all-extras python benchmarks/bench_vs_nptdms_mixed.py \
    --segments=200 --samples-per-segment=100000

uv run --all-extras python benchmarks/bench_file_vs_nptdms.py /path/to/file.tdms DAQ
```

Each script first verifies polars_tdms values against nptdms and only reports
timings once the read is known to be correct.

### Results (bench_vs_nptdms.py, 8 f64 channels)

Synthetic 64 MiB and 256 MiB TDMS files, machine timed on the host. RSS
measured as the resident set of a fresh subprocess (after the read, the result
is kept alive).

#### 1 M samples × 8 channels (64 MiB data)

```
benchmark                                     nptdms   polars_tdms   ratio
--------------------------------------------------------------
metadata (groups/channels/properties)         49.26ms      0.06ms    790x
full group read                               0.071s      0.033s    2.1x
lazy scan_tdms().collect()                        —        0.033s      —
partial read (2 channels)                     0.070s      0.009s    8.2x
peak RSS (full read)                          188 MiB     167 MiB    1.1x
peak RSS (partial read)                       188 MiB     120 MiB    1.6x
```

#### 4 M samples × 8 channels (256 MiB data)

```
benchmark                                     nptdms   polars_tdms   ratio
--------------------------------------------------------------
metadata (groups/channels/properties)        160.89ms      0.10ms    1609x
full group read                               0.264s      0.129s     2.1x
lazy scan_tdms().collect()                        —        0.128s       —
partial read (2 channels)                     0.269s      0.033s     8.2x
peak RSS (full read)                          371 MiB     359 MiB     1.0x
peak RSS (partial read)                       371 MiB     168 MiB     2.2x
```

The metadata row is reported in milliseconds. Numbers are from a host run and
will vary by machine and file layout.

## Architecture

| Layer | Location | Purpose |
|---|---|---|
| Rust core (`_core`) | `src/lib.rs` | PyO3 extension: opens `TdmsFile`, reads channels as raw Arrow-layout byte buffers. |
| Python API | `src/polars_tdms/__init__.py` | Lazy nodes via `map_batches`, metadata dataclasses, chunked reads. |

Numeric/Boolean and String reads use zero-copy raw-byte (<code>read_channel_range_buffers</code> / <code>read_channel_strings_buffers</code>) paths built into `pl.Series` via pyarrow (`Array.from_buffers`); pyarrow is a required dependency.

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
