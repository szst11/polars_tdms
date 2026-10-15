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

This branch temporarily patches `tdms-rs` to the sibling `../tdms-rs` checkout,
which adds the read-only `TdmsChannel.data_locations()` API. Builds require that
checkout with the API change. Once the parser change is published, update the
pinned Git revision in `Cargo.toml` and remove the local `[patch]` section before
distributing wheels or source archives.

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
| `bench_scatter_gather.py` | an existing file + group, optional `--columns` | experimental Rust scalar/vectored/coalesced reads vs current reader |

```bash
uv run --all-extras python benchmarks/bench_file_vs_nptdms.py /path/to/file.tdms DAQ

uv run --all-extras python benchmarks/bench_tdms_index.py \
    --n-files=100 --samples=200000 --segments=200
```

Each script first verifies polars_tdms values against nptdms and only reports
timings once the read is known to be correct.

### Scatter-gather experiment

```powershell
conda activate uv_rust
uv run --all-extras python benchmarks\bench_scatter_gather.py "F:\data\file.tdms" DAQ --repeat 7
```

For its experimental rows, this benchmark builds a numeric
fragment plan from nptdms private metadata, then compares disk-order scalar
reads, contiguous runs of `File::read_vectored` (up to 64 buffers), and bounded
coalesced reads followed by scatter copies (8 MiB spans, gaps up to 64 KiB,
at most 2x read amplification).
Vectored I/O scatters a **contiguous file range** into multiple destinations;
it cannot gather arbitrary file offsets. On Windows the standard library may
read only the first buffer per call, so vectored reads need not reduce read
calls. Counters measure Rust read/seek API calls, not traced OS syscalls.

Only complete, little-endian, non-interleaved, non-DAQmx segments are supported
by the experiment; other layouts fail explicitly. Strings and Boolean still
use the production reader in full-group comparisons. Numeric values are checked
exactly against nptdms (including NaNs), and full groups against the current API.
The benchmark reports best/median warm-cache times, separately for raw reads,
full-group reads using a reusable plan, and full-group reads including plan
construction. It also times the production batched reader using the existing
Rust index, and the former scalar reader via a private benchmark reference.
Full-group timings include opening and metadata parsing; production and legacy
frames must match before timing. It does not create or modify companion indexes.
The nptdms reusable-plan rows remain experimental, not end-to-end measurements.

The initial experiment, before production integration, measured a Windows
23.65 MB / 2,735-segment file, a group containing
44 numeric channels plus one string channel, and 37,312 numeric fragments
(9 measured runs after 2 warmups):

| Read path | Raw numeric median | Full-group median (including string) |
|---|---:|---:|
| Former scalar reader | 132.9 ms | 211.3 ms |
| Disk-order scalar, reused plan | 73.0 ms | 158.6 ms |
| Vectored, reused plan | 68.2 ms | 126.3 ms |
| Coalesced, reused plan | 29.1 ms | 86.8 ms |
| Coalesced, including plan construction | - | 249.8 ms |

Vectored reads reduced seeks from 37,312 to 1,696 but still issued 37,312 read
calls on this Windows host. Coalescing issued 3 reads / 3 seeks, reading
23.17 MB instead of the 20.15 MB selected numeric payload. The measured
reused-plan gains therefore favor coalescing, not Windows vectored syscall
batching. Including nptdms planning made that prototype slower than the former API.
These are warm-cache measurements on one machine, not cold-storage guarantees.

After production integration, the same file's `Messungen` group was measured
with 9 runs after 2 warmups, comparing both paths in the same process:

| Production comparison | Former scalar reader | Coalesced reader |
|---|---:|---:|
| Numeric raw reads, reused Rust metadata | 136.3 ms | 29.9 ms |
| Full group, including open and string reads | 189.8 ms | 88.8 ms |

The full-group improvement is about 2.1x end-to-end, without nptdms plan
construction. With the read-amplification cap enabled, coalescing used 5 bounded
reads for the numeric payload. All five groups in the file matched nptdms
exactly, including chunked reads and a lazy two-channel projection.

## Architecture

| Layer | Location | Purpose |
|---|---|---|
| Rust core (`_core`) | `src/lib.rs` | PyO3 extension: opens `TdmsFile`, reads channels as raw Arrow-layout byte buffers. |
| Python API | `src/polars_tdms/__init__.py` | Lazy nodes via `map_batches`, metadata dataclasses, chunked reads. |


Numeric/Boolean and String reads use raw-byte (<code>read_channels_range_buffers</code> / <code>read_channel_range_buffers</code> / <code>read_channel_strings_buffers</code>) paths built into `pl.Series` via pyarrow (`Array.from_buffers`). pyarrow is a hard dependency, so there is no fallback path.

### Coalesced production reads

Eager reads, collected lazy scans, and `TdmsSource.read()` batch the selected
numeric/Boolean channels for each sample chunk. Rust borrows the parser's
existing fragment index, sorts requested ranges into disk order, reads bounded
contiguous spans, and scatters them into per-channel Arrow buffers. There is no
second metadata parser or nptdms runtime dependency. Single-channel reads use
the same coalescing engine. Strings retain the existing string-buffer reader.

Coalesced scratch spans are limited to 8 MiB, gaps to 64 KiB, and read
amplification to 2x selected raw bytes so narrow projections do not repeatedly
read the entire file. A single fragment larger than 8 MiB is read directly into
its destination without a scratch allocation. Big-endian fragments are
converted to Arrow little-endian bytes; Boolean samples are packed LSB-first.

Only projected channels receive buffers or are decoded, although coalesced
spans may read skipped gap bytes. `chunk_size` still limits samples per channel
per batch, and concatenation retains `rechunk=False`. Transient memory includes
the selected channels' chunk buffers plus at most 8 MiB of scratch, in addition
to their Arrow/Python buffers. Invalid ranges, incomplete indexed ranges, and
I/O failures raise rather than silently returning partial results.
This integration uses the parser's existing supported layouts; it does not
extend its TDMS format support. The nptdms-only experimental planner can index
repeated raw chunks within a segment, which the pinned parser does not yet
fully index.

### Zero-copy read path

Channel samples cross into polars without an intermediate NumPy array being
built:

- The Rust core hands back buffers that already match Arrow's memory layout:
  little-endian scalars for numeric channels (`read_channels_range_buffers`), a
  packed LSB-first bitmap for Boolean, and for String channels (`read_channel_strings_buffers`)
  a pair of buffers — cumulative `i64` offsets plus one contiguous UTF-8 block —
  which is exactly the payload of an Arrow `LargeUtf8` array.
- pyarrow wraps those buffers with `Array.from_buffers` and polars adopts the
  resulting arrays directly. Coalesced reads copy from their bounded scratch
  span into channel destinations, followed by `PyBytes` marshalling inside
  `_core` and pyarrow's import.
- Chunked reads concatenate the per-chunk series with `rechunk=False`, so a
  multi-GB channel streams into the frame without ever allocating a full-size
  intermediate array.

LazyFrame nodes use `validate_output_schema=False` (polars 1.33+); projection
pushdown is verified (only requested columns are materialized).

## License

MIT
