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

This branch uses `tdms-rs` from the sibling `../tdms-rs` checkout, which adds the
read-only `TdmsChannel.data_locations()` API and optimized index generation.
Builds require that modified checkout. Once the parser changes are published,
replace the local path dependency in `Cargo.toml` with a pinned Git revision
before distributing wheels or source archives.

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

Initial index generation uses a bounded 4 MiB write buffer to avoid hundreds
of small writes for metadata-heavy files on high-latency storage. The index
format is unchanged; indexes smaller than the buffer are emitted in one final
write, and the final buffer is flushed before atomic publication.

## Benchmarks

All scripts live in `benchmarks/`. The synthetic script accepts `--skip-write`
to reuse a previously generated file instead of rewriting it.

| Script | Input | Measures |
|---|---|---|
| `bench_file_vs_nptdms.py` | an existing file + group (positional args) | pyarrow path vs nptdms, full/chunked/per-channel |
| `bench_tdms_index.py` | N synthetic files of identical structure, mixed content, many segments (~500 MB total) | vertical lazy-union build (scan + schema, no raw reads) with vs without `.tdms_index` |
| `bench_index_creation.py` | an existing file or folder of `.tdms` files | metadata scan vs first-index creation vs existing-index open on temporary copies |
| `bench_scatter_gather.py` | an existing file + group, optional `--columns` | experimental Rust scalar/vectored/coalesced reads vs current reader |

```bash
uv run --all-extras python benchmarks/bench_file_vs_nptdms.py /path/to/file.tdms DAQ

uv run --all-extras python benchmarks/bench_tdms_index.py \
    --n-files=100 --samples=200000 --segments=200
```

Scripts verify data values or metadata against nptdms and only report timings
once their correctness checks pass.

### First-index creation

```powershell
conda activate uv_rust
uv run --all-extras python benchmarks\bench_index_creation.py "F:\data\folder" --repeat 7
```

This benchmark preserves original files and companion indexes. Temporary copies
are created beside the inputs to measure the same filesystem (`--workspace`
overrides that location). Copying, cleanup, and correctness checks are excluded
from the reported best/median timings. Each first-open measurement starts with
no index, verifies that an index was actually generated, compares metadata with
unindexed and indexed opens, and checks `verify_index=True`. Group/channel names
and lengths are also checked against nptdms. Repeated reads use the OS cache;
these are not cold-storage measurements.

On three supplied ~23.63 MB files on `F:`, each producing a ~1.93 MB index,
7 measured opens after 1 warmup gave these median **scan + index creation**
times on temporary copies on the same drive:

| File suffix | Former 8 KiB write buffer | Bounded 4 MiB buffer |
|---|---:|---:|
| `179_2025618161517.tdms` | 493.7 ms | 216.3 ms |
| `180_2025618161517.tdms` | 480.4 ms | 185.4 ms |
| `182_2025618161517.tdms` | 547.4 ms | 224.8 ms |

That is approximately 2.3-2.6x faster for first-index creation. Scanning
metadata without writing an index remained around 50-55 ms. A 1 MiB-buffer
trial was also much faster than the default, but the filesystem timing
variation did not demonstrate a consistent additional benefit from 4 MiB.
The 4 MiB buffer lets these indexes be submitted as one final write while
keeping memory bounded for larger files. Existing-index opens and the
remaining filesystem create/flush/rename latency are not eliminated.

#### Position tracking and buffered skips

Metadata scans now track logical read positions in Rust instead of querying
the file position around every segment. Nearby seeks preserve the existing
read buffer when possible. This changes neither the 4 MiB index writer nor
the index format.

A paired comparison loaded the previous and optimized compiled extensions in
the same process, alternating their execution order on safe same-drive copies.
Both versions retained the 4 MiB writer. After 3 warmups, a 21-run confirmation
gave these medians:

| File suffix | Metadata scan, before / after | Existing index, before / after | First index, before / after |
|---|---:|---:|---:|
| `179_2025618161517.tdms` | 45.4 / 35.4 ms | 51.7 / 47.6 ms | 214.5 / 176.7 ms |
| `180_2025618161517.tdms` | 39.1 / 33.3 ms | 44.1 / 41.7 ms | 187.2 / 176.8 ms |
| `182_2025618161517.tdms` | 39.0 / 33.0 ms | 46.7 / 43.7 ms | 202.1 / 193.7 ms |

Across the initial 15-run comparison and this confirmation, median paired
speedups were 1.14-1.20x for metadata scans and 1.06-1.12x for indexed opens.
First-index creation was much noisier: the initial comparison gave paired
speedups of 1.03x, 0.95x, and 1.02x. The position optimization therefore helps
parsing consistently, but a substantial reduction in first-index creation
latency is not established. Metadata equality and `verify_index=True` passed
for every measured open.

#### Metadata read-ahead experiment

An isolated release/LTO build of the actual `tdms-rs` parser compared 8 KiB,
64 KiB, 256 KiB, 1 MiB, and 4 MiB metadata read buffers. Decoder and index-building
logic were unchanged; only buffer capacity and benchmark-only file I/O counters
varied. This experiment preceded the adaptive production policy below.

The first-observed pass randomly allocated 95 distinct files larger than 20 MB
from the supplied `F:` folder across the five capacities, with 19 files per
capacity. Index usage and writing were disabled:

| Read buffer | Median scan | Median read calls | Median seek calls | Median bytes fetched |
|---|---:|---:|---:|---:|
| 8 KiB | 668.1 ms | 1,067 | 717 | 8.73 MB |
| 64 KiB | 252.4 ms | 300 | 192 | 19.57 MB |
| 256 KiB | 221.3 ms | 81 | 76 | 20.94 MB |
| 1 MiB | 187.8 ms | 23 | 19 | 22.06 MB |
| 4 MiB | 199.5 ms | 7 | 4 | 22.14 MB |

The 1 MiB buffer reduced the median first-observed scan by about 72% (3.56x).
It fetched about 2.5x as many bytes to avoid hundreds of small reads and seeks.
The 4 MiB result did not improve on 1 MiB. These are file API counters, not SMB
request counts or measured network-wire throughput.

Repeated scans randomized capacity order on the three previously used files,
with 2 warmups and 13 measured repeats per file:

| Read buffer | Warm unindexed median | Warm existing-index median |
|---|---:|---:|
| 8 KiB | 63.2 ms | 57.0 ms |
| 64 KiB | 63.6 ms | 55.1 ms |
| 256 KiB | 62.9 ms | 56.3 ms |
| 1 MiB | 64.3 ms | 54.0 ms |
| 4 MiB | 65.8 ms | 59.7 ms |

Warm scans showed no comparable gain, consistent with parsing CPU cost becoming
dominant. All 485 timed opens matched the 8 KiB baseline's full Rust metadata,
including properties and raw-data locations. Original files, indexes, sizes,
and modification times were unchanged.

No shared caches were flushed: "first-observed" does not guarantee cold client
or server caches. First-pass capacities used different randomized files rather
than repeated cold reads of the same file. Timings measure Rust metadata open,
not Python conversion or index creation. The result favors bounded 1 MiB
read-ahead for this dense-segment network workload and motivated the adaptive
policy below, including sparse/large-file amplification tests.

#### Adaptive metadata read-ahead

The sibling `tdms-rs` parser now chooses read-ahead from segment layout:

- Start data-file metadata scans at 64 KiB.
- After four consecutive data-bearing segments with raw-data skips of at most
  64 KiB, use 1 MiB refills.
- On a larger gap, immediately reduce future refills to 8 KiB, seek past the
  gap, and require four nearby gaps before growing again.
- Ignore metadata-only segments and incomplete final segments when training
  the policy. Existing indexes use 1 MiB directly because their metadata is
  contiguous.

The allocation is bounded at 1 MiB per metadata scan. Only the refill limit
changes, so unread buffered bytes and logical positions survive transitions
without copying or repositioning the file. There is no drive-name detection,
timing-based tuning, or new Python option.

A temporary release/LTO benchmark used all 103 matching files in the supplied
network folder and `C:\temp\tmp`, plus both copies of the supplied
6,588,248,070-byte file in the network `test_data` folder and `C:\temp\large`.
Index writing was disabled throughout. Medians:

| Workload | Previous 8 KiB | Fixed 1 MiB | Adaptive |
|---|---:|---:|---:|
| Small files, network first pass | 708.7 ms | 134.7 ms | 137.5 ms |
| Small files, local first pass | 202.9 ms | 52.3 ms | 54.6 ms |
| Small files, network cached | 41.8 ms | 41.1 ms | 42.3 ms |
| Small files, local cached | 34.7 ms | 37.2 ms | 37.2 ms |
| Existing indexes, network cached | 36.1 ms | 37.3 ms | 39.4 ms |
| Existing indexes, local cached | 30.0 ms | 31.4 ms | 31.0 ms |
| Large file, network cached | 15.8 ms | 219.0 ms | 17.3 ms |
| Large file, local cached | 12.6 ms | 284.3 ms | 13.3 ms |

The first pass shuffled the 103 file pairs and assigned 35/34/34 pairs to the
three variants, rather than repeatedly reading one supposedly cold file.
Cached small-file/index measurements used three files per location, two
warmups and seven measured repeats with randomized variant order. Large-file
cached measurements used five randomized repeats per location. The fixed
1 MiB variant happened to run first on each large-file copy; its initial
3,432 ms network / 698 ms local times are not fair cold-cache comparisons
against the later variants and are excluded from the table.

For the fragmented small files, adaptive first-pass medians were about 5.15x
faster on the network and 3.71x locally. Typical read/seek counts fell from
1,067/717 to 27/23, at the cost of fetching about 22.1 MB instead of 8.7 MB.
On the large file, 1,098 raw-data gaps exceeded 64 KiB, with a maximum around
12.2 MB. Blind 1 MiB read-ahead fetched 1,125.1 MB; adaptive fetched 10.2 MB,
close to the previous reader's 9.1 MB. Adaptation therefore avoids the severe
sparse-file amplification, but retains a small overhead relative to 8 KiB.
Cached scans and indexed opens did not show a consistent speedup and some
were modestly slower; the policy is a latency/read-amplification compromise,
not a guarantee of the globally fastest buffer.

All 494 timed opens matched full baseline Rust metadata, including properties
and raw locations; matching network/local copies also agreed. Fifteen parser
source files, including decoder and metadata-building logic, were unchanged
in the benchmark snapshot; only private benchmark hooks/counters differed.
The rebuilt Python extension also matched nptdms on 48 first/middle/last
numeric and string sample ranges covering all four supplied locations.
Original file/index inventories, sizes and modification times were unchanged.
Counters are file API reads/seeks and returned bytes, not SMB requests or
network-wire bandwidth. No shared caches were flushed, the small network
files had been used in earlier experiments, and "first pass" does not imply
cold client or server caches. These timings exclude Python conversion and
index serialization/publication.

#### Object-ID reuse

The production sibling parser now interns object paths once per file, reuses
integer IDs for active raw metadata and object order, and caches resolved
group/channel targets. Segment objects share canonical paths instead of
rebuilding their strings and components on every appearance. This applies to
both TDMS metadata scans and existing-index opens. The TDMS/index format and
property/update semantics are unchanged. Streaming model construction is
not included.

A paired comparison loaded the previous and rebuilt production extensions in
the same process. It used all 103 files in the supplied network folder and
their local copies in `C:\temp\tmp`, with existing indexes, writing disabled,
one warmup and three measured randomized before/after repeats per file:

| Location | Previous: 103 indexed opens | Object-ID reuse | Speedup |
|---|---:|---:|---:|
| Network | 3.697 s | 1.377 s | 2.68x |
| Local | 2.838 s | 0.741 s | 3.83x |

Folder totals sum individual constructor times and report the median of the
three repeats. They include native parsing, model construction and temporary
segment cleanup, but exclude Python metadata conversion and comparison.
Caches were not flushed; these are warm measurements, not cold-network claims.
All 1,236 timed opens matched the previous extension's exposed metadata,
properties, sample counts, dtypes and group/channel ordering. Original data
files and index inventories, sizes and modification times were unchanged.

The Rust regression generates a mixed file at runtime, checking reordered and
late-added objects, metadata-only property updates, raw-index reuse,
continuations, varied fragment sizes, padding, quoted slashes in paths, empty
channels, six data types, exact fragment offsets and cross-fragment ranges.
It covers direct reads, generated/verified indexes, and forced index-only
reads. An independent nptdms read of the generated file also matched all
30,482 samples across its nine channels.

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
