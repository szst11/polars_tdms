"""A polars IO extension for reading TDMS files, backed by tdms-rs (Rust).

The heavy lifting (parsing the TDMS metadata and reading raw samples) happens
in the compiled ``polars_tdms._core`` extension. This module layers a polars
friendly API on top:

* :func:`read_metadata` -- read the full file metadata (groups/channels,
  properties, dtypes, sample counts) without touching raw data.
* :func:`scan_tdms` -- return a ``pl.LazyFrame`` that only reads the
  underlying channel data when collected.
* :func:`read_tdms` -- eagerly materialize a ``pl.DataFrame``.

Memory efficiency
-----------------
``tdms-rs`` indexes all segment metadata eagerly but never loads raw data.
``scan_tdms`` exploits that: constructing the lazy frame is O(metadata) only.
At execution time only the requested group/channels are read, in row batches
of at most ``chunk_size`` samples (default 1M). Streaming sinks can consume
these batches without materializing all selected data. ``collect()`` still
retains the complete result, and blocking query operations may buffer data.
"""

from __future__ import annotations

import datetime as _dt
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

import polars as pl
import pyarrow as pa
from polars.io.plugins import register_io_source

from . import _core

__version__ = _core.__version__

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "Channel",
    "Group",
    "TdmsMetadata",
    "TdmsSource",
    "open_tdms",
    "read_metadata",
    "read_tdms",
    "scan_tdms",
]

# Maximum number of samples per source batch.
DEFAULT_CHUNK_SIZE = 1_000_000

# TDMS "zero" time: the LabVIEW epoch used for timestamps.
_TDMS_EPOCH = _dt.datetime(1904, 1, 1, tzinfo=_dt.UTC)

#: TDMS DataType name (as reported by tdms-rs) -> polars dtype.
DTYPE_TO_POLARS: dict[str, type[pl.DataType] | pl.DataType] = {
    "Void": pl.Null,
    "I8": pl.Int8,
    "I16": pl.Int16,
    "I32": pl.Int32,
    "I64": pl.Int64,
    "U8": pl.UInt8,
    "U16": pl.UInt16,
    "U32": pl.UInt32,
    "U64": pl.UInt64,
    "Float": pl.Float32,
    "Double": pl.Float64,
    "Boolean": pl.Boolean,
    "String": pl.Utf8,
    "TimeStamp": pl.Datetime("us"),
}

#: Channel data types whose samples can be loaded into columns (all POD types
#: plus String). TimeStamp and Void remain metadata-only and are skipped when
#: building frames.
_READABLE_DTYPES = frozenset(
    {
        "I8",
        "I16",
        "I32",
        "I64",
        "U8",
        "U16",
        "U32",
        "U64",
        "Float",
        "Double",
        "Boolean",
        "String",
    }
)


def _convert_property(value: Any) -> Any:
    """Convert raw values returned by the Rust core into Python objects."""
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(x, int) for x in value)
    ):
        seconds, fraction = value
        return _TDMS_EPOCH + _dt.timedelta(seconds=seconds + fraction / (1 << 64))
    return value


def _convert_properties(props: Mapping[str, Any]) -> dict[str, Any]:
    return {k: _convert_property(v) for k, v in props.items()}


@dataclass(frozen=True)
class Channel:
    """A single TDMS channel and its metadata."""

    name: str
    dtype: str
    length: int
    properties: Mapping[str, Any]

    @property
    def polars_dtype(self) -> type[pl.DataType] | pl.DataType:
        return DTYPE_TO_POLARS[self.dtype]

    @property
    def readable(self) -> bool:
        return self.dtype in _READABLE_DTYPES


@dataclass(frozen=True)
class Group:
    """A TDMS group and its channels."""

    name: str
    properties: Mapping[str, Any]
    channels: tuple[Channel, ...]

    def channel(self, name: str) -> Channel | None:
        for c in self.channels:
            if c.name == name:
                return c
        return None

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.channels)


@dataclass(frozen=True)
class TdmsMetadata:
    """Full metadata of a TDMS file: properties, groups and channels."""

    path: str
    properties: Mapping[str, Any]
    groups: tuple[Group, ...]

    def group(self, name: str) -> Group | None:
        for g in self.groups:
            if g.name == name:
                return g
        return None

    @property
    def group_names(self) -> tuple[str, ...]:
        return tuple(g.name for g in self.groups)


@dataclass(frozen=True)
class _ChannelRequest:
    group: str
    channel: str
    name: str
    dtype: str
    length: int


def _build_metadata(path: str, handle: _core.TdmsHandle) -> TdmsMetadata:
    file_props = _convert_properties(handle.file_properties())
    groups: list[Group] = []
    for gname in handle.groups():
        info = handle.group_metadata(gname)
        group_props = _convert_properties(info["properties"])
        channels = tuple(
            Channel(
                name=c["name"],
                dtype=c["dtype"],
                length=int(c["len"]),
                properties=_convert_properties(c["properties"]),
            )
            for c in info["channels"]
        )
        groups.append(Group(name=gname, properties=group_props, channels=channels))
    return TdmsMetadata(path=path, properties=file_props, groups=tuple(groups))


def _select_nonempty_group(meta: TdmsMetadata, name: str) -> Group:
    g = meta.group(name)
    if g is None:
        raise ValueError(
            f"group {name!r} not found in {meta.path!r}; "
            f"available: {list(meta.group_names)}"
        )
    return g


def _resolve_channels(
    meta: TdmsMetadata,
    group: str | Sequence[str] | None,
    columns: Sequence[str] | None,
) -> list[_ChannelRequest]:
    """Resolve the requested group(s)/channels into concrete read requests."""
    if group is None:
        groups = list(meta.groups)
    elif isinstance(group, str):
        groups = [_select_nonempty_group(meta, group)]
    else:
        groups = [_select_nonempty_group(meta, g) for g in group]
    if not groups:
        raise ValueError(
            f"no groups found in {meta.path!r}; available: {list(meta.group_names)}"
        )

    multi = len(groups) > 1
    requests: list[_ChannelRequest] = []
    for g in groups:
        for c in g.channels:
            if not c.readable:
                continue
            out_name = f"{g.name}/{c.name}" if multi else c.name
            requests.append(
                _ChannelRequest(
                    group=g.name,
                    channel=c.name,
                    name=out_name,
                    dtype=c.dtype,
                    length=c.length,
                )
            )

    if columns is not None:
        wanted = set(columns)
        requests = [r for r in requests if r.name in wanted]
        missing = wanted - {r.name for r in requests}
        if missing:
            raise ValueError(f"columns not found: {sorted(missing)}")
    return requests


def _series_from_string_buffers(name: str, offsets: bytes, data: bytes, n: int) -> pl.Series:
    arr = pa.Array.from_buffers(
        pa.large_string(),
        n,
        [None, pa.py_buffer(offsets), pa.py_buffer(data)],
    )
    return pl.Series(name, arr)


def _arrow_dtype(name: str) -> Any:
    return {
        "I8": pa.int8(),
        "I16": pa.int16(),
        "I32": pa.int32(),
        "I64": pa.int64(),
        "U8": pa.uint8(),
        "U16": pa.uint16(),
        "U32": pa.uint32(),
        "U64": pa.uint64(),
        "Float": pa.float32(),
        "Double": pa.float64(),
        "Boolean": pa.bool_(),
    }[name]


def _series_from_numeric_buffers(name: str, dtype: str, data: bytes, n: int) -> pl.Series:
    arr = pa.Array.from_buffers(_arrow_dtype(dtype), n, [None, pa.py_buffer(data)])
    return pl.Series(name, arr)


def _read_requested_range(
    handle: _core.TdmsHandle,
    requests: Sequence[_ChannelRequest],
    start: int,
    end: int,
) -> pl.DataFrame:
    length = end - start
    numeric = [r for r in requests if r.length > 0 and r.dtype != "String"]
    batched: dict[str, pl.Series] = {}
    if len(numeric) > 1:
        channels = [(r.group, r.channel) for r in numeric]
        buffers = handle.read_channels_range_buffers(channels, start, end)
        batched = {
            req.name: _series_from_numeric_buffers(req.name, req.dtype, data, length)
            for req, data in zip(numeric, buffers, strict=True)
        }
    columns: list[pl.Series] = []
    for req in requests:
        if req.length == 0:
            series = pl.Series(
                req.name, [None], dtype=DTYPE_TO_POLARS[req.dtype]
            ).new_from_index(0, length)
        elif req.name in batched:
            series = batched[req.name]
        elif req.dtype == "String":
            offsets, data = handle.read_channel_strings_buffers(
                req.group, req.channel, start, end
            )
            series = _series_from_string_buffers(req.name, offsets, data, length)
        else:
            data = handle.read_channel_range_buffers(
                req.group, req.channel, start, end
            )
            series = _series_from_numeric_buffers(req.name, req.dtype, data, length)
        columns.append(series)
    return pl.DataFrame(columns)


class TdmsSource:
    """A lazily opened TDMS file.

    Opening only parses the metadata (the same work as :func:`read_metadata`);
    raw samples are read on demand. Use it as a context manager, or drop it to
    release the underlying file handle.

    The metadata index can come from a sibling ``<file>.tdms_index`` companion
    file (see the ``use_index_file`` / ``create_index_if_missing`` /
    ``verify_index`` parameters). Set ``copy_to_temp=True`` to read a temporary
    local copy of the TDMS and any existing companion index.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        use_index_file: bool = True,
        create_index_if_missing: bool = True,
        verify_index: bool = False,
        copy_to_temp: bool = False,
    ):
        self._path = os.fspath(path)
        self._temporary_directory = None
        self._access_path = self._path
        if copy_to_temp:
            self._temporary_directory = tempfile.TemporaryDirectory(
                prefix="polars_tdms_"
            )
            try:
                filename = os.path.basename(self._path)
                self._access_path = os.path.join(
                    self._temporary_directory.name, filename
                )
                shutil.copy2(self._path, self._access_path)
                source_index = f"{self._path}_index"
                if os.path.isfile(source_index):
                    shutil.copy2(source_index, f"{self._access_path}_index")
            except Exception:
                self._temporary_directory.cleanup()
                self._temporary_directory = None
                raise
        self._handle: _core.TdmsHandle | None = _core.TdmsHandle(
            self._access_path,
            use_index_file=use_index_file,
            create_index_if_missing=create_index_if_missing,
            verify_index=verify_index,
        )
        self._metadata: TdmsMetadata | None = None

    @property
    def path(self) -> str:
        return self._path

    @property
    def metadata(self) -> TdmsMetadata:
        if self._metadata is None:
            if self._handle is None:
                raise RuntimeError("TdmsSource has been closed")
            self._metadata = _build_metadata(self._path, self._handle)
        return self._metadata

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        self._handle = None
        self._temporary_directory = None

    def scan(
        self,
        group: str | Sequence[str] | None = None,
        columns: Sequence[str] | None = None,
        chunk_size: int | None = DEFAULT_CHUNK_SIZE,
    ) -> pl.LazyFrame:
        """Return a ``LazyFrame`` over the requested group/channels.

        The TDMS data is not read until the returned frame is executed.
        Thanks to projection pushdown, ``scan(...).select([...]).collect()``
        only loads the selected channels. Streaming sinks consume successive
        row batches without loading the complete selection into memory.

        Parameters
        ----------
        group:
            Group name, a list of group names, or ``None`` to merge all groups.
            When more than one group is selected, column names are prefixed
            with ``"GroupName/"`` to stay unique.
        columns:
            Optional subset of channels to load.
        chunk_size:
            Maximum samples per source batch; Polars may request smaller batches.
            Pass ``None`` or a nonpositive value to read the whole selection in
            one batch (disables bounded-memory source streaming).
        """
        meta = self.metadata
        if self._handle is None:
            raise RuntimeError("TdmsSource has been closed")
        handle = self._handle
        temporary_directory = self._temporary_directory
        requests = _resolve_channels(meta, group, columns)
        schema: dict[str, type[pl.DataType] | pl.DataType] = {
            r.name: DTYPE_TO_POLARS[r.dtype] for r in requests
        }
        if not schema:
            raise ValueError(f"no readable channel data in {self._path!r}")

        def _read_batches(
            with_columns: list[str] | None,
            predicate: pl.Expr | None,
            n_rows: int | None,
            batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            _ = temporary_directory
            by_name = {r.name: r for r in requests}
            selected = (
                requests if with_columns is None
                else [by_name[name] for name in with_columns]
            )
            nonempty_lengths = {r.length for r in selected if r.length > 0}
            if len(nonempty_lengths) > 1:
                details = ", ".join(f"{r.name}={r.length}" for r in selected)
                raise ValueError(
                    "requested channels have differing sample counts, which polars "
                    f"DataFrames cannot represent as columns of equal length: {details}"
                )
            target_length = next(
                iter(nonempty_lengths), max(r.length for r in requests)
            )
            if predicate is None and n_rows is not None:
                target_length = min(target_length, n_rows)
            if target_length == 0 or n_rows == 0:
                yield pl.DataFrame(
                    schema={r.name: DTYPE_TO_POLARS[r.dtype] for r in selected}
                )
                return
            size = target_length
            if chunk_size is not None and chunk_size > 0:
                size = chunk_size
                if batch_size is not None and batch_size > 0:
                    size = min(size, batch_size)
            remaining = n_rows
            for start in range(0, target_length, size):
                end = min(start + size, target_length)
                batch = (
                    _read_requested_range(handle, selected, start, end)
                    if selected else pl.DataFrame([{}] * (end - start))
                )
                if predicate is not None:
                    batch = batch.filter(predicate)
                if remaining is not None:
                    batch = batch.head(remaining)
                    remaining -= batch.height
                yield batch
                if remaining == 0:
                    return

        return register_io_source(
            _read_batches,
            schema=schema,
            validate_schema=True,
        )

    def read(
        self,
        group: str | Sequence[str] | None = None,
        columns: Sequence[str] | None = None,
        chunk_size: int | None = DEFAULT_CHUNK_SIZE,
    ) -> pl.DataFrame:
        """Eagerly build a :class:`pl.DataFrame` for the requested group/channels."""
        return self.scan(group=group, columns=columns, chunk_size=chunk_size).collect()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def open_tdms(
    path: str | os.PathLike[str],
    *,
    use_index_file: bool = True,
    create_index_if_missing: bool = True,
    verify_index: bool = False,
    copy_to_temp: bool = False,
) -> TdmsSource:
    """Open a TDMS file, indexing its metadata immediately.

    When a sibling ``<file>.tdms_index`` companion file exists (and
    ``use_index_file`` is true) it is used for the metadata index, so opening
    never scans the raw data. A missing, empty, stale, or corrupt index is
    skipped in favor of the data file and (when ``create_index_if_missing``)
    regenerated best-effort. ``verify_index`` re-parses both files and raises
    ``ValueError`` on mismatch. ``copy_to_temp`` copies the TDMS and any
    existing companion index into a temporary directory for local reads; the
    directory is removed when the source and its lazy scans are released.
    """
    return TdmsSource(
        path,
        use_index_file=use_index_file,
        create_index_if_missing=create_index_if_missing,
        verify_index=verify_index,
        copy_to_temp=copy_to_temp,
    )


def read_metadata(
    path: str | os.PathLike[str],
    *,
    use_index_file: bool = True,
    create_index_if_missing: bool = True,
    verify_index: bool = False,
    copy_to_temp: bool = False,
) -> TdmsMetadata:
    """Read the full TDMS metadata without loading any raw channel data.

    See :func:`open_tdms` for the ``use_index_file`` / ``create_index_if_missing``
    / ``verify_index`` / ``copy_to_temp`` options.
    """
    with TdmsSource(
        path,
        use_index_file=use_index_file,
        create_index_if_missing=create_index_if_missing,
        verify_index=verify_index,
        copy_to_temp=copy_to_temp,
    ) as src:
        return src.metadata


def scan_tdms(
    path: str | os.PathLike[str],
    group: str | Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    chunk_size: int | None = DEFAULT_CHUNK_SIZE,
    *,
    use_index_file: bool = True,
    create_index_if_missing: bool = True,
    verify_index: bool = False,
    copy_to_temp: bool = False,
) -> pl.LazyFrame:
    """Lazily scan a TDMS file as a ``pl.LazyFrame``.

    Only metadata is parsed up front; channel data is read when the frame is
    collected (see :meth:`TdmsSource.scan` for the parameters). See
    :func:`open_tdms` for the ``use_index_file`` / ``create_index_if_missing``
    / ``verify_index`` options.
    """
    return TdmsSource(
        path,
        use_index_file=use_index_file,
        create_index_if_missing=create_index_if_missing,
        verify_index=verify_index,
        copy_to_temp=copy_to_temp,
    ).scan(group=group, columns=columns, chunk_size=chunk_size)


def read_tdms(
    path: str | os.PathLike[str],
    group: str | Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    chunk_size: int | None = DEFAULT_CHUNK_SIZE,
    *,
    use_index_file: bool = True,
    create_index_if_missing: bool = True,
    verify_index: bool = False,
    copy_to_temp: bool = False,
) -> pl.DataFrame:
    """Eagerly read a TDMS group into a ``pl.DataFrame``.

    See :func:`open_tdms` for the ``use_index_file`` / ``create_index_if_missing``
    / ``verify_index`` / ``copy_to_temp`` options.
    """
    return scan_tdms(
        path,
        group=group,
        columns=columns,
        chunk_size=chunk_size,
        use_index_file=use_index_file,
        create_index_if_missing=create_index_if_missing,
        verify_index=verify_index,
        copy_to_temp=copy_to_temp,
    ).collect()
