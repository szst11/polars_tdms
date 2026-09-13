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
At ``collect()`` time only the requested group/channels are read, and large
channels are streamed in chunks of ``chunk_size`` samples (default 1M) so the
peak memory footprint stays bounded even for multi-GB files.
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import polars as pl

from . import _core

__version__ = _core.__version__

__all__ = [
    "Channel",
    "DEFAULT_CHUNK_SIZE",
    "Group",
    "TdmsMetadata",
    "TdmsSource",
    "open_tdms",
    "read_tdms",
    "read_metadata",
    "scan_tdms",
]

# Number of samples read per chunk when materializing a large channel.
DEFAULT_CHUNK_SIZE = 1_000_000

# TDMS "zero" time: the LabVIEW epoch used for timestamps.
_TDMS_EPOCH = _dt.datetime(1904, 1, 1)

#: TDMS DataType name (as reported by tdms-rs) -> polars dtype.
DTYPE_TO_POLARS: dict[str, pl.DataType] = {
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

#: Channel data types whose samples can be loaded into columns (all POD types).
#: String / TimeStamp channels are exposed in the metadata but their raw data
#: is not decompressed by tdms-rs 2.x, so they are skipped when building frames.
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
    }
)


def _convert_property(value: Any) -> Any:
    """Convert raw values returned by the Rust core into Python objects."""
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(x, int) for x in value):
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
    def polars_dtype(self) -> pl.DataType:
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


def _series_from_numpy(name: str, arr: Any) -> pl.Series:
    return pl.Series(name, arr, strict=False)


def _read_channel_series(
    handle: _core.TdmsHandle, req: _ChannelRequest, chunk_size: int | None
) -> pl.Series:
    if req.length == 0:
        return pl.Series(req.name, [], dtype=DTYPE_TO_POLARS[req.dtype])

    if chunk_size is None or chunk_size <= 0 or chunk_size >= req.length:
        arr = handle.read_channel_range(req.group, req.channel, 0, req.length)
        return _series_from_numpy(req.name, arr)

    pieces: list[pl.Series] = []
    for start in range(0, req.length, chunk_size):
        end = min(start + chunk_size, req.length)
        arr = handle.read_channel_range(req.group, req.channel, start, end)
        pieces.append(_series_from_numpy(req.name, arr))
    if len(pieces) == 1:
        return pieces[0]
    return pl.concat(pieces, rechunk=False)


def _read_requested(
    handle: _core.TdmsHandle,
    requests: Sequence[_ChannelRequest],
    chunk_size: int | None,
) -> pl.DataFrame:
    if not requests:
        return pl.DataFrame()
    lengths = {r.length for r in requests}
    if len(lengths) > 1:
        details = ", ".join(f"{r.name}={r.length}" for r in requests)
        raise ValueError(
            "requested channels have differing sample counts, which polars "
            f"DataFrames cannot represent as columns of equal length: {details}"
        )
    columns = [_read_channel_series(handle, r, chunk_size) for r in requests]
    return pl.DataFrame(columns)


class TdmsSource:
    """A lazily opened TDMS file.

    Opening only parses the metadata (the same work as :func:`read_metadata`);
    raw samples are read on demand. Use it as a context manager, or drop it to
    release the underlying file handle.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self._path = os.fspath(path)
        self._handle: _core.TdmsHandle | None = _core.TdmsHandle(self._path)
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

    def scan(
        self,
        group: str | Sequence[str] | None = None,
        columns: Sequence[str] | None = None,
        chunk_size: int | None = DEFAULT_CHUNK_SIZE,
    ) -> pl.LazyFrame:
        """Return a ``LazyFrame`` over the requested group/channels.

        The TDMS data is not read until the returned frame is collected.
        Thanks to projection pushdown, ``scan(...).select([...]).collect()``
        only loads the selected channels.

        Parameters
        ----------
        group:
            Group name, a list of group names, or ``None`` to merge all groups.
            When more than one group is selected, column names are prefixed
            with ``"GroupName/"`` to stay unique.
        columns:
            Optional subset of channels to load.
        chunk_size:
            Samples read per chunk for channels wider than this (memory bound).
            Pass ``None`` to load each channel in a single allocation.
        """
        meta = self.metadata
        if self._handle is None:
            raise RuntimeError("TdmsSource has been closed")
        handle = self._handle
        requests = _resolve_channels(meta, group, columns)
        schema: dict[str, pl.DataType] = {
            r.name: DTYPE_TO_POLARS[r.dtype] for r in requests
        }
        if not schema:
            raise ValueError(f"no readable channel data in {self._path!r}")

        def _load_batch(_in: pl.DataFrame) -> pl.DataFrame:
            wanted = set(_in.columns)
            selected = [r for r in requests if r.name in wanted]
            if not selected:
                return pl.DataFrame(schema=schema)
            return _read_requested(handle, selected, chunk_size)

        placeholder = pl.DataFrame(schema=schema)
        return placeholder.lazy().map_batches(
            _load_batch,
            schema=schema,
            predicate_pushdown=False,
            projection_pushdown=True,
            slice_pushdown=False,
            validate_output_schema=False,
            streamable=False,
        )

    def read(
        self,
        group: str | Sequence[str] | None = None,
        columns: Sequence[str] | None = None,
        chunk_size: int | None = DEFAULT_CHUNK_SIZE,
    ) -> pl.DataFrame:
        """Eagerly build a :class:`pl.DataFrame` for the requested group/channels."""
        return self.scan(group=group, columns=columns, chunk_size=chunk_size).collect()

    def __enter__(self) -> "TdmsSource":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def open_tdms(path: str | os.PathLike[str]) -> TdmsSource:
    """Open a TDMS file, indexing its metadata immediately."""
    return TdmsSource(path)


def read_metadata(path: str | os.PathLike[str]) -> TdmsMetadata:
    """Read the full TDMS metadata without loading any raw channel data."""
    with TdmsSource(path) as src:
        return src.metadata


def scan_tdms(
    path: str | os.PathLike[str],
    group: str | Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    chunk_size: int | None = DEFAULT_CHUNK_SIZE,
) -> pl.LazyFrame:
    """Lazily scan a TDMS file as a ``pl.LazyFrame``.

    Only metadata is parsed up front; channel data is read when the frame is
    collected (see :meth:`TdmsSource.scan` for the parameters).
    """
    return TdmsSource(path).scan(group=group, columns=columns, chunk_size=chunk_size)


def read_tdms(
    path: str | os.PathLike[str],
    group: str | Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    chunk_size: int | None = DEFAULT_CHUNK_SIZE,
) -> pl.DataFrame:
    """Eagerly read a TDMS group into a ``pl.DataFrame``."""
    return scan_tdms(path, group=group, columns=columns, chunk_size=chunk_size).collect()