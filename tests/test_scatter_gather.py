from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import polars_tdms as pt
import pytest
from nptdms import ChannelObject, GroupObject, TdmsWriter
from polars_tdms import _core


@pytest.mark.parametrize("strategy", ["scalar", "vectored", "coalesced"])
def test_fragment_reader_scattered_destinations(tmp_path, strategy):
    path = tmp_path / "raw.bin"
    path.write_bytes(b"ABcdEF__ghIJ")
    # Disk order differs from destination order, including repeated channels.
    plan = [(0, 0, 2, 2), (2, 1, 0, 2), (4, 0, 0, 2),
            (8, 1, 2, 2), (10, 0, 4, 2)]
    buffers, seeks, reads, transferred = _core._read_fragment_buffers(
        str(path), [6, 4], plan[::-1], strategy
    )
    assert buffers == [b"EFABIJ", b"cdgh"]
    if strategy == "scalar":
        assert (seeks, reads, transferred) == (5, 5, 10)
    elif strategy == "vectored":
        assert seeks == 2
        assert 2 <= reads <= 5
        assert transferred == 10
    else:
        assert (seeks, reads, transferred) == (1, 1, 12)


@pytest.mark.parametrize("strategy", ["scalar", "vectored", "coalesced"])
def test_fragment_reader_empty(tmp_path, strategy):
    path = tmp_path / "raw.bin"
    path.write_bytes(b"")
    assert _core._read_fragment_buffers(str(path), [0], [], strategy) == ([b""], 0, 0, 0)


@pytest.mark.parametrize(
    "sizes,fragments,message",
    [
        ([4], [(0, 1, 0, 4)], "channel index"),
        ([4], [(0, 0, 2, 4)], "range out of bounds"),
        ([4], [(7, 0, 0, 4)], "range out of bounds"),
        ([4], [(0, 0, 0, 0)], "range out of bounds"),
        ([4], [(0, 0, 0, 2)], "do not fill"),
        ([4], [(0, 0, 0, 2), (2, 0, 1, 2)], "exactly once"),
        ([4], [(0, 0, 0, 2), (1, 0, 2, 2)], "overlapping file"),
        ([20], [], "exceed file size"),
    ],
)
def test_fragment_reader_invalid_plan(tmp_path, sizes, fragments, message):
    path = tmp_path / "raw.bin"
    path.write_bytes(b"01234567")
    with pytest.raises(ValueError, match=message):
        _core._read_fragment_buffers(str(path), sizes, fragments, "coalesced")


def test_fragment_reader_errors(tmp_path):
    path = tmp_path / "raw.bin"
    with pytest.raises(ValueError, match="unknown"):
        _core._read_fragment_buffers(str(path), [], [], "unknown")
    with pytest.raises(OSError):
        _core._read_fragment_buffers(str(path), [], [], "scalar")


def test_coalescing_respects_gap_and_span_limits(tmp_path):
    path = tmp_path / "raw.bin"
    gap = 64 * 1024 + 1
    span = 8 * 1024 * 1024
    data = bytearray(span + 3)
    data[0] = 1
    data[gap + 1] = 2
    data[-1] = 3
    path.write_bytes(data)
    buffers, seeks, reads, transferred = _core._read_fragment_buffers(
        str(path), [3],
        [(0, 0, 0, 1), (gap + 1, 0, 1, 1), (span + 2, 0, 2, 1)],
        "coalesced",
    )
    assert buffers == [b"\x01\x02\x03"]
    assert (seeks, reads, transferred) == (3, 3, 3)


def test_coalescing_respects_span_limit(tmp_path):
    path = tmp_path / "raw.bin"
    span = 8 * 1024 * 1024
    path.write_bytes(b"A" * (span + 1))
    buffers, seeks, reads, transferred = _core._read_fragment_buffers(
        str(path), [span + 1], [(0, 0, 0, span), (span, 0, span, 1)], "coalesced",
    )
    assert buffers == [b"A" * (span + 1)]
    assert (seeks, reads, transferred) == (2, 2, span + 1)


def test_coalescing_bounds_read_amplification(tmp_path):
    path = tmp_path / "raw.bin"
    path.write_bytes(b"A" + b"x" * 100 + b"B")
    buffers, seeks, reads, transferred = _core._read_fragment_buffers(
        str(path), [2], [(0, 0, 0, 1), (101, 0, 1, 1)], "coalesced",
    )
    assert buffers == [b"AB"]
    assert (seeks, reads, transferred) == (2, 2, 2)


def _benchmark_module():
    script = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_scatter_gather.py"
    spec = importlib.util.spec_from_file_location("bench_scatter_gather", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fragment_plan_repeated_chunks_and_selection(tmp_path):
    path = tmp_path / "chunks.tdms"
    with TdmsWriter(path) as writer:
        writer.write_segment([
            GroupObject("G"),
            ChannelObject("G", "A", np.array([1, 2], dtype=np.int32)),
            ChannelObject("G", "B", np.array([3, 4], dtype=np.int32)),
        ])
    data = bytearray(path.read_bytes())
    data.extend(data[-16:])
    data[12:20] = (len(data) - 28).to_bytes(8, "little")
    path.write_bytes(data)
    plan = _benchmark_module().build_plan(str(path), "G", ["B"])
    assert plan.sizes == [16]
    assert len(plan.fragments) == 2
    for strategy in ("scalar", "vectored", "coalesced"):
        buffers, _, _, _ = _core._read_fragment_buffers(
            str(path), plan.sizes, plan.fragments, strategy
        )
        np.testing.assert_array_equal(np.frombuffer(buffers[0], dtype="<i4"), [3, 4, 3, 4])
    assert not Path(f"{path}_index").exists()


@pytest.mark.parametrize("toc_bit", [1 << 5, 1 << 6, 1 << 7])
def test_fragment_plan_rejects_unsupported_layout(tmp_path, toc_bit):
    path = tmp_path / "layout.tdms"
    with TdmsWriter(path) as writer:
        writer.write_segment([ChannelObject("G", "A", np.arange(3, dtype=np.int32))])
    module = _benchmark_module()
    # Exercise rejection using parsed metadata so big-endian/DAQmx flags do
    # not invalidate the synthetic file before the layout guard is reached.
    original_open = module.TdmsFile.open

    def altered_open(*args, **kwargs):
        source = original_open(*args, **kwargs)
        source._reader._segments[0].toc_mask |= toc_bit
        return source

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module.TdmsFile, "open", altered_open)
        with pytest.raises(ValueError, match="unsupported layout"):
            module.build_plan(str(path), "G")


@pytest.fixture
def production_file(tmp_path):
    path = tmp_path / "production.tdms"
    expected = {
        dtype: np.arange(25).astype(dtype)
        for dtype in (
            "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64", "float32", "float64",
        )
    }
    expected["int64"] -= 2**60
    expected["uint64"] += 2**63
    expected["float32"][3] = np.nan
    expected["float64"][10] = -np.inf
    expected["bool"] = np.arange(25) % 3 != 0
    expected["Label"] = [f"row-{i}" for i in range(25)]
    with TdmsWriter(path) as writer:
        for start, end in ((0, 9), (9, 14), (14, 25)):
            writer.write_segment([
                GroupObject("G"),
                *[
                    ChannelObject("G", name, values[start:end])
                    for name, values in expected.items()
                ],
                ChannelObject("G", "Empty", np.array([], dtype=np.float64)),
            ])
    return path, expected


@pytest.mark.parametrize("chunk_size", [None, 0, -1, 1, 7, 100])
def test_production_batched_all_types(production_file, chunk_size):
    path, expected = production_file
    result = pt.read_tdms(path, group="G", chunk_size=chunk_size)
    assert result.columns == [*expected, "Empty"]
    assert result.shape == (25, 13)
    for name, values in expected.items():
        if name == "Label":
            assert result[name].to_list() == values
        else:
            np.testing.assert_array_equal(result[name].to_numpy(), values)
    assert result["Empty"].null_count() == 25


def test_production_projection_batches_only_selected(production_file):
    path, expected = production_file
    with pt.open_tdms(path) as source:
        real = source._handle
        assert real is not None

        class SpyHandle:
            def __init__(self):
                self.calls = []

            def __getattr__(self, name):
                return getattr(real, name)

            def read_channels_range_buffers(self, channels, start, end):
                self.calls.append((channels, start, end))
                return real.read_channels_range_buffers(channels, start, end)

            def read_channel_range_buffers(self, *args):
                raise AssertionError("batched channels must not be read one by one")

            def read_channel_strings_buffers(self, *args):
                raise AssertionError("unprojected string data must not be decoded")

        spy = SpyHandle()
        source._handle = spy
        result = source.scan(group="G", chunk_size=7).select("int32", "bool").collect()
        assert spy.calls == [
            ([("G", "int32"), ("G", "bool")], start, min(start + 7, 25))
            for start in range(0, 25, 7)
        ]
        np.testing.assert_array_equal(result["int32"].to_numpy(), expected["int32"])
        np.testing.assert_array_equal(result["bool"].to_numpy(), expected["bool"])


def test_production_multigroup_batch(tmp_path):
    path = tmp_path / "groups.tdms"
    with TdmsWriter(path) as writer:
        for start in (0, 5):
            writer.write_segment([
                ChannelObject("First", "A", np.arange(start, start + 5, dtype=np.int32)),
                ChannelObject("Second", "A", np.arange(start + 10, start + 15, dtype=np.int32)),
                ChannelObject("Second", "Skipped", ["text"] * 5),
            ])
    result = pt.scan_tdms(path, chunk_size=3).select("Second/A", "First/A").collect()
    assert result.columns == ["Second/A", "First/A"]
    assert result["First/A"].to_list() == list(range(10))
    assert result["Second/A"].to_list() == list(range(10, 20))


def _endian_segment(endian, offset):
    arrays = [
        ("Ints", 3, np.array([offset, offset + 1, offset + 2], dtype=np.int32)),
        ("Floats", 10, np.array([offset + 0.5, np.nan, -np.inf], dtype=np.float64)),
        ("Flags", 33, np.array([True, False, True])),
    ]
    metadata = bytearray(struct.pack(f"{endian}I", len(arrays)))
    payload = bytearray()
    for name, dtype, values in arrays:
        path = f"/'G'/'{name}'".encode("utf-8")
        metadata.extend(struct.pack(f"{endian}I", len(path)))
        metadata.extend(path)
        metadata.extend(struct.pack(f"{endian}IIIQI", 20, dtype, 1, len(values), 0))
        payload.extend(values.astype(values.dtype.newbyteorder(endian)).tobytes())
    toc = 14 | (64 if endian == ">" else 0)
    lead = b"TDSm" + struct.pack("<I", toc)
    lead += struct.pack(f"{endian}IQQ", 4713, len(metadata) + len(payload), len(metadata))
    return lead + metadata + payload


@pytest.mark.parametrize("chunk_size", [None, 2, 4])
def test_production_mixed_endian_fragments(tmp_path, chunk_size):
    from nptdms import TdmsFile

    path = tmp_path / "endian.tdms"
    path.write_bytes(_endian_segment(">", -3) + _endian_segment("<", 10))
    result = pt.read_tdms(path, group="G", chunk_size=chunk_size, create_index_if_missing=False)
    reference = TdmsFile.read(path)
    for name in ("Ints", "Floats", "Flags"):
        np.testing.assert_array_equal(result[name].to_numpy(), reference["G"][name][:])
    assert result["Ints"].to_list() == [-3, -2, -1, 10, 11, 12]
    assert not Path(f"{path}_index").exists()


def test_production_batch_range_and_errors(production_file):
    path, expected = production_file
    handle = _core.TdmsHandle(str(path), create_index_if_missing=False)
    buffers = handle.read_channels_range_buffers([("G", "bool"), ("G", "int64")], 5, 20)
    actual_bool = np.unpackbits(np.frombuffer(buffers[0], dtype=np.uint8), bitorder="little")[:15]
    np.testing.assert_array_equal(actual_bool, expected["bool"][5:20])
    np.testing.assert_array_equal(np.frombuffer(buffers[1], dtype="<i8"), expected["int64"][5:20])
    assert handle.read_channels_range_buffers([("G", "int32"), ("G", "bool")], 25, 25) == [b"", b""]
    assert handle.read_channels_range_buffers([], 0, 0) == []
    with pytest.raises(ValueError, match="end must be"):
        handle.read_channels_range_buffers([("G", "int32")], 2, 1)
    with pytest.raises(ValueError):
        handle.read_channels_range_buffers([("G", "int32")], 0, 26)
    with pytest.raises(ValueError, match="duplicate"):
        handle.read_channels_range_buffers([("G", "int32"), ("G", "int32")], 0, 2)
    with pytest.raises(KeyError):
        handle.read_channels_range_buffers([("Missing", "int32")], 0, 2)
    with pytest.raises(KeyError):
        handle.read_channels_range_buffers([("G", "Missing")], 0, 2)
    with pytest.raises(NotImplementedError, match="String"):
        handle.read_channels_range_buffers([("G", "Label")], 0, 2)


def test_production_io_errors_are_not_hidden(production_file):
    path, _ = production_file
    handle = _core.TdmsHandle(str(path), create_index_if_missing=False)
    with path.open("r+b") as stream:
        stream.truncate(28)
    with pytest.raises(ValueError, match="file size|range out of bounds"):
        handle.read_channels_range_buffers([("G", "int32"), ("G", "float64")], 0, 25)
    path.unlink()
    with pytest.raises(OSError):
        handle.read_channel_range_buffers("G", "int32", 0, 2)
