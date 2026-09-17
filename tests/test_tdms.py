from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import polars_tdms as pt
import pytest
from nptdms import TdmsFile, TdmsWriter, ChannelObject, GroupObject, RootObject

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def tdms_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("tdms") / "sample.tdms"
    now = dt.datetime(2024, 5, 17, 12, 30, 45, tzinfo=dt.timezone.utc)
    with TdmsWriter(path) as w:
        root = RootObject(properties={"Title": "Demo", "Version": 1.5, "Flag": True, "Time": now})
        sensors = GroupObject("Sensors", properties={"Desc": "sensor group"})
        other = GroupObject("Other")
        w.write_segment(
            [
                root,
                sensors,
                other,
                ChannelObject("Sensors", "Voltage", np.arange(20.0), properties={"unit_string": "V"}),
                ChannelObject("Sensors", "Current", np.arange(100, 120, dtype=np.int32)),
                ChannelObject("Sensors", "On", np.array([True, False] * 10)),
                ChannelObject("Other", "Temp", np.linspace(0, 1, 20)),
            ]
        )
    return path


def test_read_metadata(tdms_file):
    meta = pt.read_metadata(tdms_file)
    assert isinstance(meta, pt.TdmsMetadata)
    assert meta.properties["Title"] == "Demo"
    assert meta.properties["Version"] == 1.5
    assert meta.properties["Flag"] is True
    assert meta.properties["Time"] == dt.datetime(2024, 5, 17, 12, 30, 45)

    assert meta.group_names == ("Sensors", "Other")
    sensors = meta.group("Sensors")
    assert sensors.properties["Desc"] == "sensor group"
    assert sensors.channel_names == ("Voltage", "Current", "On")

    voltage = sensors.channel("Voltage")
    assert voltage.dtype == "Double"
    assert voltage.length == 20
    assert voltage.polars_dtype == pl.Float64
    assert voltage.properties["unit_string"] == "V"
    assert voltage.readable

    current = sensors.channel("Current")
    assert current.dtype == "I32"
    assert current.polars_dtype == pl.Int32


def test_scan_is_lazy(tdms_file):
    lf = pt.scan_tdms(tdms_file, group="Sensors")
    assert isinstance(lf, pl.LazyFrame)
    assert lf.collect_schema() == {
        "Voltage": pl.Float64,
        "Current": pl.Int32,
        "On": pl.Boolean,
    }


def test_read_tdms_values(tdms_file):
    df = pt.read_tdms(tdms_file, group="Sensors")
    assert df.shape == (20, 3)
    assert df.schema == {
        "Voltage": pl.Float64,
        "Current": pl.Int32,
        "On": pl.Boolean,
    }
    assert df["Voltage"].to_list() == list(range(20))
    assert df["Current"].to_list() == list(range(100, 120))
    assert df["On"].to_list() == [True, False] * 10


def test_matches_nptdms(tdms_file):
    ours = pt.read_tdms(tdms_file, group="Other")
    ref = TdmsFile.read(tdms_file)
    assert np.allclose(ours["Temp"].to_numpy(), ref["Other"]["Temp"][:])


def test_all_groups_merged_with_prefix(tdms_file):
    df = pt.read_tdms(tdms_file)
    assert list(df.columns) == [
        "Sensors/Voltage",
        "Sensors/Current",
        "Sensors/On",
        "Other/Temp",
    ]
    assert df.shape == (20, 4)


def test_single_group_no_prefix(tdms_file):
    df = pt.read_tdms(tdms_file, group="Other")
    assert list(df.columns) == ["Temp"]


def test_columns_selection(tdms_file):
    df = pt.read_tdms(tdms_file, group="Sensors", columns=["Current"])
    assert list(df.columns) == ["Current"]

    with pytest.raises(ValueError, match="columns not found"):
        pt.read_tdms(tdms_file, group="Sensors", columns=["nope"])


def test_projection_pushdown_only_loads_selected(tdms_file):
    src = pt.TdmsSource(tdms_file)
    real = src._handle

    class SpyHandle:
        def __init__(self, h):
            self._h = h
            self.calls: list[tuple] = []

        def __getattr__(self, name):
            return getattr(self._h, name)

        def read_channel_range_buffers(self, group, channel, start, end):
            self.calls.append((group, channel, start, end))
            return self._h.read_channel_range_buffers(group, channel, start, end)

    src._handle = SpyHandle(real)
    df = src.scan(group="Sensors").select("Current").collect()
    assert df.schema == {"Current": pl.Int32}
    channels_read = {c for _, c, _, _ in src._handle.calls}
    assert channels_read == {"Current"}


def test_chunked_matches_whole(tdms_file):
    whole = pt.read_tdms(tdms_file, group="Other")
    chunked = pt.read_tdms(tdms_file, group="Other", chunk_size=7)
    assert whole.equals(chunked)

    single_pass = pt.read_tdms(tdms_file, group="Other", chunk_size=None)
    assert whole.equals(single_pass)


def test_empty_channel():
    path = None
    with TdmsWriter("/tmp/opencode/_empty_ch.tdms") as w:
        w.write_segment(
            [
                GroupObject("G"),
                ChannelObject("G", "Sig", np.array([], dtype=np.float64)),
            ]
        )
        path = w._file
    df = pt.read_tdms("/tmp/opencode/_empty_ch.tdms", group="G")
    assert df.shape == (0, 1)
    assert df.schema == {"Sig": pl.Float64}
    _ = path


@pytest.fixture(scope="module")
def multiseg_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("tdms") / "multiseg.tdms"
    rng = np.random.default_rng(7)
    lenses = [50_000, 20_000, 30_000]
    with TdmsWriter(path) as w:
        grp = GroupObject("DAQ")
        for n in lenses:
            w.write_segment(
                [
                    grp,
                    ChannelObject("DAQ", "SigA", rng.standard_normal(n)),
                    ChannelObject("DAQ", "SigB", rng.uniform(0, 10, n)),
                ]
            )
    return path


def test_multi_segment(multiseg_file):
    df = pt.read_tdms(multiseg_file, group="DAQ")
    assert df.shape == (100_000, 2)
    chunked = pt.read_tdms(multiseg_file, group="DAQ", chunk_size=9_999)
    assert df.equals(chunked)

    ref = TdmsFile.read(multiseg_file)
    assert np.allclose(ref["DAQ"]["SigB"][:], df["SigB"].to_numpy())


def test_differing_lengths_raises():
    path = "/tmp/opencode/_mismatch.tdms"
    with TdmsWriter(path) as w:
        w.write_segment(
            [
                GroupObject("G"),
                ChannelObject("G", "A", np.arange(5.0)),
                ChannelObject("G", "B", np.arange(3.0)),
            ]
        )
    with pytest.raises(ValueError, match="differing sample counts"):
        pt.read_tdms(path, group="G")


def test_string_channel_roundtrip():
    path = "/tmp/opencode/_str_ch.tdms"
    with TdmsWriter(path) as w:
        ch = ChannelObject("G", "Label", ["a", "bb", "ccc"])
        w.write_segment([GroupObject("G"), ch])
    df = pt.read_tdms(path, group="G")
    assert df.shape == (3, 1)
    assert df.schema == {"Label": pl.Utf8}
    assert df["Label"].to_list() == ["a", "bb", "ccc"]


def test_string_channel_buffers_path():
    path = "/tmp/opencode/_str_buf.tdms"
    with TdmsWriter(path) as w:
        w.write_segment(
            [GroupObject("G"), ChannelObject("G", "Label", ["ec", "", "a\U0001F600c"])]
        )

    src = pt.TdmsSource(path)
    real = src._handle

    class SpyHandle:
        def __init__(self, h):
            self._h = h
            self.buffer_calls = 0

        def __getattr__(self, name):
            return getattr(self._h, name)

        def read_channel_strings_buffers(self, *a):
            self.buffer_calls += 1
            return self._h.read_channel_strings_buffers(*a)

    src._handle = SpyHandle(real)
    df = src.read(group="G")
    assert df["Label"].to_list() == ["ec", "", "a\U0001F600c"]
    assert df.schema["Label"] == pl.Utf8
    assert src._handle.buffer_calls > 0

    offs, data = real.read_channel_strings_buffers("G", "Label", 0, 3)
    n = len(offs) // 8 - 1
    assert n == 3
    import struct

    parsed = []
    for i in range(n):
        s, e = struct.unpack_from("<qq", offs, i * 8)
        parsed.append(data[s:e].decode("utf-8"))
    assert parsed == ["ec", "", "a\U0001F600c"]


def test_mixed_group_with_string_channel(tmp_path):
    path = tmp_path / "mixed.tdms"
    with TdmsWriter(path) as w:
        w.write_segment(
            [
                GroupObject("G"),
                ChannelObject("G", "Sig", np.arange(3.0)),
                ChannelObject("G", "Label", ["a", "bb", "ccc"]),
            ]
        )

    df = pt.read_tdms(path, group="G")
    assert set(df.columns) == {"Sig", "Label"}
    assert df.schema["Sig"] == pl.Float64
    assert df.schema["Label"] == pl.Utf8
    assert df["Sig"].to_list() == [0.0, 1.0, 2.0]
    assert df["Label"].to_list() == ["a", "bb", "ccc"]

    label = pt.read_metadata(path).group("G").channel("Label")
    assert label.dtype == "String"
    assert label.readable
    assert label.length == 3


def test_string_chunked_and_scan():
    path = "/tmp/opencode/_str_scan.tdms"
    strs = [f"s{i:04d}" for i in range(10)]
    with TdmsWriter(path) as w:
        w.write_segment([GroupObject("G"), ChannelObject("G", "Label", strs)])

    lf = pt.scan_tdms(path, group="G")
    assert lf.collect_schema() == {"Label": pl.Utf8}
    assert lf.filter(pl.col("Label").str.starts_with("s00")).collect()["Label"].to_list() == strs

    whole = pt.read_tdms(path, group="G")
    chunked = pt.read_tdms(path, group="G", chunk_size=3)
    single = pt.read_tdms(path, group="G", chunk_size=None)
    assert whole.equals(chunked)
    assert whole.equals(single)
    assert chunked["Label"].to_list() == strs


def test_numeric_buffer_path(tmp_path):
    import struct

    path = tmp_path / "numeric.tdms"
    with TdmsWriter(path) as w:
        w.write_segment(
            [
                GroupObject("G"),
                ChannelObject("G", "F", np.array([1.5, -2.5, 3.25], dtype=np.float32)),
                ChannelObject(
                    "G",
                    "B",
                    np.array([True, True, False, True, False, False, False, True, True]),
                ),
            ]
        )

    src = pt.TdmsSource(path)
    real = src._handle

    class SpyHandle:
        def __init__(self, h):
            self._h = h
            self.buffer_calls = 0

        def __getattr__(self, name):
            return getattr(self._h, name)

        def read_channel_range_buffers(self, *a):
            self.buffer_calls += 1
            return self._h.read_channel_range_buffers(*a)

    src._handle = SpyHandle(real)
    df = src.read(group="G", columns=["F"])
    assert df["F"].to_list() == [1.5, -2.5, 3.25]
    assert df.schema["F"] == pl.Float32
    assert src._handle.buffer_calls > 0

    raw = real.read_channel_range_buffers("G", "F", 0, 3)
    vals = struct.unpack("<3f", raw)
    assert vals == (1.5, -2.5, 3.25)

    assert src.read(group="G", columns=["B"])["B"].to_list() == [
        True,
        True,
        False,
        True,
        False,
        False,
        False,
        True,
        True,
    ]


def test_unknown_group_error(tdms_file):
    with pytest.raises(ValueError, match="not found"):
        pt.read_tdms(tdms_file, group="Missing")