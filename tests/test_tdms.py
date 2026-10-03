from __future__ import annotations

import datetime as dt
import gc
import os
import tempfile

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


def test_copy_to_temp_includes_index_and_cleans_up(tdms_file):
    with pt.open_tdms(tdms_file):
        pass
    assert os.path.isfile(f"{tdms_file}_index")

    source = pt.TdmsSource(tdms_file, copy_to_temp=True)
    local_path = source._access_path
    local_directory = os.path.dirname(local_path)
    assert source.path == os.fspath(tdms_file)
    assert local_path != source.path
    assert os.path.isfile(local_path)
    assert os.path.isfile(f"{local_path}_index")
    assert source.read(group="Sensors", columns=["Current"])["Current"].to_list() == list(
        range(100, 120)
    )
    assert pt.read_tdms(
        tdms_file, group="Sensors", columns=["Current"], copy_to_temp=True
    )["Current"].to_list() == list(range(100, 120))

    source.close()
    assert not os.path.exists(local_directory)


def test_copy_to_temp_lives_as_long_as_lazy_scan(tdms_file):
    source = pt.TdmsSource(tdms_file, copy_to_temp=True)
    local_path = source._access_path
    local_directory = os.path.dirname(local_path)
    lazy_frame = source.scan(group="Sensors", columns=["Current"])

    source.close()
    assert os.path.isfile(local_path)
    assert lazy_frame.collect()["Current"].to_list() == list(range(100, 120))

    del lazy_frame
    gc.collect()
    assert not os.path.exists(local_directory)

    lazy_frame = pt.scan_tdms(
        tdms_file, group="Sensors", columns=["Current"], copy_to_temp=True
    )
    assert lazy_frame.collect()["Current"].to_list() == list(range(100, 120))


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


def test_projection_pushdown_drop_to_parquet_wide_schema(tmp_path):
    path = tmp_path / "wide.tdms"
    channels = [
        ChannelObject("G", f"C{i}", ["text", "value"] if i == 2 else np.arange(2))
        for i in range(45)
    ]
    with TdmsWriter(path) as writer:
        writer.write_segment([GroupObject("G"), *channels])

    parquet_path = tmp_path / "wide.parquet"
    pt.scan_tdms(path, group="G").drop("C2").sink_parquet(parquet_path)

    result = pl.read_parquet(parquet_path)
    assert result.shape == (2, 44)
    assert "C2" not in result.columns
    assert result["C44"].to_list() == [0, 1]


def test_fragmented_string_and_numeric_reads(tmp_path):
    path = tmp_path / "fragmented.tdms"
    expected_numeric = []
    expected_labels = []
    with TdmsWriter(path) as writer:
        for segment in range(12):
            values = np.arange(segment * 5, (segment + 1) * 5, dtype=np.int32)
            labels = [f"segment-{segment}-value-{i}" for i in range(5)]
            writer.write_segment(
                [
                    GroupObject("G"),
                    ChannelObject("G", "Numeric", values),
                    ChannelObject("G", "Label", labels),
                ]
            )
            expected_numeric.extend(values.tolist())
            expected_labels.extend(labels)

    result = pt.read_tdms(path, group="G")
    assert result["Numeric"].to_list() == expected_numeric
    assert result["Label"].to_list() == expected_labels


def test_chunked_matches_whole(tdms_file):
    whole = pt.read_tdms(tdms_file, group="Other")
    chunked = pt.read_tdms(tdms_file, group="Other", chunk_size=7)
    assert whole.equals(chunked)

    single_pass = pt.read_tdms(tdms_file, group="Other", chunk_size=None)
    assert whole.equals(single_pass)


def test_empty_channel():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "empty_ch.tdms")
        with TdmsWriter(path) as w:
            w.write_segment(
                [
                    GroupObject("G"),
                    ChannelObject("G", "Sig", np.array([], dtype=np.float64)),
                ]
            )
        df = pt.read_tdms(path, group="G")
        assert df.shape == (0, 1)
        assert df.schema == {"Sig": pl.Float64}


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
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "mismatch.tdms")
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
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "str_ch.tdms")
        with TdmsWriter(path) as w:
            ch = ChannelObject("G", "Label", ["a", "bb", "ccc"])
            w.write_segment([GroupObject("G"), ch])
        df = pt.read_tdms(path, group="G")
        assert df.shape == (3, 1)
        assert df.schema == {"Label": pl.Utf8}
        assert df["Label"].to_list() == ["a", "bb", "ccc"]


def test_string_channel_buffers_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "str_buf.tdms")
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
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "str_scan.tdms")
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


def _write_single_channel(tmp_path: object, name: str = "sig.tdms"):
    path = tmp_path / name
    with TdmsWriter(path) as w:
        w.write_segment([GroupObject("G"), ChannelObject("G", "A", np.arange(5.0))])
    return path


def _index_path(path: object) -> object:
    return path.with_suffix(path.suffix + "_index")


def test_default_open_generates_index_file(tmp_path):
    path = _write_single_channel(tmp_path)
    idx = _index_path(path)
    assert not os.path.exists(idx)

    df = pt.read_tdms(path, group="G")
    assert df["A"].to_list() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert os.path.exists(idx)
    with open(idx, "rb") as f:
        assert f.read(4) == b"TDSh"

    again = pt.read_tdms(path, group="G")
    assert again.equals(df)


def test_create_index_if_missing_false(tmp_path):
    path = _write_single_channel(tmp_path)
    idx = _index_path(path)
    df = pt.read_tdms(path, group="G", create_index_if_missing=False)
    assert df["A"].to_list() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert not os.path.exists(idx)


def test_use_index_file_false_ignores_index(tmp_path):
    path = _write_single_channel(tmp_path)
    idx = _index_path(path)
    idx.write_bytes(b"bogus")

    df = pt.read_tdms(path, group="G", use_index_file=False)
    assert df["A"].to_list() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert idx.read_bytes() == b"bogus"


def test_corrupt_index_falls_back_and_regenerates(tmp_path):
    path = _write_single_channel(tmp_path)
    idx = _index_path(path)
    pt.read_tdms(path, group="G")
    idx.write_bytes(b"corrupt index, not a valid tdms index")

    df = pt.read_tdms(path, group="G")
    assert df["A"].to_list() == [0.0, 1.0, 2.0, 3.0, 4.0]
    with open(idx, "rb") as f:
        assert f.read(4) == b"TDSh"


def test_stale_index_falls_back_and_regenerates(tmp_path):
    path = _write_single_channel(tmp_path)
    idx = _index_path(path)
    pt.read_tdms(path, group="G")

    # Age the index into the past so the data file looks newer than it
    # (a stale index must never be trusted).
    past = os.path.getmtime(path) - 100
    os.utime(idx, (past, past))

    df = pt.read_tdms(path, group="G")
    assert df["A"].to_list() == [0.0, 1.0, 2.0, 3.0, 4.0]

    # The stale index was regenerated and is now current with the data file.
    assert os.path.getmtime(idx) >= os.path.getmtime(path)


def test_verify_index_true_passes_on_matching_index(tmp_path):
    path = _write_single_channel(tmp_path)
    pt.read_tdms(path, group="G")
    df = pt.read_tdms(path, group="G", verify_index=True)
    assert df["A"].to_list() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_verify_index_true_raises_on_mismatch(tmp_path):
    path = _write_single_channel(tmp_path, name="a.tdms")
    other = tmp_path / "b.tdms"
    with TdmsWriter(other) as w:
        w.write_segment(
            [GroupObject("G"), ChannelObject("G", "A", np.arange(3, dtype=np.int32))]
        )
    pt.read_tdms(other, group="G")

    target = _index_path(path)
    os.replace(_index_path(other), target)
    new_mtime = os.path.getmtime(path) + 100
    os.utime(target, (new_mtime, new_mtime))

    with pytest.raises(ValueError, match="index"):
        pt.read_tdms(path, group="G", verify_index=True)