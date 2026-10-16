from __future__ import annotations

import numpy as np
import polars as pl
import polars_tdms as pt
import pytest
from nptdms import ChannelObject, GroupObject, TdmsWriter


@pytest.fixture
def mixed_file(tmp_path):
    path = tmp_path / "streaming.tdms"
    with TdmsWriter(path) as writer:
        for start, end in ((0, 3), (3, 11), (11, 23)):
            writer.write_segment([
                GroupObject("G"),
                ChannelObject("G", "Value", np.arange(start, end, dtype=np.int32)),
                ChannelObject("G", "Flag", np.arange(start, end) % 3 == 0),
                ChannelObject("G", "Label", [f"row-{i}" for i in range(start, end)]),
                ChannelObject("G", "Empty", np.array([], dtype=np.float64)),
            ])
    return path


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_streaming_mixed_types_and_repeat_collection(mixed_file, engine):
    lazy = pt.scan_tdms(mixed_file, group="G", chunk_size=5)
    expected = pl.DataFrame({
        "Value": pl.Series(range(23), dtype=pl.Int32),
        "Flag": [i % 3 == 0 for i in range(23)],
        "Label": [f"row-{i}" for i in range(23)],
        "Empty": pl.Series([None] * 23, dtype=pl.Float64),
    })
    assert lazy.collect(engine=engine).equals(expected)
    assert lazy.collect(engine=engine).equals(expected)
    assert lazy.select("Empty").collect(engine=engine).equals(expected.select("Empty"))
    assert lazy.select(pl.len()).collect(engine=engine).item() == 23


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_streaming_filter_projection_and_slice(mixed_file, engine):
    lazy = pt.scan_tdms(mixed_file, group="G", chunk_size=5)
    result = (
        lazy.filter(pl.col("Value") >= 9)
        .select("Label", "Flag")
        .slice(2, 7)
        .collect(engine=engine)
    )
    assert result.to_dict(as_series=False) == {
        "Label": [f"row-{i}" for i in range(11, 18)],
        "Flag": [i % 3 == 0 for i in range(11, 18)],
    }
    assert (
        lazy.filter(pl.col("Value") > 100)
        .collect(engine=engine)
        .schema
        == lazy.collect_schema()
    )
    assert lazy.filter(pl.col("Value") > 100).collect(engine=engine).height == 0
    assert lazy.head(0).collect(engine=engine).schema == lazy.collect_schema()


def test_streaming_head_reads_only_requested_rows(mixed_file, monkeypatch):
    calls = []
    original = pt._read_requested_range

    def read_range(handle, requests, start, end):
        calls.append((start, end))
        return original(handle, requests, start, end)

    monkeypatch.setattr(pt, "_read_requested_range", read_range)
    result = pt.scan_tdms(mixed_file, group="G", chunk_size=5).head(7).collect()
    assert result["Value"].to_list() == list(range(7))
    assert calls == [(0, 5), (5, 7)]


def test_source_honors_batch_hint_and_post_filter_limit(mixed_file, monkeypatch):
    sources = []
    original = pt.register_io_source

    def register(source, **kwargs):
        sources.append(source)
        return original(source, **kwargs)

    monkeypatch.setattr(pt, "register_io_source", register)
    lazy = pt.scan_tdms(mixed_file, group="G", chunk_size=5)
    batches = list(sources[0](
        ["Label", "Value"], pl.col("Value") >= 8, 4, 3
    ))
    assert all(batch.height <= 3 for batch in batches)
    assert pl.concat(batches).to_dict(as_series=False) == {
        "Label": [f"row-{i}" for i in range(8, 12)],
        "Value": list(range(8, 12)),
    }
    assert lazy.collect_schema()["Value"] == pl.Int32


def test_streaming_parquet_batches_and_projection(mixed_file, tmp_path, monkeypatch):
    calls = []
    original = pt._read_requested_range

    def read_range(handle, requests, start, end):
        batch = original(handle, requests, start, end)
        calls.append((batch.columns, start, end, batch.height))
        return batch

    monkeypatch.setattr(pt, "_read_requested_range", read_range)
    output = tmp_path / "streaming.parquet"
    (
        pt.scan_tdms(mixed_file, group="G", chunk_size=5)
        .select("Label", "Value", "Flag", "Empty")
        .sink_parquet(output, engine="streaming", row_group_size=4)
    )
    assert [(start, end, height) for _, start, end, height in calls] == [
        (start, min(start + 5, 23), min(5, 23 - start))
        for start in range(0, 23, 5)
    ]
    result = pl.read_parquet(output)
    assert result.columns == ["Label", "Value", "Flag", "Empty"]
    assert result["Value"].to_list() == list(range(23))
    assert result["Flag"].to_list() == [i % 3 == 0 for i in range(23)]
    assert result["Label"].to_list() == [f"row-{i}" for i in range(23)]
    assert result["Empty"].null_count() == 23

    calls.clear()
    (
        pt.scan_tdms(mixed_file, group="G", chunk_size=5)
        .select("Value")
        .sink_parquet(output, engine="streaming")
    )
    assert all(columns == ["Value"] for columns, *_ in calls)
    assert pl.read_parquet(output)["Value"].to_list() == list(range(23))


def test_parquet_consumes_before_source_finishes(tmp_path, monkeypatch):
    path = tmp_path / "larger.tdms"
    rows = 1_048_576
    with TdmsWriter(path) as writer:
        writer.write_segment([
            ChannelObject("G", "Value", np.arange(rows, dtype=np.int32)),
        ])

    read_ends = []
    consumed_at = []
    original = pt._read_requested_range

    def read_range(handle, requests, start, end):
        assert end - start <= 4096
        read_ends.append(end)
        return original(handle, requests, start, end)

    def consume(batch):
        consumed_at.append(read_ends[-1])
        return batch

    monkeypatch.setattr(pt, "_read_requested_range", read_range)
    output = tmp_path / "larger.parquet"
    (
        pt.scan_tdms(path, group="G", chunk_size=4096)
        .map_batches(consume, streamable=True)
        .sink_parquet(output, engine="streaming")
    )
    assert read_ends == list(range(4096, rows + 1, 4096))
    assert consumed_at[0] < rows
    assert pl.read_parquet(output)["Value"].to_list() == list(range(rows))


def test_streaming_concat_and_cross_batch_operations(mixed_file, tmp_path):
    lazy = pt.scan_tdms(mixed_file, group="G", chunk_size=5).select("Value")
    assert lazy.select(pl.col("Value").cum_sum()).collect()["Value"].to_list() == [
        i * (i + 1) // 2 for i in range(23)
    ]
    assert lazy.filter(pl.col("Value") > pl.col("Value").mean()).collect()[
        "Value"
    ].to_list() == list(range(12, 23))
    output = tmp_path / "concat.parquet"
    pl.concat([lazy, lazy]).sink_parquet(output, engine="streaming")
    assert pl.read_parquet(output)["Value"].to_list() == list(range(23)) * 2


def test_streaming_empty_file(tmp_path):
    path = tmp_path / "empty.tdms"
    with TdmsWriter(path) as writer:
        writer.write_segment([
            ChannelObject("G", "Empty", np.array([], dtype=np.int32)),
        ])
    lazy = pt.scan_tdms(path, group="G", chunk_size=5)
    output = tmp_path / "empty.parquet"
    lazy.sink_parquet(output, engine="streaming")
    assert pl.read_parquet(output).schema == {"Empty": pl.Int32}
    assert pl.read_parquet(output).height == 0


def test_streaming_read_errors_propagate(mixed_file, tmp_path, monkeypatch):
    original = pt._read_requested_range

    def read_range(handle, requests, start, end):
        if start >= 5:
            raise OSError("injected TDMS read failure")
        return original(handle, requests, start, end)

    monkeypatch.setattr(pt, "_read_requested_range", read_range)
    with pytest.raises(
        (OSError, pl.exceptions.ComputeError), match="injected TDMS read failure"
    ):
        pt.scan_tdms(mixed_file, group="G", chunk_size=5).sink_parquet(
            tmp_path / "failure.parquet", engine="streaming"
        )
