"""Tests for router request metrics (metrics.jsonl)."""

import json

from claude_router import metrics as metrics_module
from claude_router.metrics import (
    MetricsRecorder,
    extract_usage_event,
    format_summary,
    load_records,
    summarize,
    write_record,
)


def test_extract_usage_from_anthropic_events() -> None:
    record = metrics_module.new_record("zai", "glm-5.3-flash")

    extract_usage_event(
        record,
        b'event: message_start\ndata: {"type":"message_start","message":{"usage":'
        b'{"input_tokens":234,"cache_read_input_tokens":7200,"cache_creation_input_tokens":5}}}\n\n',
    )
    extract_usage_event(
        record,
        b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":56}}\n\n',
    )

    assert record["input_tokens"] == 234
    assert record["output_tokens"] == 56
    assert record["cache_read_tokens"] == 7200
    assert record["cache_creation_tokens"] == 5


def test_extract_usage_ignores_garbage() -> None:
    record = metrics_module.new_record("zai", "glm-5.3-flash")

    extract_usage_event(record, b"event: ping\ndata: not-json\n\n")

    assert record["input_tokens"] is None


def test_recorder_computes_speed_and_writes(tmp_path, monkeypatch) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(metrics_module, "metrics_path", lambda: metrics_path)

    recorder = MetricsRecorder("zai", "glm-5.3-flash")
    recorder.mark_first_byte()
    recorder.observe_sse(
        b'event: message_start\ndata: {"type":"message_start","message":{"usage":'
        b'{"input_tokens":100}}}\n\n'
    )
    recorder.observe_sse(
        b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":50}}\n\n'
    )
    recorder.finish(200)

    records = load_records(days=1)
    assert len(records) == 1
    record = records[0]
    assert record["route"] == "zai"
    assert record["model"] == "glm-5.3-flash"
    assert record["status"] == 200
    assert record["input_tokens"] == 100
    assert record["output_tokens"] == 50
    assert record["duration_ms"] >= 0
    assert isinstance(record["ttft_ms"], int)
    assert record["tokens_per_sec"] is not None


def test_apply_cursor_usage_maps_camel_case() -> None:
    record = metrics_module.new_record("cursor", "composer-2.5")

    metrics_module.apply_cursor_usage(
        record,
        {"inputTokens": 6320, "outputTokens": 1450, "cacheReadTokens": 21300,
         "cacheWriteTokens": 7100},
    )

    assert record["input_tokens"] == 6320
    assert record["output_tokens"] == 1450
    assert record["cache_read_tokens"] == 21300
    assert record["cache_creation_tokens"] == 7100


def test_summarize_aggregates_per_model(tmp_path, monkeypatch) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(metrics_module, "metrics_path", lambda: metrics_path)
    write_record(
        {
            "at": "2026-09-10T12:00:00+00:00",
            "route": "zai",
            "model": "glm-5.3-flash",
            "stream": True,
            "status": 200,
            "error": None,
            "duration_ms": 1000,
            "ttft_ms": 400,
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_tokens": 200,
            "cache_creation_tokens": 10,
            "tokens_per_sec": 50.0,
        }
    )
    write_record(
        {
            "at": "2026-09-10T13:00:00+00:00",
            "route": "zai",
            "model": "glm-5.3-flash",
            "stream": True,
            "status": 200,
            "error": None,
            "duration_ms": 2000,
            "ttft_ms": 600,
            "input_tokens": 100,
            "output_tokens": 100,
            "cache_read_tokens": 300,
            "cache_creation_tokens": 0,
            "tokens_per_sec": 50.0,
        }
    )

    summary = summarize(load_records(days=7))

    assert summary["totals"]["requests"] == 2
    assert summary["totals"]["input_tokens"] == 200
    assert summary["totals"]["output_tokens"] == 150
    assert summary["totals"]["cache_read_tokens"] == 500
    row = summary["models"][0]
    assert row["model"] == "glm-5.3-flash"
    assert row["tokens_per_sec"] == 50.0  # 150 tokens / 3 seconds
    assert row["avg_ttft_ms"] == 500


def test_format_summary_renders_table(tmp_path, monkeypatch, capsys) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(metrics_module, "metrics_path", lambda: metrics_path)
    write_record(
        {
            "at": "2026-09-10T12:00:00+00:00",
            "route": "cursor",
            "model": "composer-2.5",
            "stream": False,
            "status": 200,
            "error": None,
            "duration_ms": 42000,
            "ttft_ms": 30000,
            "input_tokens": 7982,
            "output_tokens": 46,
            "cache_read_tokens": 6688,
            "cache_creation_tokens": 0,
            "tokens_per_sec": None,
        }
    )

    out = format_summary(days=7)

    assert "cursor" in out
    assert "composer-2.5" in out
    assert "7982" in out


def test_rotation_rewrites_old_file(tmp_path, monkeypatch) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(metrics_module, "metrics_path", lambda: metrics_path)
    monkeypatch.setattr(metrics_module, "MAX_METRICS_BYTES", 300)
    record = {"at": "2026-09-10T12:00:00+00:00", "route": "zai", "model": "m"}

    for _ in range(5):
        write_record(record)

    rotated = metrics_path.with_suffix(".jsonl.1")
    assert metrics_path.exists()
    lines = metrics_path.read_text().strip().splitlines()
    rotated_lines = len(rotated.read_text().strip().splitlines()) if rotated.exists() else 0
    assert len(lines) + rotated_lines == 5


def test_extract_usage_reads_cumulative_message_delta() -> None:
    """Z.ai reports final input/cache usage in message_delta, unlike Anthropic."""
    record = metrics_module.new_record("zai", "glm-5.3-flash")

    extract_usage_event(
        record,
        b'event: message_start\ndata: {"type":"message_start","message":{"usage":'
        b'{"input_tokens":0,"cache_read_input_tokens":0}}}\n\n',
    )
    extract_usage_event(
        record,
        b'event: message_delta\ndata: {"type":"message_delta","usage":'
        b'{"input_tokens":15,"output_tokens":8,"cache_read_input_tokens":2240}}\n\n',
    )

    assert record["input_tokens"] == 15
    assert record["output_tokens"] == 8
    assert record["cache_read_tokens"] == 2240


def test_format_histogram_renders_scaled_bars(tmp_path, monkeypatch) -> None:
    from claude_router.metrics import format_histogram

    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(metrics_module, "metrics_path", lambda: metrics_path)
    for i in range(4):
        write_record(
            {
                "at": f"2026-09-10T1{i}:15:00+00:00",
                "route": "zai",
                "model": "glm-5.3",
                "stream": True,
                "status": 200,
                "error": None,
                "duration_ms": 2000,
                "ttft_ms": 1000,
                "input_tokens": 10,
                "output_tokens": 100,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
                "tokens_per_sec": 50.0,
            }
        )

    out = format_histogram(days=7)

    assert "Requests per hour" in out
    assert "legend" in out
    assert "09-10 1" in out
    assert "▓" in out
    assert "50.0" in out


def test_recorder_computes_decode_rate(tmp_path, monkeypatch) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(metrics_module, "metrics_path", lambda: metrics_path)

    recorder = MetricsRecorder("inco", "glm-5.3-flash:fast")
    recorder.mark_first_byte()
    recorder.observe_sse(
        b'event: message_delta\ndata: {"type":"message_delta","usage":'
        b'{"output_tokens":100}}\n\n'
    )
    recorder.finish(200)
    record = json.loads(metrics_path.read_text().strip())

    assert record["tokens_per_sec"] is not None
    assert record["decode_tokens_per_sec"] is not None
    assert record["decode_tokens_per_sec"] >= record["tokens_per_sec"]
