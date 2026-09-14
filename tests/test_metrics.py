"""Tests for router request metrics (metrics.jsonl)."""

import json
from datetime import datetime, timezone

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

    clock = {"now": 0.0}
    monkeypatch.setattr(metrics_module.time, "monotonic", lambda: clock["now"])
    recorder = MetricsRecorder("zai", "glm-5.3-flash")
    clock["now"] = 0.5
    recorder.mark_first_byte()
    clock["now"] = 0.6
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


def test_summarize_decodes_only_measurable_streams(tmp_path, monkeypatch) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(metrics_module, "metrics_path", lambda: metrics_path)
    for out, dur, ttft in ((500, 4000, 1000), (10, 3000, 1000)):
        write_record(
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "route": "inco",
                "model": "glm-5.3-flash:fast",
                "stream": True,
                "status": 200,
                "error": None,
                "duration_ms": dur,
                "ttft_ms": ttft,
                "input_tokens": 1,
                "output_tokens": out,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
                "tokens_per_sec": None,
                "decode_tokens_per_sec": None,
            }
        )

    row = summarize(load_records(days=1))["models"][0]

    # only the 500-token request counts: 500 tokens / 3s generation
    assert row["decode_tokens_per_sec"] == round(500 / 3, 2)
    # e2e still counts both: 510 tokens / 7s
    assert row["tokens_per_sec"] == round(510 / 7, 2)


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
    assert "dec t/s" in out
    assert "fit t/s" in out


def test_recorder_computes_decode_rate(tmp_path, monkeypatch) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(metrics_module, "metrics_path", lambda: metrics_path)
    clock = {"now": 0.0}
    monkeypatch.setattr(metrics_module.time, "monotonic", lambda: clock["now"])

    recorder = MetricsRecorder("inco", "glm-5.3-flash:fast")
    clock["now"] = 0.5
    recorder.mark_first_byte()
    clock["now"] = 3.0
    recorder.observe_sse(
        b'event: message_delta\ndata: {"type":"message_delta","usage":'
        b'{"output_tokens":100}}\n\n'
    )
    recorder.finish(200)
    record = json.loads(metrics_path.read_text().strip())

    assert record["tokens_per_sec"] is not None
    assert record["decode_tokens_per_sec"] is not None
    assert record["decode_tokens_per_sec"] >= record["tokens_per_sec"]


def test_decode_rate_skips_tiny_responses(tmp_path, monkeypatch) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(metrics_module, "metrics_path", lambda: metrics_path)

    clock = {"now": 0.0}
    monkeypatch.setattr(metrics_module.time, "monotonic", lambda: clock["now"])
    recorder = MetricsRecorder("zai", "glm-5.3-flash")
    clock["now"] = 0.5
    recorder.mark_first_byte()
    clock["now"] = 0.6
    recorder.observe_sse(
        b'event: message_delta\ndata: {"type":"message_delta","usage":'
        b'{"output_tokens":8}}\n\n'
    )
    recorder.finish(200)
    record = json.loads(metrics_path.read_text().strip())

    assert record["decode_tokens_per_sec"] is None  # too small to measure


def test_fit_generation_curve_recovers_slope_and_floor() -> None:
    # generation_ms = 1500ms floor + 10ms per token -> 100 tok/s, 1500ms floor
    points = [(tokens, 1500 + 10 * tokens) for tokens in range(20, 200, 7)]
    floor, slope, r2 = metrics_module.fit_generation_curve(points)
    assert r2 > 0.99

    assert floor == 1500
    assert slope == 10


def test_fit_generation_curve_needs_samples() -> None:
    assert metrics_module.fit_generation_curve([(100, 2000)]) is None
    assert metrics_module.fit_generation_curve([(100, 2000)] * 10) is None  # zero variance


def test_min_tokens_threshold_raises_decode_population(tmp_path, monkeypatch) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(metrics_module, "metrics_path", lambda: metrics_path)
    for out in (150, 300, 800):
        write_record(
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "route": "inco",
                "model": "glm-5.3-flash:fast",
                "stream": True,
                "status": 200,
                "error": None,
                "duration_ms": 2000 + out * 10,
                "ttft_ms": 1000,
                "input_tokens": 1,
                "output_tokens": out,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
                "tokens_per_sec": None,
                "decode_tokens_per_sec": None,
            }
        )

    records = load_records(days=1)
    row100 = summarize(records, min_tokens=100)["models"][0]
    row500 = summarize(records, min_tokens=500)["models"][0]
    totals500 = summarize(records, min_tokens=500)["totals"]

    # at 100 all three count: 1250 tokens over (2.5+4+9)s = 80.65
    assert row100["requests"] == 3
    assert row100["decode_tokens_per_sec"] == round(1250 / 15.5, 2)
    # at 500 every column reflects only the 800-token response
    assert row500["requests"] == 1
    assert totals500["requests"] == 1
    assert totals500["output_tokens"] == 800
    assert row500["decode_tokens_per_sec"] == round(800 / 9.0, 2)


def test_floor_from_tiny_turns(tmp_path, monkeypatch) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(metrics_module, "metrics_path", lambda: metrics_path)
    for out, dur, ttft in ((5, 3200, 800), (12, 3500, 900), (8, 3400, 850), (600, 9000, 1000)):
        write_record(
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "route": "inco",
                "model": "glm-5.3-flash:fast",
                "stream": True,
                "status": 200,
                "error": None,
                "duration_ms": dur,
                "ttft_ms": ttft,
                "input_tokens": 1,
                "output_tokens": out,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
                "tokens_per_sec": None,
                "decode_tokens_per_sec": None,
                "effort": None,
            }
        )

    row = summarize(load_records(days=1))["models"][0]

    # tiny turns (5, 12, 8) give generation 2400, 2600, 2550 -> median 2550
    assert row["floor_ms"] == 2550
