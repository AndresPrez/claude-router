"""Request metrics for the hybrid router.

One JSONL record per request in the state directory (``metrics.jsonl``,
rotated at 10 MB). Records carry route, model, timing (duration, time to
first byte), token usage, and cache counters as reported by upstreams —
Z.ai and OpenRouter report usage inline in the response stream; the Cursor
bridge fetches it from the usage endpoint after a run.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any

from .paths import metrics_path

MAX_METRICS_BYTES = 10 * 1024 * 1024
_MAX_JSON_TEE_BYTES = 4 * 1024 * 1024
_WRITE_LOCK = threading.Lock()


def new_record(route: str, model: str) -> dict[str, Any]:
    return {
        "at": datetime.now(timezone.utc).isoformat(),
        "route": route,
        "model": model,
        "stream": None,
        "status": None,
        "error": None,
        "duration_ms": None,
        "ttft_ms": None,
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_tokens": None,
        "cache_creation_tokens": None,
        "tokens_per_sec": None,
        "decode_tokens_per_sec": None,
    }


def extract_usage_event(record: dict[str, Any], raw_event: bytes) -> None:
    """Update ``record`` in place from one upstream SSE event ( Anthropic shape)."""
    for line in raw_event.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data:"):
            continue
        try:
            document = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(document, dict):
            continue
        kind = document.get("type")
        if kind == "message_start":
            message = document.get("message")
            usage = message.get("usage") if isinstance(message, dict) else None
            if isinstance(usage, dict):
                record["input_tokens"] = _count(usage.get("input_tokens"))
                record["cache_read_tokens"] = _count(usage.get("cache_read_input_tokens"))
                record["cache_creation_tokens"] = _count(
                    usage.get("cache_creation_input_tokens")
                )
        elif kind == "message_delta":
            usage = document.get("usage")
            if isinstance(usage, dict):
                record["output_tokens"] = _count(usage.get("output_tokens"))
                # Z.ai reports the final cumulative input/cache usage here
                # instead of message_start; only overwrite present values.
                for field, key in (
                    ("input_tokens", "input_tokens"),
                    ("cache_read_tokens", "cache_read_input_tokens"),
                    ("cache_creation_tokens", "cache_creation_input_tokens"),
                ):
                    value = usage.get(key)
                    if isinstance(value, int) and value > 0:
                        record[field] = value
            if record["input_tokens"] is None:
                record["input_tokens"] = 0


def _count(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def fit_generation_curve(
    points: list[tuple[int, int]],
) -> tuple[float, float, float] | None:
    """Least-squares fit of ``generation_ms = floor + tokens * per_token_ms``.

    The slope isolates true decode speed from the fixed event-pipeline floor
    (SSE batching, turn finalization) that dominates short responses; the
    intercept is that floor in milliseconds. Returns ``(floor, slope, r2)``;
    a low r2 means the pooled data mixes regimes (for example peak and
    off-peak hours) and the fit should not be trusted.
    """
    if len(points) < 10:
        return None
    n = float(len(points))
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    var_x = sum((x - mean_x) ** 2 for x, _ in points)
    var_y = sum((y - mean_y) ** 2 for _, y in points)
    if var_x == 0 or var_y == 0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in points)
    slope = cov / var_x
    if slope <= 0:
        return None
    r2 = (cov * cov) / (var_x * var_y)
    return max(mean_y - slope * mean_x, 0.0), slope, r2


def parse_json_usage(record: dict[str, Any], raw: bytes) -> None:
    """Update ``record`` from a complete non-stream Anthropic JSON response."""
    try:
        document = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return
    usage = document.get("usage") if isinstance(document, dict) else None
    if not isinstance(usage, dict):
        return
    record["input_tokens"] = _count(usage.get("input_tokens"))
    record["output_tokens"] = _count(usage.get("output_tokens"))
    record["cache_read_tokens"] = _count(usage.get("cache_read_input_tokens"))
    record["cache_creation_tokens"] = _count(usage.get("cache_creation_input_tokens"))


def apply_cursor_usage(record: dict[str, Any], usage: dict[str, Any] | None) -> None:
    """Fold a Cursor ``Get Agent Usage`` run-usage object into ``record``."""
    if not isinstance(usage, dict):
        return
    record["input_tokens"] = _count(usage.get("inputTokens"))
    record["output_tokens"] = _count(usage.get("outputTokens"))
    record["cache_read_tokens"] = _count(usage.get("cacheReadTokens"))
    record["cache_creation_tokens"] = _count(usage.get("cacheWriteTokens"))


class MetricsRecorder:
    """Collect one request's metrics and append a record on ``finish``."""

    def __init__(self, route: str, model: str) -> None:
        self.record = new_record(route, model)
        self._started = time.monotonic()
        self._finished = False

    def mark_first_byte(self) -> None:
        if self.record["ttft_ms"] is None:
            self.record["ttft_ms"] = int((time.monotonic() - self._started) * 1000)

    def observe_sse(self, raw_event: bytes) -> None:
        extract_usage_event(self.record, raw_event)

    def observe_json(self, raw: bytes) -> None:
        parse_json_usage(self.record, raw[:_MAX_JSON_TEE_BYTES])

    def finish(
        self,
        status: int,
        *,
        error: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        if usage is not None:
            apply_cursor_usage(self.record, usage)
        duration = time.monotonic() - self._started
        self.record["status"] = status
        self.record["error"] = error[:500] if error else None
        self.record["duration_ms"] = int(duration * 1000)
        output = self.record.get("output_tokens")
        if isinstance(output, int) and output > 0 and duration > 0:
            self.record["tokens_per_sec"] = round(output / duration, 2)
            ttft = self.record.get("ttft_ms")
            generation = duration - (ttft / 1000 if isinstance(ttft, int) else 0)
            # Short responses can arrive inside the first read, leaving no
            # measurable generation phase; a floor keeps the rate meaningful.
            if generation >= 0.1 and output >= 100:
                self.record["decode_tokens_per_sec"] = round(output / generation, 2)
        write_record(self.record)


def write_record(record: dict[str, Any]) -> None:
    path = metrics_path()
    line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
    with _WRITE_LOCK:
        _rotate_if_needed(path, len(line.encode()))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def _rotate_if_needed(path: Path, incoming: int) -> None:
    try:
        size = path.stat().st_size if path.exists() else 0
    except OSError:
        return
    if size == 0 or size + incoming <= MAX_METRICS_BYTES:
        return
    rotated = path.with_suffix(".jsonl.1")
    try:
        rotated.unlink(missing_ok=True)
        path.rename(rotated)
    except OSError:
        pass


def load_records(
    days: int,
    model_filter: str | None = None,
    route_filter: str | None = None,
) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    needle = model_filter.casefold() if model_filter else None
    route_needle = route_filter.casefold() if route_filter else None
    records: list[dict[str, Any]] = []
    path = metrics_path()
    for candidate in (path.with_suffix(".jsonl.1"), path):
        try:
            handle = candidate.open("r", encoding="utf-8")
        except FileNotFoundError:
            continue
        with handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                at = record.get("at")
                if isinstance(at, str):
                    try:
                        if datetime.fromisoformat(at) < cutoff:
                            continue
                    except ValueError:
                        pass
                if needle and needle not in str(record.get("model", "")).casefold():
                    continue
                if route_needle and route_needle not in str(record.get("route", "")).casefold():
                    continue
                records.append(record)
    return records


def summarize(records: list[dict[str, Any]], min_tokens: int = 100) -> dict[str, Any]:
    """Aggregate records; decode rates count responses of >= min_tokens."""
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "requests": 0,
            "errors": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "output_tokens_streamed": 0,
            "stream_seconds": 0.0,
            "decode_output_tokens": 0,
            "generation_seconds": 0.0,
            "gen_points": [],
            "ttft_total_ms": 0,
            "ttft_samples": 0,
        }
    )
    for record in records:
        key = (str(record.get("route")), str(record.get("model")))
        bucket = groups[key]
        bucket["requests"] += 1
        if record.get("error"):
            bucket["errors"] += 1
        for field in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_creation_tokens",
        ):
            value = record.get(field)
            if isinstance(value, int):
                bucket[field] += value
        output = record.get("output_tokens")
        duration = record.get("duration_ms")
        ttft = record.get("ttft_ms")
        if (
            record.get("stream")
            and isinstance(output, int)
            and output > 0
            and isinstance(duration, int)
            and duration > 0
        ):
            bucket["output_tokens_streamed"] += output
            bucket["stream_seconds"] += duration / 1000
            # Decode rate only counts responses with a measurable generation
            # phase; tiny responses are chunk-arrival-bound, not decode-bound.
            generation_ms = max(duration - (ttft if isinstance(ttft, int) else 0), 100)
            if output >= 100:
                bucket["decode_output_tokens"] += output
                bucket["generation_seconds"] += generation_ms / 1000
            if output >= 20:
                bucket["gen_points"].append((output, generation_ms))
        ttft = record.get("ttft_ms")
        if isinstance(ttft, int) and ttft >= 0:
            bucket["ttft_total_ms"] += ttft
            bucket["ttft_samples"] += 1

    models = []
    totals = {
        "requests": 0,
        "errors": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }
    for (route, model), bucket in sorted(groups.items()):
        tps = (
            bucket["output_tokens_streamed"] / bucket["stream_seconds"]
            if bucket["stream_seconds"] > 0
            else None
        )
        decode_tps = (
            bucket["decode_output_tokens"] / bucket["generation_seconds"]
            if bucket["generation_seconds"] > 0
            else None
        )
        curve = fit_generation_curve(bucket["gen_points"])
        # Only trust the fit when generation time actually tracks length;
        # pooled peak/off-peak mixtures produce confident nonsense.
        fit_tps = 1000.0 / curve[1] if curve and curve[2] >= 0.5 else None
        floor_ms = round(curve[0]) if curve and curve[2] >= 0.5 else None
        avg_ttft = (
            bucket["ttft_total_ms"] / bucket["ttft_samples"]
            if bucket["ttft_samples"]
            else None
        )
        models.append(
            {
                "route": route,
                "model": model,
                "requests": bucket["requests"],
                "errors": bucket["errors"],
                "input_tokens": bucket["input_tokens"],
                "output_tokens": bucket["output_tokens"],
                "cache_read_tokens": bucket["cache_read_tokens"],
                "cache_creation_tokens": bucket["cache_creation_tokens"],
                "tokens_per_sec": round(tps, 2) if tps else None,
                "decode_tokens_per_sec": round(decode_tps, 2) if decode_tps else None,
                "fit_decode_tokens_per_sec": round(fit_tps, 1) if fit_tps else None,
                "turn_floor_ms": floor_ms,
                "avg_ttft_ms": round(avg_ttft) if avg_ttft is not None else None,
            }
        )
        for field in totals:
            totals[field] += bucket[field]
    return {"models": models, "totals": totals}


def format_summary(
    days: int,
    model_filter: str | None = None,
    route_filter: str | None = None,
    min_tokens: int = 100,
) -> str:
    records = load_records(days, model_filter, route_filter)
    if not records:
        return f"No recorded requests in the last {days} day(s) at {metrics_path()}."
    summary = summarize(records, min_tokens)
    scope = "" if min_tokens == 100 else f" (decode >= {min_tokens}-token responses)"
    lines = [
        f"Router metrics — last {days} day(s) — {summary['totals']['requests']} request(s){scope}",
        "",
        f"{'route':<11} {'model':<22} {'req':>5} {'err':>4} {'in tok':>10} "
        f"{'out tok':>9} {'cache rd':>10} {'cache wr':>10} {'tok/s':>7} "
        f"{'dec t/s':>7} {'fit t/s':>7} {'ttft ms':>8}",
    ]
    for row in summary["models"]:
        lines.append(
            f"{row['route']:<11.11} {row['model']:<22.22} {row['requests']:>5} "
            f"{row['errors']:>4} {row['input_tokens']:>10} {row['output_tokens']:>9} "
            f"{row['cache_read_tokens']:>10} {row['cache_creation_tokens']:>10} "
            f"{_fmt(row['tokens_per_sec']):>7} {_fmt(row['decode_tokens_per_sec']):>7} "
            f"{_fmt(row['fit_decode_tokens_per_sec']):>7} "
            f"{_fmt(row['avg_ttft_ms']):>8}"
        )
    totals = summary["totals"]
    lines.append(
        f"{'TOTAL':<11} {'':<22} {totals['requests']:>5} {totals['errors']:>4} "
        f"{totals['input_tokens']:>10} {totals['output_tokens']:>9} "
        f"{totals['cache_read_tokens']:>10} {totals['cache_creation_tokens']:>10}"
    )
    return "\n".join(lines)


def format_histogram(
    days: int,
    model_filter: str | None = None,
    route_filter: str | None = None,
    min_tokens: int = 100,
) -> str:
    """Render requests per hour as an ASCII histogram segmented by route."""
    records = load_records(days, model_filter, route_filter)
    if model_filter:
        needle = model_filter.casefold()
        records = [r for r in records if needle in str(r.get("model", "")).casefold()]
    if not records:
        return f"No recorded requests in the last {days} day(s) at {metrics_path()}."

    hours: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "routes": defaultdict(int),
            "tps": [],
            "decode": [],
            "gen_points": [],
            "ttft": [],
            "errors": 0,
        }
    )
    for record in records:
        at = record.get("at")
        try:
            hour = datetime.fromisoformat(str(at)).strftime("%m-%d %H:00")
        except ValueError:
            continue
        bucket = hours[hour]
        bucket["routes"][str(record.get("route"))] += 1
        if record.get("error"):
            bucket["errors"] += 1
        if isinstance(record.get("tokens_per_sec"), (int, float)):
            bucket["tps"].append(record["tokens_per_sec"])
        out = record.get("output_tokens")
        dur = record.get("duration_ms")
        ttft = record.get("ttft_ms")
        if isinstance(out, int) and isinstance(dur, int) and dur > 0:
            generation = max(dur - (ttft if isinstance(ttft, int) else 0), 100)
            if out >= max(min_tokens, 100):
                bucket["decode"].append(out / generation * 1000)
            if out >= 20:
                bucket["gen_points"].append((out, generation))
        if isinstance(record.get("ttft_ms"), int):
            bucket["ttft"].append(record["ttft_ms"])

    def hour_fit(points: list[tuple[int, int]]) -> float | None:
        curve = fit_generation_curve(points)
        return round(1000.0 / curve[1], 1) if curve and curve[2] >= 0.5 else None

    characters = {
        "anthropic": "█",
        "zai": "▓",
        "cursor": "░",
        "wafer": "▒",
        "fireworks": "▚",
        "inco": "▞",
        "openrouter": "░",
        "rejected": "·",
    }
    width = 26
    busiest = max(
        (sum(bucket["routes"].values()) for bucket in hours.values()), default=0
    )
    scale = max(busiest, 1) / width

    scope = f" — model ~{model_filter}" if model_filter else ""
    lines = [
        f"Requests per hour — last {days} day(s) — {len(records)} total{scope}",
        "",
        f"{'hour':<12} {'req':>4} {'err':>4} {'distribution':<28} "
        f"{'tok/s':>7} {'dec t/s':>8} {'fit t/s':>8} {'ttft':>6}",
    ]
    for hour in sorted(hours):
        bucket = hours[hour]
        total = sum(bucket["routes"].values())
        bar = "".join(
            characters.get(route, "?") * max(int(round(count / scale)), 1 if count else 0)
            for route, count in sorted(bucket["routes"].items())
        )[:width]
        tps = f"{median(bucket['tps']):.1f}" if bucket["tps"] else "-"
        decode = f"{median(bucket['decode']):.1f}" if bucket["decode"] else "-"
        if min_tokens != 100 and not bucket["decode"]:
            decode = f"<{min_tokens}"
        fit = hour_fit(bucket["gen_points"])
        fit_txt = f"{fit:.1f}" if fit else "-"
        ttft = f"{median(bucket['ttft']) / 1000:.1f}s" if bucket["ttft"] else "-"
        lines.append(
            f"{hour:<12} {total:>4} {bucket['errors']:>4} {bar:<28} "
            f"{tps:>7} {decode:>8} {fit_txt:>8} {ttft:>6}"
        )
    lines.append("")
    lines.append(
        "legend: " + ", ".join(f"{char} {name}" for name, char in characters.items())
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)
