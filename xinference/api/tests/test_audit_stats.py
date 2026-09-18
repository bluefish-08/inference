# Copyright 2022-2026 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from datetime import datetime, timedelta, timezone

import pytest

from ..routers.admin import _compute_audit_stats, get_audit_stats

NOW = datetime.now(timezone.utc)


def _entry(  # noqa: D401
    minutes_ago,
    user,
    model,
    status="success",
    latency=100.0,
    tokens=None,
):
    entry = {
        "@timestamp": (NOW - timedelta(minutes=minutes_ago))
        .isoformat()
        .replace("+00:00", "Z"),
        "user": user,
        "api_key_name": "key",
        "model_id": model + "-uid",
        "model_name": model,
        "model_type": "llm",
        "endpoint": "/v1/chat/completions",
        "status": status,
        "category": "inference",
        "auth_type": "api_key",
        "latency_ms": latency,
        "client_ip": "10.0.0.1",
    }
    if tokens:
        entry.update(tokens)
    return entry


STREAMED = {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "ttft_ms": 80.0,
    "output_tps": 25.0,
    "stream": True,
}

ENTRIES = [
    _entry(10, "alice", "qwen3", latency=100.0, tokens=STREAMED),
    _entry(20, "alice", "qwen3", latency=300.0, tokens=STREAMED),
    _entry(30, "bob", "qwen3", latency=200.0),
    _entry(40, "bob", "deepseek", status="error", latency=50.0),
    _entry(1000, "carol", "deepseek", latency=400.0),  # outside a 1h window
]


@pytest.fixture
def audit_log(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "audit.log").write_text(
        "\n".join(json.dumps(e) for e in ENTRIES[:3]) + "\n", encoding="utf-8"
    )
    # Rotated file: must be picked up too, otherwise multi-day ranges under-count.
    (log_dir / "audit.log.2026-09-16").write_text(
        "\n".join(json.dumps(e) for e in ENTRIES[3:]) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr("xinference.constants.XINFERENCE_LOG_DIR", str(log_dir))
    monkeypatch.delenv("XINFERENCE_ES_URL", raising=False)
    return log_dir


def test_compute_stats():
    stats = _compute_audit_stats(
        ENTRIES[:4], t_from=NOW - timedelta(hours=1), t_to=NOW, top_n=10
    )
    assert stats["total"] == 4
    assert stats["errors"] == 1
    assert stats["unique_users"] == 2
    assert stats["unique_models"] == 2
    # latency now samples streamed calls only, so the two non-streaming
    # entries (200 ms, 50 ms) are out and the median moves.
    assert stats["latency"]["p50"] == 100.0

    by_model = {row["key"]: row for row in stats["by_model"]}
    assert by_model["qwen3"]["count"] == 3
    assert by_model["qwen3"]["share"] == 75.0
    assert by_model["deepseek"]["errors"] == 1

    # Grouping is by API key, not by user: one account can hold several keys
    # and that is the split worth seeing.
    by_key = {row["key"]: row for row in stats["by_api_key"]}
    assert set(by_key) == {"key"}
    assert by_key["key"]["models"] == 2
    assert stats["unique_api_keys"] == 1

    # Token/speed fields are optional per entry; the rollup must ignore the
    # rows that lack them rather than treating them as zero.
    assert stats["prompt_tokens"] == 200
    assert stats["completion_tokens"] == 100
    assert stats["ttft"]["avg"] == 80.0
    assert stats["tps"]["avg"] == 25.0
    assert by_model["qwen3"]["completion_tokens"] == 100
    assert by_model["deepseek"]["completion_tokens"] == 0

    # Only streamed calls carry a meaningful total duration.
    assert stats["latency"]["max"] == 300.0

    # Every group gets a bucket-aligned sparkline of the same length.
    assert len(by_model["qwen3"]["series"]) == len(stats["series"])
    assert sum(by_model["qwen3"]["series"]) == 3

    # The series covers the whole window with zero-filled buckets.
    assert stats["interval"] == "5m"
    assert len(stats["series"]) == 13
    assert sum(b["count"] for b in stats["series"]) == 4


@pytest.mark.asyncio
async def test_stats_endpoint_reads_rotated_files(audit_log):
    resp = await get_audit_stats(time_from="now-30d", time_to="now")
    data = json.loads(resp.body)
    assert data["total"] == 5
    assert data["unique_users"] == 3


@pytest.mark.asyncio
async def test_stats_endpoint_filters(audit_log):
    resp = await get_audit_stats(time_from="now-30d", time_to="now", user="alice")
    data = json.loads(resp.body)
    assert data["total"] == 2
    assert data["unique_users"] == 1
