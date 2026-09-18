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

import asyncio
import json
import time

import pytest

from ..stream_probe import StreamStats, probe_body


def _sse(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\r\n\r\n"


DELTA = {"choices": [{"delta": {"content": "hi"}}]}
USAGE_CHUNK = {
    "choices": [],
    "usage": {"prompt_tokens": 12, "completion_tokens": 40, "total_tokens": 52},
}


async def _drain(chunks, *, drop_usage_chunk=False, delay=0.0):
    async def source():
        for c in chunks:
            if delay:
                await asyncio.sleep(delay)
            yield c

    stats = StreamStats()
    out = [
        c
        async for c in probe_body(
            source(),
            stats,
            request_start=time.perf_counter(),
            drop_usage_chunk=drop_usage_chunk,
        )
    ]
    return stats, out


@pytest.mark.asyncio
async def test_collects_usage_from_final_sse_chunk():
    stats, out = await _drain([_sse(DELTA), _sse(DELTA), _sse(USAGE_CHUNK)])
    assert stats.prompt_tokens == 12
    assert stats.completion_tokens == 40
    assert stats.chunks == 3
    assert len(out) == 3, "client asked for usage, so the chunk must pass through"


@pytest.mark.asyncio
async def test_drops_injected_usage_chunk():
    stats, out = await _drain(
        [_sse(DELTA), _sse(USAGE_CHUNK)], drop_usage_chunk=True
    )
    assert stats.completion_tokens == 40, "still accounted for"
    assert out == [_sse(DELTA)], "but never reaches a client that did not ask"


@pytest.mark.asyncio
async def test_keeps_content_chunk_that_also_carries_usage():
    """Some engines attach usage to the last content delta; dropping it would
    swallow generated text."""
    merged = {"choices": [{"delta": {"content": "!"}}], "usage": {"completion_tokens": 3}}
    stats, out = await _drain([_sse(merged)], drop_usage_chunk=True)
    assert stats.completion_tokens == 3
    assert out == [_sse(merged)]


@pytest.mark.asyncio
async def test_non_streaming_json_body():
    body = json.dumps({"choices": [{"message": {"content": "hi"}}], "usage": {
        "prompt_tokens": 5, "completion_tokens": 7}}).encode()
    stats, out = await _drain([body])
    assert (stats.prompt_tokens, stats.completion_tokens) == (5, 7)
    assert out == [body]


@pytest.mark.asyncio
async def test_timings_and_tps():
    stats, _ = await _drain([_sse(DELTA), _sse(USAGE_CHUNK)], delay=0.02)
    assert stats.ttft_ms is not None and stats.ttft_ms > 0
    assert stats.duration_ms > stats.ttft_ms
    assert stats.output_tps is not None and stats.output_tps > 0


@pytest.mark.asyncio
async def test_duration_recorded_when_client_disconnects():
    async def source():
        yield _sse(DELTA)
        raise asyncio.CancelledError

    stats = StreamStats()
    with pytest.raises(asyncio.CancelledError):
        async for _ in probe_body(source(), stats, request_start=time.perf_counter()):
            pass
    assert stats.duration_ms > 0
    assert stats.chunks == 1


@pytest.mark.asyncio
async def test_ignores_done_sentinel_and_malformed_json():
    stats, out = await _drain([b'data: [DONE]\r\n\r\n', b'data: {"usage": broken}\r\n\r\n'])
    assert stats.completion_tokens is None
    assert len(out) == 2, "unparseable chunks still reach the client untouched"
