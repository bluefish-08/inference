# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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
"""Derive per-request timing and token usage by observing the response body.

Sits at the HTTP boundary so it works for every engine and for both streaming
and non-streaming responses, and so the API key / user context is still at hand
(the worker-side metrics in ``core/metrics.py`` know the timings but not who
made the call).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

# Cheap pre-filter: most SSE deltas carry no usage, and parsing every one of
# them costs more than the whole probe is worth.
_USAGE_MARKER = b'"usage"'
_SSE_DATA_PREFIX = b"data:"


@dataclass
class StreamStats:
    """What one response turned out to cost."""

    ttft_ms: Optional[float] = None
    duration_ms: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    chunks: int = 0

    @property
    def output_tps(self) -> Optional[float]:
        """Decode speed: output tokens over the time after the first token."""
        if not self.completion_tokens or self.ttft_ms is None:
            return None
        decode_ms = self.duration_ms - self.ttft_ms
        if decode_ms <= 0:
            return None
        return round(self.completion_tokens / (decode_ms / 1000), 1)

    def as_audit_fields(self) -> dict:
        fields: dict[str, Any] = {}
        if self.ttft_ms is not None:
            fields["ttft_ms"] = round(self.ttft_ms, 1)
        if self.prompt_tokens is not None:
            fields["prompt_tokens"] = self.prompt_tokens
        if self.completion_tokens is not None:
            fields["completion_tokens"] = self.completion_tokens
        tps = self.output_tps
        if tps is not None:
            fields["output_tps"] = tps
        return fields


def _iter_payloads(chunk: bytes) -> list[dict]:
    """JSON objects carried by one body chunk, SSE-framed or bare."""
    if _USAGE_MARKER not in chunk:
        return []

    payloads = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(_SSE_DATA_PREFIX):
            line = line[len(_SSE_DATA_PREFIX) :].strip()
        if line == b"[DONE]" or not line.startswith(b"{"):
            continue
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _is_accounting_only_chunk(payload: dict) -> bool:
    """A usage chunk with no content, i.e. the one include_usage appends.

    Dropping it keeps the wire format identical for clients that never asked
    for ``stream_options.include_usage``.
    """
    return payload.get("usage") is not None and payload.get("choices") == []


async def probe_body(
    iterator: AsyncIterator[bytes],
    stats: StreamStats,
    *,
    request_start: float,
    drop_usage_chunk: bool = False,
) -> AsyncIterator[bytes]:
    """Pass the body through, recording timings and usage as it goes.

    ``request_start`` is a ``time.perf_counter()`` reading from before the
    handler ran, so ttft covers queueing and prefill too — not just the time
    the body spent streaming.
    """
    try:
        async for chunk in iterator:
            if stats.ttft_ms is None:
                stats.ttft_ms = (time.perf_counter() - request_start) * 1000
            stats.chunks += 1

            suppress = False
            for payload in _iter_payloads(chunk):
                usage = payload.get("usage")
                if isinstance(usage, dict):
                    prompt = usage.get("prompt_tokens")
                    completion = usage.get("completion_tokens")
                    if isinstance(prompt, int):
                        stats.prompt_tokens = prompt
                    if isinstance(completion, int):
                        stats.completion_tokens = completion
                if drop_usage_chunk and _is_accounting_only_chunk(payload):
                    suppress = True

            if suppress:
                continue
            yield chunk
    finally:
        # Also runs when the client disconnects mid-stream, so a cancelled
        # request still gets an audit record with what it consumed.
        stats.duration_ms = (time.perf_counter() - request_start) * 1000
