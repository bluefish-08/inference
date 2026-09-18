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
"""The cluster-metrics loop must be driven by the lifespan.

FastAPI(lifespan=...) makes Starlette ignore on_event("startup") handlers, so
registering the loop there left every supervisor gauge (workers_total,
supervisor_uptime, model_info, ...) permanently unpopulated.
"""

import asyncio

import pytest

from ..restful_api import RESTfulAPI


class _FakeAPI:
    def __init__(self, enabled):
        self._cluster_metrics_enabled = enabled
        self.loop_started = asyncio.Event()
        self.closed = False

    async def _cluster_metrics_update_loop(self):
        self.loop_started.set()
        await asyncio.sleep(3600)

    async def _close_token_router_client(self):
        self.closed = True


@pytest.mark.asyncio
async def test_lifespan_starts_metrics_loop():
    fake = _FakeAPI(enabled=True)
    async with RESTfulAPI._lifespan(fake, None):
        await asyncio.wait_for(fake.loop_started.wait(), timeout=1)
    assert fake.closed


@pytest.mark.asyncio
async def test_lifespan_skips_loop_when_metrics_disabled():
    fake = _FakeAPI(enabled=False)
    async with RESTfulAPI._lifespan(fake, None):
        await asyncio.sleep(0)
    assert not fake.loop_started.is_set()
    assert fake.closed
