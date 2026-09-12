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

import pytest

from ..media_source import (
    MediaSourceError,
    validate_media_source,
    validate_messages_media,
)


@pytest.mark.parametrize(
    "source",
    [
        "data:image/jpeg;base64,AAAA",
        "https://example.com/cat.png",
        "http://example.com/cat.png",
    ],
)
def test_remote_and_inline_sources_allowed(source):
    validate_media_source(source)


@pytest.mark.parametrize(
    "source",
    [
        # Observed in production: a scanner probing for local file reads.
        "/etc/hostname",
        "/etc/os-release",
        "/root/.xinference/config.json",
        "/root/.xinference/",
        "file:///etc/passwd",
        "relative/path.png",
    ],
)
def test_local_paths_rejected_by_default(source):
    with pytest.raises(MediaSourceError):
        validate_media_source(source)


def test_local_paths_allowed_when_opted_in():
    validate_media_source("/etc/hostname", allow_local_path=True)


def test_private_addresses_rejected_only_when_opted_in():
    # Default keeps intranet media hosts working.
    validate_media_source("http://127.0.0.1:8080/a.png")
    with pytest.raises(MediaSourceError):
        validate_media_source("http://127.0.0.1:8080/a.png", block_private_address=True)


def test_non_string_sources_pass_through():
    # Already-decoded images carry no fetch.
    validate_media_source(object())


def test_validate_messages_media_checks_openai_shape():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "/etc/hostname"}},
            ],
        }
    ]
    with pytest.raises(MediaSourceError):
        validate_messages_media(messages)


def test_validate_messages_media_checks_transformed_shape():
    messages = [
        {"role": "user", "content": [{"type": "image", "image": "file:///etc/passwd"}]}
    ]
    with pytest.raises(MediaSourceError):
        validate_messages_media(messages)


@pytest.mark.parametrize("key", ["video_url", "audio_url"])
def test_validate_messages_media_covers_video_and_audio(key):
    messages = [{"role": "user", "content": [{key: {"url": "/etc/hostname"}}]}]
    with pytest.raises(MediaSourceError):
        validate_messages_media(messages)


def test_validate_messages_media_tolerates_plain_content():
    validate_messages_media([{"role": "user", "content": "hello"}])
    validate_messages_media([])


def _engine_source(relative_path: str) -> str:
    """Return an engine module's text.

    Read as text rather than imported: these modules pull in vllm / sglang,
    which are not installed in the base test environment.
    """
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / relative_path
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "module, helpers",
    [
        (
            "llm/vllm/core.py",
            (
                "process_mm_info",
                "process_audio_info",
                "process_vision_info",
                "self._transform_messages",
            ),
        ),
        (
            "llm/sglang/core.py",
            ("process_vision_info", "self._transform_messages"),
        ),
    ],
)
def test_media_helpers_are_not_called_on_the_event_loop(module, helpers):
    # async_chat is awaited directly by the model actor, so a blocking download
    # inside it freezes every other request that actor serves -- including
    # terminate_model. Each helper must be dispatched through to_thread, which
    # takes the callable by name rather than calling it inline.
    lines = _engine_source(module).splitlines()
    for helper in helpers:
        called_inline = [
            line.strip()
            for line in lines
            # A bare "helper(" is an inline call; "to_thread(helper," is not.
            if f"{helper}(" in line and "import" not in line
        ]
        assert called_inline == [], (
            f"{module} calls {helper} inline, blocking the event loop: "
            f"{called_inline}"
        )
        assert any(
            f"asyncio.to_thread(" in line or f"{helper}," in line for line in lines
        ), f"{module} no longer dispatches {helper}"
