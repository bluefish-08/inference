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

"""Validation for client-supplied media references.

Chat requests can carry ``image_url`` / ``video_url`` / ``audio_url`` values
that the server fetches itself. Whatever accepts them is therefore a
server-side fetcher driven by untrusted input, which without limits is both an
SSRF primitive and a local-file-read primitive.
"""

import ipaddress
import logging
import socket
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from ..constants import (
    XINFERENCE_MEDIA_ALLOW_LOCAL_PATH,
    XINFERENCE_MEDIA_BLOCK_PRIVATE_ADDRESS,
)

logger = logging.getLogger(__name__)

REMOTE_SCHEMES = ("http://", "https://")
DATA_URI_PREFIX = "data:"
FILE_URI_PREFIX = "file://"


class MediaSourceError(ValueError):
    """Raised when a client-supplied media reference must not be fetched."""


def _resolved_addresses(host: str) -> List[str]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        # Let the fetcher surface the resolution failure with its own error.
        return []
    return [str(info[4][0]) for info in infos]


def _is_blocked_address(host: str) -> bool:
    for address in _resolved_addresses(host):
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return True
    return False


def validate_media_source(
    source: Any,
    *,
    allow_local_path: Optional[bool] = None,
    block_private_address: Optional[bool] = None,
) -> None:
    """Raise :class:`MediaSourceError` if ``source`` must not be fetched.

    Non-string sources (already-decoded ``PIL.Image``, numpy arrays, ...) are
    passed through: they carry no fetch.
    """
    if not isinstance(source, str):
        return
    if allow_local_path is None:
        allow_local_path = XINFERENCE_MEDIA_ALLOW_LOCAL_PATH
    if block_private_address is None:
        block_private_address = XINFERENCE_MEDIA_BLOCK_PRIVATE_ADDRESS

    reference = source.strip()
    if reference.startswith(DATA_URI_PREFIX):
        return

    if reference.startswith(REMOTE_SCHEMES):
        if not block_private_address:
            return
        try:
            host = urlparse(reference).hostname
        except ValueError as e:
            # Malformed URLs (bad IPv6 brackets, ...) raise instead of parsing.
            raise MediaSourceError(f"Invalid media URL: {e}") from e
        # Resolution here is advisory: the fetcher resolves again, so a rebinding
        # DNS entry can still slip through. Network policy remains the real control.
        if host and _is_blocked_address(host):
            raise MediaSourceError(
                "Refusing to fetch media from a private or loopback address. "
                "Unset XINFERENCE_MEDIA_BLOCK_PRIVATE_ADDRESS to allow it."
            )
        return

    # Anything else is read off the server's filesystem: an explicit file://
    # URI, or a bare path, which is what a permissive fetcher falls back to.
    if allow_local_path:
        return
    raise MediaSourceError(
        "Refusing to read media from the server filesystem. Send the media as a "
        "data: URI or an http(s) URL, or set XINFERENCE_MEDIA_ALLOW_LOCAL_PATH=1 "
        "to allow local paths."
    )


_MEDIA_CONTENT_KEYS = ("image_url", "video_url", "audio_url")
_TRANSFORMED_MEDIA_KEYS = ("image", "video", "audio")


def validate_messages_media(
    messages: Union[List[Dict], List[Any]],
) -> None:
    """Validate every media reference carried by OpenAI-style chat messages.

    Accepts both the raw OpenAI shape (``{"image_url": {"url": ...}}``) and the
    transformed shape (``{"type": "image", "image": ...}``) so it can guard the
    message list before or after normalisation.
    """
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            for key in _MEDIA_CONTENT_KEYS:
                value = item.get(key)
                if isinstance(value, dict):
                    validate_media_source(value.get("url"))
                elif value is not None:
                    validate_media_source(value)
            for key in _TRANSFORMED_MEDIA_KEYS:
                if key in item:
                    validate_media_source(item[key])
