# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Proxy engine forwarding chat requests to an external OpenAI-compatible server."""

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from ....types import ChatCompletion, ChatCompletionChunk
from ..core import LLM

if TYPE_CHECKING:
    from ..llm_family import LLMFamilyV2, LLMSpecV1

logger = logging.getLogger(__name__)

# Params the OpenAI SDK accepts directly; anything else the backend understands
# (top_k, repetition_penalty, chat_template_kwargs, ...) is tunnelled via extra_body.
_OPENAI_PARAMS = frozenset(
    {
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "n",
        "parallel_tool_calls",
        "presence_penalty",
        "response_format",
        "seed",
        "stop",
        "stream_options",
        "temperature",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "user",
    }
)
# Xinference-internal keys that must never reach the remote server.
_INTERNAL_PARAMS = frozenset({"lora_name", "request_id", "stream_interval", "echo"})


class ExternalChatModel(LLM):
    # The remote server does its own continuous batching; without this the model
    # actor wraps every request in a global asyncio.Lock and serialises them.
    allow_batch = True

    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV2",
        model_path: str,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_uid, model_family, model_path)
        config = dict(model_config or {})
        self._base_url = config.pop("base_url", None)
        self._api_key = config.pop("api_key", "EMPTY")
        self._remote_model_name = config.pop("remote_model_name", None)
        self._timeout = float(config.pop("timeout", 600))
        self._client = None

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        import importlib.util

        if importlib.util.find_spec("openai") is None:
            return False, "openai is required by the External engine"
        return True

    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if llm_spec.model_format != "external":
            return False, "External engine only supports model_format 'external'"
        if "chat" not in (llm_family.model_ability or []):
            return False, "External engine requires the chat ability"
        return True

    def load(self):
        from openai import AsyncOpenAI

        if not self._base_url:
            raise ValueError(
                f"External model {self.model_uid} requires 'base_url', e.g. "
                "--base_url http://10.0.0.1:8100/v1"
            )
        if not self._remote_model_name:
            raise ValueError(
                f"External model {self.model_uid} requires 'remote_model_name', "
                "the model name served by the remote endpoint"
            )
        self._client = AsyncOpenAI(
            base_url=self._base_url, api_key=self._api_key, timeout=self._timeout
        )
        logger.info(
            "External model %s proxies to %s (remote model %s)",
            self.model_uid,
            self._base_url,
            self._remote_model_name,
        )

    def _split_config(
        self, generate_config: Optional[Dict]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        config = dict(generate_config or {})
        stream = bool(config.pop("stream", False))
        config.pop("model", None)
        kwargs: Dict[str, Any] = {}
        extra_body: Dict[str, Any] = {}
        for key, value in config.items():
            if value is None or key in _INTERNAL_PARAMS:
                continue
            if key in _OPENAI_PARAMS:
                kwargs[key] = value
            else:
                extra_body[key] = value
        return kwargs, extra_body, stream

    async def async_chat(
        self,
        messages: List[Dict],
        generate_config: Optional[Dict] = None,
        request_id: Optional[str] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        assert self._client is not None, "External model is not loaded"
        kwargs, extra_body, stream = self._split_config(generate_config)
        if extra_body:
            kwargs["extra_body"] = extra_body

        if not stream:
            completion = await self._client.chat.completions.create(
                model=self._remote_model_name,
                messages=messages,
                stream=False,
                **kwargs,
            )
            return completion.model_dump()  # type: ignore[return-value]

        async def _stream() -> AsyncGenerator[ChatCompletionChunk, None]:
            remote_stream = await self._client.chat.completions.create(
                model=self._remote_model_name,
                messages=messages,
                stream=True,
                **kwargs,
            )
            async for chunk in remote_stream:
                yield chunk.model_dump()  # type: ignore[misc]

        return _stream()
