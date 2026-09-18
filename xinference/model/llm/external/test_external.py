# Copyright 2022-2026 Xinference Holdings Pte. Ltd
import asyncio
import types

import pytest

from .core import ExternalChatModel


class _Family:
    def __init__(self, ability):
        self.model_ability = ability
        self.model_name = "ext"
        self.model_specs = [types.SimpleNamespace(quantization="none")]


class _Spec:
    def __init__(self, fmt):
        self.model_format = fmt


def test_match():
    chat, vision = _Family(["chat"]), _Family(["chat", "vision"])
    assert ExternalChatModel.match_json(chat, _Spec("external"), "none") is True
    assert ExternalChatModel.match_json(chat, _Spec("pytorch"), "none") != True
    assert (
        ExternalChatModel.match_json(_Family(["generate"]), _Spec("external"), "none")
        != True
    )
    # one class serves both; vision is gated by the family's model_ability
    assert ExternalChatModel.match_json(vision, _Spec("external"), "none") is True


def test_split_config():
    m = ExternalChatModel.__new__(ExternalChatModel)
    kwargs, extra, stream = m._split_config(
        {
            "max_tokens": 16,
            "temperature": 0.5,
            "stream": True,
            "top_k": 20,
            "chat_template_kwargs": {"thinking": True},
            "request_id": "r1",
            "lora_name": None,
            "stop": None,
        }
    )
    assert stream is True
    assert kwargs == {"max_tokens": 16, "temperature": 0.5}
    assert extra == {"top_k": 20, "chat_template_kwargs": {"thinking": True}}
    # internal keys never leak to the remote server
    assert "request_id" not in kwargs and "request_id" not in extra
    assert "lora_name" not in extra and "stop" not in kwargs


def test_load_requires_endpoint():
    m = ExternalChatModel.__new__(ExternalChatModel)
    m.model_uid, m._base_url, m._remote_model_name = "u", None, None
    try:
        m.load()
    except ValueError as e:
        assert "base_url" in str(e)
    else:
        raise AssertionError("missing base_url must fail")


def test_forwards_and_streams():
    class _Chunk:
        def __init__(self, i):
            self.i = i

        def model_dump(self):
            return {"id": self.i}

    class _Completions:
        def __init__(self):
            self.seen = {}

        async def create(self, **kw):
            self.seen = kw
            if kw["stream"]:

                async def gen():
                    for i in range(2):
                        yield _Chunk(i)

                return gen()
            return _Chunk("done")

    comp = _Completions()
    m = ExternalChatModel.__new__(ExternalChatModel)
    m._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=comp))
    m._remote_model_name = "dsv4-vision"

    out = asyncio.run(
        m.async_chat([{"role": "user", "content": "hi"}], {"max_tokens": 8})
    )
    assert out == {"id": "done"}
    assert comp.seen["model"] == "dsv4-vision" and comp.seen["max_tokens"] == 8

    async def collect():
        agen = await m.async_chat([{"role": "user", "content": "hi"}], {"stream": True})
        return [c async for c in agen]

    assert asyncio.run(collect()) == [{"id": 0}, {"id": 1}]


if __name__ == "__main__":
    test_match()
    test_split_config()
    test_load_requires_endpoint()
    test_forwards_and_streams()
    print("EXTERNAL_OK")


def test_allow_batch_is_enabled():
    # ModelActor serialises every request behind a lock when this is False,
    # which silently caps the remote endpoint at one concurrent request.
    assert ExternalChatModel.allow_batch is True


def test_remote_error_is_picklable_and_keeps_the_message():
    # The original bug: openai errors carry an httpx.Response, so unpickling
    # them across the actor boundary raised TypeError about missing kwargs,
    # replacing the real error with a meaningless one.
    import pickle

    import httpx
    from openai import APIConnectionError, BadRequestError, InternalServerError

    from .core import _remote_error

    req = httpx.Request("POST", "http://remote/v1/chat/completions")
    body = {
        "error": {"message": "This model's maximum context length is 131072 tokens"}
    }
    resp = httpx.Response(400, request=req, json=body)

    for original, expected in (
        (BadRequestError("bad", response=resp, body=body), ValueError),
        (
            InternalServerError(
                "boom", response=httpx.Response(500, request=req), body=None
            ),
            RuntimeError,
        ),
        (APIConnectionError(request=req), RuntimeError),
    ):
        with pytest.raises(TypeError):
            pickle.loads(pickle.dumps(original))  # the bug, still present upstream

        converted = _remote_error(original)
        assert isinstance(converted, expected)
        assert pickle.loads(pickle.dumps(converted)).args == converted.args

    assert "maximum context length" in str(
        _remote_error(BadRequestError("bad", response=resp, body=body))
    )
    assert "HTTP 400" in str(
        _remote_error(BadRequestError("bad", response=resp, body=body))
    )


def test_async_chat_converts_remote_errors_on_both_paths():
    import pickle

    import httpx
    from openai import BadRequestError

    req = httpx.Request("POST", "http://remote/v1/chat/completions")
    body = {"error": {"message": "maximum context length is 131072 tokens"}}

    def _fresh():
        # a new object per raise: reusing one accumulates traceback/context
        return BadRequestError(
            "bad", response=httpx.Response(400, request=req, json=body), body=body
        )

    class _Completions:
        async def create(self, **kw):
            raise _fresh()

    m = ExternalChatModel.__new__(ExternalChatModel)
    m._client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_Completions())
    )
    m._remote_model_name = "remote"
    msgs = [{"role": "user", "content": "hi"}]

    async def collect():
        agen = await m.async_chat(msgs, {"stream": True})
        return [c async for c in agen]

    for call in (
        lambda: asyncio.run(m.async_chat(msgs, {})),
        lambda: asyncio.run(collect()),
    ):
        with pytest.raises(Exception) as excinfo:
            call()
        raised = excinfo.value
        assert not isinstance(raised, BadRequestError), "openai error must not escape"
        assert "maximum context length" in str(raised)
        # __context__ would otherwise still hold the unpicklable openai error
        assert raised.__context__ is None
        pickle.loads(pickle.dumps(raised))  # must survive the actor boundary
