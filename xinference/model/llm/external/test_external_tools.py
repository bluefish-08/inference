# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""The tool-call family whitelist must not apply to external models."""

from ....api.restful_api import _supports_tool_calls

FAMILIES = {"deepseek-chat", "qwen2.5-instruct"}


def test_external_bypasses_the_whitelist():
    desc = {"model_format": "external"}
    assert _supports_tool_calls(desc, "deepseek-v4", FAMILIES) is True


def test_local_model_outside_the_whitelist_still_rejected():
    desc = {"model_format": "pytorch"}
    assert _supports_tool_calls(desc, "deepseek-v4", FAMILIES) is False


def test_local_model_inside_the_whitelist_allowed():
    desc = {"model_format": "pytorch"}
    assert _supports_tool_calls(desc, "deepseek-chat", FAMILIES) is True


def test_missing_format_falls_back_to_the_whitelist():
    assert _supports_tool_calls({}, "deepseek-v4", FAMILIES) is False


if __name__ == "__main__":
    for f in list(globals().values()):
        if callable(f) and getattr(f, "__name__", "").startswith("test_"):
            f()
    print("TOOLS_GATE_OK")
