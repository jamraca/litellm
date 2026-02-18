import json
from unittest.mock import MagicMock

from litellm.llms.nvidia_nim.chat.transformation import NvidiaNimConfig
from litellm.types.utils import ModelResponse


def test_nvidia_nim_fallback_tool_call_with_colon_argument_string():
    config = NvidiaNimConfig()
    optional_params = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Explore",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                        },
                        "required": ["task"],
                    },
                },
            }
        ]
    }

    tool_call = config._check_and_fix_if_content_is_tool_call(
        content="Explore(: Deep analysis of last DVC training run and performance gaps)",
        optional_params=optional_params,
    )

    assert tool_call is not None
    assert tool_call.function.name == "Explore"
    assert json.loads(tool_call.function.arguments) == {
        "task": "Deep analysis of last DVC training run and performance gaps"
    }


def test_nvidia_nim_fallback_tool_call_with_bullet_prefix_and_multiline_content():
    config = NvidiaNimConfig()
    optional_params = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Explore",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                        },
                        "required": ["task"],
                    },
                },
            }
        ]
    }

    tool_call = config._check_and_fix_if_content_is_tool_call(
        content=(
            "● Explore(: Explore existing options features)\n"
            "  ⎿ ❯ Search the codebase for existing options-related features"
        ),
        optional_params=optional_params,
    )

    assert tool_call is not None
    assert tool_call.function.name == "Explore"
    assert json.loads(tool_call.function.arguments) == {
        "task": "Explore existing options features"
    }


def test_nvidia_nim_fallback_tool_call_matches_tool_name_case_insensitively():
    config = NvidiaNimConfig()
    optional_params = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "explore",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                        },
                        "required": ["task"],
                    },
                },
            }
        ]
    }

    tool_call = config._check_and_fix_if_content_is_tool_call(
        content="Explore(: Explore existing options features)",
        optional_params=optional_params,
    )

    assert tool_call is not None
    assert tool_call.function.name == "explore"


def test_nvidia_nim_transform_response_normalizes_textual_tool_call_content():
    config = NvidiaNimConfig()
    optional_params = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Explore",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                        },
                        "required": ["task"],
                    },
                },
            }
        ]
    }

    raw_response = MagicMock()
    raw_response.text = "mock-response"
    raw_response.status_code = 200
    raw_response.headers = {}
    raw_response.json.return_value = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 123,
        "model": "nvidia/kimi-2.5",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "● Explore(: Explore existing options features)",
                },
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    logging_obj = MagicMock()
    logging_obj.model_call_details = {}

    transformed = config.transform_response(
        model="nvidia/kimi-2.5",
        raw_response=raw_response,
        model_response=ModelResponse(),
        logging_obj=logging_obj,
        request_data={},
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional_params,
        litellm_params={},
        encoding=None,
    )

    assert transformed.choices[0].message.content is None
    assert transformed.choices[0].finish_reason == "tool_calls"
    assert transformed.choices[0].message.tool_calls is not None
    assert transformed.choices[0].message.tool_calls[0].function.name == "Explore"
    assert json.loads(transformed.choices[0].message.tool_calls[0].function.arguments) == {
        "task": "Explore existing options features"
    }


def test_nvidia_nim_fallback_tool_call_with_shell_symbol_prefix():
    config = NvidiaNimConfig()
    optional_params = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                        },
                        "required": ["command"],
                    },
                },
            }
        ]
    }

    tool_call = config._check_and_fix_if_content_is_tool_call(
        content="> Bash(: ls -la)",
        optional_params=optional_params,
    )

    assert tool_call is not None
    assert tool_call.function.name == "Bash"
    assert json.loads(tool_call.function.arguments) == {"command": "ls -la"}


def test_nvidia_nim_fallback_tool_call_with_nested_shell_symbols():
    config = NvidiaNimConfig()
    optional_params = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                        },
                        "required": ["command"],
                    },
                },
            }
        ]
    }

    tool_call = config._check_and_fix_if_content_is_tool_call(
        content="⎿ ❯ Bash(: git status)",
        optional_params=optional_params,
    )

    assert tool_call is not None
    assert tool_call.function.name == "Bash"
    assert json.loads(tool_call.function.arguments) == {"command": "git status"}


def test_nvidia_nim_fallback_tool_call_preserves_nested_parentheses_in_content():
    config = NvidiaNimConfig()
    optional_params = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Explore",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                        },
                        "required": ["task"],
                    },
                },
            }
        ]
    }

    content = (
        "● Explore(: Deep analysis."
        " Search(pattern: \"catalyst/features/*options*\", path: \"src/(legacy)/options\"))"
    )
    tool_call = config._check_and_fix_if_content_is_tool_call(
        content=content,
        optional_params=optional_params,
    )

    assert tool_call is not None
    assert tool_call.function.name == "Explore"
    assert json.loads(tool_call.function.arguments) == {
        "task": "Deep analysis. Search(pattern: \"catalyst/features/*options*\", path: \"src/(legacy)/options\")"
    }


def test_nvidia_nim_fallback_tool_call_preserves_parentheses_for_bash_command():
    config = NvidiaNimConfig()
    optional_params = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                        },
                        "required": ["command"],
                    },
                },
            }
        ]
    }

    tool_call = config._check_and_fix_if_content_is_tool_call(
        content="> Bash(grep -E \"(foo|bar)\" file.txt)",
        optional_params=optional_params,
    )

    assert tool_call is not None
    assert tool_call.function.name == "Bash"
    assert json.loads(tool_call.function.arguments) == {
        "command": "grep -E \"(foo|bar)\" file.txt"
    }


def test_nvidia_nim_fallback_tool_call_supports_bash_without_colon_prefix():
    config = NvidiaNimConfig()
    optional_params = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                        },
                        "required": ["command"],
                    },
                },
            }
        ]
    }

    tool_call = config._check_and_fix_if_content_is_tool_call(
        content="Bash(grep -n \"TODO\" src/main.py)",
        optional_params=optional_params,
    )

    assert tool_call is not None
    assert tool_call.function.name == "Bash"
    assert json.loads(tool_call.function.arguments) == {
        "command": "grep -n \"TODO\" src/main.py"
    }
