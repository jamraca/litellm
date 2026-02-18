import json

from litellm.llms.nvidia_nim.chat.transformation import NvidiaNimConfig


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
