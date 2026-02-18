"""
Nvidia NIM endpoint: https://docs.api.nvidia.com/nim/reference/databricks-dbrx-instruct-infer 

This is OpenAI compatible 

This file only contains param mapping logic

API calling is done using the OpenAI SDK with an api_base
"""
import json
import re
from typing import Optional

from litellm.types.utils import ChatCompletionMessageToolCall, Function

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class NvidiaNimConfig(OpenAIGPTConfig):
    """
    Reference: https://docs.api.nvidia.com/nim/reference/databricks-dbrx-instruct-infer

    The class `NvidiaNimConfig` provides configuration for the Nvidia NIM's Chat Completions API interface. Below are the parameters:
    """

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the supported OpenAI params for the given model


        Updated on July 5th, 2024 - based on https://docs.api.nvidia.com/nim/reference
        """
        if model in [
            "google/recurrentgemma-2b",
            "google/gemma-2-27b-it",
            "google/gemma-2-9b-it",
            "gemma-2-9b-it",
        ]:
            return ["stream", "temperature", "top_p", "max_tokens", "stop", "seed"]
        elif model == "nvidia/nemotron-4-340b-instruct":
            return [
                "stream",
                "temperature",
                "top_p",
                "max_tokens",
                "max_completion_tokens",
            ]
        elif model == "nvidia/nemotron-4-340b-reward":
            return [
                "stream",
            ]
        elif model in ["google/codegemma-1.1-7b"]:
            # most params - but no 'seed' :(
            return [
                "stream",
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "max_tokens",
                "max_completion_tokens",
                "stop",
            ]
        else:
            # DEFAULT Case - The vast majority of Nvidia NIM Models lie here
            # "upstage/solar-10.7b-instruct",
            # "snowflake/arctic",
            # "seallms/seallm-7b-v2.5",
            # "nvidia/llama3-chatqa-1.5-8b",
            # "nvidia/llama3-chatqa-1.5-70b",
            # "mistralai/mistral-large",
            # "mistralai/mixtral-8x22b-instruct-v0.1",
            # "mistralai/mixtral-8x7b-instruct-v0.1",
            # "mistralai/mistral-7b-instruct-v0.3",
            # "mistralai/mistral-7b-instruct-v0.2",
            # "mistralai/codestral-22b-instruct-v0.1",
            # "microsoft/phi-3-small-8k-instruct",
            # "microsoft/phi-3-small-128k-instruct",
            # "microsoft/phi-3-mini-4k-instruct",
            # "microsoft/phi-3-mini-128k-instruct",
            # "microsoft/phi-3-medium-4k-instruct",
            # "microsoft/phi-3-medium-128k-instruct",
            # "meta/llama3-70b-instruct",
            # "meta/llama3-8b-instruct",
            # "meta/llama2-70b",
            # "meta/codellama-70b",
            return [
                "stream",
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "max_tokens",
                "max_completion_tokens",
                "stop",
                "seed",
                "tools",
                "tool_choice",
                "parallel_tool_calls",
                "response_format",
            ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                optional_params[param] = value
        return optional_params

    def _check_and_fix_if_content_is_tool_call(
        self, content: str, optional_params: dict
    ) -> Optional[ChatCompletionMessageToolCall]:
        """
        Nvidia Kimi models can occasionally emit tool calls as plain text like:
        `Explore(: some free-form instruction)`.

        Convert this fallback format into an OpenAI-compatible tool call.
        """
        tool_call = super()._check_and_fix_if_content_is_tool_call(
            content=content, optional_params=optional_params
        )
        if tool_call is not None:
            return tool_call

        if optional_params.get("tools") is None:
            return None

        tool_call_match = re.match(
            r"^\s*([A-Za-z0-9_-]{1,64})\(\s*(.*?)\s*\)\s*$",
            content,
            re.DOTALL,
        )
        if tool_call_match is None:
            return None

        tool_name = tool_call_match.group(1)
        tool_argument_string = tool_call_match.group(2).strip()
        if tool_argument_string.startswith(":"):
            tool_argument_string = tool_argument_string[1:].strip()

        tool_definition = None
        for tool in optional_params.get("tools", []):
            if tool.get("function", {}).get("name") == tool_name:
                tool_definition = tool
                break

        if tool_definition is None:
            return None

        try:
            parsed_tool_arguments = json.loads(tool_argument_string)
            if isinstance(parsed_tool_arguments, str):
                raise ValueError("Expected object-like tool arguments")
            arguments = json.dumps(parsed_tool_arguments)
        except Exception:
            function_definition = tool_definition.get("function", {})
            parameters = function_definition.get("parameters", {})
            properties = parameters.get("properties", {})
            required_params = parameters.get("required", [])
            selected_param_name = (
                required_params[0]
                if isinstance(required_params, list) and len(required_params) > 0
                else next(iter(properties), "input")
            )
            arguments = json.dumps({selected_param_name: tool_argument_string})

        return ChatCompletionMessageToolCall(
            function=Function(name=tool_name, arguments=arguments)
        )
