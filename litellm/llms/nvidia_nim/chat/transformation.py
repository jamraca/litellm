"""
Nvidia NIM endpoint: https://docs.api.nvidia.com/nim/reference/databricks-dbrx-instruct-infer 

This is OpenAI compatible 

This file only contains param mapping logic

API calling is done using the OpenAI SDK with an api_base
"""
import json
import re
from typing import Any, List, Optional

from litellm.types.utils import ChatCompletionMessageToolCall, Function
from litellm.types.utils import Message, ModelResponse

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

    @staticmethod
    def _extract_balanced_tool_call(content: str) -> Optional[tuple[str, str]]:
        """
        Extract tool name + argument string from textual invocations while preserving
        argument content exactly (including nested parentheses).
        """
        tool_call_start = re.search(
            # Handles variants such as:
            # - Explore(: task)
            # - ● Explore(: task)
            # - > Bash(: ls -la)
            # - ⎿ ❯ Bash(: git status)
            r"(?:^|\n)\s*(?:(?:[\-\*•●>❯⎿]+|❯)\s*)*([A-Za-z0-9_-]{1,64})\(",
            content,
        )
        if tool_call_start is None:
            return None

        tool_name = tool_call_start.group(1)
        opening_paren_idx = content.find("(", tool_call_start.start(1) + len(tool_name))
        if opening_paren_idx == -1:
            return None

        depth = 0
        closing_paren_idx = -1
        for i, char in enumerate(content[opening_paren_idx:], start=opening_paren_idx):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    closing_paren_idx = i
                    break

        if closing_paren_idx == -1:
            return None

        argument_string = content[opening_paren_idx + 1 : closing_paren_idx].strip()
        return tool_name, argument_string

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

        extracted_tool_call = self._extract_balanced_tool_call(content=content)
        if extracted_tool_call is None:
            return None

        matched_tool_name, tool_argument_string = extracted_tool_call
        if tool_argument_string.startswith(":"):
            tool_argument_string = tool_argument_string[1:].strip()

        tool_definition = None
        canonical_tool_name = matched_tool_name
        for tool in optional_params.get("tools", []):
            tool_name = tool.get("function", {}).get("name")
            if isinstance(tool_name, str) and tool_name.lower() == matched_tool_name.lower():
                tool_definition = tool
                canonical_tool_name = tool_name
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
            function=Function(name=canonical_tool_name, arguments=arguments)
        )

    def _normalize_textual_tool_call_in_message(
        self,
        message: Message,
        optional_params: dict,
    ) -> bool:
        """
        Convert textual fallback tool calls into structured tool_calls.

        Returns True when conversion happened.
        """
        content = message.content
        if message.tool_calls is not None or not isinstance(content, str):
            return False

        parsed_tool_call = self._check_and_fix_if_content_is_tool_call(
            content=content,
            optional_params=optional_params,
        )
        if parsed_tool_call is None:
            return False

        message.tool_calls = [parsed_tool_call]
        message.content = None
        return True

    def transform_response(
        self,
        model: str,
        raw_response: Any,
        model_response: ModelResponse,
        logging_obj: Any,
        request_data: dict,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        transformed_response = super().transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )

        converted_tool_call = False
        if transformed_response.choices is not None:
            for choice in transformed_response.choices:
                if self._normalize_textual_tool_call_in_message(
                    message=choice.message,
                    optional_params=optional_params,
                ):
                    converted_tool_call = True
                    if choice.finish_reason in (None, "stop"):
                        choice.finish_reason = "tool_calls"

        if converted_tool_call and transformed_response._hidden_params is not None:
            transformed_response._hidden_params["nvidia_nim_text_tool_call_fixed"] = True

        return transformed_response
