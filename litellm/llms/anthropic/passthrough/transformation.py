"""
Anthropic Passthrough Configuration

Enables direct HTTP passthrough to Anthropic API while supporting LiteLLM callbacks.
This is used for OAuth token passthrough from Claude Code.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig

if TYPE_CHECKING:
    from httpx import URL

    from litellm.types.llms.openai import AllMessageValues


class AnthropicPassthroughConfig(BasePassthroughConfig):
    """
    Passthrough configuration for direct Anthropic API requests.

    Supports OAuth bearer token passthrough while enabling LiteLLM callbacks
    for context injection and conversation storage.
    """

    def is_streaming_request(self, endpoint: str, request_data: dict) -> bool:
        """
        Check if the request is a streaming request.

        For Anthropic, streaming is determined by the 'stream' field in request body.
        """
        return request_data.get("stream", False)

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        endpoint: str,
        request_query_params: Optional[dict],
        litellm_params: dict,
    ) -> Tuple["URL", str]:
        """
        Get the complete URL for the Anthropic API request.

        Returns:
            - complete_url: URL - the complete url for the request
            - base_target_url: str - the base url (for auth headers)
        """
        # Use configured api_base or default to Anthropic API
        base_target_url = api_base or os.getenv("ANTHROPIC_API_BASE") or "https://api.anthropic.com"

        # Ensure endpoint starts with '/'
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint

        # Build complete URL
        complete_url = self.format_url(
            endpoint=endpoint,
            base_target_url=base_target_url,
            request_query_params=request_query_params,
        )

        return complete_url, base_target_url

    # ========================================
    # BaseLLMModelInfo abstract method implementations
    # ========================================

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        """
        Returns a list of models supported by Anthropic.
        For passthrough, we don't enumerate - the model comes from the request.
        """
        return []

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        """
        Get API key from parameter or environment.
        For OAuth passthrough, the key comes from the Authorization header.
        """
        return api_key or os.getenv("ANTHROPIC_API_KEY")

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        """
        Get API base URL from parameter or environment.
        """
        return api_base or os.getenv("ANTHROPIC_API_BASE") or "https://api.anthropic.com"

    @property
    def forward_request_headers(self) -> bool:
        """
        Anthropic passthrough must forward client request headers.

        OAuth Bearer tokens arrive in the Authorization header from the client.
        Anthropic accepts Authorization: Bearer <token> with the
        anthropic-beta: oauth-2025-04-20 header. Without forwarding,
        the upstream request has no auth and returns 401.
        """
        return True

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List["AllMessageValues"],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate and prepare headers for Anthropic API.

        For OAuth passthrough, client headers (including Authorization) are
        forwarded via forward_request_headers=True. If an API key is available
        (from config or ANTHROPIC_API_KEY env), set x-api-key as fallback.
        """
        if api_key:
            headers["x-api-key"] = api_key
        return headers

    @staticmethod
    def get_base_model(model: str) -> Optional[str]:
        """
        Returns the base model name.
        For Anthropic, model names are typically passed as-is.
        """
        # Strip any provider prefix if present
        if model.startswith("anthropic/"):
            return model[len("anthropic/"):]
        return model
