# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Cohere AI model integration for Genkit.

This module provides the model implementation for Cohere AI,
converting between Genkit and Cohere SDK (V2 API) formats.

Uses ``cohere.AsyncClientV2`` for chat completions with support
for tool calling, structured output, and streaming.

See:
    - https://docs.cohere.com/reference/chat
    - https://docs.cohere.com/docs/models
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

import cohere
from cohere.v2.types.v2chat_response import V2ChatResponse
from cohere.v2.types.v2chat_stream_response import (
    ContentDeltaV2ChatStreamResponse,
    MessageEndV2ChatStreamResponse,
    ToolCallDeltaV2ChatStreamResponse,
    ToolCallStartV2ChatStreamResponse,
)
from genkit.core.action import ActionRunContext
from genkit.core.logging import get_logger
from genkit.core.typing import (
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChunk,
    Message,
    Part,
    Role,
    TextPart,
    ToolRequest,
    ToolRequestPart,
)
from genkit.plugins.cohere.converters import (
    FINISH_REASON_MAP,
    convert_messages,
    convert_response,
    convert_tools,
    extract_content_delta_text,
    extract_finish_reason,
    extract_tool_call_delta_args,
    extract_tool_call_start,
    get_response_format,
    parse_tool_arguments,
)
from genkit.plugins.cohere.model_info import (
    SUPPORTED_COHERE_MODELS,
    get_default_model_info,
)

COHERE_PLUGIN_NAME = 'cohere'

logger = get_logger(__name__)


def cohere_name(name: str) -> str:
    """Create a Cohere action name.

    Args:
        name: Base name for the action.

    Returns:
        The fully qualified Cohere action name.
    """
    return f'{COHERE_PLUGIN_NAME}/{name}'


class CohereConfig(BaseModel):
    """Configuration options for Cohere chat completions (V2 API).

    See: https://docs.cohere.com/reference/chat

    Attributes:
        temperature: Sampling temperature (0.0–1.0). Lower = more deterministic.
        max_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling probability cutoff (0.01–0.99).
        top_k: Top-K sampling cutoff (0–500). 0 disables.
        frequency_penalty: Penalises token frequency (0.0–1.0).
        presence_penalty: Penalises token presence (0.0–1.0).
        seed: Seed for deterministic sampling.
        stop_sequences: Stop generation when these sequences appear.
    """

    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, ge=0.01, le=0.99)
    top_k: int | None = Field(default=None, ge=0, le=500)
    frequency_penalty: float | None = Field(default=None, ge=0.0, le=1.0)
    presence_penalty: float | None = Field(default=None, ge=0.0, le=1.0)
    seed: int | None = None
    stop_sequences: list[str] | None = None


# Config keys forwarded to the Cohere V2 chat API.
_CONFIG_KEYS = (
    'temperature',
    'max_tokens',
    'stop_sequences',
    'seed',
    'frequency_penalty',
    'presence_penalty',
)

# Cohere V2 API parameter name mappings: Genkit config → API parameter.
_CONFIG_ALIASES: dict[str, str] = {
    'top_k': 'k',
    'top_p': 'p',
}


class CohereModel:
    """Manages Cohere AI model integration for Genkit.

    This class provides integration with Cohere's official Python SDK,
    allowing Cohere models to be exposed as Genkit models via the V2 API.

    All type-conversion logic is delegated to :mod:`converters` — this
    class is responsible only for orchestrating API calls and managing
    the streaming lifecycle.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
    ) -> None:
        """Initialize the Cohere model instance.

        Args:
            model: The name of the specific Cohere model.
            api_key: Cohere API key for authentication.
        """
        self.name = model
        self.client = cohere.AsyncClientV2(api_key=api_key)

    def get_model_info(self) -> dict[str, Any] | None:
        """Retrieve metadata and supported features for the specified model.

        Returns:
            A dictionary containing the model's 'name' and 'supports' features.
        """
        model_info = SUPPORTED_COHERE_MODELS.get(self.name, get_default_model_info(self.name))
        supports_dict = model_info.supports.model_dump() if model_info.supports else {}
        return {
            'name': model_info.label,
            'supports': supports_dict,
        }

    async def generate(
        self,
        request: GenerateRequest,
        ctx: ActionRunContext | None = None,
    ) -> GenerateResponse:
        """Generate a response using the Cohere model.

        Args:
            request: The generation request.
            ctx: Optional action run context for streaming.

        Returns:
            The generated response.
        """
        messages = convert_messages(request.messages)

        # Build request parameters.
        params: dict[str, Any] = {
            'model': self.name,
            'messages': messages,
        }

        # Add tools if provided.
        if request.tools:
            params['tools'] = convert_tools(request.tools)

        # Handle structured output.
        if request.output:
            response_format = get_response_format(request.output)
            if response_format:
                params['response_format'] = response_format

        # Apply config if provided.
        if request.config:
            config = request.config
            if isinstance(config, dict):
                for key in _CONFIG_KEYS:
                    if config.get(key) is not None:
                        params[key] = config[key]
                # Handle aliased parameters (top_k → k, top_p → p).
                for genkit_key, api_key in _CONFIG_ALIASES.items():
                    if config.get(genkit_key) is not None:
                        params[api_key] = config[genkit_key]

        # Handle streaming.
        if ctx and ctx.send_chunk:
            logger.debug('Cohere generate request', model=self.name, streaming=True)
            return await self._generate_streaming(params, ctx)

        # Non-streaming request.
        logger.debug('Cohere generate request', model=self.name, streaming=False)
        response: V2ChatResponse = await self.client.chat(**params)
        logger.debug(
            'Cohere raw API response',
            model=self.name,
            message_content=str(response.message.content) if response.message else None,
            tool_calls=str(response.message.tool_calls) if response.message and response.message.tool_calls else None,
            finish_reason=str(response.finish_reason),
        )
        return convert_response(response)

    async def _generate_streaming(
        self,
        params: dict[str, Any],
        ctx: ActionRunContext,
    ) -> GenerateResponse:
        """Generate a streaming response.

        Uses the Cohere V2 chat_stream API. Stream events are typed
        discriminated unions; extraction helpers in :mod:`converters`
        use ``getattr`` for safe attribute access since each event
        type has different fields.

        Args:
            params: Request parameters.
            ctx: Action run context with send_chunk callback.

        Returns:
            The complete generated response.
        """
        full_text = ''
        finish_reason: FinishReason = FinishReason.STOP
        accumulated_content: list[Part] = []

        # Track tool calls being streamed (by index).
        tool_calls: dict[int, dict[str, str]] = {}

        async for event in self.client.chat_stream(**params):
            if isinstance(event, ContentDeltaV2ChatStreamResponse):
                text = extract_content_delta_text(event)
                if text:
                    full_text += text
                    ctx.send_chunk(
                        GenerateResponseChunk(
                            role=Role.MODEL,
                            content=[Part(root=TextPart(text=text))],
                        )
                    )

            elif isinstance(event, ToolCallStartV2ChatStreamResponse):
                idx = event.index if event.index is not None else 0
                tc_id, tc_name = extract_tool_call_start(event)
                tool_calls[idx] = {
                    'id': tc_id,
                    'name': tc_name,
                    'arguments': '',
                }

            elif isinstance(event, ToolCallDeltaV2ChatStreamResponse):
                idx = event.index if event.index is not None else 0
                if idx not in tool_calls:
                    tool_calls[idx] = {'id': '', 'name': '', 'arguments': ''}
                args_chunk = extract_tool_call_delta_args(event)
                if args_chunk:
                    tool_calls[idx]['arguments'] += args_chunk

            elif isinstance(event, MessageEndV2ChatStreamResponse):
                fr = extract_finish_reason(event)
                if fr:
                    finish_reason = FINISH_REASON_MAP.get(fr, FinishReason.OTHER)

        # Build final content.
        if full_text:
            accumulated_content.append(Part(root=TextPart(text=full_text)))

        # Add accumulated tool calls.
        for tc in tool_calls.values():
            args: dict[str, Any] | str = {}
            if tc['arguments']:
                args = parse_tool_arguments(tc['arguments'])

            tool_part = Part(
                root=ToolRequestPart(
                    tool_request=ToolRequest(
                        ref=tc['id'] or None,
                        name=tc['name'],
                        input=args,
                    )
                )
            )
            accumulated_content.append(tool_part)
            ctx.send_chunk(
                GenerateResponseChunk(
                    role=Role.MODEL,
                    content=[tool_part],
                )
            )

        return GenerateResponse(
            message=Message(role=Role.MODEL, content=accumulated_content),
            finish_reason=finish_reason,
        )

    def to_generate_fn(self) -> Callable:
        """Convert the Cohere model into a Genkit-compatible generate function.

        Returns:
            A callable function that can be used by Genkit.
        """
        return self.generate
