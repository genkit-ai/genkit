# Copyright 2025 Google LLC
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

"""xAI model implementations."""

import asyncio
import contextlib
import json
from typing import Any, cast

from pydantic import Field, ValidationError
from xai_sdk import Client as XAIClient
from xai_sdk.proto.v6 import chat_pb2, image_pb2

from genkit.ai import ActionRunContext
from genkit.blocks.model import get_basic_usage_stats
from genkit.core.logging import get_logger
from genkit.core.schema import to_json_schema
from genkit.plugins.xai.converters import DEFAULT_MAX_OUTPUT_TOKENS, FINISH_REASON_MAP
from genkit.plugins.xai.model_info import get_model_info
from genkit.types import (
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChunk,
    GenerationCommonConfig,
    GenerationUsage,
    MediaPart,
    Message,
    Part,
    Role,
    TextPart,
    ToolRequest,
    ToolRequestPart,
    ToolResponsePart,
)


class XAIConfig(GenerationCommonConfig):
    deferred: bool | None = None
    reasoning_effort: str | None = Field(None, pattern='^(low|medium|high)$')
    web_search_options: dict | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None


# Tool type mapping for xAI(function only, for now)
TOOL_TYPE_MAP = {
    'function': chat_pb2.ToolCallType.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
}


logger = get_logger(__name__)

__all__ = ['XAIModel']

ROLE_MAP = {
    Role.SYSTEM: chat_pb2.MessageRole.ROLE_SYSTEM,
    Role.USER: chat_pb2.MessageRole.ROLE_USER,
    Role.MODEL: chat_pb2.MessageRole.ROLE_ASSISTANT,
    Role.TOOL: chat_pb2.MessageRole.ROLE_TOOL,
}


def build_generation_usage(
    final_response: Any | None,  # noqa: ANN401
    basic_usage: GenerationUsage,
) -> GenerationUsage:
    """Builds a GenerationUsage object from a final_response and basic_usage."""
    return GenerationUsage(
        input_tokens=getattr(final_response.usage, 'prompt_tokens', 0) if final_response else 0,
        output_tokens=getattr(final_response.usage, 'completion_tokens', 0) if final_response else 0,
        total_tokens=getattr(final_response.usage, 'total_tokens', 0) if final_response else 0,
        input_characters=basic_usage.input_characters,
        output_characters=basic_usage.output_characters,
        input_images=basic_usage.input_images,
        output_images=basic_usage.output_images,
    )


class XAIModel:
    """xAI Grok model for Genkit."""

    def __init__(self, model_name: str, client: XAIClient) -> None:
        """Initialize the model."""
        model_info = get_model_info(model_name)
        self.model_name = model_info.versions[0] if model_info.versions else model_name
        self.client = client

    async def generate(self, request: GenerateRequest, ctx: ActionRunContext | None = None) -> GenerateResponse:
        """Generate content using the model.

        Args:
            request: The generate request.
            ctx: The action run context.

        Returns:
            The generate response.
        """
        params = self._build_params(request)
        streaming = ctx and ctx.is_streaming

        logger.debug('xAI generate request', model=self.model_name, streaming=bool(streaming))

        if streaming:
            assert ctx is not None  # streaming requires ctx
            return await self._generate_streaming(params, request, ctx)

        def _sample() -> Any:  # noqa: ANN401
            chat = self.client.chat.create(**cast(dict[str, Any], params))
            return chat.sample()

        response: Any = await asyncio.to_thread(_sample)
        logger.debug(
            'xAI raw API response',
            model=self.model_name,
            content=str(response.content)[:500] if response.content else None,
            tool_calls=str(response.tool_calls) if response.tool_calls else None,
            finish_reason=str(response.finish_reason),
        )
        content = self._to_genkit_content(response)
        response_message = Message(role=Role.MODEL, content=content)
        basic_usage = get_basic_usage_stats(input_=request.messages, response=response_message)

        return GenerateResponse(
            message=response_message,
            usage=build_generation_usage(response, basic_usage),
            finish_reason=FINISH_REASON_MAP.get(response.finish_reason, FinishReason.UNKNOWN),
        )

    def _build_params(self, request: GenerateRequest) -> dict[str, object]:
        """Build xAI API parameters from request using validated config."""
        config = request.config or {}
        if not isinstance(config, XAIConfig):
            try:
                config = XAIConfig.model_validate(config)
            except ValidationError:
                config = XAIConfig()

        params: dict[str, object] = {
            'model': self.model_name,
            'messages': self._to_xai_messages(request.messages),
            'max_tokens': int(config.max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS),
        }
        if config.temperature is not None:
            params['temperature'] = config.temperature
        if config.top_p is not None:
            params['top_p'] = config.top_p
        if config.stop_sequences:
            params['stop'] = config.stop_sequences
        if getattr(config, 'frequency_penalty', None) is not None:
            params['frequency_penalty'] = config.frequency_penalty
        if getattr(config, 'presence_penalty', None) is not None:
            params['presence_penalty'] = config.presence_penalty
        if config.web_search_options is not None:
            params['web_search_options'] = config.web_search_options
        if config.deferred is not None:
            params['deferred'] = config.deferred
        if config.reasoning_effort is not None:
            params['reasoning_effort'] = config.reasoning_effort

        # Structured output: set response_format for JSON output requests.
        if request.output and request.output.format == 'json':
            if request.output.schema:
                params['response_format'] = chat_pb2.ResponseFormat(
                    format_type=chat_pb2.FormatType.FORMAT_TYPE_JSON_SCHEMA,
                    schema=json.dumps(request.output.schema),
                )
            else:
                params['response_format'] = chat_pb2.ResponseFormat(
                    format_type=chat_pb2.FormatType.FORMAT_TYPE_JSON_OBJECT,
                )

        if request.tools:
            params['tools'] = [
                chat_pb2.Tool(
                    function=chat_pb2.Function(
                        name=t.name,
                        description=t.description or '',
                        parameters=json.dumps(to_json_schema(t.input_schema or {})),
                    ),
                )
                for t in request.tools
            ]

        return params

    async def _generate_streaming(
        self, params: dict[str, object], request: GenerateRequest, ctx: ActionRunContext
    ) -> GenerateResponse:
        loop = asyncio.get_running_loop()

        def _sync_stream() -> GenerateResponse:
            accumulated_content = []
            final_response = None

            chat = self.client.chat.create(**cast(dict[str, Any], params))
            for response, chunk in chat.stream():
                final_response = response

                if chunk.content:
                    loop.call_soon_threadsafe(
                        ctx.send_chunk,
                        GenerateResponseChunk(
                            role=Role.MODEL,
                            index=0,
                            content=[Part(root=TextPart(text=chunk.content))],
                        ),
                    )
                    accumulated_content.append(Part(root=TextPart(text=chunk.content)))

                for choice in chunk.choices:
                    if choice.tool_calls:
                        for tool_call in choice.tool_calls:
                            if tool_call.function:
                                tool_input = tool_call.function.arguments
                                if isinstance(tool_input, str):
                                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                                        tool_input = json.loads(tool_input)

                                tool_part = Part(
                                    root=ToolRequestPart(
                                        tool_request=ToolRequest(
                                            ref=tool_call.id,
                                            name=tool_call.function.name,
                                            input=tool_input,
                                        )
                                    )
                                )
                                loop.call_soon_threadsafe(
                                    ctx.send_chunk,
                                    GenerateResponseChunk(
                                        role=Role.MODEL,
                                        index=0,
                                        content=[tool_part],
                                    ),
                                )
                                accumulated_content.append(tool_part)

            response_message = Message(role=Role.MODEL, content=accumulated_content)
            basic_usage = get_basic_usage_stats(input_=request.messages, response=response_message)

            finish_reason = (
                FINISH_REASON_MAP.get(final_response.finish_reason, FinishReason.UNKNOWN)
                if final_response
                else FinishReason.UNKNOWN
            )

            return GenerateResponse(
                message=response_message,
                usage=build_generation_usage(final_response, basic_usage),
                finish_reason=finish_reason,
            )

        return await asyncio.to_thread(_sync_stream)

    def _to_xai_messages(self, messages: list[Message]) -> list[chat_pb2.Message]:
        result = []

        for msg in messages:
            # msg.role can be Role or str; ROLE_MAP keys are Role enum values
            if isinstance(msg.role, Role):
                role = ROLE_MAP.get(msg.role, chat_pb2.MessageRole.ROLE_USER)
            else:
                role = chat_pb2.MessageRole.ROLE_USER
            content = []
            tool_calls = []

            for part in msg.content:
                actual_part = part.root if isinstance(part, Part) else part

                if isinstance(actual_part, TextPart):
                    content.append(chat_pb2.Content(text=actual_part.text))
                elif isinstance(actual_part, MediaPart):
                    if not actual_part.media.url:
                        raise ValueError('xAI models require a URL for media parts.')
                    content.append(
                        chat_pb2.Content(image_url=image_pb2.ImageUrlContent(image_url=actual_part.media.url))
                    )
                elif isinstance(actual_part, ToolRequestPart):
                    tool_type = getattr(actual_part.tool_request, 'type', 'function')
                    tool_calls.append(
                        chat_pb2.ToolCall(
                            id=actual_part.tool_request.ref,
                            type=TOOL_TYPE_MAP.get(tool_type, chat_pb2.ToolCallType.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL),
                            function=chat_pb2.FunctionCall(
                                name=actual_part.tool_request.name,
                                arguments=json.dumps(actual_part.tool_request.input),
                            ),
                        )
                    )
                elif isinstance(actual_part, ToolResponsePart):
                    result.append(
                        chat_pb2.Message(
                            role=chat_pb2.MessageRole.ROLE_TOOL,
                            content=[chat_pb2.Content(text=str(actual_part.tool_response.output))],
                        )
                    )
                    continue

            pb_message = chat_pb2.Message(role=role, content=content or [chat_pb2.Content(text='')])
            if tool_calls:
                pb_message.tool_calls.extend(tool_calls)

            result.append(pb_message)

        return result

    def _to_genkit_content(self, response: Any) -> list[Part]:  # noqa: ANN401
        content: list[Part] = []

        if response.content:
            content.append(Part(root=TextPart(text=response.content)))

        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_input = tool_call.function.arguments
                if isinstance(tool_input, str):
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        tool_input = json.loads(tool_input)

                content.append(
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(
                                ref=tool_call.id,
                                name=tool_call.function.name,
                                input=tool_input,
                            )
                        )
                    )
                )

        return content
