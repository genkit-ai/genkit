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

"""Microsoft Foundry model implementation for Genkit.

This module implements the model interface for Microsoft Foundry chat completions
using the OpenAI-compatible API.

See:
- Microsoft Foundry: https://ai.azure.com/
- Model Catalog: https://ai.azure.com/catalog/models
- SDK Overview: https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/sdk-overview

Key features:
- Chat completions with GPT-4o, GPT-4, GPT-3.5, o1/o3/o4 series, GPT-5 series
- Claude, DeepSeek, Grok, Llama, Mistral models
- Tool/function calling support
- Streaming responses
- Multimodal inputs (images with visual_detail_level control)
- JSON output mode
"""

from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.lib._pydantic import _ensure_strict_json_schema
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage

from genkit.ai import ActionRunContext
from genkit.core.logging import get_logger
from genkit.plugins.microsoft_foundry.models.converters import (
    build_usage,
    from_openai_tool_calls,
    map_finish_reason,
    normalize_config,
    parse_tool_call_args,
    to_openai_messages,
    to_openai_tool,
)
from genkit.plugins.microsoft_foundry.models.model_info import MODELS_SUPPORTING_RESPONSE_FORMAT, get_model_info
from genkit.plugins.microsoft_foundry.typing import MicrosoftFoundryConfig, VisualDetailLevel
from genkit.types import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChunk,
    GenerationUsage,
    Message,
    Part,
    Role,
    TextPart,
    ToolRequest,
    ToolRequestPart,
)

logger = get_logger(__name__)


class MicrosoftFoundryModel:
    """Microsoft Foundry model for chat completions.

    This class handles the conversion between Genkit's message format
    and the Microsoft Foundry/Azure OpenAI chat completion API format.

    Attributes:
        model_name: The model name (e.g., 'gpt-4o', 'DeepSeek-V3.2').
        client: AsyncAzureOpenAI or AsyncOpenAI client instance.
        deployment: Optional deployment name override.
    """

    def __init__(
        self,
        model_name: str,
        client: AsyncAzureOpenAI | AsyncOpenAI,
        deployment: str | None = None,
    ) -> None:
        """Initialize the Microsoft Foundry model.

        Args:
            model_name: Name of the model (e.g., 'gpt-4o', 'DeepSeek-V3.2').
            client: AsyncAzureOpenAI or AsyncOpenAI client instance.
            deployment: Optional deployment name (defaults to model_name).
        """
        self.model_name = model_name
        self.client = client
        self.deployment = deployment or model_name

    async def generate(
        self,
        request: GenerateRequest,
        ctx: ActionRunContext | None = None,
    ) -> GenerateResponse:
        """Generate a response from Azure OpenAI.

        Args:
            request: The generation request containing messages and config.
            ctx: Action run context for streaming support.

        Returns:
            GenerateResponse with the model's output.
        """
        config = normalize_config(request.config)
        params = self._build_request_body(request, config)
        streaming = ctx is not None and ctx.is_streaming

        logger.debug(
            'Microsoft Foundry generate request',
            model=self.model_name,
            streaming=streaming,
        )

        if streaming and ctx is not None:
            return await self._generate_streaming(params, ctx, request)

        # Non-streaming request
        response: ChatCompletion = await self.client.chat.completions.create(**params)
        logger.debug(
            'Microsoft Foundry raw API response',
            model=self.model_name,
            choices=len(response.choices),
            finish_reason=str(response.choices[0].finish_reason) if response.choices else None,
        )
        choice = response.choices[0]
        message = choice.message

        # Convert response to Genkit format
        content = self._from_openai_message(message)
        response_message = Message(role=Role.MODEL, content=content)

        # Build usage statistics
        usage = build_usage(
            response.usage.prompt_tokens if response.usage else 0,
            response.usage.completion_tokens if response.usage else 0,
            response.usage.total_tokens if response.usage else 0,
        )

        finish_reason = map_finish_reason(choice.finish_reason or '')

        return GenerateResponse(
            message=response_message,
            usage=usage,
            finish_reason=finish_reason,
            request=request,
        )

    async def _generate_streaming(
        self,
        params: dict[str, Any],
        ctx: ActionRunContext,
        request: GenerateRequest,
    ) -> GenerateResponse:
        """Handle streaming generation.

        Args:
            params: Request parameters for the API.
            ctx: Action run context for sending chunks.
            request: Original generation request.

        Returns:
            Final GenerateResponse after streaming completes.
        """
        params['stream'] = True
        params['stream_options'] = {'include_usage': True}

        stream = await self.client.chat.completions.create(**params)

        accumulated_content: list[Part] = []
        accumulated_tool_calls: dict[int, dict[str, Any]] = {}
        final_usage: GenerationUsage | None = None

        async for chunk in stream:
            chunk: ChatCompletionChunk
            if not chunk.choices:
                # May contain usage info at the end
                if chunk.usage:
                    final_usage = build_usage(
                        chunk.usage.prompt_tokens,
                        chunk.usage.completion_tokens,
                        chunk.usage.total_tokens,
                    )
                continue

            delta = chunk.choices[0].delta
            parts: list[Part] = []

            # Handle text content
            if delta.content:
                text_part = Part(root=TextPart(text=delta.content))
                parts.append(text_part)
                accumulated_content.append(text_part)

            # Handle tool calls (accumulated across chunks)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {
                            'id': tc.id or '',
                            'name': tc.function.name if tc.function else '',
                            'arguments': '',
                        }
                    if tc.function and tc.function.arguments:
                        accumulated_tool_calls[idx]['arguments'] += tc.function.arguments

            # Send chunk to callback
            if parts:
                ctx.send_chunk(
                    GenerateResponseChunk(
                        role=Role.MODEL,
                        content=parts,
                        index=chunk.choices[0].index,
                    )
                )

        # Add accumulated tool calls to content and emit as chunks.
        for tc_data in accumulated_tool_calls.values():
            args = parse_tool_call_args(tc_data['arguments'])

            tool_part = Part(
                root=ToolRequestPart(
                    tool_request=ToolRequest(
                        ref=tc_data['id'],
                        name=tc_data['name'],
                        input=args,
                    )
                )
            )
            accumulated_content.append(tool_part)
            ctx.send_chunk(
                GenerateResponseChunk(
                    role=Role.MODEL,
                    content=[tool_part],
                    index=0,
                )
            )

        return GenerateResponse(
            message=Message(role=Role.MODEL, content=accumulated_content),
            usage=final_usage,
            request=request,
        )

    # _normalize_config is now delegated to the converters module.
    # See: normalize_config() imported at the top of this file.

    def _build_request_body(
        self,
        request: GenerateRequest,
        config: MicrosoftFoundryConfig,
    ) -> dict[str, Any]:
        """Build the Azure OpenAI API request body.

        This follows the same logic as `toOpenAiRequestBody` in the JS plugin.

        Args:
            request: The generation request.
            config: Normalized configuration.

        Returns:
            Dictionary suitable for chat.completions.create().
        """
        visual_detail = config.visual_detail_level or VisualDetailLevel.AUTO

        body: dict[str, Any] = {
            'model': config.model or self.deployment,
            'messages': to_openai_messages(request.messages, visual_detail),
        }

        # Add optional parameters
        # Use max_completion_tokens for o-series models, fall back to max_tokens
        if config.max_completion_tokens is not None:
            body['max_completion_tokens'] = config.max_completion_tokens
        elif config.max_tokens is not None:
            body['max_tokens'] = config.max_tokens
        if config.temperature is not None:
            body['temperature'] = config.temperature
        if config.top_p is not None:
            body['top_p'] = config.top_p
        if config.stop is not None:
            body['stop'] = config.stop
        if config.frequency_penalty is not None:
            body['frequency_penalty'] = config.frequency_penalty
        if config.presence_penalty is not None:
            body['presence_penalty'] = config.presence_penalty
        if config.logit_bias is not None:
            body['logit_bias'] = config.logit_bias
        if config.logprobs is not None:
            body['logprobs'] = config.logprobs
        if config.top_logprobs is not None:
            body['top_logprobs'] = config.top_logprobs
        if config.seed is not None:
            body['seed'] = config.seed
        if config.user is not None:
            body['user'] = config.user

        # Number of completions: prefer config.n, fall back to request.candidates
        if config.n is not None:
            body['n'] = config.n
        elif request.candidates:
            body['n'] = request.candidates

        # Output modalities (text, audio)
        if config.modalities is not None:
            body['modalities'] = config.modalities

        # Reasoning model parameters (o1, o3, o4 series)
        if config.reasoning_effort is not None:
            body['reasoning_effort'] = config.reasoning_effort
        if config.verbosity is not None:
            body['verbosity'] = config.verbosity

        # Handle tools
        if request.tools:
            body['tools'] = [to_openai_tool(t) for t in request.tools]
            if request.tool_choice:
                body['tool_choice'] = request.tool_choice
            # Allow explicit control over parallel tool calls (defaults to True in API)
            if config.parallel_tool_calls is not None:
                body['parallel_tool_calls'] = config.parallel_tool_calls

        # Handle response format (JSON mode / Structured Outputs)
        # Config response_format takes priority over request.output.format
        if config.response_format is not None:
            body['response_format'] = config.response_format
        elif request.output and request.output.format:
            model_name = config.model or self.model_name
            if model_name in MODELS_SUPPORTING_RESPONSE_FORMAT:
                model_info = get_model_info(model_name)
                output_modes = (model_info.supports.output or []) if model_info.supports else []

                if request.output.format == 'json' and 'json' in output_modes:
                    if request.output.schema:
                        # Use Structured Outputs (json_schema) when a schema
                        # is provided — the API constrains the model to emit
                        # JSON that conforms exactly to the schema.
                        body['response_format'] = {
                            'type': 'json_schema',
                            'json_schema': {
                                'name': request.output.schema.get('title', 'Response'),
                                'schema': _ensure_strict_json_schema(
                                    request.output.schema,
                                    path=(),
                                    root=request.output.schema,
                                ),
                                'strict': True,
                            },
                        }
                    else:
                        body['response_format'] = {'type': 'json_object'}
                elif request.output.format == 'text' and 'text' in output_modes:
                    body['response_format'] = {'type': 'text'}

        return body

    # _to_openai_tool, _to_openai_messages, _to_openai_role, and _extract_text
    # are now delegated to the converters module.
    # See: to_openai_tool(), to_openai_messages(), to_openai_role(), and
    # extract_text() imported at the top of this file.

    def _from_openai_message(self, message: ChatCompletionMessage) -> list[Part]:
        """Convert OpenAI response message to Genkit parts.

        Args:
            message: OpenAI ChatCompletionMessage.

        Returns:
            List of Genkit Part objects.
        """
        parts: list[Part] = []

        if message.content:
            parts.append(Part(root=TextPart(text=message.content)))

        if message.tool_calls:
            parts.extend(from_openai_tool_calls(message.tool_calls))

        return parts
