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

"""Anthropic model implementations.

Supports Prompt Caching, PDF/Document input, and extended thinking in
addition to standard chat, vision, and tool-calling capabilities.

See:
    - Cache control: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
    - Document input: https://docs.anthropic.com/en/docs/build-with-claude/pdf-support
"""

import json
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import Message as AnthropicMessage
from genkit.ai import ActionRunContext
from genkit.blocks.model import get_basic_usage_stats
from genkit.core.logging import get_logger
from genkit.plugins.anthropic.model_info import get_model_info
from genkit.plugins.anthropic.utils import (
    build_cache_usage,
    get_cache_control,
    maybe_strip_fences,
    to_anthropic_media,
)
from genkit.types import (
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChunk,
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

logger = get_logger(__name__)

DEFAULT_MAX_OUTPUT_TOKENS = 4096


def _to_anthropic_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Transform a JSON schema for Anthropic structured output.

    Anthropic requires ``additionalProperties: false`` on all object
    types.  This recursively adds it.

    See:
        https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs#json-schema-limitations
    """
    out = dict(schema)
    out.pop('$schema', None)
    if out.get('type') == 'object':
        out['additionalProperties'] = False
    for key, value in out.items():
        if isinstance(value, dict):
            out[key] = _to_anthropic_schema(value)
    return out


class AnthropicModel:
    """Represents an Anthropic language model for use with Genkit.

    Encapsulates interaction logic for a specific Claude model,
    enabling its use within Genkit for generative tasks.

    Supports:
        - Prompt caching via ``cache_control`` metadata on content parts
        - PDF and plain-text document input via ``DocumentBlockParam``
        - Extended thinking via ``thinking`` config parameter
        - Tool use / function calling
    """

    def __init__(self, model_name: str, client: AsyncAnthropic) -> None:
        """Initialize Anthropic model.

        Sets up the client for communicating with the Anthropic API
        and stores the model name.

        Args:
            model_name: Name of the Anthropic model.
            client: AsyncAnthropic client instance.
        """
        model_info = get_model_info(model_name)
        self._model_info = model_info
        self.model_name = model_info.versions[0] if model_info.versions else model_name
        self.client = client

    async def generate(self, request: GenerateRequest, ctx: ActionRunContext | None = None) -> GenerateResponse:
        """Generate response from Anthropic.

        Args:
            request: Generation request.
            ctx: Action run context for streaming.

        Returns:
            Generated response.
        """
        params = self._build_params(request)
        streaming = ctx and ctx.is_streaming

        logger.debug('Anthropic generate request', model=self.model_name, streaming=bool(streaming))

        if streaming:
            assert ctx is not None  # streaming requires ctx
            response = await self._generate_streaming(params, ctx)
        else:
            response = await self.client.messages.create(**params)

        logger.debug(
            'Anthropic raw API response',
            model=self.model_name,
            stop_reason=str(response.stop_reason),
            content_blocks=len(response.content),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        content = self._to_genkit_content(response.content)
        content = maybe_strip_fences(request, content)

        response_message = Message(role=Role.MODEL, content=content)
        basic_usage = get_basic_usage_stats(input_=request.messages, response=response_message)

        finish_reason_map: dict[str, FinishReason] = {
            'end_turn': FinishReason.STOP,
            'max_tokens': FinishReason.LENGTH,
            'stop_sequence': FinishReason.STOP,
            'tool_use': FinishReason.STOP,
        }
        stop_reason_str = str(response.stop_reason) if response.stop_reason else ''
        finish_reason = finish_reason_map.get(stop_reason_str, FinishReason.UNKNOWN)

        # Build usage with cache-aware token counts.
        usage = self._build_usage(response, basic_usage)

        return GenerateResponse(
            message=response_message,
            usage=usage,
            finish_reason=finish_reason,
        )

    def _build_usage(self, response: AnthropicMessage, basic_usage: GenerationUsage) -> GenerationUsage:
        """Build usage stats including cache read/write token counts.

        Delegates to :func:`utils.build_cache_usage` for the actual
        construction.

        Args:
            response: The Anthropic API response.
            basic_usage: Basic character/image usage from message content.

        Returns:
            GenerationUsage with token and character counts.
        """
        return build_cache_usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            basic_usage=basic_usage,
            cache_creation_input_tokens=getattr(response.usage, 'cache_creation_input_tokens', None) or 0,
            cache_read_input_tokens=getattr(response.usage, 'cache_read_input_tokens', None) or 0,
        )

    def _build_params(self, request: GenerateRequest) -> dict[str, Any]:
        """Build Anthropic API parameters."""
        config = request.config
        params: dict[str, Any] = {}

        if isinstance(config, dict):
            params = config.copy()
        elif config:
            if hasattr(config, 'model_dump'):
                params = config.model_dump(exclude_none=True)
            else:
                params = {k: v for k, v in vars(config).items() if v is not None}

        # Handle mapped parameters
        max_tokens = params.pop('max_output_tokens', None)
        if max_tokens is None:
            max_tokens = params.get('max_tokens', DEFAULT_MAX_OUTPUT_TOKENS)

        params.get('temperature')
        params.get('top_p')
        params.get('stop_sequences')
        thinking = params.pop('thinking', None)
        metadata = params.pop('metadata', None)

        params['model'] = self.model_name
        params['messages'] = self._to_anthropic_messages(request.messages)
        params['max_tokens'] = int(max_tokens)

        # Remove known genkit keys that don't map directly or are handled
        params.pop('version', None)  # If version was passed through config

        if thinking and isinstance(thinking, dict):
            anthropic_thinking: dict[str, str | int] = {}
            # Handle boolean enabled -> type="enabled"
            if thinking.get('enabled') is True or thinking.get('type') == 'enabled':
                anthropic_thinking['type'] = 'enabled'

            # Handle camelCase -> snake_case for budget tokens
            tokens = thinking.get('budgetTokens', thinking.get('budget_tokens'))
            if tokens:
                anthropic_thinking['budget_tokens'] = int(tokens)

            if anthropic_thinking.get('type') == 'enabled':
                params['thinking'] = anthropic_thinking

        if metadata:
            params['metadata'] = metadata

        system = self._extract_system(request.messages)

        # Handle JSON output constraint
        if request.output and request.output.format == 'json':
            supports_json = 'json' in (self._model_info.supports.output or []) if self._model_info.supports else False
            if request.output.schema and supports_json:
                # Use native structured outputs via output_config.
                params['output_config'] = {
                    'format': {
                        'type': 'json_schema',
                        'schema': _to_anthropic_schema(request.output.schema),
                    }
                }
            else:
                # Fall back to system prompt instruction.
                instruction = '\n\nOutput valid JSON. Do not wrap the JSON in markdown code fences.'
                if request.output.schema:
                    schema_str = json.dumps(request.output.schema, indent=2)
                    instruction += f'\n\nFollow this JSON schema:\n{schema_str}'
                system = (system or '') + instruction

        if system:
            params['system'] = system

        if request.tools:
            params['tools'] = [
                {
                    'name': t.name,
                    'description': t.description,
                    'input_schema': t.input_schema,
                }
                for t in request.tools
            ]

            if request.tool_choice:
                if request.tool_choice == 'required':
                    params['tool_choice'] = {'type': 'any'}
                elif request.tool_choice == 'auto':
                    params['tool_choice'] = {'type': 'auto'}
                elif isinstance(request.tool_choice, dict):
                    params['tool_choice'] = request.tool_choice

        return params

    async def _generate_streaming(self, params: dict[str, Any], ctx: ActionRunContext) -> AnthropicMessage:
        """Handle streaming generation.

        Processes Anthropic streaming events including text deltas and
        tool-use blocks.  Tool-use blocks arrive as:

        1. ``content_block_start`` with ``content_block.type == 'tool_use'``
        2. Zero or more ``content_block_delta`` with ``delta.type == 'input_json_delta'``
        3. ``content_block_stop``

        We track in-progress tool calls and emit a
        :class:`GenerateResponseChunk` containing the tool request when
        the block finishes.
        """
        # Track in-progress tool-use blocks by index.
        pending_tools: dict[int, dict[str, Any]] = {}

        async with self.client.messages.stream(**params) as stream:
            async for chunk in stream:
                if chunk.type == 'content_block_start' and hasattr(chunk, 'content_block'):
                    block = chunk.content_block
                    if getattr(block, 'type', None) == 'tool_use':
                        idx = getattr(chunk, 'index', None)
                        if idx is not None:
                            pending_tools[idx] = {
                                'id': getattr(block, 'id', ''),
                                'name': getattr(block, 'name', ''),
                                'input_json': '',
                            }

                elif chunk.type == 'content_block_delta' and hasattr(chunk, 'delta'):
                    delta = chunk.delta
                    if getattr(delta, 'type', None) == 'text_delta' and hasattr(delta, 'text'):
                        ctx.send_chunk(
                            GenerateResponseChunk(
                                role=Role.MODEL,
                                index=0,
                                content=[Part(root=TextPart(text=str(delta.text)))],
                            )
                        )
                    elif getattr(delta, 'type', None) == 'input_json_delta' and hasattr(delta, 'partial_json'):
                        idx = getattr(chunk, 'index', None)
                        if idx is not None and idx in pending_tools:
                            pending_tools[idx]['input_json'] += delta.partial_json

                elif chunk.type == 'content_block_stop':
                    idx = getattr(chunk, 'index', None)
                    if idx is not None and idx in pending_tools:
                        tool_info = pending_tools.pop(idx)
                        tool_input: object = {}
                        if tool_info['input_json']:
                            try:
                                tool_input = json.loads(tool_info['input_json'])
                            except (json.JSONDecodeError, TypeError):
                                tool_input = tool_info['input_json']
                        ctx.send_chunk(
                            GenerateResponseChunk(
                                role=Role.MODEL,
                                index=0,
                                content=[
                                    Part(
                                        root=ToolRequestPart(
                                            tool_request=ToolRequest(
                                                ref=tool_info['id'],
                                                name=tool_info['name'],
                                                input=tool_input,
                                            )
                                        )
                                    )
                                ],
                            )
                        )

            return await stream.get_final_message()

    def _extract_system(self, messages: list[Message]) -> str | None:
        """Extract system prompt from messages."""
        for msg in messages:
            if msg.role == Role.SYSTEM:
                texts = []
                for part in msg.content:
                    actual_part = part.root if isinstance(part, Part) else part
                    if isinstance(actual_part, TextPart):
                        texts.append(actual_part.text)
                return ''.join(texts) if texts else None
        return None

    def _to_anthropic_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Genkit messages to Anthropic format.

        Handles text, media (images), tool use/result, and document
        (PDF/plain-text) content parts. Applies ``cache_control``
        metadata when present on a part's metadata.
        """
        result = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                continue
            role = 'assistant' if msg.role == Role.MODEL else 'user'
            content: list[dict[str, Any]] = []
            for part in msg.content:
                actual_part = part.root if isinstance(part, Part) else part
                block = self._to_anthropic_block(actual_part)
                if block is not None:
                    # Apply cache_control from part metadata if present.
                    cache_meta = get_cache_control(actual_part)
                    if cache_meta:
                        block['cache_control'] = cache_meta
                    content.append(block)
            result.append({'role': role, 'content': content})
        return result

    def _to_anthropic_block(self, part: Any) -> dict[str, Any] | None:  # noqa: ANN401
        """Convert a single Genkit content part to an Anthropic content block.

        Handles TextPart, MediaPart (images + PDFs), ToolRequestPart,
        and ToolResponsePart.

        Args:
            part: The actual (unwrapped) content part.

        Returns:
            An Anthropic content block dict, or None if unrecognized.
        """
        if isinstance(part, TextPart):
            return {'type': 'text', 'text': part.text}
        if isinstance(part, MediaPart):
            return to_anthropic_media(part)
        if isinstance(part, ToolRequestPart):
            return {
                'type': 'tool_use',
                'id': part.tool_request.ref,
                'name': part.tool_request.name,
                'input': part.tool_request.input,
            }
        if isinstance(part, ToolResponsePart):
            return {
                'type': 'tool_result',
                'tool_use_id': part.tool_response.ref,
                'content': str(part.tool_response.output),
            }
        return None

    def _to_genkit_content(self, content_blocks: list[Any]) -> list[Part]:
        """Convert Anthropic response to Genkit format."""
        parts = []
        for block in content_blocks:
            if block.type == 'text':
                parts.append(Part(root=TextPart(text=block.text)))
            elif block.type == 'tool_use':
                parts.append(
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(
                                ref=block.id,
                                name=block.name,
                                input=block.input,
                            )
                        )
                    )
                )
        return parts
