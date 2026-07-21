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
import math
import time
from email.utils import parsedate_to_datetime
from typing import Any, Protocol, cast

import structlog
from anthropic import APIError, AsyncAnthropic
from anthropic.types import Message as AnthropicMessage

from genkit import (
    Constrained,
    CustomPart,
    ErrorResponseMetadata,
    FinishReason,
    GenkitError,
    MediaPart,
    Message,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
    ModelUsage,
    Part,
    ReasoningPart,
    Role,
    TextPart,
    ToolRequest,
    ToolRequestPart,
    ToolResponsePart,
)
from genkit.model import get_basic_usage_stats
from genkit.plugin_api import ActionRunContext, StatusName
from genkit_anthropic.config import BETA_KWARG_KEYS, STABLE_KWARG_KEYS, AnthropicConfig
from genkit_anthropic.model_info import get_model_info
from genkit_anthropic.utils import (
    build_cache_usage,
    get_cache_control,
    get_redacted_thinking_data,
    get_thinking_signature,
    maybe_strip_fences,
    to_anthropic_media,
)

logger = structlog.get_logger(__name__)

DEFAULT_MAX_OUTPUT_TOKENS = 4096
_THINKING_MODE_KEYS = frozenset({'adaptive', 'budget_tokens', 'enabled', 'type'})


class _ModelDumpable(Protocol):
    """Minimal protocol for Pydantic-like config objects."""

    def model_dump(self, *, exclude_none: bool = False, by_alias: bool = False) -> dict[str, object]:
        """Dump model fields."""
        ...


_ANTHROPIC_STATUS_MAP: dict[int, StatusName] = {
    400: 'INVALID_ARGUMENT',
    401: 'UNAUTHENTICATED',
    403: 'PERMISSION_DENIED',
    429: 'RESOURCE_EXHAUSTED',
    500: 'INTERNAL',
    503: 'UNAVAILABLE',
    529: 'UNAVAILABLE',
}


def _parse_retry_after_ms(value: str) -> float | None:
    """Parse an HTTP Retry-After value into milliseconds.

    Supports both delay-seconds and HTTP-date values, matching the
    JavaScript Anthropic adapter.
    """
    value = value.strip()
    if not value:
        return None

    try:
        seconds = float(value)
    except ValueError:
        pass
    else:
        # Check the scaled value: a large finite input can overflow to inf.
        retry_after_ms = seconds * 1000
        if seconds >= 0 and math.isfinite(retry_after_ms):
            return retry_after_ms

    try:
        retry_at_ms = parsedate_to_datetime(value).timestamp() * 1000
    except (OSError, OverflowError, TypeError, ValueError):
        return None
    return max(0.0, retry_at_ms - time.time() * 1000)


def _from_anthropic_error(error: APIError) -> GenkitError:
    """Convert an Anthropic SDK error to its Genkit equivalent."""
    status_code = getattr(error, 'status_code', None)
    status = _ANTHROPIC_STATUS_MAP.get(status_code, 'UNKNOWN') if isinstance(status_code, int) else 'UNKNOWN'

    response = getattr(error, 'response', None)
    retry_after_header = response.headers.get('retry-after') if response is not None else None
    retry_after_ms = _parse_retry_after_ms(retry_after_header) if retry_after_header else None
    response_metadata: ErrorResponseMetadata | None = None
    if retry_after_ms is not None:
        response_metadata = {'retry_after_ms': retry_after_ms}

    return GenkitError(
        status=status,
        message=error.message,
        response_metadata=response_metadata,
    )


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


def _to_tool_input_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Ensure a tool input schema is valid for the Anthropic API.

    Anthropic requires ``input_schema.type`` to be present and rejects a
    missing or empty schema with a 400, so no-input tools get a default
    object schema.
    """
    if not schema:
        return {'type': 'object', 'properties': {}}
    if 'type' not in schema:
        return {**schema, 'type': 'object'}
    return schema


def _normalize_config(config: object | None) -> AnthropicConfig:
    """Normalize supported config inputs to ``AnthropicConfig``."""
    if config is None:
        return AnthropicConfig()
    if isinstance(config, AnthropicConfig):
        return config
    if isinstance(config, dict):
        return AnthropicConfig.model_validate(config)
    if hasattr(config, 'model_dump'):
        return AnthropicConfig.model_validate(
            cast(_ModelDumpable, config).model_dump(exclude_none=True, by_alias=False)
        )
    return AnthropicConfig.model_validate({k: v for k, v in vars(config).items() if v is not None})


def _to_anthropic_thinking_config(thinking: dict[str, Any] | None) -> dict[str, Any] | None:
    """Translate the public thinking config to the Anthropic SDK shape."""
    if not thinking:
        return None

    thinking_type = thinking.get('type')
    budget_tokens = thinking.get('budget_tokens')
    adaptive = thinking.get('adaptive') is True or thinking_type == 'adaptive'
    enabled = thinking.get('enabled') is True or thinking_type == 'enabled'
    disabled = thinking.get('enabled') is False or thinking_type == 'disabled'

    # Keys that are not mode toggles (display, and any forward-compatible field) pass through unchanged.
    result: dict[str, Any] = {key: value for key, value in thinking.items() if key not in _THINKING_MODE_KEYS}

    if adaptive:
        result['type'] = 'adaptive'
        return result

    if enabled or (budget_tokens is not None and not disabled):
        if budget_tokens is None:
            raise ValueError('budgetTokens is required when thinking is enabled')
        if not float(budget_tokens).is_integer():
            raise ValueError('budgetTokens must be an integer when thinking is enabled')
        result['type'] = 'enabled'
        result['budget_tokens'] = int(budget_tokens)
        return result

    if disabled:
        result['type'] = 'disabled'
        return result

    if thinking_type is not None:
        result['type'] = thinking_type
    if 'type' not in result:
        return None
    return result


def _move_unknown_params_to_extra_body(params: dict[str, Any], use_beta: bool) -> None:
    """Route passthrough body params through the SDK's ``extra_body`` escape hatch."""
    allowed = BETA_KWARG_KEYS if use_beta else STABLE_KWARG_KEYS
    unknown_keys = [key for key in params if key not in allowed]
    if not unknown_keys:
        return

    extra_body = params.get('extra_body')
    if extra_body is None:
        body: dict[str, Any] = {}
    elif isinstance(extra_body, dict):
        body = dict(extra_body)
    else:
        body = {'extra_body': extra_body}

    for key in unknown_keys:
        body[key] = params.pop(key)
    params['extra_body'] = body


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

    async def generate(self, request: ModelRequest, ctx: ActionRunContext | None = None) -> ModelResponse:
        """Generate response from Anthropic.

        Args:
            request: Generation request.
            ctx: Action run context for streaming.

        Returns:
            Generated response.
        """
        config = _normalize_config(request.config)
        use_beta = self._uses_beta_api(config)
        client = self._client_for_config(config)
        params = self._build_params(request, config=config, use_beta=use_beta)
        streaming = ctx and ctx.is_streaming

        logger.debug('Anthropic generate request', model=self.model_name, streaming=bool(streaming))

        try:
            if streaming:
                assert ctx is not None  # streaming requires ctx
                response = await self._generate_streaming(params, ctx, client=client, use_beta=use_beta)
            else:
                active_client = cast(Any, client)
                messages_client = active_client.beta.messages if use_beta else active_client.messages
                response = await messages_client.create(**params)
        except APIError as error:
            raise _from_anthropic_error(error) from error

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
            'compaction': FinishReason.OTHER,
            'end_turn': FinishReason.STOP,
            'max_tokens': FinishReason.LENGTH,
            'model_context_window_exceeded': FinishReason.LENGTH,
            'pause_turn': FinishReason.OTHER,
            'refusal': FinishReason.BLOCKED,
            'stop_sequence': FinishReason.STOP,
            'tool_use': FinishReason.STOP,
        }
        stop_reason_str = str(response.stop_reason) if response.stop_reason else ''
        finish_reason = finish_reason_map.get(stop_reason_str, FinishReason.UNKNOWN)

        # Build usage with cache-aware token counts.
        usage = self._build_usage(response, basic_usage)

        return ModelResponse(
            message=response_message,
            usage=usage,
            finish_reason=finish_reason,
        )

    def _build_usage(self, response: AnthropicMessage, basic_usage: ModelUsage) -> ModelUsage:
        """Build usage stats including cache read/write token counts.

        Delegates to :func:`utils.build_cache_usage` for the actual
        construction.

        Args:
            response: The Anthropic API response.
            basic_usage: Basic character/image usage from message content.

        Returns:
            ModelUsage with token and character counts.
        """
        return build_cache_usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            basic_usage=basic_usage,
            cache_creation_input_tokens=getattr(response.usage, 'cache_creation_input_tokens', None) or 0,
            cache_read_input_tokens=getattr(response.usage, 'cache_read_input_tokens', None) or 0,
        )

    def _client_for_config(self, config: AnthropicConfig) -> object:
        """Return the request client, applying a per-request API key when supported."""
        if not config.api_key:
            return self.client

        if not isinstance(self.client, AsyncAnthropic):
            logger.warning('Ignored per-request Anthropic apiKey because the configured client does not support it')
            return self.client

        # copy() cannot unset these, so the override would leave the base credential authenticating the request.
        if self.client.auth_token is not None:
            logger.warning('Ignored per-request Anthropic apiKey because the client authenticates with an auth token')
            return self.client

        if any(name.lower() == 'x-api-key' for name in self.client._custom_headers):  # noqa: SLF001
            logger.warning('Ignored per-request Anthropic apiKey because the client pins an x-api-key header')
            return self.client

        # copy() keeps every other client setting and shares the pooled HTTP transport.
        return self.client.copy(api_key=config.api_key)

    def _uses_beta_api(self, config: AnthropicConfig) -> bool:
        """Whether this request should use the Anthropic beta API surface.

        Any beta-only field selects the beta surface, so the requested features
        are honored instead of being rejected by the stable surface.
        """
        return config.api_version == 'beta' or bool(config.beta_only_fields())

    def _build_params(
        self,
        request: ModelRequest,
        config: AnthropicConfig | None = None,
        use_beta: bool | None = None,
    ) -> dict[str, Any]:
        """Build Anthropic API parameters."""
        config = config or _normalize_config(request.config)
        if use_beta is None:
            use_beta = self._uses_beta_api(config)
        params = config.model_dump(exclude_none=True, by_alias=False)

        # Handle mapped parameters
        max_tokens = params.pop('max_output_tokens', None)
        if max_tokens is None:
            max_tokens = params.pop('max_tokens', DEFAULT_MAX_OUTPUT_TOKENS)

        thinking = params.pop('thinking', None)
        metadata = params.pop('metadata', None)
        version = params.pop('version', None)
        betas = params.pop('betas', None)

        params['model'] = version or self.model_name
        params['messages'] = self._to_anthropic_messages(request.messages)
        params['max_tokens'] = int(max_tokens)

        # api_version and api_key select the API surface and client; they are not create() kwargs.
        for key in AnthropicConfig.SDK_UNSUPPORTED_KEYS:
            params.pop(key, None)

        # Genkit selects the streaming surface from the request context.
        params.pop('stream', None)

        if use_beta and betas:
            params['betas'] = betas

        if isinstance(thinking, dict):
            anthropic_thinking = _to_anthropic_thinking_config(thinking)
            if anthropic_thinking is not None:
                params['thinking'] = anthropic_thinking

        if metadata is not None:
            params['metadata'] = metadata

        system = self._extract_system(request.messages)

        # Handle JSON output constraint
        if request.output_format == 'json':
            use_native = (
                request.output_schema is not None
                and bool(request.output_constrained)
                and self._supports_constrained(bool(request.tools))
            )
            if use_native:
                assert request.output_schema is not None
                # Use native structured outputs via output_config.
                output_config = params.get('output_config') or {}
                params['output_config'] = {
                    **output_config,
                    'format': {
                        'type': 'json_schema',
                        'schema': _to_anthropic_schema(request.output_schema),
                    },
                }
            else:
                # Fall back to system prompt instruction.
                instruction = '\n\nOutput valid JSON. Do not wrap the JSON in markdown code fences.'
                if request.output_schema is not None:
                    schema_str = json.dumps(request.output_schema, indent=2)
                    instruction += f'\n\nFollow this JSON schema:\n{schema_str}'
                system = (system or '') + instruction

        if system:
            params['system'] = system

        if request.tools:
            params['tools'] = [
                {
                    'name': t.name,
                    'description': t.description,
                    'input_schema': _to_tool_input_schema(t.input_schema),
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

        # The API rejects tool_choice when the request carries no tools.
        if not params.get('tools'):
            params.pop('tool_choice', None)

        _move_unknown_params_to_extra_body(params, use_beta)
        return params

    def _supports_constrained(self, has_tools: bool) -> bool:
        """Return whether this model supports native constrained output."""
        supports = self._model_info.supports
        constrained = supports.constrained if supports else None
        if constrained is None or constrained == Constrained.NONE:
            return False
        return constrained != Constrained.NO_TOOLS or not has_tools

    async def _generate_streaming(
        self,
        params: dict[str, Any],
        ctx: ActionRunContext,
        client: object | None = None,
        use_beta: bool = False,
    ) -> AnthropicMessage:
        """Handle streaming generation.

        Processes Anthropic streaming events including text deltas,
        thinking deltas, redacted thinking blocks, and tool-use blocks.
        Tool-use blocks arrive as:

        1. ``content_block_start`` with ``content_block.type == 'tool_use'``
        2. Zero or more ``content_block_delta`` with ``delta.type == 'input_json_delta'``
        3. ``content_block_stop``

        We track in-progress tool calls and emit a
        :class:`ModelResponseChunk` containing the tool request when
        the block finishes.
        """
        # Track in-progress tool-use blocks by index.
        pending_tools: dict[int, dict[str, Any]] = {}

        active_client = cast(Any, client or self.client)
        messages_client = active_client.beta.messages if use_beta else active_client.messages

        async with messages_client.stream(**params) as stream:
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
                    elif getattr(block, 'type', None) == 'redacted_thinking' and hasattr(block, 'data'):
                        # Redacted thinking arrives complete in the start event; no deltas follow.
                        ctx.send_chunk(
                            ModelResponseChunk(
                                role=Role.MODEL,
                                index=0,
                                content=[Part(root=CustomPart(custom={'redactedThinking': block.data}))],  # pyright: ignore[reportAttributeAccessIssue]
                            )
                        )

                elif chunk.type == 'content_block_delta' and hasattr(chunk, 'delta'):
                    delta = chunk.delta
                    if getattr(delta, 'type', None) == 'text_delta' and hasattr(delta, 'text'):
                        ctx.send_chunk(
                            ModelResponseChunk(
                                role=Role.MODEL,
                                index=0,
                                content=[Part(root=TextPart(text=str(delta.text)))],  # pyright: ignore[reportAttributeAccessIssue]
                            )
                        )
                    elif getattr(delta, 'type', None) == 'thinking_delta' and hasattr(delta, 'thinking'):
                        ctx.send_chunk(
                            ModelResponseChunk(
                                role=Role.MODEL,
                                index=0,
                                content=[Part(root=ReasoningPart(reasoning=str(delta.thinking)))],  # pyright: ignore[reportAttributeAccessIssue]
                            )
                        )
                    # signature_delta is intentionally not streamed. The signature
                    # is recovered from the final message via _to_genkit_content.
                    elif getattr(delta, 'type', None) == 'input_json_delta' and hasattr(delta, 'partial_json'):
                        idx = getattr(chunk, 'index', None)
                        if idx is not None and idx in pending_tools:
                            pending_tools[idx]['input_json'] += delta.partial_json  # pyright: ignore[reportAttributeAccessIssue]

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
                            ModelResponseChunk(
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

            return cast(AnthropicMessage, await stream.get_final_message())

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
                    # Apply cache_control from part metadata if present; the API
                    # rejects it on thinking blocks.
                    cache_meta = get_cache_control(actual_part)
                    if cache_meta and block['type'] not in ('thinking', 'redacted_thinking'):
                        block['cache_control'] = cache_meta
                    content.append(block)
            result.append({'role': role, 'content': content})
        return result

    def _to_anthropic_block(self, part: Any) -> dict[str, Any] | None:  # noqa: ANN401
        """Convert a single Genkit content part to an Anthropic content block.

        Handles reasoning parts, redacted thinking custom parts, TextPart,
        MediaPart (images + PDFs), ToolRequestPart, and ToolResponsePart.

        Args:
            part: The actual (unwrapped) content part.

        Returns:
            An Anthropic content block dict, or None if unrecognized.
        """
        # Attribute check (not isinstance): JSON-parsed reasoning parts deserialize as DataPart.
        reasoning = getattr(part, 'reasoning', None)
        if reasoning:
            signature = get_thinking_signature(part)
            if not signature:
                raise ValueError(
                    'Anthropic thinking parts require a signature when sending back '
                    'to the API. Preserve the `metadata.thoughtSignature` value from '
                    'the original response.'
                )
            return {'type': 'thinking', 'thinking': reasoning, 'signature': signature}

        redacted_thinking = get_redacted_thinking_data(part)
        if redacted_thinking is not None:
            return {'type': 'redacted_thinking', 'data': redacted_thinking}

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
            elif block.type == 'thinking':
                signature = getattr(block, 'signature', None)
                parts.append(
                    Part(
                        root=ReasoningPart(
                            reasoning=block.thinking,
                            metadata={'thoughtSignature': signature} if signature else None,
                        )
                    )
                )
            elif block.type == 'redacted_thinking':
                parts.append(Part(root=CustomPart(custom={'redactedThinking': block.data})))
        return parts
