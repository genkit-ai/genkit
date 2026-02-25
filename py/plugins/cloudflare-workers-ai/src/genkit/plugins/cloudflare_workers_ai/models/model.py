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

"""CF AI model implementation for Genkit - Cloudflare Workers AI.

This module implements the model interface for Cloudflare Workers AI,
supporting text generation with streaming and tool calling.

See:
    - REST API: https://developers.cloudflare.com/workers-ai/get-started/rest-api/
    - Model params: https://developers.cloudflare.com/workers-ai/models/llama-3.1-8b-instruct/
    - Multimodal: https://developers.cloudflare.com/workers-ai/models/llama-4-scout-17b-16e-instruct/

Key Features
------------
- Chat completions using messages format
- Tool/function calling support (for supported models)
- Streaming responses via Server-Sent Events (SSE)
- Multimodal inputs (for Llama 4 Scout and similar models)

Implementation Notes & Edge Cases
---------------------------------

**Media URL Fetching (Cloudflare-Specific Requirement)**

Unlike other AI providers (Anthropic, OpenAI, Google GenAI, xAI) that accept media URLs
directly in their APIs and fetch the content server-side, Cloudflare Workers AI
**only accepts base64 data URIs** for images. The API explicitly states:

    "url string - image uri with data (e.g. data:image/jpeg;base64,/9j/...).
    HTTP URL will not be accepted"

This means we must fetch media content client-side before sending to Cloudflare::

    # Other providers (e.g., Google GenAI):
    {'type': 'image_url', 'image_url': {'url': 'https://example.com/image.jpg'}}

    # Cloudflare requires:
    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,/9j/...'}}

We use ``httpx.AsyncClient`` (via the existing client) to fetch images. This approach:

- Uses the same httpx client already initialized for API calls
- True async I/O (no thread pool needed)
- Doesn't block the event loop during network I/O
- Automatically handles content-type detection from response headers

**User-Agent Header Requirement**

Some servers (notably Wikipedia/Wikimedia) block requests without a proper ``User-Agent``
header, returning HTTP 403 Forbidden. We include a standard User-Agent header to ensure
compatibility when fetching images from such servers::

    headers = {
        'User-Agent': 'Genkit/1.0 (https://github.com/firebase/genkit; genkit@google.com)',
    }

**Base64 Data URL Handling**

Data URLs (``data:image/png;base64,...``) are passed through directly without
modification since they're already in the required format.

**Tool Calling Format**

Cloudflare expects tool call responses as assistant messages with the tool call
serialized to JSON in the ``content`` field::

    # When the model wants to call a tool, we send:
    {'role': 'assistant', 'content': '{"name": "get_weather", "arguments": {...}}'}

    # When providing tool results, we send:
    {'role': 'tool', 'name': 'get_weather', 'content': '{"result": "Sunny, 72°F"}'}

Note: Cloudflare's tool call format differs from OpenAI's which uses a dedicated
``tool_calls`` array. This format was determined through testing and aligns with
Cloudflare's function calling documentation.

**Tool Input Schema Wrapping**

Cloudflare requires tool parameters to be an object schema. If a tool has a primitive
type schema (e.g., ``{'type': 'string'}``), we automatically wrap it::

    # Original tool input schema:
    {'type': 'string'}

    # Wrapped for Cloudflare:
    {'type': 'object', 'properties': {'input': {'type': 'string'}}, 'required': ['input']}

**Server-Sent Events (SSE) Streaming**

The Cloudflare Workers AI API uses SSE for streaming responses. Each event is prefixed
with "data: " and contains a JSON payload. The stream ends with a special "[DONE]"
message::

    data: {'response': 'Hello'}
    data: {'response': ' world'}
    data: [DONE]

We parse these events using httpx async streaming and accumulate text chunks.

**JSON Output Mode**

Cloudflare supports two JSON modes via ``response_format``:

- Unstructured JSON: ``{'response_format': {'type': 'json_object'}}``
- Schema-constrained: ``{'response_format': {'type': 'json_schema', 'json_schema': {...}}}``

See: https://developers.cloudflare.com/workers-ai/json-mode/

When a JSON schema is provided, we use ``type: json_schema`` with the schema inlined.
Cloudflare does NOT support ``$ref`` / ``$defs`` in schemas, so we resolve all
references before sending. Note: JSON Mode does not support streaming.

When no schema is provided, we use ``type: json_object`` for unstructured JSON.

**Logging & Error Handling**

All API calls and media fetches are logged for debugging:

- ``logger.debug()`` for successful operations (request start, media fetch)
- ``logger.exception()`` for failures (API errors, fetch failures)

Exceptions from httpx are logged with full context before being re-raised,
ensuring errors are visible in logs even when caught by upstream code.
"""

import base64
import json
from typing import Any, cast

import httpx

from genkit.ai import ActionRunContext
from genkit.core.http_client import get_cached_client
from genkit.core.logging import get_logger
from genkit.plugins.cloudflare_workers_ai.constants import CF_API_BASE_URL
from genkit.plugins.cloudflare_workers_ai.models.model_info import get_model_info
from genkit.plugins.cloudflare_workers_ai.typing import CloudflareConfig
from genkit.types import (
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChunk,
    GenerationCommonConfig,
    GenerationUsage,
    Media,
    MediaPart,
    Message,
    Part,
    Role,
    TextPart,
    ToolDefinition,
    ToolRequest,
    ToolRequestPart,
    ToolResponsePart,
)

logger = get_logger(__name__)


class CfModel:
    """Cloudflare Workers AI model for chat completions.

    This class handles the conversion between Genkit's message format
    and the Cloudflare Workers AI API format.

    Attributes:
        model_id: The Cloudflare model ID (e.g., '@cf/meta/llama-3.1-8b-instruct').
        account_id: The Cloudflare account ID.
        client: httpx.AsyncClient for making API requests.
    """

    # TODO(#4360): Replace with downloadRequestMedia middleware (G15 parity).
    # User-Agent is required because many servers (e.g. Wikipedia) return
    # 403 Forbidden for the default httpx user-agent string.
    _DOWNLOAD_HEADERS: dict[str, str] = {
        'User-Agent': 'Genkit/1.0 (https://github.com/firebase/genkit; genkit@google.com)',
    }

    def __init__(
        self,
        model_id: str,
        account_id: str,
        client: httpx.AsyncClient,
    ) -> None:
        """Initialize the Cloudflare model.

        Args:
            model_id: Cloudflare model ID (e.g., '@cf/meta/llama-3.1-8b-instruct').
            account_id: Cloudflare account ID.
            client: Configured httpx.AsyncClient with auth headers.
        """
        self.model_id = model_id
        self.account_id = account_id
        self.client = client
        self._model_info = get_model_info(model_id)

    def _get_api_url(self) -> str:
        """Get the API URL for this model.

        Returns:
            Full URL for the model's inference endpoint.
        """
        return f'{CF_API_BASE_URL}/{self.account_id}/ai/run/{self.model_id}'

    async def generate(
        self,
        request: GenerateRequest,
        ctx: ActionRunContext | None = None,
    ) -> GenerateResponse:
        """Generate a response from Cloudflare Workers AI.

        Args:
            request: The generation request containing messages and config.
            ctx: Action run context for streaming support.

        Returns:
            GenerateResponse with the model's output.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        config = self._normalize_config(request.config)
        body = await self._build_request_body(request, config)
        streaming = ctx is not None and ctx.is_streaming

        logger.debug(
            'Cloudflare generate request',
            model_id=self.model_id,
            streaming=streaming,
        )

        if streaming and ctx is not None:
            body['stream'] = True
            return await self._generate_streaming(body, ctx, request)

        try:
            response = await self.client.post(self._get_api_url(), json=body)
            response.raise_for_status()
            data = response.json()
            logger.debug(
                'Cloudflare raw API response',
                model_id=self.model_id,
                data=data,
            )
        except httpx.HTTPStatusError as e:
            logger.exception(
                'Cloudflare API call failed',
                model_id=self.model_id,
                status_code=e.response.status_code,
                error=str(e),
            )
            raise

        return self._parse_response(data, request)

    async def _generate_streaming(
        self,
        body: dict[str, Any],
        ctx: ActionRunContext,
        request: GenerateRequest,
    ) -> GenerateResponse:
        """Handle streaming generation using SSE.

        Args:
            body: Request body with stream=True.
            ctx: Action run context for sending chunks.
            request: Original generation request.

        Returns:
            Final GenerateResponse after streaming completes.
        """
        accumulated_text = ''
        accumulated_tool_calls: list[dict[str, Any]] = []
        final_usage: GenerationUsage | None = None

        try:
            async with self.client.stream('POST', self._get_api_url(), json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith('data: '):
                        continue

                    data_str = line[6:]  # Remove 'data: ' prefix

                    # Check for end of stream
                    if data_str.strip() == '[DONE]':
                        break

                    try:
                        chunk_data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Extract text response
                    if 'response' in chunk_data:
                        text = chunk_data['response']
                        if text and isinstance(text, str):  # Guard against None and non-string types
                            accumulated_text += text

                            text_part = Part(root=TextPart(text=text))
                            ctx.send_chunk(
                                GenerateResponseChunk(
                                    role=Role.MODEL,
                                    content=[text_part],
                                    index=0,
                                )
                            )

                    # Extract usage if present
                    if 'usage' in chunk_data:
                        usage_data = chunk_data['usage']
                        final_usage = GenerationUsage(
                            input_tokens=usage_data.get('prompt_tokens', 0),
                            output_tokens=usage_data.get('completion_tokens', 0),
                            total_tokens=usage_data.get('total_tokens', 0),
                        )

                    # Extract tool calls if present
                    if 'tool_calls' in chunk_data:
                        accumulated_tool_calls.extend(chunk_data['tool_calls'])

        except httpx.HTTPStatusError as e:
            logger.exception(
                'Cloudflare streaming API call failed',
                model_id=self.model_id,
                status_code=e.response.status_code,
                error=str(e),
            )
            raise

        # Build final response content
        content: list[Part] = []
        if accumulated_text:
            content.append(Part(root=TextPart(text=accumulated_text)))

        # Add tool calls to content (OpenAI-compatible nested format).
        for tool_call in accumulated_tool_calls:
            func = tool_call.get('function', {})
            tc_name = func.get('name', '') or tool_call.get('name', '')
            tc_args_raw = func.get('arguments', '') or tool_call.get('arguments', {})
            if isinstance(tc_args_raw, str):
                try:
                    tc_args = json.loads(tc_args_raw)
                except (json.JSONDecodeError, ValueError):
                    tc_args = {}
            else:
                tc_args = tc_args_raw
            content.append(
                Part(
                    root=ToolRequestPart(
                        tool_request=ToolRequest(
                            name=tc_name,
                            input=tc_args,
                        )
                    )
                )
            )

        finish_reason = FinishReason.STOP
        if accumulated_tool_calls:
            finish_reason = FinishReason.STOP  # Tool calls also use STOP

        return GenerateResponse(
            message=Message(role=Role.MODEL, content=content),
            usage=final_usage,
            finish_reason=finish_reason,
            request=request,
        )

    def _parse_response(
        self,
        data: dict[str, Any],
        request: GenerateRequest,
    ) -> GenerateResponse:
        """Parse Cloudflare API response into GenerateResponse.

        Args:
            data: Raw API response data.
            request: Original generation request.

        Returns:
            Parsed GenerateResponse.
        """
        result = data.get('result', data)  # API may wrap in 'result'

        content: list[Part] = []

        # Extract text response (only if non-empty).
        # In JSON Mode, Cloudflare returns a dict instead of a string.
        # Serialize to JSON so TextPart gets a string and the framework
        # can parse structured output downstream.
        text_response = result.get('response', '')
        if text_response:
            if isinstance(text_response, dict):
                text_response = json.dumps(text_response)
            content.append(Part(root=TextPart(text=text_response)))

        # Extract tool calls.
        # Cloudflare returns OpenAI-compatible format with a nested
        # ``function`` object: {"type": "function", "function": {"name": ..., "arguments": ...}}
        # We also handle a flat format for backwards compatibility.
        tool_calls = result.get('tool_calls', [])
        for tool_call in tool_calls:
            func = tool_call.get('function', {})
            tc_name = func.get('name', '') or tool_call.get('name', '')
            tc_args_raw = func.get('arguments', '') or tool_call.get('arguments', {})
            # arguments may be a JSON string or already a dict.
            if isinstance(tc_args_raw, str):
                try:
                    tc_args = json.loads(tc_args_raw)
                except (json.JSONDecodeError, ValueError):
                    tc_args = {}
            else:
                tc_args = tc_args_raw
            content.append(
                Part(
                    root=ToolRequestPart(
                        tool_request=ToolRequest(
                            name=tc_name,
                            input=tc_args,
                        )
                    )
                )
            )

        # Build usage statistics
        usage_data = result.get('usage', {})
        usage = GenerationUsage(
            input_tokens=usage_data.get('prompt_tokens', 0),
            output_tokens=usage_data.get('completion_tokens', 0),
            total_tokens=usage_data.get('total_tokens', 0),
        )

        finish_reason = FinishReason.STOP

        return GenerateResponse(
            message=Message(role=Role.MODEL, content=content),
            usage=usage,
            finish_reason=finish_reason,
            request=request,
        )

    def _normalize_config(self, config: object) -> CloudflareConfig:
        """Normalize config to CloudflareConfig.

        Args:
            config: Request configuration (dict, CloudflareConfig, or GenerationCommonConfig).

        Returns:
            Normalized CloudflareConfig instance.
        """
        if config is None:
            return CloudflareConfig()

        if isinstance(config, CloudflareConfig):
            return config

        if isinstance(config, GenerationCommonConfig):
            return CloudflareConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                top_p=config.top_p,
                stop_sequences=config.stop_sequences,
            )

        if isinstance(config, dict):
            # Handle camelCase to snake_case mapping
            mapped: dict[str, Any] = {}
            key_map: dict[str, str] = {
                'maxOutputTokens': 'max_output_tokens',
                'maxTokens': 'max_output_tokens',
                'topP': 'top_p',
                'topK': 'top_k',
                'stopSequences': 'stop_sequences',
                'repetitionPenalty': 'repetition_penalty',
                'frequencyPenalty': 'frequency_penalty',
                'presencePenalty': 'presence_penalty',
            }
            for key, value in config.items():
                str_key = str(key)
                mapped_key = key_map.get(str_key, str_key)
                mapped[mapped_key] = value
            return CloudflareConfig(**mapped)

        return CloudflareConfig()

    async def _build_request_body(
        self,
        request: GenerateRequest,
        config: CloudflareConfig,
    ) -> dict[str, Any]:
        """Build the Cloudflare API request body.

        Args:
            request: The generation request.
            config: Normalized configuration.

        Returns:
            Dictionary suitable for the API request.
        """
        body: dict[str, Any] = {
            'messages': await self._to_cloudflare_messages(request.messages),
        }

        # Add configuration parameters
        if config.max_output_tokens is not None:
            body['max_tokens'] = config.max_output_tokens

        if config.temperature is not None:
            body['temperature'] = config.temperature

        if config.top_p is not None:
            body['top_p'] = config.top_p

        if config.top_k is not None:
            body['top_k'] = config.top_k

        if config.seed is not None:
            body['seed'] = config.seed

        if config.repetition_penalty is not None:
            body['repetition_penalty'] = config.repetition_penalty

        if config.frequency_penalty is not None:
            body['frequency_penalty'] = config.frequency_penalty

        if config.presence_penalty is not None:
            body['presence_penalty'] = config.presence_penalty

        if config.lora is not None:
            body['lora'] = config.lora

        if config.raw is not None:
            body['raw'] = config.raw

        # Handle JSON output format
        # See: https://developers.cloudflare.com/workers-ai/json-mode/
        if request.output and request.output.format == 'json':
            if request.output.schema:
                # Schema-constrained JSON mode: resolve $ref/$defs since
                # Cloudflare doesn't support JSON Schema references.
                resolved = _resolve_json_schema_refs(request.output.schema)
                body['response_format'] = {
                    'type': 'json_schema',
                    'json_schema': resolved,
                }
            else:
                # Unstructured JSON mode (no schema)
                body['response_format'] = {'type': 'json_object'}

        # Handle tools
        if request.tools:
            body['tools'] = [self._to_cloudflare_tool(t) for t in request.tools]

        return body

    def _to_cloudflare_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        """Convert a Genkit tool definition to Cloudflare format.

        Args:
            tool: Genkit ToolDefinition.

        Returns:
            Cloudflare-compatible tool specification.
        """
        # Cloudflare expects parameters to be an object schema.
        # If the tool has a primitive type schema (e.g., {'type': 'string'}),
        # we need to wrap it in an object schema.
        params = tool.input_schema or {'type': 'object', 'properties': {}}

        # If the schema is not an object type, wrap it
        if params.get('type') != 'object':
            params = {
                'type': 'object',
                'properties': {
                    'input': params,
                },
                'required': ['input'],
            }

        return {
            'type': 'function',
            'function': {
                'name': tool.name,
                'description': tool.description or '',
                'parameters': params,
            },
        }

    async def _to_cloudflare_messages(
        self,
        messages: list[Message],
    ) -> list[dict[str, Any]]:
        """Convert Genkit messages to Cloudflare API message format.

        Args:
            messages: List of Genkit messages.

        Returns:
            List of Cloudflare-compatible message dictionaries.
        """
        cloudflare_msgs: list[dict[str, Any]] = []

        for msg in messages:
            role = self._to_cloudflare_role(msg.role)
            content_parts: list[dict[str, Any] | str] = []
            text_content = ''

            for part in msg.content:
                root = part.root if isinstance(part, Part) else part

                if isinstance(root, TextPart):
                    text_content += root.text

                elif isinstance(root, MediaPart):
                    # Handle multimodal content
                    media_content = await self._convert_media_to_cloudflare(root.media)
                    content_parts.append(media_content)

                elif isinstance(root, ToolRequestPart):
                    # Tool requests in assistant messages
                    # Cloudflare expects the assistant content to be a JSON string
                    # of the tool call, not an object with tool_calls property.
                    # See: https://developers.cloudflare.com/workers-ai/function-calling/
                    tool_req = root.tool_request
                    tool_call_obj = {
                        'name': tool_req.name,
                        'arguments': tool_req.input if isinstance(tool_req.input, dict) else {'input': tool_req.input},
                    }
                    cloudflare_msgs.append({
                        'role': 'assistant',
                        'content': json.dumps(tool_call_obj),
                    })
                    continue

                elif isinstance(root, ToolResponsePart):
                    # Tool responses
                    tool_resp = root.tool_response
                    output = tool_resp.output
                    output_str = json.dumps(output) if isinstance(output, dict) else str(output)

                    cloudflare_msgs.append({
                        'role': 'tool',
                        'name': tool_resp.name,
                        'content': output_str,
                    })
                    continue

            # Build message
            if text_content and not content_parts:
                # Simple text message
                cloudflare_msgs.append({
                    'role': role,
                    'content': text_content,
                })
            elif content_parts:
                # Multimodal message
                if text_content:
                    content_parts.insert(0, {'type': 'text', 'text': text_content})
                cloudflare_msgs.append({
                    'role': role,
                    'content': content_parts,
                })

        return cloudflare_msgs

    def _to_cloudflare_role(self, role: Role | str) -> str:
        """Convert Genkit role to Cloudflare role string.

        Args:
            role: Genkit Role enum or string.

        Returns:
            Cloudflare-compatible role string.
        """
        if isinstance(role, str):
            role_str = role.lower()
        else:
            role_str = role.value.lower() if hasattr(role, 'value') else str(role).lower()

        role_mapping = {
            'user': 'user',
            'model': 'assistant',
            'system': 'system',
            'tool': 'tool',
        }

        return role_mapping.get(role_str, 'user')

    async def _convert_media_to_cloudflare(self, media: Media) -> dict[str, Any]:
        """Convert Genkit Media to Cloudflare image format.

        Cloudflare Workers AI does NOT accept HTTP URLs for images. It only
        accepts base64 data URIs (e.g., `data:image/jpeg;base64,/9j/...`).
        This method fetches images from URLs and converts them to base64.

        See: https://developers.cloudflare.com/workers-ai/models/llama-4-scout-17b-16e-instruct/

        Args:
            media: Genkit Media object.

        Returns:
            Cloudflare-compatible image content block with base64 data URI.

        Raises:
            ValueError: If the media URL cannot be processed.
        """
        url = media.url

        # Handle base64 data URLs - already in correct format
        if url.startswith('data:'):
            return {
                'type': 'image_url',
                'image_url': {
                    'url': url,
                },
            }

        # For HTTP/HTTPS URLs, we must fetch and convert to base64
        # because Cloudflare does NOT accept URLs directly
        try:
            client = get_cached_client(
                cache_key='cloudflare-workers-ai/media-fetch',
                headers=self._DOWNLOAD_HEADERS,
                follow_redirects=True,
            )
            response = await client.get(url)
            response.raise_for_status()
            image_bytes = response.content

            # Determine content type from response or URL
            content_type = media.content_type
            if not content_type:
                content_type = response.headers.get('content-type', 'image/jpeg')
                # Strip any charset or parameters from content type
                if ';' in content_type:
                    content_type = content_type.split(';')[0].strip()

            # Convert to base64 data URI
            base64_data = base64.b64encode(image_bytes).decode('utf-8')
            data_uri = f'data:{content_type};base64,{base64_data}'

            return {
                'type': 'image_url',
                'image_url': {
                    'url': data_uri,
                },
            }
        except httpx.HTTPStatusError as e:
            raise ValueError(f'Failed to fetch image from URL {url}: {e}') from e


# JSON Schema nodes can be dicts, lists, or scalar values.
# A proper recursive type isn't needed here — we only care about
# dict/list dispatch in _resolve, and isinstance handles the rest.
JsonSchemaNode = dict[str, object] | list[object] | str | int | float | bool | None


def _resolve_json_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``$ref`` / ``$defs`` in a JSON Schema by inlining definitions.

    Cloudflare Workers AI does not support ``$ref`` or ``$defs`` in
    ``json_schema``.  Pydantic generates these for nested models (e.g.
    ``RpgCharacter`` containing ``Skills``).  This helper recursively
    inlines all ``$ref`` pointers so the resulting schema is self-contained.

    Only ``#/$defs/<name>`` references are resolved (the format Pydantic
    uses).  Circular references are **not** handled because Pydantic
    doesn't produce them for concrete models.

    Args:
        schema: JSON Schema dict, possibly containing ``$defs`` and ``$ref``.

    Returns:
        A new dict with all ``$ref`` pointers replaced by their definitions
        and the top-level ``$defs`` key removed.
    """
    defs: dict[str, Any] = schema.get('$defs', {})
    if not defs:
        return schema

    def _resolve(node: JsonSchemaNode) -> JsonSchemaNode:
        if isinstance(node, dict):
            # Replace {"$ref": "#/$defs/Foo"} with the definition of Foo
            if '$ref' in node:
                ref_val = node['$ref']
                if isinstance(ref_val, str) and ref_val.startswith('#/$defs/'):
                    ref_name = ref_val[len('#/$defs/') :]
                    if ref_name in defs:
                        return _resolve(defs[ref_name])
                # Unknown ref — leave as-is (shouldn't happen with Pydantic)
                return node
            return {k: _resolve(cast(JsonSchemaNode, v)) for k, v in node.items() if k != '$defs'}
        if isinstance(node, list):
            return [_resolve(cast(JsonSchemaNode, item)) for item in node]
        return node

    resolved = _resolve(schema)
    # _resolve always returns a dict when given a dict input
    assert isinstance(resolved, dict)  # noqa: S101 - type narrowing assertion
    return resolved


__all__ = ['CF_API_BASE_URL', 'CfModel']
