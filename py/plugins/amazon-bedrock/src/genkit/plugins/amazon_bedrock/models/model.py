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

"""AWS Bedrock model implementation for Genkit.

This module implements the model interface for AWS Bedrock using the Converse API.

See:
- Converse API: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
- Boto3 Reference: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html

Key Features
------------
- Chat completions using Converse API
- Tool/function calling support
- Streaming responses via ConverseStream
- Multimodal inputs (images, video for supported models)
- Reasoning content extraction (for DeepSeek-R1, etc.)

Implementation Notes & Edge Cases
---------------------------------

**Media URL Fetching (Bedrock-Specific Requirement)**

Unlike other AI providers (Anthropic, OpenAI, Google GenAI, xAI) that accept media URLs
directly in their APIs and fetch the content server-side, AWS Bedrock's Converse API
**only accepts inline bytes**. The API does not support URL references.

This means we must fetch media content client-side before sending to Bedrock::

    # Other providers (e.g., Anthropic):
    {'type': 'url', 'url': 'https://example.com/image.jpg'}  # API fetches it

    # AWS Bedrock requires:
    {'image': {'format': 'jpeg', 'source': {'bytes': b'...actual bytes...'}}}

We use ``httpx.AsyncClient`` for true async HTTP requests. This approach:

- Uses httpx which is already a genkit core dependency
- True async I/O (no thread pool needed)
- Doesn't block the event loop during network I/O
- Includes a 30-second timeout for fetch operations
- Supports both images and videos
- Better error handling with ``HTTPStatusError`` for HTTP errors

**User-Agent Header Requirement**

Some servers (notably Wikipedia/Wikimedia) block requests without a proper ``User-Agent``
header, returning HTTP 403 Forbidden. We include a standard User-Agent header to ensure
compatibility::

    headers = {
        'User-Agent': 'Genkit/1.0 (https://github.com/firebase/genkit; Python httpx)',
        'Accept': 'image/*,video/*,*/*',
    }

**Base64 Data URL Handling**

Data URLs (``data:image/png;base64,...``) are handled inline without network requests.
The base64 payload is decoded directly to bytes.

**JSON Output Mode (Prompt Engineering)**

The Bedrock Converse API doesn't have a native JSON mode like OpenAI's ``response_format``.
When JSON output is requested via ``request.output.format == 'json'``, we inject
instructions into the system prompt to guide the model::

    IMPORTANT: You MUST respond with valid JSON only.
    Do not include any text before or after the JSON.
    Do not wrap the JSON in markdown code blocks.
    Your response MUST conform to this JSON schema:
    {...schema...}

This prompt engineering approach works across all Bedrock models but is not as
reliable as native JSON mode. Models may occasionally include extra text.

**Automatic Inference Profile Conversion**

When using API key authentication (``AWS_BEARER_TOKEN_BEDROCK``), AWS Bedrock
requires inference profile IDs with regional prefixes instead of direct model IDs.
The plugin automatically detects API key auth and adds the appropriate regional
prefix (``us.``, ``eu.``, ``apac.``) based on ``AWS_REGION``::

    # User specifies direct model ID
    model = 'amazon-bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0'

    # With API key auth and AWS_REGION=us-east-1, automatically converts to:
    modelId = 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'

This means users don't need to manually use ``inference_profile()`` - the plugin
handles the conversion transparently. If the model ID already has a regional
prefix, no conversion is performed.

**Logging & Error Handling**

All API calls and media fetches are logged for debugging:

- ``logger.debug()`` for successful operations (request start, media fetch)
- ``logger.exception()`` for failures (API errors, fetch failures)

Exceptions from boto3 are logged with full context before being re-raised,
ensuring errors are visible in logs even when caught by upstream code.

**Streaming Implementation**

Streaming uses the ``converse_stream`` API. Tool use deltas are accumulated
across multiple events and assembled into complete tool requests at the end
of the stream.
"""

import base64
import json
import os
from typing import Any

import httpx

from genkit.ai import ActionRunContext
from genkit.core.http_client import get_cached_client
from genkit.core.logging import get_logger
from genkit.plugins.amazon_bedrock.models.converters import (
    StreamingFenceStripper,
    maybe_strip_fences,
)
from genkit.plugins.amazon_bedrock.typing import BedrockConfig
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

# Logger for this module
logger = get_logger(__name__)

# Regional prefixes for inference profiles (us, eu, apac)
_INFERENCE_PROFILE_PREFIXES = ('us.', 'eu.', 'apac.')

# Model provider prefixes that support cross-region inference profiles.
# These providers can use regional prefixes (us., eu., apac.) with API key auth.
# Other providers (ai21., stability.) require direct model IDs only.
_INFERENCE_PROFILE_SUPPORTED_PROVIDERS = (
    'anthropic.',  # Claude models
    'amazon.',  # Nova, Titan models
    'meta.',  # Llama models
    'mistral.',  # Mistral models
    'cohere.',  # Command R models
    'deepseek.',  # DeepSeek models
)

# Mapping from Bedrock stop reasons to Genkit finish reasons
FINISH_REASON_MAP: dict[str, FinishReason] = {
    'end_turn': FinishReason.STOP,
    'stop_sequence': FinishReason.STOP,
    'max_tokens': FinishReason.LENGTH,
    'tool_use': FinishReason.STOP,
    'content_filtered': FinishReason.BLOCKED,
    'guardrail_intervened': FinishReason.BLOCKED,
}


class BedrockModel:
    """AWS Bedrock model for chat completions using the Converse API.

    This class handles the conversion between Genkit's message format
    and the AWS Bedrock Converse API format.

    Attributes:
        model_id: The Bedrock model ID (e.g., 'anthropic.claude-sonnet-4-5-20250929-v1:0').
        client: boto3 bedrock-runtime client instance.
    """

    def __init__(
        self,
        model_id: str,
        client: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the Bedrock model.

        Args:
            model_id: Bedrock model ID (e.g., 'anthropic.claude-sonnet-4-5-20250929-v1:0').
            client: boto3 bedrock-runtime client instance.
        """
        self.model_id = model_id
        self.client = client

    def _get_effective_model_id(self) -> str:
        """Get the effective model ID, adding inference profile prefix if needed.

        When using API key authentication (AWS_BEARER_TOKEN_BEDROCK), AWS Bedrock
        requires inference profile IDs with regional prefixes (us., eu., apac.)
        instead of direct model IDs for supported providers.

        **Important**: Not all model providers support cross-region inference profiles.
        Only certain providers (Anthropic, Amazon, Meta, Mistral, Cohere, DeepSeek)
        support the regional prefix. Other providers (AI21, Stability) require
        direct model IDs and will fail if a regional prefix is added.

        This method automatically detects if API keys are being used and adds
        the appropriate regional prefix based on AWS_REGION if:
        1. The model ID doesn't already have a prefix
        2. The model provider supports inference profiles

        Returns:
            The model ID to use for the API call.
        """
        # Check if already has an inference profile prefix
        if self.model_id.startswith(_INFERENCE_PROFILE_PREFIXES):
            return self.model_id

        # Check if using API key authentication
        if 'AWS_BEARER_TOKEN_BEDROCK' not in os.environ:
            return self.model_id

        # Check if this model provider supports inference profiles
        if not self.model_id.startswith(_INFERENCE_PROFILE_SUPPORTED_PROVIDERS):
            logger.debug(
                'Model provider does not support inference profiles, using direct model ID',
                model_id=self.model_id,
            )
            return self.model_id

        # API key auth requires inference profiles - add regional prefix
        region = os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION')
        if not region:
            logger.warning('API key auth detected but AWS_REGION not set. Using direct model ID which may fail.')
            return self.model_id

        # Determine regional prefix
        region_lower = region.lower()
        if region_lower.startswith('us-') or region_lower.startswith('us_'):
            prefix = 'us'
        elif region_lower.startswith('eu-') or region_lower.startswith('eu_'):
            prefix = 'eu'
        elif region_lower.startswith(('ap-', 'ap_', 'cn-', 'cn_', 'me-', 'me_', 'af-', 'af_', 'sa-', 'sa_')):
            prefix = 'apac'
        else:
            prefix = 'us'  # Default to US for unknown regions

        effective_id = f'{prefix}.{self.model_id}'
        logger.debug(
            'Auto-converting model ID to inference profile',
            original=self.model_id,
            effective=effective_id,
        )
        return effective_id

    async def generate(
        self,
        request: GenerateRequest,
        ctx: ActionRunContext | None = None,
    ) -> GenerateResponse:
        """Generate a response from AWS Bedrock.

        Args:
            request: The generation request containing messages and config.
            ctx: Action run context for streaming support.

        Returns:
            GenerateResponse with the model's output.
        """
        config = self._normalize_config(request.config)
        params = await self._build_request_body(request, config)
        streaming = ctx is not None and ctx.is_streaming

        logger.debug(
            'Bedrock generate request',
            model_id=self.model_id,
            streaming=streaming,
        )

        try:
            if streaming and ctx is not None:
                return await self._generate_streaming(params, ctx, request)

            # Non-streaming request using Converse API
            response = await self.client.converse(**params)
        except Exception as e:
            logger.exception(
                'Bedrock API call failed',
                model_id=self.model_id,
                error=str(e),
            )
            raise

        # Extract the output message
        output = response.get('output', {})
        message_data = output.get('message', {})

        # Convert response to Genkit format
        content = self._from_bedrock_content(message_data.get('content', []))
        content = maybe_strip_fences(request, content)
        response_message = Message(role=Role.MODEL, content=content)

        # Build usage statistics
        usage_data = response.get('usage', {})
        usage = GenerationUsage(
            input_tokens=usage_data.get('inputTokens', 0),
            output_tokens=usage_data.get('outputTokens', 0),
            total_tokens=usage_data.get('totalTokens', 0),
        )

        stop_reason = response.get('stopReason', '')
        finish_reason = FINISH_REASON_MAP.get(stop_reason, FinishReason.UNKNOWN)

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
        """Handle streaming generation using ConverseStream.

        Args:
            params: Request parameters for the API.
            ctx: Action run context for sending chunks.
            request: Original generation request.

        Returns:
            Final GenerateResponse after streaming completes.
        """
        try:
            # Use converse_stream for streaming
            response = await self.client.converse_stream(**params)
        except Exception as e:
            logger.exception(
                'Bedrock streaming API call failed',
                model_id=self.model_id,
                error=str(e),
            )
            raise

        accumulated_content: list[Part] = []
        accumulated_tool_uses: dict[str, dict[str, Any]] = {}
        final_usage: GenerationUsage | None = None
        stop_reason: str = ''

        json_mode = bool(request.output and request.output.format == 'json')
        fence_stripper = StreamingFenceStripper(json_mode=json_mode)

        def _send_text(text: str) -> None:
            """Send a text chunk and accumulate it."""
            if not text:
                return
            part = Part(root=TextPart(text=text))
            accumulated_content.append(part)
            ctx.send_chunk(GenerateResponseChunk(role=Role.MODEL, content=[part], index=0))

        # Process the event stream
        stream = response.get('stream', [])
        async for event in stream:
            # Handle content block delta (text chunks)
            if 'contentBlockDelta' in event:
                delta = event['contentBlockDelta'].get('delta', {})
                if 'text' in delta:
                    _send_text(fence_stripper.process(delta['text']))
                # Handle tool use delta (input fragments only).
                # contentBlockStart always fires first with name/toolUseId,
                # so by the time we get here the entry should already exist.
                if 'toolUse' in delta:
                    tool_use = delta['toolUse']
                    block_idx = str(event['contentBlockDelta'].get('contentBlockIndex', 0))
                    if block_idx not in accumulated_tool_uses:
                        # Defensive fallback — should not happen in practice
                        # because contentBlockStart always precedes deltas.
                        accumulated_tool_uses[block_idx] = {
                            'toolUseId': tool_use.get('toolUseId', ''),
                            'name': tool_use.get('name', ''),
                            'input': '',
                        }
                    if 'input' in tool_use:
                        tool_input = tool_use['input']
                        if isinstance(tool_input, str):
                            accumulated_tool_uses[block_idx]['input'] += tool_input

            # Handle content block start (for tool use)
            if 'contentBlockStart' in event:
                start = event['contentBlockStart'].get('start', {})
                if 'toolUse' in start:
                    tool_use = start['toolUse']
                    block_index = event['contentBlockStart'].get('contentBlockIndex', 0)
                    accumulated_tool_uses[str(block_index)] = {
                        'toolUseId': tool_use.get('toolUseId', ''),
                        'name': tool_use.get('name', ''),
                        'input': '',
                    }

            # Handle metadata (usage)
            if 'metadata' in event:
                metadata = event['metadata']
                usage_data = metadata.get('usage', {})
                final_usage = GenerationUsage(
                    input_tokens=usage_data.get('inputTokens', 0),
                    output_tokens=usage_data.get('outputTokens', 0),
                    total_tokens=usage_data.get('totalTokens', 0),
                )

            # Handle content block stop — emit tool request chunks
            if 'contentBlockStop' in event:
                block_idx = str(event['contentBlockStop'].get('contentBlockIndex', 0))
                if block_idx in accumulated_tool_uses:
                    tool_data = accumulated_tool_uses[block_idx]
                    try:
                        tool_input = json.loads(tool_data['input']) if tool_data['input'] else {}
                    except json.JSONDecodeError:
                        tool_input = tool_data['input']
                    ctx.send_chunk(
                        GenerateResponseChunk(
                            role=Role.MODEL,
                            content=[
                                Part(
                                    root=ToolRequestPart(
                                        tool_request=ToolRequest(
                                            ref=tool_data['toolUseId'],
                                            name=tool_data['name'],
                                            input=tool_input,
                                        )
                                    )
                                )
                            ],
                            index=0,
                        )
                    )

            # Handle message stop
            if 'messageStop' in event:
                stop_reason = event['messageStop'].get('stopReason', '')

        # Flush any remaining fence buffer (short streams).
        _send_text(fence_stripper.flush())

        # Add accumulated tool uses to content
        for tool_data in accumulated_tool_uses.values():
            try:
                tool_input = json.loads(tool_data['input']) if tool_data['input'] else {}
            except json.JSONDecodeError:
                tool_input = tool_data['input']

            accumulated_content.append(
                Part(
                    root=ToolRequestPart(
                        tool_request=ToolRequest(
                            ref=tool_data['toolUseId'],
                            name=tool_data['name'],
                            input=tool_input,
                        )
                    )
                )
            )

        finish_reason = FINISH_REASON_MAP.get(stop_reason, FinishReason.UNKNOWN)

        return GenerateResponse(
            message=Message(role=Role.MODEL, content=accumulated_content),
            usage=final_usage,
            finish_reason=finish_reason,
            request=request,
        )

    def _normalize_config(self, config: object) -> BedrockConfig:
        """Normalize config to BedrockConfig.

        Args:
            config: Request configuration (dict, BedrockConfig, or GenerationCommonConfig).

        Returns:
            Normalized BedrockConfig instance.
        """
        if config is None:
            return BedrockConfig()

        if isinstance(config, BedrockConfig):
            return config

        if isinstance(config, GenerationCommonConfig):
            max_tokens = int(config.max_output_tokens) if config.max_output_tokens is not None else None
            return BedrockConfig(
                temperature=config.temperature,
                max_tokens=max_tokens,
                top_p=config.top_p,
                stop_sequences=config.stop_sequences,
            )

        if isinstance(config, dict):
            # Handle camelCase to snake_case mapping
            mapped: dict[str, Any] = {}
            key_map: dict[str, str] = {
                'maxOutputTokens': 'max_tokens',
                'maxTokens': 'max_tokens',
                'topP': 'top_p',
                'topK': 'top_k',
                'stopSequences': 'stop_sequences',
            }
            for key, value in config.items():
                str_key = str(key)
                mapped_key = key_map.get(str_key, str_key)
                mapped[mapped_key] = value
            return BedrockConfig(**mapped)

        return BedrockConfig()

    async def _build_request_body(
        self,
        request: GenerateRequest,
        config: BedrockConfig,
    ) -> dict[str, Any]:
        """Build the AWS Bedrock Converse API request body.

        Args:
            request: The generation request.
            config: Normalized configuration.

        Returns:
            Dictionary suitable for client.converse().
        """
        # Separate system messages from conversation messages
        system_messages, conversation_messages = self._separate_system_messages(request.messages)

        body: dict[str, Any] = {
            'modelId': self._get_effective_model_id(),
            'messages': await self._to_bedrock_messages(conversation_messages),
        }

        # Handle JSON output format by injecting instructions into system prompt
        # The Converse API doesn't have native JSON mode, so we use prompt engineering
        json_instruction = self._build_json_instruction(request)
        if json_instruction:
            system_messages.append(json_instruction)

        # Add system prompt if present
        if system_messages:
            body['system'] = [{'text': msg} for msg in system_messages]

        # Build inference config
        inference_config: dict[str, Any] = {}

        if config.max_tokens is not None:
            inference_config['maxTokens'] = config.max_tokens
        elif config.max_output_tokens is not None:
            inference_config['maxTokens'] = config.max_output_tokens

        if config.temperature is not None:
            inference_config['temperature'] = config.temperature
        if config.top_p is not None:
            inference_config['topP'] = config.top_p
        if config.stop_sequences is not None:
            inference_config['stopSequences'] = config.stop_sequences

        if inference_config:
            body['inferenceConfig'] = inference_config

        # Handle tools
        if request.tools:
            body['toolConfig'] = {
                'tools': [self._to_bedrock_tool(t) for t in request.tools],
            }

        return body

    def _build_json_instruction(self, request: GenerateRequest) -> str | None:
        """Build a JSON output instruction based on request.output configuration.

        The Bedrock Converse API doesn't have native JSON mode like OpenAI's response_format.
        Instead, we inject instructions into the system prompt to ensure JSON output.

        Args:
            request: The generation request.

        Returns:
            JSON instruction string if JSON output is requested, None otherwise.
        """
        if not request.output:
            return None

        output_format = request.output.format
        schema = request.output.schema

        if output_format != 'json':
            return None

        # Build instruction for JSON output
        instruction_parts = [
            'IMPORTANT: You MUST respond with valid JSON only.',
            'Do not include any text before or after the JSON.',
            'Do not wrap the JSON in markdown code blocks.',
        ]

        if schema:
            # Include the schema in the instruction
            schema_str = json.dumps(schema, indent=2)
            instruction_parts.append(f'Your response MUST conform to this JSON schema:\n{schema_str}')

        return '\n'.join(instruction_parts)

    def _separate_system_messages(
        self,
        messages: list[Message],
    ) -> tuple[list[str], list[Message]]:
        """Separate system messages from conversation messages.

        The Converse API requires system messages to be passed separately.

        Args:
            messages: List of Genkit messages.

        Returns:
            Tuple of (system_texts, conversation_messages).
        """
        system_texts: list[str] = []
        conversation_messages: list[Message] = []

        for msg in messages:
            if msg.role == Role.SYSTEM or (isinstance(msg.role, str) and msg.role.lower() == 'system'):
                # Extract text from system message
                text_parts: list[str] = []
                for part in msg.content:
                    root = part.root if isinstance(part, Part) else part
                    if isinstance(root, TextPart):
                        text_parts.append(root.text)
                if text_parts:
                    system_texts.append(''.join(text_parts))
            else:
                conversation_messages.append(msg)

        return system_texts, conversation_messages

    def _to_bedrock_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        """Convert a Genkit tool definition to Bedrock format.

        Args:
            tool: Genkit ToolDefinition.

        Returns:
            Bedrock-compatible tool specification.
        """
        input_schema = tool.input_schema or {'type': 'object', 'properties': {}}

        return {
            'toolSpec': {
                'name': tool.name,
                'description': tool.description or '',
                'inputSchema': {
                    'json': input_schema,
                },
            },
        }

    async def _to_bedrock_messages(
        self,
        messages: list[Message],
    ) -> list[dict[str, Any]]:
        """Convert Genkit messages to Bedrock Converse API message format.

        Args:
            messages: List of Genkit messages (excluding system messages).

        Returns:
            List of Bedrock-compatible message dictionaries.
        """
        bedrock_msgs: list[dict[str, Any]] = []

        for msg in messages:
            role = self._to_bedrock_role(msg.role)
            content: list[dict[str, Any]] = []

            for part in msg.content:
                root = part.root if isinstance(part, Part) else part

                if isinstance(root, TextPart):
                    content.append({'text': root.text})

                elif isinstance(root, MediaPart):
                    media = root.media
                    content.append(await self._convert_media_to_bedrock(media))

                elif isinstance(root, ToolRequestPart):
                    # Tool use from assistant
                    tool_req = root.tool_request
                    content.append({
                        'toolUse': {
                            'toolUseId': tool_req.ref or '',
                            'name': tool_req.name,
                            'input': tool_req.input if isinstance(tool_req.input, dict) else {},
                        },
                    })

                elif isinstance(root, ToolResponsePart):
                    # Tool result from user
                    tool_resp = root.tool_response
                    output = tool_resp.output
                    result_content = [{'text': output}] if isinstance(output, str) else [{'json': output}]

                    content.append({
                        'toolResult': {
                            'toolUseId': tool_resp.ref or '',
                            'content': result_content,
                        },
                    })

            if content:
                bedrock_msgs.append({
                    'role': role,
                    'content': content,
                })

        return bedrock_msgs

    async def _convert_media_to_bedrock(self, media: Media) -> dict[str, Any]:
        """Convert Genkit Media to Bedrock image/video format.

        Args:
            media: Genkit Media object.

        Returns:
            Bedrock-compatible media content block.

        Raises:
            ValueError: If the media URL cannot be fetched or is invalid.
        """
        url = media.url
        content_type = media.content_type or ''

        # Determine if this is an image or video
        is_image = content_type.startswith('image/') or any(
            ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        )
        is_video = content_type.startswith('video/') or any(ext in url.lower() for ext in ['.mp4', '.webm', '.mov'])

        # Extract format from content type or URL
        if content_type:
            format_str = content_type.split('/')[-1]
        else:
            # Guess from URL extension
            for ext in ['jpeg', 'jpg', 'png', 'gif', 'webp', 'mp4', 'webm', 'mov']:
                if f'.{ext}' in url.lower():
                    format_str = ext if ext != 'jpg' else 'jpeg'
                    break
            else:
                format_str = 'jpeg'  # Default

        # Handle base64 data URLs
        if url.startswith('data:'):
            # Parse data URL: data:image/png;base64,<data>
            parts = url.split(',', 1)
            if len(parts) == 2:
                media_bytes = base64.b64decode(parts[1])
                if is_image or not is_video:
                    return {
                        'image': {
                            'format': format_str,
                            'source': {'bytes': media_bytes},
                        },
                    }
                else:
                    return {
                        'video': {
                            'format': format_str,
                            'source': {'bytes': media_bytes},
                        },
                    }

        # For regular URLs, fetch the content asynchronously
        # Bedrock doesn't support URL sources directly, we must provide bytes
        media_bytes, format_str = await self._fetch_media_from_url(url, format_str)

        if is_image or not is_video:
            return {
                'image': {
                    'format': format_str,
                    'source': {'bytes': media_bytes},
                },
            }
        else:
            return {
                'video': {
                    'format': format_str,
                    'source': {'bytes': media_bytes},
                },
            }

    # TODO(#4360): Replace with downloadRequestMedia middleware (G15 parity).
    # User-Agent is required because many servers (e.g. Wikipedia) return
    # 403 Forbidden for the default httpx user-agent string.
    _DOWNLOAD_HEADERS: dict[str, str] = {
        'User-Agent': 'Genkit/1.0 (https://github.com/firebase/genkit; genkit@google.com)',
        'Accept': 'image/*,video/*,*/*',
    }

    async def _fetch_media_from_url(self, url: str, default_format: str) -> tuple[bytes, str]:
        """Fetch media content from a URL asynchronously.

        Args:
            url: The URL to fetch media from.
            default_format: Default format to use if not detected from response.

        Returns:
            Tuple of (media_bytes, format_string).

        Raises:
            ValueError: If the URL cannot be fetched.
        """
        logger.debug('Fetching media from URL', url=url[:100])

        try:
            client = get_cached_client(
                cache_key='amazon-bedrock/media-fetch',
                headers=self._DOWNLOAD_HEADERS,
                timeout=30.0,
                follow_redirects=True,
            )
            response = await client.get(url)
            response.raise_for_status()

            media_bytes = response.content
            format_str = default_format

            # Update format from response content-type if available
            resp_content_type = response.headers.get('content-type', '')
            if resp_content_type and '/' in resp_content_type:
                format_str = resp_content_type.split('/')[-1].split(';')[0]
                if format_str == 'jpg':
                    format_str = 'jpeg'

            logger.debug('Fetched media', size=len(media_bytes), format=format_str)
            return media_bytes, format_str
        except httpx.HTTPStatusError as e:
            logger.exception('HTTP error fetching media URL', url=url[:100], status=e.response.status_code)
            raise ValueError(f'HTTP {e.response.status_code} fetching media from URL: {url[:100]}...') from e
        except Exception as e:
            logger.exception('Failed to fetch media URL', url=url[:100], error=str(e))
            raise ValueError(f'Failed to fetch media from URL: {url[:100]}... Error: {e}') from e

    def _to_bedrock_role(self, role: Role | str) -> str:
        """Convert Genkit role to Bedrock role.

        Args:
            role: Genkit message role.

        Returns:
            Bedrock role string ('user' or 'assistant').
        """
        if isinstance(role, str):
            str_role_map = {
                'user': 'user',
                'model': 'assistant',
                'assistant': 'assistant',
                'tool': 'user',  # Tool responses come from user role
            }
            return str_role_map.get(role.lower(), 'user')

        role_map = {
            Role.USER: 'user',
            Role.MODEL: 'assistant',
            Role.TOOL: 'user',  # Tool responses come from user role
        }
        return role_map.get(role, 'user')

    def _from_bedrock_content(self, content_blocks: list[dict[str, Any]]) -> list[Part]:
        """Convert Bedrock response content to Genkit parts.

        Args:
            content_blocks: List of Bedrock content blocks.

        Returns:
            List of Genkit Part objects.
        """
        parts: list[Part] = []

        for block in content_blocks:
            # Handle text content
            if 'text' in block:
                parts.append(Part(root=TextPart(text=block['text'])))

            # Handle tool use
            if 'toolUse' in block:
                tool_use = block['toolUse']
                parts.append(
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(
                                ref=tool_use.get('toolUseId', ''),
                                name=tool_use.get('name', ''),
                                input=tool_use.get('input', {}),
                            )
                        )
                    )
                )

            # Handle reasoning content (DeepSeek-R1, etc.)
            if 'reasoningContent' in block:
                reasoning = block['reasoningContent']
                if 'reasoningText' in reasoning:
                    # Include reasoning as a text part with prefix
                    reasoning_text = reasoning['reasoningText']
                    if isinstance(reasoning_text, dict) and 'text' in reasoning_text:
                        parts.append(Part(root=TextPart(text=f'[Reasoning]\n{reasoning_text["text"]}\n[/Reasoning]\n')))
                    elif isinstance(reasoning_text, str):
                        parts.append(Part(root=TextPart(text=f'[Reasoning]\n{reasoning_text}\n[/Reasoning]\n')))

        return parts
