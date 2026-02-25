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

"""Models package for Ollama plugin.

This module implements the model interface for Ollama using its Python client.

See:
- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md
- Ollama Python Client: https://github.com/ollama/ollama-python

Key Features
------------
- Chat completions using the ``/api/chat`` endpoint
- Text generation using the ``/api/generate`` endpoint
- Tool/function calling support
- Streaming responses
- Multimodal inputs (images for vision models like ``llava``)

Implementation Notes & Edge Cases
----------------------------------

**Media URL Handling (Ollama-Specific Requirement)**

The Ollama Python client's ``Image`` type only accepts base64 strings, raw
bytes, or local file paths. It does **not** accept HTTP URLs or full data
URIs. When a string value ending in a known image extension (e.g. ``.jpg``,
``.png``) is passed, the client attempts to interpret it as a local file path
and raises ``ValueError: File ... does not exist`` if the path doesn't exist.

This means we must resolve media URLs client-side before passing to Ollama::

    # Ollama client raises ValueError for HTTP URLs:
    ollama.Image(value='https://example.com/cat.jpg')  # ❌ ValueError

    # We resolve to raw bytes first:
    image_bytes = await fetch(url)
    ollama.Image(value=image_bytes)  # ✅ Works

The ``_resolve_image()`` method handles three cases:

- **Data URIs** (``data:image/jpeg;base64,...``): Strips the prefix and
  returns the raw base64 string, matching the JS canonical Ollama plugin.
- **HTTP/HTTPS URLs**: Downloads the image using the shared
  ``get_cached_client()`` utility and returns raw bytes.
- **Other strings** (local file paths, raw base64): Passed through
  unchanged for the ``Image`` type to handle.

**User-Agent Header Requirement**

Some servers (notably Wikipedia/Wikimedia) block requests without a proper
``User-Agent`` header, returning HTTP 403 Forbidden. We include a standard
User-Agent header when fetching images::

    headers = {
        'User-Agent': 'Genkit/1.0 (https://github.com/firebase/genkit; genkit@google.com)',
    }

**JS Canonical Parity**

The JS Ollama plugin (``js/plugins/ollama/src/index.ts``) bypasses the
client library and constructs raw HTTP requests to ``/api/chat``, passing
image data as plain strings in the ``images[]`` array. It only strips data
URI prefixes but does **not** download HTTP URLs — the JS Ollama server
handles URL fetching natively.

The Python ``ollama`` client library adds stricter validation (via Pydantic)
that rejects URLs, so we must download images explicitly. This is the only
behavioral divergence from the JS plugin.
"""

import mimetypes
from collections.abc import Callable
from typing import Any, Literal, cast

import structlog
from pydantic import BaseModel

import ollama as ollama_api
from genkit.ai import ActionRunContext
from genkit.blocks.model import get_basic_usage_stats
from genkit.core.http_client import get_cached_client
from genkit.plugins.ollama.constants import (
    OllamaAPITypes,
)
from genkit.types import (
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
    ToolRequest,
    ToolRequestPart,
    ToolResponsePart,
)

logger = structlog.get_logger(__name__)


class OllamaSupports(BaseModel):
    """Supports for Ollama models."""

    tools: bool = True


class ModelDefinition(BaseModel):
    """Meta definition for Ollama models."""

    name: str
    api_type: OllamaAPITypes = OllamaAPITypes.CHAT
    supports: OllamaSupports = OllamaSupports()


class OllamaModel:
    """Represents an Ollama language model for use with Genkit.

    This class encapsulates the interaction logic for a specific Ollama model,
    allowing it to be integrated into the Genkit framework for generative tasks.
    """

    def __init__(self, client: Callable, model_definition: ModelDefinition) -> None:
        """Initializes the OllamaModel.

        Sets up the client factory for communicating with the Ollama server and stores
        the definition of the model.

        Note: We store the client factory (not the client instance) to avoid async
        event loop binding issues. The client is created fresh per request to ensure
        it's bound to the correct event loop.

        Args:
            client: A callable that returns an asynchronous Ollama client instance.
            model_definition: The definition describing the specific Ollama model
                to be used (e.g., its name, API type, supported features).
        """
        self._client_factory = client
        self.model_definition = model_definition

    def _get_client(self) -> ollama_api.AsyncClient:
        """Creates a fresh async client bound to the current event loop.

        This ensures the httpx client is not reused across different event loops,
        which would cause 'bound to a different event loop' errors.

        Returns:
            A fresh Ollama async client instance.
        """
        return self._client_factory()

    async def generate(self, request: GenerateRequest, ctx: ActionRunContext | None = None) -> GenerateResponse:
        """Generate a response from Ollama.

        Args:
            request: The request to generate a response for.
            ctx: The context to generate a response for.

        Returns:
            The generated response.
        """
        content = [Part(root=TextPart(text='Failed to get response from Ollama API'))]

        logger.debug(
            'Ollama generate request',
            model=self.model_definition.name,
            api_type=str(self.model_definition.api_type),
            streaming=self.is_streaming_request(ctx=ctx),
        )

        if self.model_definition.api_type == OllamaAPITypes.CHAT:
            api_response = await self._chat_with_ollama(request=request, ctx=ctx)
            if api_response:
                logger.debug(
                    'Ollama raw API response',
                    model=self.model_definition.name,
                    content=str(api_response.message.content)[:500] if api_response.message else None,
                )
                content = self._build_multimodal_chat_response(
                    chat_response=api_response,
                )
        elif self.model_definition.api_type == OllamaAPITypes.GENERATE:
            api_response = await self._generate_ollama_response(request=request, ctx=ctx)
            if api_response:
                logger.debug(
                    'Ollama raw API response',
                    model=self.model_definition.name,
                    response=str(api_response.response)[:500],
                )
                content = [Part(root=TextPart(text=api_response.response))]
        else:
            raise ValueError(f'Unresolved API type: {self.model_definition.api_type}')

        if self.is_streaming_request(ctx=ctx):
            content = []

        response_message = Message(
            role=Role.MODEL,
            content=content,
        )

        basic_generation_usage = get_basic_usage_stats(
            input_=request.messages,
            response=response_message,
        )

        return GenerateResponse(
            message=Message(
                role=Role.MODEL,
                content=content,
            ),
            usage=self.get_usage_info(
                basic_generation_usage=basic_generation_usage,
                api_response=api_response,
            ),
        )

    async def _chat_with_ollama(
        self, request: GenerateRequest, ctx: ActionRunContext | None = None
    ) -> ollama_api.ChatResponse | None:
        """Chat with Ollama.

        Args:
            request: The request to chat with Ollama for.
            ctx: The context to chat with Ollama for.

        Returns:
            The chat response from Ollama.
        """
        messages = await self.build_chat_messages(request)
        streaming_request = self.is_streaming_request(ctx=ctx)

        if request.output:
            # ollama api either accepts 'json' literal, or the JSON schema
            if request.output.schema:
                fmt = request.output.schema
            elif request.output.format:
                fmt = request.output.format
            else:
                fmt = ''
        else:
            fmt = ''

        # Build common kwargs for both streaming and non-streaming calls
        tools = [
            ollama_api.Tool(
                function=ollama_api.Tool.Function(
                    name=tool.name,
                    description=tool.description,
                    parameters=_convert_parameters(tool.input_schema or {}),
                )
            )
            for tool in request.tools or []
        ]
        options = self.build_request_options(config=request.config)

        if streaming_request:
            # Streaming call with literal stream=True for proper overload resolution
            chat_response = await self._get_client().chat(  # type: ignore[no-matching-overload]
                model=self.model_definition.name,
                messages=messages,
                tools=tools,
                options=options,
                format=fmt,
                stream=True,
            )
            idx = 0
            async for chunk in chat_response:
                idx += 1
                role = Role.MODEL if chunk.message.role == 'assistant' else Role.TOOL
                if ctx:
                    ctx.send_chunk(
                        chunk=GenerateResponseChunk(
                            role=role,
                            index=idx,
                            content=self._build_multimodal_chat_response(chat_response=chunk),
                        )
                    )
            # For streaming requests, we return None because the response chunks
            # have already been sent via ctx.send_chunk() above. The async generator
            # is now exhausted, and the caller should not expect a return value.
            return None
        else:
            # Non-streaming call with literal stream=False for proper overload resolution
            chat_response = await self._get_client().chat(  # type: ignore[no-matching-overload]
                model=self.model_definition.name,
                messages=messages,
                tools=tools,
                options=options,
                format=fmt,
                stream=False,
            )
            return chat_response

    async def _generate_ollama_response(
        self, request: GenerateRequest, ctx: ActionRunContext | None = None
    ) -> ollama_api.GenerateResponse | None:
        """Generate a response from Ollama.

        Args:
            request: The request to generate a response for.
            ctx: The context to generate a response for.

        Returns:
            The generated response.
        """
        prompt = self.build_prompt(request)
        streaming_request = self.is_streaming_request(ctx=ctx)
        options = self.build_request_options(config=request.config)

        if streaming_request:
            # Streaming call with literal stream=True for proper overload resolution
            generate_response = await self._get_client().generate(
                model=self.model_definition.name,
                prompt=prompt,
                options=options,
                stream=True,
            )
            idx = 0
            async for chunk in generate_response:
                idx += 1
                if ctx:
                    ctx.send_chunk(
                        chunk=GenerateResponseChunk(
                            role=Role.MODEL,
                            index=idx,
                            content=[Part(root=TextPart(text=chunk.response))],
                        )
                    )
            # For streaming requests, we return None because the response chunks
            # have already been sent via ctx.send_chunk() above. The async generator
            # is now exhausted, and the caller should not expect a return value.
            return None
        else:
            # Non-streaming call with literal stream=False for proper overload resolution
            generate_response = await self._get_client().generate(
                model=self.model_definition.name,
                prompt=prompt,
                options=options,
                stream=False,
            )
            return generate_response

    @staticmethod
    def _build_multimodal_chat_response(
        chat_response: ollama_api.ChatResponse,
    ) -> list[Part]:
        """Build the multimodal chat response.

        Args:
            chat_response: The chat response to build the multimodal response for.

        Returns:
            The multimodal chat response.
        """
        content = []
        chat_response_message = chat_response.message
        if chat_response_message.content:
            content.append(Part(root=TextPart(text=chat_response.message.content or '')))
        if chat_response_message.images:
            for image in chat_response_message.images:
                content.append(
                    Part(
                        root=MediaPart(
                            media=Media(
                                content_type=mimetypes.guess_type(str(image.value), strict=False)[0]
                                or 'application/octet-stream',
                                url=str(image.value),
                            )
                        )
                    )
                )
        if chat_response_message.tool_calls:
            for tool_call in chat_response_message.tool_calls:
                content.append(
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(
                                name=tool_call.function.name,
                                input=tool_call.function.arguments,
                            )
                        )
                    )
                )
        return content

    @staticmethod
    def build_request_options(
        config: GenerationCommonConfig | ollama_api.Options | dict[str, object] | None,
    ) -> ollama_api.Options:
        """Build request options for the generate API.

        Args:
            config: The configuration to build the request options for.

        Returns:
            The request options for the generate API.
        """
        if config is None:
            return ollama_api.Options()
        if isinstance(config, GenerationCommonConfig):
            config = dict(
                top_k=config.top_k,
                topP=config.top_p,
                stop=config.stop_sequences,
                temperature=config.temperature,
                num_predict=config.max_output_tokens,
            )
        if isinstance(config, dict):
            # Use cast to avoid type error with **spread of dict[str, object]
            config = ollama_api.Options(**cast(dict[str, Any], config))

        return config

    @staticmethod
    def build_prompt(request: GenerateRequest) -> str:
        """Build the prompt for the generate API.

        Args:
            request: The request to build the prompt for.

        Returns:
            The prompt for the generate API.
        """
        prompt = ''
        for message in request.messages:
            for text_part in message.content:
                if isinstance(text_part.root, TextPart):
                    prompt += text_part.root.text
                else:
                    logger.error('Non-text messages are not supported')
        return prompt

    @classmethod
    async def build_chat_messages(cls, request: GenerateRequest) -> list[ollama_api.Message]:
        """Build the messages for the chat API.

        Handles MediaPart by converting image URLs to the format expected
        by the Ollama Python client's ``Image`` type, which only accepts
        base64 strings, raw bytes, or local file paths — not HTTP URLs
        or full data URIs.

        For HTTP/HTTPS URLs, the image is downloaded and passed as raw
        bytes. For data URIs, the ``data:...;base64,`` prefix is stripped
        to extract the base64 payload. This matches the JS canonical
        Ollama plugin's ``toOllamaRequest()`` behavior.

        Args:
            request: The request to build the messages for.

        Returns:
            The messages for the chat API.
        """
        messages: list[ollama_api.Message] = []
        for message in request.messages:
            item = ollama_api.Message(
                role=cls._to_ollama_role(role=cast(Role, message.role)),
                content='',
                images=[],
            )
            for text_part in message.content:
                if isinstance(text_part.root, TextPart):
                    item.content = (item.content or '') + text_part.root.text
                elif isinstance(text_part.root, ToolResponsePart):
                    item.content = (item.content or '') + str(text_part.root.tool_response.output)
                elif isinstance(text_part.root, MediaPart):
                    image_value = await cls._resolve_image(text_part.root.media.url)
                    item['images'].append(ollama_api.Image(value=image_value))
            messages.append(item)
        return messages

    @staticmethod
    async def _resolve_image(url: str) -> str | bytes:
        """Convert a media URL to a value the Ollama Image type accepts.

        The Ollama Python client's ``Image`` type only accepts base64
        strings, raw bytes, or local file paths. This method handles:

        - **Data URIs**: Strips the ``data:...;base64,`` prefix and
          returns the raw base64 string.
        - **HTTP/HTTPS URLs**: Downloads the image and returns the raw
          bytes.
        - **Other strings** (e.g. local file paths or raw base64):
          Passed through unchanged.

        Args:
            url: The media URL from a ``MediaPart``.

        Returns:
            A value suitable for ``ollama.Image(value=...)``.
        """
        if url.startswith('data:'):
            # Strip data URI prefix → raw base64: "data:image/jpeg;base64,ABC" → "ABC"
            comma_idx = url.index(',')
            return url[comma_idx + 1 :]

        if url.startswith(('http://', 'https://')):
            # TODO(#4360): Replace with downloadRequestMedia middleware (G15 parity).
            # Some servers (e.g., Wikipedia/Wikimedia) block requests
            # without a proper User-Agent, returning HTTP 403 Forbidden.
            client = get_cached_client(
                cache_key='ollama/image-fetch',
                timeout=60.0,
                headers={
                    'User-Agent': 'Genkit/1.0 (https://github.com/firebase/genkit; genkit@google.com)',
                },
                follow_redirects=True,
            )
            response = await client.get(url)
            response.raise_for_status()
            return response.content

        # Local file path or raw base64 — pass through to Image.
        return url

    @staticmethod
    def _to_ollama_role(
        role: Role,
    ) -> Literal['user', 'assistant', 'system', 'tool']:
        match role:
            case Role.USER:
                return 'user'
            case Role.MODEL:
                return 'assistant'
            case Role.TOOL:
                return 'tool'
            case Role.SYSTEM:
                return 'system'
            case _:
                raise ValueError(f'Unknown role: {role}')

    @staticmethod
    def is_streaming_request(ctx: ActionRunContext | None) -> bool:
        """Determines if streaming mode is requested."""
        return bool(ctx and ctx.is_streaming)

    @staticmethod
    def get_usage_info(
        basic_generation_usage: GenerationUsage,
        api_response: ollama_api.GenerateResponse | ollama_api.ChatResponse | None,
    ) -> GenerationUsage:
        """Extracts and calculates token usage information from an Ollama API response.

        Updates a basic generation usage object with input, output, and total token counts
        based on the details provided in the Ollama API response.

        Args:
            basic_generation_usage: An existing GenerationUsage object to update.
            api_response: The response object received from the Ollama API,
                containing token count details.

        Returns:
            The updated GenerationUsage object with token counts populated.
        """
        if api_response:
            basic_generation_usage.input_tokens = api_response.prompt_eval_count or 0
            basic_generation_usage.output_tokens = api_response.eval_count or 0
            basic_generation_usage.total_tokens = (
                basic_generation_usage.input_tokens + basic_generation_usage.output_tokens
            )
        return basic_generation_usage


def _convert_parameters(input_schema: dict[str, object]) -> ollama_api.Tool.Function.Parameters | None:
    """Sanitizes a schema to be compatible with Ollama API."""
    if not input_schema or 'type' not in input_schema:
        return None

    schema = ollama_api.Tool.Function.Parameters()
    if 'required' in input_schema:
        required = input_schema['required']
        if isinstance(required, list):
            schema.required = cast(list[str], required)

    if 'type' in input_schema:
        schema_type = input_schema['type']
        if schema_type == 'object':
            schema.type = 'object'

        if schema_type == 'object':
            schema.properties = {}
            properties_raw = input_schema.get('properties', {})
            if isinstance(properties_raw, dict):
                properties = cast(dict[str, dict[str, Any]], properties_raw)
                for key in properties:
                    schema.properties[key] = ollama_api.Tool.Function.Parameters.Property(
                        type=properties[key]['type'], description=properties[key].get('description', '')
                    )

    return schema
