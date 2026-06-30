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
        'User-Agent': 'Genkit/1.0 (https://github.com/genkit-ai/genkit; genkit@google.com)',
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
import re
from collections.abc import Callable
from typing import Any, Literal, cast

import structlog
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel, to_snake

import ollama as ollama_api
from genkit import (
    Media,
    MediaPart,
    Message,
    ModelConfig,
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
from genkit.plugin_api import ActionRunContext, get_cached_client
from genkit.plugins.ollama._errors import wrap_connection_errors
from genkit.plugins.ollama.constants import (
    DEFAULT_OLLAMA_SERVER_URL,
    OllamaAPITypes,
)

logger = structlog.get_logger(__name__)

# Matches <think>/<thinking> blocks case-insensitively (``i``) across newlines
# (``s``), non-greedy (``.*?``) so multiple blocks in one response are captured
# individually. Mirrors the Go plugin's ``thinkingRegex``.
_THINKING_RE = re.compile(r'(?is)<(?:think|thinking)>(.*?)</(?:think|thinking)>')


def _parse_thinking(content: str) -> tuple[str, str]:
    """Split inline ``<think>``/``<thinking>`` blocks out of model content.

    Mirrors the Go plugin's ``parseThinking``: returns the joined reasoning text
    and the remaining content with the thinking blocks removed (both stripped).

    Args:
        content: The raw model content possibly containing thinking tags.

    Returns:
        A ``(reasoning, rest)`` tuple. ``reasoning`` is empty when no tags match,
        in which case ``rest`` is the original content unchanged.
    """
    blocks = _THINKING_RE.findall(content)
    if not blocks:
        return '', content
    reasoning = '\n\n'.join(block.strip() for block in blocks)
    rest = _THINKING_RE.sub('', content).strip()
    return reasoning, rest


class OllamaConfig(ModelConfig):
    """Configuration schema for Ollama models.

    Extends the shared :class:`ModelConfig` with Ollama-specific sampler
    knobs and the ``think`` chain-of-thought control. Unknown keys are
    accepted (``extra='allow'``) and forwarded to the Ollama server's
    ``options`` so newer sampler parameters work without an SDK bump.
    """

    model_config = ConfigDict(alias_generator=to_camel, extra='allow', populate_by_name=True)

    think: bool | Literal['low', 'medium', 'high'] | None = None
    keep_alive: float | str | None = None
    num_ctx: int | None = None
    min_p: float | None = None
    seed: int | None = None
    num_predict: int | None = None


class OllamaSupports(BaseModel):
    """Supports for Ollama models."""

    tools: bool = True
    media: bool = False


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

    def __init__(
        self,
        client: Callable,
        model_definition: ModelDefinition,
        server_address: str = DEFAULT_OLLAMA_SERVER_URL,
    ) -> None:
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
            server_address: The Ollama server URL, surfaced in connectivity errors.
        """
        self._client_factory = client
        self.model_definition = model_definition
        self._server_address = server_address

    def _get_client(self) -> ollama_api.AsyncClient:
        """Creates a fresh async client bound to the current event loop.

        This ensures the httpx client is not reused across different event loops,
        which would cause 'bound to a different event loop' errors.

        Returns:
            A fresh Ollama async client instance.
        """
        return self._client_factory()

    async def generate(
        self,
        request: ModelRequest,
        ctx: ActionRunContext | None = None,
        client: ollama_api.AsyncClient | None = None,
    ) -> ModelResponse:
        """Generate a response from Ollama.

        Args:
            request: The request to generate a response for.
            ctx: The context to generate a response for.
            client: An optional pre-resolved Ollama client to use for this request
                (e.g. one built with per-request headers). Falls back to the stored
                client factory when omitted.

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
            api_response = await self._chat_with_ollama(request=request, ctx=ctx, client=client)
            if api_response:
                logger.debug(
                    'Ollama raw API response',
                    model=self.model_definition.name,
                    content=str(api_response.message.content)[:500] if api_response.message else None,
                )
                content = self._build_multimodal_chat_response(
                    chat_response=api_response,
                    thinking_enabled=self._thinking_requested(request.config),
                )
        elif self.model_definition.api_type == OllamaAPITypes.GENERATE:
            api_response = await self._generate_ollama_response(request=request, ctx=ctx, client=client)
            if api_response:
                logger.debug(
                    'Ollama raw API response',
                    model=self.model_definition.name,
                    response=str(api_response.response)[:500],
                )
                content = self._build_generate_response(
                    generate_response=api_response,
                    thinking_enabled=self._thinking_requested(request.config),
                )
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

        return ModelResponse(
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
        self,
        request: ModelRequest,
        ctx: ActionRunContext | None = None,
        client: ollama_api.AsyncClient | None = None,
    ) -> ollama_api.ChatResponse | None:
        """Chat with Ollama.

        Args:
            request: The request to chat with Ollama for.
            ctx: The context to chat with Ollama for.
            client: An optional pre-resolved Ollama client; falls back to the
                stored client factory when omitted.

        Returns:
            The chat response from Ollama.
        """
        # Resolve media URLs first. build_chat_messages may perform HTTP fetches
        # for image parts, and those must stay *outside* wrap_connection_errors so
        # an image-host failure isn't misreported as an Ollama server outage.
        messages = await self.build_chat_messages(request)
        if client is None:
            client = self._get_client()
        streaming_request = self.is_streaming_request(ctx=ctx)

        if request.output_format or request.output_schema:
            # ollama api either accepts 'json' literal, or the JSON schema
            if request.output_schema:
                fmt = request.output_schema
            elif request.output_format:
                fmt = request.output_format
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
        extra_kwargs = self.build_request_kwargs(config=request.config)

        # Only the Ollama SDK call (and, when streaming, its iteration — where a
        # connection failure can first surface) is wrapped, so transport errors are
        # attributed to the Ollama server rather than to media-URL fetches above.
        if streaming_request:
            async with wrap_connection_errors(self._server_address):
                # Streaming call with literal stream=True for proper overload resolution
                chat_response = await client.chat(  # type: ignore[no-matching-overload]
                    model=self.model_definition.name,
                    messages=messages,
                    tools=tools,
                    options=options,
                    format=fmt,
                    stream=True,
                    **extra_kwargs,
                )
                idx = 0
                async for chunk in chat_response:
                    idx += 1
                    role = self._from_ollama_role(chunk.message.role)
                    if ctx:
                        ctx.send_chunk(
                            chunk=ModelResponseChunk(
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
            async with wrap_connection_errors(self._server_address):
                # Non-streaming call with literal stream=False for proper overload resolution
                chat_response = await client.chat(  # type: ignore[no-matching-overload]
                    model=self.model_definition.name,
                    messages=messages,
                    tools=tools,
                    options=options,
                    format=fmt,
                    stream=False,
                    **extra_kwargs,
                )
            return chat_response

    async def _generate_ollama_response(
        self,
        request: ModelRequest,
        ctx: ActionRunContext | None = None,
        client: ollama_api.AsyncClient | None = None,
    ) -> ollama_api.GenerateResponse | None:
        """Generate a response from Ollama.

        Args:
            request: The request to generate a response for.
            ctx: The context to generate a response for.
            client: An optional pre-resolved Ollama client; falls back to the
                stored client factory when omitted.

        Returns:
            The generated response.
        """
        prompt = self.build_prompt(request)
        if client is None:
            client = self._get_client()
        streaming_request = self.is_streaming_request(ctx=ctx)
        options = self.build_request_options(config=request.config)
        extra_kwargs = self.build_request_kwargs(config=request.config)

        # Wrap only the Ollama SDK call (and its streamed iteration) so transport
        # errors are attributed to the Ollama server, matching the chat path.
        if streaming_request:
            async with wrap_connection_errors(self._server_address):
                # Streaming call with literal stream=True for proper overload resolution
                generate_response = await client.generate(
                    model=self.model_definition.name,
                    prompt=prompt,
                    options=options,
                    stream=True,
                    **extra_kwargs,
                )
                idx = 0
                async for chunk in generate_response:
                    idx += 1
                    if ctx:
                        ctx.send_chunk(
                            chunk=ModelResponseChunk(
                                role=Role.MODEL,
                                index=idx,
                                content=self._build_generate_response(generate_response=chunk),
                            )
                        )
            # For streaming requests, we return None because the response chunks
            # have already been sent via ctx.send_chunk() above. The async generator
            # is now exhausted, and the caller should not expect a return value.
            return None
        else:
            async with wrap_connection_errors(self._server_address):
                # Non-streaming call with literal stream=False for proper overload resolution
                generate_response = await client.generate(
                    model=self.model_definition.name,
                    prompt=prompt,
                    options=options,
                    stream=False,
                    **extra_kwargs,
                )
            return generate_response

    @staticmethod
    def _build_multimodal_chat_response(
        chat_response: ollama_api.ChatResponse,
        thinking_enabled: bool = False,
    ) -> list[Part]:
        """Build the multimodal chat response.

        Args:
            chat_response: The chat response to build the multimodal response for.
            thinking_enabled: Whether the request explicitly enabled thinking. When
                the model returns no dedicated ``thinking`` field, this allows the
                ``<think>``/``<thinking>`` content fallback to run (matching the Go
                plugin). It is only applied to complete (non-streaming) responses,
                never to partial streamed chunks where a tag may be split.

        Returns:
            The multimodal chat response.
        """
        content = []
        chat_response_message = chat_response.message
        text = chat_response_message.content or ''
        # ``think`` chain-of-thought arrives on ``message.thinking``; surface it
        # as a leading ReasoningPart so the Dev UI renders it separately from
        # the answer text. Covers both streaming deltas and the final message.
        thinking = getattr(chat_response_message, 'thinking', None)
        if thinking:
            content.append(Part(root=ReasoningPart(reasoning=thinking)))
        elif thinking_enabled and text:
            # Fallback for models that inline <think>…</think> in content instead
            # of populating the dedicated field. Gated on an explicit think request
            # so ordinary text containing these tags is never hijacked.
            reasoning, text = _parse_thinking(text)
            if reasoning:
                content.append(Part(root=ReasoningPart(reasoning=reasoning)))
        if text:
            content.append(Part(root=TextPart(text=text)))
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
    def _build_generate_response(
        generate_response: ollama_api.GenerateResponse,
        thinking_enabled: bool = False,
    ) -> list[Part]:
        """Build the response parts for a ``generate`` endpoint response.

        Mirrors :meth:`_build_multimodal_chat_response` for the ``generate`` API,
        which returns plain text (no media/tool calls): ``think`` reasoning is
        surfaced as a leading ReasoningPart so the Dev UI renders it separately
        from the answer text.

        Args:
            generate_response: A complete generate response or a streamed chunk.
            thinking_enabled: Whether the request explicitly enabled thinking. When
                the model returns no dedicated ``thinking`` field, this allows the
                ``<think>``/``<thinking>`` content fallback to run (matching the Go
                plugin). It is only applied to complete (non-streaming) responses,
                never to partial streamed chunks where a tag may be split.

        Returns:
            The reasoning/text parts for the response.
        """
        content: list[Part] = []
        text = generate_response.response or ''
        thinking = getattr(generate_response, 'thinking', None)
        if thinking:
            content.append(Part(root=ReasoningPart(reasoning=thinking)))
        elif thinking_enabled and text:
            reasoning, text = _parse_thinking(text)
            if reasoning:
                content.append(Part(root=ReasoningPart(reasoning=reasoning)))
        if text:
            content.append(Part(root=TextPart(text=text)))
        return content

    @staticmethod
    def build_request_options(
        config: ModelConfig | ollama_api.Options | dict[str, object] | None,
    ) -> dict[str, Any]:
        """Build the sampler ``options`` mapping for the chat/generate APIs.

        Accepts an :class:`OllamaConfig`/:class:`ModelConfig` instance, a raw
        ``Options``, or a plain dict (e.g. a config already dumped to JSON by
        the framework — see :meth:`build_request_kwargs`). All inputs are
        normalised to snake-cased Ollama option fields:

        - ``think``/``keep_alive`` are stripped — they are top-level request
          kwargs, not sampler options (Ollama rejects them inside ``options``).
        - Genkit's ``max_output_tokens`` maps to Ollama's ``num_predict``; an
          explicit ``num_predict`` wins when both are present.
        - ``stop_sequences`` maps to ``stop``; ``version``/``api_key`` (genkit
          bookkeeping) are dropped.
        - ``OllamaConfig`` extras (e.g. ``repeatPenalty``) are forwarded
          snake-cased so newer sampler knobs pass through untouched.

        Known knobs are routed through ``ollama_api.Options`` purely for type
        coercion (genkit types ``max_output_tokens``/``top_k`` as floats, but
        Ollama's ``num_predict``/``top_k`` are integers). The result is then
        returned as a plain mapping — *not* an ``Options`` — and any knob the
        installed ``Options`` model doesn't yet field (e.g. ``min_p``) is merged
        back in, so newer sampler parameters still reach the server.

        Args:
            config: The configuration to build the request options for.

        Returns:
            A mapping of snake-cased Ollama sampler options.
        """
        if config is None:
            return {}
        if isinstance(config, ollama_api.Options):
            return config.model_dump(exclude_none=True)

        if isinstance(config, ModelConfig):
            # Covers OllamaConfig (a ModelConfig subclass) and plain ModelConfig.
            # model_dump defaults to by_alias=False, so declared fields come out
            # snake_cased; only extras keep the key they were supplied with.
            # to_snake below normalises both.
            raw: dict[str, Any] = config.model_dump(exclude_none=True)
        else:
            raw = {k: v for k, v in cast(dict[str, Any], config).items() if v is not None}

        # Snake-case so camelCase knobs (e.g. ``topP``) hit the server field
        # instead of being silently dropped.
        knobs = {to_snake(k): v for k, v in raw.items()}

        # Top-level request kwargs, not sampler options.
        knobs.pop('think', None)
        knobs.pop('keep_alive', None)
        # Genkit bookkeeping that Ollama does not understand.
        knobs.pop('version', None)
        knobs.pop('api_key', None)

        if 'stop_sequences' in knobs:
            knobs['stop'] = knobs.pop('stop_sequences')

        max_tokens = knobs.pop('max_output_tokens', None)
        if max_tokens is not None and knobs.get('num_predict') is None:
            knobs['num_predict'] = max_tokens

        # Coerce the knobs Options models (int num_predict/top_k, etc.), then
        # merge back any it drops (e.g. min_p) so they still reach the server.
        options: dict[str, Any] = ollama_api.Options(**knobs).model_dump(exclude_none=True)
        for key, value in knobs.items():
            options.setdefault(key, value)
        return options

    @staticmethod
    def build_request_kwargs(
        config: ModelConfig | ollama_api.Options | dict[str, object] | None,
    ) -> dict[str, Any]:
        """Extract top-level chat/generate kwargs from the config.

        ``think`` and ``keep_alive`` are top-level parameters of the Ollama
        ``chat``/``generate`` calls — not sampler ``options``. The framework
        dumps a ``BaseModel`` config to a dict before the model fn sees it, so
        this reads them from any :class:`ModelConfig` instance *or* a dumped
        dict. Both paths snake-case the keys (declared fields and ``extra``
        keys can arrive camelCased) and return only the values that are set.

        Args:
            config: The configuration to extract request kwargs from.

        Returns:
            A dict with ``think``/``keep_alive`` entries that are not ``None``.
        """
        if isinstance(config, ModelConfig):
            snake = {to_snake(k): v for k, v in config.model_dump(exclude_none=True).items()}
            think: Any = snake.get('think')
            keep_alive: Any = snake.get('keep_alive')
        elif isinstance(config, dict):
            snake = {to_snake(k): v for k, v in cast(dict[str, Any], config).items()}
            think = snake.get('think')
            keep_alive = snake.get('keep_alive')
        else:
            return {}

        kwargs: dict[str, Any] = {}
        if think is not None:
            kwargs['think'] = think
        if keep_alive is not None:
            kwargs['keep_alive'] = keep_alive
        return kwargs

    @staticmethod
    def _thinking_requested(
        config: ModelConfig | ollama_api.Options | dict[str, object] | None,
    ) -> bool:
        """Whether the request explicitly enabled thinking.

        Mirrors the Go plugin's ``ThinkOption.IsEnabled``: a boolean ``think`` is
        taken as-is, a non-empty effort string (``low``/``medium``/``high``) counts
        as enabled, and anything else is disabled. Used to gate the ``<think>`` tag
        content fallback in :meth:`_build_multimodal_chat_response`.

        Args:
            config: The request configuration.

        Returns:
            ``True`` when thinking was explicitly requested, ``False`` otherwise.
        """
        think = OllamaModel.build_request_kwargs(config).get('think')
        if isinstance(think, bool):
            return think
        if isinstance(think, str):
            return think != ''
        return False

    @staticmethod
    def build_prompt(request: ModelRequest) -> str:
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
    async def build_chat_messages(cls, request: ModelRequest) -> list[ollama_api.Message]:
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
            comma_idx = url.find(',')
            if comma_idx == -1:
                raise ValueError(f'Malformed data URI (missing comma separator): {url!r}')
            return url[comma_idx + 1 :]

        if url.startswith(('http://', 'https://')):
            # TODO(#4360): Replace with downloadRequestMedia middleware (G15 parity).
            # Some servers (e.g., Wikipedia/Wikimedia) block requests
            # without a proper User-Agent, returning HTTP 403 Forbidden.
            client = get_cached_client(
                cache_key='ollama/image-fetch',
                timeout=60.0,
                headers={
                    'User-Agent': 'Genkit/1.0 (https://github.com/genkit-ai/genkit; genkit@google.com)',
                },
                follow_redirects=True,
            )
            response = await client.get(url)
            response.raise_for_status()
            return response.content

        # Local file path or raw base64 — pass through to Image.
        return url

    @staticmethod
    def _from_ollama_role(role: str | None) -> Role:
        """Map an Ollama message role onto a Genkit :class:`Role`.

        Ollama streams deltas with an empty role and labels the rest as
        ``assistant``/``tool``/``user``/``system``. Anything unexpected falls
        back to ``MODEL`` (with a warning) so an unknown role never aborts a
        stream.

        Args:
            role: The role string from an Ollama message, possibly empty.

        Returns:
            The corresponding Genkit role.
        """
        match role:
            case 'assistant':
                return Role.MODEL
            case 'tool':
                return Role.TOOL
            case 'user':
                return Role.USER
            case 'system':
                return Role.SYSTEM
            case '' | None:
                # Ollama commonly sends an empty role on streamed deltas.
                return Role.MODEL
            case _:
                logger.warning('Unknown Ollama role; defaulting to MODEL', role=role)
                return Role.MODEL

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
        basic_generation_usage: ModelUsage,
        api_response: ollama_api.GenerateResponse | ollama_api.ChatResponse | None,
    ) -> ModelUsage:
        """Extracts and calculates token usage information from an Ollama API response.

        Updates a basic generation usage object with input, output, and total token counts
        based on the details provided in the Ollama API response.

        Args:
            basic_generation_usage: An existing ModelUsage object to update.
            api_response: The response object received from the Ollama API,
                containing token count details.

        Returns:
            The updated ModelUsage object with token counts populated.
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
    if not input_schema:
        return None

    schema_type = input_schema.get('type')
    if schema_type is None and 'properties' in input_schema:
        # Infer an object schema when properties are present but ``type`` is omitted.
        schema_type = 'object'
    if schema_type != 'object':
        # JS parity (isValidOllamaTool): Ollama only supports object-typed tool inputs.
        raise ValueError(f'Unsupported schema type {schema_type!r}: Ollama only supports tools with object inputs')

    schema = ollama_api.Tool.Function.Parameters()
    schema.type = 'object'

    required = input_schema.get('required')
    if isinstance(required, list):
        schema.required = cast(list[str], required)

    schema.properties = {}
    properties_raw = input_schema.get('properties', {})
    if isinstance(properties_raw, dict):
        properties = cast(dict[str, dict[str, Any]], properties_raw)
        for key in properties:
            schema.properties[key] = ollama_api.Tool.Function.Parameters.Property(
                type=_property_type(properties[key]), description=properties[key].get('description', '')
            )

    return schema


def _property_type(prop: dict[str, Any]) -> str | list[str] | None:
    """Resolves a JSON-schema property to a type Ollama's Property accepts.

    Optional/Union fields serialize as ``anyOf`` with no top-level ``type``
    (e.g. ``Optional[str]`` -> ``{'anyOf': [{'type': 'string'}, {'type': 'null'}]}``).
    Map those to the list form ``Property.type`` accepts instead of crashing on a
    missing ``type`` key or dropping the property (which would leave ``required``
    pointing at a property that no longer exists). Schemas with no resolvable type
    (e.g. ``Any``) fall back to ``None``, which Ollama treats as untyped.
    """
    if 'type' in prop:
        return cast(str | list[str], prop['type'])
    union = prop.get('anyOf') or prop.get('oneOf')
    if isinstance(union, list):
        types: list[str] = []
        for entry in union:
            if isinstance(entry, dict):
                entry_type = entry.get('type')
                if isinstance(entry_type, str):
                    types.append(entry_type)
                elif isinstance(entry_type, list):
                    types.extend(t for t in entry_type if isinstance(t, str))
        if types:
            return list(dict.fromkeys(types))  # order-preserving dedup
    return None
