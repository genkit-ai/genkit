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


"""Utility functions for OpenAI compatible models."""

import base64
import json
import re
from collections.abc import Callable
from typing import Any

from genkit.types import (
    GenerateRequest,
    MediaPart,
    Message,
    Part,
    ReasoningPart,
    Role,
    TextPart,
    ToolRequest,
    ToolRequestPart,
    ToolResponsePart,
)


def strip_markdown_fences(text: str) -> str:
    r"""Strip markdown code fences from a JSON response.

    Models sometimes wrap JSON output in markdown fences like
    ``\`\`\`json ... \`\`\``` even when instructed to output raw
    JSON.  This helper removes the fences.

    Args:
        text: The response text, possibly wrapped in fences.

    Returns:
        The text with markdown fences removed, or the original
        text if no fences are found.
    """
    stripped = text.strip()
    match = re.match(r'^```(?:json)?\s*\n?(.*?)\n?\s*```$', stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _find_text(request: GenerateRequest) -> str | None:
    """Find the first text content from the first message, if any.

    Args:
        request: The generate request.

    Returns:
        The text content, or None if no text is found.
    """
    if not request.messages:
        return None

    return next(
        (part.root.text for part in request.messages[0].content if isinstance(part.root, TextPart) and part.root.text),
        None,
    )


def _extract_text(request: GenerateRequest) -> str:
    """Extract text content from the first message.

    Args:
        request: The generate request.

    Returns:
        The text content.

    Raises:
        ValueError: If no text content is found.
    """
    text = _find_text(request)
    if text is not None:
        return text

    if not request.messages:
        raise ValueError('No messages found in the request')
    raise ValueError('No text content found in the first message')


def parse_data_uri_content_type(url: str) -> str:
    """Extract the content type from a data URI.

    Parses the header part of a ``data:`` URI to extract the media type.
    Handles URIs with and without the ``;base64`` qualifier.

    Args:
        url: A data URI string (must start with ``data:``).

    Returns:
        The extracted content type, or an empty string if parsing fails.

    Examples:
        >>> parse_data_uri_content_type('data:audio/mpeg;base64,AAAA')
        'audio/mpeg'
        >>> parse_data_uri_content_type('data:text/plain,hello')
        'text/plain'
        >>> parse_data_uri_content_type('data:;base64,AAAA')
        ''
    """
    if not url.startswith('data:'):
        return ''
    try:
        header, _ = url.split(',', 1)
        media_type_part = header[len('data:') :]
        return media_type_part.split(';', 1)[0]
    except ValueError:
        return ''


def decode_data_uri_bytes(url: str) -> bytes:
    """Decode the payload of a data URI or raw base64 string to bytes.

    Supports three formats:
    - ``data:`` URIs — extracts and decodes the base64 payload after the comma
    - Raw base64 strings — decoded directly
    - Remote URLs (``http://``, ``https://``) — raises ``ValueError``

    Args:
        url: A data URI, raw base64 string, or URL.

    Returns:
        The decoded bytes.

    Raises:
        ValueError: If the URL is a remote URL or contains invalid base64.
    """
    if url.startswith('data:'):
        try:
            _, b64_data = url.split(',', 1)
            return base64.b64decode(b64_data)
        except (ValueError, TypeError) as e:
            raise ValueError('Invalid data URI format') from e

    if url.startswith(('http://', 'https://')):
        raise ValueError(f'Remote URLs are not supported; provide a base64 data URI instead: {url[:50]}...')

    try:
        return base64.b64decode(url)
    except (ValueError, TypeError) as e:
        raise ValueError('Invalid base64 data provided in media URL') from e


def extract_config_dict(request: GenerateRequest) -> dict[str, Any]:
    """Extract the config from a GenerateRequest as a mutable dictionary.

    Handles both dict configs and Pydantic model configs uniformly.

    Args:
        request: The generate request.

    Returns:
        A mutable copy of the config as a dictionary, or an empty dict.
    """
    if not request.config:
        return {}
    if isinstance(request.config, dict):
        return request.config.copy()
    if hasattr(request.config, 'model_dump'):
        return request.config.model_dump(exclude_none=True)
    return {}


def _extract_media(request: GenerateRequest) -> tuple[str, str]:
    """Extract media content from the first message.

    Finds the first part with a MediaPart root and returns its URL and
    content type. If the content type is missing, attempts to parse it
    from a data URI.

    Args:
        request: The generate request.

    Returns:
        A tuple of (media_url, content_type).

    Raises:
        ValueError: If no media content is found.
    """
    if not request.messages:
        raise ValueError('No messages found in the request')

    part_with_media = next(
        (p for p in request.messages[0].content if isinstance(p.root, MediaPart) and p.root.media),
        None,
    )

    if not part_with_media:
        raise ValueError('No media content found in the first message')

    # Re-assert to help type checkers narrow through the generator boundary.
    assert isinstance(part_with_media.root, MediaPart)
    media = part_with_media.root.media
    content_type = media.content_type or ''
    url = media.url
    if not content_type and url.startswith('data:'):
        content_type = parse_data_uri_content_type(url)
    return url, content_type


class DictMessageAdapter:
    """Adapter for dictionary-based chat message objects with OpenAI-compatible fields."""

    def __init__(self, data: dict) -> None:
        """Initializes the adapter with a dictionary.

        Args:
            data: Dictionary with keys like 'content', 'tool_calls', and 'role'.
        """
        self._data = data

    @property
    def content(self) -> str | None:
        """Returns the 'content' of the message if available.

        Returns:
            The message content or None.
        """
        return self._data.get('content', None)

    @property
    def tool_calls(self) -> list | None:
        """Returns the 'tool_calls' list if present in the message.

        Returns:
            A list of tool calls or None.
        """
        return self._data.get('tool_calls', None)

    @property
    def role(self) -> str | None:
        """Returns the role of the message.

        Returns:
            The role string or None.
        """
        return self._data.get('role', None)

    @property
    def reasoning_content(self) -> str | None:
        """Returns the 'reasoning_content' if present in the message.

        Returns:
            The reasoning content string or None.
        """
        return self._data.get('reasoning_content', None)


class MessageAdapter:
    """Adapter for object-based chat message objects with OpenAI-compatible fields."""

    def __init__(self, data: object) -> None:
        """Initializes the adapter with an object.

        Args:
            data: An object expected to have attributes 'content', 'tool_calls', and 'role'.
        """
        self._data = data

    @property
    def content(self) -> str | None:
        """Returns the 'content' attribute of the message if available.

        Returns:
            The message content or None.
        """
        return getattr(self._data, 'content', None)

    @property
    def tool_calls(self) -> list | None:
        """Returns the 'tool_calls' attribute of the message if available.

        Returns:
            A list of tool calls or None.
        """
        return getattr(self._data, 'tool_calls', None)

    @property
    def role(self) -> str | None:
        """Returns the 'role' attribute of the message if available.

        Returns:
            The role string or None.
        """
        return getattr(self._data, 'role', None)

    @property
    def reasoning_content(self) -> str | None:
        """Returns the 'reasoning_content' attribute if available.

        DeepSeek R1/reasoner models return chain-of-thought reasoning
        in this separate field alongside the regular content.

        Note: Pydantic models (like openai's ChatCompletionMessage) raise
        AttributeError in __getattr__ for unknown fields, so getattr()
        with a default doesn't work. We must catch the exception.

        Returns:
            The reasoning content string or None.
        """
        try:
            return self._data.reasoning_content  # type: ignore[union-attr]
        except AttributeError:
            return None


ChatCompletionMessageAdapter = DictMessageAdapter | MessageAdapter


class MessageConverter:
    """Converts between internal `Message` objects and OpenAI-compatible chat message dicts."""

    _openai_role_map: dict[Role, str] = {Role.MODEL: 'assistant'}
    _genkit_role_map: dict[str, Role] = {'assistant': Role.MODEL}

    @classmethod
    def _get_openai_role(cls, role: Role | str) -> str:
        """Convert a Role to its OpenAI string representation."""
        if isinstance(role, Role):
            return cls._openai_role_map.get(role, role.value)

        if role == 'model':
            return 'assistant'
        return str(role)

    @classmethod
    def to_openai(cls, message: Message) -> list[dict]:
        """Converts an internal `Message` object to OpenAI-compatible chat messages.

        Handles TextPart, MediaPart (images), ToolRequestPart, and
        ToolResponsePart. When a message contains MediaPart content, the
        ``content`` field uses the array-of-content-blocks format required
        by the OpenAI Chat Completions API for multimodal requests.

        Matches the JS canonical implementation in ``toOpenAIMessages()``.

        Args:
            message: The internal `Message` instance.

        Returns:
            A list of OpenAI-compatible message dictionaries.
        """
        content_parts: list[dict[str, Any]] = []
        tool_calls = []
        tool_messages = []
        has_media = False

        for part in message.content:
            root = part.root

            # Skip ReasoningPart — reasoning_content must not be sent back
            # in multi-turn context. DeepSeek's API rejects it, and the JS
            # canonical implementation naturally excludes it by using msg.text
            # (which only returns text parts) for assistant messages.
            if isinstance(root, ReasoningPart):
                continue

            if isinstance(root, TextPart):
                content_parts.append({'type': 'text', 'text': root.text})

            elif isinstance(root, MediaPart):
                has_media = True
                content_parts.append({
                    'type': 'image_url',
                    'image_url': {'url': root.media.url},
                })

            elif isinstance(root, ToolRequestPart):
                tool_calls.append({
                    'id': root.tool_request.ref,
                    'type': 'function',
                    'function': {
                        'name': root.tool_request.name,
                        'arguments': json.dumps(root.tool_request.input),
                    },
                })

            elif isinstance(root, ToolResponsePart):
                tool_call = root.tool_response
                tool_messages.append({
                    'role': cls._get_openai_role(message.role),
                    'tool_call_id': tool_call.ref,
                    'content': str(tool_call.output),
                })

        result: list[dict[str, Any]] = []

        if content_parts:
            role = cls._get_openai_role(message.role)
            if has_media:
                # Multimodal: content is an array of typed content blocks.
                result.append({'role': role, 'content': content_parts})
            else:
                # Text-only: content is a plain string (matching JS behavior
                # where text-only messages use string content for
                # compatibility with older model endpoints).
                result.append({
                    'role': role,
                    'content': ''.join(p['text'] for p in content_parts),
                })

        if tool_calls:
            result.append({
                'role': cls._get_openai_role(message.role),
                'tool_calls': tool_calls,
            })

        result.extend(tool_messages)
        return result

    @classmethod
    def to_genkit(cls, message: ChatCompletionMessageAdapter) -> Message:
        """Converts an OpenAI-style message into a Genkit `Message` object.

        Handles tool calls, reasoning content (from DeepSeek R1 / reasoner),
        and regular text content. Matches the JS canonical implementation
        in fromOpenAIChoice().

        Args:
            message: A ChatCompletionMessageAdapter instance.

        Returns:
            A Genkit `Message` object.

        Raises:
            ValueError: If neither content, tool_calls, nor reasoning_content
                are present in the message.
        """
        content: list[Part] = []

        if message.tool_calls:
            content = [cls.tool_call_to_genkit(tool_call, args_parser=json.loads) for tool_call in message.tool_calls]
        else:
            # Reasoning content comes before regular content (matching JS order).
            reasoning = message.reasoning_content
            if reasoning:
                content.append(Part(root=ReasoningPart(reasoning=reasoning)))

            if message.content:
                content.append(cls.text_part_to_genkit(message.content))

        if not content:
            raise ValueError('Unable to determine content part')

        role = message.role or Role.MODEL
        return Message(role=cls._genkit_role_map.get(role, role), content=content)

    @classmethod
    def text_part_to_genkit(cls, content: str) -> Part:
        """Converts plain text to a Genkit `Part`.

        Args:
            content: The text content.

        Returns:
            A `Part` instance containing the text.
        """
        return Part(root=TextPart(text=content))

    @classmethod
    def tool_call_to_genkit(
        cls, tool_call: object, args_segment: str | None = None, args_parser: Callable[[str], dict] | None = None
    ) -> Part:
        """Converts a tool call into a Genkit `Part`.

        Args:
            tool_call: The tool call object containing function info.
            args_segment: Optional pre-parsed arguments string.
            args_parser: Optional parser to deserialize arguments.

        Returns:
            A `Part` instance containing a `ToolRequest`.
        """
        # Get function info from tool_call (could be dict or object)
        if hasattr(tool_call, 'function') and hasattr(tool_call, 'id'):
            func = tool_call.function
            tool_id = tool_call.id
            func_name = func.name if hasattr(func, 'name') else ''
            func_args = func.arguments if hasattr(func, 'arguments') else ''
        else:
            # Assume dict-like access
            func = tool_call.get('function', {})  # type: ignore[attr-defined]
            tool_id = tool_call.get('id', '')  # type: ignore[attr-defined]
            func_name = func.get('name', '')
            func_args = func.get('arguments', '')

        # args can be str from streaming or parsed dict from args_parser
        default_args = str(func_args) if func_args else ''
        args_input: str | dict[str, Any] | None = args_segment if args_segment is not None else default_args
        if args_parser and isinstance(args_input, str):
            args_input = args_parser(args_input)

        return Part(
            root=ToolRequestPart(
                tool_request=ToolRequest(
                    ref=str(tool_id) if tool_id else None,
                    name=str(func_name) if func_name else '',
                    input=args_input,
                )
            )
        )
