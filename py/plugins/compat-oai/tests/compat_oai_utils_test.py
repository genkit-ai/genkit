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

"""Exhaustive tests for models/utils.py utility functions."""

import base64

import pytest
from pydantic import BaseModel

from genkit.plugins.compat_oai.models.utils import (
    DictMessageAdapter,
    MessageAdapter,
    MessageConverter,
    _extract_media,
    _extract_text,
    _find_text,
    decode_data_uri_bytes,
    extract_config_dict,
    parse_data_uri_content_type,
)
from genkit.types import (
    GenerateRequest,
    Media,
    MediaPart,
    Message,
    Part,
    ReasoningPart,
    Role,
    TextPart,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)


class TestParseDataUriContentType:
    """Tests for parse_data_uri_content_type."""

    def test_audio_mpeg_with_base64(self) -> None:
        """Parse content type from a standard audio data URI."""
        url = 'data:audio/mpeg;base64,AAAA'
        assert parse_data_uri_content_type(url) == 'audio/mpeg'

    def test_text_plain_without_base64(self) -> None:
        """Parse content type from a data URI without ;base64 qualifier."""
        url = 'data:text/plain,hello world'
        assert parse_data_uri_content_type(url) == 'text/plain'

    def test_image_png_with_base64(self) -> None:
        """Parse content type from an image data URI."""
        url = 'data:image/png;base64,iVBOR...'
        assert parse_data_uri_content_type(url) == 'image/png'

    def test_empty_content_type_with_base64(self) -> None:
        """Return empty string when content type is missing from data URI."""
        url = 'data:;base64,AAAA'
        assert parse_data_uri_content_type(url) == ''

    def test_no_data_prefix(self) -> None:
        """Return empty string for non-data-URI URLs."""
        assert parse_data_uri_content_type('https://example.com/file.mp3') == ''

    def test_raw_base64_string(self) -> None:
        """Return empty string for raw base64 strings."""
        assert parse_data_uri_content_type('AAAA') == ''

    def test_empty_string(self) -> None:
        """Return empty string for empty input."""
        assert parse_data_uri_content_type('') == ''

    def test_data_prefix_no_comma(self) -> None:
        """Return empty string for malformed data URI without comma."""
        assert parse_data_uri_content_type('data:audio/mpeg;base64') == ''

    def test_application_json(self) -> None:
        """Parse content type from a JSON data URI."""
        url = 'data:application/json;base64,eyJ0ZXN0IjogdHJ1ZX0='
        assert parse_data_uri_content_type(url) == 'application/json'

    def test_audio_wav_with_extra_params(self) -> None:
        """Parse content type from a data URI with extra parameters."""
        url = 'data:audio/wav;rate=44100;base64,AAAA'
        assert parse_data_uri_content_type(url) == 'audio/wav'

    def test_content_type_with_charset(self) -> None:
        """Parse content type from a data URI with charset parameter."""
        url = 'data:text/html;charset=utf-8,<h1>hi</h1>'
        assert parse_data_uri_content_type(url) == 'text/html'


class TestDecodeDataUriBytes:
    """Tests for decode_data_uri_bytes."""

    def test_valid_data_uri(self) -> None:
        """Decode bytes from a valid base64 data URI."""
        payload = b'hello world'
        b64 = base64.b64encode(payload).decode('ascii')
        url = f'data:audio/mpeg;base64,{b64}'
        assert decode_data_uri_bytes(url) == payload

    def test_raw_base64(self) -> None:
        """Decode bytes from a raw base64 string without data: prefix."""
        payload = b'test data'
        b64 = base64.b64encode(payload).decode('ascii')
        assert decode_data_uri_bytes(b64) == payload

    def test_remote_http_url_raises(self) -> None:
        """Raise ValueError for http:// URLs."""
        with pytest.raises(ValueError, match='Remote URLs are not supported'):
            decode_data_uri_bytes('http://example.com/audio.mp3')

    def test_remote_https_url_raises(self) -> None:
        """Raise ValueError for https:// URLs."""
        with pytest.raises(ValueError, match='Remote URLs are not supported'):
            decode_data_uri_bytes('https://example.com/audio.mp3')

    def test_invalid_data_uri_format_raises(self) -> None:
        """Raise ValueError for data URI with invalid base64 payload."""
        with pytest.raises(ValueError, match='Invalid data URI format'):
            decode_data_uri_bytes('data:audio/mpeg;base64,NOT_VALID_B64!!!')

    def test_invalid_raw_base64_raises(self) -> None:
        """Raise ValueError for invalid raw base64 strings."""
        with pytest.raises(ValueError, match='Invalid base64 data'):
            decode_data_uri_bytes('NOT_VALID_B64!!!')

    def test_empty_payload_data_uri(self) -> None:
        """Decode empty bytes from a data URI with empty payload."""
        url = 'data:audio/mpeg;base64,'
        assert decode_data_uri_bytes(url) == b''

    def test_data_uri_without_base64_qualifier(self) -> None:
        """Decode bytes from a data URI that omits ;base64 qualifier."""
        payload = b'test'
        b64 = base64.b64encode(payload).decode('ascii')
        url = f'data:text/plain,{b64}'
        assert decode_data_uri_bytes(url) == payload

    def test_data_uri_with_no_content_type(self) -> None:
        """Decode bytes from a data URI with empty content type."""
        payload = b'data'
        b64 = base64.b64encode(payload).decode('ascii')
        url = f'data:;base64,{b64}'
        assert decode_data_uri_bytes(url) == payload


class TestExtractConfigDict:
    """Tests for extract_config_dict."""

    def _make_request(self, config: object = None) -> GenerateRequest:
        """Create a minimal GenerateRequest with the given config."""
        return GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='hello'))],
                )
            ],
            config=config,
        )

    def test_no_config_returns_empty_dict(self) -> None:
        """Return empty dict when request has no config."""
        request = self._make_request(config=None)
        assert extract_config_dict(request) == {}

    def test_dict_config_returns_copy(self) -> None:
        """Return a copy of the config when it is a dict."""
        original = {'temperature': 0.5, 'model': 'gpt-4'}
        request = self._make_request(config=original)
        result = extract_config_dict(request)
        assert result == original
        assert result is not original

    def test_dict_config_mutation_does_not_affect_original(self) -> None:
        """Verify that mutating the returned dict does not affect the original."""
        original = {'temperature': 0.5}
        request = self._make_request(config=original)
        result = extract_config_dict(request)
        result['temperature'] = 1.0
        assert original['temperature'] == 0.5

    def test_empty_dict_config(self) -> None:
        """Return empty dict when config is an empty dict."""
        request = self._make_request(config={})
        assert extract_config_dict(request) == {}


class TestFindText:
    """Tests for _find_text."""

    def test_returns_text_from_first_message(self) -> None:
        """Find and return text from the first message's text part."""
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='hello'))],
                )
            ]
        )
        assert _find_text(request) == 'hello'

    def test_returns_none_for_no_messages(self) -> None:
        """Return None when there are no messages."""
        request = GenerateRequest(messages=[])
        assert _find_text(request) is None

    def test_returns_none_for_no_text_parts(self) -> None:
        """Return None when message has only media parts."""
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=MediaPart(media=Media(url='data:audio/mpeg;base64,AAAA')))],
                )
            ]
        )
        assert _find_text(request) is None

    def test_returns_first_text_part(self) -> None:
        """Return the first text part when multiple exist."""
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        Part(root=TextPart(text='first')),
                        Part(root=TextPart(text='second')),
                    ],
                )
            ]
        )
        assert _find_text(request) == 'first'


class TestExtractText:
    """Tests for _extract_text."""

    def test_returns_text_when_present(self) -> None:
        """Extract and return text when present."""
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='hello'))],
                )
            ]
        )
        assert _extract_text(request) == 'hello'

    def test_raises_for_no_messages(self) -> None:
        """Raise ValueError when request has no messages."""
        request = GenerateRequest(messages=[])
        with pytest.raises(ValueError, match='No messages found'):
            _extract_text(request)

    def test_raises_for_no_text_content(self) -> None:
        """Raise ValueError when no text parts exist."""
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=MediaPart(media=Media(url='data:audio/mpeg;base64,AAAA')))],
                )
            ]
        )
        with pytest.raises(ValueError, match='No text content found'):
            _extract_text(request)


class TestExtractMedia:
    """Tests for _extract_media."""

    def test_extracts_media_url_and_content_type(self) -> None:
        """Extract URL and content type from a media part."""
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        Part(
                            root=MediaPart(
                                media=Media(
                                    url='data:audio/mpeg;base64,AAAA',
                                    content_type='audio/mpeg',
                                )
                            )
                        )
                    ],
                )
            ]
        )
        url, ct = _extract_media(request)
        assert url == 'data:audio/mpeg;base64,AAAA'
        assert ct == 'audio/mpeg'

    def test_parses_content_type_from_data_uri_when_missing(self) -> None:
        """Parse content type from data URI when not explicitly provided."""
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        Part(
                            root=MediaPart(
                                media=Media(
                                    url='data:audio/wav;base64,AAAA',
                                )
                            )
                        )
                    ],
                )
            ]
        )
        url, ct = _extract_media(request)
        assert url == 'data:audio/wav;base64,AAAA'
        assert ct == 'audio/wav'

    def test_raises_for_no_messages(self) -> None:
        """Raise ValueError when request has no messages."""
        request = GenerateRequest(messages=[])
        with pytest.raises(ValueError, match='No messages found'):
            _extract_media(request)

    def test_raises_for_no_media_parts(self) -> None:
        """Raise ValueError when message has no media parts."""
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='just text'))],
                )
            ]
        )
        with pytest.raises(ValueError, match='No media content found'):
            _extract_media(request)

    def test_skips_text_parts_finds_media(self) -> None:
        """Find media part even when text parts come first."""
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        Part(root=TextPart(text='instructions')),
                        Part(
                            root=MediaPart(
                                media=Media(
                                    url='data:image/png;base64,iVBOR',
                                    content_type='image/png',
                                )
                            )
                        ),
                    ],
                )
            ]
        )
        _, ct = _extract_media(request)
        assert ct == 'image/png'

    def test_content_type_from_data_uri_without_base64_qualifier(self) -> None:
        """Parse content type from data URI that omits ;base64 qualifier."""
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=MediaPart(media=Media(url='data:text/plain,hello')))],
                )
            ]
        )
        _, ct = _extract_media(request)
        assert ct == 'text/plain'


class TestDictMessageAdapterReasoningContent:
    """Tests for DictMessageAdapter.reasoning_content property."""

    def test_returns_reasoning_content_when_present(self) -> None:
        """Return reasoning_content from the dict."""
        adapter = DictMessageAdapter({
            'content': 'The answer is 42.',
            'reasoning_content': 'Let me think step by step...',
            'role': 'assistant',
        })
        assert adapter.reasoning_content == 'Let me think step by step...'

    def test_returns_none_when_missing(self) -> None:
        """Return None when reasoning_content is not in the dict."""
        adapter = DictMessageAdapter({
            'content': 'Hello',
            'role': 'assistant',
        })
        assert adapter.reasoning_content is None


class TestMessageAdapterReasoningContent:
    """Tests for MessageAdapter.reasoning_content property."""

    def test_returns_reasoning_content_when_present(self) -> None:
        """Return reasoning_content from the object."""

        class FakeMessage:
            content = 'The answer is 42.'
            reasoning_content = 'Let me think step by step...'
            tool_calls = None
            role = 'assistant'

        adapter = MessageAdapter(FakeMessage())
        assert adapter.reasoning_content == 'Let me think step by step...'

    def test_returns_none_when_missing(self) -> None:
        """Return None when object has no reasoning_content attribute."""

        class FakeMessage:
            content = 'Hello'
            tool_calls = None
            role = 'assistant'

        adapter = MessageAdapter(FakeMessage())
        assert adapter.reasoning_content is None

    def test_returns_none_for_pydantic_model_without_field(self) -> None:
        """Return None for Pydantic models that raise AttributeError on unknown attrs.

        The openai library's ChatCompletionMessage is a Pydantic model whose
        __getattr__ raises AttributeError for unknown fields, bypassing
        Python's getattr(obj, name, default) fallback. This test verifies
        the try/except pattern handles this correctly.
        """

        class PydanticMessage(BaseModel):
            content: str | None = None
            tool_calls: list | None = None
            role: str = 'assistant'

        adapter = MessageAdapter(PydanticMessage(content='Hello'))
        assert adapter.reasoning_content is None


class TestMessageConverterReasoningContent:
    """Tests for reasoning_content handling in MessageConverter.to_genkit()."""

    def test_reasoning_content_only(self) -> None:
        """Convert message with only reasoning_content to ReasoningPart."""
        adapter = DictMessageAdapter({
            'content': None,
            'reasoning_content': 'Let me think about this step by step...',
            'role': 'assistant',
        })
        msg = MessageConverter.to_genkit(adapter)
        assert len(msg.content) == 1
        assert isinstance(msg.content[0].root, ReasoningPart)
        assert msg.content[0].root.reasoning == 'Let me think about this step by step...'

    def test_reasoning_and_text_content(self) -> None:
        """Convert message with both reasoning_content and content."""
        adapter = DictMessageAdapter({
            'content': 'The answer is 42.',
            'reasoning_content': 'Let me think...',
            'role': 'assistant',
        })
        msg = MessageConverter.to_genkit(adapter)
        # Reasoning comes first, then text (matching JS order).
        assert len(msg.content) == 2
        assert isinstance(msg.content[0].root, ReasoningPart)
        assert msg.content[0].root.reasoning == 'Let me think...'
        assert isinstance(msg.content[1].root, TextPart)
        assert msg.content[1].root.text == 'The answer is 42.'

    def test_text_content_without_reasoning(self) -> None:
        """Convert a regular message without reasoning_content."""
        adapter = DictMessageAdapter({
            'content': 'Hello!',
            'role': 'assistant',
        })
        msg = MessageConverter.to_genkit(adapter)
        assert len(msg.content) == 1
        assert isinstance(msg.content[0].root, TextPart)
        assert msg.content[0].root.text == 'Hello!'

    def test_empty_reasoning_content_is_ignored(self) -> None:
        """Ignore reasoning_content when it is an empty string."""
        adapter = DictMessageAdapter({
            'content': 'Hello!',
            'reasoning_content': '',
            'role': 'assistant',
        })
        msg = MessageConverter.to_genkit(adapter)
        # Empty reasoning is falsy, so only text part is created.
        assert len(msg.content) == 1
        assert isinstance(msg.content[0].root, TextPart)

    def test_raises_when_no_content_at_all(self) -> None:
        """Raise ValueError when all content fields are None/empty."""
        adapter = DictMessageAdapter({
            'content': None,
            'role': 'assistant',
        })
        with pytest.raises(ValueError, match='Unable to determine content part'):
            MessageConverter.to_genkit(adapter)

    def test_tool_calls_take_precedence_over_reasoning(self) -> None:
        """Tool calls take precedence; reasoning_content is ignored."""
        adapter = DictMessageAdapter({
            'content': None,
            'reasoning_content': 'Some reasoning',
            'tool_calls': [
                {
                    'id': 'call_1',
                    'function': {
                        'name': 'get_weather',
                        'arguments': '{"location": "NYC"}',
                    },
                }
            ],
            'role': 'assistant',
        })
        msg = MessageConverter.to_genkit(adapter)
        # Should produce tool request parts, not reasoning.
        assert len(msg.content) == 1

        assert isinstance(msg.content[0].root, ToolRequestPart)

    def test_role_defaults_to_model(self) -> None:
        """Default role should be MODEL when not provided."""
        adapter = DictMessageAdapter({
            'content': None,
            'reasoning_content': 'Thinking...',
        })
        msg = MessageConverter.to_genkit(adapter)
        assert msg.role == Role.MODEL


class TestMessageConverterToOpenAI:
    """Tests for MessageConverter.to_openai()."""

    def test_text_only_message_uses_string_content(self) -> None:
        """Text-only messages should produce a plain string content field."""
        message = Message(
            role=Role.USER,
            content=[Part(root=TextPart(text='Hello world'))],
        )
        result = MessageConverter.to_openai(message)
        assert len(result) == 1
        assert result[0] == {'role': 'user', 'content': 'Hello world'}

    def test_multiple_text_parts_concatenated(self) -> None:
        """Multiple text parts should be concatenated into one string."""
        message = Message(
            role=Role.USER,
            content=[
                Part(root=TextPart(text='Hello ')),
                Part(root=TextPart(text='world')),
            ],
        )
        result = MessageConverter.to_openai(message)
        assert len(result) == 1
        assert result[0]['content'] == 'Hello world'

    def test_media_part_produces_image_url_block(self) -> None:
        """A MediaPart should produce an image_url content block."""
        message = Message(
            role=Role.USER,
            content=[
                Part(root=MediaPart(media=Media(url='https://example.com/cat.jpg', content_type='image/jpeg'))),
            ],
        )
        result = MessageConverter.to_openai(message)
        assert len(result) == 1
        assert result[0]['role'] == 'user'
        content = result[0]['content']
        assert isinstance(content, list)
        assert len(content) == 1
        assert content[0] == {
            'type': 'image_url',
            'image_url': {'url': 'https://example.com/cat.jpg'},
        }

    def test_text_and_media_produces_content_array(self) -> None:
        """Mixed text + media should produce an array of content blocks.

        This is the multimodal vision format required by the OpenAI Chat
        Completions API, matching the JS canonical toOpenAIMessages().
        """
        message = Message(
            role=Role.USER,
            content=[
                Part(root=TextPart(text='Describe this image')),
                Part(root=MediaPart(media=Media(url='https://example.com/cat.jpg', content_type='image/jpeg'))),
            ],
        )
        result = MessageConverter.to_openai(message)
        assert len(result) == 1
        content = result[0]['content']
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0] == {'type': 'text', 'text': 'Describe this image'}
        assert content[1] == {
            'type': 'image_url',
            'image_url': {'url': 'https://example.com/cat.jpg'},
        }

    def test_tool_request_parts(self) -> None:
        """ToolRequestParts should produce tool_calls entries."""
        message = Message(
            role=Role.MODEL,
            content=[
                Part(
                    root=ToolRequestPart(
                        tool_request=ToolRequest(
                            ref='call_1',
                            name='get_weather',
                            input={'location': 'NYC'},
                        )
                    )
                )
            ],
        )
        result = MessageConverter.to_openai(message)
        assert len(result) == 1
        assert result[0]['role'] == 'assistant'
        assert 'tool_calls' in result[0]
        tc = result[0]['tool_calls'][0]
        assert tc['id'] == 'call_1'
        assert tc['function']['name'] == 'get_weather'

    def test_tool_response_parts(self) -> None:
        """ToolResponseParts should produce tool role messages."""
        message = Message(
            role=Role.TOOL,
            content=[
                Part(
                    root=ToolResponsePart(
                        tool_response=ToolResponse(
                            ref='call_1',
                            name='get_weather',
                            output='Sunny, 72F',
                        )
                    )
                )
            ],
        )
        result = MessageConverter.to_openai(message)
        assert len(result) == 1
        assert result[0]['role'] == 'tool'
        assert result[0]['tool_call_id'] == 'call_1'
        assert result[0]['content'] == 'Sunny, 72F'

    def test_model_role_maps_to_assistant(self) -> None:
        """Role.MODEL should map to 'assistant' in OpenAI format."""
        message = Message(
            role=Role.MODEL,
            content=[Part(root=TextPart(text='Hi there'))],
        )
        result = MessageConverter.to_openai(message)
        assert result[0]['role'] == 'assistant'

    def test_data_uri_media_url_preserved(self) -> None:
        """Data URI media URLs should be passed through unchanged."""
        data_uri = 'data:image/png;base64,iVBORw0KGgo='
        message = Message(
            role=Role.USER,
            content=[
                Part(root=TextPart(text='What is this?')),
                Part(root=MediaPart(media=Media(url=data_uri))),
            ],
        )
        result = MessageConverter.to_openai(message)
        content = result[0]['content']
        assert isinstance(content, list)
        assert content[1]['image_url']['url'] == data_uri

    def test_reasoning_part_stripped_from_assistant_message(self) -> None:
        """ReasoningPart should be stripped when converting back to OpenAI format.

        DeepSeek's API rejects reasoning_content in context messages. The JS
        canonical implementation naturally excludes it by using msg.text (which
        only returns text parts) for assistant messages. We must explicitly skip
        ReasoningPart instances.
        """
        message = Message(
            role=Role.MODEL,
            content=[
                Part(root=ReasoningPart(reasoning='Let me think step by step...')),
                Part(root=TextPart(text='The answer is 42.')),
            ],
        )
        result = MessageConverter.to_openai(message)
        assert len(result) == 1
        assert result[0] == {'role': 'assistant', 'content': 'The answer is 42.'}

    def test_reasoning_only_message_produces_empty_result(self) -> None:
        """A message with only ReasoningPart should produce an empty result.

        This can happen when a DeepSeek R1 model returns only reasoning_content
        without any text content. The reasoning must not be sent back.
        """
        message = Message(
            role=Role.MODEL,
            content=[
                Part(root=ReasoningPart(reasoning='Let me think about this...')),
            ],
        )
        result = MessageConverter.to_openai(message)
        assert result == []

    def test_multi_turn_with_reasoning_strips_all_reasoning(self) -> None:
        """In a multi-turn conversation, all ReasoningParts should be stripped.

        Simulates a multi-turn context where a previous assistant message
        contained both reasoning and text content.
        """
        # Previous assistant message with reasoning + text
        assistant_msg = Message(
            role=Role.MODEL,
            content=[
                Part(root=ReasoningPart(reasoning='Step 1: analyze the question...')),
                Part(root=ReasoningPart(reasoning='Step 2: formulate answer...')),
                Part(root=TextPart(text='Paris is the capital of France.')),
            ],
        )
        result = MessageConverter.to_openai(assistant_msg)
        assert len(result) == 1
        assert result[0] == {'role': 'assistant', 'content': 'Paris is the capital of France.'}

    def test_empty_message_produces_no_result(self) -> None:
        """A message with no content parts should produce an empty result."""
        message = Message(role=Role.USER, content=[])
        result = MessageConverter.to_openai(message)
        assert result == []
