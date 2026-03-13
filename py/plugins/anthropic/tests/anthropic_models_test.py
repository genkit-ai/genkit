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

"""Tests for Anthropic models."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from genkit.plugins.anthropic.models import AnthropicModel
from genkit.plugins.anthropic.utils import maybe_strip_fences, strip_markdown_fences
from genkit.types import (
    GenerateRequest,
    GenerateResponseChunk,
    GenerationCommonConfig,
    Media,
    MediaPart,
    Message,
    Metadata,
    OutputConfig,
    Part,
    Role,
    TextPart,
    ToolDefinition,
    ToolRequestPart,
)


def _create_sample_request() -> GenerateRequest:
    """Create a sample generation request for testing."""
    return GenerateRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text='Hello, how are you?'))],
            )
        ],
        config=GenerationCommonConfig(),
        tools=[
            ToolDefinition(
                name='get_weather',
                description='Get weather for a location',
                input_schema={
                    'type': 'object',
                    'properties': {'location': {'type': 'string', 'description': 'Location name'}},
                    'required': ['location'],
                },
            )
        ],
    )


@pytest.mark.asyncio
async def test_generate_basic() -> None:
    """Test basic generation."""
    sample_request = _create_sample_request()

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type='text', text="Hello! I'm doing well.")]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=15)
    mock_response.stop_reason = 'end_turn'

    mock_client.messages.create = AsyncMock(return_value=mock_response)

    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)
    response = await model.generate(sample_request)

    assert response.message is not None
    assert response.message.content is not None
    assert len(response.message.content) == 1
    part = response.message.content[0]
    actual_part = part.root if isinstance(part, Part) else part
    assert isinstance(actual_part, TextPart)
    assert actual_part.text == "Hello! I'm doing well."
    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 15
    assert response.finish_reason == 'stop'


@pytest.mark.asyncio
async def test_generate_with_tools() -> None:
    """Test generation with tool calls."""
    sample_request = _create_sample_request()

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.type = 'tool_use'
    mock_block.id = 'tool_123'
    mock_block.name = 'get_weather'
    mock_block.input = {'location': 'Paris'}
    mock_response.content = [mock_block]
    mock_response.usage = MagicMock(input_tokens=20, output_tokens=10)
    mock_response.stop_reason = 'tool_use'

    mock_client.messages.create = AsyncMock(return_value=mock_response)

    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)
    response = await model.generate(sample_request)

    assert response.message is not None
    assert response.message.content is not None
    assert len(response.message.content) == 1
    part = response.message.content[0]
    actual_part = part.root if isinstance(part, Part) else part
    assert isinstance(actual_part, ToolRequestPart)
    assert actual_part.tool_request is not None
    assert actual_part.tool_request.name == 'get_weather'
    assert actual_part.tool_request.ref == 'tool_123'
    assert actual_part.tool_request.input == {'location': 'Paris'}


@pytest.mark.asyncio
async def test_generate_with_config() -> None:
    """Test generation with custom config."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type='text', text='Response')]
    mock_response.usage = MagicMock(input_tokens=5, output_tokens=5)
    mock_response.stop_reason = 'end_turn'

    mock_client.messages.create = AsyncMock(return_value=mock_response)

    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    request = GenerateRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Test'))])],
        config=GenerationCommonConfig(
            temperature=0.7,
            max_output_tokens=100,
            top_p=0.9,
        ),
    )

    await model.generate(request)

    call_args = mock_client.messages.create.call_args
    assert call_args.kwargs['temperature'] == 0.7
    assert call_args.kwargs['max_tokens'] == 100
    assert call_args.kwargs['top_p'] == 0.9


def test_extract_system() -> None:
    """Test system prompt extraction."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    messages = [
        Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='You are helpful.'))]),
        Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))]),
    ]

    system = model._extract_system(messages)
    assert system == 'You are helpful.'


def test_to_anthropic_messages() -> None:
    """Test message conversion."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    messages = [
        Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))]),
        Message(role=Role.MODEL, content=[Part(root=TextPart(text='Hi there'))]),
    ]

    anthropic_messages = model._to_anthropic_messages(messages)

    assert len(anthropic_messages) == 2
    assert anthropic_messages[0]['role'] == 'user'
    assert anthropic_messages[0]['content'][0]['text'] == 'Hello'
    assert anthropic_messages[1]['role'] == 'assistant'
    assert anthropic_messages[1]['content'][0]['text'] == 'Hi there'


class MockStreamManager:
    """Mock stream manager for testing Anthropic streaming."""

    def __init__(self, chunks: list[Any], final_content: list[Any] | None = None) -> None:
        """Initialize the MockStreamManager."""
        self.chunks = chunks
        self.final_message = MagicMock()
        self.final_message.content = final_content if final_content else []
        self.final_message.usage = MagicMock(input_tokens=10, output_tokens=20)
        self.final_message.stop_reason = 'end_turn'

    async def __aenter__(self) -> 'MockStreamManager':
        """Enter the async context manager."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit the async context manager."""
        pass

    def __aiter__(self) -> 'MockStreamManager':
        """Return the async iterator."""
        return self

    async def __anext__(self) -> object:
        """Return the next chunk from the stream."""
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)

    async def get_final_message(self) -> object:
        """Get the final message from the stream."""
        return self.final_message


@pytest.mark.asyncio
async def test_streaming_generation() -> None:
    """Test streaming generation."""
    sample_request = _create_sample_request()

    mock_client = MagicMock()

    chunks = [
        MagicMock(type='content_block_delta', delta=MagicMock(type='text_delta', text='Hello')),
        MagicMock(type='content_block_delta', delta=MagicMock(type='text_delta', text=' world')),
        MagicMock(type='content_block_delta', delta=MagicMock(type='text_delta', text='!')),
    ]

    final_content = [MagicMock(type='text', text='Hello world!')]
    mock_stream = MockStreamManager(chunks, final_content=final_content)
    mock_client.messages.stream.return_value = mock_stream

    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    ctx = MagicMock()
    ctx.is_streaming = True
    collected_chunks: list[GenerateResponseChunk] = []

    def send_chunk(chunk: GenerateResponseChunk) -> None:
        collected_chunks.append(chunk)

    ctx.send_chunk = send_chunk

    response = await model.generate(sample_request, ctx)

    assert len(collected_chunks) == 3
    chunk0_part = collected_chunks[0].content[0]
    chunk0_actual = chunk0_part.root if isinstance(chunk0_part, Part) else chunk0_part
    assert chunk0_actual.text == 'Hello'

    chunk1_part = collected_chunks[1].content[0]
    chunk1_actual = chunk1_part.root if isinstance(chunk1_part, Part) else chunk1_part
    assert chunk1_actual.text == ' world'

    chunk2_part = collected_chunks[2].content[0]
    chunk2_actual = chunk2_part.root if isinstance(chunk2_part, Part) else chunk2_part
    assert chunk2_actual.text == '!'

    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 20

    # Verify final response content is populated
    assert response.message is not None
    assert len(response.message.content) == 1
    final_part = response.message.content[0]
    assert isinstance(final_part, Part)
    assert isinstance(final_part.root, TextPart)
    assert final_part.root.text == 'Hello world!'


@pytest.mark.asyncio
async def test_streaming_tool_request() -> None:
    """Test streaming generation with tool use blocks."""
    sample_request = _create_sample_request()

    mock_client = MagicMock()

    # Simulate: text chunk, then tool_use block (start + json deltas + stop).
    tool_block = MagicMock(type='tool_use', id='tool_abc')
    tool_block.name = 'get_weather'
    chunks = [
        MagicMock(type='content_block_delta', delta=MagicMock(type='text_delta', text='Let me check.')),
        MagicMock(type='content_block_start', index=1, content_block=tool_block),
        MagicMock(
            type='content_block_delta',
            index=1,
            delta=MagicMock(type='input_json_delta', partial_json='{"location"'),
        ),
        MagicMock(
            type='content_block_delta',
            index=1,
            delta=MagicMock(type='input_json_delta', partial_json=': "Paris"}'),
        ),
        MagicMock(type='content_block_stop', index=1),
    ]

    final_tool = MagicMock(type='tool_use', id='tool_abc', input={'location': 'Paris'})
    final_tool.name = 'get_weather'
    final_content = [
        MagicMock(type='text', text='Let me check.'),
        final_tool,
    ]
    mock_stream = MockStreamManager(chunks, final_content=final_content)
    mock_client.messages.stream.return_value = mock_stream

    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    ctx = MagicMock()
    ctx.is_streaming = True
    collected_chunks: list[GenerateResponseChunk] = []
    ctx.send_chunk = lambda chunk: collected_chunks.append(chunk)

    response = await model.generate(sample_request, ctx)

    # Should have 2 chunks: one text, one tool request.
    assert len(collected_chunks) == 2

    text_part = collected_chunks[0].content[0].root
    assert isinstance(text_part, TextPart)
    assert text_part.text == 'Let me check.'

    tool_part = collected_chunks[1].content[0].root
    assert isinstance(tool_part, ToolRequestPart)
    assert tool_part.tool_request.name == 'get_weather'
    assert tool_part.tool_request.ref == 'tool_abc'
    assert tool_part.tool_request.input == {'location': 'Paris'}

    # Final response should also contain the tool request.
    assert response.message is not None
    assert len(response.message.content) == 2


class TestStripMarkdownFences:
    """Tests for strip_markdown_fences."""

    def test_strips_json_fences(self) -> None:
        """Strips ```json ... ``` fences."""
        text = '```json\n{"name": "John", "age": 30}\n```'
        assert strip_markdown_fences(text) == '{"name": "John", "age": 30}'

    def test_strips_plain_fences(self) -> None:
        """Strips ``` ... ``` fences without language tag."""
        text = '```\n{"name": "John"}\n```'
        assert strip_markdown_fences(text) == '{"name": "John"}'

    def test_strips_fences_with_surrounding_whitespace(self) -> None:
        """Strips fences even with leading/trailing whitespace."""
        text = '  \n```json\n{"a": 1}\n```\n  '
        assert strip_markdown_fences(text) == '{"a": 1}'

    def test_preserves_plain_json(self) -> None:
        """Does not alter valid JSON without fences."""
        text = '{"name": "John", "age": 30}'
        assert strip_markdown_fences(text) == text

    def test_preserves_non_json_text(self) -> None:
        """Does not alter plain text."""
        text = 'Hello, world!'
        assert strip_markdown_fences(text) == text

    def test_strips_multiline_json_in_fences(self) -> None:
        """Strips fences around multiline JSON."""
        text = '```json\n{\n  "name": "John",\n  "age": 30\n}\n```'
        result = strip_markdown_fences(text)
        assert result == '{\n  "name": "John",\n  "age": 30\n}'


class TestMaybeStripFences:
    """Tests for maybe_strip_fences."""

    def test_strips_fences_for_json_output(self) -> None:
        """Strips markdown fences when JSON output is requested."""
        request = GenerateRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
            output=OutputConfig(format='json', schema={'type': 'object'}),
        )
        parts = [Part(root=TextPart(text='```json\n{"a": 1}\n```'))]
        result = maybe_strip_fences(request, parts)
        assert result[0].root.text == '{"a": 1}'

    def test_no_op_for_text_output(self) -> None:
        """Does not modify responses when output format is not json."""
        request = GenerateRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
            output=OutputConfig(format='text'),
        )
        fenced = '```json\n{"a": 1}\n```'
        parts = [Part(root=TextPart(text=fenced))]
        result = maybe_strip_fences(request, parts)
        assert result[0].root.text == fenced

    def test_no_op_for_no_output(self) -> None:
        """Does not modify responses when no output config is set."""
        request = GenerateRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
        )
        fenced = '```json\n{"a": 1}\n```'
        parts = [Part(root=TextPart(text=fenced))]
        result = maybe_strip_fences(request, parts)
        assert result[0].root.text == fenced

    def test_no_op_when_no_fences(self) -> None:
        """Does not modify clean JSON responses."""
        request = GenerateRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
            output=OutputConfig(format='json', schema={'type': 'object'}),
        )
        text = '{"name": "John"}'
        parts = [Part(root=TextPart(text=text))]
        result = maybe_strip_fences(request, parts)
        assert result is parts


def test_cache_control_on_text_block() -> None:
    """Test that cache_control metadata is forwarded to Anthropic blocks."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    messages = [
        Message(
            role=Role.USER,
            content=[
                Part(root=TextPart(text='Cached context', metadata=Metadata({'cache_control': {'type': 'ephemeral'}}))),
                Part(root=TextPart(text='Question about the context')),
            ],
        ),
    ]

    anthropic_messages = model._to_anthropic_messages(messages)

    assert len(anthropic_messages) == 1
    blocks = anthropic_messages[0]['content']
    assert len(blocks) == 2

    # First block should have cache_control.
    assert blocks[0]['type'] == 'text'
    assert blocks[0]['text'] == 'Cached context'
    assert blocks[0]['cache_control'] == {'type': 'ephemeral'}

    # Second block should not have cache_control.
    assert blocks[1]['type'] == 'text'
    assert 'cache_control' not in blocks[1]


def test_cache_control_not_applied_without_metadata() -> None:
    """Test that no cache_control is applied when metadata is absent."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    messages = [
        Message(
            role=Role.USER,
            content=[Part(root=TextPart(text='No cache'))],
        ),
    ]

    anthropic_messages = model._to_anthropic_messages(messages)
    blocks = anthropic_messages[0]['content']
    assert 'cache_control' not in blocks[0]


@pytest.mark.asyncio
async def test_cache_token_tracking_in_usage() -> None:
    """Test that cache creation/read tokens are included in usage."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type='text', text='Cached response')]
    mock_response.usage = MagicMock(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=80,
        cache_read_input_tokens=20,
    )
    mock_response.stop_reason = 'end_turn'

    mock_client.messages.create = AsyncMock(return_value=mock_response)

    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)
    request = GenerateRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Test'))])],
    )

    response = await model.generate(request)

    assert response.usage is not None
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 50
    assert response.usage.custom is not None
    assert response.usage.custom['cache_creation_input_tokens'] == 80
    assert response.usage.custom['cache_read_input_tokens'] == 20


@pytest.mark.asyncio
async def test_no_cache_tokens_when_caching_not_used() -> None:
    """Test that custom is None when no cache tokens are present."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type='text', text='Response')]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    mock_response.stop_reason = 'end_turn'
    # Simulate no cache attributes.
    del mock_response.usage.cache_creation_input_tokens
    del mock_response.usage.cache_read_input_tokens

    mock_client.messages.create = AsyncMock(return_value=mock_response)

    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)
    request = GenerateRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Test'))])],
    )

    response = await model.generate(request)
    assert response.usage is not None
    assert response.usage.custom is None


def test_pdf_base64_becomes_document_block() -> None:
    """Test that a base64 PDF MediaPart converts to Anthropic document block."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    pdf_data = 'data:application/pdf;base64,JVBERi0xLjQ='
    messages = [
        Message(
            role=Role.USER,
            content=[
                Part(root=MediaPart(media=Media(url=pdf_data, content_type='application/pdf'))),
                Part(root=TextPart(text='Summarize this PDF')),
            ],
        ),
    ]

    anthropic_messages = model._to_anthropic_messages(messages)
    blocks = anthropic_messages[0]['content']

    assert blocks[0]['type'] == 'document'
    assert blocks[0]['source']['type'] == 'base64'
    assert blocks[0]['source']['media_type'] == 'application/pdf'
    assert blocks[0]['source']['data'] == 'JVBERi0xLjQ='

    assert blocks[1]['type'] == 'text'
    assert blocks[1]['text'] == 'Summarize this PDF'


def test_pdf_url_becomes_document_block() -> None:
    """Test that a URL-based PDF converts to Anthropic document block."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    messages = [
        Message(
            role=Role.USER,
            content=[
                Part(
                    root=MediaPart(
                        media=Media(
                            url='https://example.com/doc.pdf',
                            content_type='application/pdf',
                        )
                    )
                ),
            ],
        ),
    ]

    anthropic_messages = model._to_anthropic_messages(messages)
    blocks = anthropic_messages[0]['content']

    assert blocks[0]['type'] == 'document'
    assert blocks[0]['source']['type'] == 'url'
    assert blocks[0]['source']['url'] == 'https://example.com/doc.pdf'


def test_image_still_works() -> None:
    """Test that non-document images still produce image blocks."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    messages = [
        Message(
            role=Role.USER,
            content=[
                Part(root=MediaPart(media=Media(url='https://example.com/cat.jpg', content_type='image/jpeg'))),
            ],
        ),
    ]

    anthropic_messages = model._to_anthropic_messages(messages)
    blocks = anthropic_messages[0]['content']

    assert blocks[0]['type'] == 'image'
    assert blocks[0]['source']['type'] == 'url'


def test_pdf_with_cache_control() -> None:
    """Test that cache_control can be applied to document blocks."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    pdf_data = 'data:application/pdf;base64,JVBERi0xLjQ='
    messages = [
        Message(
            role=Role.USER,
            content=[
                Part(
                    root=MediaPart(
                        media=Media(url=pdf_data, content_type='application/pdf'),
                        metadata=Metadata({'cache_control': {'type': 'ephemeral'}}),
                    )
                ),
            ],
        ),
    ]

    anthropic_messages = model._to_anthropic_messages(messages)
    blocks = anthropic_messages[0]['content']

    assert blocks[0]['type'] == 'document'
    assert blocks[0]['cache_control'] == {'type': 'ephemeral'}


def test_structured_output_uses_native_output_config() -> None:
    """Test that JSON schema uses native output_config when model supports it."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-opus-4-6', client=mock_client)

    request = GenerateRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Generate a cat'))])],
        output=OutputConfig(
            format='json',
            schema={'type': 'object', 'properties': {'name': {'type': 'string'}}},
        ),
    )

    params = model._build_params(request)

    assert 'output_config' in params
    assert params['output_config']['format']['type'] == 'json_schema'
    assert params['output_config']['format']['schema']['additionalProperties'] is False


def test_structured_output_falls_back_to_system_prompt() -> None:
    """Test that JSON without schema falls back to system prompt instruction."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    request = GenerateRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Generate JSON'))])],
        output=OutputConfig(format='json'),
    )

    params = model._build_params(request)

    assert 'output_config' not in params
    assert 'system' in params
    assert 'Output valid JSON' in params['system']


def test_structured_output_falls_back_for_unsupported_models() -> None:
    """Test that JSON with schema falls back to system prompt for unsupported models."""
    mock_client = MagicMock()
    # Claude 3.5 Haiku is marked as not supporting JSON natively in model_info.py
    model = AnthropicModel(model_name='claude-3-5-haiku', client=mock_client)

    request = GenerateRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Generate a cat'))])],
        output=OutputConfig(
            format='json',
            schema={'type': 'object', 'properties': {'name': {'type': 'string'}}},
        ),
    )

    params = model._build_params(request)

    assert 'output_config' not in params
    assert 'system' in params
    assert 'Output valid JSON' in params['system']
    assert 'Follow this JSON schema' in params['system']
    assert '"name"' in params['system']
