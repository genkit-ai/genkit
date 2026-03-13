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


"""Tests for OpenAI compatible model implementation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from genkit.core.action._action import ActionRunContext
from genkit.plugins.compat_oai.models import OpenAIModel
from genkit.plugins.compat_oai.models.utils import strip_markdown_fences
from genkit.plugins.compat_oai.typing import OpenAIConfig
from genkit.types import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChunk,
    GenerationCommonConfig,
    Message,
    OutputConfig,
    Part,
    Role,
    TextPart,
)


def test_get_messages(sample_request: GenerateRequest) -> None:
    """Test _get_messages method.

    Ensures the method correctly converts GenerateRequest messages into OpenAI-compatible ChatMessage format.
    """
    model = OpenAIModel(model='gpt-4', client=MagicMock())
    messages = model._get_messages(sample_request.messages)

    assert len(messages) == 2
    assert messages[0]['role'] == 'system'
    assert messages[0]['content'] == 'You are an assistant'
    assert messages[1]['role'] == 'user'
    assert messages[1]['content'] == 'Hello, world!'


@pytest.mark.asyncio
async def test_get_openai_config(sample_request: GenerateRequest) -> None:
    """Test _get_openai_request_config method.

    Ensures the method correctly constructs the OpenAI API configuration dictionary.
    """
    model = OpenAIModel(model='gpt-4', client=MagicMock())
    openai_config = await model._get_openai_request_config(sample_request)

    assert isinstance(openai_config, dict)
    assert openai_config['model'] == 'gpt-4'
    assert 'messages' in openai_config
    assert isinstance(openai_config['messages'], list)


@pytest.mark.asyncio
async def test__generate(sample_request: GenerateRequest) -> None:
    """Test generate method calls OpenAI API and returns GenerateResponse."""
    mock_message = MagicMock()
    mock_message.content = 'Hello, user!'
    mock_message.role = 'model'
    mock_message.tool_calls = None
    mock_message.reasoning_content = None

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    model = OpenAIModel(model='gpt-4', client=mock_client)
    response = await model._generate(sample_request)

    mock_client.chat.completions.create.assert_called_once()
    assert isinstance(response, GenerateResponse)
    assert response.message is not None
    assert response.message.role == Role.MODEL
    assert response.message.content[0].root.text == 'Hello, user!'


@pytest.mark.asyncio
async def test__generate_stream(sample_request: GenerateRequest) -> None:
    """Test generate_stream method ensures it processes streamed responses correctly."""
    mock_client = MagicMock()

    class MockStream:
        def __init__(self, data: list[str]) -> None:
            self._data = data
            self._current = 0

        def __aiter__(self) -> 'MockStream':
            return self

        async def __anext__(self) -> object:
            if self._current >= len(self._data):
                raise StopAsyncIteration

            content = self._data[self._current]
            self._current += 1

            delta_mock = MagicMock()
            delta_mock.content = content
            delta_mock.role = None
            delta_mock.tool_calls = None
            delta_mock.reasoning_content = None

            choice_mock = MagicMock()
            choice_mock.delta = delta_mock

            return MagicMock(choices=[choice_mock])

    mock_client.chat.completions.create = AsyncMock(return_value=MockStream(['Hello', ', world!']))

    model = OpenAIModel(model='gpt-4', client=mock_client)
    collected_chunks = []

    def callback(chunk: GenerateResponseChunk) -> None:
        collected_chunks.append(chunk.content[0].root.text)

    await model._generate_stream(sample_request, callback)

    assert collected_chunks == ['Hello', ', world!']


@pytest.mark.parametrize(
    'stream',
    [
        True,
        False,
    ],
)
@pytest.mark.asyncio
async def test_generate(stream: bool, sample_request: GenerateRequest) -> None:
    """Tests for generate."""
    ctx_mock = MagicMock(spec=ActionRunContext)
    ctx_mock.is_streaming = stream

    mock_response = GenerateResponse(message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='mocked'))]))

    model = OpenAIModel(model='gpt-4', client=MagicMock())
    model._generate_stream = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]
    model._generate = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]
    model.normalize_config = MagicMock(return_value={})  # type: ignore[method-assign]
    response = await model.generate(sample_request, ctx_mock)

    assert response == mock_response
    if stream:
        model._generate_stream.assert_called_once()  # type: ignore[union-attr]
    else:
        model._generate.assert_called_once()  # type: ignore[union-attr]


@pytest.mark.parametrize(
    'config, expected',
    [
        (OpenAIConfig(model='test'), OpenAIConfig(model='test')),
        ({'model': 'test'}, OpenAIConfig(model='test')),
        (
            GenerationCommonConfig(temperature=0.7),
            OpenAIConfig(temperature=0.7),
        ),
        (
            None,
            Exception(),
        ),
    ],
)
def test_normalize_config(config: object, expected: object) -> None:
    """Tests for _normalize_config."""
    if isinstance(expected, Exception):
        with pytest.raises(ValueError, match=r'Expected request.config to be a dict or OpenAIConfig, got .*'):
            OpenAIModel.normalize_config(config)
    else:
        response = OpenAIModel.normalize_config(config)
        assert response == expected


_SAMPLE_SCHEMA: dict[str, object] = {
    'type': 'object',
    'title': 'RpgCharacter',
    'properties': {
        'name': {'type': 'string'},
        'level': {'type': 'integer'},
    },
    'required': ['name', 'level'],
}


class TestNeedsSchemaInPrompt:
    """Tests for _needs_schema_in_prompt."""

    def test_true_for_deepseek_with_json_and_schema(self) -> None:
        """Returns True for DeepSeek model with json format and schema."""
        model = OpenAIModel(model='deepseek-chat', client=MagicMock())
        output = OutputConfig(format='json', schema=_SAMPLE_SCHEMA)
        assert model._needs_schema_in_prompt(output) is True

    def test_false_for_gpt_with_json_and_schema(self) -> None:
        """Returns False for GPT models even with json format and schema."""
        model = OpenAIModel(model='gpt-4o', client=MagicMock())
        output = OutputConfig(format='json', schema=_SAMPLE_SCHEMA)
        assert model._needs_schema_in_prompt(output) is False

    def test_false_for_deepseek_without_schema(self) -> None:
        """Returns False for DeepSeek when no schema is provided."""
        model = OpenAIModel(model='deepseek-chat', client=MagicMock())
        output = OutputConfig(format='json')
        assert model._needs_schema_in_prompt(output) is False

    def test_false_for_deepseek_with_text_format(self) -> None:
        """Returns False for DeepSeek when format is text."""
        model = OpenAIModel(model='deepseek-chat', client=MagicMock())
        output = OutputConfig(format='text')
        assert model._needs_schema_in_prompt(output) is False

    def test_false_for_no_format(self) -> None:
        """Returns False when output has no format set."""
        model = OpenAIModel(model='deepseek-chat', client=MagicMock())
        output = OutputConfig()
        assert model._needs_schema_in_prompt(output) is False


class TestBuildSchemaInstruction:
    """Tests for _build_schema_instruction."""

    def test_returns_system_message(self) -> None:
        """Returns a dict with role 'system'."""
        result = OpenAIModel._build_schema_instruction(_SAMPLE_SCHEMA)
        assert result['role'] == 'system'

    def test_content_contains_schema(self) -> None:
        """Content includes the schema's field names and title."""
        result = OpenAIModel._build_schema_instruction(_SAMPLE_SCHEMA)
        assert '"RpgCharacter"' in result['content']
        assert '"name"' in result['content']
        assert '"level"' in result['content']

    def test_content_contains_instructions(self) -> None:
        """Content includes directive keywords."""
        result = OpenAIModel._build_schema_instruction(_SAMPLE_SCHEMA)
        assert 'EXACTLY' in result['content']
        assert 'JSON schema' in result['content']


class TestSchemaInjectionInConfig:
    """Tests for schema injection in _get_openai_request_config."""

    @pytest.mark.asyncio
    async def test_deepseek_injects_schema_message(self) -> None:
        """DeepSeek request prepends a schema system message."""
        model = OpenAIModel(model='deepseek-chat', client=MagicMock())
        request = GenerateRequest(
            messages=[
                Message(role=Role.USER, content=[Part(root=TextPart(text='Generate a character'))]),
            ],
            output=OutputConfig(format='json', schema=_SAMPLE_SCHEMA),
        )
        config = await model._get_openai_request_config(request)

        messages = config['messages']
        # Schema instruction is prepended as the first message.
        assert messages[0]['role'] == 'system'
        assert 'RpgCharacter' in messages[0]['content']
        # Original user message follows.
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == 'Generate a character'

    @pytest.mark.asyncio
    async def test_gpt_does_not_inject_schema_message(self) -> None:
        """GPT request does not prepend a schema system message."""
        model = OpenAIModel(model='gpt-4o', client=MagicMock())
        request = GenerateRequest(
            messages=[
                Message(role=Role.USER, content=[Part(root=TextPart(text='Generate a character'))]),
            ],
            output=OutputConfig(format='json', schema=_SAMPLE_SCHEMA),
        )
        config = await model._get_openai_request_config(request)

        messages = config['messages']
        # No extra system message — only the original user message.
        assert len(messages) == 1
        assert messages[0]['role'] == 'user'

    @pytest.mark.asyncio
    async def test_deepseek_without_schema_no_injection(self) -> None:
        """DeepSeek request without a schema does not inject anything."""
        model = OpenAIModel(model='deepseek-chat', client=MagicMock())
        request = GenerateRequest(
            messages=[
                Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))]),
            ],
            output=OutputConfig(format='json'),
        )
        config = await model._get_openai_request_config(request)

        messages = config['messages']
        assert len(messages) == 1
        assert messages[0]['role'] == 'user'

    @pytest.mark.asyncio
    async def test_deepseek_preserves_existing_system_message(self) -> None:
        """Schema injection does not clobber an existing system message."""
        model = OpenAIModel(model='deepseek-chat', client=MagicMock())
        request = GenerateRequest(
            messages=[
                Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='You are helpful'))]),
                Message(role=Role.USER, content=[Part(root=TextPart(text='Generate'))]),
            ],
            output=OutputConfig(format='json', schema=_SAMPLE_SCHEMA),
        )
        config = await model._get_openai_request_config(request)

        messages = config['messages']
        # Schema instruction prepended, then original system, then user.
        assert len(messages) == 3
        assert messages[0]['role'] == 'system'
        assert 'RpgCharacter' in messages[0]['content']
        assert messages[1]['role'] == 'system'
        assert messages[1]['content'] == 'You are helpful'
        assert messages[2]['role'] == 'user'


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


class TestCleanJsonResponse:
    """Tests for _clean_json_response."""

    def test_cleans_deepseek_json_response(self) -> None:
        """Strips markdown fences from DeepSeek JSON response."""
        model = OpenAIModel(model='deepseek-chat', client=MagicMock())
        request = GenerateRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
            output=OutputConfig(format='json', schema=_SAMPLE_SCHEMA),
        )
        response = GenerateResponse(
            request=request,
            message=Message(
                role=Role.MODEL,
                content=[Part(root=TextPart(text='```json\n{"name": "John", "level": 5}\n```'))],
            ),
        )
        cleaned = model._clean_json_response(response, request)
        assert cleaned.message is not None
        assert cleaned.message.content[0].root.text == '{"name": "John", "level": 5}'

    def test_no_op_for_gpt_model(self) -> None:
        """Does not modify responses from non-DeepSeek models."""
        model = OpenAIModel(model='gpt-4o', client=MagicMock())
        request = GenerateRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
            output=OutputConfig(format='json', schema=_SAMPLE_SCHEMA),
        )
        fenced_text = '```json\n{"name": "John", "level": 5}\n```'
        response = GenerateResponse(
            request=request,
            message=Message(
                role=Role.MODEL,
                content=[Part(root=TextPart(text=fenced_text))],
            ),
        )
        result = model._clean_json_response(response, request)
        assert result.message is not None
        assert result.message.content[0].root.text == fenced_text

    def test_no_op_for_text_output(self) -> None:
        """Does not modify responses when output format is not json."""
        model = OpenAIModel(model='deepseek-chat', client=MagicMock())
        request = GenerateRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
            output=OutputConfig(format='text'),
        )
        text = '```json\n{"a": 1}\n```'
        response = GenerateResponse(
            request=request,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text=text))]),
        )
        result = model._clean_json_response(response, request)
        assert result.message is not None
        assert result.message.content[0].root.text == text

    def test_no_op_for_no_output(self) -> None:
        """Does not modify responses when no output config is set."""
        model = OpenAIModel(model='deepseek-chat', client=MagicMock())
        request = GenerateRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
        )
        text = '```json\n{"a": 1}\n```'
        response = GenerateResponse(
            request=request,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text=text))]),
        )
        result = model._clean_json_response(response, request)
        assert result.message is not None
        assert result.message.content[0].root.text == text

    def test_no_op_when_no_fences(self) -> None:
        """Does not modify clean JSON responses."""
        model = OpenAIModel(model='deepseek-chat', client=MagicMock())
        request = GenerateRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
            output=OutputConfig(format='json', schema=_SAMPLE_SCHEMA),
        )
        text = '{"name": "John", "level": 5}'
        response = GenerateResponse(
            request=request,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text=text))]),
        )
        result = model._clean_json_response(response, request)
        # Should return the exact same object (no copy).
        assert result is response
