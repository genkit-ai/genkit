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

from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from genkit_openai.models import OpenAIModel
from genkit_openai.models.model import _usage_from_completion
from genkit_openai.models.utils import strip_markdown_fences
from genkit_openai.typing import OpenAIConfig
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel

from genkit import (
    GenkitError,
    Message,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
    Part,
    Role,
    TextPart,
)
from genkit._core._model import OutputConfig
from genkit.plugin_api import ActionRunContext, ModelConfig


def test_unknown_chat_id_json_mode_uses_json_object() -> None:
    """An unlisted chat id that asked for JSON gets json_object, not a KeyError."""
    model = OpenAIModel(model='my-custom-ft', client=MagicMock())
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
        output=OutputConfig(format='json'),
    )
    assert model._get_response_format(request) == {'type': 'json_object'}


def test_get_messages(sample_request: ModelRequest) -> None:
    """Test _get_messages method.

    Ensures the method correctly converts ModelRequest messages into OpenAI-compatible ChatMessage format.
    """
    model = OpenAIModel(model='gpt-4', client=MagicMock())
    messages = model._get_messages(sample_request.messages)

    assert len(messages) == 2
    assert messages[0]['role'] == 'system'
    assert messages[0]['content'] == 'You are an assistant'
    assert messages[1]['role'] == 'user'
    assert messages[1]['content'] == 'Hello, world!'


@pytest.mark.asyncio
async def test_get_openai_config(sample_request: ModelRequest) -> None:
    """Test _get_openai_request_config method.

    Ensures the method correctly constructs the OpenAI API configuration dictionary.
    """
    model = OpenAIModel(model='gpt-4', client=MagicMock())
    openai_config = await model._get_openai_request_config(sample_request)

    assert isinstance(openai_config, dict)
    assert openai_config['model'] == 'gpt-4'
    assert 'messages' in openai_config
    assert isinstance(openai_config['messages'], list)
    assert openai_config['top_p'] == 0.9
    assert openai_config['temperature'] == 0.7
    assert openai_config['stop'] == ['stop']
    assert openai_config['max_tokens'] == 100
    assert 'topP' not in openai_config
    assert 'maxTokens' not in openai_config
    assert 'max_output_tokens' not in openai_config


@pytest.mark.asyncio
async def test_get_openai_config_peels_genkit_keys_and_passes_the_rest() -> None:
    """Genkit-only keys stay off create(); declared OpenAI fields and extras go out."""
    model = OpenAIModel(model='gpt-4o', client=MagicMock())
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])],
        config=OpenAIConfig.model_validate({
            'temperature': 0.5,
            'max_output_tokens': 128,
            'stop_sequences': ['END'],
            'api_key': 'secret',
            'top_k': 8,
            'version': 'gpt-4o-2024-08-06',
            'prompt_cache_key': 'abc',
            'some_new_openai_knob': 1,
        }),
    )
    body = await model._get_openai_request_config(request)
    assert body['temperature'] == 0.5
    assert body['stop'] == ['END']
    assert body['prompt_cache_key'] == 'abc'
    assert body['some_new_openai_knob'] == 1
    assert body['model'] == 'gpt-4o-2024-08-06'
    assert 'max_output_tokens' not in body
    assert 'stop_sequences' not in body
    assert 'api_key' not in body
    assert 'top_k' not in body
    assert 'version' not in body


@pytest.mark.asyncio
async def test_get_openai_config_model_field_overrides_version() -> None:
    """OpenAIConfig.model is the create() model id; it wins over version."""
    model = OpenAIModel(model='gpt-4o', client=MagicMock())
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])],
        config=OpenAIConfig(version='gpt-4o-2024-08-06', model='gpt-4.1'),
    )
    body = await model._get_openai_request_config(request)
    assert body['model'] == 'gpt-4.1'
    assert 'version' not in body


@pytest.mark.asyncio
async def test__generate(sample_request: ModelRequest) -> None:
    """Test generate method calls OpenAI API and returns ModelResponse."""
    mock_message = MagicMock()
    mock_message.content = 'Hello, user!'
    mock_message.role = 'model'
    mock_message.tool_calls = None
    mock_message.reasoning_content = None

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_response.usage = None

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    model = OpenAIModel(model='gpt-4', client=mock_client)
    response = await model._generate(sample_request)

    mock_client.chat.completions.create.assert_called_once()
    assert isinstance(response, ModelResponse)
    assert response.message is not None
    assert response.message.role == Role.MODEL
    assert response.message.content[0].root.text == 'Hello, user!'


@pytest.mark.asyncio
async def test__generate_stream(sample_request: ModelRequest) -> None:
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

            return MagicMock(choices=[choice_mock], usage=None)

    mock_client.chat.completions.create = AsyncMock(return_value=MockStream(['Hello', ', world!']))

    model = OpenAIModel(model='gpt-4', client=mock_client)
    collected_chunks = []

    def callback(chunk: ModelResponseChunk) -> None:
        collected_chunks.append(chunk.content[0].root.text)

    await model._generate_stream(sample_request, callback)

    assert collected_chunks == ['Hello', ', world!']


_USAGE_PAYLOAD: dict[str, Any] = {
    'prompt_tokens': 10,
    'completion_tokens': 7,
    'total_tokens': 17,
    'prompt_cache_hit_tokens': 6,
    'prompt_cache_miss_tokens': 4,
    'completion_tokens_details': {'reasoning_tokens': 5},
    'cost': 0.00042,
}


def _assert_usage_payload_reported(usage: Any) -> None:  # noqa: ANN401
    """Assert every part of _USAGE_PAYLOAD reached the response usage."""
    assert usage is not None
    assert usage.input_tokens == 10
    assert usage.output_tokens == 7
    assert usage.total_tokens == 17
    assert usage.thoughts_tokens == 5
    assert usage.cached_content_tokens == 6
    assert usage.custom == {'cost': 0.00042}


def _text_chunk(text: str) -> ChatCompletionChunk:
    """A content chunk with no usage, as the API sends mid-stream."""
    return ChatCompletionChunk.model_validate({
        'id': '1',
        'object': 'chat.completion.chunk',
        'created': 1,
        'model': 'gpt-4',
        'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': text}, 'finish_reason': None}],
        'usage': None,
    })


def _usage_chunk() -> ChatCompletionChunk:
    """The final usage chunk, which carries no choices."""
    return ChatCompletionChunk.model_validate({
        'id': '1',
        'object': 'chat.completion.chunk',
        'created': 1,
        'model': 'gpt-4',
        'choices': [],
        'usage': _USAGE_PAYLOAD,
    })


def _mock_stream(chunks: list[ChatCompletionChunk]) -> AsyncMock:
    """A create() mock returning chunks from an async iterator."""

    async def iterator() -> AsyncIterator[ChatCompletionChunk]:
        for chunk in chunks:
            yield chunk

    return AsyncMock(return_value=iterator())


@pytest.mark.asyncio
async def test__generate_reports_usage(sample_request: ModelRequest) -> None:
    """A non-streaming response's token usage reaches the ModelResponse."""
    mock_message = MagicMock()
    mock_message.content = 'Hello, user!'
    mock_message.role = 'model'
    mock_message.tool_calls = None
    mock_message.reasoning_content = None

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_response.usage = CompletionUsage.model_validate(_USAGE_PAYLOAD)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    model = OpenAIModel(model='gpt-4', client=mock_client)
    response = await model._generate(sample_request)

    _assert_usage_payload_reported(response.usage)
    # include_usage is added on the streaming path only.
    assert 'stream_options' not in mock_client.chat.completions.create.call_args.kwargs


@pytest.mark.asyncio
async def test__generate_reports_extra_token_counts() -> None:
    """Counts Genkit has no field for land in usage.custom; zeroes are dropped."""
    mock_message = MagicMock()
    mock_message.content = 'hi'
    mock_message.role = 'model'
    mock_message.tool_calls = None
    mock_message.reasoning_content = None

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_response.usage = CompletionUsage.model_validate({
        'prompt_tokens': 3,
        'completion_tokens': 2,
        'total_tokens': 5,
        'prompt_tokens_details': {'cached_tokens': 0, 'image_tokens': 9},
        'completion_tokens_details': {
            'audio_tokens': 4,
            'accepted_prediction_tokens': 0,
            'rejected_prediction_tokens': 2,
            'reasoning_tokens': 0,
        },
        'num_sources_used': 8,
    })

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    model = OpenAIModel(model='gpt-4', client=mock_client)
    request = ModelRequest(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])])
    response = await model._generate(request)

    assert response.usage is not None
    assert response.usage.custom == {
        'audioTokens': 4,
        'rejectedPredictionTokens': 2,
        'imageTokens': 9,
        'numSourcesUsed': 8,
    }
    assert response.usage.thoughts_tokens is None
    assert response.usage.cached_content_tokens is None


@pytest.mark.parametrize(
    ('usage_fields', 'expected'),
    [
        ({'prompt_tokens_details': {'cached_tokens': 64}}, 64),
        ({'prompt_cache_hit_tokens': 6}, 6),
        ({'prompt_tokens_details': {'cached_tokens': 64}, 'prompt_cache_hit_tokens': 6}, 64),
    ],
    ids=['prompt_tokens_details', 'top_level_fallback', 'details_take_precedence'],
)
def test_cached_content_tokens(usage_fields: dict[str, Any], expected: float) -> None:
    """cached_content_tokens prefers prompt_tokens_details over the top-level count."""
    usage = CompletionUsage.model_validate({
        'prompt_tokens': 10,
        'completion_tokens': 2,
        'total_tokens': 12,
        **usage_fields,
    })

    assert _usage_from_completion(usage).cached_content_tokens == expected


def test_accepted_prediction_tokens() -> None:
    """A non-zero predicted-outputs count reaches usage.custom."""
    usage = CompletionUsage.model_validate({
        'prompt_tokens': 10,
        'completion_tokens': 4,
        'total_tokens': 14,
        'completion_tokens_details': {'accepted_prediction_tokens': 3},
    })

    assert _usage_from_completion(usage).custom == {'acceptedPredictionTokens': 3}


@pytest.mark.asyncio
async def test__generate_stream_requests_usage(sample_request: ModelRequest) -> None:
    """A stream asks for its usage, keeping any stream_options the caller set."""
    config = OpenAIConfig(model='gpt-4', stream_options={'other': 'keep'})
    sample_request.config = config

    mock_client = MagicMock()
    mock_client.chat.completions.create = _mock_stream([_text_chunk('Hello')])

    model = OpenAIModel(model='gpt-4', client=mock_client)
    await model._generate_stream(sample_request, lambda chunk: None)

    assert mock_client.chat.completions.create.call_args.kwargs['stream_options'] == {
        'other': 'keep',
        'include_usage': True,
    }
    # The caller's config is reused across requests, so it must not be mutated.
    assert config.stream_options == {'other': 'keep'}


@pytest.mark.asyncio
async def test__generate_stream_reports_usage(sample_request: ModelRequest) -> None:
    """The choice-less usage chunk is read, not indexed into."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = _mock_stream([_text_chunk('Hello'), _usage_chunk()])

    model = OpenAIModel(model='gpt-4', client=mock_client)
    collected_chunks = []

    def callback(chunk: ModelResponseChunk) -> None:
        collected_chunks.append(chunk.content[0].root.text)

    response = await model._generate_stream(sample_request, callback)

    assert collected_chunks == ['Hello']
    _assert_usage_payload_reported(response.usage)


@pytest.mark.parametrize(
    'stream',
    [
        True,
        False,
    ],
)
@pytest.mark.asyncio
async def test_generate(stream: bool, sample_request: ModelRequest) -> None:
    """Tests for generate."""
    ctx_mock = MagicMock(spec=ActionRunContext)
    type(ctx_mock).is_streaming = PropertyMock(return_value=stream)

    mock_response = ModelResponse(message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='mocked'))]))

    model = OpenAIModel(model='gpt-4', client=MagicMock())
    # monkey-patch real methods with mocks; sidestep the static signatures.
    model_any = cast(Any, model)
    model_any._generate_stream = AsyncMock(return_value=mock_response)
    model_any._generate = AsyncMock(return_value=mock_response)
    model_any.normalize_config = MagicMock(return_value={})
    response = await model.generate(sample_request, ctx_mock)

    assert response == mock_response
    if stream:
        model_any._generate_stream.assert_called_once()
    else:
        model_any._generate.assert_called_once()


@pytest.mark.asyncio
async def test_generate_classifies_bad_config_type() -> None:
    """A config type we cannot send is INVALID_ARGUMENT so retry skips it."""
    ctx_mock = MagicMock(spec=ActionRunContext)
    type(ctx_mock).is_streaming = PropertyMock(return_value=False)
    model = OpenAIModel(model='gpt-4', client=MagicMock())

    class OtherConfig(BaseModel):
        pass

    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])],
        config=OtherConfig(),
    )

    with pytest.raises(GenkitError) as raised:
        await model.generate(request, ctx_mock)
    assert raised.value.status == 'INVALID_ARGUMENT'


@pytest.mark.parametrize(
    'config, expected',
    [
        (OpenAIConfig(model='test'), OpenAIConfig(model='test')),
        ({'model': 'test'}, OpenAIConfig(model='test')),
        (
            ModelConfig(temperature=0.7),
            OpenAIConfig(temperature=0.7),
        ),
        (
            ModelConfig(version='gpt-4o-2024-08-06'),
            OpenAIConfig(version='gpt-4o-2024-08-06'),
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
        request = ModelRequest(
            messages=[],
            output=OutputConfig(format='json', json_schema=_SAMPLE_SCHEMA),
        )
        assert model._needs_schema_in_prompt(request) is True

    def test_false_for_gpt_with_json_and_schema(self) -> None:
        """Returns False for GPT models even with json format and schema."""
        model = OpenAIModel(model='gpt-4o', client=MagicMock())
        request = ModelRequest(
            messages=[],
            output=OutputConfig(format='json', json_schema=_SAMPLE_SCHEMA),
        )
        assert model._needs_schema_in_prompt(request) is False

    def test_false_for_deepseek_without_schema(self) -> None:
        """Returns False for DeepSeek when no schema is provided."""
        model = OpenAIModel(model='deepseek-chat', client=MagicMock())
        request = ModelRequest(messages=[], output=OutputConfig(format='json'))
        assert model._needs_schema_in_prompt(request) is False

    def test_false_for_deepseek_with_text_format(self) -> None:
        """Returns False for DeepSeek when format is text."""
        model = OpenAIModel(model='deepseek-chat', client=MagicMock())
        request = ModelRequest(messages=[], output=OutputConfig(format='text'))
        assert model._needs_schema_in_prompt(request) is False

    def test_false_for_no_format(self) -> None:
        """Returns False when output has no format set."""
        model = OpenAIModel(model='deepseek-chat', client=MagicMock())
        request = ModelRequest(messages=[])
        assert model._needs_schema_in_prompt(request) is False


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
        request = ModelRequest(
            messages=[
                Message(role=Role.USER, content=[Part(root=TextPart(text='Generate a character'))]),
            ],
            output=OutputConfig(format='json', json_schema=_SAMPLE_SCHEMA),
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
        request = ModelRequest(
            messages=[
                Message(role=Role.USER, content=[Part(root=TextPart(text='Generate a character'))]),
            ],
            output=OutputConfig(format='json', json_schema=_SAMPLE_SCHEMA),
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
        request = ModelRequest(
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
        request = ModelRequest(
            messages=[
                Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='You are helpful'))]),
                Message(role=Role.USER, content=[Part(root=TextPart(text='Generate'))]),
            ],
            output=OutputConfig(format='json', json_schema=_SAMPLE_SCHEMA),
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
        request = ModelRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
            output=OutputConfig(format='json', json_schema=_SAMPLE_SCHEMA),
        )
        response = ModelResponse(
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
        request = ModelRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
            output=OutputConfig(format='json', json_schema=_SAMPLE_SCHEMA),
        )
        fenced_text = '```json\n{"name": "John", "level": 5}\n```'
        response = ModelResponse(
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
        request = ModelRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
            output=OutputConfig(format='text'),
        )
        text = '```json\n{"a": 1}\n```'
        response = ModelResponse(
            request=request,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text=text))]),
        )
        result = model._clean_json_response(response, request)
        assert result.message is not None
        assert result.message.content[0].root.text == text

    def test_no_op_for_no_output(self) -> None:
        """Does not modify responses when no output config is set."""
        model = OpenAIModel(model='deepseek-chat', client=MagicMock())
        request = ModelRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
        )
        text = '```json\n{"a": 1}\n```'
        response = ModelResponse(
            request=request,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text=text))]),
        )
        result = model._clean_json_response(response, request)
        assert result.message is not None
        assert result.message.content[0].root.text == text

    def test_no_op_when_no_fences(self) -> None:
        """Does not modify clean JSON responses."""
        model = OpenAIModel(model='deepseek-chat', client=MagicMock())
        request = ModelRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
            output=OutputConfig(format='json', json_schema=_SAMPLE_SCHEMA),
        )
        text = '{"name": "John", "level": 5}'
        response = ModelResponse(
            request=request,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text=text))]),
        )
        result = model._clean_json_response(response, request)
        # Should return the exact same object (no copy).
        assert result is response
