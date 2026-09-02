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

import json
from collections.abc import AsyncIterator, Callable
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from genkit_openai.models import OpenAIModel
from genkit_openai.models.model import _usage_from_completion
from genkit_openai.models.utils import strip_markdown_fences
from genkit_openai.typing import OpenAIConfig
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel

from genkit import (
    FinishReason,
    GenkitError,
    Message,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
    Part,
    ReasoningPart,
    Role,
    TextPart,
    ToolRequestPart,
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
    mock_message.refusal = None

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
async def test__generate_no_choices(sample_request: ModelRequest) -> None:
    """A completion carrying no choices fails with a status the caller can handle."""
    mock_response = MagicMock()
    mock_response.choices = []

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    model = OpenAIModel(model='gpt-4', client=mock_client)

    with pytest.raises(GenkitError) as exc_info:
        await model._generate(sample_request)

    assert exc_info.value.status == 'INTERNAL'
    assert 'No choices in completion.' in str(exc_info.value)


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
            delta_mock.refusal = None

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


def _chunk(
    *,
    content: str | None = None,
    refusal: str | None = None,
    finish_reason: str | None = None,
    error: object | None = None,
) -> ChatCompletionChunk:
    """A one-choice chunk with no usage, as the API sends mid-stream.

    Built with ``construct``, which is how the SDK reads a response off the
    wire: a finish reason outside OpenAI's own five and an error object beside
    the choice both survive it, where validation would reject them.
    """
    choice: dict[str, Any] = {
        'index': 0,
        'delta': {'role': 'assistant', 'content': content, 'refusal': refusal},
        'finish_reason': finish_reason,
    }
    if error is not None:
        choice['error'] = error
    return ChatCompletionChunk.construct(
        id='1',
        object='chat.completion.chunk',
        created=1,
        model='gpt-4',
        choices=[choice],
        usage=None,
    )


def _completion(
    *,
    content: str | None = 'Hello, user!',
    refusal: str | None = None,
    finish_reason: str | None = 'stop',
    error: object | None = None,
) -> ChatCompletion:
    """A one-choice completion, built the same way as :func:`_chunk`."""
    choice: dict[str, Any] = {
        'index': 0,
        'message': {'role': 'assistant', 'content': content, 'refusal': refusal},
        'finish_reason': finish_reason,
    }
    if error is not None:
        choice['error'] = error
    return ChatCompletion.construct(
        id='1',
        object='chat.completion',
        created=1,
        model='gpt-4',
        choices=[choice],
        usage=None,
    )


def _mock_completion(completion: ChatCompletion) -> MagicMock:
    """A client whose create() answers with one completion."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=completion)
    return client


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


def _tool_call_chunk(call_id: str, name: str, args_segment: str) -> ChatCompletionChunk:
    """A chunk whose single choice carries a tool call delta."""
    return ChatCompletionChunk.model_validate({
        'id': '1',
        'object': 'chat.completion.chunk',
        'created': 1,
        'model': 'gpt-4',
        'choices': [
            {
                'index': 0,
                'delta': {
                    'role': 'assistant',
                    'tool_calls': [
                        {
                            'index': 0,
                            'id': call_id,
                            'type': 'function',
                            'function': {'name': name, 'arguments': args_segment},
                        }
                    ],
                },
                'finish_reason': None,
            }
        ],
        'usage': None,
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
    mock_message.refusal = None

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
    mock_message.refusal = None

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
    mock_client.chat.completions.create = _mock_stream([_chunk(content='Hello')])

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
    mock_client.chat.completions.create = _mock_stream([_chunk(content='Hello'), _usage_chunk()])

    model = OpenAIModel(model='gpt-4', client=mock_client)
    collected_chunks = []

    def callback(chunk: ModelResponseChunk) -> None:
        collected_chunks.append(chunk.content[0].root.text)

    response = await model._generate_stream(sample_request, callback)

    assert collected_chunks == ['Hello']
    _assert_usage_payload_reported(response.usage)


@pytest.mark.asyncio
async def test__generate_stream_no_choices(sample_request: ModelRequest) -> None:
    """A stream carrying only the usage chunk fails rather than returning an empty response."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = _mock_stream([_usage_chunk()])

    model = OpenAIModel(model='gpt-4', client=mock_client)

    with pytest.raises(GenkitError) as exc_info:
        await model._generate_stream(sample_request, lambda _: None)

    assert exc_info.value.status == 'INTERNAL'
    assert 'No choices in completion.' in str(exc_info.value)


@pytest.mark.asyncio
async def test__generate_stream_tool_calls_only(sample_request: ModelRequest) -> None:
    """A stream carrying only tool calls accumulates no text but still succeeds."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = _mock_stream([
        _tool_call_chunk('tool123', 'tool_fn', '{"a": '),
        _tool_call_chunk('tool123', 'tool_fn', '1}'),
        _usage_chunk(),
    ])

    model = OpenAIModel(model='gpt-4', client=mock_client)
    response = await model._generate_stream(sample_request, lambda _: None)

    assert response.message is not None
    assert response.text == ''
    tool_requests = [p.root.tool_request for p in response.message.content if p.root.tool_request]
    assert len(tool_requests) == 1
    assert tool_requests[0].name == 'tool_fn'


@pytest.mark.asyncio
async def test__generate_stream_empty_deltas(sample_request: ModelRequest) -> None:
    """A stream that carries choices but no content succeeds with an empty message."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = _mock_stream([_chunk(content=''), _chunk(content='')])

    model = OpenAIModel(model='gpt-4', client=mock_client)
    response = await model._generate_stream(sample_request, lambda _: None)

    assert response.message is not None
    assert response.text == ''


_FINISH_REASONS: list[tuple[str | None, FinishReason]] = [
    ('stop', FinishReason.STOP),
    ('tool_calls', FinishReason.STOP),
    ('end_turn', FinishReason.STOP),
    ('length', FinishReason.LENGTH),
    ('model_context_window_exceeded', FinishReason.LENGTH),
    ('content_filter', FinishReason.BLOCKED),
    ('sensitive', FinishReason.BLOCKED),
    ('error', FinishReason.OTHER),
    ('function_call', FinishReason.OTHER),
    ('network_error', FinishReason.OTHER),
    ('insufficient_system_resource', FinishReason.OTHER),
    ('a_reason_no_provider_has_sent_yet', FinishReason.UNKNOWN),
    (None, FinishReason.UNKNOWN),
]


@pytest.mark.parametrize(('reason', 'expected'), _FINISH_REASONS)
@pytest.mark.asyncio
async def test__generate_maps_finish_reason(
    reason: str | None, expected: FinishReason, sample_request: ModelRequest
) -> None:
    """Every reason a compatible provider ends a generation with is reported."""
    model = OpenAIModel(model='gpt-4', client=_mock_completion(_completion(finish_reason=reason)))

    response = await model._generate(sample_request)

    assert response.finish_reason == expected
    assert response.finish_message is None


@pytest.mark.parametrize(('reason', 'expected'), _FINISH_REASONS)
@pytest.mark.asyncio
async def test__generate_stream_maps_finish_reason(
    reason: str | None, expected: FinishReason, sample_request: ModelRequest
) -> None:
    """The reason on a stream's last chunk reaches the response."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = _mock_stream([_chunk(content='Hello'), _chunk(finish_reason=reason)])

    model = OpenAIModel(model='gpt-4', client=mock_client)
    response = await model._generate_stream(sample_request, lambda chunk: None)

    assert response.finish_reason == expected
    assert response.finish_message is None


@pytest.mark.asyncio
async def test__generate_refusal_only_is_blocked(sample_request: ModelRequest) -> None:
    """A message carrying only a refusal is a blocked response, not a failure."""
    model = OpenAIModel(
        model='gpt-4',
        client=_mock_completion(_completion(content=None, refusal='I cannot help with that.')),
    )

    response = await model._generate(sample_request)

    assert response.finish_reason == FinishReason.BLOCKED
    assert response.finish_message == 'I cannot help with that.'
    assert response.message is not None
    assert response.message.content == []


@pytest.mark.asyncio
async def test__generate_refusal_wins_over_a_clean_stop(sample_request: ModelRequest) -> None:
    """A refusal beside content blocks the response and keeps the content."""
    model = OpenAIModel(
        model='gpt-4',
        client=_mock_completion(_completion(content='Some of it.', refusal='Not the rest.')),
    )

    response = await model._generate(sample_request)

    assert response.finish_reason == FinishReason.BLOCKED
    assert response.finish_message == 'Not the rest.'
    assert response.message is not None
    assert response.message.content[0].root.text == 'Some of it.'


@pytest.mark.asyncio
async def test__generate_refusal_wins_over_a_failure_message(sample_request: ModelRequest) -> None:
    """A refusal beside a gateway's error object is the finish message."""
    model = OpenAIModel(
        model='gpt-4',
        client=_mock_completion(
            _completion(
                content=None,
                refusal='I cannot help with that.',
                finish_reason='error',
                error={'message': 'upstream timed out', 'code': 504},
            )
        ),
    )

    response = await model._generate(sample_request)

    assert response.finish_reason == FinishReason.BLOCKED
    assert response.finish_message == 'I cannot help with that.'


@pytest.mark.asyncio
async def test__generate_reports_the_failure_message_on_a_choice(sample_request: ModelRequest) -> None:
    """The message of an error object on a failing choice is the finish message."""
    model = OpenAIModel(
        model='gpt-4',
        client=_mock_completion(
            _completion(
                content='Partial ',
                finish_reason='error',
                error={'message': 'upstream timed out', 'code': 504},
            )
        ),
    )

    response = await model._generate(sample_request)

    assert response.finish_reason == FinishReason.OTHER
    assert response.finish_message == 'upstream timed out'


@pytest.mark.asyncio
async def test__generate_clean_stop_reports_no_failure_message(sample_request: ModelRequest) -> None:
    """A choice that stopped cleanly reports no finish message."""
    model = OpenAIModel(
        model='gpt-4',
        client=_mock_completion(_completion(error={'message': 'not this generation'})),
    )

    response = await model._generate(sample_request)

    assert response.finish_reason == FinishReason.STOP
    assert response.finish_message is None


@pytest.mark.asyncio
async def test__generate_ignores_an_error_field_that_is_not_an_object(sample_request: ModelRequest) -> None:
    """An error field that is not an object leaves the finish message unset."""
    model = OpenAIModel(
        model='gpt-4',
        client=_mock_completion(_completion(finish_reason='error', error='upstream timed out')),
    )

    response = await model._generate(sample_request)

    assert response.finish_reason == FinishReason.OTHER
    assert response.finish_message is None


@pytest.mark.asyncio
async def test__generate_stream_refusal_is_blocked(sample_request: ModelRequest) -> None:
    """A streamed refusal blocks the response and is not sent as content."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = _mock_stream([
        _chunk(refusal='I cannot '),
        _chunk(refusal='help with that.'),
        _chunk(finish_reason='stop'),
    ])

    model = OpenAIModel(model='gpt-4', client=mock_client)
    collected_chunks: list[ModelResponseChunk] = []

    response = await model._generate_stream(sample_request, collected_chunks.append)

    assert response.finish_reason == FinishReason.BLOCKED
    assert response.finish_message == 'I cannot help with that.'
    assert collected_chunks == []
    assert response.message is not None
    assert response.message.content == []


@pytest.mark.asyncio
async def test__generate_stream_reports_the_failure_message_on_a_choice(sample_request: ModelRequest) -> None:
    """A stream whose upstream failed reports the message the gateway sent."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = _mock_stream([
        _chunk(content='Partial '),
        _chunk(finish_reason='error', error={'message': 'upstream timed out', 'code': 504}),
    ])

    model = OpenAIModel(model='gpt-4', client=mock_client)
    response = await model._generate_stream(sample_request, lambda chunk: None)

    assert response.finish_reason == FinishReason.OTHER
    assert response.finish_message == 'upstream timed out'
    assert response.message is not None
    assert response.message.content[0].root.text == 'Partial '


@pytest.mark.asyncio
async def test__generate_stream_refusal_wins_over_a_failure_message(sample_request: ModelRequest) -> None:
    """A streamed refusal beside a gateway's error object is the finish message."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = _mock_stream([
        _chunk(refusal='I cannot help with that.'),
        _chunk(finish_reason='error', error={'message': 'upstream timed out', 'code': 504}),
    ])

    model = OpenAIModel(model='gpt-4', client=mock_client)
    response = await model._generate_stream(sample_request, lambda chunk: None)

    assert response.finish_reason == FinishReason.BLOCKED
    assert response.finish_message == 'I cannot help with that.'


@pytest.mark.asyncio
async def test_generate_classifies_an_unreadable_message(sample_request: ModelRequest) -> None:
    """A message with nothing to read at all is still INTERNAL, so retry runs."""
    ctx_mock = MagicMock(spec=ActionRunContext)
    type(ctx_mock).is_streaming = PropertyMock(return_value=False)
    model = OpenAIModel(model='gpt-4', client=_mock_completion(_completion(content=None)))

    with pytest.raises(GenkitError) as raised:
        await model.generate(sample_request, ctx_mock)
    assert raised.value.status == 'INTERNAL'


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


def _delta_chunk(make_chunk: Callable[..., ChatCompletionChunk], **delta: Any) -> ChatCompletionChunk:
    """A chunk whose delta carries exactly the given fields."""
    return make_chunk(choice={'delta': delta})


def _roots(chunk: ModelResponseChunk) -> list[Any]:
    """The part roots of a streamed chunk."""
    return [part.root for part in chunk.content]


@pytest.mark.asyncio
async def test__generate_stream_emits_reasoning_text_and_tool_call_from_one_chunk(
    sample_request: ModelRequest, make_chunk: Callable[..., ChatCompletionChunk]
) -> None:
    """A delta carrying all three kinds of content yields all three parts."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = _mock_stream([
        _delta_chunk(
            make_chunk,
            content='Checking.',
            reasoning_content='Need the weather.',
            tool_calls=[
                {
                    'index': 0,
                    'id': 'call_1',
                    'type': 'function',
                    'function': {'name': 'get_weather', 'arguments': '{"city": "NYC"}'},
                }
            ],
        )
    ])

    model = OpenAIModel(model='deepseek-reasoner', client=mock_client)
    collected: list[ModelResponseChunk] = []

    response = await model._generate_stream(sample_request, collected.append)

    assert len(collected) == 1
    streamed = _roots(collected[0])
    assert len(streamed) == 3
    assert isinstance(streamed[0], ReasoningPart)
    assert streamed[0].reasoning == 'Need the weather.'
    assert isinstance(streamed[1], TextPart)
    assert streamed[1].text == 'Checking.'
    assert isinstance(streamed[2], ToolRequestPart)
    assert streamed[2].tool_request.name == 'get_weather'
    assert streamed[2].tool_request.ref == 'call_1'

    assert response.message is not None
    final = [part.root for part in response.message.content]
    assert len(final) == 3
    assert isinstance(final[0], ReasoningPart)
    assert isinstance(final[1], TextPart)
    assert isinstance(final[2], ToolRequestPart)
    assert final[2].tool_request.input == {'city': 'NYC'}


@pytest.mark.asyncio
async def test__generate_stream_keeps_reasoning_interleaved_with_content(
    sample_request: ModelRequest, make_chunk: Callable[..., ChatCompletionChunk]
) -> None:
    """Reasoning and text that share a delta both reach the callback and the aggregate."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = _mock_stream([
        _delta_chunk(make_chunk, reasoning_content='Think'),
        _delta_chunk(make_chunk, reasoning_content=' harder.', content='The'),
        _delta_chunk(make_chunk, content=' answer.'),
    ])

    model = OpenAIModel(model='deepseek-reasoner', client=mock_client)
    collected: list[ModelResponseChunk] = []

    response = await model._generate_stream(sample_request, collected.append)

    assert [[type(root).__name__ for root in _roots(chunk)] for chunk in collected] == [
        ['ReasoningPart'],
        ['ReasoningPart', 'TextPart'],
        ['TextPart'],
    ]
    assert response.message is not None
    final = [part.root for part in response.message.content]
    assert [root.reasoning for root in final if isinstance(root, ReasoningPart)] == ['Think', ' harder.']
    assert [root.text for root in final if isinstance(root, TextPart)] == ['The', ' answer.']


@pytest.mark.asyncio
async def test__generate_stream_keeps_text_riding_with_tool_call_arguments(
    sample_request: ModelRequest, make_chunk: Callable[..., ChatCompletionChunk]
) -> None:
    """Text sharing a delta with an argument fragment is emitted and the arguments still accumulate."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = _mock_stream([
        _delta_chunk(
            make_chunk,
            tool_calls=[
                {'index': 0, 'id': 'tool123', 'type': 'function', 'function': {'name': 'tool_fn', 'arguments': ''}}
            ],
        ),
        _delta_chunk(make_chunk, content='Calling', tool_calls=[{'index': 0, 'function': {'arguments': '{"a": '}}]),
        _delta_chunk(make_chunk, content=' the tool.', tool_calls=[{'index': 0, 'function': {'arguments': '1}'}}]),
    ])

    model = OpenAIModel(model='gpt-4', client=mock_client)
    collected: list[ModelResponseChunk] = []

    response = await model._generate_stream(sample_request, collected.append)

    streamed = [root for chunk in collected for root in _roots(chunk)]
    assert [root.text for root in streamed if isinstance(root, TextPart)] == ['Calling', ' the tool.']
    fragments = [root for root in streamed if isinstance(root, ToolRequestPart)]
    assert len(fragments) == 3
    assert all(root.tool_request.name == 'tool_fn' for root in fragments)
    assert all(root.tool_request.ref == 'tool123' for root in fragments)
    assert json.loads(''.join(str(root.tool_request.input) for root in fragments)) == {'a': 1}

    assert response.message is not None
    final = [part.root for part in response.message.content]
    assert [root.text for root in final if isinstance(root, TextPart)] == ['Calling', ' the tool.']
    requests = [root.tool_request for root in final if isinstance(root, ToolRequestPart)]
    assert len(requests) == 1
    assert requests[0].input == {'a': 1}


@pytest.mark.asyncio
async def test__generate_stream_skips_a_delta_with_nothing_to_report(
    sample_request: ModelRequest, make_chunk: Callable[..., ChatCompletionChunk]
) -> None:
    """A delta with no content, reasoning or tool calls emits no chunk."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = _mock_stream([
        _delta_chunk(make_chunk, content='Hi'),
        _delta_chunk(make_chunk),
    ])

    model = OpenAIModel(model='gpt-4', client=mock_client)
    collected: list[ModelResponseChunk] = []

    response = await model._generate_stream(sample_request, collected.append)

    assert len(collected) == 1
    assert collected[0].content[0].root.text == 'Hi'
    assert response.message is not None
    assert len(response.message.content) == 1


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


async def _stream(chunks: list[ChatCompletionChunk]) -> AsyncIterator[ChatCompletionChunk]:
    """Yield chunks the way an AsyncStream does."""
    for chunk in chunks:
        yield chunk


def _client(response: object) -> MagicMock:
    """A client whose chat completion call answers with response."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


class TestResponseMetadata:
    """Tests for the response metadata both chat paths report."""

    @pytest.mark.asyncio
    async def test_generate_reports_ids_and_fingerprint(
        self, sample_request: ModelRequest, make_completion: Callable[..., ChatCompletion]
    ) -> None:
        """The fingerprint, model and id reach both custom and raw."""
        model = OpenAIModel(model='gpt-4o', client=_client(make_completion(system_fingerprint='fp_44709d6fcb')))
        response = await model._generate(sample_request)

        expected = {
            'systemFingerprint': 'fp_44709d6fcb',
            'model': 'gpt-4o-2024-08-06',
            'id': 'chatcmpl-abc',
        }
        assert response.custom == expected
        assert response.raw == expected

    @pytest.mark.asyncio
    async def test_generate_reports_citations_without_a_fingerprint(
        self, sample_request: ModelRequest, make_completion: Callable[..., ChatCompletion]
    ) -> None:
        """Citations survive on providers that never send a fingerprint."""
        citations = ['https://a.example', 'https://b.example']
        model = OpenAIModel(model='grok-4', client=_client(make_completion(citations=citations)))
        response = await model._generate(sample_request)

        assert response.raw is not None
        assert response.raw['citations'] == citations
        assert response.raw['id'] == 'chatcmpl-abc'
        assert 'systemFingerprint' not in response.raw

    @pytest.mark.asyncio
    async def test_generate_reports_the_choice_error_object(
        self, sample_request: ModelRequest, make_completion: Callable[..., ChatCompletion]
    ) -> None:
        """A gateway's error object rides on the metadata whole."""
        failure = {'message': 'Provider returned error', 'code': 429, 'metadata': {'provider_name': 'xai'}}
        completion = make_completion(choice={'finish_reason': 'error', 'error': failure})
        model = OpenAIModel(model='gpt-4o', client=_client(completion))
        response = await model._generate(sample_request)

        assert response.raw is not None
        assert response.raw['error'] == failure

    @pytest.mark.asyncio
    async def test_generate_omits_absent_fields(
        self, sample_request: ModelRequest, make_completion: Callable[..., ChatCompletion]
    ) -> None:
        """A response without the optional fields grows no null-valued keys."""
        model = OpenAIModel(model='gpt-4o', client=_client(make_completion()))
        response = await model._generate(sample_request)

        assert response.raw == {'model': 'gpt-4o-2024-08-06', 'id': 'chatcmpl-abc'}

    @pytest.mark.asyncio
    async def test_generate_stream_reports_chunk_metadata(
        self, sample_request: ModelRequest, make_chunk: Callable[..., ChatCompletionChunk]
    ) -> None:
        """Metadata spread across chunks is collected onto the final response."""
        chunks = [
            make_chunk(content='Hello', system_fingerprint='fp_stream'),
            make_chunk(content=', world!'),
            make_chunk(
                citations=['https://x.example'],
                choice={'finish_reason': 'error', 'error': {'message': 'upstream gave up'}},
            ),
        ]
        model = OpenAIModel(model='grok-4', client=_client(_stream(chunks)))

        collected = []

        def callback(chunk: ModelResponseChunk) -> None:
            collected.append(chunk.content[0].root.text)

        response = await model._generate_stream(sample_request, callback)

        assert collected == ['Hello', ', world!']
        assert response.custom == {
            'systemFingerprint': 'fp_stream',
            'model': 'grok-4',
            'id': 'chatcmpl-stream',
            'citations': ['https://x.example'],
            'error': {'message': 'upstream gave up'},
        }
        assert response.raw == response.custom

    @pytest.mark.asyncio
    async def test_cleaned_json_response_keeps_metadata(self, make_completion: Callable[..., ChatCompletion]) -> None:
        """Stripping markdown fences does not drop the metadata."""
        request = ModelRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Generate'))])],
            output=OutputConfig(format='json'),
        )
        completion = make_completion(
            content='```json\n{"name": "John", "level": 5}\n```',
            system_fingerprint='fp_deepseek',
        )
        model = OpenAIModel(model='deepseek-chat', client=_client(completion))
        response = await model._generate(request)

        assert response.message is not None
        assert response.message.content[0].root.text == '{"name": "John", "level": 5}'
        assert response.raw is not None
        assert response.raw['systemFingerprint'] == 'fp_deepseek'
        assert response.custom == response.raw
