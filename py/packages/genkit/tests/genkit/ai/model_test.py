#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the action module."""

import warnings

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel, ConfigDict, ValidationError

from genkit import FinishReason, Message, ModelResponse, ModelResponseChunk, Part, Role
from genkit._ai._model import define_model, text_from_content
from genkit._core._action import ActionRunContext
from genkit._core._error import RuntimeErrorReason
from genkit._core._model import OutputConfig, chunk_for_stream
from genkit._core._reflection import create_reflection_asgi_app
from genkit._core._registry import Registry
from genkit._core._schema import InvalidOutputSchemaError, to_json_schema
from genkit._core._typing import (
    ActionMetadata,
    Operation,
    ToolRequest,
)
from genkit.model import (
    ModelInfo,
    ModelRequest,
    ModelUsage,
    background_model,
    get_basic_usage_stats,
    model,
    model_action_metadata,
)


class PluginConfig(BaseModel):
    """Stand-in for a plugin-specific config schema (e.g. LyriaConfig)."""

    model_config = ConfigDict(extra='allow')
    api_key: str | None = None
    response_modalities: list[str] | None = None


def test_message_wrapper_text() -> None:
    """Test text property of Message."""
    wrapper = Message(
        role='model',
        content=[Part.from_text('hello'), Part.from_text(' world')],
    )

    assert wrapper.text == 'hello world'


def test_response_wrapper_text() -> None:
    """Test text property of ModelResponse."""
    wrapper = ModelResponse(
        message=Message(
            role='model',
            content=[Part.from_text('hello'), Part.from_text(' world')],
        ),
    )
    wrapper.request = ModelRequest(messages=[])

    assert wrapper.text == 'hello world'


def test_response_wrapper_output() -> None:
    """A finished reply split across parts as `{"foo":` + `"bar"}` reads back as the dict."""
    wrapper = ModelResponse(
        message=Message(
            role='model',
            content=[Part.from_text('{"foo":'), Part.from_text('"bar"}')],
        ),
    )
    wrapper.request = ModelRequest(messages=[])

    assert wrapper.output == {'foo': 'bar'}


def test_response_output_cut_off_reply_is_none() -> None:
    """A finished reply cut off at `{"foo": "bar` gives `output is None`."""
    wrapper = ModelResponse(
        message=Message(
            role='model',
            content=[Part.from_text('{"foo":'), Part.from_text('"bar')],
        ),
    )
    wrapper.request = ModelRequest(messages=[])

    assert wrapper.output is None


def test_response_wrapper_messages() -> None:
    """Test messages property of ModelResponse."""
    wrapper = ModelResponse(
        message=Message(
            role='model',
            content=[Part.from_text('baz')],
        )
    )
    wrapper.request = ModelRequest(
        messages=[
            Message(
                role='user',
                content=[Part.from_text('foo')],
            ),
            Message(
                role='tool',
                content=[Part.from_text('bar')],
            ),
        ],
    )

    assert wrapper.messages == [
        Message(
            role='user',
            content=[Part.from_text('foo')],
        ),
        Message(
            role='tool',
            content=[Part.from_text('bar')],
        ),
        Message(
            role='model',
            content=[Part.from_text('baz')],
        ),
    ]


def test_response_wrapper_output_uses_parser() -> None:
    """Test that ModelResponse uses the provided message_parser."""
    wrapper = ModelResponse(
        message=Message(
            role='model',
            content=[Part.from_text('{"foo":'), Part.from_text('"bar')],
        ),
    )
    wrapper.request = ModelRequest(messages=[])
    wrapper._message_parser = lambda x: 'banana'

    assert wrapper.output == 'banana'


def test_stream_chunk_stamps_index_and_parser_on_a_copy() -> None:
    """chunk_for_stream stamps index/parser on a copy."""
    source: ModelResponseChunk[str] = ModelResponseChunk(content=[Part.from_text('hi')])
    wrapped = chunk_for_stream(source, index=0, previous_chunks=[], chunk_parser=lambda _c: 'parsed')
    assert wrapped is not source
    assert wrapped.index == 0
    assert wrapped.text == 'hi'
    assert wrapped.output == 'parsed'


def test_chunk_wrapper_text() -> None:
    """Test text property of ModelResponseChunk."""
    wrapper = chunk_for_stream(
        ModelResponseChunk(content=[Part.from_text('hello'), Part.from_text(' world')]),
        index=0,
        previous_chunks=[],
    )

    assert wrapper.text == 'hello world'


def test_chunk_wrapper_accumulated_text() -> None:
    """Test accumulated_text property of ModelResponseChunk."""
    wrapper = chunk_for_stream(
        ModelResponseChunk(content=[Part.from_text(' PS: aliens')]),
        index=0,
        previous_chunks=[
            ModelResponseChunk(content=[Part.from_text('hello'), Part.from_text(' ')]),
            ModelResponseChunk(content=[Part.from_text('world!')]),
        ],
    )

    assert wrapper.accumulated_text == 'hello world! PS: aliens'


def test_chunk_wrapper_output() -> None:
    """Test output property of ModelResponseChunk."""
    wrapper = chunk_for_stream(
        ModelResponseChunk(content=[Part.from_text(', "baz":[1,2,')]),
        index=0,
        previous_chunks=[
            ModelResponseChunk(content=[Part.from_text('{"foo":'), Part.from_text('"ba')]),
            ModelResponseChunk(content=[Part.from_text('r"')]),
        ],
    )

    assert wrapper.output == {'foo': 'bar', 'baz': [1, 2]}


def test_chunk_wrapper_output_uses_parser() -> None:
    """Test that ModelResponseChunk uses the provided chunk_parser."""
    wrapper = chunk_for_stream(
        ModelResponseChunk(content=[Part.from_text(', "baz":[1,2,')]),
        index=0,
        previous_chunks=[
            ModelResponseChunk(content=[Part.from_text('{"foo":'), Part.from_text('"ba')]),
            ModelResponseChunk(content=[Part.from_text('r"')]),
        ],
        chunk_parser=lambda x: 'banana',
    )

    assert wrapper.output == 'banana'


@pytest.mark.parametrize(
    'test_input,test_response,expected_output',
    (
        [
            [],
            Message(role='model', content=[]),
            ModelUsage(
                input_images=0,
                input_videos=0,
                input_characters=0,
                input_audio_files=0,
                output_audio_files=0,
                output_characters=0,
                output_images=0,
                output_videos=0,
            ),
        ],
        [
            [
                Message(
                    role='user',
                    content=[
                        Part.from_text('1'),
                        Part.from_text('2'),
                    ],
                ),
                Message(
                    role='user',
                    content=[
                        Part.from_media('', content_type='image'),
                        Part.from_media('data:image'),
                        Part.from_media('', content_type='audio'),
                        Part.from_media('data:audio'),
                        Part.from_media('', content_type='video'),
                        Part.from_media('data:video'),
                    ],
                ),
            ],
            Message(
                role='model',
                content=[
                    Part.from_text('3'),
                    Part.from_media('', content_type='image'),
                    Part.from_media('data:image'),
                    Part.from_media('', content_type='audio'),
                    Part.from_media('data:audio'),
                    Part.from_media('', content_type='video'),
                    Part.from_media('data:video'),
                ],
            ),
            ModelUsage(
                input_images=2,
                input_videos=2,
                input_characters=2,
                input_audio_files=2,
                output_audio_files=2,
                output_characters=1,
                output_images=2,
                output_videos=2,
            ),
        ],
    ),
)
def test_get_basic_usage_stats(
    test_input: list[Message],
    test_response: Message,
    expected_output: ModelUsage,
) -> None:
    """Test get_basic_usage_stats utility."""
    assert get_basic_usage_stats(input_=test_input, response=test_response) == expected_output


def test_response_wrapper_tool_requests() -> None:
    """Test tool_requests property of ModelResponse."""
    wrapper = ModelResponse(
        message=Message(
            role='model',
            content=[Part.from_text('bar')],
        )
    )
    wrapper.request = ModelRequest(
        messages=[
            Message(
                role='user',
                content=[Part.from_text('foo')],
            ),
        ],
    )

    assert wrapper.tool_requests == []

    wrapper = ModelResponse(
        message=Message(
            role='model',
            content=[
                Part(tool_request=ToolRequest(name='tool', input={'abc': 3})),
                Part.from_text('bar'),
            ],
        )
    )
    wrapper.request = ModelRequest(
        messages=[
            Message(
                role='user',
                content=[Part.from_text('foo')],
            ),
        ],
    )

    assert wrapper.tool_requests == [Part(tool_request=ToolRequest(name='tool', input={'abc': 3}))]


def test_response_wrapper_interrupts() -> None:
    """Test interrupts property of ModelResponse."""
    wrapper = ModelResponse(
        message=Message(
            role='model',
            content=[Part.from_text('bar')],
        )
    )
    wrapper.request = ModelRequest(
        messages=[
            Message(
                role='user',
                content=[Part.from_text('foo')],
            ),
        ],
    )

    assert wrapper.interrupts == []

    wrapper = ModelResponse(
        message=Message(
            role='model',
            content=[
                Part(tool_request=ToolRequest(name='tool1', input={'abc': 3})),
                Part(
                    tool_request=ToolRequest(name='tool2', input={'bcd': 4}), metadata={'interrupt': {'banana': 'yes'}}
                ),
                Part.from_text('bar'),
            ],
        )
    )
    wrapper.request = ModelRequest(
        messages=[
            Message(
                role='user',
                content=[Part.from_text('foo')],
            ),
        ],
    )

    assert wrapper.interrupts == [
        Part(
            tool_request=ToolRequest(name='tool2', input={'bcd': 4}),
            metadata={'interrupt': {'banana': 'yes'}},
        )
    ]


def test_model_action_metadata() -> None:
    """Test for model_action_metadata."""
    action_metadata = model_action_metadata(
        name='test_model',
        info={'label': 'test_label'},
        config_schema=None,
    )

    assert isinstance(action_metadata, ActionMetadata)
    assert action_metadata.input_json_schema is not None
    assert action_metadata.output_json_schema is not None
    assert action_metadata.metadata == {'model': {'customOptions': None, 'label': 'test_label'}}


_QUALITY_SCHEMA = {
    'type': 'object',
    'properties': {'quality': {'type': 'string', 'enum': ['low', 'high']}},
}


async def _echo_model(request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
    return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))


def _model_card(action: object) -> dict[str, object]:
    metadata = getattr(action, 'metadata', None)
    assert isinstance(metadata, dict)
    card = metadata['model']
    assert isinstance(card, dict)
    return card


def test_model_info_config_schema_becomes_custom_options() -> None:
    """model(..., info=ModelInfo(config_schema=S)) has metadata['model']['customOptions'] == S."""
    action = model('acme/m', _echo_model, info=ModelInfo(config_schema=_QUALITY_SCHEMA))
    assert _model_card(action)['customOptions'] == _QUALITY_SCHEMA


def test_model_info_config_schema_does_not_emit_config_schema_key() -> None:
    """The same metadata has no configSchema key."""
    action = model('acme/m', _echo_model, info=ModelInfo(config_schema=_QUALITY_SCHEMA))
    assert 'configSchema' not in _model_card(action)


def test_model_config_schema_class_wins_over_info_config_schema() -> None:
    """With both a config class and an info schema set, customOptions is the class's JSON schema."""

    class Quality(BaseModel):
        quality: str

    action = model(
        'acme/m',
        _echo_model,
        config_schema=Quality,
        info=ModelInfo(config_schema=_QUALITY_SCHEMA),
    )
    card = _model_card(action)
    assert card['customOptions'] == to_json_schema(Quality)
    assert card['customOptions'] != _QUALITY_SCHEMA
    assert 'configSchema' not in card


def test_model_without_any_config_schema_has_no_custom_options() -> None:
    """A model with no config class and no info schema has no config form."""
    action = model('acme/m', _echo_model)
    assert _model_card(action) == {'label': 'acme/m'}


def test_model_action_metadata_accepts_model_info() -> None:
    """model_action_metadata(info=ModelInfo(...)) returns the same customOptions as the dumped dict."""
    info = ModelInfo(label='acme/m', config_schema=_QUALITY_SCHEMA)
    from_info = model_action_metadata('acme/m', info=info)
    from_dict = model_action_metadata('acme/m', info=info.model_dump(by_alias=True, exclude_none=True))
    assert from_info.metadata is not None
    assert from_dict.metadata is not None
    assert from_info.metadata['model']['customOptions'] == _QUALITY_SCHEMA
    assert from_dict.metadata['model']['customOptions'] == from_info.metadata['model']['customOptions']
    assert 'configSchema' not in from_info.metadata['model']
    assert 'configSchema' not in from_dict.metadata['model']


async def _start_background(request: ModelRequest, _ctx: ActionRunContext) -> Operation:
    return Operation(id='op')


async def _check_background(operation: Operation, _ctx: ActionRunContext) -> Operation:
    return operation


def test_background_model_info_config_schema_becomes_custom_options() -> None:
    """background_model(..., info=ModelInfo(config_schema=S)) advertises S as customOptions."""
    action = background_model(
        'acme/bg',
        _start_background,
        _check_background,
        info=ModelInfo(config_schema=_QUALITY_SCHEMA),
    )
    card = _model_card(action.start_action)
    assert card['customOptions'] == _QUALITY_SCHEMA
    assert 'configSchema' not in card


@pytest.mark.asyncio
async def test_reflection_list_actions_model_custom_options_from_info() -> None:
    """/api/actions shows the info schema under metadata.model.customOptions."""
    registry = Registry()
    define_model(registry, 'acme/m', _echo_model, info=ModelInfo(config_schema=_QUALITY_SCHEMA))
    app = create_reflection_asgi_app(registry)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.get('/api/actions')
    assert response.status_code == 200
    card = response.json()['/model/acme/m']['metadata']['model']
    assert card['customOptions'] == _QUALITY_SCHEMA
    assert 'configSchema' not in card


def test_text_from_content_with_parts() -> None:
    """Test text_from_content with list of Part objects."""
    content = [Part.from_text('hello'), Part.from_text(' world')]
    assert text_from_content(content) == 'hello world'


def test_text_from_content_with_empty_list() -> None:
    """Test text_from_content with empty list."""
    assert text_from_content([]) == ''


def test_text_from_content_with_none_text() -> None:
    """Test text_from_content handles parts without text content."""
    content = [
        Part.from_text('hello'),
        Part.from_media('http://example.com/image.png'),
        Part.from_text(' world'),
    ]
    assert text_from_content(content) == 'hello world'


def test_text_from_content_skips_thoughts() -> None:
    """Thoughts are scratch work — they do not show up on ``.text``."""
    content = [
        Part.from_reasoning('let me think'),
        Part.from_text('hello'),
    ]
    assert text_from_content(content) == 'hello'


def test_schema_check_marks_error_when_output_does_not_conform() -> None:
    """Structured output that is the wrong shape stays on the response as error."""

    class Person(BaseModel):
        name: str
        age: int

    response = ModelResponse(
        message=Message(role=Role.MODEL, content=[Part.from_text('{"name": "John", "age": "30"}')]),
        finish_reason=FinishReason.STOP,
    )
    response.request = ModelRequest(
        messages=[],
        output=OutputConfig(json_schema=Person.model_json_schema()),
    )
    response._schema_type = Person

    response._assert_valid_schema()
    assert response.finish_reason == FinishReason.STOP
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is RuntimeErrorReason.INVALID_OUTPUT
    assert response.output is None
    assert response.text == '{"name": "John", "age": "30"}'


def test_schema_check_passes_when_output_conforms() -> None:
    """Structured output that matches the schema is a usable reply."""

    class Person(BaseModel):
        name: str
        age: int

    response = ModelResponse[Person](
        message=Message(role=Role.MODEL, content=[Part.from_text('{"name": "John", "age": 30}')]),
        finish_reason=FinishReason.STOP,
    )
    response.request = ModelRequest(
        messages=[],
        output=OutputConfig(json_schema=Person.model_json_schema()),
    )
    response._schema_type = Person

    response._assert_valid_schema()
    assert response.output is not None
    assert response.output.name == 'John'
    assert response.output.age == 30


def test_schema_check_names_non_json_output() -> None:
    """A raw echo string is a schema miss, not a json5 column error."""
    response = ModelResponse(
        message=Message(role=Role.MODEL, content=[Part.from_text('[ECHO] hi')]),
        finish_reason=FinishReason.STOP,
    )
    response.request = ModelRequest(messages=[], output=OutputConfig(json_schema={'type': 'object'}))

    response._assert_valid_schema()
    assert response.finish_reason == FinishReason.STOP
    assert response.output is None
    assert response.finish_message is None
    assert response.error is not None
    assert 'not valid JSON' in response.error.message


def test_schema_check_keeps_blocked_finish() -> None:
    """A safety refusal keeps finish_reason=blocked; the text stays on .text."""
    response = ModelResponse(
        finish_reason=FinishReason.BLOCKED,
        finish_message='Content was blocked',
        message=Message(role=Role.MODEL, content=[Part.from_text('nope')]),
    )
    response.request = ModelRequest(messages=[], output=OutputConfig(json_schema={'type': 'object'}))

    response._assert_valid_schema()
    assert response.finish_reason == FinishReason.BLOCKED
    assert response.text == 'nope'
    assert response.output is None


def test_schema_check_keeps_length_on_truncated_json() -> None:
    """Hit the token cap — output validation does not replace the model's reason."""
    response = ModelResponse(
        finish_reason=FinishReason.LENGTH,
        message=Message(role=Role.MODEL, content=[Part.from_text('The recipe starts with')]),
    )
    response.request = ModelRequest(messages=[], output=OutputConfig(json_schema={'type': 'object'}))

    response._assert_valid_schema()
    assert response.finish_reason == FinishReason.LENGTH
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.text == 'The recipe starts with'
    assert response.output is None


def test_length_finish_still_parses_complete_json() -> None:
    """A full Recipe that also hit the token cap is still a Recipe."""

    class Recipe(BaseModel):
        title: str

    response = ModelResponse[Recipe](
        finish_reason=FinishReason.LENGTH,
        message=Message(role=Role.MODEL, content=[Part.from_text('{"title": "Soup"}')]),
    )
    response.request = ModelRequest(messages=[], output=OutputConfig(json_schema=Recipe.model_json_schema()))
    response._schema_type = Recipe

    response._assert_valid_schema()
    assert response.finish_reason == FinishReason.LENGTH
    assert response.output is not None
    assert response.output.title == 'Soup'


def test_schema_check_marks_error_when_output_is_empty() -> None:
    """An empty reply is a miss when a schema was requested."""
    response = ModelResponse(
        message=Message(role=Role.MODEL, content=[Part.from_text('')]),
        finish_reason=FinishReason.STOP,
    )
    response.request = ModelRequest(messages=[], output=OutputConfig(json_schema={'type': 'object'}))

    response._assert_valid_schema()
    assert response.finish_reason == FinishReason.STOP
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.output is None


def test_schema_check_keeps_other_finish() -> None:
    """No-image / unspecified image stop is other, not a schema miss."""
    response = ModelResponse(
        finish_reason=FinishReason.OTHER,
        message=Message(role=Role.MODEL, content=[Part.from_text('{"title": "Soup"}')]),
    )
    response.request = ModelRequest(messages=[], output=OutputConfig(json_schema={'type': 'object'}))

    response._assert_valid_schema()
    assert response.finish_reason == FinishReason.OTHER
    assert response.output is None


def test_schema_check_broken_schema_still_throws() -> None:
    """A caller-broken schema is not stamped as a model miss."""
    response = ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part.from_text('{"title": "Soup"}')]),
    )
    response.request = ModelRequest(
        messages=[],
        output=OutputConfig(json_schema={'type': 'not-a-json-type'}),
    )

    with pytest.raises(InvalidOutputSchemaError):
        response._assert_valid_schema()
    assert response.finish_reason == FinishReason.STOP


def test_bare_model_request_accepts_plugin_config_instance() -> None:
    """Bare ModelRequest(config=PluginConfig) keeps the plugin schema instance."""
    plugin_config = PluginConfig(api_key='k', response_modalities=['audio'])
    request = ModelRequest(
        messages=[Message(role='user', content=[Part.from_text('hi')])],
        config=plugin_config,
    )
    assert request.config is plugin_config
    assert isinstance(request.config, PluginConfig)


def test_bare_model_request_keeps_dict_config() -> None:
    """Dict configs stay dicts on bare ModelRequest; Action coerces to the plugin schema."""
    request = ModelRequest(
        messages=[Message(role='user', content=[Part.from_text('hi')])],
        config={'temperature': 0.5, 'api_key': 'k'},
    )
    assert request.config == {'temperature': 0.5, 'api_key': 'k'}


def test_parameterized_model_request_coerces_dict_to_plugin_config() -> None:
    """ModelRequest[PluginConfig](config={'api_key': 'k'}) builds a PluginConfig."""
    request = ModelRequest[PluginConfig](
        messages=[Message(role='user', content=[Part.from_text('hi')])],
        config={'api_key': 'k'},
    )
    assert isinstance(request.config, PluginConfig)
    assert request.config.api_key == 'k'


def test_parameterized_model_request_rejects_mismatched_config_instance() -> None:
    """ModelRequest[PluginConfig](config=OtherConfig()) is a ValidationError."""

    class OtherConfig(BaseModel):
        top_k: int | None = None

    with pytest.raises(ValidationError):
        ModelRequest[PluginConfig](
            messages=[Message(role='user', content=[Part.from_text('hi')])],
            config=OtherConfig(top_k=3),  # pyright: ignore[reportArgumentType]
        )


def test_model_request_rejects_non_model_non_dict_config() -> None:
    """ModelRequest(config='not-a-config') is a ValidationError."""
    with pytest.raises(ValidationError, match='config must be a BaseModel or mapping'):
        ModelRequest(
            messages=[Message(role='user', content=[Part.from_text('hi')])],
            config='not-a-config',  # pyright: ignore[reportArgumentType]
        )


def test_parameterized_model_request_config_json_schema_refs_plugin_schema() -> None:
    """Verify JSON schema for ModelRequest[PluginConfig] includes a $ref to PluginConfig for Reflection API / Dev UI."""
    schema = to_json_schema(ModelRequest[PluginConfig])
    config_prop = schema['properties']['config']
    assert any('$ref' in arm for arm in config_prop.get('anyOf', [])), config_prop


def test_model_request_dump_emits_no_serializer_warnings() -> None:
    """Verify model_dump() and model_dump_json() execute without triggering Pydantic serialization warnings."""
    request = ModelRequest[PluginConfig](
        messages=[Message(role='user', content=[Part.from_text('hi')])],
        config={'api_key': 'k'},
    )
    # Convert all Python/Pydantic warnings into hard errors so silent serialization warnings fail the test.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        request.model_dump(mode='python')
        request.model_dump_json()


def test_output_returns_none_on_unparseable_text_without_schema() -> None:
    """Reading .output never raises ValueError, even when no schema was requested."""
    response = ModelResponse(
        message=Message(role=Role.MODEL, content=[Part.from_text('plain unparseable text')]),
        finish_reason=FinishReason.STOP,
        request=ModelRequest(messages=[]),
    )
    assert response.output is None


def test_output_returns_none_on_unparseable_text_with_json_format_no_schema() -> None:
    """Reading .output returns None when format='json' is requested but output is unparseable."""
    response = ModelResponse(
        message=Message(role=Role.MODEL, content=[Part.from_text('plain unparseable text')]),
        finish_reason=FinishReason.STOP,
        request=ModelRequest(messages=[], output=OutputConfig(format='json')),
    )
    assert response.output is None


def test_schema_check_marks_invalid_output_when_json_format_no_schema_unparseable() -> None:
    """Unparseable text with format='json' (no schema) records INVALID_OUTPUT."""
    response = ModelResponse(
        message=Message(role=Role.MODEL, content=[Part.from_text('plain unparseable text')]),
        finish_reason=FinishReason.STOP,
        request=ModelRequest(messages=[], output=OutputConfig(format='json')),
    )
    response._assert_valid_schema()
    assert response.finish_reason == FinishReason.STOP
    assert response.output is None
    assert response.text == 'plain unparseable text'
    assert response.error is not None
    assert response.error.reason is RuntimeErrorReason.INVALID_OUTPUT
    assert 'not valid JSON' in response.error.message


def test_schema_check_marks_invalid_output_when_array_format_no_schema_unparseable() -> None:
    """Unparseable text with format='array' (no schema) records INVALID_OUTPUT."""
    from genkit._ai._formats._array import ArrayFormat

    fmt = ArrayFormat()
    formatter = fmt.handle(None)
    response = ModelResponse(
        message=Message(role=Role.MODEL, content=[Part.from_text('plain unparseable text')]),
        finish_reason=FinishReason.STOP,
        request=ModelRequest(messages=[], output=OutputConfig(format='array')),
    )
    response._message_parser = formatter.parse_message
    response._assert_valid_schema()
    assert response.finish_reason == FinishReason.STOP
    assert response.output is None
    assert response.text == 'plain unparseable text'
    assert response.error is not None
    assert response.error.reason is RuntimeErrorReason.INVALID_OUTPUT


def test_schema_check_passes_when_json_format_no_schema_valid() -> None:
    """Valid JSON with format='json' (no schema) parses cleanly without error."""
    response = ModelResponse(
        message=Message(role=Role.MODEL, content=[Part.from_text('{"item": "bread"}')]),
        finish_reason=FinishReason.STOP,
        request=ModelRequest(messages=[], output=OutputConfig(format='json')),
    )
    response._assert_valid_schema()
    assert response.finish_reason == FinishReason.STOP
    assert response.error is None
    assert response.output == {'item': 'bread'}
