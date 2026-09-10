#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the action module."""

import warnings

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from genkit import (
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
    ModelUsage,
    Part,
    Role,
)
from genkit._ai._model import text_from_content
from genkit._core._model import OutputConfig
from genkit._core._schema import InvalidOutputSchemaError, to_json_schema
from genkit._core._typing import (
    ActionMetadata,
    ToolRequest,
)
from genkit.model import get_basic_usage_stats, model_action_metadata


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
    """Test output property of ModelResponse."""
    wrapper = ModelResponse(
        message=Message(
            role='model',
            content=[Part.from_text('{"foo":'), Part.from_text('"bar')],
        ),
    )
    wrapper.request = ModelRequest(messages=[])

    assert wrapper.output == {'foo': 'bar'}


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


def test_chunk_wrapper_text() -> None:
    """Test text property of ModelResponseChunk."""
    wrapper = ModelResponseChunk(
        ModelResponseChunk(content=[Part.from_text('hello'), Part.from_text(' world')]),
        index=0,
        previous_chunks=[],
    )

    assert wrapper.text == 'hello world'


def test_chunk_wrapper_accumulated_text() -> None:
    """Test accumulated_text property of ModelResponseChunk."""
    wrapper = ModelResponseChunk(
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
    wrapper = ModelResponseChunk(
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
    wrapper = ModelResponseChunk(
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


def test_assert_valid_schema_marks_failed_when_output_does_not_conform() -> None:
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

    response.assert_valid_schema()
    assert response.finish_reason == FinishReason.FAILED
    assert response.output is None
    assert response.text == '{"name": "John", "age": "30"}'


def test_assert_valid_schema_passes_when_output_conforms() -> None:
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

    response.assert_valid_schema()
    assert response.output is not None
    assert response.output.name == 'John'
    assert response.output.age == 30


def test_assert_valid_schema_names_non_json_output() -> None:
    """A leftover echo string is a schema miss, not a json5 column error."""
    response = ModelResponse(
        message=Message(role=Role.MODEL, content=[Part.from_text('[ECHO] hi')]),
        finish_reason=FinishReason.STOP,
    )
    response.request = ModelRequest(messages=[], output=OutputConfig(json_schema={'type': 'object'}))

    response.assert_valid_schema()
    assert response.finish_reason == FinishReason.FAILED
    assert response.output is None
    assert 'not valid JSON' in (response.finish_message or '')


def test_assert_valid_schema_keeps_blocked_finish() -> None:
    """A safety refusal keeps finish_reason=blocked; leftover is on .text."""
    response = ModelResponse(
        finish_reason=FinishReason.BLOCKED,
        finish_message='Content was blocked',
        message=Message(role=Role.MODEL, content=[Part.from_text('nope')]),
    )
    response.request = ModelRequest(messages=[], output=OutputConfig(json_schema={'type': 'object'}))

    response.assert_valid_schema()
    assert response.finish_reason == FinishReason.BLOCKED
    assert response.text == 'nope'
    assert response.output is None


def test_assert_valid_schema_marks_failed_on_truncated_json() -> None:
    """Hit the token cap — non-conforming json becomes FAILED."""
    response = ModelResponse(
        finish_reason=FinishReason.LENGTH,
        message=Message(role=Role.MODEL, content=[Part.from_text('The recipe starts with')]),
    )
    response.request = ModelRequest(messages=[], output=OutputConfig(json_schema={'type': 'object'}))

    response.assert_valid_schema()
    assert response.finish_reason == FinishReason.FAILED
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

    response.assert_valid_schema()
    assert response.finish_reason == FinishReason.LENGTH
    assert response.output is not None
    assert response.output.title == 'Soup'


def test_assert_valid_schema_marks_failed_when_output_is_empty() -> None:
    """An empty reply is a miss when a schema was requested."""
    response = ModelResponse(
        message=Message(role=Role.MODEL, content=[Part.from_text('')]),
        finish_reason=FinishReason.STOP,
    )
    response.request = ModelRequest(messages=[], output=OutputConfig(json_schema={'type': 'object'}))

    response.assert_valid_schema()
    assert response.finish_reason == FinishReason.FAILED
    assert response.output is None


def test_assert_valid_schema_keeps_other_finish() -> None:
    """No-image / unspecified image stop is other, not a schema miss."""
    response = ModelResponse(
        finish_reason=FinishReason.OTHER,
        message=Message(role=Role.MODEL, content=[Part.from_text('{"title": "Soup"}')]),
    )
    response.request = ModelRequest(messages=[], output=OutputConfig(json_schema={'type': 'object'}))

    response.assert_valid_schema()
    assert response.finish_reason == FinishReason.OTHER
    assert response.output is None


def test_assert_valid_schema_broken_schema_still_throws() -> None:
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
        response.assert_valid_schema()
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
