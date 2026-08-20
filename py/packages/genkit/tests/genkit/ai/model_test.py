#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the action module."""

import warnings

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from genkit import Message, ModelRequest, ModelResponse, ModelResponseChunk, ModelUsage
from genkit._ai._model import text_from_content
from genkit._core._schema import to_json_schema
from genkit._core._typing import (
    ActionMetadata,
    DocumentPart,
    Media,
    MediaPart,
    Part,
    TextPart,
    ToolRequest,
    ToolRequestPart,
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
        Message(
            role='model',
            content=[Part(root=TextPart(text='hello')), Part(root=TextPart(text=' world'))],
        ),
    )

    assert wrapper.text == 'hello world'


def test_response_wrapper_text() -> None:
    """Test text property of ModelResponse."""
    wrapper = ModelResponse(
        message=Message(
            role='model',
            content=[Part(root=TextPart(text='hello')), Part(root=TextPart(text=' world'))],
        ),
    )
    wrapper.request = ModelRequest(messages=[])

    assert wrapper.text == 'hello world'


def test_response_wrapper_output() -> None:
    """Test output property of ModelResponse."""
    wrapper = ModelResponse(
        message=Message(
            role='model',
            content=[Part(root=TextPart(text='{"foo":')), Part(root=TextPart(text='"bar'))],
        ),
    )
    wrapper.request = ModelRequest(messages=[])

    assert wrapper.output == {'foo': 'bar'}


def test_response_wrapper_messages() -> None:
    """Test messages property of ModelResponse."""
    wrapper = ModelResponse(
        message=Message(
            role='model',
            content=[Part(root=TextPart(text='baz'))],
        )
    )
    wrapper.request = ModelRequest(
        messages=[
            Message(
                role='user',
                content=[Part(root=TextPart(text='foo'))],
            ),
            Message(
                role='tool',
                content=[Part(root=TextPart(text='bar'))],
            ),
        ],
    )

    assert wrapper.messages == [
        Message(
            role='user',
            content=[Part(root=TextPart(text='foo'))],
        ),
        Message(
            role='tool',
            content=[Part(root=TextPart(text='bar'))],
        ),
        Message(
            role='model',
            content=[Part(root=TextPart(text='baz'))],
        ),
    ]


def test_response_wrapper_output_uses_parser() -> None:
    """Test that ModelResponse uses the provided message_parser."""
    wrapper = ModelResponse(
        message=Message(
            role='model',
            content=[Part(root=TextPart(text='{"foo":')), Part(root=TextPart(text='"bar'))],
        ),
    )
    wrapper.request = ModelRequest(messages=[])
    wrapper._message_parser = lambda x: 'banana'

    assert wrapper.output == 'banana'


def test_chunk_wrapper_text() -> None:
    """Test text property of ModelResponseChunk."""
    wrapper = ModelResponseChunk(
        chunk=ModelResponseChunk(content=[Part(root=TextPart(text='hello')), Part(root=TextPart(text=' world'))]),
        index=0,
        previous_chunks=[],
    )

    assert wrapper.text == 'hello world'


def test_chunk_wrapper_accumulated_text() -> None:
    """Test accumulated_text property of ModelResponseChunk."""
    wrapper = ModelResponseChunk(
        ModelResponseChunk(content=[Part(root=TextPart(text=' PS: aliens'))]),
        index=0,
        previous_chunks=[
            ModelResponseChunk(content=[Part(root=TextPart(text='hello')), Part(root=TextPart(text=' '))]),
            ModelResponseChunk(content=[Part(root=TextPart(text='world!'))]),
        ],
    )

    assert wrapper.accumulated_text == 'hello world! PS: aliens'


def test_chunk_wrapper_output() -> None:
    """Test output property of ModelResponseChunk."""
    wrapper = ModelResponseChunk(
        ModelResponseChunk(content=[Part(root=TextPart(text=', "baz":[1,2,'))]),
        index=0,
        previous_chunks=[
            ModelResponseChunk(content=[Part(root=TextPart(text='{"foo":')), Part(root=TextPart(text='"ba'))]),
            ModelResponseChunk(content=[Part(root=TextPart(text='r"'))]),
        ],
    )

    assert wrapper.output == {'foo': 'bar', 'baz': [1, 2]}


def test_chunk_wrapper_output_uses_parser() -> None:
    """Test that ModelResponseChunk uses the provided chunk_parser."""
    wrapper = ModelResponseChunk(
        ModelResponseChunk(content=[Part(root=TextPart(text=', "baz":[1,2,'))]),
        index=0,
        previous_chunks=[
            ModelResponseChunk(content=[Part(root=TextPart(text='{"foo":')), Part(root=TextPart(text='"ba'))]),
            ModelResponseChunk(content=[Part(root=TextPart(text='r"'))]),
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
                        Part(root=TextPart(text='1')),
                        Part(root=TextPart(text='2')),
                    ],
                ),
                Message(
                    role='user',
                    content=[
                        Part(root=MediaPart(media=Media(content_type='image', url=''))),
                        Part(root=MediaPart(media=Media(url='data:image'))),
                        Part(root=MediaPart(media=Media(content_type='audio', url=''))),
                        Part(root=MediaPart(media=Media(url='data:audio'))),
                        Part(root=MediaPart(media=Media(content_type='video', url=''))),
                        Part(root=MediaPart(media=Media(url='data:video'))),
                    ],
                ),
            ],
            Message(
                role='model',
                content=[
                    Part(root=TextPart(text='3')),
                    Part(root=MediaPart(media=Media(content_type='image', url=''))),
                    Part(root=MediaPart(media=Media(url='data:image'))),
                    Part(root=MediaPart(media=Media(content_type='audio', url=''))),
                    Part(root=MediaPart(media=Media(url='data:audio'))),
                    Part(root=MediaPart(media=Media(content_type='video', url=''))),
                    Part(root=MediaPart(media=Media(url='data:video'))),
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
            content=[Part(root=TextPart(text='bar'))],
        )
    )
    wrapper.request = ModelRequest(
        messages=[
            Message(
                role='user',
                content=[Part(root=TextPart(text='foo'))],
            ),
        ],
    )

    assert wrapper.tool_requests == []

    wrapper = ModelResponse(
        message=Message(
            role='model',
            content=[
                Part(root=ToolRequestPart(tool_request=ToolRequest(name='tool', input={'abc': 3}))),
                Part(root=TextPart(text='bar')),
            ],
        )
    )
    wrapper.request = ModelRequest(
        messages=[
            Message(
                role='user',
                content=[Part(root=TextPart(text='foo'))],
            ),
        ],
    )

    assert wrapper.tool_requests == [ToolRequestPart(tool_request=ToolRequest(name='tool', input={'abc': 3}))]


def test_response_wrapper_interrupts() -> None:
    """Test interrupts property of ModelResponse."""
    wrapper = ModelResponse(
        message=Message(
            role='model',
            content=[Part(root=TextPart(text='bar'))],
        )
    )
    wrapper.request = ModelRequest(
        messages=[
            Message(
                role='user',
                content=[Part(root=TextPart(text='foo'))],
            ),
        ],
    )

    assert wrapper.interrupts == []

    wrapper = ModelResponse(
        message=Message(
            role='model',
            content=[
                Part(root=ToolRequestPart(tool_request=ToolRequest(name='tool1', input={'abc': 3}))),
                Part(
                    root=ToolRequestPart(
                        tool_request=ToolRequest(name='tool2', input={'bcd': 4}),
                        metadata={'interrupt': {'banana': 'yes'}},
                    )
                ),
                Part(root=TextPart(text='bar')),
            ],
        )
    )
    wrapper.request = ModelRequest(
        messages=[
            Message(
                role='user',
                content=[Part(root=TextPart(text='foo'))],
            ),
        ],
    )

    assert wrapper.interrupts == [
        ToolRequestPart(
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
    content = [Part(root=TextPart(text='hello')), Part(root=TextPart(text=' world'))]
    assert text_from_content(content) == 'hello world'


def test_text_from_content_with_document_parts() -> None:
    """Test text_from_content with list of DocumentPart objects."""
    content = [DocumentPart(root=TextPart(text='doc1')), DocumentPart(root=TextPart(text=' doc2'))]
    assert text_from_content(content) == 'doc1 doc2'


def test_text_from_content_with_mixed_parts() -> None:
    """Test text_from_content with mixed Part and DocumentPart objects."""
    content = [
        Part(root=TextPart(text='part')),
        DocumentPart(root=TextPart(text=' text')),
    ]
    assert text_from_content(content) == 'part text'


def test_text_from_content_with_empty_list() -> None:
    """Test text_from_content with empty list."""
    assert text_from_content([]) == ''


def test_text_from_content_with_none_text() -> None:
    """Test text_from_content handles parts without text content."""
    content = [
        Part(root=TextPart(text='hello')),
        Part(root=MediaPart(media=Media(url='http://example.com/image.png'))),
        Part(root=TextPart(text=' world')),
    ]
    assert text_from_content(content) == 'hello world'


def test_bare_model_request_accepts_plugin_config_instance() -> None:
    """Bare ModelRequest(config=PluginConfig) keeps the plugin schema instance."""
    plugin_config = PluginConfig(api_key='k', response_modalities=['audio'])
    request = ModelRequest(
        messages=[Message(role='user', content=[Part(root=TextPart(text='hi'))])],
        config=plugin_config,
    )
    assert request.config is plugin_config
    assert isinstance(request.config, PluginConfig)


def test_bare_model_request_keeps_dict_config() -> None:
    """Dict configs stay dicts on bare ModelRequest; Action coerces to the plugin schema."""
    request = ModelRequest(
        messages=[Message(role='user', content=[Part(root=TextPart(text='hi'))])],
        config={'temperature': 0.5, 'api_key': 'k'},
    )
    assert request.config == {'temperature': 0.5, 'api_key': 'k'}


def test_parameterized_model_request_coerces_dict_to_plugin_config() -> None:
    """ModelRequest[PluginConfig](config={'api_key': 'k'}) builds a PluginConfig."""
    request = ModelRequest[PluginConfig](
        messages=[Message(role='user', content=[Part(root=TextPart(text='hi'))])],
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
            messages=[Message(role='user', content=[Part(root=TextPart(text='hi'))])],
            config=OtherConfig(top_k=3),  # pyright: ignore[reportArgumentType]
        )


def test_model_request_rejects_non_model_non_dict_config() -> None:
    """ModelRequest(config='not-a-config') is a ValidationError."""
    with pytest.raises(ValidationError, match='config must be a BaseModel or mapping'):
        ModelRequest(
            messages=[Message(role='user', content=[Part(root=TextPart(text='hi'))])],
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
        messages=[Message(role='user', content=[Part(root=TextPart(text='hi'))])],
        config={'api_key': 'k'},
    )
    # Convert all Python/Pydantic warnings into hard errors so silent serialization warnings fail the test.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        request.model_dump(mode='python')
        request.model_dump_json()
