#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the action module."""

import pytest
from pydantic import BaseModel

from genkit import (
    FinishReason,
    GenerationBlockedError,
    GenerationResponseError,
    GenkitError,
    Message,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
    ModelUsage,
    Role,
)
from genkit._ai._model import text_from_content
from genkit._core._typing import (
    ActionMetadata,
    DocumentPart,
    Media,
    MediaPart,
    Part,
    ReasoningPart,
    TextPart,
    ToolRequest,
    ToolRequestPart,
)
from genkit.model import get_basic_usage_stats, model_action_metadata


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


def test_text_from_content_skips_thoughts() -> None:
    """Thoughts are scratch work — they do not show up on ``.text``."""
    content = [
        Part(root=ReasoningPart(reasoning='let me think', text='secret thought')),
        Part(root=TextPart(text='hello')),
    ]
    assert text_from_content(content) == 'hello'


def test_assert_valid_throws_on_blocked() -> None:
    """Blocked finish is a refusal, not a usable reply."""
    response = ModelResponse(
        finish_reason=FinishReason.BLOCKED,
        finish_message='Content was blocked',
        message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='nope'))]),
    )

    with pytest.raises(GenerationBlockedError, match='Generation blocked: Content was blocked') as raised:
        response.assert_valid()
    assert isinstance(raised.value.details['response'], dict)
    assert raised.value.response.finish_reason == FinishReason.BLOCKED


def test_assert_valid_throws_when_no_message() -> None:
    """A length stop with no message has nothing to return."""
    response = ModelResponse(finish_reason=FinishReason.LENGTH, finish_message='Reached max tokens')

    with pytest.raises(GenerationResponseError, match='Model did not generate a message'):
        response.assert_valid()


def test_assert_valid_schema_throws_when_output_does_not_conform() -> None:
    """Structured output that is the wrong shape fails generate()."""

    class Person(BaseModel):
        name: str
        age: int

    response = ModelResponse(
        message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='{"name": "John", "age": "30"}'))]),
        finish_reason=FinishReason.STOP,
    )
    response.request = ModelRequest(
        messages=[],
        output_schema=Person.model_json_schema(),
    )
    response._schema_type = Person

    with pytest.raises(GenkitError, match='must be integer|age') as raised:
        response.assert_valid_schema()
    assert raised.value.status == 'INVALID_ARGUMENT'


def test_assert_valid_schema_passes_when_output_conforms() -> None:
    """Structured output that matches the schema is a usable reply."""

    class Person(BaseModel):
        name: str
        age: int

    response = ModelResponse(
        message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='{"name": "John", "age": 30}'))]),
        finish_reason=FinishReason.STOP,
    )
    response.request = ModelRequest(
        messages=[],
        output_schema=Person.model_json_schema(),
    )
    response._schema_type = Person

    response.assert_valid_schema()
    assert response.output.name == 'John'
    assert response.output.age == 30


def test_assert_valid_schema_names_non_json_output() -> None:
    """A leftover echo string is a schema miss, not a json5 column error."""
    response = ModelResponse(
        message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='[ECHO] hi'))]),
        finish_reason=FinishReason.STOP,
    )
    response.request = ModelRequest(messages=[], output_schema={'type': 'object'})

    with pytest.raises(GenkitError, match='not valid JSON for the requested schema') as raised:
        response.assert_valid_schema()
    assert raised.value.status == 'INVALID_ARGUMENT'
