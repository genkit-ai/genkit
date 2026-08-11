#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for typed streaming output (issue #6007).

Covers:
- ModelResponseChunk.output partial validation into the output schema type
- ActionRunContext genericity over the chunk type
- End-to-end generate_stream with output_schema producing typed chunks
"""

import pytest
from pydantic import BaseModel

from genkit import Genkit, Message, ModelResponse, ModelResponseChunk
from genkit._ai._testing import define_programmable_model
from genkit._core._action import ActionRunContext
from genkit._core._typing import Part, Role, TextPart


class Recipe(BaseModel):
    """Test output schema."""

    title: str
    steps: list[str]


def _chunk(text: str, schema_type: type[BaseModel] | None = None) -> ModelResponseChunk:
    """Build a chunk whose accumulated text is exactly ``text``."""
    return ModelResponseChunk(
        role='model',
        content=[Part(root=TextPart(text=text))],
        schema_type=schema_type,
    )


class TestChunkPartialValidation:
    """chunk.output with a schema type validates partials into that type."""

    def test_unparseable_prefix_returns_none(self) -> None:
        assert _chunk('{"ti', schema_type=Recipe).output is None

    def test_missing_required_field_returns_none(self) -> None:
        # title present but steps has not started streaming yet
        assert _chunk('{"title": "Chocolate C', schema_type=Recipe).output is None

    def test_partial_trailing_value_returns_model_instance(self) -> None:
        out = _chunk('{"title": "Chocolate Cake", "steps": ["mi', schema_type=Recipe).output
        assert isinstance(out, Recipe)
        assert out.title == 'Chocolate Cake'
        assert out.steps == ['mi']

    def test_complete_json_returns_model_instance(self) -> None:
        out = _chunk('{"title": "Cake", "steps": ["mix", "bake"]}', schema_type=Recipe).output
        assert isinstance(out, Recipe)
        assert out.steps == ['mix', 'bake']

    def test_no_schema_type_preserves_raw_json_behavior(self) -> None:
        # Backward compat: without a schema type, output is the raw extracted JSON
        out = _chunk('{"title": "Cake", "steps": ["mix"]}').output
        assert out == {'title': 'Cake', 'steps': ['mix']}
        assert not isinstance(out, Recipe)

    def test_schema_validation_applies_to_chunk_parser_result(self) -> None:
        # chunk_parser output (used by format definitions) is also validated
        wrapper = ModelResponseChunk(
            role='model',
            content=[Part(root=TextPart(text='ignored'))],
            chunk_parser=lambda _c: {'title': 'Parsed', 'steps': ['a']},
            schema_type=Recipe,
        )
        out = wrapper.output
        assert isinstance(out, Recipe)
        assert out.title == 'Parsed'

    def test_non_dict_parse_result_passes_through(self) -> None:
        # A scalar/array payload can't be validated into an object schema;
        # it is returned as-is rather than silently dropped.
        assert _chunk('[1, 2, 3]', schema_type=Recipe).output == [1, 2, 3]


class TestActionRunContextGenerics:
    """ActionRunContext is generic over the chunk type, with a default."""

    def test_unparameterized_usage_still_works(self) -> None:
        received: list[object] = []
        ctx = ActionRunContext(streaming_callback=received.append)
        ctx.send_chunk({'anything': 1})
        assert received == [{'anything': 1}]

    def test_parameterized_usage_works_at_runtime(self) -> None:
        received: list[Recipe] = []
        ctx: ActionRunContext[Recipe] = ActionRunContext(streaming_callback=received.append)
        ctx.send_chunk(Recipe(title='t', steps=[]))
        assert received[0].title == 't'

    def test_class_is_subscriptable(self) -> None:
        # Generic alias construction must not raise
        assert ActionRunContext[Recipe] is not None


@pytest.mark.asyncio
async def test_generate_stream_with_output_schema_yields_typed_chunks() -> None:
    """End to end: generate_stream(output_schema=...) chunks validate into the schema."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    final_text = '{"title": "Chocolate Cake", "steps": ["mix", "bake"]}'
    pm.chunks = [
        [
            ModelResponseChunk(role=Role.MODEL, content=[Part(root=TextPart(text='{"title": "Chocolate C'))]),
            ModelResponseChunk(role=Role.MODEL, content=[Part(root=TextPart(text='ake", "steps": ["mi'))]),
            ModelResponseChunk(role=Role.MODEL, content=[Part(root=TextPart(text='x", "bake"]}'))]),
        ]
    ]
    pm.responses = [
        ModelResponse(
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text=final_text))]),
        )
    ]

    stream_result = ai.generate_stream(prompt='hi', output_schema=Recipe)

    outputs: list[object] = []
    async for chunk in stream_result.stream:
        outputs.append(chunk.output)

    # First chunk: required field `steps` hasn't started -> None.
    assert outputs[0] is None
    # Later chunks: real, partially-populated Recipe instances.
    assert isinstance(outputs[1], Recipe)
    assert outputs[1].title == 'Chocolate Cake'
    assert outputs[1].steps == ['mi']
    assert isinstance(outputs[2], Recipe)
    assert outputs[2].steps == ['mix', 'bake']

    # Final response is fully validated, as before.
    response = await stream_result.response
    assert isinstance(response.output, Recipe)
    assert response.output.title == 'Chocolate Cake'
    assert response.output.steps == ['mix', 'bake']


@pytest.mark.asyncio
async def test_generate_stream_without_schema_chunks_unchanged() -> None:
    """Backward compat: no output_schema keeps raw JSON chunk output."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[Part(root=TextPart(text='{"a": 1}'))])]]
    pm.responses = [
        ModelResponse(message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='{"a": 1}'))])),
    ]

    stream_result = ai.generate_stream(prompt='hi')
    async for chunk in stream_result.stream:
        assert chunk.output == {'a': 1}
    await stream_result.response
