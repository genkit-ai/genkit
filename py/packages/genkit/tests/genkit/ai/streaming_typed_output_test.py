#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for typed streaming output (issue #6007).

Covers:
- ModelResponseChunk.output wrapping extracted JSON in a synthesized
  all-optional partial of the output schema type
- ActionRunContext genericity over the chunk type
- End-to-end generate_stream with output_schema producing partial chunks
"""

from typing import Annotated, Any, TypeVar, cast, overload

import pytest
from pydantic import BaseModel, Field, field_validator

from genkit import Genkit, Message, ModelResponse, ModelResponseChunk
from genkit._ai._testing import define_programmable_model
from genkit._core._action import ActionRunContext
from genkit._core._partial import partial_model
from genkit._core._typing import Part, Role, TextPart

OutputT = TypeVar('OutputT', bound=BaseModel)


class Recipe(BaseModel):
    """Test output schema."""

    title: str
    steps: list[str]


@overload
def _chunk(text: str, schema_type: type[OutputT]) -> ModelResponseChunk[OutputT]: ...
@overload
def _chunk(text: str, schema_type: None = None) -> ModelResponseChunk[object]: ...
def _chunk(text: str, schema_type: type[BaseModel] | None = None) -> ModelResponseChunk[Any]:
    """Build a chunk whose accumulated text is exactly ``text``."""
    return ModelResponseChunk(
        role='model',
        content=[Part(root=TextPart(text=text))],
        schema_type=schema_type,
    )


class TestChunkPartialOutput:
    """chunk.output with a schema type wraps extracted JSON in a partial."""

    def test_preamble_returns_none(self) -> None:
        # No JSON object has started yet.
        assert _chunk('Here is the JSON:', schema_type=Recipe).output is None
        assert _chunk('Here is the JSON:').output is None

    def test_prefix_with_no_fields_is_empty_partial(self) -> None:
        # The object has started but no key has finished; all fields None.
        out = _chunk('{"ti', schema_type=Recipe).output
        assert out is not None
        assert not isinstance(out, Recipe)
        assert out.title is None
        assert out.steps is None

    def test_first_field_is_available_immediately(self) -> None:
        # title present but steps has not started streaming yet:
        # the partial carries title, steps stays None. Not None overall.
        out = _chunk('{"title": "Chocolate C', schema_type=Recipe).output
        assert out is not None
        assert not isinstance(out, Recipe)
        assert out.title == 'Chocolate C'
        assert out.steps is None

    def test_partial_trailing_value(self) -> None:
        out = _chunk('{"title": "Chocolate Cake", "steps": ["mi', schema_type=Recipe).output
        assert out is not None
        assert out.title == 'Chocolate Cake'
        assert out.steps == ['mi']

    def test_complete_json_is_still_the_partial_type(self) -> None:
        # Chunks never hand back the real model; only ModelResponse.output does.
        out = _chunk('{"title": "Cake", "steps": ["mix", "bake"]}', schema_type=Recipe).output
        assert out is not None
        assert not isinstance(out, Recipe)
        assert type(out).__name__ == 'RecipePartial'
        assert out.steps == ['mix', 'bake']

    def test_no_schema_type_preserves_raw_json_behavior(self) -> None:
        # Backward compat: without a schema type, output is the raw extracted JSON
        out = _chunk('{"title": "Cake", "steps": ["mix"]}').output
        assert out == {'title': 'Cake', 'steps': ['mix']}
        assert not isinstance(out, Recipe)

    def test_partial_wraps_chunk_parser_result(self) -> None:
        # chunk_parser output (used by format definitions) is also wrapped
        wrapper: ModelResponseChunk[Recipe] = ModelResponseChunk(
            role='model',
            content=[Part(root=TextPart(text='ignored'))],
            chunk_parser=lambda _c: {'title': 'Parsed', 'steps': ['a']},
            schema_type=Recipe,
        )
        out = wrapper.output
        assert out is not None
        assert not isinstance(out, Recipe)
        assert out.title == 'Parsed'

    def test_non_dict_parse_result_passes_through(self) -> None:
        # A scalar/array payload can't be validated into an object schema;
        # it is returned as-is rather than silently dropped.
        assert _chunk('[1, 2, 3]', schema_type=Recipe).output == [1, 2, 3]

    def test_wrong_typed_value_yields_none_instead_of_raising(self) -> None:
        # A wrong-typed value (title as a number) can't fit even the partial.
        # The chunk degrades to None instead of crashing the caller's loop;
        # the final response is where the real ValidationError surfaces.
        assert _chunk('{"title": 123}', schema_type=Recipe).output is None


class TestPartialModelSynthesis:
    """partial_model rewrites nested annotations and drops constraints."""

    def test_constraints_and_validators_are_dropped(self) -> None:
        # A half-streamed value legitimately violates constraints (a streamed
        # int passes through 0, a capitalized string starts mid-word), so the
        # partial keeps only the types. The real model still enforces
        # everything on the final response.
        class Strict(BaseModel):
            servings: int = Field(gt=0)
            rating: Annotated[int, Field(ge=1, le=5)]
            title: str

            @field_validator('title')
            @classmethod
            def _capitalized(cls, v: str) -> str:
                if not v[0].isupper():
                    raise ValueError('must be capitalized')
                return v

        out = cast('Any', partial_model(Strict).model_validate({'servings': -5, 'rating': 99, 'title': 'lowercase'}))
        assert out.servings == -5
        assert out.rating == 99
        assert out.title == 'lowercase'

    def test_union_members_become_partials(self) -> None:
        # A mid-stream nested object under a multi-member union must validate
        # against partial members, not the real models (which require fields).
        class Cat(BaseModel):
            meow: str

        class Dog(BaseModel):
            bark: str
            volume: int

        class Pet(BaseModel):
            animal: Cat | Dog

        out = cast('Any', partial_model(Pet).model_validate({'animal': {'bark': 'woof'}}))
        assert out.animal is not None
        assert out.animal.bark == 'woof'
        assert out.animal.volume is None

    def test_dict_and_tuple_values_become_partials(self) -> None:
        class Item(BaseModel):
            name: str
            qty: int

        class Inventory(BaseModel):
            by_id: dict[str, Item]
            featured: tuple[Item, ...]

        out = cast(
            'Any',
            partial_model(Inventory).model_validate({
                'by_id': {'a': {'name': 'axe'}},
                'featured': [{'qty': 2}],
            }),
        )
        assert out.by_id['a'].name == 'axe'
        assert out.by_id['a'].qty is None
        assert out.featured[0].qty == 2
        assert out.featured[0].name is None

    def test_self_referential_model(self) -> None:
        class Node(BaseModel):
            name: str
            child: 'Node | None' = None

        partial = partial_model(Node)
        out = cast('Any', partial.model_validate({'name': 'root', 'child': {'child': {'name': 'leaf'}}}))
        assert out.name == 'root'
        assert out.child.name is None
        assert out.child.child.name == 'leaf'
        assert type(out.child).__name__ == 'NodePartial'

    def test_mutually_recursive_models(self) -> None:
        class Author(BaseModel):
            name: str
            posts: list['Post'] = []

        class Post(BaseModel):
            title: str
            author: Author | None = None

        Author.model_rebuild()
        out = cast('Any', partial_model(Author).model_validate({'name': 'a', 'posts': [{'author': {'posts': []}}]}))
        assert out.posts[0].title is None
        assert out.posts[0].author.name is None

    def test_partial_is_cached_per_class(self) -> None:
        assert partial_model(Recipe) is partial_model(Recipe)


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
async def test_generate_stream_with_output_schema_yields_partial_chunks() -> None:
    """End to end: generate_stream(output_schema=...) chunks are partials of the schema."""
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

    outputs: list[Any] = []
    async for chunk in stream_result.stream:
        outputs.append(chunk.output)

    # First chunk: title is already usable; steps has not started -> None.
    assert outputs[0] is not None
    assert not isinstance(outputs[0], Recipe)
    assert outputs[0].title == 'Chocolate C'
    assert outputs[0].steps is None
    # Later chunks fill in as keys arrive; still partials, never the real model.
    assert outputs[1].title == 'Chocolate Cake'
    assert outputs[1].steps == ['mi']
    assert outputs[2].steps == ['mix', 'bake']
    assert not isinstance(outputs[2], Recipe)

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
