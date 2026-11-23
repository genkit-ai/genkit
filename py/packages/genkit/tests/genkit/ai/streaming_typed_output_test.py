#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for typed streaming output (issue #6007).

Covers:
- ModelResponseChunk.output constructing the output schema from extracted JSON
  with optional defaults filled, closed subtrees validated, and missing required
  fields set to None
- ActionRunContext genericity over the chunk type
- End-to-end generate_stream with output_schema producing typed chunks
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, TypeVar, overload

import pytest
from pydantic import BaseModel, ConfigDict, Discriminator, Field, RootModel, Tag, field_validator, model_validator
from pydantic.alias_generators import to_camel

from genkit import Genkit, Message, ModelResponse, ModelResponseChunk, Part
from genkit._core import _partial
from genkit._core._action import ActionRunContext
from genkit._core._model import chunk_for_stream
from genkit._core._typing import Role
from genkit.formats import FormatHandler, Formatter, FormatterConfig
from genkit.testing import define_scripted_model

OutputT = TypeVar('OutputT', bound=BaseModel)


class Recipe(BaseModel):
    """Test output schema."""

    title: str
    steps: list[str]


class Author(BaseModel):
    name: str
    posts: list['Post'] = []


class Post(BaseModel):
    title: str
    author: Author | None = None


Author.model_rebuild()


class QuotedItem(BaseModel):
    name: str
    qty: int


class QuotedInventory(BaseModel):
    by_id: dict[str, 'QuotedItem']
    featured: tuple['QuotedItem', ...]


class NodeTree(BaseModel):
    name: str
    children: list['NodeTree'] = []


class UnknownQuotedList(BaseModel):
    items: list['NoSuchModel'] = []  # noqa: F821  # ty: ignore[unresolved-reference]


@overload
def _chunk(text: str, schema_type: type[OutputT]) -> ModelResponseChunk[OutputT]: ...
@overload
def _chunk(text: str, schema_type: None = None) -> ModelResponseChunk[object]: ...
def _chunk(text: str, schema_type: type[BaseModel] | None = None) -> ModelResponseChunk[Any]:
    """Build a chunk whose accumulated text is exactly ``text``."""
    return chunk_for_stream(
        ModelResponseChunk(
            role='model',
            content=[Part.from_text(text)],
        ),
        schema_type=schema_type,
    )


class TestChunkPartialOutput:
    """chunk.output with a schema type constructs that class from extracted JSON."""

    def test_preamble_returns_none(self) -> None:
        # No JSON object has started yet.
        assert _chunk('Here is the JSON:', schema_type=Recipe).output is None
        assert _chunk('Here is the JSON:').output is None

    def test_prefix_with_no_fields_is_empty_instance(self) -> None:
        # The object has started but no key has finished; all fields None.
        out = _chunk('{"ti', schema_type=Recipe).output
        assert isinstance(out, Recipe)
        assert out.title is None
        assert out.steps is None

    def test_first_field_is_available_immediately(self) -> None:
        out = _chunk('{"title": "Chocolate C', schema_type=Recipe).output
        assert isinstance(out, Recipe)
        assert out.title == 'Chocolate C'
        assert out.steps is None

    def test_partial_trailing_value(self) -> None:
        out = _chunk('{"title": "Chocolate Cake", "steps": ["mi', schema_type=Recipe).output
        assert isinstance(out, Recipe)
        assert out.title == 'Chocolate Cake'
        assert out.steps == ['mi']

    def test_complete_json_chunk_returns_validated_instance(self) -> None:
        # Once the root JSON object closes, chunk.output validates into Recipe.
        out = _chunk('{"title": "Cake", "steps": ["mix", "bake"]}', schema_type=Recipe).output
        assert isinstance(out, Recipe)
        assert out.steps == ['mix', 'bake']

    def test_no_schema_type_preserves_raw_json_behavior(self) -> None:
        out = _chunk('{"title": "Cake", "steps": ["mix"]}').output
        assert out == {'title': 'Cake', 'steps': ['mix']}
        assert not isinstance(out, Recipe)

    def test_constructs_chunk_parser_result(self) -> None:
        wrapper: ModelResponseChunk[Recipe] = chunk_for_stream(
            ModelResponseChunk(
                role='model',
                content=[Part.from_text('ignored')],
            ),
            chunk_parser=lambda _c: {'title': 'Parsed', 'steps': ['a']},
            schema_type=Recipe,
        )
        out = wrapper.output
        assert isinstance(out, Recipe)
        assert out.title == 'Parsed'

    def test_non_dict_parse_result_passes_through(self) -> None:
        # A scalar/array payload can't be constructed into an object schema;
        # it is returned as-is rather than silently dropped.
        assert _chunk('[1, 2, 3]', schema_type=Recipe).output == [1, 2, 3]

    def test_wrong_typed_value_does_not_raise(self) -> None:
        # Failed validation falls back to the raw value instead of crashing
        # the caller's loop; the final response surfaces the ValidationError.
        out = _chunk('{"title": 123}', schema_type=Recipe).output
        assert isinstance(out, Recipe)
        assert out.title == 123
        assert out.steps is None

    def test_camel_case_alias_populates_python_field(self) -> None:
        class UserProfile(BaseModel):
            model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
            first_name: str
            last_name: str

        out = _chunk('{"firstName": "Ada"', schema_type=UserProfile).output
        assert isinstance(out, UserProfile)
        assert out.first_name == 'Ada'
        assert out.last_name is None

    def test_explicit_field_alias_populates_python_field(self) -> None:
        class User(BaseModel):
            first_name: str = Field(alias='firstName')

        out = _chunk('{"firstName": "Ada"}', schema_type=User).output
        assert isinstance(out, User)
        assert out.first_name == 'Ada'

    def test_root_model_dict_stays_extracted_json(self) -> None:
        class DictRoot(RootModel[dict[str, int]]):
            pass

        out = _chunk('{"a": 1, "b": 2}', schema_type=DictRoot).output
        assert out == {'a': 1, 'b': 2}


class TestNestedAndConstrainedOutput:
    """Nested models and constraint fallbacks on a streaming chunk."""

    def test_failed_constraints_and_validators_fall_back_to_raw(self) -> None:
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

        out = _chunk(
            '{"servings": -5, "rating": 99, "title": "lowercase"}',
            schema_type=Strict,
        ).output
        assert isinstance(out, Strict)
        assert out.servings == -5
        assert out.rating == 99
        assert out.title == 'lowercase'

    def test_union_member_is_the_matching_class(self) -> None:
        class Cat(BaseModel):
            meow: str

        class Dog(BaseModel):
            bark: str
            volume: int

        class Pet(BaseModel):
            animal: Cat | Dog

        out = _chunk('{"animal": {"bark": "woof"}}', schema_type=Pet).output
        assert isinstance(out, Pet)
        assert isinstance(out.animal, Dog)
        assert out.animal.bark == 'woof'
        assert out.animal.volume is None

    def test_sequence_and_mapping_values_are_constructed(self) -> None:
        class Step(BaseModel):
            title: str
            duration: int

        class Item(BaseModel):
            name: str
            qty: int

        class Plan(BaseModel):
            steps: Sequence[Step]
            by_id: Mapping[str, Item]

        out = _chunk(
            '{"steps": [{"title": "mix"}], "by_id": {"a": {"name": "axe"}}}',
            schema_type=Plan,
        ).output
        assert isinstance(out, Plan)
        assert isinstance(out.steps[0], Step)
        assert out.steps[0].title == 'mix'
        assert out.steps[0].duration is None
        assert isinstance(out.by_id['a'], Item)
        assert out.by_id['a'].name == 'axe'
        assert out.by_id['a'].qty is None

    def test_dict_and_tuple_values_are_constructed(self) -> None:
        class Item(BaseModel):
            name: str
            qty: int

        class Inventory(BaseModel):
            by_id: dict[str, Item]
            featured: tuple[Item, ...]

        out = _chunk(
            '{"by_id": {"a": {"name": "axe"}}, "featured": [{"qty": 2}]}',
            schema_type=Inventory,
        ).output
        assert isinstance(out, Inventory)
        assert out.by_id['a'].name == 'axe'
        assert out.by_id['a'].qty is None
        assert out.featured[0].qty == 2
        assert out.featured[0].name is None

    def test_self_referential_model(self) -> None:
        class Node(BaseModel):
            name: str
            child: 'Node | None' = None

        out = _chunk(
            '{"name": "root", "child": {"child": {"name": "leaf"}}}',
            schema_type=Node,
        ).output
        assert isinstance(out, Node)
        assert out.name == 'root'
        assert isinstance(out.child, Node)
        assert out.child.name is None
        assert isinstance(out.child.child, Node)
        assert out.child.child.name == 'leaf'

    def test_mutually_recursive_models(self) -> None:
        """A quoted sibling in list['Post'] is a Post, with holes as None."""
        out = _chunk(
            '{"name": "a", "posts": [{"author": {"posts": []}}]}',
            schema_type=Author,
        ).output
        assert isinstance(out, Author)
        assert isinstance(out.posts[0], Post)
        assert out.posts[0].title is None
        assert isinstance(out.posts[0].author, Author)
        assert out.posts[0].author.name is None

    def test_half_arrived_nested_post_is_post_with_prefix_title(self) -> None:
        """A nested object cut mid-string is a Post; the prefix is title, missing fields are None."""
        out = _chunk(
            '{"posts": [{"title": "Hel',
            schema_type=Author,
        ).output
        assert isinstance(out, Author)
        assert out.name is None
        assert isinstance(out.posts[0], Post)
        assert out.posts[0].title == 'Hel'
        assert out.posts[0].author is None

    def test_half_arrived_nested_author_fills_name_prefix(self) -> None:
        """A nested object cut mid-string is an Author; missing optional fields get defaults."""
        out = _chunk(
            '{"posts": [{"author": {"name": "Jo',
            schema_type=Author,
        ).output
        assert isinstance(out, Author)
        assert out.name is None
        assert isinstance(out.posts[0], Post)
        assert out.posts[0].title is None
        assert isinstance(out.posts[0].author, Author)
        assert out.posts[0].author.name == 'Jo'
        assert out.posts[0].author.posts == []

    def test_half_arrived_empty_nested_post_has_holes(self) -> None:
        """A nested object that has only opened is a Post with every field None."""
        out = _chunk(
            '{"posts": [{',
            schema_type=Author,
        ).output
        assert isinstance(out, Author)
        assert out.name is None
        assert isinstance(out.posts[0], Post)
        assert out.posts[0].title is None
        assert out.posts[0].author is None

    def test_half_arrived_nested_key_without_value_is_none(self) -> None:
        """A nested key whose value has not arrived is None; keys that arrived stay."""
        out = _chunk(
            '{"posts": [{"title": "Hel", "author":',
            schema_type=Author,
        ).output
        assert isinstance(out, Author)
        assert out.name is None
        assert isinstance(out.posts[0], Post)
        assert out.posts[0].title == 'Hel'
        assert out.posts[0].author is None

    def test_half_arrived_deep_child_name_is_prefix(self) -> None:
        """A deeply nested object cut mid-string is a NodeTree; missing optional fields get defaults."""
        out = _chunk(
            '{"children": [{"children": [{"name": "lea',
            schema_type=NodeTree,
        ).output
        assert isinstance(out, NodeTree)
        assert out.name is None
        assert isinstance(out.children[0], NodeTree)
        assert out.children[0].name is None
        assert isinstance(out.children[0].children[0], NodeTree)
        assert out.children[0].children[0].name == 'lea'
        assert out.children[0].children[0].children == []

    def test_dict_quoted_post_yields_post_instances(self) -> None:
        """dict[str, 'QuotedItem'] values are QuotedItem, with holes as None."""
        out = _chunk(
            '{"by_id": {"a": {"name": "axe"}}, "featured": []}',
            schema_type=QuotedInventory,
        ).output
        assert isinstance(out, QuotedInventory)
        assert isinstance(out.by_id['a'], QuotedItem)
        assert out.by_id['a'].name == 'axe'
        assert out.by_id['a'].qty is None

    def test_tuple_quoted_post_yields_post_instances(self) -> None:
        """tuple['QuotedItem', ...] values are QuotedItem, with holes as None."""
        out = _chunk(
            '{"by_id": {}, "featured": [{"qty": 2}]}',
            schema_type=QuotedInventory,
        ).output
        assert isinstance(out, QuotedInventory)
        assert isinstance(out.featured[0], QuotedItem)
        assert out.featured[0].qty == 2
        assert out.featured[0].name is None

    def test_list_quoted_self_yields_node_instances(self) -> None:
        """list['NodeTree'] children are NodeTree, with holes as None."""
        out = _chunk(
            '{"name": "root", "children": [{"children": [{"name": "leaf"}]}]}',
            schema_type=NodeTree,
        ).output
        assert isinstance(out, NodeTree)
        assert out.name == 'root'
        assert isinstance(out.children[0], NodeTree)
        assert out.children[0].name is None
        assert isinstance(out.children[0].children[0], NodeTree)
        assert out.children[0].children[0].name == 'leaf'

    def test_list_quoted_unknown_name_stays_dicts(self) -> None:
        """A quoted name that is not a model stays extracted JSON; the loop does not crash."""
        out = _chunk(
            '{"items": [{"name": "x"}]}',
            schema_type=UnknownQuotedList,
        ).output
        assert isinstance(out, UnknownQuotedList)
        assert out.items[0] == {'name': 'x'}


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
        assert ActionRunContext[Recipe] is not None


@pytest.mark.asyncio
async def test_generate_stream_with_output_schema_yields_typed_chunks() -> None:
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    final_text = '{"title": "Chocolate Cake", "steps": ["mix", "bake"]}'
    pm.chunks = [
        [
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('{"title": "Chocolate C')]),
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('ake", "steps": ["mi')]),
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('x", "bake"]}')]),
        ]
    ]
    pm.responses = [
        ModelResponse(
            message=Message(role=Role.MODEL, content=[Part.from_text(final_text)]),
        )
    ]

    stream_result = ai.generate_stream(prompt='hi', output_schema=Recipe)

    outputs: list[Any] = []
    async for chunk in stream_result.stream:
        outputs.append(chunk.output)

    assert isinstance(outputs[0], Recipe)
    assert outputs[0].title == 'Chocolate C'
    assert outputs[0].steps is None
    assert isinstance(outputs[1], Recipe)
    assert outputs[1].title == 'Chocolate Cake'
    assert outputs[1].steps == ['mi']
    assert isinstance(outputs[2], Recipe)
    assert outputs[2].steps == ['mix', 'bake']

    response = await stream_result.response
    assert isinstance(response.output, Recipe)
    assert response.output.title == 'Chocolate Cake'
    assert response.output.steps == ['mix', 'bake']


@pytest.mark.asyncio
async def test_generate_stream_without_schema_chunks_unchanged() -> None:
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('{"a": 1}')])]]
    pm.responses = [
        ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('{"a": 1}')])),
    ]

    stream_result = ai.generate_stream(prompt='hi')
    async for chunk in stream_result.stream:
        assert chunk.output == {'a': 1}
    await stream_result.response


@pytest.mark.asyncio
async def test_generate_stream_camel_case_alias_fills_fields() -> None:
    class UserProfile(BaseModel):
        model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
        first_name: str
        last_name: str

    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    final_text = '{"firstName": "Ada", "lastName": "Lovelace"}'
    pm.chunks = [
        [
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('{"firstName": "Ada"')]),
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text(', "lastName": "Lovelace"}')]),
        ]
    ]
    pm.responses = [
        ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text(final_text)])),
    ]

    stream_result = ai.generate_stream(prompt='hi', output_schema=UserProfile)
    outputs: list[Any] = []
    async for chunk in stream_result.stream:
        outputs.append(chunk.output)

    assert isinstance(outputs[0], UserProfile)
    assert outputs[0].first_name == 'Ada'
    assert outputs[0].last_name is None
    assert outputs[1].first_name == 'Ada'
    assert outputs[1].last_name == 'Lovelace'

    response = await stream_result.response
    assert isinstance(response.output, UserProfile)
    assert response.output.first_name == 'Ada'
    assert response.output.last_name == 'Lovelace'


@pytest.mark.asyncio
async def test_generate_stream_dict_schema_chunks_stay_dicts() -> None:
    """A dict output_schema leaves chunk.output as extracted JSON, not a class."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    schema = {'type': 'object', 'properties': {'title': {'type': 'string'}}}
    pm.chunks = [
        [
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('{"title": "Chocolate C')]),
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('ake"}')]),
        ]
    ]
    pm.responses = [
        ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('{"title": "Chocolate Cake"}')])),
    ]

    stream_result = ai.generate_stream(prompt='hi', output_schema=schema)
    outputs: list[Any] = []
    async for chunk in stream_result.stream:
        outputs.append(chunk.output)

    assert outputs[0] == {'title': 'Chocolate C'}
    assert isinstance(outputs[0], dict)
    assert outputs[1] == {'title': 'Chocolate Cake'}
    response = await stream_result.response
    assert response.output == {'title': 'Chocolate Cake'}
    assert isinstance(response.output, dict)


@pytest.mark.asyncio
async def test_define_prompt_stream_yields_typed_chunks() -> None:
    """define_prompt(...).stream() yields the same Recipe holes as generate_stream."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    final_text = '{"title": "Chocolate Cake", "steps": ["mix", "bake"]}'
    pm.chunks = [
        [
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('{"title": "Chocolate C')]),
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('ake", "steps": ["mi')]),
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('x", "bake"]}')]),
        ]
    ]
    pm.responses = [
        ModelResponse(
            message=Message(role=Role.MODEL, content=[Part.from_text(final_text)]),
        )
    ]

    recipe_prompt = ai.define_prompt(prompt='hi', output_schema=Recipe)
    stream_result = recipe_prompt.stream()

    outputs: list[Any] = []
    async for chunk in stream_result.stream:
        outputs.append(chunk.output)

    assert isinstance(outputs[0], Recipe)
    assert outputs[0].title == 'Chocolate C'
    assert outputs[0].steps is None
    assert isinstance(outputs[1], Recipe)
    assert outputs[1].title == 'Chocolate Cake'
    assert outputs[1].steps == ['mi']
    assert isinstance(outputs[2], Recipe)
    assert outputs[2].steps == ['mix', 'bake']

    response = await stream_result.response
    assert isinstance(response.output, Recipe)
    assert response.output.title == 'Chocolate Cake'
    assert response.output.steps == ['mix', 'bake']


async def _stream_outputs(
    chunk_texts: list[str],
    final_text: str,
    **generate_kwargs: Any,  # noqa: ANN401
) -> tuple[list[Any], ModelResponse[Any]]:
    """Stream ``chunk_texts`` then ``final_text`` and collect every ``chunk.output``."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[Part.from_text(t)]) for t in chunk_texts]]
    pm.responses = [ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text(final_text)]))]

    stream_result = ai.generate_stream(prompt='hi', **generate_kwargs)
    outputs: list[Any] = []
    async for chunk in stream_result.stream:
        outputs.append(chunk.output)
    return outputs, await stream_result.response


@pytest.mark.asyncio
async def test_stream_chunk_output_is_partial_instance_of_schema_class() -> None:
    """With output_schema=Recipe, an early chunk `{"title": "Fluffy Panc` is a Recipe.

    Its title is the cut-short 'Fluffy Panc' and steps is None, with no validation.
    """
    outputs, _ = await _stream_outputs(
        ['{"title": "Fluffy Panc'],
        '{"title": "Fluffy Pancakes", "steps": ["mix"]}',
        output_schema=Recipe,
    )

    assert isinstance(outputs[0], Recipe)
    assert outputs[0].title == 'Fluffy Panc'
    assert outputs[0].steps is None


@pytest.mark.asyncio
async def test_stream_chunk_output_is_none_before_object_starts() -> None:
    """A chunk with prose and no `{` yet has `chunk.output is None`."""
    outputs, _ = await _stream_outputs(
        ['Sure, here is ', 'your recipe: {"title": "Pan'],
        'Sure, here is your recipe: {"title": "Pancakes", "steps": []}',
        output_schema=Recipe,
    )

    assert outputs[0] is None
    assert isinstance(outputs[1], Recipe)
    assert outputs[1].title == 'Pan'


@pytest.mark.asyncio
async def test_stream_chunk_output_skips_non_json_braces() -> None:
    """Chunks `{1, 2}` then ` then {"a": 1}` give chunk.output None then {'a': 1}."""
    outputs, _ = await _stream_outputs(['{1, 2}', ' then {"a": 1}'], '{1, 2} then {"a": 1}')

    assert outputs == [None, {'a': 1}]


@pytest.mark.asyncio
async def test_stream_chunk_output_schema_post_init_error_is_none() -> None:
    """generate_stream(output_schema=Recipe) whose model_post_init reads title.

    A chunk missing title gives chunk.output is None and the loop keeps going.
    """

    class RecipeNeedsTitle(BaseModel):
        title: str

        def model_post_init(self, __context: Any) -> None:
            _ = self.title.lower()

    outputs, response = await _stream_outputs(
        ['{"other": 1', ', "title": "X"}'],
        '{"other": 1, "title": "X"}',
        output_schema=RecipeNeedsTitle,
    )

    assert outputs[0] is None
    assert isinstance(outputs[1], RecipeNeedsTitle)
    assert outputs[1].title == 'X'
    assert response.output == RecipeNeedsTitle(title='X')
    assert response.error is None


@pytest.mark.asyncio
async def test_stream_chunk_output_custom_chunk_parser_error_is_none() -> None:
    """A custom format whose chunk parser raises gives `chunk.output is None` instead of ending the stream."""

    class BoomChunks(Formatter):
        def __init__(self) -> None:
            super().__init__('boom-chunks', FormatterConfig(format='json'))

        def handle(self, schema: dict[str, object] | None) -> FormatHandler[object, object]:
            def chunk_parser(_chunk: ModelResponseChunk) -> object:
                raise TypeError('chunk parser exploded')

            return FormatHandler(message_parser=lambda msg: msg.text, chunk_parser=chunk_parser, instructions=None)

    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    ai.define_format(BoomChunks())
    pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[Part.from_text(t)]) for t in ('a', 'b')]]
    pm.responses = [ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ab')]))]

    stream_result = ai.generate_stream(prompt='hi', output_format='boom-chunks', output_instructions=False)
    outputs = [chunk.output async for chunk in stream_result.stream]

    assert outputs == [None, None]
    assert (await stream_result.response).text == 'ab'


@pytest.mark.asyncio
async def test_stream_final_output_is_validated_instance() -> None:
    """After partial chunks, `(await stream.response).output` is a fully validated Recipe."""
    outputs, response = await _stream_outputs(
        ['{"title": "Fluffy Panc', 'akes", "steps": ["mix"]}'],
        '{"title": "Fluffy Pancakes", "steps": ["mix"]}',
        output_schema=Recipe,
    )

    assert outputs[0].steps is None
    assert response.output == Recipe.model_validate({'title': 'Fluffy Pancakes', 'steps': ['mix']})
    assert response.error is None


@pytest.mark.asyncio
async def test_stream_final_output_missing_field_is_none_after_partial_chunks() -> None:
    """Chunks showed a Recipe with steps None, but a final reply that never sends steps gives `output is None`."""
    outputs, response = await _stream_outputs(
        ['{"title": "Fluffy Pancakes"}'],
        '{"title": "Fluffy Pancakes"}',
        output_schema=Recipe,
    )

    assert isinstance(outputs[0], Recipe)
    assert outputs[0].steps is None
    assert response.output is None
    assert response.error is not None


@pytest.mark.asyncio
async def test_stream_cut_off_final_output_is_none() -> None:
    """A stream that ends on a cut-off object gives partial chunk outputs, but the final `output is None`."""
    outputs, response = await _stream_outputs(
        ['{"title": "Fluffy Pancakes", "steps": ["mi'],
        '{"title": "Fluffy Pancakes", "steps": ["mi',
        output_schema=Recipe,
    )

    assert isinstance(outputs[0], Recipe)
    assert outputs[0].steps == ['mi']
    assert response.output is None
    assert response.error is not None


class OrderStatus(str, Enum):
    PENDING = 'pending'
    SHIPPED = 'shipped'


class Ingredient(BaseModel):
    name: str
    grams: int

    @field_validator('name')
    @classmethod
    def _strip_name(cls, v: str) -> str:
        return v.strip()


class FullRecipe(BaseModel):
    title: str
    servings: int = 2
    status: OrderStatus
    tags: list[str] = []
    notes: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    ingredients: list[Ingredient]

    @field_validator('title')
    @classmethod
    def _strip_title(cls, v: str) -> str:
        return v.strip()


class CatPet(BaseModel):
    type: Literal['cat']
    meows: int


class DogPet(BaseModel):
    type: Literal['dog']
    barks: int


class CatfishPet(BaseModel):
    type: Literal['catfish']
    whiskers: int


PetUnion = Annotated[CatPet | DogPet, Field(discriminator='type')]


class TestDefaultsAndClosedSubtreeValidation:
    """Optional defaults, closed-subtree validation, and discriminated unions."""

    def test_optional_defaults_populate_and_do_not_share_mutable_state(self) -> None:
        first = _chunk('{"title": "Panc', schema_type=FullRecipe).output
        second = _chunk('{"title": "Pancakes', schema_type=FullRecipe).output

        assert isinstance(first, FullRecipe)
        assert isinstance(second, FullRecipe)
        assert first.servings == 2
        assert first.tags == []
        assert first.notes == []
        assert first.status is None
        assert first.ingredients is None

        first.tags.append('breakfast')
        first.notes.append('hot')
        assert second.tags == []
        assert second.notes == []

    def test_closed_enum_datetime_and_field_validator_run_mid_stream(self) -> None:
        # While status / created_at are trailing open keys, half-sent non-str scalars stay None.
        open_out = _chunk('{"title": "  Pancakes  ", "status": "pend', schema_type=FullRecipe).output
        assert isinstance(open_out, FullRecipe)
        assert open_out.title == 'Pancakes'
        assert open_out.status is None

        half_dt = _chunk(
            '{"title": "  Pancakes  ", "status": "pending", "created_at": "2026-04-',
            schema_type=FullRecipe,
        ).output
        assert isinstance(half_dt, FullRecipe)
        assert half_dt.status is OrderStatus.PENDING
        assert half_dt.created_at is None

        # Once ingredients starts, created_at is closed and becomes a real datetime.
        next_out = _chunk(
            '{"title": "  Pancakes  ", "status": "pending", "created_at": "2026-04-20T12:00:00Z", "ingredients": [',
            schema_type=FullRecipe,
        ).output
        assert isinstance(next_out, FullRecipe)
        assert next_out.status is OrderStatus.PENDING
        assert next_out.created_at == datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)

        class ModeOrder(BaseModel):
            mode: Literal['express', 'standard']

        open_mode = _chunk('{"mode": "exp', schema_type=ModeOrder).output
        assert isinstance(open_mode, ModeOrder)
        assert open_mode.mode is None
        closed_mode = _chunk('{"mode": "express"', schema_type=ModeOrder).output
        assert isinstance(closed_mode, ModeOrder)
        assert closed_mode.mode == 'express'

    def test_closed_earlier_list_items_validate_while_last_item_stays_partial(self) -> None:
        out = _chunk(
            '{"title": "Pancakes", "status": "pending", '
            '"ingredients": [{"name": "  flour  ", "grams": 200}, {"name": "  mi',
            schema_type=FullRecipe,
        ).output
        assert isinstance(out, FullRecipe)
        assert len(out.ingredients) == 2
        assert out.ingredients[0] == Ingredient(name='flour', grams=200)
        assert isinstance(out.ingredients[1], Ingredient)
        assert out.ingredients[1].name == '  mi'
        assert out.ingredients[1].grams is None

    def test_out_of_order_json_keys_track_wire_arrival_order(self) -> None:
        # 'ingredients' arrives before 'status', opposite of class field order.
        out = _chunk(
            '{"ingredients": [{"name": "  flour  ", "grams": 200}], "status": "pend',
            schema_type=FullRecipe,
        ).output
        assert isinstance(out, FullRecipe)
        assert out.ingredients == [Ingredient(name='flour', grams=200)]
        assert out.status is None
        assert out.title is None
        assert out.servings == 2

    def test_invalid_closed_field_falls_back_without_dropping_valid_siblings(self) -> None:
        class ConstrainedOrder(BaseModel):
            qty: int = Field(gt=0)
            status: OrderStatus
            note: str

        out = _chunk(
            '{"qty": -3, "status": "shipped", "note": "frag',
            schema_type=ConstrainedOrder,
        ).output
        assert isinstance(out, ConstrainedOrder)
        assert out.qty == -3
        assert out.status is OrderStatus.SHIPPED
        assert out.note == 'frag'

    def test_discriminated_union_waits_until_tag_matches(self) -> None:
        class Household(BaseModel):
            owner: str
            pet: PetUnion
            backup: PetUnion | None = None
            all_pets: list[PetUnion] = []

        # Half-typed tag 'do' -> None instead of guessing CatPet.
        c1 = _chunk('{"owner": "Ada", "pet": {"type": "do', schema_type=Household).output
        assert isinstance(c1, Household)
        assert c1.pet is None

        # Matched tag 'dog' with missing required barks -> DogPet(type='dog', barks=None).
        c2 = _chunk('{"owner": "Ada", "pet": {"type": "dog"', schema_type=Household).output
        assert isinstance(c2, Household)
        assert isinstance(c2.pet, DogPet)
        assert c2.pet.type == 'dog'
        assert c2.pet.barks is None

        # Optional discriminated union with half-typed tag stays None.
        c3 = _chunk(
            '{"owner": "Ada", "pet": {"type": "dog", "barks": 3}, "backup": {"type": "ca',
            schema_type=Household,
        ).output
        assert isinstance(c3, Household)
        assert c3.pet == DogPet(type='dog', barks=3)
        assert c3.backup is None

        # List of discriminated unions omits the trailing unresolved tag until it matches.
        c4 = _chunk(
            '{"owner": "Ada", "pet": {"type": "dog", "barks": 3}, '
            '"all_pets": [{"type": "dog", "barks": 1}, {"type": "ca',
            schema_type=Household,
        ).output
        assert isinstance(c4, Household)
        assert c4.all_pets == [DogPet(type='dog', barks=1)]

    def test_discriminated_union_waits_when_open_tag_is_prefix_of_another_tag(self) -> None:
        class Tank(BaseModel):
            animal: Annotated[CatPet | CatfishPet, Field(discriminator='type')]

        # While "type": "cat is open, 'cat' is still a prefix of 'catfish'.
        open_cat = _chunk('{"animal": {"type": "cat', schema_type=Tank).output
        assert isinstance(open_cat, Tank)
        assert open_cat.animal is None

        # Once the next key starts, 'cat' is closed and unambiguously CatPet.
        closed_tag = _chunk('{"animal": {"type": "cat", "meows": 2', schema_type=Tank).output
        assert isinstance(closed_tag, Tank)
        assert isinstance(closed_tag.animal, CatPet)
        assert closed_tag.animal.meows == 2

    def test_discriminated_union_dict_tag_reads_as_unknown(self) -> None:
        class Household(BaseModel):
            owner: str
            pet: PetUnion
            all_pets: list[PetUnion] = []

        # A dict tag can't match a Literal, so pet stays None instead of raising.
        out = _chunk('{"owner": "Ada", "pet": {"type": {"name": "dog"}', schema_type=Household).output
        assert isinstance(out, Household)
        assert out.owner == 'Ada'
        assert out.pet is None

        # Same inside a list: the item with a list tag is dropped, valid siblings stay.
        listed = _chunk(
            '{"owner": "Ada", "all_pets": [{"type": "dog", "barks": 1}, {"type": ["cat"]',
            schema_type=Household,
        ).output
        assert isinstance(listed, Household)
        assert listed.all_pets == [DogPet(type='dog', barks=1)]

    def test_annotated_with_no_args_does_not_hang(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # get_args() returning () used to spin the Annotated unwrap loops forever.
        monkeypatch.setattr(_partial, 'get_args', lambda _tp: ())
        tp = Annotated[int, 'meta']
        assert _partial.unwrap_annotated(annotation=tp) is tp
        assert _partial.extract_discriminator(annotation=tp) is None

    def test_callable_discriminator_with_tag_waits_until_closed(self) -> None:
        def pet_kind(v: Any) -> str | None:  # noqa: ANN401
            return v.get('kind') if isinstance(v, dict) else getattr(v, 'kind', None)

        class Cat(BaseModel):
            kind: Literal['cat']
            name: str
            lives: int = 9

        class Dog(BaseModel):
            kind: Literal['dog']
            name: str
            good: bool = True

        CallablePetUnion = Annotated[  # noqa: N806
            Annotated[Cat, Tag('cat')] | Annotated[Dog, Tag('dog')],
            Discriminator(pet_kind),
        ]

        class Owner(BaseModel):
            pet: CallablePetUnion
            pets: list[CallablePetUnion] = []
            owner_name: str = 'Ada'

        # Open chunks must stay None rather than guessing Cat by key overlap.
        c1 = _chunk('{"pet": {"kind": "dog", ', schema_type=Owner).output
        assert isinstance(c1, Owner)
        assert c1.pet is None

        c2 = _chunk('{"pet": {"kind": "dog", "name": "Rex"', schema_type=Owner).output
        assert isinstance(c2, Owner)
        assert c2.pet is None

        # Once a sibling key closes "pet" mid-stream, Pydantic resolves Dog via Discriminator(pet_kind).
        c3 = _chunk(
            '{"pet": {"kind": "dog", "name": "Rex", "good": false}, "owner_name": "Al',
            schema_type=Owner,
        ).output
        assert isinstance(c3, Owner)
        assert c3.pet == Dog(kind='dog', name='Rex', good=False)
        assert c3.owner_name == 'Al'

        # Once the root object closes, Owner validates cleanly into Dog.
        c4 = _chunk('{"pet": {"kind": "dog", "name": "Rex", "good": false}}', schema_type=Owner).output
        assert isinstance(c4, Owner)
        assert c4.pet == Dog(kind='dog', name='Rex', good=False)

        # In a list, closed items resolve via Discriminator(pet_kind) while the trailing open item waits.
        c5 = _chunk(
            '{"pet": {"kind": "dog", "name": "Rex"}, '
            '"pets": [{"kind": "dog", "name": "Buddy"}, {"kind": "cat", "name": "Mi',
            schema_type=Owner,
        ).output
        assert isinstance(c5, Owner)
        assert c5.pets == [Dog(kind='dog', name='Buddy', good=True)]

    def test_field_validators_run_mid_stream_without_firing_model_validators(self) -> None:
        before_calls = 0
        after_calls = 0

        class Booking(BaseModel):
            guest: str
            nights: int
            total: float

            @field_validator('guest')
            @classmethod
            def _upper_guest(cls, v: str) -> str:
                return v.upper()

            @model_validator(mode='before')
            @classmethod
            def _before_check(cls, data: Any) -> Any:  # noqa: ANN401
                nonlocal before_calls
                before_calls += 1
                return data

            @model_validator(mode='after')
            def _after_check(self) -> 'Booking':
                nonlocal after_calls
                after_calls += 1
                if self.total != self.nights * 100:
                    raise ValueError('total must equal nights * 100')
                return self

        # On chunk 1, 'guest' is still the last parsed key in data, so it streams raw;
        # once 'nights' arrives on chunk 2, 'guest' is closed and uppercased to 'ANN'.
        b1 = _chunk('{"guest": "ann", ', schema_type=Booking).output
        assert isinstance(b1, Booking)
        assert b1.guest == 'ann'
        assert b1.nights is None
        assert b1.total is None
        assert (before_calls, after_calls) == (0, 0)

        b2 = _chunk('{"guest": "ann", "nights": 2, ', schema_type=Booking).output
        assert isinstance(b2, Booking)
        assert b2.guest == 'ANN'
        assert b2.nights == 2
        assert b2.total is None
        assert (before_calls, after_calls) == (0, 0)

        b3 = _chunk('{"guest": "ann", "nights": 2, "total": 20', schema_type=Booking).output
        assert isinstance(b3, Booking)
        assert b3.guest == 'ANN'
        assert b3.nights == 2
        assert b3.total == 20
        assert (before_calls, after_calls) == (0, 0)

        b4 = _chunk('{"guest": "ann", "nights": 2, "total": 200.0}', schema_type=Booking).output
        assert (before_calls, after_calls) == (1, 1)
        assert b4 == Booking(guest='ANN', nights=2, total=200.0)

        # If the closed root violates @model_validator, fallback still preserves @field_validator outputs.
        b_invalid = _chunk('{"guest": "ann", "nights": 2, "total": 999.0}', schema_type=Booking).output
        assert isinstance(b_invalid, Booking)
        assert b_invalid.guest == 'ANN'
        assert b_invalid.nights == 2
        assert b_invalid.total == 999.0

    def test_open_numbers_and_invalid_closed_enum_match_documented_edge_cases(self) -> None:
        class EventPayload(BaseModel):
            at: datetime | None = None
            code: Literal[1, 12] | None = None
            c: OrderStatus | None = None
            n: int | None = None

        open_dt = _chunk('{"at": 17000', schema_type=EventPayload).output
        assert isinstance(open_dt, EventPayload)
        assert open_dt.at == 17000

        open_num_lit = _chunk('{"code": 1', schema_type=EventPayload).output
        assert isinstance(open_num_lit, EventPayload)
        assert open_num_lit.code == 1

        invalid_closed_enum = _chunk('{"c": "blue", "n": 1', schema_type=EventPayload).output
        assert isinstance(invalid_closed_enum, EventPayload)
        assert invalid_closed_enum.c == 'blue'
        assert invalid_closed_enum.n == 1


@pytest.mark.asyncio
async def test_generate_stream_progressively_validates_closed_subtrees() -> None:
    outputs, response = await _stream_outputs(
        [
            '{"title": "Pancakes", "status": "p',
            'ending", "ingredients": [{"name": "f',
            'lour", "grams": 200}, {"name": "milk", "grams": ',
            '150}]}',
        ],
        '{"title": "Pancakes", "status": "pending", '
        '"ingredients": [{"name": "flour", "grams": 200}, {"name": "milk", "grams": 150}]}',
        output_schema=FullRecipe,
    )

    assert outputs[0].title == 'Pancakes'
    assert outputs[0].servings == 2
    assert outputs[0].tags == []
    assert outputs[0].status is None
    assert outputs[0].ingredients is None

    assert outputs[1].status is OrderStatus.PENDING
    assert len(outputs[1].ingredients) == 1
    assert outputs[1].ingredients[0].name == 'f'
    assert outputs[1].ingredients[0].grams is None

    assert outputs[2].ingredients[0] == Ingredient(name='flour', grams=200)
    assert outputs[2].ingredients[1].name == 'milk'
    assert outputs[2].ingredients[1].grams is None

    assert outputs[3] == response.output
    assert outputs[3].status is OrderStatus.PENDING
    assert outputs[3].ingredients == [
        Ingredient(name='flour', grams=200),
        Ingredient(name='milk', grams=150),
    ]
