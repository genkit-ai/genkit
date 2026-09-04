#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for typed streaming output (issue #6007).

Covers:
- ModelResponseChunk.output constructing the output schema from extracted JSON
  with missing fields set to None
- ActionRunContext genericity over the chunk type
- End-to-end generate_stream with output_schema producing typed chunks
"""

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, TypeVar, overload

import pytest
from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator
from pydantic.alias_generators import to_camel

from genkit import Genkit, Message, ModelResponse, ModelResponseChunk
from genkit._ai._testing import define_programmable_model
from genkit._core._action import ActionRunContext
from genkit._core._typing import Part, Role, TextPart

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
    return ModelResponseChunk(
        role='model',
        content=[Part(TextPart(text=text))],
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

    def test_complete_json_is_still_unvalidated(self) -> None:
        # A complete-looking chunk is still constructed, not validated.
        # Only ModelResponse.output runs the real model.
        out = _chunk('{"title": "Cake", "steps": ["mix", "bake"]}', schema_type=Recipe).output
        assert isinstance(out, Recipe)
        assert out.steps == ['mix', 'bake']

    def test_no_schema_type_preserves_raw_json_behavior(self) -> None:
        out = _chunk('{"title": "Cake", "steps": ["mix"]}').output
        assert out == {'title': 'Cake', 'steps': ['mix']}
        assert not isinstance(out, Recipe)

    def test_constructs_chunk_parser_result(self) -> None:
        wrapper: ModelResponseChunk[Recipe] = ModelResponseChunk(
            role='model',
            content=[Part(TextPart(text='ignored'))],
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
        # Chunks skip validation, so a wrong-typed field is stored as-is
        # instead of crashing the caller's loop. The final response is
        # where the real ValidationError surfaces.
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
    """Nested models and dropped constraints on a streaming chunk."""

    def test_constraints_and_validators_are_skipped(self) -> None:
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    final_text = '{"title": "Chocolate Cake", "steps": ["mix", "bake"]}'
    pm.chunks = [
        [
            ModelResponseChunk(role=Role.MODEL, content=[Part(TextPart(text='{"title": "Chocolate C'))]),
            ModelResponseChunk(role=Role.MODEL, content=[Part(TextPart(text='ake", "steps": ["mi'))]),
            ModelResponseChunk(role=Role.MODEL, content=[Part(TextPart(text='x", "bake"]}'))]),
        ]
    ]
    pm.responses = [
        ModelResponse(
            message=Message(role=Role.MODEL, content=[Part(TextPart(text=final_text))]),
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[Part(TextPart(text='{"a": 1}'))])]]
    pm.responses = [
        ModelResponse(message=Message(role=Role.MODEL, content=[Part(TextPart(text='{"a": 1}'))])),
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

    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    final_text = '{"firstName": "Ada", "lastName": "Lovelace"}'
    pm.chunks = [
        [
            ModelResponseChunk(role=Role.MODEL, content=[Part(TextPart(text='{"firstName": "Ada"'))]),
            ModelResponseChunk(role=Role.MODEL, content=[Part(TextPart(text=', "lastName": "Lovelace"}'))]),
        ]
    ]
    pm.responses = [
        ModelResponse(message=Message(role=Role.MODEL, content=[Part(TextPart(text=final_text))])),
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
