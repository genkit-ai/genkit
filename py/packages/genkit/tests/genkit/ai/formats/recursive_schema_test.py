# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Recursive schemas retain their reference context in output format instructions."""

import copy
import json

import pytest
from jsonschema import Draft202012Validator, ValidationError
from pydantic import BaseModel, Field, TypeAdapter

from genkit import Genkit, Message, ModelResponse, Part
from genkit._ai._formats._array import ArrayFormat
from genkit._ai._formats._jsonl import JsonlFormat
from genkit.model import ModelContext, ModelRequest


class TreeNode(BaseModel):
    """A self-referential model."""

    name: str
    children: list['TreeNode'] = Field(default_factory=list)


class Branch(BaseModel):
    """One half of a mutually recursive pair."""

    leaves: list['Leaf'] = Field(default_factory=list)


class Leaf(BaseModel):
    """A leaf can contain another branch."""

    label: str
    parent: Branch | None = None


Branch.model_rebuild()


@pytest.mark.parametrize('formatter', [ArrayFormat, JsonlFormat])
@pytest.mark.parametrize(
    ('schema', 'data', 'invalid'),
    [
        pytest.param(
            TypeAdapter(list[TreeNode]).json_schema(),
            [{'name': 'root', 'children': [{'name': 'child'}]}],
            [{'name': 'root', 'children': [{'name': 42}]}],
            id='self-recursive',
        ),
        pytest.param(
            TypeAdapter(list[Branch]).json_schema(),
            [{'leaves': [{'label': 'outer', 'parent': {'leaves': [{'label': 'inner'}]}}]}],
            [{'leaves': [{'label': 'outer', 'parent': {'leaves': [{'label': 42}]}}]}],
            id='mutually-recursive',
        ),
    ],
)
def test_recursive_format_instructions(
    formatter: type[ArrayFormat] | type[JsonlFormat],
    schema: dict[str, object],
    data: list[dict[str, object]],
    invalid: list[dict[str, object]],
) -> None:
    """The emitted schema validates nested data and still rejects a wrong nested type."""
    original = copy.deepcopy(schema)
    handler = formatter().handle(schema)
    assert schema == original
    assert handler.instructions is not None
    instruction_schema = json.loads(handler.instructions.split('```')[1])
    validator = Draft202012Validator(instruction_schema)
    validator.validate(data)
    with pytest.raises(ValidationError):
        validator.validate(invalid)

    text = json.dumps(data) if formatter is ArrayFormat else '\n'.join(json.dumps(item) for item in data)
    assert handler.parse_message(Message(role='model', content=[Part.from_text(text)])) == data


@pytest.mark.asyncio
@pytest.mark.parametrize('output_format', ['array', 'jsonl'])
async def test_generate_with_recursive_output(output_format: str) -> None:
    """Recursive output schemas reach an in-process model and return nested objects."""
    ai = Genkit()
    requests: list[ModelRequest] = []
    data = [{'name': 'root', 'children': [{'name': 'child'}]}]

    async def tree_model(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        requests.append(request)
        text = json.dumps(data) if output_format == 'array' else json.dumps(data[0])
        return ModelResponse(message=Message(role='model', content=[Part.from_text(text)]))

    ai.define_model(name='recursiveTree', fn=tree_model)
    response = await ai.generate(
        model='recursiveTree',
        prompt='Return a tree.',
        output_format=output_format,
        output_schema=TypeAdapter(list[TreeNode]).json_schema(),
    )
    assert requests
    assert response.output == data
