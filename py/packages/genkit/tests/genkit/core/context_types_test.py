# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""FlowContext, ToolContext, and ModelContext: one class hierarchy, generic over the chunk type."""

import types
from typing import Any, get_args, get_origin

import pytest
from typing_extensions import assert_type

import genkit
import genkit.model
import genkit.plugin_api
from genkit import FlowContext, Genkit, ModelResponseChunk, ToolContext
from genkit._core._action import ActionKind, ActionRunContext, find_input_and_context
from genkit.model import ModelContext


def test_flow_context_is_action_run_context() -> None:
    """`FlowContext` is the same class as the internal `ActionRunContext` base."""
    assert FlowContext is ActionRunContext


def test_model_context_is_exported_from_genkit_model() -> None:
    """`ModelContext` is `ActionRunContext[ModelResponseChunk]`, exported from `genkit.model`."""
    assert 'ModelContext' in genkit.model.__all__
    assert get_origin(ModelContext) is ActionRunContext
    assert get_args(ModelContext) == (ModelResponseChunk,)


def test_action_run_context_is_not_publicly_exported() -> None:
    """`ActionRunContext` stays internal; public packages export `FlowContext`, `ToolContext`, and `ModelContext`."""
    assert 'ActionRunContext' not in genkit.__all__
    assert not hasattr(genkit, 'ActionRunContext')
    assert 'ActionRunContext' not in genkit.plugin_api.__all__
    assert not hasattr(genkit.plugin_api, 'ActionRunContext')
    assert 'ActionRunContext' not in genkit.model.__all__


def test_contexts_subscript_at_runtime() -> None:
    """`FlowContext[str]` and `ToolContext[int]` are real generic aliases."""
    assert get_origin(FlowContext[str]) is ActionRunContext
    assert get_origin(ToolContext[int]) is ToolContext
    assert issubclass(ToolContext, FlowContext)


def test_tool_context_is_a_flow_context_for_type_checkers() -> None:
    """A `ToolContext[int]` goes wherever a `FlowContext[int]` is expected; checked by pyright, pyrefly and ty."""

    def plate(ctx: FlowContext[int]) -> FlowContext[int]:
        ctx.send_chunk(3)
        return ctx

    tool_ctx: ToolContext[int] = ToolContext(FlowContext[int]())
    flow_ctx: FlowContext[str] = FlowContext[str]()

    assert_type(plate(tool_ctx), FlowContext[int])
    assert_type(flow_ctx.context, dict[str, Any])
    assert_type(tool_ctx.is_resumed(), bool)


@pytest.mark.parametrize(
    'annotation',
    [
        'FlowContext',
        'ToolContext',
        'ModelContext',
        'ActionRunContext',
        'FlowContext[str]',
        'ToolContext[int]',
        'FlowContext | None',
        'ModelContext | None',
        'Optional[ToolContext]',
        'Optional[ModelContext]',
        'genkit.FlowContext',
        'genkit.model.ModelContext',
        'genkit.ToolContext[int] | None',
    ],
)
def test_string_annotation_naming_a_context_marks_the_context(annotation: str) -> None:
    """A string naming FlowContext, ToolContext, or ModelContext is recognized as the context."""

    async def check_allergens(dish: str, ctx: object) -> str:
        return dish

    check_allergens.__annotations__['ctx'] = annotation
    params = find_input_and_context(check_allergens, {'dish': str}, kind=ActionKind.FLOW, name='check_allergens')

    assert params.input is not None and params.input.name == 'dish'
    assert params.context is not None and params.context.name == 'ctx'


@pytest.mark.parametrize('annotation', ['FlowContexts', 'Context', 'ToolContextual', 'ModelContexts', 'str'])
def test_string_annotation_not_naming_a_context_is_a_second_input(annotation: str) -> None:
    """Near-miss names are not the ctx, so a second parameter still raises."""

    async def check_allergens(dish: str, ctx: object) -> str:
        return dish

    check_allergens.__annotations__['ctx'] = annotation

    with pytest.raises(TypeError, match="'ctx' is a second parameter"):
        find_input_and_context(check_allergens, {'dish': str}, kind=ActionKind.FLOW, name='check_allergens')


_FUTURE_MODEL_CONTEXT_MODULE = """
from __future__ import annotations

from typing import TYPE_CHECKING

from genkit import Message, ModelResponse, Part, Role

if TYPE_CHECKING:
    from genkit.model import ModelContext, ModelRequest


async def future_chef(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
    table = ctx.context.get('table', 'walk-in')
    return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text(f'Tartine for {table}')]))
"""


@pytest.mark.asyncio
async def test_future_annotations_model_context_receives_context_at_runtime() -> None:
    """`from __future__ import annotations` with `ctx: ModelContext` on a model action receives context at runtime."""
    module = types.ModuleType('future_model_ctx')
    exec(_FUTURE_MODEL_CONTEXT_MODULE, module.__dict__)  # noqa: S102 - builds a module with postponed annotations

    ai = Genkit()
    action = ai.define_model(name='future-chef', fn=module.future_chef)

    assert action.params.context is not None and action.params.context.name == 'ctx'
    response = await ai.generate(model='future-chef', prompt='Suggest a dish.', context={'table': 7})
    assert response.text == 'Tartine for 7'
