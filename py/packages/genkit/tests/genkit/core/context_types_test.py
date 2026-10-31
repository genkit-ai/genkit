# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""FlowContext, ToolContext, ModelContext, and AgentContext: subclasses of ActionRunContext."""

import types
from typing import Any, cast, get_args, get_origin

import pytest
from typing_extensions import assert_type

import genkit
import genkit.exp.agent
import genkit.model
import genkit.plugin_api
from genkit import (
    Document,
    Embedding,
    FlowContext,
    Genkit,
    Message,
    ModelResponse,
    ModelResponseChunk,
    Operation,
    Part,
    Role,
    ToolContext,
)
from genkit._core._action import ActionKind, ActionRunContext, find_input_and_context
from genkit.embedder import EmbedRequest, EmbedResponse
from genkit.exp import Genkit as ExpGenkit
from genkit.exp.agent import (
    AgentContext,
    AgentFinishReason,
    AgentInput,
    AgentResult,
    AgentStreamChunk,
    InMemorySessionStore,
    SessionRunner,
    TurnContext,
    TurnResult,
)
from genkit.model import ModelContext, ModelInfo, ModelRequest, ModelSupports


def test_public_contexts_are_distinct_subclasses_of_action_run_context() -> None:
    """FlowContext, ModelContext, ToolContext, and AgentContext are distinct subclasses of ActionRunContext."""
    public_classes = (FlowContext, ModelContext, ToolContext, AgentContext)
    for cls in public_classes:
        assert cls is not ActionRunContext
        assert issubclass(cls, ActionRunContext)

    for cls_a in public_classes:
        for cls_b in public_classes:
            if cls_a is not cls_b:
                assert not issubclass(cls_a, cls_b)


def test_model_context_is_exported_from_genkit_model() -> None:
    """`ModelContext` subclasses `ActionRunContext[ModelResponseChunk]` and is exported from `genkit.model`."""
    assert 'ModelContext' in genkit.model.__all__
    assert issubclass(ModelContext, ActionRunContext)
    orig_bases = cast(tuple[Any, ...], getattr(ModelContext, '__orig_bases__', ()))
    assert len(orig_bases) == 1
    assert get_origin(orig_bases[0]) is ActionRunContext
    assert get_args(orig_bases[0]) == (ModelResponseChunk,)


def test_model_context_has_one_definition() -> None:
    """`genkit.model.ModelContext` and the model module's import are the same class."""
    from genkit._ai import _model
    from genkit._core import _action

    assert _model.ModelContext is _action.ModelContext
    assert genkit.model.ModelContext is _action.ModelContext


def test_isinstance_works_and_distinguishes_each_context_class() -> None:
    """`isinstance` works at runtime for every context class without raising `TypeError`."""
    instances: list[tuple[ActionRunContext[Any], type[ActionRunContext[Any]]]] = [
        (FlowContext(), FlowContext),
        (ModelContext(), ModelContext),
        (ToolContext(ActionRunContext()), ToolContext),
        (AgentContext(), AgentContext),
    ]

    for ctx, expected_cls in instances:
        assert isinstance(ctx, ActionRunContext)
        assert isinstance(ctx, expected_cls)
        for _, other_cls in instances:
            if other_cls is not expected_cls:
                assert not isinstance(ctx, other_cls)


def test_action_run_context_is_not_publicly_exported() -> None:
    """`ActionRunContext` stays internal; public packages export only concrete subclasses."""
    assert 'ActionRunContext' not in genkit.__all__
    assert not hasattr(genkit, 'ActionRunContext')
    assert 'ActionRunContext' not in genkit.plugin_api.__all__
    assert not hasattr(genkit.plugin_api, 'ActionRunContext')
    assert 'ActionRunContext' not in genkit.model.__all__
    assert 'ActionRunContext' not in genkit.exp.agent.__all__


def test_contexts_subscript_at_runtime() -> None:
    """`FlowContext[str]` and `ToolContext[int]` subscript cleanly with their own class as origin."""
    assert get_origin(FlowContext[str]) is FlowContext
    assert get_args(FlowContext[str]) == (str,)
    assert get_origin(ToolContext[int]) is ToolContext
    assert get_args(ToolContext[int]) == (int,)


def test_context_subclasses_satisfy_action_run_context_for_type_checkers() -> None:
    """Each context subclass satisfies `ActionRunContext[ChunkT]`; checked by pyright, pyrefly and ty."""

    def stream_step(ctx: ActionRunContext[int]) -> ActionRunContext[int]:
        ctx.send_chunk(3)
        return ctx

    def stream_model_chunk(ctx: ActionRunContext[ModelResponseChunk]) -> ActionRunContext[ModelResponseChunk]:
        return ctx

    def stream_agent_chunk(ctx: ActionRunContext[AgentStreamChunk]) -> ActionRunContext[AgentStreamChunk]:
        return ctx

    flow_ctx: FlowContext[int] = FlowContext[int]()
    tool_ctx: ToolContext[int] = ToolContext(ActionRunContext[int]())
    model_ctx: ModelContext = ModelContext()
    agent_ctx: AgentContext = AgentContext()

    assert_type(stream_step(flow_ctx), ActionRunContext[int])
    assert_type(stream_step(tool_ctx), ActionRunContext[int])
    assert_type(stream_model_chunk(model_ctx), ActionRunContext[ModelResponseChunk])
    assert_type(stream_agent_chunk(agent_ctx), ActionRunContext[AgentStreamChunk])
    assert_type(flow_ctx.context, dict[str, Any])
    assert_type(tool_ctx.is_resumed(), bool)
    assert_type(model_ctx.is_streaming, bool)
    assert_type(agent_ctx.is_streaming, bool)


@pytest.mark.asyncio
async def test_runtime_actions_deliver_matching_context_subclasses() -> None:
    """Flows, models, background models, tools, embedders, and agents receive their concrete context subclass."""
    ai = Genkit()
    seen: dict[str, object] = {}

    @ai.flow()
    async def plate_dish(dish: str, ctx: FlowContext[str]) -> str:
        seen['flow'] = ctx
        ctx.send_chunk('searing')
        return f'{dish} ({ctx.context.get("table", "walk-in")})'

    @ai.tool()
    async def check_allergens(dish: str, ctx: ToolContext) -> str:
        seen['tool'] = ctx
        return f'{dish}: nut-free'

    call_count = 0

    async def chef(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        seen['model'] = ctx
        if call_count == 1:
            return ModelResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part.from_tool_request(name='check_allergens', input='salmon')],
                )
            )
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('Smoked Salmon Tartine')]))

    ai.define_model(
        name='bistro/chef',
        fn=chef,
        info=ModelInfo(supports=ModelSupports(tools=True, multiturn=True)),
    )

    async def start_video(_request: ModelRequest, ctx: ModelContext) -> Operation:
        seen['bg_start'] = ctx
        return Operation(id='op-42', done=False)

    async def check_video(op: Operation[ModelResponse], ctx: ModelContext) -> Operation:
        seen['bg_check'] = ctx
        return Operation(
            id=op.id,
            done=True,
            output=ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('plated video')])),
        )

    async def cancel_video(op: Operation[ModelResponse], ctx: ModelContext) -> Operation:
        seen['bg_cancel'] = ctx
        return Operation(id=op.id, done=True)

    ai.define_background_model(
        name='bistro/menu-video',
        start=start_video,
        check=check_video,
        cancel=cancel_video,
    )

    async def menu_embedder(request: EmbedRequest, ctx: FlowContext) -> EmbedResponse:
        seen['embedder'] = ctx
        return EmbedResponse(embeddings=[Embedding(embedding=[0.25, 0.75]) for _ in request.input])

    ai.define_embedder(name='bistro/menu-embedder', fn=menu_embedder)

    # 1. Flow
    assert await plate_dish('salmon', context={'table': 7}) == 'salmon (7)'
    assert isinstance(seen['flow'], FlowContext)
    assert type(seen['flow']) is FlowContext

    # 2. Model + Tool inside generate()
    gen_resp = await ai.generate(model='bistro/chef', prompt='Suggest a dish.', tools=[check_allergens])
    assert gen_resp.text == 'Smoked Salmon Tartine'
    assert isinstance(seen['model'], ModelContext)
    assert type(seen['model']) is ModelContext
    assert isinstance(seen['tool'], ToolContext)
    assert type(seen['tool']) is ToolContext

    # 3. Background model (start / check / cancel)
    op = await ai.generate_operation(model='bistro/menu-video', prompt='Film the kitchen.')
    checked = await ai.check_operation(op)
    assert checked.done is True
    cancelled = await ai.cancel_operation(op)
    assert cancelled.done is True
    assert isinstance(seen['bg_start'], ModelContext)
    assert type(seen['bg_start']) is ModelContext
    assert isinstance(seen['bg_check'], ModelContext)
    assert type(seen['bg_check']) is ModelContext
    assert isinstance(seen['bg_cancel'], ModelContext)
    assert type(seen['bg_cancel']) is ModelContext

    # 4. Embedder
    embeddings = await ai.embed(embedder='bistro/menu-embedder', content=Document.from_text('Smoked Salmon Tartine'))
    assert embeddings[0].embedding == [0.25, 0.75]
    assert isinstance(seen['embedder'], FlowContext)
    assert type(seen['embedder']) is FlowContext

    # 5. Custom Agent
    exp_ai = ExpGenkit()

    async def concierge(sess: SessionRunner, ctx: AgentContext) -> AgentResult:
        seen['agent'] = ctx

        async def handle_turn(_: AgentInput, __: TurnContext) -> TurnResult | None:
            await sess.add_messages([Message(role=Role.MODEL, content=[Part.from_text('Table reserved')])])
            return TurnResult(finish_reason=AgentFinishReason.STOP)

        await sess.run(handle_turn)
        return await sess.result()

    agent = exp_ai.define_custom_agent(name='concierge', fn=concierge, store=InMemorySessionStore())
    turn = await agent.chat().send('Reserve table 7')
    assert turn.text == 'Table reserved'
    assert isinstance(seen['agent'], AgentContext)
    assert type(seen['agent']) is AgentContext


@pytest.mark.parametrize(
    'annotation',
    [
        'FlowContext',
        'ToolContext',
        'ModelContext',
        'AgentContext',
        'ActionRunContext',
        'FlowContext[str]',
        'ToolContext[int]',
        'FlowContext | None',
        'ModelContext | None',
        'AgentContext | None',
        'Optional[ToolContext]',
        'Optional[ModelContext]',
        'genkit.FlowContext',
        'genkit.model.ModelContext',
        'genkit.exp.agent.AgentContext',
        'genkit.ToolContext[int] | None',
    ],
)
def test_string_annotation_naming_a_context_marks_the_context(annotation: str) -> None:
    """A string naming FlowContext, ToolContext, ModelContext, or AgentContext is recognized as the context."""

    async def check_allergens(dish: str, ctx: object) -> str:
        return dish

    check_allergens.__annotations__['ctx'] = annotation
    params = find_input_and_context(check_allergens, {'dish': str}, kind=ActionKind.FLOW, name='check_allergens')

    assert params.input is not None and params.input.name == 'dish'
    assert params.context is not None and params.context.name == 'ctx'


@pytest.mark.parametrize(
    'annotation',
    ['FlowContexts', 'Context', 'ToolContextual', 'ModelContexts', 'AgentContexts', 'str'],
)
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
from genkit.model import ModelContext as RuntimeModelContext

if TYPE_CHECKING:
    from genkit.model import ModelContext, ModelRequest


async def future_chef(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
    assert isinstance(ctx, RuntimeModelContext)
    table = ctx.context.get('table', 'walk-in')
    return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text(f'Tartine for {table}')]))
"""


@pytest.mark.asyncio
async def test_future_annotations_model_context_receives_context_at_runtime() -> None:
    """`from __future__ import annotations` with `ctx: ModelContext` on a model action receives `ModelContext`."""
    module = types.ModuleType('future_model_ctx')
    exec(_FUTURE_MODEL_CONTEXT_MODULE, module.__dict__)  # noqa: S102 - builds a module with postponed annotations

    ai = Genkit()
    action = ai.define_model(name='future-chef', fn=module.future_chef)

    assert action.params.context is not None and action.params.context.name == 'ctx'
    response = await ai.generate(model='future-chef', prompt='Suggest a dish.', context={'table': 7})
    assert response.text == 'Tartine for 7'
