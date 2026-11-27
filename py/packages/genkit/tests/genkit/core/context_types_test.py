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
from pydantic import BaseModel

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
    InMemorySessionStore,
    SessionRunner,
    TurnContext,
    TurnResult,
)
from genkit.model import ModelContext, ModelInfo, ModelRequest, ModelSupports
from genkit.plugin_api import BackgroundAction


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


@pytest.mark.asyncio
async def test_flow_and_tool_contexts_pass_shared_isinstance_check() -> None:
    """A shared helper narrows with `isinstance(ctx, (FlowContext, ToolContext))` on both contexts."""
    ai = Genkit()
    seen: list[object] = []

    def user_id(ctx: FlowContext | ToolContext) -> str:
        seen.append(ctx)
        return ctx.context['auth']['uid']

    @ai.flow()
    async def reorder_last(dish: str, ctx: FlowContext) -> str:
        return f'{dish} for {user_id(ctx)}'

    @ai.tool()
    async def check_allergies(dish: str, ctx: ToolContext) -> str:
        return f'{dish}: nut-free for {user_id(ctx)}'

    auth = {'auth': {'uid': 'diner-42'}}
    assert await reorder_last('tartine', context=auth) == 'tartine for diner-42'
    assert await check_allergies('tartine', context=auth) == 'tartine: nut-free for diner-42'

    flow_ctx, tool_ctx = seen
    assert type(flow_ctx) is FlowContext
    assert type(tool_ctx) is ToolContext
    for ctx in (flow_ctx, tool_ctx):
        assert isinstance(ctx, (FlowContext, ToolContext))
    assert not isinstance(ModelContext(), (FlowContext, ToolContext))
    assert not isinstance(AgentContext(), (FlowContext, ToolContext))


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


@pytest.mark.asyncio
async def test_runtime_actions_deliver_matching_context_subclasses() -> None:
    """Flows, models, background models, custom background jobs, tools, embedders, and agents get matching contexts."""
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

    class RenderRequest(BaseModel):
        quarter: str

    class Report(BaseModel):
        url: str

    class ReportJob(BackgroundAction[RenderRequest, Report]):
        pass

    async def start_report(request: RenderRequest, ctx: FlowContext) -> Operation:
        seen['custom_bg_start'] = ctx
        return Operation(id=f'report-{request.quarter}', done=False)

    async def check_report(op: Operation[Report], ctx: FlowContext) -> Operation:
        seen['custom_bg_check'] = ctx
        return Operation(id=op.id, done=True, output=Report(url=f'https://reports.example/{op.id}.pdf'))

    async def cancel_report(op: Operation[Report], ctx: FlowContext) -> Operation:
        seen['custom_bg_cancel'] = ctx
        return Operation(id=op.id, done=True)

    report_job = ReportJob(
        'render-report',
        start_report,
        check_report,
        kind=ActionKind.CUSTOM,
        cancel=cancel_report,
    )
    for each in report_job.actions:
        ai._registry.register_action_from_instance(each)

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

    # 3. Background model (start / check / cancel -> ModelContext)
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

    # 4. Custom BackgroundAction (start / check / cancel -> FlowContext)
    report_op = await report_job.start(RenderRequest(quarter='q3'))
    checked_report = await ai.check_operation(report_op)
    assert checked_report.done is True
    cancelled_report = await ai.cancel_operation(report_op)
    assert cancelled_report.done is True
    assert isinstance(seen['custom_bg_start'], FlowContext)
    assert type(seen['custom_bg_start']) is FlowContext
    assert isinstance(seen['custom_bg_check'], FlowContext)
    assert type(seen['custom_bg_check']) is FlowContext
    assert isinstance(seen['custom_bg_cancel'], FlowContext)
    assert type(seen['custom_bg_cancel']) is FlowContext

    # 5. Embedder
    embeddings = await ai.embed(embedder='bistro/menu-embedder', content=Document.from_text('Smoked Salmon Tartine'))
    assert embeddings[0].embedding == [0.25, 0.75]
    assert isinstance(seen['embedder'], FlowContext)
    assert type(seen['embedder']) is FlowContext

    # 6. Custom Agent
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


def test_background_check_signature_error_names_owner_context() -> None:
    """Signature fix-it hints on check/cancel name FlowContext for custom jobs and ModelContext for models."""
    ai = Genkit()

    class RenderRequest(BaseModel):
        quarter: str

    class Report(BaseModel):
        url: str

    class ReportJob(BackgroundAction[RenderRequest, Report]):
        pass

    async def start_report(request: RenderRequest, ctx: FlowContext) -> Operation:
        return Operation(id=request.quarter)

    async def bad_check_report(op: Operation[Report], extra: int) -> Operation:
        return op

    with pytest.raises(TypeError, match="annotate 'extra' as FlowContext"):
        ReportJob('render-report', start_report, cast(Any, bad_check_report), kind=ActionKind.CUSTOM)

    async def start_video(request: ModelRequest, ctx: ModelContext) -> Operation:
        return Operation(id='video-1')

    async def bad_check_video(op: Operation[ModelResponse], extra: int) -> Operation:
        return op

    with pytest.raises(TypeError, match="annotate 'extra' as ModelContext"):
        ai.define_background_model(name='bistro/menu-video', start=start_video, check=cast(Any, bad_check_video))


_FUTURE_BACKGROUND_JOB_MODULE = """
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from genkit import FlowContext as RuntimeFlowContext, Operation as RuntimeOperation
from genkit.plugin_api import ActionKind, BackgroundAction

if TYPE_CHECKING:
    from genkit import FlowContext, Operation


class RenderRequest(BaseModel):
    quarter: str


class Report(BaseModel):
    url: str


class ReportJob(BackgroundAction[RenderRequest, Report]):
    pass


async def start_report(request: RenderRequest, ctx: FlowContext) -> Operation:
    assert type(ctx) is RuntimeFlowContext
    return RuntimeOperation(id=f'report-{request.quarter}', done=False)


async def check_report(op: Operation[Report], ctx: FlowContext) -> Operation:
    assert type(ctx) is RuntimeFlowContext
    return RuntimeOperation(id=op.id, done=True, output=Report(url=f'https://reports.example/{op.id}.pdf'))


async def cancel_report(op: Operation[Report], ctx: FlowContext) -> Operation:
    assert type(ctx) is RuntimeFlowContext
    return RuntimeOperation(id=op.id, done=True)
"""


@pytest.mark.asyncio
async def test_future_annotations_custom_background_job_receives_flow_context() -> None:
    """`from __future__ import annotations` on a custom `BackgroundAction` resolves `Operation` and `FlowContext`."""
    module = types.ModuleType('future_bg_job')
    exec(_FUTURE_BACKGROUND_JOB_MODULE, module.__dict__)  # noqa: S102 - builds a module with postponed annotations

    ai = Genkit()
    job = module.ReportJob(
        'render-report',
        module.start_report,
        module.check_report,
        kind=ActionKind.CUSTOM,
        cancel=module.cancel_report,
    )
    for each in job.actions:
        ai._registry.register_action_from_instance(each)

    op = await job.start(module.RenderRequest(quarter='q4'))
    checked = await ai.check_operation(op)
    assert checked.done is True
    assert checked.output is not None
    assert checked.output.url == 'https://reports.example/report-q4.pdf'
    cancelled = await ai.cancel_operation(op)
    assert cancelled.done is True
