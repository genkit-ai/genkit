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

"""Type-checker contract for FlowContext, ToolContext, ModelContext, and AgentContext.

pyright, pyrefly, and ty include ``packages/*/tests/typecheck/`` with strict
argument and overload checks enabled, so any signature mismatch here fails CI.
"""

from typing import Any

import pytest
from pydantic import BaseModel
from typing_extensions import assert_type

from genkit import (
    Embedding,
    Flow,
    FlowContext,
    Genkit,
    Message,
    ModelResponse,
    ModelResponseChunk,
    Operation,
    Part,
    Role,
    Tool,
    ToolContext,
)
from genkit._core._action import ActionRunContext
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
from genkit.model import ModelContext, ModelRequest
from genkit.plugin_api import ActionKind, BackgroundAction


class RenderRequest(BaseModel):
    quarter: str


class Report(BaseModel):
    url: str


class ReportJob(BackgroundAction[RenderRequest, Report]):
    pass


def test_context_subclasses_satisfy_action_run_context() -> None:
    """Each public context subclass satisfies ``ActionRunContext[ChunkT]``."""

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


def test_registration_overloads_accept_public_and_internal_contexts() -> None:
    """Registration APIs accept handlers annotated with public subclasses or internal ActionRunContext."""
    ai = Genkit()
    exp_ai = ExpGenkit()

    @ai.flow
    async def plate_dish(dish: str, ctx: FlowContext[str]) -> str:
        ctx.send_chunk('searing')
        return dish

    @ai.flow
    async def plate_dish_internal(dish: str, ctx: ActionRunContext[str]) -> str:
        ctx.send_chunk('plating')
        return dish

    @ai.flow
    async def ping_kitchen(ctx: FlowContext[int]) -> str:
        ctx.send_chunk(1)
        return 'ready'

    async def raw_plate_dish(dish: str, ctx: FlowContext[str]) -> str:
        ctx.send_chunk('garnishing')
        return dish

    standalone_flow = Flow(raw_plate_dish, name='standalonePlateDish')

    assert_type(plate_dish, Flow[str, str, str])
    assert_type(plate_dish_internal, Flow[str, str, str])
    assert_type(ping_kitchen, Flow[None, str, int])
    assert_type(standalone_flow, Flow[str, str, str])

    @ai.tool
    async def check_allergens(dish: str, ctx: ToolContext[str]) -> str:
        ctx.send_chunk('checking')
        return dish if not ctx.is_resumed() else f'{dish} (resumed)'

    assert_type(check_allergens, Tool[str, str, str])

    async def chef(req: ModelRequest, ctx: ModelContext) -> ModelResponse:
        del req
        assert_type(ctx.is_streaming, bool)
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))

    async def chef_internal(req: ModelRequest, ctx: ActionRunContext[ModelResponseChunk]) -> ModelResponse:
        del req, ctx
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))

    ai.define_model(name='bistro/chef', fn=chef)
    ai.define_model(name='bistro/chef-internal', fn=chef_internal)

    async def start_video(req: ModelRequest, ctx: ModelContext) -> Operation:
        del req, ctx
        return Operation(id='op-1', done=False)

    async def check_video(op: Operation[ModelResponse], ctx: ModelContext) -> Operation:
        del ctx
        return op

    async def cancel_video(op: Operation[ModelResponse], ctx: ModelContext) -> Operation:
        del ctx
        return op

    ai.define_background_model(
        name='bistro/video',
        start=start_video,
        check=check_video,
        cancel=cancel_video,
    )

    async def start_report(req: RenderRequest, ctx: FlowContext) -> Operation:
        del ctx
        return Operation(id=f'report-{req.quarter}', done=False)

    async def check_report(op: Operation[Report], ctx: FlowContext) -> Operation:
        del ctx
        return op

    async def cancel_report(op: Operation[Report], ctx: FlowContext) -> Operation:
        del ctx
        return op

    job = ReportJob('render-report', start_report, check_report, kind=ActionKind.CUSTOM, cancel=cancel_report)
    assert job.name == 'render-report'

    async def embed_menu(req: EmbedRequest, ctx: FlowContext) -> EmbedResponse:
        del ctx
        return EmbedResponse(embeddings=[Embedding(embedding=[0.5]) for _ in req.input])

    ai.define_embedder(name='bistro/embedder', fn=embed_menu)

    async def concierge(sess: SessionRunner, ctx: AgentContext) -> AgentResult:
        assert_type(ctx.is_streaming, bool)

        async def handle_turn(_: AgentInput, __: TurnContext) -> TurnResult | None:
            return TurnResult(finish_reason=AgentFinishReason.STOP)

        await sess.run(handle_turn)
        return await sess.result()

    exp_ai.define_custom_agent(name='concierge', fn=concierge, store=InMemorySessionStore())


@pytest.mark.asyncio
async def test_shared_flow_and_tool_helper_union_typechecks() -> None:
    """A helper shared between a flow and a tool types cleanly with ``FlowContext | ToolContext``."""
    ai = Genkit()

    def read_table(ctx: FlowContext | ToolContext) -> str:
        if isinstance(ctx, ToolContext) and ctx.is_resumed():
            return 'resumed-table'
        return str(ctx.context.get('table', 'walk-in'))

    def emit_step(ctx: FlowContext[str] | ToolContext[str], step: str) -> None:
        ctx.send_chunk(step)

    @ai.tool
    async def check_table(dish: str, ctx: ToolContext[str]) -> str:
        emit_step(ctx, 'tool-step')
        return f'{dish}@{read_table(ctx)}'

    @ai.flow
    async def serve_order(dish: str, ctx: FlowContext[str]) -> str:
        emit_step(ctx, 'flow-step')
        return f'{dish}@{read_table(ctx)}'

    assert await serve_order('tartine', context={'table': 7}) == 'tartine@7'
    assert await check_table('tartine', context={'table': 9}) == 'tartine@9'


@pytest.mark.asyncio
async def test_user_id_helper_accepts_flow_and_tool_contexts() -> None:
    """``def user_id(ctx: FlowContext | ToolContext)`` accepts both contexts; ``ActionRunContext`` stays private."""
    ai = Genkit()

    def user_id(ctx: FlowContext | ToolContext) -> str:
        return ctx.context['auth']['uid']

    flow_ctx: FlowContext = FlowContext(context={'auth': {'uid': 'diner-42'}})
    tool_ctx: ToolContext = ToolContext(ActionRunContext(context={'auth': {'uid': 'diner-7'}}))
    assert_type(user_id(flow_ctx), str)
    assert_type(user_id(tool_ctx), str)
    assert user_id(flow_ctx) == 'diner-42'
    assert user_id(tool_ctx) == 'diner-7'

    @ai.flow
    async def reorder_last(dish: str, ctx: FlowContext) -> str:
        return f'{dish} for {user_id(ctx)}'

    @ai.tool
    async def check_allergies(dish: str, ctx: ToolContext) -> str:
        return f'{dish}: nut-free for {user_id(ctx)}'

    auth = {'auth': {'uid': 'diner-42'}}
    assert await reorder_last('tartine', context=auth) == 'tartine for diner-42'
    assert await check_allergies('tartine', context=auth) == 'tartine: nut-free for diner-42'
