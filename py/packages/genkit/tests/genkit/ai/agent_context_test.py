# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""AgentContext: the ctx a custom agent fn gets, where send_chunk takes an AgentStreamChunk."""

from typing import get_args, get_origin

import pytest

from genkit import Message, ModelResponseChunk, Part, Role
from genkit._core._action import ActionRunContext
from genkit.exp import Genkit
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


def test_agent_context_is_action_run_context_of_agent_stream_chunk() -> None:
    """`AgentContext` is `ActionRunContext[AgentStreamChunk]`, not a new class."""
    assert get_origin(AgentContext) is ActionRunContext
    assert get_args(AgentContext) == (AgentStreamChunk,)


def _send_status(ctx: AgentContext) -> None:
    """Type-checked only: an AgentStreamChunk is accepted and a str is rejected.

    pyright and pyrefly flag the str here. The repo turns off ty's
    invalid-argument-type under tests/; ty flags the same line in src.
    """
    ctx.send_chunk(AgentStreamChunk(model_chunk=ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('ok')])))
    ctx.send_chunk('ok')  # pyright: ignore[reportArgumentType]  # pyrefly: ignore[bad-argument-type]


@pytest.mark.asyncio
async def test_custom_agent_fn_typed_agent_context_streams_chunks() -> None:
    """A custom agent fn annotated `ctx: AgentContext` streams through `ctx.send_chunk` to the turn's stream."""
    ai = Genkit()

    async def concierge(sess: SessionRunner, ctx: AgentContext) -> AgentResult:
        async def handle_turn(_: AgentInput, __: TurnContext) -> TurnResult | None:
            ctx.send_chunk(
                AgentStreamChunk(
                    model_chunk=ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('Table for two at 7pm')])
                )
            )
            await sess.add_messages([Message(role=Role.MODEL, content=[Part.from_text('Table for two at 7pm')])])
            return TurnResult(finish_reason=AgentFinishReason.STOP)

        await sess.run(handle_turn)
        return await sess.result()

    agent = ai.define_custom_agent(name='concierge', fn=concierge, store=InMemorySessionStore())
    turn = agent.chat().send_stream('Book dinner')
    texts = [chunk.text async for chunk in turn.stream if chunk.text]

    assert texts == ['Table for two at 7pm']
