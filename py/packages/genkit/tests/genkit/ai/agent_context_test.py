# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""AgentContext: the ctx a custom agent fn gets, where send_chunk takes an AgentStreamChunk."""

from typing import Any, cast, get_args, get_origin

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


def test_agent_context_is_subclass_of_action_run_context() -> None:
    """`AgentContext` subclasses `ActionRunContext[AgentStreamChunk]` so `isinstance(ctx, AgentContext)` works."""
    assert issubclass(AgentContext, ActionRunContext)
    orig_bases = cast(tuple[Any, ...], getattr(AgentContext, '__orig_bases__', ()))
    assert len(orig_bases) == 1
    assert get_origin(orig_bases[0]) is ActionRunContext
    assert get_args(orig_bases[0]) == (AgentStreamChunk,)
    ctx = AgentContext()
    assert isinstance(ctx, AgentContext)
    assert isinstance(ctx, ActionRunContext)


def _send_status(ctx: AgentContext) -> None:
    """Type-checked only: an AgentStreamChunk is accepted and a str is rejected.

    pyright and pyrefly flag the str here. The repo turns off ty's
    invalid-argument-type under tests/; ty flags the same line in src.
    """
    ctx.send_chunk(AgentStreamChunk(model_chunk=ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('ok')])))
    ctx.send_chunk('ok')  # pyright: ignore[reportArgumentType]  # pyrefly: ignore[bad-argument-type]


@pytest.mark.asyncio
async def test_custom_agent_fn_typed_agent_context_streams_chunks() -> None:
    """A custom agent fn annotated `ctx: AgentContext` receives an `AgentContext` and streams chunks."""
    ai = Genkit()
    seen_contexts: list[object] = []

    async def concierge(sess: SessionRunner, ctx: AgentContext) -> AgentResult:
        seen_contexts.append(ctx)

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
    assert len(seen_contexts) == 1
    assert isinstance(seen_contexts[0], AgentContext)
