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

"""CallScope: every generate path gets its own child of the app's registry and the app as ctx.ai."""

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from genkit import FinishReason, Genkit, Message, ModelResponse, Part, Role, Tool, tool
from genkit._ai._generate import CallScope
from genkit.exp import Genkit as ExpGenkit
from genkit.middleware import BaseMiddleware, GenerateMiddlewareContext, ModelHookParams
from genkit.model import ModelContext, ModelRequest, ToolRequest
from genkit.plugin_api import ActionKind

NextModel = Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]]


def _define_tool_caller(ai: Genkit, name: str = 'caller') -> None:
    """Model that asks for `lookup` once, then answers with whatever the tool returned."""

    async def fn(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        last = request.messages[-1]
        for part in last.content:
            if part.tool_response is not None:
                return ModelResponse(
                    finish_reason=FinishReason.STOP,
                    message=Message(role=Role.MODEL, content=[Part.from_text(str(part.tool_response.output))]),
                )
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[Part(tool_request=ToolRequest(name='lookup', ref='r1', input={}))],
            ),
        )

    ai.define_model(name=name, fn=fn)


def _inline_lookup(answer: str, delay: float = 0.0) -> Tool:
    async def lookup() -> str:
        """Answers for this call only."""
        await asyncio.sleep(delay)
        return answer

    return tool(lookup)


def test_call_scope_registry_is_a_fresh_child_of_its_ai() -> None:
    """Each scope gets its own child of the app registry; writes stay in it."""
    # 1. Two scopes over one app
    ai = Genkit()
    ai.define_value(kind='catalog', name='banner', value='spring menu')
    first, second = CallScope(ai), CallScope(ai)

    # 2. Both see the app, neither is the app, and they aren't each other
    assert first.ai is ai
    assert first.registry.is_child
    assert first.registry is not second.registry
    assert first.registry.lookup_value('catalog', 'banner') == 'spring menu'

    # 3. A per-call registration stays in its scope
    first.registry.register_value('catalog', 'scratch', 'only here')
    assert second.registry.lookup_value('catalog', 'scratch') is None
    assert ai.lookup_value(kind='catalog', name='scratch') is None


@pytest.mark.asyncio
async def test_concurrent_generate_calls_keep_inline_tools_apart() -> None:
    """Two overlapping calls with a same-named inline tool each run their own, and neither leaks."""
    ai = Genkit()
    _define_tool_caller(ai)

    slow, fast = await asyncio.gather(
        ai.generate(model='caller', prompt='hi', tools=[_inline_lookup('slow answer', delay=0.05)]),
        ai.generate(model='caller', prompt='hi', tools=[_inline_lookup('fast answer')]),
    )

    assert slow.text == 'slow answer'
    assert fast.text == 'fast answer'
    assert await ai.lookup_tool('lookup') is None


@pytest.mark.asyncio
async def test_inline_tool_on_prompt_call_runs_and_does_not_outlive_the_call() -> None:
    """`prompt(tools=[tool(fn)])` resolves the tool for that call only."""
    ai = Genkit()
    _define_tool_caller(ai)
    prompt = ai.define_prompt(model='caller', prompt='hi')

    response = await prompt(tools=[_inline_lookup('from prompt call')])

    assert response.text == 'from prompt call'
    assert await ai.lookup_tool('lookup') is None


@pytest.mark.asyncio
async def test_prompt_action_render_lists_inline_tool_definitions() -> None:
    """The Dev UI prompt action renders a request whose tools include an inline define-time tool."""
    ai = Genkit()
    _define_tool_caller(ai)
    ai.define_prompt(name='withInline', model='caller', prompt='hi', tools=[_inline_lookup('unused')])

    prompt_action = await ai._registry.resolve_action(ActionKind.PROMPT, 'withInline')  # pyright: ignore[reportPrivateUsage]
    assert prompt_action is not None
    rendered = (await prompt_action.run()).response

    assert [t.name for t in rendered.tools or []] == ['lookup']
    assert await ai.lookup_tool('lookup') is None


@pytest.mark.asyncio
async def test_middleware_ctx_ai_is_the_app_inside_an_agent_turn() -> None:
    """Agent turns go through the prompt's CallScope, so middleware still sees the app as `ctx.ai`."""
    ai = ExpGenkit()
    seen: list[object] = []

    async def echo(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('ok')]),
        )

    ai.define_model(name='echo', fn=echo)

    class PeekAi(BaseMiddleware):
        async def wrap_model(
            self, params: ModelHookParams, ctx: GenerateMiddlewareContext, next_fn: NextModel
        ) -> ModelResponse:
            seen.append(ctx.ai)
            return await next_fn(params, ctx)

    agent = ai.define_agent(name='waiter', model='echo', system='Reply briefly.', use=[PeekAi()])
    out = await agent.chat().send('hello')

    assert out.text == 'ok'
    assert seen == [ai]
    assert seen[0] is ai
