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

from __future__ import annotations

import pytest

from genkit._ai._aio import Genkit
from genkit._ai._testing import define_programmable_model
from genkit._ai._tools import ToolRunContext
from genkit._core._model import Message, ModelResponse
from genkit._core._typing import FinishReason, Part, Role, ToolRequest, ToolRequestPart, TextPart


@pytest.mark.asyncio
async def test_agent_propagates_context_to_tools() -> None:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    context_seen = []

    @ai.tool(name='getContextTool')
    async def get_context_tool(_: dict, ctx: ToolRunContext) -> str:
        context_seen.append(ctx.context)
        return f"auth: {ctx.context.get('auth', 'missing')}"

    agent = ai.define_agent(
        name='contextAgent',
        model='programmableModel',
        system='Context tester.',
        tools=[get_context_tool],
    )

    # 1. First model call: request the tool call
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(
                                name='getContextTool',
                                ref='ref-1',
                                input={},
                            )
                        )
                    )
                ],
            ),
        )
    )

    # 2. Second model call: accept tool result and stop
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='done'))]),
        )
    )

    conn = await agent.stream_bidi(context={'auth': 'secret'})
    await conn.send_text('run')
    await conn.close()
    async for chunk in conn.receive():
        pass
    out = await conn.output()

    assert out.message is not None
    assert context_seen == [{'auth': 'secret'}]


@pytest.mark.asyncio
async def test_prompt_agent_propagates_context_to_template() -> None:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    ai.define_prompt(
        name='templatedAgent',
        model='programmableModel',
        system='System context: {{@auth.email}}',
    )
    agent = ai.define_prompt_agent(name='templatedAgent')

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='ack'))]),
        )
    )

    conn = await agent.stream_bidi(context={'auth': {'email': 'secret@agent.com'}})
    await conn.send_text('hello')
    await conn.close()
    async for chunk in conn.receive():
        pass
    await conn.output()

    assert pm.request_count == 1
    assert pm.last_request is not None
    system_msg = next((m for m in pm.last_request.messages if m.role == Role.SYSTEM), None)
    assert system_msg is not None
    assert 'secret@agent.com' in system_msg.content[0].root.text


@pytest.mark.asyncio
async def test_agent_lookup_and_execution() -> None:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    ai.define_agent(
        name='lookupAgent',
        model='programmableModel',
        system='Hello from lookup agent.',
    )

    # Lookup asynchronously
    agent = await ai.agent('lookupAgent')
    assert agent.name == 'lookupAgent'

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='lookup response'))]),
        )
    )

    # Execution runs it
    conn = await agent.stream_bidi()
    await conn.send_text('hello')
    await conn.close()
    async for chunk in conn.receive():
        pass
    out = await conn.output()

    assert out.message is not None
    assert out.message.content[0].root.text == 'lookup response'


@pytest.mark.asyncio
async def test_agent_lookup_not_found() -> None:
    ai = Genkit()

    with pytest.raises(Exception, match="Agent 'missingAgent' not found in registry."):
        await ai.agent('missingAgent')
