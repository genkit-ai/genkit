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

"""Pin: agents are only on `from genkit.exp import Genkit`.

The leftover a caller reuses is the instance they constructed. Stable
`from genkit import Genkit` has no agent methods. Experimental
`from genkit.exp import Genkit` does, and still runs generate.
"""

from __future__ import annotations

import pytest

from genkit import Genkit as StableGenkit
from genkit._ai._testing import define_programmable_model
from genkit._core._action import ActionRunContext
from genkit._core._model import Message, ModelResponse
from genkit._core._typing import (
    AgentFinishReason,
    AgentInput,
    AgentResult,
    FinishReason,
    MessageData,
    Part,
    Role,
    TextPart,
)
from genkit.exp import (
    FileSessionStore,
    Genkit,
    InMemorySessionStore,
    SessionRunner,
    TurnContext,
    TurnResult,
    remote_agent,
)


def test_stable_genkit_has_no_agent_methods() -> None:
    ai = StableGenkit()
    assert not hasattr(ai, 'define_agent')
    assert not hasattr(ai, 'define_prompt_agent')
    assert not hasattr(ai, 'define_custom_agent')
    assert not hasattr(ai, 'agent')


def test_stable_genkit_keeps_graduated_methods() -> None:
    ai = StableGenkit()
    assert hasattr(ai, 'define_interrupt')
    assert hasattr(ai, 'generate')
    assert hasattr(ai, 'generate_operation')


def test_exp_types_import() -> None:
    assert FileSessionStore is not None
    assert InMemorySessionStore is not None
    assert remote_agent is not None


def test_from_genkit_agent_is_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError):
        import genkit.agent  # ty: ignore[unresolved-import]  # noqa: F401


def test_from_genkit_exp_agent_imports() -> None:
    from genkit.exp.agent import (  # noqa: F401
        Agent,
        FileSessionStore,
        InMemorySessionStore,
        remote_agent as _remote_agent,
    )

    assert Agent is not None
    assert InMemorySessionStore is not None
    assert _remote_agent is not None


@pytest.mark.asyncio
async def test_exp_genkit_define_agent_one_turn() -> None:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='ok'))]),
        )
    )

    agent = ai.define_agent(name='echoAgent', model='programmableModel', system='Reply briefly.')
    out = await agent.chat().send('hello')

    assert out.text == 'ok'


@pytest.mark.asyncio
async def test_exp_genkit_define_prompt_agent_one_turn() -> None:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    ai.define_prompt(name='promptAgent', model='programmableModel', system='Reply briefly.')
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='from-prompt'))]),
        )
    )

    agent = ai.define_prompt_agent(name='promptAgent')
    out = await agent.chat().send('hello')

    assert out.text == 'from-prompt'


@pytest.mark.asyncio
async def test_exp_genkit_define_custom_agent_one_turn() -> None:
    ai = Genkit()

    async def echo_fn(session_runner: SessionRunner, _: ActionRunContext) -> AgentResult:
        async def handle_turn(inp: AgentInput, __: TurnContext) -> TurnResult | None:
            text = ''
            if inp.message and inp.message.content:
                root = inp.message.content[0].root
                text = getattr(root, 'text', '') or ''
            await session_runner.add_messages([
                MessageData(role='model', content=[Part(root=TextPart(text=f'Echo: {text}'))])
            ])
            return TurnResult(finish_reason=AgentFinishReason.STOP)

        await session_runner.run(handle_turn)
        return await session_runner.result()

    agent = ai.define_custom_agent(name='customEcho', fn=echo_fn)
    out = await agent.chat().send('hi')

    assert out.text == 'Echo: hi'


@pytest.mark.asyncio
async def test_exp_genkit_agent_lookup() -> None:
    ai = Genkit()
    define_programmable_model(ai)
    defined = ai.define_agent(name='lookupAgent', model='programmableModel')
    found = await ai.agent('lookupAgent')

    assert found is defined


@pytest.mark.asyncio
async def test_exp_genkit_still_generates() -> None:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='gen'))]),
        )
    )

    response = await ai.generate(model='programmableModel', prompt='hello')

    assert response.text == 'gen'
