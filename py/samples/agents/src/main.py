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

"""Python backend for the useChat audit demo — all agent scenarios."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from genkit import Genkit, GenkitError, ToolRunContext
from genkit._ai._agent import SessionRunner, TurnResult, _to_agent_finish_reason
from genkit._ai._generate import generate_action
from genkit._ai._prompt import PromptConfig, to_generate_action_options
from genkit._core._action import ActionRunContext
from genkit._core._typing import (
    AgentFinishReason,
    AgentInput,
    AgentResult,
    AgentStreamChunk,
    Artifact,
    MessageData,
    Part,
    TextPart,
)
from genkit.agent import InMemorySessionStore
from genkit.plugins.fastapi import genkit_fastapi_handler
from genkit.plugins.google_genai import GoogleAI
from genkit.plugins.middleware import Middleware, ToolApproval

ai = Genkit(plugins=[GoogleAI(), Middleware()])
agents: dict[str, object] = {}
long_task_store = InMemorySessionStore()


class WeatherInput(BaseModel):
    location: str


class WeatherOutput(BaseModel):
    weather: str
    temperature: str


class ApprovalInput(BaseModel):
    action: str
    details: str


class TransferInput(BaseModel):
    amount: float
    to_account: str = Field(alias='toAccount')

    model_config = {'populate_by_name': True}


class TransferOutput(BaseModel):
    success: bool
    transaction_id: str = Field(alias='transactionId')

    model_config = {'populate_by_name': True}


def _register_agents() -> dict[str, object]:
    weather_store = InMemorySessionStore()
    banking_store = InMemorySessionStore()
    greeter_store = InMemorySessionStore()
    custom_store = InMemorySessionStore()
    stateful_store = InMemorySessionStore()
    flaky_store = InMemorySessionStore()
    global long_task_store
    long_task_store = InMemorySessionStore()

    @ai.tool(name='getWeather', description='Get weather for a city.')
    async def get_weather(input: WeatherInput) -> WeatherOutput:
        import random

        return WeatherOutput(
            weather=f'{random.choice(["Sunny", "Cloudy", "Rainy"])} in {input.location}',
            temperature=f'{random.randint(5, 34)}°C',
        )

    weather_agent = ai.define_agent(
        name='weatherAgent',
        model='googleai/gemini-flash-latest',
        system='Weather assistant. Use getWeather for weather questions.',
        tools=[get_weather],
        store=weather_store,
    )

    echo_agent = ai.define_agent(
        name='echoNoStore',
        model='googleai/gemini-flash-latest',
        system='Echo assistant. Answer briefly and remember context.',
    )

    user_approval = ai.define_interrupt(
        name='userApproval',
        description='Ask the user to approve a sensitive action.',
        input_schema=ApprovalInput,
    )

    @ai.tool(name='transferMoney', description='Transfer money.')
    async def transfer_money(_input: TransferInput) -> TransferOutput:
        return TransferOutput(success=True, transactionId=f'txn-{uuid4().hex[:12]}')

    banking_agent = ai.define_agent(
        name='bankingAgent',
        model='googleai/gemini-flash-latest',
        system='Banking assistant. Use userApproval before transferMoney.',
        tools=[user_approval, transfer_money],
        store=banking_store,
    )

    approval_no_store = ai.define_agent(
        name='approvalNoStore',
        model='googleai/gemini-flash-latest',
        system='Banking assistant. Call transferMoney when the user asks to transfer money.',
        tools=[transfer_money],
        use=[ToolApproval(allowed_tools=[])],
    )

    ai.define_prompt(
        name='greeterPrompt',
        model='googleai/gemini-flash-latest',
        system='You are a greeter. Be warm and brief.',
    )
    greeter_agent = ai.define_prompt_agent(name='greeterPrompt', store=greeter_store)

    async def custom_coder_fn(sess: SessionRunner, ctx: ActionRunContext) -> AgentResult:
        async def handle_turn(inp: AgentInput) -> TurnResult | None:
            history = await sess.get_messages()
            child = ai.registry.new_child()
            pc = PromptConfig(
                model='googleai/gemini-flash-latest',
                system='Concise coding assistant.',
                messages=history or None,
            )
            opts = await to_generate_action_options(child, pc)

            def on_chunk(chunk) -> None:
                ctx.send_chunk(AgentStreamChunk(model_chunk=chunk))

            res = await generate_action(child, opts, on_chunk=on_chunk, abort_signal=ctx.abort_signal)
            if res.message:
                await sess.add_messages(res.message)
            return TurnResult(finish_reason=_to_agent_finish_reason(res.finish_reason))

        await sess.run(handle_turn)
        return await sess.result()

    custom_agent = ai.define_custom_agent(name='customCoder', fn=custom_coder_fn, store=custom_store)

    async def stateful_fn(sess: SessionRunner, ctx: ActionRunContext) -> AgentResult:
        async def handle_turn(inp: AgentInput) -> TurnResult | None:
            await sess.update_custom(lambda c: {'turns': (c or {}).get('turns', 0) + 1})
            await sess.add_artifacts(
                Artifact(name='status', parts=[Part(TextPart(text=f'turn {sess.turn_index + 1}'))])
            )
            history = await sess.get_messages()
            child = ai.registry.new_child()
            pc = PromptConfig(
                model='googleai/gemini-flash-latest',
                system='Acknowledge progress in one sentence.',
                messages=history or None,
            )
            opts = await to_generate_action_options(child, pc)

            def on_chunk(chunk) -> None:
                ctx.send_chunk(AgentStreamChunk(model_chunk=chunk))

            res = await generate_action(child, opts, on_chunk=on_chunk, abort_signal=ctx.abort_signal)
            if res.message:
                await sess.add_messages(res.message)
            return TurnResult(finish_reason=_to_agent_finish_reason(res.finish_reason))

        await sess.run(handle_turn)
        return await sess.result()

    stateful_agent = ai.define_custom_agent(name='statefulAgent', fn=stateful_fn, store=stateful_store)

    async def flaky_fn(sess: SessionRunner, _ctx: ActionRunContext) -> AgentResult:
        async def handle_turn(inp: AgentInput) -> TurnResult | None:
            text = ''
            if inp.messages:
                for part in inp.messages[-1].content or []:
                    root = getattr(part, 'root', part)
                    if isinstance(root, TextPart) and root.text:
                        text += root.text
            if 'fail' in text.lower():
                raise GenkitError(status='INTERNAL', message='Simulated turn failure')
            msgs = await sess.get_messages()
            await sess.set_messages(msgs + [MessageData(role='model', content=[Part(TextPart(text='OK'))])])
            return TurnResult(finish_reason=AgentFinishReason.STOP)

        await sess.run(handle_turn)
        return await sess.result()

    flaky_agent = ai.define_custom_agent(name='flakyAgent', fn=flaky_fn, store=flaky_store)

    @ai.tool(name='slowWork', description='Simulate long background work.')
    async def slow_work(_: dict, ctx: ToolRunContext) -> dict:
        for _i in range(30):
            if ctx.abort_signal.is_set():
                raise GenkitError(status='ABORTED', message='Task aborted')
            await asyncio.sleep(0.5)
        return {'done': True}

    long_task_agent = ai.define_agent(
        name='longTaskAgent',
        model='googleai/gemini-flash-latest',
        system='When asked for a long task, call slowWork.',
        tools=[slow_work],
        store=long_task_store,
    )

    return {
        'weather': weather_agent,
        'banking': banking_agent,
        'echo_no_store': echo_agent,
        'approval_no_store': approval_no_store,
        'greeter': greeter_agent,
        'custom_coder': custom_agent,
        'stateful': stateful_agent,
        'flaky': flaky_agent,
        'long_task': long_task_agent,
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    global agents
    agents = _register_agents()
    yield


app = FastAPI(title='Genkit Agent Audit Demo', lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000', 'http://127.0.0.1:3000'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class AbortBody(BaseModel):
    snapshot_id: str


@app.post('/api/agents/long-task/abort')
async def abort_long_task(body: AbortBody) -> dict:
    status = await long_task_store.abort_snapshot(body.snapshot_id)
    return {'snapshotId': body.snapshot_id, 'status': status}


@genkit_fastapi_handler(app, ai, path='/api/chat/weather')
async def chat_weather():
    return agents['weather']


@genkit_fastapi_handler(app, ai, path='/api/chat/banking')
async def chat_banking():
    return agents['banking']


@genkit_fastapi_handler(app, ai, path='/api/chat/echo-no-store')
async def chat_echo_no_store():
    return agents['echo_no_store']


@genkit_fastapi_handler(app, ai, path='/api/chat/approval-no-store')
async def chat_approval_no_store():
    return agents['approval_no_store']


@genkit_fastapi_handler(app, ai, path='/api/chat/greeter')
async def chat_greeter():
    return agents['greeter']


@genkit_fastapi_handler(app, ai, path='/api/chat/custom-coder')
async def chat_custom_coder():
    return agents['custom_coder']


@genkit_fastapi_handler(app, ai, path='/api/chat/stateful')
async def chat_stateful():
    return agents['stateful']


@genkit_fastapi_handler(app, ai, path='/api/chat/flaky')
async def chat_flaky():
    return agents['flaky']


@genkit_fastapi_handler(app, ai, path='/api/chat/long-task')
async def chat_long_task():
    return agents['long_task']
