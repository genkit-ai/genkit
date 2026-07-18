#!/usr/bin/env python3
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

"""Fire a long turn into the background and poll it to completion.

chat.detach() submits a turn and returns immediately with a snapshot id instead of
streaming — the work runs server-side. task.wait() resolves once the snapshot
reaches a terminal status (task.poll() streams status for a live UI). This is the
shape of a job-queue / async-task API on top of an agent. Requires GEMINI_API_KEY.
"""

from __future__ import annotations

import asyncio

from genkit_google_genai import GoogleAI
from pydantic import BaseModel

from genkit import ActionRunContext, FinishReason, Genkit, GenkitError, Message, ToolRunContext
from genkit.agent import (
    AgentFinishReason,
    AgentInput,
    AgentResult,
    InMemorySessionStore,
    SessionRunner,
    SessionSnapshot,
    SnapshotStatus,
    TurnResult,
)


class JobState(BaseModel):
    step: int = 0
    completed: bool = False


ai = Genkit(plugins=[GoogleAI()])
store = InMemorySessionStore()


@ai.tool(name='slowWork', description='Simulate long background work.')
async def slow_work(_: dict, ctx: ToolRunContext) -> dict:
    for _i in range(10):
        if ctx.abort_signal.is_set():
            raise GenkitError(status='ABORTED', message='Task aborted')
        await asyncio.sleep(0.5)  # pretend each step is real work
    return {'done': True}


async def long_task_fn(sess: SessionRunner, ctx: ActionRunContext) -> AgentResult:
    # Define tool inside turn handler closure so it can mutate custom session state on each step:
    @ai.tool(name='slowWork', description='Simulate long background work.')
    async def slow_work_closure(_: dict, tool_ctx: ToolRunContext) -> dict:
        for i in range(1, 11):
            if tool_ctx.abort_signal.is_set():
                raise GenkitError(status='ABORTED', message='Task aborted')
            await asyncio.sleep(0.5)
            await sess.update_custom(lambda _, step=i: JobState(step=step, completed=(step == 10)))
        return {'done': True}

    async def handle_turn(inp: AgentInput) -> TurnResult | None:
        history = await sess.get_messages()
        messages = [Message(m) for m in history] if history else None
        res = await ai.generate(
            model='googleai/gemini-flash-latest',
            system='When asked for a long task, call slowWork.',
            messages=messages,
            tools=[slow_work_closure],
        )
        if res.message:
            await sess.add_messages(res.message)
        fr = AgentFinishReason.STOP if res.finish_reason == FinishReason.STOP else AgentFinishReason.UNKNOWN
        return TurnResult(finish_reason=fr)

    await sess.run(handle_turn)
    return await sess.result()


agent = ai.define_custom_agent(
    name='longTaskAgent',
    fn=long_task_fn,
    state_schema=JobState,
    store=store,
)


async def main() -> None:
    chat = agent.chat(state=JobState(step=0, completed=False))

    # Submit the turn and return right away — the work continues in the background.
    task = await chat.detach('Please run a long task using slowWork.')
    assert task.snapshot_id  # the handle you poll on, hand off, or persist

    # Re-read the server snapshot every 0.5s and yield live status until terminal:
    last_snap: SessionSnapshot[JobState] | None = None
    async for snap in task.poll(interval=0.5):
        last_snap = snap
        if snap.state and snap.state.custom:
            print(
                f'Live poll -> status: {snap.status}, step: {snap.state.custom.step}, '
                f'completed: {snap.state.custom.completed}'
            )

    assert last_snap is not None and last_snap.status == SnapshotStatus.COMPLETED
    assert last_snap.state
    assert last_snap.state.custom is not None
    assert last_snap.state.custom.step == 10 and last_snap.state.custom.completed is True

    # Access the agent's completed output message off the terminal snapshot
    assert last_snap.state.messages is not None
    latest_message = last_snap.state.messages[-1]
    print('Completed background task output:', latest_message.content[0].root.text)

    # To resume the conversation later, load the chat by snapshot_id
    loaded_chat = await agent.load_chat(snapshot_id=task.snapshot_id)
    assert len(loaded_chat.messages) == len(last_snap.state.messages)


if __name__ == '__main__':
    ai.run_main(main())
