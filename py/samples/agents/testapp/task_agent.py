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

"""A task list the model edits by calling tools that mutate typed session state.

The user chats naturally ("add buy groceries", "mark task 1 done") and the model
reaches for tools that read and write a structured task list living in the
session's custom state. Declaring a ``state_schema`` means that state comes back
typed — so a UI can render the live task list straight off ``response.state``.

Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from typing import Any

from _ai import ai
from pydantic import BaseModel

from genkit import ActionRunContext


class TaskItem(BaseModel):
    id: int
    title: str
    done: bool = False


class TaskState(BaseModel):
    tasks: list[TaskItem] = []
    next_id: int = 1


# Custom state rides on the session as a loosely-typed blob (it may come back
# with camelCased keys after a round-trip), so normalize to a plain dict with the
# fields our tools rely on. The state_schema handles typing it on the way out.
def _tasks(custom: Any) -> dict[str, Any]:
    if isinstance(custom, BaseModel):
        custom = custom.model_dump()
    c = custom or {}
    tasks = c.get('tasks', [])
    next_id = c.get('next_id') or c.get('nextId') or (max((t.get('id', 0) for t in tasks), default=0) + 1)
    return {'tasks': tasks, 'next_id': next_id}


class AddTaskInput(BaseModel):
    title: str


class TaskIdInput(BaseModel):
    id: int


@ai.tool(name='addTask', description='Add a new task to the list. Returns the created task.')
async def add_task(input: AddTaskInput) -> TaskItem:
    created: TaskItem | None = None

    def mutate(custom: dict[str, Any] | None) -> dict[str, Any]:
        nonlocal created
        s = _tasks(custom)
        created = TaskItem(id=s['next_id'], title=input.title)
        s['tasks'].append(created.model_dump())
        s['next_id'] += 1
        return s

    await ai.current_session().update_custom(mutate)
    return created  # type: ignore[return-value]


@ai.tool(name='toggleTask', description='Toggle a task done/not-done by id.')
async def toggle_task(input: TaskIdInput) -> dict[str, Any]:
    result: dict[str, Any] = {'success': False, 'error': f'Task {input.id} not found'}

    def mutate(custom: dict[str, Any] | None) -> dict[str, Any]:
        nonlocal result
        s = _tasks(custom)
        for t in s['tasks']:
            if t['id'] == input.id:
                t['done'] = not t['done']
                result = {'success': True, 'task': t}
        return s

    await ai.current_session().update_custom(mutate)
    return result


@ai.tool(name='removeTask', description='Remove a task by id.')
async def remove_task(input: TaskIdInput) -> dict[str, Any]:
    result: dict[str, Any] = {'success': False, 'error': f'Task {input.id} not found'}

    def mutate(custom: dict[str, Any] | None) -> dict[str, Any]:
        nonlocal result
        s = _tasks(custom)
        before = len(s['tasks'])
        s['tasks'] = [t for t in s['tasks'] if t['id'] != input.id]
        if len(s['tasks']) < before:
            result = {'success': True}
        return s

    await ai.current_session().update_custom(mutate)
    return result


# state_schema types the custom state end to end: chat.state, response.state, and
# streamed chunk.custom all come back as TaskState instead of a bare dict.
task_agent = ai.define_agent(
    name='taskAgent',
    state_schema=TaskState,
    system=(
        'You are a concise task management assistant. Use addTask to add, toggleTask to '
        'mark done/undone, and removeTask to delete. After changing tasks, confirm briefly.'
    ),
    tools=[add_task, toggle_task, remove_task],
)


@ai.flow()
async def test_task_agent(text: str, ctx: ActionRunContext) -> dict[str, Any]:
    """Seed an empty list, run one turn, and hand back the live typed state."""
    chat = task_agent.chat(state={'custom': {'tasks': [], 'next_id': 1}, 'messages': [], 'artifacts': []})
    turn = chat.send(text or 'Add a task: buy groceries')
    async for chunk in turn:
        if chunk.text:
            ctx.send_chunk(chunk.text)
    res = await turn
    state = res.state
    return {'text': res.text, 'tasks': state.model_dump()['tasks'] if state else []}


if __name__ == '__main__':
    import asyncio

    ai.run_main(asyncio.sleep(0))
