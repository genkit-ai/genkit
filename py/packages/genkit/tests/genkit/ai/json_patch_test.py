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

import asyncio

import pytest

from genkit._ai._agents._base import AgentRuntime
from genkit._ai._json_patch import diff_json
from genkit._ai._session import Session
from genkit._core._typing import AgentStreamChunk, ModelResponseChunk, Part, SessionState, TextPart


def test_diff_object_field_replace() -> None:
    patch = diff_json({'status': 'idle'}, {'status': 'working'})
    assert len(patch) == 1
    assert patch[0].op == 'replace'
    assert patch[0].path == '/status'
    assert patch[0].value == 'working'


def test_diff_array_append() -> None:
    patch = diff_json({'items': [1]}, {'items': [1, 2]})
    assert any(op.op == 'add' and op.path == '/items/-' and op.value == 2 for op in patch)


@pytest.mark.asyncio
async def test_runtime_emits_custom_patch() -> None:
    out_queue: asyncio.Queue = asyncio.Queue()
    session = Session(SessionState(custom={'status': 'idle'}))
    AgentRuntime(
        name='test',
        session=session,
        parent_snapshot=None,
        store=None,
        snapshot_callback=None,
        client_transform=None,
        out_queue=out_queue,
    )

    await session.update_custom(lambda c: {**(c or {}), 'status': 'working'})
    chunk = out_queue.get_nowait()
    assert chunk.custom_patch is not None
    ops = chunk.custom_patch.root
    assert len(ops) == 1
    assert ops[0].op == 'replace'
    assert ops[0].path == ''
    assert ops[0].value == {'status': 'working'}


@pytest.mark.asyncio
async def test_runtime_incremental_custom_patch_within_turn() -> None:
    out_queue: asyncio.Queue = asyncio.Queue()
    session = Session(SessionState(custom={'status': 'idle'}))
    rt = AgentRuntime(
        name='test',
        session=session,
        parent_snapshot=None,
        store=None,
        snapshot_callback=None,
        client_transform=None,
        out_queue=out_queue,
    )

    await session.update_custom(lambda c: {**(c or {}), 'status': 'working'})
    out_queue.get_nowait()

    await session.update_custom(lambda c: {**(c or {}), 'status': 'done'})
    chunk = out_queue.get_nowait()
    ops = chunk.custom_patch.root if chunk.custom_patch else []
    assert len(ops) == 1
    assert ops[0].op == 'replace'
    assert ops[0].path == '/status'
    assert ops[0].value == 'done'

    await rt._reset_custom_patch_turn()
    await session.update_custom(lambda c: {**(c or {}), 'status': 'idle'})
    chunk = out_queue.get_nowait()
    ops = chunk.custom_patch.root if chunk.custom_patch else []
    assert ops[0].path == ''


@pytest.mark.asyncio
async def test_runtime_custom_patch_honors_state_transform() -> None:
    out_queue: asyncio.Queue = asyncio.Queue()
    session = Session(SessionState(custom={'public': 'ok', 'secret': 'hidden'}))

    def redact(state: SessionState) -> SessionState:
        custom = dict(state.custom or {})
        custom.pop('secret', None)
        return state.model_copy(update={'custom': custom})

    AgentRuntime(
        name='test',
        session=session,
        parent_snapshot=None,
        store=None,
        snapshot_callback=None,
        client_transform={'state': redact},
        out_queue=out_queue,
    )

    await session.update_custom(lambda c: c)
    chunk = out_queue.get_nowait()
    assert chunk.custom_patch is not None
    assert chunk.custom_patch.root[0].value == {'public': 'ok'}


@pytest.mark.asyncio
async def test_runtime_chunk_transform_can_drop_chunks() -> None:
    out_queue: asyncio.Queue = asyncio.Queue()
    session = Session()
    rt = AgentRuntime(
        name='test',
        session=session,
        parent_snapshot=None,
        store=None,
        snapshot_callback=None,
        client_transform={'chunk': lambda _chunk: None},
        out_queue=out_queue,
    )

    rt._send_chunk(AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part(root=TextPart(text='hi'))])))
    assert out_queue.empty()


@pytest.mark.asyncio
async def test_runtime_chunk_transform_can_redact_model_chunks() -> None:
    out_queue: asyncio.Queue = asyncio.Queue()
    session = Session()
    rt = AgentRuntime(
        name='test',
        session=session,
        parent_snapshot=None,
        store=None,
        snapshot_callback=None,
        client_transform={
            'chunk': lambda chunk: (
                chunk.model_copy(
                    update={
                        'model_chunk': ModelResponseChunk(
                            content=[Part(root=TextPart(text='[redacted]'))],
                        )
                    }
                )
                if chunk.model_chunk is not None
                else chunk
            )
        },
        out_queue=out_queue,
    )

    rt._send_chunk(AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part(root=TextPart(text='secret'))])))
    chunk = out_queue.get_nowait()
    assert chunk.model_chunk is not None
    assert chunk.model_chunk.content[0].root.text == '[redacted]'
