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

from genkit._ai._agents._runtime import AgentRuntime
from genkit._ai._agents._session import Session
from genkit._ai._json_patch import apply_json_patch, diff_json
from genkit._core._channel import CloseableQueue
from genkit._core._typing import (
    AgentStreamChunk,
    JsonPatchOp,
    JsonPatchOperation,
    ModelResponseChunk,
    Part,
    SessionState,
    TextPart,
)


def test_diff_object_field_replace() -> None:
    patch = diff_json(from_value={'status': 'idle'}, to_value={'status': 'working'})
    assert len(patch) == 1
    assert patch[0].op == 'replace'
    assert patch[0].path == '/status'
    assert patch[0].value == 'working'


def test_diff_array_append() -> None:
    patch = diff_json(from_value={'items': [1]}, to_value={'items': [1, 2]})
    assert any(op.op == 'add' and op.path == '/items/-' and op.value == 2 for op in patch)


# ---------------------------------------------------------------------------
# apply_json_patch: leniency + full op set (aligned with the JS/Go runtimes)
# ---------------------------------------------------------------------------


def test_apply_add_creates_missing_parent() -> None:
    # Lenient: a missing intermediate container is initialized rather than raising.
    res = apply_json_patch(doc={}, patch=[JsonPatchOperation(op=JsonPatchOp.ADD, path='/a/b', value=1)])
    assert res == {'a': {'b': 1}}


def test_apply_remove_missing_member_is_noop() -> None:
    doc = {'a': 1}
    res = apply_json_patch(doc=doc, patch=[JsonPatchOperation(op=JsonPatchOp.REMOVE, path='/missing')])
    assert res == {'a': 1}


def test_apply_replace_missing_parent_is_lenient() -> None:
    res = apply_json_patch(doc={}, patch=[JsonPatchOperation(op=JsonPatchOp.REPLACE, path='/x/y', value='v')])
    assert res == {'x': {'y': 'v'}}


def test_apply_test_op_passes() -> None:
    doc = {'status': 'idle'}
    res = apply_json_patch(doc=doc, patch=[JsonPatchOperation(op=JsonPatchOp.TEST, path='/status', value='idle')])
    assert res == {'status': 'idle'}


def test_apply_test_op_fails() -> None:
    with pytest.raises(ValueError, match='test failed'):
        apply_json_patch(
            doc={'status': 'idle'}, patch=[JsonPatchOperation(op=JsonPatchOp.TEST, path='/status', value='busy')]
        )


def test_apply_move_op() -> None:
    doc = {'a': 1}
    res = apply_json_patch(doc=doc, patch=[JsonPatchOperation(op=JsonPatchOp.MOVE, path='/b', **{'from': '/a'})])
    assert res == {'b': 1}


def test_apply_copy_op() -> None:
    doc = {'a': 1}
    res = apply_json_patch(doc=doc, patch=[JsonPatchOperation(op=JsonPatchOp.COPY, path='/b', **{'from': '/a'})])
    assert res == {'a': 1, 'b': 1}


def test_apply_does_not_mutate_input() -> None:
    doc = {'a': {'b': 1}}
    apply_json_patch(doc=doc, patch=[JsonPatchOperation(op=JsonPatchOp.REPLACE, path='/a/b', value=2)])
    assert doc == {'a': {'b': 1}}


def test_apply_invalid_pointer_raises() -> None:
    with pytest.raises(ValueError, match='must start with'):
        apply_json_patch(doc={}, patch=[JsonPatchOperation(op=JsonPatchOp.ADD, path='nope', value=1)])


@pytest.mark.asyncio
async def test_runtime_emits_custom_patch() -> None:
    out_queue = CloseableQueue()
    session = Session(SessionState(custom={'status': 'idle'}))
    AgentRuntime(
        name='test',
        session=session,
        parent_snapshot=None,
        store=None,
        state_transform=None,
        chunk_transform=None,
        emit_chunk=out_queue.put_nowait,
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
    out_queue = CloseableQueue()
    session = Session(SessionState(custom={'status': 'idle'}))
    rt = AgentRuntime(
        name='test',
        session=session,
        parent_snapshot=None,
        store=None,
        state_transform=None,
        chunk_transform=None,
        emit_chunk=out_queue.put_nowait,
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

    await rt.reset_custom_patch_turn()
    await session.update_custom(lambda c: {**(c or {}), 'status': 'idle'})
    chunk = out_queue.get_nowait()
    ops = chunk.custom_patch.root if chunk.custom_patch else []
    assert ops[0].path == ''


@pytest.mark.asyncio
async def test_runtime_custom_patch_honors_state_transform() -> None:
    out_queue = CloseableQueue()
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
        state_transform=redact,
        chunk_transform=None,
        emit_chunk=out_queue.put_nowait,
    )

    await session.update_custom(lambda c: c)
    chunk = out_queue.get_nowait()
    assert chunk.custom_patch is not None
    assert chunk.custom_patch.root[0].value == {'public': 'ok'}


@pytest.mark.asyncio
async def test_runtime_chunk_transform_can_drop_chunks() -> None:
    out_queue = CloseableQueue()
    session = Session()
    rt = AgentRuntime(
        name='test',
        session=session,
        parent_snapshot=None,
        store=None,
        state_transform=None,
        chunk_transform=lambda _chunk: None,
        emit_chunk=out_queue.put_nowait,
    )

    rt.send_chunk(AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part(root=TextPart(text='hi'))])))
    assert out_queue.empty()


@pytest.mark.asyncio
async def test_runtime_chunk_transform_can_redact_model_chunks() -> None:
    out_queue = CloseableQueue()
    session = Session()
    rt = AgentRuntime(
        name='test',
        session=session,
        parent_snapshot=None,
        store=None,
        state_transform=None,
        chunk_transform=lambda chunk: (
            chunk.model_copy(
                update={
                    'model_chunk': ModelResponseChunk(
                        content=[Part(root=TextPart(text='[redacted]'))],
                    )
                }
            )
            if chunk.model_chunk is not None
            else chunk
        ),
        emit_chunk=out_queue.put_nowait,
    )

    rt.send_chunk(AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part(root=TextPart(text='secret'))])))
    chunk = out_queue.get_nowait()
    assert chunk.model_chunk is not None
    assert chunk.model_chunk.content[0].root.text == '[redacted]'
