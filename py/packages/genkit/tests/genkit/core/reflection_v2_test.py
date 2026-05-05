# Copyright 2025 Google LLC
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

"""Tests for Reflection API v2 (WebSocket JSON-RPC client).

Design notes:

- **fakeManager pattern**: A minimal in-process WebSocket *server* stands in for
  the CLI ``RuntimeManagerV2``. The runtime under test is the *client*. This
  isolates protocol handling without the full tools server or Dev UI.
- **Explicit JSON-RPC sequencing**: Tests ``read`` the next frame, assert
  ``method`` / ``id`` / ``params``, then ``write`` responses. This catches
  wrong ordering (e.g. ``register`` vs first ``listActions``) deterministically.
- **ackRegister helper**: The runtime sends ``register`` and awaits a result;
  most tests must reply with a minimal ``result`` so the client does not stall.
- **Draining notifications**: ``runAction`` may emit ``runActionState`` frames
  before the final ``result`` or ``error``; tests loop until they see the
  response shape they need rather than asserting on the very next frame.
- **Parallel failure modes**: ``cancelAction`` tests assert on *two* correlated
  replies (cancel ack + runAction error) without assuming order.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
import pytest_asyncio
from websockets.asyncio.server import serve

from genkit._core._action import Action, ActionKind, ActionRunContext
from genkit._core._reflection_v2 import (
    JSON_RPC_INVALID_PARAMS,
    JSON_RPC_METHOD_NOT_FOUND,
    JSON_RPC_SERVER_ERROR,
    ReflectionServerV2,
)
from genkit._core._registry import Registry


class FakeReflectionManager:
    """Minimal WebSocket server that accepts one runtime client (CLI stand-in)."""

    def __init__(self) -> None:
        self._stop = asyncio.Event()
        self._client_ws: Any = None
        self._server: Any = None
        self._serve_ctx: Any = None
        self._host = '127.0.0.1'
        self._port = 0
        self._ready: asyncio.Future[None] | None = None

    @property
    def url(self) -> str:
        return f'ws://{self._host}:{self._port}'

    async def _handler(self, ws: Any) -> None:
        self._client_ws = ws
        if self._ready is not None and not self._ready.done():
            self._ready.set_result(None)
        await self._stop.wait()

    async def start(self) -> None:
        self._ready = asyncio.get_running_loop().create_future()
        self._serve_ctx = serve(self._handler, self._host, 0)
        self._server = await self._serve_ctx.__aenter__()
        first_socket = next(iter(self._server.sockets))
        self._port = first_socket.getsockname()[1]

    async def aclose(self) -> None:
        self._stop.set()
        if self._client_ws is not None:
            await self._client_ws.close()
        if self._serve_ctx is not None:
            await self._serve_ctx.__aexit__(None, None, None)

    async def wait_connected(self, timeout: float = 2.0) -> None:
        assert self._ready is not None
        await asyncio.wait_for(self._ready, timeout=timeout)

    async def read_rpc(self, timeout: float = 2.0) -> dict[str, Any]:
        assert self._client_ws is not None
        raw = await asyncio.wait_for(self._client_ws.recv(), timeout=timeout)
        return json.loads(raw)

    async def write_rpc(self, msg: dict[str, Any]) -> None:
        assert self._client_ws is not None
        await self._client_ws.send(json.dumps(msg))


async def ack_register(fm: FakeReflectionManager) -> dict[str, Any]:
    msg = await fm.read_rpc()
    assert msg.get('method') == 'register'
    req_id = msg['id']
    assert isinstance(req_id, str) and req_id != ''
    await fm.write_rpc({'jsonrpc': '2.0', 'result': {}, 'id': req_id})
    return msg


@pytest_asyncio.fixture(loop_scope='function')
async def fake_manager() -> Any:
    fm = FakeReflectionManager()
    await fm.start()
    try:
        yield fm
    finally:
        await fm.aclose()


async def _run_client_lifecycle(
    registry: Registry,
    fm: FakeReflectionManager,
    *,
    app_name: str = 'test-app',
) -> tuple[ReflectionServerV2, asyncio.Task[None]]:
    client = ReflectionServerV2(registry, fm.url, app_name=app_name)
    task = asyncio.create_task(client.run_forever())
    await fm.wait_connected()
    await asyncio.sleep(0)  # let register task schedule
    return client, task


async def _stop_client(client: ReflectionServerV2, task: asyncio.Task[None]) -> None:
    client.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_reflection_server_v2_register(fake_manager: FakeReflectionManager) -> None:
    registry = Registry()
    client, task = await _run_client_lifecycle(registry, fake_manager)
    try:
        msg = await fake_manager.read_rpc()
        assert msg.get('method') == 'register'
        assert isinstance(msg.get('id'), str)
        params = msg.get('params')
        assert isinstance(params, dict)
        assert params.get('name') == 'test-app'
        assert params.get('id')
        assert isinstance(params.get('pid'), (int, float))
        assert str(params.get('genkitVersion', '')).startswith('py/')
        assert isinstance(params.get('reflectionApiSpecVersion'), (int, float))
        envs = params.get('envs')
        assert isinstance(envs, list) and envs == ['dev']
    finally:
        await _stop_client(client, task)


@pytest.mark.asyncio
async def test_reflection_server_v2_register_handshake_telemetry(fake_manager: FakeReflectionManager) -> None:
    registry = Registry()
    client, task = await _run_client_lifecycle(registry, fake_manager)
    try:
        msg = await fake_manager.read_rpc()
        assert msg.get('method') == 'register'
        req_id = msg['id']
        await fake_manager.write_rpc({
            'jsonrpc': '2.0',
            'result': {'telemetryServerUrl': 'http://127.0.0.1:9999'},
            'id': req_id,
        })
    finally:
        await _stop_client(client, task)


@pytest.mark.asyncio
async def test_reflection_server_v2_list_actions(fake_manager: FakeReflectionManager) -> None:
    """listActions returns the same action map as HTTP reflection (:func:`_get_actions_payload`)."""
    registry = Registry()

    async def inc(x: int) -> int:
        return x + 1

    registry.register_action_from_instance(Action(ActionKind.CUSTOM, 'test/inc', inc))

    client, task = await _run_client_lifecycle(registry, fake_manager)
    try:
        await ack_register(fake_manager)
        await fake_manager.write_rpc({
            'jsonrpc': '2.0',
            'method': 'listActions',
            'id': '1',
        })
        resp = await fake_manager.read_rpc()
        assert resp.get('id') == '1'
        result = resp.get('result')
        assert isinstance(result, dict)
        actions = result.get('actions')
        assert isinstance(actions, dict)
        assert actions == {
            '/custom/test/inc': {
                'key': '/custom/test/inc',
                'name': 'test/inc',
                'actionType': 'custom',
                'inputSchema': {'type': 'integer'},
                'outputSchema': {'type': 'integer'},
                'metadata': {
                    'inputSchema': {'type': 'integer'},
                    'outputSchema': {'type': 'integer'},
                },
            }
        }
    finally:
        await _stop_client(client, task)


@pytest.mark.asyncio
async def test_reflection_server_v2_list_values(fake_manager: FakeReflectionManager) -> None:
    registry = Registry()
    registry.register_value('defaultModel', 'defaultModel', 'my-model')

    client, task = await _run_client_lifecycle(registry, fake_manager)
    try:
        await ack_register(fake_manager)
        await fake_manager.write_rpc({
            'jsonrpc': '2.0',
            'method': 'listValues',
            'params': {'type': 'defaultModel'},
            'id': '2',
        })
        resp = await fake_manager.read_rpc()
        assert resp.get('id') == '2'
        result = resp.get('result')
        assert isinstance(result, dict)
        values = result.get('values')
        assert isinstance(values, dict)
        assert values.get('defaultModel') == 'my-model'
    finally:
        await _stop_client(client, task)


@pytest.mark.asyncio
async def test_reflection_server_v2_list_values_rejects_unsupported_type(
    fake_manager: FakeReflectionManager,
) -> None:
    registry = Registry()
    client, task = await _run_client_lifecycle(registry, fake_manager)
    try:
        await ack_register(fake_manager)
        await fake_manager.write_rpc({
            'jsonrpc': '2.0',
            'method': 'listValues',
            'params': {'type': 'prompt'},
            'id': '2a',
        })
        resp = await fake_manager.read_rpc()
        err = resp.get('error')
        assert isinstance(err, dict)
        assert err.get('code') == JSON_RPC_INVALID_PARAMS
    finally:
        await _stop_client(client, task)


@pytest.mark.asyncio
async def test_reflection_server_v2_run_action(fake_manager: FakeReflectionManager) -> None:
    registry = Registry()

    async def inc(x: int) -> int:
        return x + 1

    registry.register_action_from_instance(Action(ActionKind.CUSTOM, 'test/inc', inc))

    client, task = await _run_client_lifecycle(registry, fake_manager)
    try:
        await ack_register(fake_manager)
        await fake_manager.write_rpc({
            'jsonrpc': '2.0',
            'method': 'runAction',
            'params': {'key': '/custom/test/inc', 'input': 3},
            'id': '3',
        })
        resp: dict[str, Any] | None = None
        while resp is None:
            msg = await fake_manager.read_rpc()
            if msg.get('method') == 'runActionState':
                continue
            resp = msg
        assert resp.get('id') == '3'
        assert resp.get('error') is None
        result = resp.get('result')
        assert isinstance(result, dict)
        assert result.get('result') == 4
        telemetry = result.get('telemetry')
        assert isinstance(telemetry, dict)
        assert telemetry.get('traceId')
    finally:
        await _stop_client(client, task)


@pytest.mark.asyncio
async def test_reflection_server_v2_streaming_run_action(fake_manager: FakeReflectionManager) -> None:
    registry = Registry()

    async def stream_inc(x: int, ctx: ActionRunContext) -> int:
        for i in range(x):
            ctx.send_chunk(i)
        return x

    registry.register_action_from_instance(Action(ActionKind.CUSTOM, 'test/streaming', stream_inc))

    client, task = await _run_client_lifecycle(registry, fake_manager)
    try:
        await ack_register(fake_manager)
        await fake_manager.write_rpc({
            'jsonrpc': '2.0',
            'method': 'runAction',
            'params': {'key': '/custom/test/streaming', 'input': 3, 'stream': True},
            'id': '4',
        })
        chunks: list[Any] = []
        final: dict[str, Any] | None = None
        while final is None:
            msg = await fake_manager.read_rpc()
            if msg.get('method') == 'streamChunk':
                params = msg.get('params')
                assert isinstance(params, dict)
                assert params.get('requestId') == '4'
                chunks.append(params.get('chunk'))
                continue
            if msg.get('method') == 'runActionState':
                continue
            final = msg
        assert len(chunks) == 3
        for i, c in enumerate(chunks):
            assert c == i
        assert final is not None
        result = final.get('result')
        assert isinstance(result, dict)
        assert result.get('result') == 3
    finally:
        await _stop_client(client, task)


@pytest.mark.asyncio
async def test_reflection_server_v2_run_action_not_found(fake_manager: FakeReflectionManager) -> None:
    registry = Registry()
    client, task = await _run_client_lifecycle(registry, fake_manager)
    try:
        await ack_register(fake_manager)
        await fake_manager.write_rpc({
            'jsonrpc': '2.0',
            'method': 'runAction',
            'params': {'key': '/custom/does-not-exist', 'input': None},
            'id': '5',
        })
        resp = await fake_manager.read_rpc()
        err = resp.get('error')
        assert isinstance(err, dict)
        assert err.get('code') == JSON_RPC_INVALID_PARAMS
        assert 'not found' in str(err.get('message', '')).lower()
    finally:
        await _stop_client(client, task)


@pytest.mark.asyncio
async def test_reflection_server_v2_cancel_action(fake_manager: FakeReflectionManager) -> None:
    registry = Registry()
    started = asyncio.Event()

    async def slow(_: Any = None) -> Any:
        started.set()
        await asyncio.sleep(10**6)

    registry.register_action_from_instance(Action(ActionKind.CUSTOM, 'test/slow', slow))

    client, task = await _run_client_lifecycle(registry, fake_manager)
    try:
        await ack_register(fake_manager)
        await fake_manager.write_rpc({
            'jsonrpc': '2.0',
            'method': 'runAction',
            'params': {'key': '/custom/test/slow', 'input': None},
            'id': '6',
        })
        await asyncio.wait_for(started.wait(), timeout=2.0)
        trace_id = ''
        while not trace_id:
            msg = await fake_manager.read_rpc()
            if msg.get('method') == 'runActionState':
                params = msg.get('params')
                assert isinstance(params, dict)
                state = params.get('state')
                assert isinstance(state, dict)
                tid = state.get('traceId')
                if tid:
                    trace_id = str(tid)

        await fake_manager.write_rpc({
            'jsonrpc': '2.0',
            'method': 'cancelAction',
            'params': {'traceId': trace_id},
            'id': '7',
        })

        saw_cancel = False
        saw_run_err = False
        while not saw_cancel or not saw_run_err:
            msg = await fake_manager.read_rpc()
            mid = msg.get('id')
            if mid == '7':
                result = msg.get('result')
                assert isinstance(result, dict)
                assert result.get('message') == 'Action cancelled'
                saw_cancel = True
            elif mid == '6':
                err = msg.get('error')
                assert isinstance(err, dict)
                assert 'cancel' in str(err.get('message', '')).lower()
                saw_run_err = True
    finally:
        await _stop_client(client, task)


@pytest.mark.asyncio
@pytest.mark.parametrize('stream_method', ('sendInputStreamChunk', 'endInputStream'))
async def test_reflection_server_v2_input_stream_not_implemented_js_style(
    fake_manager: FakeReflectionManager,
    stream_method: str,
) -> None:
    """Unimplemented input-stream methods return -32000 + data.stack when id is set (JS parity)."""
    registry = Registry()
    client, task = await _run_client_lifecycle(registry, fake_manager)
    try:
        await ack_register(fake_manager)
        await fake_manager.write_rpc({
            'jsonrpc': '2.0',
            'method': stream_method,
            'params': {},
            'id': 'stream-1',
        })
        resp = await fake_manager.read_rpc()
        err = resp.get('error')
        assert isinstance(err, dict)
        assert err.get('code') == JSON_RPC_SERVER_ERROR
        assert 'not implemented' in str(err.get('message', '')).lower()
        data = err.get('data')
        assert isinstance(data, dict)
        assert 'stack' in data and str(data.get('stack', '')).strip()
    finally:
        await _stop_client(client, task)


@pytest.mark.asyncio
async def test_reflection_server_v2_method_not_found(fake_manager: FakeReflectionManager) -> None:
    registry = Registry()
    client, task = await _run_client_lifecycle(registry, fake_manager)
    try:
        await ack_register(fake_manager)
        await fake_manager.write_rpc({
            'jsonrpc': '2.0',
            'method': 'unknownMethod',
            'id': '8',
        })
        resp = await fake_manager.read_rpc()
        err = resp.get('error')
        assert isinstance(err, dict)
        assert err.get('code') == JSON_RPC_METHOD_NOT_FOUND
    finally:
        await _stop_client(client, task)


@pytest.mark.asyncio
async def test_reflection_server_v2_omits_data_for_simple_errors(
    fake_manager: FakeReflectionManager,
) -> None:
    """Plain validation errors omit ``error.data`` to match JS / Go reflection-v2.

    JS's ``JSON.stringify`` drops ``undefined`` props and Go's struct uses
    ``json:",omitempty"`` on ``Data``, so ``sendError(id, code, message)`` with
    no extra payload produces a frame without a ``data`` key at all. Only
    handlers that assemble a Status-shaped payload (runAction errors) emit one.
    """

    registry = Registry()
    client, task = await _run_client_lifecycle(registry, fake_manager)
    try:
        await ack_register(fake_manager)

        await fake_manager.write_rpc({
            'jsonrpc': '2.0',
            'method': 'unknownMethod',
            'id': 'e1',
        })
        resp = await fake_manager.read_rpc()
        err = resp.get('error') or {}
        assert err.get('code') == JSON_RPC_METHOD_NOT_FOUND
        assert 'data' not in err, 'error.data must be omitted for plain JSON-RPC errors'

        await fake_manager.write_rpc({
            'jsonrpc': '2.0',
            'method': 'runAction',
            'params': {'key': '/model/missing', 'input': {}},
            'id': 'e2',
        })
        resp = await fake_manager.read_rpc()
        err = resp.get('error') or {}
        assert err.get('code') == JSON_RPC_INVALID_PARAMS
        assert 'not found' in str(err.get('message', '')).lower()
        assert 'data' not in err, 'error.data must be omitted when no Status payload is built'
    finally:
        await _stop_client(client, task)


def test_reflection_run_action_params_accepts_dev_ui_telemetry_labels() -> None:
    """Dev UI sends telemetryLabels as a string record (e.g. genkitx:ignore-trace)."""

    from genkit._core._typing import ReflectionRunActionParams

    p = ReflectionRunActionParams.model_validate({
        'key': '/executable-prompt/story',
        'telemetryLabels': {'genkitx:ignore-trace': 'true'},
    })
    assert p.telemetry_labels == {'genkitx:ignore-trace': 'true'}
