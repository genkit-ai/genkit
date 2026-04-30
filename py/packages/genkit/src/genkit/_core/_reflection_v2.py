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

"""Reflection API v2 (WebSocket JSON-RPC client) for Genkit Dev UI / CLI.

``runAction`` with ``stream: true`` emits ``streamChunk`` notifications (output streaming).
Bidirectional input streaming (``sendInputStreamChunk`` / ``endInputStream``) is not
implemented yet. Requests with an ``id`` receive JSON-RPC ``-32000`` with message
``Not implemented`` and ``error.data.stack`` (same pattern as JS ``throw`` in the handler).
Notifications without ``id`` are ignored except for a debug log.
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from typing import Any

import websockets
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from pydantic import BaseModel, JsonValue, ValidationError
from websockets.exceptions import ConnectionClosed

from genkit._core._constants import GENKIT_VERSION
from genkit._core._error import StatusCodes, get_reflection_json
from genkit._core._logger import get_logger
from genkit._core._registry import Registry
from genkit._core._trace._default_exporter import TraceServerExporter
from genkit._core._tracing import add_custom_exporter
from genkit._core._typing import (
    ReflectionCancelActionParams,
    ReflectionCancelActionResponse,
    ReflectionConfigureParams,
    ReflectionListValuesParams,
    ReflectionRegisterParams,
    ReflectionRunActionParams,
    ReflectionRunActionStateParams,
    ReflectionStreamChunkParams,
    State,
)

logger = get_logger(__name__)

GENKIT_REFLECTION_API_SPEC_VERSION = 1

JSON_RPC_METHOD_NOT_FOUND = -32601
JSON_RPC_INVALID_PARAMS = -32602
JSON_RPC_SERVER_ERROR = -32000

RECONNECT_BASE_DELAY_S = 0.5
RECONNECT_MAX_DELAY_S = 5.0

WRITE_TIMEOUT_S = 5.0


class JsonRpcCallError(Exception):
    """Error returned in a JSON-RPC response for a request we originated."""

    def __init__(self, code: int, message: str, data: object | None = None) -> None:
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f'JSON-RPC error {code}: {message}')


def _chunk_for_json(chunk: object) -> object:
    if isinstance(chunk, BaseModel):
        return json.loads(chunk.model_dump_json())
    return chunk


class ReflectionServerV2:
    """WebSocket client that connects to the CLI reflection manager (RuntimeManagerV2).

    See module docstring for streaming support scope.
    """

    def __init__(
        self,
        registry: Registry,
        ws_url: str,
        *,
        app_name: str | None = None,
    ) -> None:
        self._registry = registry
        self._ws_url = ws_url
        self._app_name = app_name
        self._ws: Any = None
        self._write_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future[JsonValue]] = {}
        self._request_seq = 0
        self._active_actions: dict[str, asyncio.Task[Any]] = {}
        self._stop = False
        self._reflection_handshake_telemetry_applied = False

    def _apply_handshake_telemetry(self, url: str | None) -> None:
        """Use the Dev UI trace server URL from the reflection handshake.

        The CLI manager returns ``telemetryServerUrl`` on ``register`` and may send it
        again on ``configure``. We need that base URL so OpenTelemetry spans can be
        POSTed to ``{url}/api/traces`` (see ``TraceServerExporter``).
        """
        if not url or os.environ.get('GENKIT_TELEMETRY_SERVER'):
            return
        if self._reflection_handshake_telemetry_applied:
            return
        self._reflection_handshake_telemetry_applied = True
        # Register HTTP export to this URL on the global OTel provider.
        add_custom_exporter(TraceServerExporter(telemetry_server_url=url), 'reflection_v2_telemetry')
        logger.debug('reflection V2: connected to telemetry server', url=url)

    async def run_forever(self) -> None:
        """Connect, handle requests, reconnect with backoff until stop() or process exit."""
        attempt = 0
        while not self._stop:
            try:
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    self._ws = ws
                    attempt = 0
                    _ = asyncio.create_task(self._register())
                    await self._read_loop()
            except ConnectionClosed as e:
                logger.debug('reflection V2: connection closed', code=e.code, reason=e.reason)
            except OSError as e:
                logger.debug('reflection V2: connection error', err=e)
            finally:
                self._ws = None
                self._drain_pending(ConnectionError('connection closed'))

            if self._stop:
                return

            delay = min(RECONNECT_BASE_DELAY_S * (2**attempt), RECONNECT_MAX_DELAY_S)
            attempt += 1
            logger.debug('reflection V2: reconnect scheduled', delay_s=delay, attempt=attempt)
            await asyncio.sleep(delay)

    def stop(self) -> None:
        self._stop = True

    def _drain_pending(self, exc: BaseException) -> None:
        for _rid, fut in list(self._pending.items()):
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()

    async def _send_message(self, message: dict[str, Any]) -> None:
        if self._ws is None:
            raise ConnectionError('websocket not connected')
        raw = json.dumps(message, default=str)
        async with self._write_lock:
            await asyncio.wait_for(self._ws.send(raw), timeout=WRITE_TIMEOUT_S)

    async def _send_response(self, req_id: str, result: object) -> None:
        await self._send_message({'jsonrpc': '2.0', 'result': result, 'id': req_id})

    async def _send_error(
        self,
        req_id: str,
        code: int,
        message: str,
        data: object | None = None,
    ) -> None:
        err: dict[str, Any] = {'code': code, 'message': message}
        if data is not None:
            err['data'] = data
        await self._send_message({'jsonrpc': '2.0', 'error': err, 'id': req_id})

    async def _send_notification(self, method: str, params: object) -> None:
        await self._send_message({'jsonrpc': '2.0', 'method': method, 'params': params})

    async def _send_request(self, method: str, params: object) -> JsonValue:
        self._request_seq += 1
        req_id = str(self._request_seq)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[JsonValue] = loop.create_future()
        self._pending[req_id] = fut
        try:
            await self._send_message({'jsonrpc': '2.0', 'id': req_id, 'method': method, 'params': params})
            return await fut
        finally:
            self._pending.pop(req_id, None)

    async def _register(self) -> None:
        runtime_id = os.environ.get('GENKIT_RUNTIME_ID') or str(os.getpid())
        name = self._app_name or runtime_id
        params = ReflectionRegisterParams(
            id=runtime_id,
            pid=float(os.getpid()),
            name=name,
            genkit_version='py/' + GENKIT_VERSION,
            reflection_api_spec_version=float(GENKIT_REFLECTION_API_SPEC_VERSION),
            envs=['dev'],
        ).model_dump(by_alias=True, exclude_none=True)
        try:
            result = await self._send_request('register', params)
            if isinstance(result, dict) and (telemetry_url := result.get('telemetryServerUrl')):
                self._apply_handshake_telemetry(str(telemetry_url))
        except JsonRpcCallError as e:
            logger.error('reflection V2: register failed', code=e.code, message=e.message)
        except Exception as e:
            logger.error('reflection V2: register failed', err=e)

    async def _read_loop(self) -> None:
        assert self._ws is not None
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.debug('reflection V2: invalid JSON from manager')
                continue
            if not isinstance(msg, dict):
                logger.debug('reflection V2: ignoring JSON value that is not an object', type=type(msg).__name__)
                continue
            if msg.get('jsonrpc') != '2.0':
                logger.debug(
                    'reflection V2: ignoring frame without jsonrpc 2.0',
                    jsonrpc=msg.get('jsonrpc'),
                )
                continue
            if 'method' in msg:
                _ = asyncio.create_task(self._dispatch_incoming(msg))
            elif msg.get('id') is not None:
                self._deliver_response(msg)
            else:
                logger.debug(
                    'reflection V2: ignoring JSON-RPC 2.0 object without method or id',
                    keys=list(msg.keys()),
                )

    def _deliver_response(self, msg: dict[str, Any]) -> None:
        req_id = msg.get('id')
        if req_id is None:
            return
        sid = str(req_id)
        fut = self._pending.pop(sid, None)
        if fut is None:
            logger.debug('reflection V2: response for unknown id', id=sid)
            return
        if err := msg.get('error'):
            fut.set_exception(
                JsonRpcCallError(
                    int(err.get('code', JSON_RPC_SERVER_ERROR)),
                    str(err.get('message', '')),
                    err.get('data'),
                )
            )
        else:
            fut.set_result(msg.get('result'))

    async def _dispatch_incoming(self, msg: dict[str, Any]) -> None:
        method = msg.get('method')
        req_id = msg.get('id')
        params = msg.get('params') or {}
        if not isinstance(params, dict):
            if req_id is not None:
                await self._send_error(
                    str(req_id),
                    JSON_RPC_INVALID_PARAMS,
                    'params must be a JSON object',
                )
            return
        try:
            if method == 'listActions':
                await self._handle_list_actions(req_id, params)
            elif method == 'listValues':
                await self._handle_list_values(req_id, params)
            elif method == 'runAction':
                await self._handle_run_action(req_id, params)
            elif method == 'cancelAction':
                await self._handle_cancel_action(req_id, params)
            elif method == 'configure':
                self._handle_configure(params)
            elif method in ('sendInputStreamChunk', 'endInputStream'):
                await self._handle_input_stream_unimplemented(req_id, method)
            else:
                if req_id is not None:
                    await self._send_error(
                        str(req_id),
                        JSON_RPC_METHOD_NOT_FOUND,
                        f'method not found: {method}',
                    )
                else:
                    logger.debug('reflection V2: unknown notification', method=method)
        except Exception:
            logger.exception('reflection V2: handler error', method=method)
            if req_id is not None:
                await self._send_error(str(req_id), JSON_RPC_SERVER_ERROR, 'internal error')

    async def _handle_input_stream_unimplemented(self, req_id: str | int | None, method: str) -> None:
        if req_id is None:
            logger.debug('reflection V2: input stream method not implemented (notification)', method=method)
            return
        try:
            raise NotImplementedError('Not implemented')
        except NotImplementedError as e:
            stack = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            await self._send_error(
                str(req_id),
                JSON_RPC_SERVER_ERROR,
                str(e) or 'Not implemented',
                {'stack': stack},
            )

    async def _handle_list_actions(self, req_id: str | int | None, _: dict[str, Any]) -> None:
        if req_id is None:
            return
        sid = str(req_id)
        catalog = await self._registry.list_actions()
        actions: dict[str, dict[str, Any]] = {}
        for key, meta in catalog.items():
            row: dict[str, Any] = {
                'key': key,
                'name': meta.name,
                'actionType': str(meta.action_type) if meta.action_type is not None else None,
                'description': meta.description,
                'metadata': meta.metadata,
                'inputSchema': meta.input_schema if meta.input_schema is not None else meta.input_json_schema,
                'outputSchema': meta.output_schema if meta.output_schema is not None else meta.output_json_schema,
                'streamSchema': meta.stream_schema,
            }
            actions[key] = {k: v for k, v in row.items() if v is not None}
        await self._send_response(sid, {'actions': actions})

    async def _handle_list_values(self, req_id: str | int | None, params: dict[str, Any]) -> None:
        if req_id is None:
            return
        sid = str(req_id)
        try:
            p = ReflectionListValuesParams.model_validate(params)
        except ValidationError as e:
            await self._send_error(sid, JSON_RPC_INVALID_PARAMS, f'invalid params: {e}')
            return
        if p.type not in ('defaultModel', 'middleware'):
            await self._send_error(
                sid,
                JSON_RPC_INVALID_PARAMS,
                f"'type' {p.type} is not supported. Only 'defaultModel' and 'middleware' are supported",
            )
            return
        mapped: dict[str, Any] = {}
        for name in self._registry.list_values(p.type):
            value = self._registry.lookup_value(p.type, name)
            to_json_fn = getattr(value, 'to_json', None) if value is not None else None
            if callable(to_json_fn):
                mapped[name] = to_json_fn()
            else:
                mapped[name] = value
        await self._send_response(sid, {'values': mapped})

    def _handle_configure(self, params: dict[str, Any]) -> None:
        try:
            p = ReflectionConfigureParams.model_validate(params)
        except ValidationError as e:
            logger.error('reflection V2: invalid configure params', err=e)
            return
        if p.telemetry_server_url:
            self._apply_handshake_telemetry(p.telemetry_server_url)

    async def _handle_cancel_action(self, req_id: str | int | None, params: dict[str, Any]) -> None:
        if req_id is None:
            return
        sid = str(req_id)
        try:
            p = ReflectionCancelActionParams.model_validate(params)
        except ValidationError as e:
            await self._send_error(sid, JSON_RPC_INVALID_PARAMS, f'invalid params: {e}')
            return
        if not p.trace_id:
            await self._send_error(sid, JSON_RPC_INVALID_PARAMS, 'traceId is required')
            return
        task = self._active_actions.get(p.trace_id)
        if task:
            task.cancel()
            self._active_actions.pop(p.trace_id, None)
            body = ReflectionCancelActionResponse(message='Action cancelled').model_dump(by_alias=True)
            await self._send_response(sid, body)
        else:
            await self._send_error(
                sid,
                JSON_RPC_INVALID_PARAMS,
                'Action not found or already completed',
            )

    async def _flush_tracing(self) -> None:
        provider = trace_api.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            await asyncio.to_thread(provider.force_flush)

    async def _handle_run_action(self, req_id: str | int | None, params: dict[str, Any]) -> None:
        if req_id is None:
            return
        sid = str(req_id)
        try:
            p = ReflectionRunActionParams.model_validate(params)
        except ValidationError as e:
            await self._send_error(sid, JSON_RPC_INVALID_PARAMS, f'invalid params: {e}')
            return

        action = await self._registry.resolve_action_by_key(p.key)
        if not action:
            await self._send_error(sid, JSON_RPC_INVALID_PARAMS, f'action {p.key} not found')
            return

        if p.context is not None and not isinstance(p.context, dict):
            await self._send_error(
                sid,
                JSON_RPC_INVALID_PARAMS,
                'context must be a JSON object when provided',
            )
            return

        stream = bool(p.stream)
        trace_holder: list[str | None] = [None]
        stream_chunk_tasks: list[asyncio.Task[Any]] = []

        async def on_trace_start(tid: str, span_id: str) -> None:
            trace_holder[0] = tid
            if t := asyncio.current_task():
                self._active_actions[tid] = t
            st = ReflectionRunActionStateParams(
                request_id=sid,
                state=State(trace_id=tid),
            ).model_dump(by_alias=True, exclude_none=True)
            await self._send_notification('runActionState', st)

        on_chunk = None
        if stream:

            def on_chunk_fn(chunk: object) -> None:
                chunk_payload = ReflectionStreamChunkParams(
                    request_id=sid,
                    chunk=_chunk_for_json(chunk),
                ).model_dump(by_alias=True, exclude_none=True)
                stream_chunk_tasks.append(asyncio.create_task(self._send_notification('streamChunk', chunk_payload)))

            on_chunk = on_chunk_fn

        ctx: dict[str, object] = {} if p.context is None else {str(k): v for k, v in p.context.items()}

        labels: dict[str, object] | None = None
        if p.telemetry_labels is not None:
            dumped = p.telemetry_labels.model_dump(exclude_none=True)
            labels = {str(k): v for k, v in dumped.items()} if dumped else None

        try:
            output = await action.run(
                input=p.input,
                on_chunk=on_chunk,
                context=ctx or None,
                on_trace_start=on_trace_start,
                telemetry_labels=labels,
            )
            if stream_chunk_tasks:
                await asyncio.gather(*stream_chunk_tasks)
            await self._flush_tracing()
            result_body: object
            if isinstance(output.response, BaseModel):
                result_body = output.response.model_dump(by_alias=True, exclude_none=True)
            else:
                result_body = output.response
            await self._send_response(
                sid,
                {'result': result_body, 'telemetry': {'traceId': output.trace_id}},
            )
        except asyncio.CancelledError:
            err_details: dict[str, Any] = {}
            if trace_holder[0]:
                err_details['traceId'] = trace_holder[0]
            err_data: dict[str, Any] = {
                'code': StatusCodes.CANCELLED.value,
                'message': 'Action was cancelled',
            }
            if err_details:
                err_data['details'] = err_details
            await self._send_error(sid, JSON_RPC_SERVER_ERROR, 'Action was cancelled', err_data)
            return
        except Exception as e:
            logger.exception('reflection V2: runAction error')
            ref = get_reflection_json(e)
            err_data = {'code': ref.code, 'message': ref.message}
            details_map: dict[str, Any] = {}
            if ref.details:
                d = ref.details.model_dump(by_alias=True, exclude_none=True)
                details_map = {k: v for k, v in d.items() if v is not None}
            if trace_holder[0]:
                details_map['traceId'] = trace_holder[0]
            if e.__traceback__ and not details_map.get('stack'):
                details_map['stack'] = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            if details_map:
                err_data['details'] = details_map
            await self._send_error(sid, JSON_RPC_SERVER_ERROR, ref.message, err_data)
        finally:
            tid = trace_holder[0]
            if tid:
                self._active_actions.pop(tid, None)
