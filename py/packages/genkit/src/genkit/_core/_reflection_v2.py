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

This connects out to the CLI's reflection manager and answers the JSON-RPC
requests it sends. The methods it handles are:

- ``listActions`` / ``listValues`` — enumerate what the app exposes.
- ``runAction`` — run one action. With ``stream: true`` the output comes back
  as ``streamChunk`` notifications while the action runs.
- ``cancelAction`` — stop an in-flight run.
- ``configure`` — set up the connection.
- ``sendInputStreamChunk`` / ``endInputStream`` — feed and close a streamed
  *input*. Agents (bidi actions) use these: the client drives a turn by pushing
  input chunks and ending the stream, on top of the streamed output above.
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

import websockets
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from pydantic import BaseModel, JsonValue, ValidationError
from websockets.exceptions import ConnectionClosed

from genkit._core._action import Action, BidiAction
from genkit._core._channel import CloseableQueue
from genkit._core._constants import GENKIT_VERSION
from genkit._core._error import ReflectionError, ReflectionErrorDetails, StatusCodes, get_reflection_json
from genkit._core._logger import get_logger
from genkit._core._middleware import GenerateMiddleware
from genkit._core._model import ModelRef
from genkit._core._reflection import as_agent_input_dict, resolve_agent_init
from genkit._core._registry import Registry
from genkit._core._trace._default_exporter import TraceServerExporter
from genkit._core._trace._log_exporter import enable_log_export
from genkit._core._tracing import add_custom_exporter
from genkit._core._typing import (
    AgentInput,
    ReflectionCancelActionParams,
    ReflectionCancelActionResponse,
    ReflectionConfigureParams,
    ReflectionEndInputStreamParams,
    ReflectionListValuesParams,
    ReflectionRegisterParams,
    ReflectionRunActionParams,
    ReflectionRunActionStateParams,
    ReflectionSendInputStreamChunkParams,
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


def coerce_json_rpc_message(message: object) -> str:
    """JSON-RPC and RuntimeManagerV2 require ``error.message`` to be a string."""
    if isinstance(message, str):
        return message
    if message is None:
        return 'Unknown error'
    try:
        return json.dumps(message, default=str)
    except TypeError:
        return str(message)


class JsonRpcCallError(Exception):
    """Error returned in a JSON-RPC response for a request we originated."""

    def __init__(self, code: int, message: str, data: object | None = None) -> None:
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f'JSON-RPC error {code}: {message}')


def chunk_for_json(chunk: object) -> object:
    if isinstance(chunk, BaseModel):
        return json.loads(chunk.model_dump_json(by_alias=True, exclude_none=True))
    return chunk


def omit_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if v is not None}


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
        self.registry = registry
        self.ws_url = ws_url
        self.app_name = app_name
        self.ws: Any = None
        self.write_lock = asyncio.Lock()
        self.pending: dict[str, asyncio.Future[JsonValue]] = {}
        self.request_seq = 0
        self.active_actions: dict[str, asyncio.Task[Any]] = {}
        # Fire-and-forget register/dispatch tasks. Held so the event loop can't
        # GC them mid-flight (asyncio only weakly references tasks) and so they
        # can be cancelled when the connection drops.
        self.background_tasks: set[asyncio.Task[Any]] = set()
        # request_id → live input stream feeding an active bidi (agent) run.
        # sendInputStreamChunk puts turns on it; endInputStream closes it.
        self.bidi_input_streams: dict[str, CloseableQueue[Any]] = {}
        self.stopped = False
        self.reflection_handshake_telemetry_applied = False

    def apply_handshake_telemetry(self, url: str | None) -> None:
        """Use the Dev UI trace server URL from the reflection handshake.

        The CLI manager returns ``telemetryServerUrl`` on ``register`` and may send it
        again on ``configure``. We need that base URL so OpenTelemetry spans can be
        POSTed to ``{url}/api/traces`` (see ``TraceServerExporter``).
        """
        if not url or os.environ.get('GENKIT_TELEMETRY_SERVER'):
            return
        if self.reflection_handshake_telemetry_applied:
            return
        self.reflection_handshake_telemetry_applied = True
        # Register HTTP export to this URL on the global OTel provider.
        add_custom_exporter(TraceServerExporter(telemetry_server_url=url), 'reflection_v2_telemetry')
        enable_log_export(url=url)
        logger.debug('reflection V2: connected to telemetry server', url=url)

    async def run_forever(self) -> None:
        """Connect, handle requests, reconnect with backoff until stop() or process exit."""
        attempt = 0
        while not self.stopped:
            try:
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    self.ws = ws
                    attempt = 0
                    self.spawn(self.register())
                    await self.read_loop()
            except ConnectionClosed as e:
                logger.debug('reflection V2: connection closed', code=e.code, reason=e.reason)
            except OSError as e:
                logger.debug('reflection V2: connection error', err=e)
            finally:
                self.ws = None
                self.drain_pending(ConnectionError('connection closed'))
                # Cancel in-flight register/dispatch handlers so they don't keep
                # running against a dead socket after the connection drops.
                for t in list(self.background_tasks):
                    t.cancel()
                # Close each live input stream so its run's feeder ends the turn
                # loop instead of hanging waiting for turns that can't arrive.
                for _rid, stream in list(self.bidi_input_streams.items()):
                    stream.close()
                self.bidi_input_streams.clear()

            if self.stopped:
                return

            delay = min(RECONNECT_BASE_DELAY_S * (2**attempt), RECONNECT_MAX_DELAY_S)
            attempt += 1
            logger.debug('reflection V2: reconnect scheduled', delay_s=delay, attempt=attempt)
            await asyncio.sleep(delay)

    def stop(self) -> None:
        self.stopped = True

    def spawn(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Run a fire-and-forget coroutine while keeping a reference to its task."""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.on_background_task_done)

    def on_background_task_done(self, task: asyncio.Task[Any]) -> None:
        self.background_tasks.discard(task)
        # Retrieve any exception so it isn't reported as "never retrieved".
        if not task.cancelled() and (exc := task.exception()) is not None:
            logger.debug('reflection V2: background task error', err=exc)

    def drain_pending(self, exc: Exception) -> None:
        for _rid, fut in list(self.pending.items()):
            if not fut.done():
                fut.set_exception(exc)
        self.pending.clear()

    async def send_message(self, message: dict[str, Any]) -> None:
        if self.ws is None:
            raise ConnectionError('websocket not connected')
        raw = json.dumps(message, default=str)
        async with self.write_lock:
            await asyncio.wait_for(self.ws.send(raw), timeout=WRITE_TIMEOUT_S)

    async def send_response(self, req_id: str, result: object) -> None:
        await self.send_message({'jsonrpc': '2.0', 'result': result, 'id': req_id})

    async def send_error(
        self,
        req_id: str,
        code: int,
        message: object,
        data: object | None = None,
    ) -> None:
        """Emit a JSON-RPC error."""
        err: dict[str, Any] = {'code': code, 'message': coerce_json_rpc_message(message)}
        if data is not None:
            err['data'] = data
        await self.send_message({'jsonrpc': '2.0', 'error': err, 'id': req_id})

    async def send_notification(self, method: str, params: object) -> None:
        await self.send_message({'jsonrpc': '2.0', 'method': method, 'params': params})

    async def send_request(self, method: str, params: object) -> JsonValue:
        self.request_seq += 1
        req_id = str(self.request_seq)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[JsonValue] = loop.create_future()
        self.pending[req_id] = fut
        try:
            await self.send_message({'jsonrpc': '2.0', 'id': req_id, 'method': method, 'params': params})
            return await fut
        finally:
            self.pending.pop(req_id, None)

    async def register(self) -> None:
        runtime_id = os.environ.get('GENKIT_RUNTIME_ID') or str(os.getpid())
        name = self.app_name or runtime_id
        params = ReflectionRegisterParams(
            id=runtime_id,
            pid=float(os.getpid()),
            name=name,
            genkit_version='py/' + GENKIT_VERSION,
            reflection_api_spec_version=float(GENKIT_REFLECTION_API_SPEC_VERSION),
            envs=['dev'],
        ).model_dump(by_alias=True, exclude_none=True)
        try:
            result = await self.send_request('register', params)
            if isinstance(result, dict) and (telemetry_url := result.get('telemetryServerUrl')):
                self.apply_handshake_telemetry(str(telemetry_url))
        except JsonRpcCallError as e:
            logger.error('reflection V2: register failed', code=e.code, message=e.message)
        except Exception as e:
            logger.error('reflection V2: register failed', err=e)

    async def read_loop(self) -> None:
        assert self.ws is not None
        async for raw in self.ws:
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
                self.spawn(self.dispatch_incoming(msg))
            elif msg.get('id') is not None:
                self.deliver_response(msg)
            else:
                logger.debug(
                    'reflection V2: ignoring JSON-RPC 2.0 object without method or id',
                    keys=list(msg.keys()),
                )

    def deliver_response(self, msg: dict[str, Any]) -> None:
        req_id = msg.get('id')
        if req_id is None:
            return
        sid = str(req_id)
        fut = self.pending.pop(sid, None)
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

    async def dispatch_incoming(self, msg: dict[str, Any]) -> None:
        method = msg.get('method')
        req_id = msg.get('id')
        params = msg.get('params') or {}
        if not isinstance(params, dict):
            if req_id is not None:
                await self.send_error(
                    str(req_id),
                    JSON_RPC_INVALID_PARAMS,
                    'params must be a JSON object',
                )
            return
        try:
            if method == 'listActions':
                await self.handle_list_actions(req_id, params)
            elif method == 'listValues':
                await self.handle_list_values(req_id, params)
            elif method == 'runAction':
                await self.handle_run_action(req_id, params)
            elif method == 'cancelAction':
                await self.handle_cancel_action(req_id, params)
            elif method == 'configure':
                self.handle_configure(params)
            elif method == 'sendInputStreamChunk':
                await self.handle_send_input_stream_chunk(req_id, params)
            elif method == 'endInputStream':
                await self.handle_end_input_stream(req_id, params)
            else:
                if req_id is not None:
                    await self.send_error(
                        str(req_id),
                        JSON_RPC_METHOD_NOT_FOUND,
                        f'method not found: {method}',
                    )
                else:
                    logger.debug('reflection V2: unknown notification', method=method)
        except Exception as e:
            logger.error(f'Reflection error in {method}: {type(e).__name__}: {e}')
            logger.debug('reflection V2: handler error', method=method, exc_info=e)
            if req_id is not None:
                await self.send_error(str(req_id), JSON_RPC_SERVER_ERROR, 'internal error')

    async def handle_send_input_stream_chunk(self, req_id: str | int | None, params: dict[str, Any]) -> None:
        """Feed a per-turn input chunk into an active bidi (agent) session."""
        try:
            p = ReflectionSendInputStreamChunkParams.model_validate(params)
        except Exception as e:  # noqa: BLE001
            if req_id is not None:
                await self.send_error(str(req_id), JSON_RPC_INVALID_PARAMS, f'invalid params: {e}')
            return

        stream = self.bidi_input_streams.get(p.request_id)
        if stream is None:
            # A chunk for a requestId with no live turn means the client is writing
            # to a turn that already ended (or never started). Surface it as an
            # INVALID_PARAMS error so a mis-wired Dev UI notices, same as the
            # bad-params branch above.
            if req_id is not None:
                await self.send_error(
                    str(req_id),
                    JSON_RPC_INVALID_PARAMS,
                    f'no active bidi session for requestId {p.request_id!r}',
                )
            return

        try:
            if p.chunk is None:
                inp = AgentInput()
            else:
                inp = AgentInput.model_validate(as_agent_input_dict(p.chunk))
            await stream.put(inp)
        except Exception as e:  # noqa: BLE001
            logger.warning('reflection V2: sendInputStreamChunk error', err=e)

    async def handle_end_input_stream(self, req_id: str | int | None, params: dict[str, Any]) -> None:
        """Close the input stream for an active bidi (agent) session."""
        try:
            p = ReflectionEndInputStreamParams.model_validate(params)
        except Exception as e:  # noqa: BLE001
            if req_id is not None:
                await self.send_error(str(req_id), JSON_RPC_INVALID_PARAMS, f'invalid params: {e}')
            return

        stream = self.bidi_input_streams.get(p.request_id)
        if stream is None:
            return  # already gone or never existed — no-op
        stream.close()

    async def flush_tracing(self) -> None:
        provider = trace_api.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            await asyncio.to_thread(provider.force_flush)

    @staticmethod
    def run_action_call_options(
        p: ReflectionRunActionParams,
    ) -> tuple[dict[str, object], dict[str, object] | None]:
        """Context and telemetry labels shared by one-shot and bidi runAction paths."""
        ctx = {} if p.context is None else {str(k): v for k, v in p.context.items()}
        labels: dict[str, object] | None = None
        if p.telemetry_labels is not None:
            labels = {str(k): v for k, v in p.telemetry_labels.items()}
        return ctx, labels

    async def notify_run_action_state(self, sid: str, trace_id: str) -> None:
        st = ReflectionRunActionStateParams(
            request_id=sid,
            state=State(trace_id=trace_id),
        ).model_dump(by_alias=True, exclude_none=True)
        await self.send_notification('runActionState', st)

    def trace_start_callback(
        self,
        sid: str,
        trace_holder: list[str | None],
        *,
        register_for_cancel: bool,
    ) -> Callable[[str, str], Awaitable[None]]:
        async def on_trace_start(tid: str, span_id: str) -> None:
            trace_holder[0] = tid
            if register_for_cancel and (t := asyncio.current_task()):
                self.active_actions[tid] = t
            await self.notify_run_action_state(sid, tid)

        return on_trace_start

    async def notify_stream_chunk(self, sid: str, chunk: object) -> None:
        payload = ReflectionStreamChunkParams(
            request_id=sid,
            chunk=chunk_for_json(chunk),
        ).model_dump(by_alias=True, exclude_none=True)
        await self.send_notification('streamChunk', payload)

    @staticmethod
    def run_action_success_body(result: object, trace_id: str | None) -> dict[str, Any]:
        if isinstance(result, BaseModel):
            result_body = result.model_dump(by_alias=True, exclude_none=True)
        else:
            result_body = result
        body: dict[str, Any] = {'result': result_body}
        if trace_id:
            body['telemetry'] = {'traceId': trace_id}
        return body

    async def send_run_action_error(
        self,
        sid: str,
        exc: BaseException,
        trace_holder: list[str | None],
    ) -> None:
        """Map a runAction failure to the JSON-RPC error shape the Dev UI expects."""
        if isinstance(exc, asyncio.CancelledError):
            err_details: dict[str, Any] = {}
            if trace_holder[0]:
                err_details['traceId'] = trace_holder[0]
            err_data: dict[str, Any] = {
                'code': StatusCodes.CANCELLED.value,
                'message': 'Action was cancelled',
            }
            if err_details:
                err_data['details'] = err_details
            await self.send_error(sid, JSON_RPC_SERVER_ERROR, 'Action was cancelled', err_data)
            return

        # Dev UI already shows the error + stack from the JSON-RPC response.
        logger.debug('Action failed: %s: %s', type(exc).__name__, exc, exc_info=True)
        # Wire contract requires ``details`` to carry only ``stack`` and ``traceId``
        # (see ``GenkitErrorSchema.data.genkitErrorDetails`` in genkit-tools); anything
        # else in ``GenkitError.details`` is runtime-internal and gets dropped.
        ref = get_reflection_json(exc)
        stack = ref.details.stack if ref.details else None
        if not stack and exc.__traceback__:
            stack = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        tid = trace_holder[0] or (ref.details.trace_id if ref.details else None)
        status = ReflectionError(
            code=ref.code,
            message=coerce_json_rpc_message(ref.message),
            details=ReflectionErrorDetails(stack=stack, trace_id=tid) if (stack or tid) else None,
        )
        await self.send_error(
            sid,
            JSON_RPC_SERVER_ERROR,
            status.message,
            status.model_dump(by_alias=True, exclude_none=True),
        )

    async def respond_run_action_success(
        self,
        sid: str,
        result: object,
        trace_id: str | None,
    ) -> None:
        await self.flush_tracing()
        await self.send_response(sid, self.run_action_success_body(result, trace_id))

    async def run_action(
        self,
        sid: str,
        p: ReflectionRunActionParams,
        action: Action[Any, Any, Any],
    ) -> None:
        """Execute a one-shot action and stream the runAction JSON-RPC response."""
        stream = bool(p.stream)
        trace_holder: list[str | None] = [None]
        stream_chunk_tasks: list[asyncio.Task[Any]] = []
        on_trace_start = self.trace_start_callback(sid, trace_holder, register_for_cancel=True)

        on_chunk = None
        if stream:

            def on_chunk_fn(chunk: object) -> None:
                # Chunks reach the client in order because tasks start in creation
                # order and send_message serializes on a FIFO lock with no await
                # before it — keep it that way, or streamed output can reorder.
                stream_chunk_tasks.append(asyncio.create_task(self.notify_stream_chunk(sid, chunk)))

            on_chunk = on_chunk_fn

        ctx, labels = self.run_action_call_options(p)

        async def drain_chunks() -> None:
            if stream_chunk_tasks:
                await asyncio.gather(*stream_chunk_tasks, return_exceptions=True)

        try:
            output = await action.run(
                input=p.input,
                on_chunk=on_chunk,
                context=ctx or None,
                on_trace_start=on_trace_start,
                telemetry_labels=labels,
            )
            await drain_chunks()
            await self.respond_run_action_success(
                sid,
                output.response,
                output.trace_id or trace_holder[0],
            )
        except (asyncio.CancelledError, Exception) as e:
            await drain_chunks()
            await self.send_run_action_error(sid, e, trace_holder)
            # Report the cancellation to the Dev UI, then let it propagate so the
            # task actually winds down (swallowing it fakes a clean completion and
            # breaks cooperative cancellation on shutdown).
            if isinstance(e, asyncio.CancelledError):
                raise
        finally:
            tid = trace_holder[0]
            if tid:
                self.active_actions.pop(tid, None)

    async def run_bidi_action(
        self,
        sid: str,
        p: ReflectionRunActionParams,
        action: BidiAction,
    ) -> None:
        """Drive a bidi (agent) runAction through action.run() with a per-turn input stream.

        A one-shot call passes the single resolved input; a ``streamInput`` call
        registers a live stream under ``sid`` so ``sendInputStreamChunk`` /
        ``endInputStream`` can feed and close it while the run is in flight.
        Output chunks stream back as ``streamChunk`` notifications, then the fn's
        return value becomes the final runAction response.
        """
        try:
            init = resolve_agent_init(action, p.init)
        except Exception as e:  # noqa: BLE001
            await self.send_error(sid, JSON_RPC_INVALID_PARAMS, f'invalid AgentInit input: {e}')
            return

        ctx, labels = self.run_action_call_options(p)
        trace_holder: list[str | None] = [None]
        on_trace_start = self.trace_start_callback(sid, trace_holder, register_for_cancel=True)

        stream_chunk_tasks: list[asyncio.Task[Any]] = []

        def on_chunk(chunk: object) -> None:
            # Ordering matches the one-shot path: tasks start in FIFO order and
            # send_message serializes with no await before it, so chunks reach
            # the client in emission order.
            stream_chunk_tasks.append(asyncio.create_task(self.notify_stream_chunk(sid, chunk)))

        async def drain_chunks() -> None:
            if stream_chunk_tasks:
                await asyncio.gather(*stream_chunk_tasks, return_exceptions=True)

        input_val: AgentInput | None = None
        input_stream: CloseableQueue[Any] | None = None
        if p.stream_input:
            # Register before the run starts so sendInputStreamChunk can find the
            # stream while run() is in flight (the client waits for runActionState
            # before sending turns, and that fires from on_trace_start inside run).
            input_stream = CloseableQueue()
            self.bidi_input_streams[sid] = input_stream
        else:
            try:
                if p.input is None:
                    input_val = AgentInput()
                else:
                    input_val = AgentInput.model_validate(as_agent_input_dict(p.input))
            except (TypeError, ValidationError) as e:
                await self.send_error(sid, JSON_RPC_INVALID_PARAMS, f'invalid AgentInput: {e}')
                return

        try:
            output = await action.run(
                input=input_val,
                input_stream=input_stream,
                init=init,
                on_chunk=on_chunk,
                context=ctx or None,
                on_trace_start=on_trace_start,
                telemetry_labels=labels,
            )
            await drain_chunks()
            await self.respond_run_action_success(
                sid,
                output.response,
                output.trace_id or trace_holder[0],
            )
        except (asyncio.CancelledError, Exception) as e:
            await drain_chunks()
            await self.send_run_action_error(sid, e, trace_holder)
            # Report the cancellation to the Dev UI, then let it propagate so the
            # task actually winds down (swallowing it fakes a clean completion and
            # breaks cooperative cancellation on shutdown).
            if isinstance(e, asyncio.CancelledError):
                raise
        finally:
            self.bidi_input_streams.pop(sid, None)
            # Drop the cancel registration too, or a finished turn's trace id
            # lingers in active_actions and a late cancelAction would falsely
            # report success against a task that already completed.
            tid = trace_holder[0]
            if tid:
                self.active_actions.pop(tid, None)

    async def handle_list_actions(self, req_id: str | int | None, _: dict[str, Any]) -> None:
        if req_id is None:
            return
        sid = str(req_id)
        catalog = await self.registry.list_actions()
        actions = {
            key: omit_none({
                'key': key,
                'name': meta.name,
                'actionType': meta.action_type,
                'description': meta.description,
                'metadata': meta.metadata,
                'inputSchema': meta.input_schema or meta.input_json_schema,
                'outputSchema': meta.output_schema or meta.output_json_schema,
            })
            for key, meta in catalog.items()
        }
        await self.send_response(sid, {'actions': actions})

    async def handle_list_values(self, req_id: str | int | None, params: dict[str, Any]) -> None:
        if req_id is None:
            return
        sid = str(req_id)
        try:
            p = ReflectionListValuesParams.model_validate(params)
        except ValidationError as e:
            await self.send_error(sid, JSON_RPC_INVALID_PARAMS, f'invalid params: {e}')
            return
        if p.type not in ('defaultModel', 'middleware'):
            await self.send_error(
                sid,
                JSON_RPC_INVALID_PARAMS,
                f"'type' {p.type} is not supported. Only 'defaultModel' and 'middleware' are supported",
            )
            return
        mapped: dict[str, Any] = {}
        for name in self.registry.list_values(p.type):
            value = self.registry.lookup_value(p.type, name)
            if p.type == 'middleware':
                assert isinstance(value, GenerateMiddleware), (
                    f'registry middleware/{name!r} must be GenerateMiddleware, got {type(value).__name__}'
                )
                mapped[name] = value.model_dump(by_alias=True, exclude_none=True, mode='json')
            else:
                # Dev UI lists a model name. A stored ModelRef is an object;
                # only the name is JSON-serializable here.
                mapped[name] = value.name if isinstance(value, ModelRef) else value
        await self.send_response(sid, {'values': mapped})

    def handle_configure(self, params: dict[str, Any]) -> None:
        try:
            p = ReflectionConfigureParams.model_validate(params)
        except ValidationError as e:
            logger.error('reflection V2: invalid configure params', err=e)
            return
        if p.telemetry_server_url:
            self.apply_handshake_telemetry(p.telemetry_server_url)

    async def handle_cancel_action(self, req_id: str | int | None, params: dict[str, Any]) -> None:
        if req_id is None:
            return
        sid = str(req_id)
        try:
            p = ReflectionCancelActionParams.model_validate(params)
        except ValidationError as e:
            await self.send_error(sid, JSON_RPC_INVALID_PARAMS, f'invalid params: {e}')
            return
        if not p.trace_id:
            await self.send_error(sid, JSON_RPC_INVALID_PARAMS, 'traceId is required')
            return
        task = self.active_actions.get(p.trace_id)
        if task:
            task.cancel()
            self.active_actions.pop(p.trace_id, None)
            body = ReflectionCancelActionResponse(message='Action cancelled').model_dump(by_alias=True)
            await self.send_response(sid, body)
        else:
            await self.send_error(
                sid,
                JSON_RPC_INVALID_PARAMS,
                'Action not found or already completed',
            )

    async def handle_run_action(self, req_id: str | int | None, params: dict[str, Any]) -> None:
        if req_id is None:
            return
        sid = str(req_id)
        try:
            p = ReflectionRunActionParams.model_validate(params)
        except ValidationError as e:
            await self.send_error(sid, JSON_RPC_INVALID_PARAMS, f'invalid params: {e}')
            return

        action = await self.registry.resolve_action_by_key(p.key)
        if not action:
            await self.send_error(sid, JSON_RPC_INVALID_PARAMS, f'action {p.key} not found')
            return

        if p.context is not None and not isinstance(p.context, dict):
            await self.send_error(
                sid,
                JSON_RPC_INVALID_PARAMS,
                'context must be a JSON object when provided',
            )
            return

        # --- Bidi (agent) path ---
        if isinstance(action, BidiAction):
            await self.run_bidi_action(sid, p, action)
        else:
            await self.run_action(sid, p, action)
