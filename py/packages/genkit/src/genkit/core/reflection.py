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

"""Development API for inspecting and interacting with Genkit.

This module provides a reflection API server for inspection and interaction
during development. It exposes endpoints for health checks, action discovery,
and action execution.

## Caveats

The reflection API server predates the flows server implementation and differs
in the protocol it uses to interface with the Dev UI. The streaming protocol
uses unadorned JSON per streamed chunk. This may change in the future to use
Server-Sent Events (SSE).

## Key endpoints

    | Method | Path                | Handler               |
    |--------|---------------------|-----------------------|
    | GET    | /api/__health       | Health check          |
    | GET    | /api/actions        | List actions          |
    | POST   | /api/__quitquitquit | Trigger shutdown      |
    | POST   | /api/notify         | Handle notification   |
    | POST   | /api/runAction      | Run action (streaming)|
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from typing import Any, cast

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from genkit.codec import dump_dict, dump_json
from genkit.core.action import Action
from genkit.core.action.types import ActionKind
from genkit.core.constants import DEFAULT_GENKIT_VERSION
from genkit.core.error import get_reflection_json
from genkit.core.logging import get_logger
from genkit.core.registry import Registry
from genkit.web.manager.signals import terminate_all_servers
from genkit.web.requests import (
    is_streaming_requested,
)
from genkit.web.typing import (
    Application,
    StartupHandler,
)

logger = get_logger(__name__)


async def _list_registered_actions(registry: Registry) -> dict[str, Action]:
    """Return all locally registered actions keyed as `/<kind>/<name>`.

    Uses resolve_actions_by_kind() to trigger lazy loading for any actions with
    deferred metadata (e.g., file-based prompts), ensuring schemas are available
    for the Dev UI.
    """
    registered: dict[str, Action] = {}
    for kind in ActionKind.__members__.values():
        for name, action in (await registry.resolve_actions_by_kind(kind)).items():
            registered[f'/{kind.value}/{name}'] = action
    return registered


def _build_actions_payload(
    *,
    registered_actions: dict[str, Action],
    plugin_metas: list[Any],
) -> dict[str, dict[str, Any]]:
    """Build payload for GET /api/actions."""
    actions: dict[str, dict[str, Any]] = {}

    # 1) Registered actions (flows/tools/etc).
    for key, action in registered_actions.items():
        actions[key] = {
            'key': key,
            'name': action.name,
            'type': action.kind.value,
            'description': action.description,
            'inputSchema': action.input_schema,
            'outputSchema': action.output_schema,
            'metadata': action.metadata,
        }

    # 2) Plugin-advertised actions (may not be registered yet).
    for meta in plugin_metas or []:
        try:
            key = f'/{meta.kind.value}/{meta.name}'
        except Exception as exc:
            # Defensive: skip unexpected plugin metadata objects.
            logger.warning('Skipping invalid plugin action metadata', error=str(exc))
            continue

        advertised = {
            'key': key,
            'name': meta.name,
            'type': meta.kind.value,
            'description': getattr(meta, 'description', None),
            'inputSchema': getattr(meta, 'input_json_schema', None),
            'outputSchema': getattr(meta, 'output_json_schema', None),
            'metadata': getattr(meta, 'metadata', None),
        }

        if key not in actions:
            actions[key] = advertised
            continue

        # Merge into the existing (registered) action entry; prefer registered data.
        existing = actions[key]

        if not existing.get('description') and advertised.get('description'):
            existing['description'] = advertised['description']

        if not existing.get('inputSchema') and advertised.get('inputSchema'):
            existing['inputSchema'] = advertised['inputSchema']

        if not existing.get('outputSchema') and advertised.get('outputSchema'):
            existing['outputSchema'] = advertised['outputSchema']

        existing_meta = existing.get('metadata') or {}
        advertised_meta = advertised.get('metadata') or {}
        if isinstance(existing_meta, dict) and isinstance(advertised_meta, dict):
            # Prefer registered action metadata on key conflicts.
            existing['metadata'] = {**advertised_meta, **existing_meta}

    return actions


def create_reflection_asgi_app(
    registry: Registry,
    on_app_startup: StartupHandler | None = None,
    on_app_shutdown: StartupHandler | None = None,
    version: str = DEFAULT_GENKIT_VERSION,
    _encoding: str = 'utf-8',
) -> Application:
    """Create and return a ASGI application for the Genkit reflection API.

    Caveats:

        The reflection API server predates the flows server implementation and
        differs in the protocol it uses to interface with the Dev UI. The
        streaming protocol uses unadorned JSON per streamed chunk. This may
        change in the future to use Server-Sent Events (SSE).

    Key endpoints:

        | Method | Path                | Handler               |
        |--------|---------------------|-----------------------|
        | GET    | /api/__health       | Health check          |
        | GET    | /api/actions        | List actions          |
        | POST   | /api/__quitquitquit | Trigger shutdown      |
        | POST   | /api/notify         | Handle notification   |
        | POST   | /api/runAction      | Run action (streaming)|

    Args:
        registry: The registry to use for the reflection server.
        on_app_startup: Optional callback to execute when the app's
            lifespan starts. Must be an async function.
        on_app_shutdown: Optional callback to execute when the app's
            lifespan ends. Must be an async function.
        version: The version string to use when setting the value of
            the X-GENKIT-VERSION HTTP header.
        encoding: The text encoding to use; default 'utf-8'.

    Returns:
        An ASGI application configured with the given registry.
    """

    async def handle_health_check(_request: Request) -> JSONResponse:
        """Handle health check requests.

        Args:
            _request: The Starlette request object (unused).

        Returns:
            A JSON response with status code 200.
        """
        return JSONResponse(content={'status': 'OK'})

    async def handle_terminate(_request: Request) -> JSONResponse:
        """Handle the quit endpoint.

        Args:
            _request: The Starlette request object (unused).

        Returns:
            An empty JSON response with status code 200.
        """
        await logger.ainfo('Shutting down servers...')
        terminate_all_servers()
        return JSONResponse(content={'status': 'OK'})

    async def handle_list_actions(_request: Request) -> JSONResponse:
        """Handle the request for listing available actions.

        Args:
            _request: The Starlette request object (unused).

        Returns:
            A JSON response containing all serializable actions.
        """
        registered = await _list_registered_actions(registry)
        metas = await registry.list_actions()
        actions = _build_actions_payload(registered_actions=registered, plugin_metas=metas)

        return JSONResponse(
            content=actions,
            status_code=200,
            headers={'x-genkit-version': version},
        )

    async def handle_list_values(request: Request) -> JSONResponse:
        """Handle the request for listing registered values.

        Args:
             request: The Starlette request object.

        Returns:
            A JSON response containing value names.
        """
        kind = request.query_params.get('type')
        if not kind:
            return JSONResponse(content={'error': 'Query parameter "type" is required.'}, status_code=400)

        if kind != 'defaultModel':
            return JSONResponse(
                content={'error': f"'type' {kind} is not supported. Only 'defaultModel' is supported"}, status_code=400
            )

        values = registry.list_values(kind)
        return JSONResponse(content=values, status_code=200)

    async def handle_list_envs(_request: Request) -> JSONResponse:
        """Handle the request for listing environments.

        Args:
            _request: The Starlette request object (unused).

        Returns:
             A JSON response containing environments.
        """
        return JSONResponse(content=['dev'], status_code=200)

    async def handle_notify(_request: Request) -> JSONResponse:
        """Handle the notification endpoint.

        Args:
            _request: The Starlette request object (unused).

        Returns:
            An empty JSON response with status code 200.
        """
        return JSONResponse(
            content={},
            status_code=200,
            headers={'x-genkit-version': version},
        )

    # Map of active actions indexed by trace ID for cancellation support.
    active_actions: dict[str, asyncio.Task[Any]] = {}

    async def handle_cancel_action(request: Request) -> JSONResponse:
        """Handle the cancelAction endpoint.

        Args:
            request: The Starlette request object.

        Returns:
            A JSON response.
        """
        try:
            payload = await request.json()
            trace_id = payload.get('traceId')
            if not trace_id:
                return JSONResponse(content={'error': 'traceId is required'}, status_code=400)

            task = active_actions.get(trace_id)
            if task:
                _ = task.cancel()
                return JSONResponse(content={'message': 'Action cancelled'}, status_code=200)
            else:
                return JSONResponse(content={'message': 'Action not found or already completed'}, status_code=404)
        except Exception as e:
            logger.error(f'Error cancelling action: {e}', exc_info=True)
            return JSONResponse(
                content={'error': 'An unexpected error occurred while cancelling the action.'},
                status_code=500,
            )

    async def handle_run_action(
        request: Request,
    ) -> JSONResponse | StreamingResponse:
        """Handle the runAction endpoint for executing registered actions.

        Flow:
        1. Reads and validates the request payload
        2. Looks up the requested action
        3. Executes the action with the provided input
        4. Returns the action result as JSON with trace ID

        Args:
            request: The Starlette request object.

        Returns:
            A JSON or StreamingResponse with the action result, or an error
            response.
        """
        # Get the action using async resolve.
        payload = await request.json()
        action = await registry.resolve_action_by_key(payload['key'])
        if action is None:
            return JSONResponse(
                content={'error': f'Action not found: {payload["key"]}'},
                status_code=404,
            )

        # Run the action.
        context = payload.get('context', {})
        action_input = payload.get('input')
        stream = is_streaming_requested(request)

        # Wrap execution to track the task for cancellation support
        task = asyncio.current_task()

        def on_trace_start(trace_id: str, span_id: str) -> None:
            if task:
                active_actions[trace_id] = task

        handler = run_streaming_action if stream else run_standard_action

        try:
            return await handler(action, payload, action_input, context, version, on_trace_start)
        except asyncio.CancelledError:
            logger.info('Action execution cancelled.')
            # Can't really send response if cancelled? Starlette/uvicorn closes connection?
            # Just raise.
            raise

    async def run_streaming_action(
        action: Action,
        payload: dict[str, Any],
        _action_input: object,
        context: dict[str, Any],
        version: str,
        on_trace_start: Callable[[str, str], None],
    ) -> StreamingResponse:
        """Handle streaming action execution with early header flushing.

        Uses early header flushing to send X-Genkit-Trace-Id immediately when
        the trace starts, enabling the Dev UI to subscribe to SSE for real-time
        trace updates.

        Args:
            action: The action to execute.
            payload: Request payload with input data.
            action_input: The input for the action.
            context: Execution context.
            version: The Genkit version header value.
            on_trace_start: Callback for trace start.

        Returns:
            A StreamingResponse with JSON chunks containing result or error
            events.
        """
        # Use a queue to pass chunks from the callback to the generator
        chunk_queue: asyncio.Queue[str | None] = asyncio.Queue()

        # Event to signal when trace ID is available
        trace_id_event: asyncio.Event = asyncio.Event()
        run_trace_id: str | None = None
        run_span_id: str | None = None

        def wrapped_on_trace_start(tid: str, sid: str) -> None:
            nonlocal run_trace_id, run_span_id
            run_trace_id = tid
            run_span_id = sid
            on_trace_start(tid, sid)
            trace_id_event.set()  # Signal that trace ID is ready

        async def run_action_task() -> None:
            """Run the action and put chunks on the queue."""
            try:

                def send_chunk(chunk: Any) -> None:  # noqa: ANN401
                    """Callback that puts chunks on the queue."""
                    out = dump_json(chunk)
                    chunk_queue.put_nowait(f'{out}\n')

                output = await action.arun_raw(
                    raw_input=payload.get('input'),
                    on_chunk=send_chunk,
                    context=context,
                    on_trace_start=wrapped_on_trace_start,
                )
                final_response = {
                    'result': dump_dict(output.response),
                    'telemetry': {'traceId': output.trace_id, 'spanId': output.span_id},
                }
                chunk_queue.put_nowait(json.dumps(final_response))

            except Exception as e:
                error_response = get_reflection_json(e).model_dump(by_alias=True)
                # Log with exc_info for pretty exception output via rich/structlog
                logger.exception('Error streaming action', exc_info=e)
                # Error response also should not have trailing newline (final message)
                # Wrap error in an 'error' field to match JS SDK format
                chunk_queue.put_nowait(json.dumps({'error': error_response}))
                # Ensure trace_id_event is set even on error
                trace_id_event.set()

            finally:
                if not trace_id_event.is_set():
                    trace_id_event.set()
                # Signal end of stream
                chunk_queue.put_nowait(None)
                if run_trace_id:
                    _ = active_actions.pop(run_trace_id, None)

        # Start the action task immediately so trace ID becomes available ASAP
        action_task = asyncio.create_task(run_action_task())

        # Wait for trace ID before returning response - this enables early header flushing
        _ = await trace_id_event.wait()

        # Now we have the trace ID, include it in headers
        headers = {
            'x-genkit-version': version,
            'Transfer-Encoding': 'chunked',
        }
        if run_trace_id:
            headers['X-Genkit-Trace-Id'] = run_trace_id  # pyright: ignore[reportUnreachable]
        if run_span_id:
            headers['X-Genkit-Span-Id'] = run_span_id  # pyright: ignore[reportUnreachable]

        async def stream_generator() -> AsyncGenerator[str, None]:
            """Yield chunks from the queue as they arrive."""
            try:
                while True:
                    chunk = await chunk_queue.get()
                    if chunk is None:
                        break
                    yield chunk
            finally:
                # Cancel task if still running (no-op if already done)
                _ = action_task.cancel()

        return StreamingResponse(
            stream_generator(),
            # Reflection server uses text/plain for streaming (not SSE format)
            # to match Go implementation
            media_type='text/plain',
            headers=headers,
        )

    async def run_standard_action(
        action: Action,
        payload: dict[str, Any],
        _action_input: object,
        context: dict[str, Any],
        version: str,
        on_trace_start: Callable[[str, str], None],
    ) -> StreamingResponse:
        """Handle standard (non-streaming) action execution with early header flushing.

        Uses StreamingResponse to enable sending the X-Genkit-Trace-Id header
        immediately when the trace starts, allowing the Dev UI to subscribe to
        the SSE stream for real-time trace updates.

        Args:
            action: The action to execute.
            payload: Request payload with input data.
            action_input: The input for the action.
            context: Execution context.
            version: The Genkit version header value.
            on_trace_start: Callback for trace start.

        Returns:
            A StreamingResponse that flushes headers early.
        """
        # Event to signal when trace ID is available
        trace_id_event: asyncio.Event = asyncio.Event()
        run_trace_id: str | None = None
        run_span_id: str | None = None
        action_result: dict[str, Any] | None = None
        action_error: Exception | None = None

        def wrapped_on_trace_start(tid: str, sid: str) -> None:
            nonlocal run_trace_id, run_span_id
            run_trace_id = tid
            run_span_id = sid
            on_trace_start(tid, sid)
            trace_id_event.set()  # Signal that trace ID is ready

        async def run_action_and_get_result() -> None:
            nonlocal action_result, action_error
            try:
                output = await action.arun_raw(
                    raw_input=payload.get('input'),
                    context=context,
                    on_trace_start=wrapped_on_trace_start,
                )
                action_result = {
                    'result': dump_dict(output.response),
                    'telemetry': {'traceId': output.trace_id, 'spanId': output.span_id},
                }
            except Exception as e:
                action_error = e
            finally:
                if not trace_id_event.is_set():
                    trace_id_event.set()

        # Start the action immediately so trace ID becomes available ASAP
        action_task = asyncio.create_task(run_action_and_get_result())

        # Wait for trace ID before returning response
        _ = await trace_id_event.wait()

        # Now return streaming response - headers will include trace ID
        async def body_generator() -> AsyncGenerator[bytes, None]:
            # Wait for action to complete
            await action_task

            if action_error:
                error_response = get_reflection_json(action_error).model_dump(by_alias=True)
                # Log with exc_info for pretty exception output via rich/structlog
                logger.exception('Error executing action', exc_info=action_error)
                # Wrap error in an 'error' field to match JS SDK format
                yield json.dumps({'error': error_response}).encode('utf-8')
            else:
                yield json.dumps(action_result).encode('utf-8')

            if run_trace_id:
                _ = active_actions.pop(run_trace_id, None)

        headers = {
            'x-genkit-version': version,
            'Transfer-Encoding': 'chunked',
        }
        if run_trace_id:
            headers['X-Genkit-Trace-Id'] = run_trace_id  # pyright: ignore[reportUnreachable]
        if run_span_id:
            headers['X-Genkit-Span-Id'] = run_span_id  # pyright: ignore[reportUnreachable]

        return StreamingResponse(
            body_generator(),
            media_type='application/json',
            headers=headers,
        )

    app = Starlette(
        routes=[
            Route('/api/__health', handle_health_check, methods=['GET']),
            Route('/api/__quitquitquit', handle_terminate, methods=['GET', 'POST']),  # Support both for parity
            Route('/api/actions', handle_list_actions, methods=['GET']),
            Route('/api/values', handle_list_values, methods=['GET']),
            Route('/api/envs', handle_list_envs, methods=['GET']),
            Route('/api/notify', handle_notify, methods=['POST']),
            Route('/api/runAction', handle_run_action, methods=['POST']),
            Route('/api/cancelAction', handle_cancel_action, methods=['POST']),
        ],
        middleware=[
            Middleware(
                CORSMiddleware,  # type: ignore[arg-type]
                allow_origins=['*'],
                allow_methods=['*'],
                allow_headers=['*'],
                expose_headers=['X-Genkit-Trace-Id', 'X-Genkit-Span-Id', 'x-genkit-version'],
            )
        ],
        on_startup=[on_app_startup] if on_app_startup else [],
        on_shutdown=[on_app_shutdown] if on_app_shutdown else [],
    )
    app.active_actions = active_actions  # type: ignore[attr-defined]
    return cast(Application, app)
