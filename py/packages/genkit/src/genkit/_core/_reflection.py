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

"""Reflection API server for Genkit Dev UI."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from genkit._core._action import Action
from genkit._core._constants import GENKIT_VERSION
from genkit._core._error import get_reflection_json
from genkit._core._logger import get_logger
from genkit._core._middleware._base import MiddlewareDesc
from genkit._core._registry import Registry

logger = get_logger(__name__)

LifecycleHook = Callable[[], Awaitable[None]]


@dataclass
class ServerSpec:
    port: int
    scheme: str = 'http'
    host: str = 'localhost'

    @property
    def url(self) -> str:
        return f'{self.scheme}://{self.host}:{self.port}'


@dataclass
class ActionRunner:
    """Encapsulates state for running an action with streaming support."""

    action: Action
    payload: dict[str, Any]
    stream: bool
    active_actions: dict[str, asyncio.Task[Any]]

    queue: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue)
    trace_ready: asyncio.Event = field(default_factory=asyncio.Event)
    trace_id: str | None = None
    span_id: str | None = None

    async def on_trace_start(self, tid: str, sid: str) -> None:
        self.trace_id, self.span_id = tid, sid
        if task := asyncio.current_task():
            self.active_actions[tid] = task
        self.trace_ready.set()

    async def execute(self) -> None:
        try:
            on_chunk = (
                (
                    lambda c: self.queue.put_nowait(
                        f'{c.model_dump_json() if isinstance(c, BaseModel) else json.dumps(c)}\n'
                    )
                )
                if self.stream
                else None
            )
            output = await self.action.run(
                input=self.payload.get('input'),
                on_chunk=on_chunk,
                context=self.payload.get('context', {}),
                on_trace_start=self.on_trace_start,
                telemetry_labels=self.payload.get('telemetryLabels'),
            )
            result = (
                output.response.model_dump(by_alias=True, exclude_none=True)
                if isinstance(output.response, BaseModel)
                else output.response
            )
            self.queue.put_nowait(
                json.dumps({
                    'result': result,
                    'telemetry': {'traceId': output.trace_id, 'spanId': output.span_id},
                })
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception('Error executing action')
            self.queue.put_nowait(json.dumps({'error': get_reflection_json(e).model_dump(by_alias=True)}))
        finally:
            self.trace_ready.set()
            self.queue.put_nowait(None)
            if self.trace_id:
                self.active_actions.pop(self.trace_id, None)

    async def stream_response(self, version: str) -> StreamingResponse:
        task = asyncio.create_task(self.execute())
        await self.trace_ready.wait()

        headers = {'x-genkit-version': version, 'Transfer-Encoding': 'chunked'}
        if self.trace_id:
            headers['X-Genkit-Trace-Id'] = self.trace_id
        if self.span_id:
            headers['X-Genkit-Span-Id'] = self.span_id

        async def gen() -> AsyncGenerator[str, None]:
            try:
                while (chunk := await self.queue.get()) is not None:
                    yield chunk
            finally:
                task.cancel()

        return StreamingResponse(gen(), media_type='text/plain' if self.stream else 'application/json', headers=headers)


def create_reflection_asgi_app(
    registry: Registry,
    on_startup: LifecycleHook | None = None,
    on_shutdown: LifecycleHook | None = None,
    version: str = GENKIT_VERSION,
) -> Starlette:
    active_actions: dict[str, asyncio.Task[Any]] = {}

    async def health(_: Request) -> JSONResponse:
        await registry.initialize_all_plugins()
        return JSONResponse({'status': 'OK'})

    async def terminate(_: Request) -> JSONResponse:
        logger.info('Shutting down...')
        asyncio.get_running_loop().call_soon(os.kill, os.getpid(), signal.SIGTERM)
        return JSONResponse({'status': 'OK'})

    async def actions(_: Request) -> JSONResponse:
        # Full catalog: plugins, registered actions, DAP expansions; merged with parent.
        actions = await registry.list_actions()

        def omit_none(payload: dict[str, Any]) -> dict[str, Any]:
            return {key: value for key, value in payload.items() if value is not None}

        response: dict[str, dict[str, Any]] = {}
        for key, action in actions.items():
            response[key] = omit_none({
                'key': key,
                'name': action.name,
                'description': action.description,
                'metadata': action.metadata,
                'inputSchema': action.input_schema or action.input_json_schema,
                'outputSchema': action.output_schema or action.output_json_schema,
            })

        return JSONResponse(response, headers={'x-genkit-version': version})

    async def values(req: Request) -> Response:
        raw = req.query_params.get('type')
        if not raw or not raw.strip():
            return JSONResponse(
                {'error': 'Query parameter "type" is required.'},
                status_code=400,
                headers={'x-genkit-version': version},
            )
        type_param = raw.strip()
        try:
            try:
                await registry.initialize_all_plugins()
            except Exception as e:
                logger.warning(
                    'initialize_all_plugins failed during /api/values; returning registered values only: %s',
                    e,
                    exc_info=True,
                )
            raw_values = registry.list_values(type_param)
            if type_param == 'middleware':
                serialized: dict[str, Any] = {}
                for key, val in raw_values.items():
                    if isinstance(val, MiddlewareDesc):
                        serialized[key] = val.model_dump(by_alias=True, exclude_none=True, mode='json')
                    else:
                        serialized[key] = val
                raw_values = serialized
            return JSONResponse(raw_values, headers={'x-genkit-version': version})
        except Exception:
            logger.exception('Reflection /api/values failed')
            return JSONResponse(
                {'error': 'Failed to list values', 'detail': 'See Python process logs for the traceback.'},
                status_code=500,
                headers={'x-genkit-version': version},
            )

    async def envs(_: Request) -> JSONResponse:
        return JSONResponse(['dev'])

    async def notify(_: Request) -> JSONResponse:
        return JSONResponse({}, headers={'x-genkit-version': version})

    async def cancel(req: Request) -> JSONResponse:
        trace_id = (await req.json()).get('traceId')
        if not trace_id:
            return JSONResponse({'error': 'traceId required'}, status_code=400)
        if task := active_actions.get(trace_id):
            task.cancel()
            return JSONResponse({'message': 'Cancelled'})
        return JSONResponse({'message': 'Not found'}, status_code=404)

    async def run(req: Request) -> Response:
        payload = await req.json()
        action = await registry.resolve_action_by_key(payload['key'])
        if not action:
            return JSONResponse({'error': f'Action not found: {payload["key"]}'}, status_code=404)

        runner = ActionRunner(
            action=action,
            payload=payload,
            stream=req.headers.get('accept') == 'text/event-stream' or req.query_params.get('stream') == 'true',
            active_actions=active_actions,
        )
        return await runner.stream_response(version)

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        # Eagerly initialize plugins so init()-registered actions exist before handling traffic.
        await registry.initialize_all_plugins()
        if on_startup is not None:
            await on_startup()
        yield
        if on_shutdown is not None:
            await on_shutdown()

    app = Starlette(
        routes=[
            Route('/api/__health', health, methods=['GET']),
            Route('/api/__quitquitquit', terminate, methods=['GET', 'POST']),
            Route('/api/actions', actions, methods=['GET']),
            Route('/api/values', values, methods=['GET']),
            Route('/api/envs', envs, methods=['GET']),
            Route('/api/notify', notify, methods=['POST']),
            Route('/api/runAction', run, methods=['POST']),
            Route('/api/cancelAction', cancel, methods=['POST']),
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
        lifespan=lifespan,
    )
    app.active_actions = active_actions  # type: ignore[attr-defined]
    return app


class ReflectionServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, ready: threading.Event) -> None:
        super().__init__(config)
        self._ready = ready

    async def startup(self, sockets: list | None = None) -> None:
        try:
            await super().startup(sockets=sockets)
        finally:
            self._ready.set()
