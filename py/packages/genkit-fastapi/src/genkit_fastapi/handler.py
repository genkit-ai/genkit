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

"""Genkit FastAPI handler for serving flows and agents as HTTP endpoints."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, TypeVar, cast

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from genkit import ContextProvider, Genkit, GenkitError, RequestData
from genkit.plugin_api import Action, get_callable_json

# Compact JSON (no spaces) for smaller wire payload.
JSON_SEPARATORS = (',', ':')

InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')
ChunkT = TypeVar('ChunkT')
InitT = TypeVar('InitT')


def to_dict(obj: Any) -> Any:  # noqa: ANN401
    """Convert object to dict if it's a Pydantic model, otherwise return as-is."""
    return obj.model_dump(by_alias=True, exclude_none=True) if isinstance(obj, BaseModel) else obj


class FastAPIRequestData(RequestData):
    """Wraps FastAPI request data for Genkit context."""

    def __init__(self, request: Request, body: dict[str, Any] | None) -> None:
        """Initialize request data wrapper."""
        super().__init__(request=request)
        self.method = request.method
        self.headers = {k.lower(): v for k, v in request.headers.items()}
        self.input = body.get('data') if body else None


def json_error_response(error: Exception, status_code: int = 400) -> Response:
    """Build a compact JSON error response from an exception."""
    # A wrapped cause is what the client should see. The wrapper itself is the
    # message when there isn't one (a bad body has no inner exception).
    ex = error.cause if isinstance(error, GenkitError) and error.cause is not None else error
    return Response(
        status_code=status_code,
        content=json.dumps(get_callable_json(ex), separators=JSON_SEPARATORS),
        media_type='application/json',
    )


def extract_action_input(body: dict[str, Any]) -> object:
    """Extract action input from the stable ``data`` / ``input`` / ``{}`` envelopes."""
    if 'data' in body:
        return body['data']
    if 'input' in body:
        return body['input']
    # Callable clients omit ``data`` when runFlow has no input (POST ``{}``).
    # A missing wrapper is not a wire error; the action decides if input is required.
    if not body:
        return None
    raise GenkitError(
        status='INVALID_ARGUMENT',
        message='Action request must be wrapped in {"data": ...} object',
    )


def wants_stream(request: Request) -> bool:
    """Check if the client requested an event stream or NDJSON stream."""
    accept = request.headers.get('accept', '')
    return 'text/event-stream' in accept or request.query_params.get('stream') == 'true'


def format_stream_chunk(chunk: object) -> str:
    """Format a stream chunk for SSE."""
    msg_json = json.dumps({'message': to_dict(chunk)}, separators=JSON_SEPARATORS)
    return f'data: {msg_json}\n\n'


def format_stream_result(result: object) -> str:
    """Format the final stream result for SSE."""
    res_json = json.dumps({'result': to_dict(result)}, separators=JSON_SEPARATORS)
    return f'data: {res_json}\n\n'


def format_stream_error(error: Exception) -> str:
    """Format a stream failure as a canonical SSE data event."""
    ex = error.cause if isinstance(error, GenkitError) else error
    return f'data: {json.dumps({"error": get_callable_json(ex)}, separators=JSON_SEPARATORS)}\n\n'


async def handle_genkit_request(
    request: Request,
    *,
    action: Action[InputT, OutputT, ChunkT, InitT],
    context: dict[str, object] | None = None,
    init: InitT | dict[str, Any] | None = None,
) -> Response | dict[str, Any]:
    """Run one Genkit action request and return its FastAPI response.

    This is the wire contract every stable route sits on. It reads ``data`` /
    ``input`` / ``{}`` plus body ``init``, then either streams SSE frames —
    ``data: {"message": ...}`` chunks followed by a final ``data: {"result": ...}``
    — or returns a one-shot ``{"result": ...}``.

    ``context`` and ``init`` are handed straight to the action, so you can resolve
    auth and per-request state however you like and pass them in. That makes this
    the escape hatch for full control: write your own ``@app.post`` endpoint with
    any ``Depends(...)`` params you need, build context and init, and call this to
    get the exact Genkit wire format without re-implementing it.

    Args:
        request: The incoming FastAPI request.
        action: The flow or agent action to run.
        context: Optional context dict passed through to the action.
        init: Optional init payload passed through to the action.

    Returns:
        A streaming SSE response, a ``{"result": ...}`` dict, or an error Response.
    """
    return await _handle_action_request(
        request=request,
        action=action,
        context=context,
        init=init,
    )


async def _handle_action_request(
    *,
    request: Request,
    action: Action[InputT, OutputT, ChunkT, InitT],
    context: dict[str, object] | None = None,
    init: InitT | dict[str, Any] | None = None,
    extract_input: Callable[[dict[str, Any]], object] | None = None,
    resolve_init: Callable[[dict[str, Any], Mapping[str, str]], object] | None = None,
    empty_status: int | None = None,
) -> Response | dict[str, Any]:
    body = await request.json()
    if not isinstance(body, dict):
        return json_error_response(
            GenkitError(
                status='INVALID_ARGUMENT',
                message='Action request must be a JSON object',
            )
        )

    try:
        input_data = (extract_input or extract_action_input)(body)
    except GenkitError as err:
        return json_error_response(err)

    if init is not None:
        resolved_init = init
    elif resolve_init is not None:
        resolved_init = resolve_init(body, request.query_params)
    else:
        resolved_init = body.get('init')
    action_obj = cast(Action[Any, Any, Any, Any], action)

    if wants_stream(request):

        async def event_stream() -> AsyncIterator[str]:
            try:
                stream_response = action_obj.stream(input_data, context=context, init=resolved_init)
                async for chunk in stream_response.stream:
                    yield format_stream_chunk(chunk)
                result = await stream_response.response
                yield format_stream_result(result)
            except Exception as e:
                yield format_stream_error(e)

        return StreamingResponse(event_stream(), media_type='text/event-stream')

    try:
        response = await action_obj.run(input_data, context=context, init=resolved_init)
        if response.response is None and empty_status is not None:
            return Response(status_code=empty_status)
        return {'result': to_dict(response.response)}
    except Exception as e:
        return json_error_response(e, status_code=500)


def genkit_fastapi_handler(
    ai: Genkit,
    context_provider: ContextProvider | None = None,
) -> Callable[
    [Callable[[], Awaitable[Action[InputT, OutputT, ChunkT, InitT]]] | Action[InputT, OutputT, ChunkT, InitT]],
    Callable[[Request], Awaitable[Response | dict[str, Any]]],
]:
    """Decorator for serving Genkit actions (flows, agents, tools, etc.) via FastAPI.

    Example (decorator on flow directly):
        ```python
        @app.post('/chat', response_model=None)
        @genkit_fastapi_handler(ai)
        @ai.flow()
        async def chat(prompt: str) -> str:
            response = await ai.generate(prompt=prompt)
            return response.text
        ```

    Example (wrapper when flow is defined later; must be async):
        ```python
        @app.post('/chat', response_model=None)
        @genkit_fastapi_handler(ai)
        async def chat():
            return my_flow


        @ai.flow()
        async def my_flow(prompt: str) -> str: ...
        ```

    Args:
        ai: The Genkit instance.
        context_provider: Optional function to extract context from the request.

    Returns:
        A decorator that wraps an Action or a function returning an Action.
    """

    def decorator(
        fn: Callable[[], Awaitable[Action[InputT, OutputT, ChunkT, InitT]]] | Action[InputT, OutputT, ChunkT, InitT],
    ) -> Callable[[Request], Awaitable[Response | dict[str, Any]]]:
        async def handler(request: Request) -> Response | dict[str, Any]:
            if isinstance(fn, Action):
                action = fn
            else:
                result = fn()
                if not asyncio.iscoroutine(result):
                    raise GenkitError(
                        status='INVALID_ARGUMENT',
                        message='genkit_fastapi_handler wrapper must be async when action is defined elsewhere',
                    )
                action = await result
            if not isinstance(action, Action):
                raise GenkitError(
                    status='INVALID_ARGUMENT',
                    message='genkit_fastapi_handler must wrap an Action or an async function returning an Action',
                )

            # This decorator reads context from the request itself. Routes that
            # want FastAPI's dependency graph (auth schemes, DB sessions) go
            # through serve_flow/serve_agent's context_dependency instead.
            action_context: dict[str, object] | None = None
            if context_provider:
                body = await request.json()
                request_data = FastAPIRequestData(request, body if isinstance(body, dict) else None)
                context = context_provider(request_data)
                if asyncio.iscoroutine(context):
                    context = await context
                if isinstance(context, dict):
                    action_context = context

            return await handle_genkit_request(
                request,
                action=cast(Action[InputT, OutputT, ChunkT, InitT], action),  # ty: ignore[redundant-cast]
                context=action_context,
            )

        return handler

    return decorator


def _mount_action(
    router: APIRouter,
    path: str,
    action: Action[InputT, OutputT, ChunkT, InitT],
    *,
    context_dependency: Callable[..., Any] | None = None,
    extract_input: Callable[[dict[str, Any]], object] | None = None,
    resolve_init: Callable[[dict[str, Any], Mapping[str, str]], object] | None = None,
    empty_status: int | None = None,
) -> None:
    """Register one action on the router, honoring FastAPI DI when asked.

    With a ``context_dependency`` the route's own signature carries the
    dependency, so FastAPI resolves it (and any sub-dependencies or security
    schemes) and the resulting dict is threaded into the action as context.
    """
    if context_dependency is not None:

        async def endpoint_with_context(
            request: Request,
            context: Any = Depends(context_dependency),  # noqa: ANN401, B008
        ) -> Response | dict[str, Any]:
            return await _handle_action_request(
                request=request,
                action=action,
                context=context if isinstance(context, dict) else None,
                extract_input=extract_input,
                resolve_init=resolve_init,
                empty_status=empty_status,
            )

        router.post(path, response_model=None)(endpoint_with_context)
        return

    async def endpoint(request: Request) -> Response | dict[str, Any]:
        return await _handle_action_request(
            request=request,
            action=action,
            extract_input=extract_input,
            resolve_init=resolve_init,
            empty_status=empty_status,
        )

    router.post(path, response_model=None)(endpoint)


def serve_flow(
    flow: Action[InputT, OutputT, ChunkT, InitT],
    *,
    base_path: str | None = None,
    context_dependency: Callable[..., Any] | None = None,
) -> APIRouter:
    """Build an APIRouter serving a single flow over HTTP.

    Mount the returned router like any other, so FastAPI's own prefix / dependencies handle wiring::

        app.include_router(serve_flow(chat_flow), prefix='/api')

    Args:
        flow: The flow action to serve.
        base_path: Route path. Defaults to /<flow name>.
        context_dependency: A FastAPI dependency whose resolved value becomes the
            action context. Use this to reuse existing ``Depends``-based auth /
            resources.

    Returns:
        An APIRouter with the single flow route registered.
    """
    resolved_base_path = f'/{flow.name}' if base_path is None else base_path
    router = APIRouter(tags=[flow.name])
    _mount_action(
        router,
        resolved_base_path,
        flow,
        context_dependency=context_dependency,
    )
    return router
