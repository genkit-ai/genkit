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

"""Genkit FastAPI handler for serving flows as HTTP endpoints."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, cast

from pydantic import BaseModel

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
from genkit import Action, GenkitError
from genkit.plugin_api import ContextProvider, RequestData, get_callable_json

# Compact JSON (no spaces) for smaller wire payload.
_JSON_SEPARATORS = (',', ':')


def _to_dict(obj: Any) -> Any:  # noqa: ANN401
    """Convert object to dict if it's a Pydantic model, otherwise return as-is."""
    return obj.model_dump() if isinstance(obj, BaseModel) else obj


class _FastAPIRequestData(RequestData):
    """Wraps FastAPI request data for Genkit context."""

    def __init__(self, request: Request, body: dict[str, Any] | None) -> None:
        super().__init__(request=request)
        self.method = request.method
        self.headers = {k.lower(): v for k, v in request.headers.items()}
        self.input = body.get('data') if body else None


def serve_flow(
    flow: Action,
    *,
    base_path: str | None = None,
    context_provider: ContextProvider | None = None,
) -> APIRouter:
    """Build an ``APIRouter`` serving a single flow over HTTP.

    Mount the returned router like any other, so FastAPI's own ``prefix`` /
    ``dependencies`` / ``tags`` handle the framework-level wiring::

        app.include_router(serve_flow(chat_flow), prefix='/api')
        # POST /api/chat_flow  (or pass base_path='/chat')

    The route takes ``{"data": <input>}`` and returns ``{"result": <output>}``, or
    streams Server-Sent Events when the client sends ``Accept: text/event-stream``.

    Args:
        flow: The flow to serve (from ``ai.flow()``).
        base_path: Route path. Defaults to ``/<flow name>``; pass ``''`` for the router root.
        context_provider: Reads the request and returns the context dict the flow
            runs with — the place to authenticate and attach the user. Raise to
            reject the request before the flow runs.

    Returns:
        An ``APIRouter`` with the single flow route registered.
    """
    resolved_base_path = f'/{flow.name}' if base_path is None else base_path
    router = APIRouter()

    @router.post(resolved_base_path, response_model=None)
    async def run_flow(request: Request) -> Response | dict[str, Any]:
        body = await request.json()
        if 'data' not in body:
            err = GenkitError(
                status='INVALID_ARGUMENT',
                message='Action request must be wrapped in {"data": ...} object',
            )
            return Response(
                status_code=400,
                content=json.dumps(get_callable_json(err), separators=_JSON_SEPARATORS),
                media_type='application/json',
            )

        action_context: dict[str, object] | None = None
        if context_provider:
            context = context_provider(_FastAPIRequestData(request, body))
            if asyncio.iscoroutine(context):
                context = await context
            if isinstance(context, dict):
                action_context = cast(dict[str, object], context)

        accept = request.headers.get('accept', '')
        stream = 'text/event-stream' in accept or request.query_params.get('stream') == 'true'

        if stream:

            async def event_stream() -> AsyncIterator[str]:
                try:
                    stream_response = flow.stream(body.get('data'), context=action_context)
                    async for chunk in stream_response.stream:
                        yield f'data: {json.dumps({"message": _to_dict(chunk)}, separators=_JSON_SEPARATORS)}\n\n'

                    result = await stream_response.response
                    yield f'data: {json.dumps({"result": _to_dict(result)}, separators=_JSON_SEPARATORS)}\n\n'
                except Exception as e:
                    ex = e.cause if isinstance(e, GenkitError) else e
                    yield f'error: {json.dumps({"error": get_callable_json(ex)}, separators=_JSON_SEPARATORS)}'

            return StreamingResponse(event_stream(), media_type='text/event-stream')

        try:
            response = await flow.run(body.get('data'), context=action_context)
            return {'result': _to_dict(response.response)}
        except Exception as e:
            ex = e.cause if isinstance(e, GenkitError) else e
            return Response(
                status_code=500,
                content=json.dumps(get_callable_json(ex), separators=_JSON_SEPARATORS),
                media_type='application/json',
            )

    return router
