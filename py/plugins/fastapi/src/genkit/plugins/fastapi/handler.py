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
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from genkit import Action, Genkit, GenkitError
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


def genkit_fastapi_handler(
    ai: Genkit,
    context_provider: ContextProvider | None = None,
) -> Callable[[Callable[[], Action]], Callable[[Request], Awaitable[Response | dict[str, Any]]]]:
    """Decorator for serving Genkit flows via FastAPI.

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
        fn: Callable[[], Action] | Action,
    ) -> Callable[[Request], Awaitable[Response | dict[str, Any]]]:
        async def handler(request: Request) -> Response | dict[str, Any]:
            if isinstance(fn, Action):
                flow = fn
            else:
                result = fn()
                if not asyncio.iscoroutine(result):
                    raise GenkitError(
                        status='INVALID_ARGUMENT',
                        message='genkit_fastapi_handler wrapper must be async when flow is defined elsewhere',
                    )
                flow = await result
            if not isinstance(flow, Action):
                raise GenkitError(
                    status='INVALID_ARGUMENT',
                    message='genkit_fastapi_handler must wrap a function that returns a @flow',
                )

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

            request_data = _FastAPIRequestData(request, body)
            action_context: dict[str, object] | None = None

            if context_provider:
                context = context_provider(request_data)
                if asyncio.iscoroutine(context):
                    context = await context
                if isinstance(context, dict):
                    action_context = context

            # Check if client wants streaming
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
            else:
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

        return handler

    return decorator
