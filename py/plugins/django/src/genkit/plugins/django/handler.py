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

"""Genkit Django handler for serving flows as HTTP endpoints."""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from django.http import HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from genkit import Action, Genkit, GenkitError
from genkit.plugin_api import ContextProvider, RequestData, get_callable_json

# Compact JSON (no spaces) for smaller wire payload.
_JSON_SEPARATORS = (',', ':')


def _to_dict(obj: Any) -> Any:  # noqa: ANN401
    """Convert object to dict if it's a Pydantic model, otherwise return as-is."""
    return obj.model_dump() if isinstance(obj, BaseModel) else obj


def _error_response(status: int, err: Exception) -> HttpResponse:
    """Return a JSON HttpErrorWireFormat response for an exception."""
    return HttpResponse(
        status=status,
        content=json.dumps(get_callable_json(err), separators=_JSON_SEPARATORS),
        content_type='application/json',
    )


class _DjangoRequestData(RequestData):
    """Wraps Django request data for Genkit context."""

    def __init__(self, request: HttpRequest, body: dict[str, Any] | None) -> None:
        super().__init__(request=request)
        self.method = request.method
        self.headers = {k.lower(): v for k, v in request.headers.items()}
        self.input = body.get('data') if body else None


def genkit_django_handler(
    ai: Genkit,
    context_provider: ContextProvider | None = None,
) -> Callable[[Action], Callable[[HttpRequest], Awaitable[HttpResponse]]]:
    """A decorator for serving Genkit flows via a Django ASGI app.

    ```python
    from django.urls import path
    from genkit.plugins.django import genkit_django_handler


    @genkit_django_handler(ai)
    @ai.flow()
    async def say_hi(name: str, ctx):
        return await ai.generate(
            on_chunk=ctx.send_chunk,
            prompt=f'tell a medium sized joke about {name}',
        )


    urlpatterns = [
        path('chat/', say_hi),
    ]
    ```

    Requires Django ASGI (Django 4.1+). The returned view is `csrf_exempt`
    because this is a JSON API.

    Args:
        ai: The Genkit instance.
        context_provider: Optional function to extract context from the request.

    Returns:
        A decorator that wraps an Action and returns an async Django view.
    """

    def decorator(flow: Action) -> Callable[[HttpRequest], Awaitable[HttpResponse]]:
        if not isinstance(flow, Action):
            raise GenkitError(status='INVALID_ARGUMENT', message='must apply @genkit_django_handler on a @flow')

        @csrf_exempt
        async def handler(request: HttpRequest) -> HttpResponse:
            if request.method != 'POST':
                return _error_response(
                    405,
                    GenkitError(status='INVALID_ARGUMENT', message='only POST is supported'),
                )

            try:
                body = json.loads(request.body.decode('utf-8')) if request.body else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                return _error_response(
                    400,
                    GenkitError(status='INVALID_ARGUMENT', message='request body must be valid JSON'),
                )

            if not isinstance(body, dict) or 'data' not in body:
                return _error_response(
                    400,
                    GenkitError(
                        status='INVALID_ARGUMENT',
                        message='Action request must be wrapped in {"data": ...} object',
                    ),
                )

            request_data = _DjangoRequestData(request, body)
            action_context: dict[str, object] | None = None

            if context_provider:
                context = context_provider(request_data)
                if asyncio.iscoroutine(context):
                    context = await context
                if isinstance(context, dict):
                    action_context = context

            accept = request.headers.get('Accept', '')
            stream = 'text/event-stream' in accept or request.GET.get('stream') == 'true'

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

                return StreamingHttpResponse(event_stream(), content_type='text/event-stream')

            try:
                response = await flow.run(body.get('data'), context=action_context)
                return JsonResponse({'result': _to_dict(response.response)})
            except Exception as e:
                ex = e.cause if isinstance(e, GenkitError) else e
                return _error_response(500, ex)

        return handler

    return decorator
