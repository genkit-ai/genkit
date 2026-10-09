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
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, cast

from django.core.exceptions import PermissionDenied, SuspiciousOperation
from django.http import Http404, HttpRequest, HttpResponse, HttpResponseBase, JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from pydantic import BaseModel

from genkit import ContextProvider, Genkit, GenkitError, PublicError, RequestData
from genkit.plugin_api import Action
from genkit.web import error_body, error_status, read_body, wants_stream

logger = logging.getLogger(__name__)

# Compact JSON (no spaces) for smaller wire payload.
_JSON_SEPARATORS = (',', ':')


def _log_failure(*, error: Exception, where: str) -> None:
    """Log a served-flow failure; 5xx includes the traceback."""
    if error_status(error) >= 500:
        logger.exception('served flow %s failed', where)
    else:
        logger.warning('served flow %s failed: %s', where, error)


def _to_dict(obj: Any) -> Any:  # noqa: ANN401
    """Recursively convert Pydantic models inside ``obj`` to plain JSON-friendly types.

    Flows can return a Pydantic model, a list of Pydantic models, or a dict whose
    values are Pydantic models. Django's ``JsonResponse`` and ``json.dumps`` don't
    know how to serialize ``BaseModel`` instances natively, so descend into lists,
    tuples, and dicts to convert every model we find.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, list):
        return [_to_dict(item) for item in obj]
    if isinstance(obj, tuple):
        return [_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def _error_response(err: Exception, status: int | None = None) -> HttpResponse:
    """Return a JSON error body; status comes from the error unless overridden."""
    return HttpResponse(
        status=error_status(err) if status is None else status,
        content=json.dumps(error_body(err), separators=_JSON_SEPARATORS).encode('utf-8'),
        content_type='application/json',
    )


def _request_headers(request: HttpRequest) -> Mapping[str, str]:
    """Return ``request.headers`` typed as a Mapping.

    Django's ``HttpRequest.headers`` is a ``HttpHeaders`` (a ``CaseInsensitiveMapping``)
    at runtime but is exposed as a ``cached_property`` to static type checkers, which
    then can't see ``.get()`` / ``.items()``. Casting once keeps the handler readable.
    """
    return cast(Mapping[str, str], request.headers)


class _DjangoRequestData(RequestData):
    """Wraps Django request data for Genkit context."""

    def __init__(self, request: HttpRequest, body: dict[str, Any] | None) -> None:
        super().__init__(
            request=request,
            method=request.method or '',
            headers=_request_headers(request).items(),
            input=body.get('data') if body else None,
        )


def genkit_django_handler(
    ai: Genkit,
    context_provider: ContextProvider | None = None,
) -> Callable[[Action], Callable[[HttpRequest], Awaitable[HttpResponseBase]]]:
    """A decorator for serving Genkit flows via a Django ASGI app.

    ```python
    from django.urls import path
    from genkit import ActionRunContext
    from genkit_django import genkit_django_handler


    @genkit_django_handler(ai)
    @ai.flow()
    async def say_hi(name: str, ctx: ActionRunContext) -> str:
        stream = ai.generate_stream(prompt=f'tell a joke about {name}')
        async for chunk in stream.stream:
            if chunk.text:
                ctx.send_chunk(chunk.text)
        res = await stream.response
        return res.text


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

    def decorator(flow: Action) -> Callable[[HttpRequest], Awaitable[HttpResponseBase]]:
        if not isinstance(flow, Action):
            raise GenkitError(status='INVALID_ARGUMENT', message='must apply @genkit_django_handler on a @flow')

        @csrf_exempt
        async def handler(request: HttpRequest) -> HttpResponseBase:
            if request.method != 'POST':
                return _error_response(
                    PublicError('INVALID_ARGUMENT', 'only POST is supported'),
                    status=405,
                )

            try:
                body = json.loads(request.body.decode('utf-8')) if request.body else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                return _error_response(
                    PublicError('INVALID_ARGUMENT', 'request body must be valid JSON'),
                )

            try:
                action_input = read_body(body)
            except PublicError as e:
                return _error_response(e)

            request_data = _DjangoRequestData(request, body)
            action_context: dict[str, object] | None = None

            if context_provider:
                try:
                    context = context_provider(request_data)
                    if asyncio.iscoroutine(context):
                        context = await context
                    if isinstance(context, dict):
                        action_context = context
                except (PermissionDenied, Http404, SuspiciousOperation):
                    # Django's own denial; re-raise so it stays 403/404/400.
                    raise
                except Exception as e:
                    _log_failure(error=e, where='context provider')
                    return _error_response(e)

            init = body.get('init')

            stream = cast(str | None, request.GET.get('stream'))
            if wants_stream(accept=request_data.headers.get('accept'), stream=stream):

                async def event_stream() -> AsyncIterator[str]:
                    try:
                        stream_response = flow.stream(input=action_input, context=action_context, init=init)
                        async for chunk in stream_response.stream:
                            yield f'data: {json.dumps({"message": _to_dict(chunk)}, separators=_JSON_SEPARATORS)}\n\n'

                        result = await stream_response.response
                        yield f'data: {json.dumps({"result": _to_dict(result)}, separators=_JSON_SEPARATORS)}\n\n'
                    except Exception as e:
                        _log_failure(error=e, where='stream')
                        yield f'data: {json.dumps({"error": error_body(e)}, separators=_JSON_SEPARATORS)}\n\n'

                return StreamingHttpResponse(event_stream(), content_type='text/event-stream')

            try:
                response = await flow.run(input=action_input, context=action_context, init=init)
                return JsonResponse({'result': _to_dict(response.response)})
            except Exception as e:
                _log_failure(error=e, where='run')
                return _error_response(e)

        return handler

    return decorator
