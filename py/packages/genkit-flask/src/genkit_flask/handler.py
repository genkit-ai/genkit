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

"""Genkit Flask plugin."""

import asyncio
import json
import logging
from asyncio import AbstractEventLoop
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterable
from typing import Any, TypeAlias, TypeVar, cast

from flask import Response, request
from pydantic import BaseModel

from genkit import ContextProvider, Genkit, GenkitError, PublicError, RequestData
from genkit._core._context import joined_headers
from genkit._core._error import log_served_failure, served_error_json, served_stream_error_event
from genkit.plugin_api import Action

logger = logging.getLogger(__name__)

# Compact JSON (no spaces) for smaller wire payload.
_JSON_SEPARATORS = (',', ':')


def _error_response(error: Exception, status: int | None = None) -> Response:
    resolved_status, body = served_error_json(error=error)
    return Response(
        status=resolved_status if status is None else status,
        response=body,
        mimetype='application/json',
    )


def _parse_request_json() -> object:
    """Parse the Flask body as JSON, or raise a 400 PublicError."""
    try:
        raw = request.get_data()
        return json.loads(raw.decode('utf-8')) if raw else {}
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise PublicError('INVALID_ARGUMENT', 'request body must be valid JSON') from err


def _to_dict(obj: Any) -> Any:  # noqa: ANN401
    """Convert object to dict if it's a Pydantic model, otherwise return as-is."""
    return obj.model_dump() if isinstance(obj, BaseModel) else obj


T = TypeVar('T')


def _create_loop() -> AbstractEventLoop:
    """Creates a new asyncio event loop or returns the current one."""
    try:
        return asyncio.get_event_loop()
    except Exception:
        return asyncio.new_event_loop()


def _iter_over_async(ait: AsyncIterable[T], loop: AbstractEventLoop) -> Iterable[T]:
    """Synchronously iterates over an AsyncIterable using a specified event loop."""
    ait_iter = ait.__aiter__()

    async def get_next() -> tuple[bool, T | None]:
        try:
            obj = await ait_iter.__anext__()
            return False, obj
        except StopAsyncIteration:
            return True, None

    while True:
        done, obj = loop.run_until_complete(get_next())
        if done:
            break
        assert obj is not None
        yield obj


# Type alias for Flask-compatible route handler return type
FlaskRouteReturn: TypeAlias = Response | dict[str, object] | Iterable[Any]


class _FlaskRequestData(RequestData):
    def __init__(self, input_data: dict[str, Any]) -> None:
        super().__init__(
            request=request,
            method=request.method,
            headers=joined_headers(request.headers.items()),
            input=input_data.get('data'),
        )


def genkit_flask_handler(
    ai: Genkit,
    context_provider: ContextProvider | None = None,
) -> Callable[[Action], Callable[..., Awaitable[FlaskRouteReturn]]]:
    """A decorator for serving Genkit flows via a flask sever.

    ```python
    from genkit import ActionRunContext
    from genkit_flask import genkit_flask_handler

    app = Flask(__name__)


    @app.post('/chat')
    @genkit_flask_handler(ai)
    @ai.flow()
    async def say_hi(name: str, ctx: ActionRunContext) -> str:
        stream = ai.generate_stream(prompt=f'tell a joke about {name}')
        async for chunk in stream.stream:
            if chunk.text:
                ctx.send_chunk(chunk.text)
        res = await stream.response
        return res.text
    ```

    """
    loop = _create_loop()

    def decorator(flow: Action) -> Callable[..., Awaitable[FlaskRouteReturn]]:
        if not isinstance(flow, Action):
            raise GenkitError(status='INVALID_ARGUMENT', message='must apply @genkit_flask_handler on a @flow')

        async def handler() -> FlaskRouteReturn:
            try:
                input_data = _parse_request_json()
            except PublicError as e:
                log_served_failure(adapter_logger=logger, error=e, where='run')
                return _error_response(e)
            if not isinstance(input_data, dict) or 'data' not in input_data:
                return _error_response(
                    PublicError('INVALID_ARGUMENT', 'flow request must be wrapped in {"data": data} object')
                )
            input_data = cast(dict[str, Any], input_data)

            request_data = _FlaskRequestData(input_data)
            action_context: dict[str, object] | None = None
            if context_provider:
                try:
                    context = context_provider(request_data)
                    if asyncio.iscoroutine(context):
                        context = await context
                    if isinstance(context, dict):
                        action_context = context
                except Exception as e:
                    log_served_failure(adapter_logger=logger, error=e, where='context provider')
                    return _error_response(e)

            # Substring match so Accept: text/event-stream, */* (and similar) still streams.
            accept = request_data.headers.get('accept', '')
            stream = 'text/event-stream' in accept or request.args.get('stream') == 'true'
            init = input_data.get('init')
            if stream:

                async def async_gen() -> AsyncIterator[str]:
                    try:
                        stream_response = flow.stream(input_data.get('data'), context=action_context, init=init)
                        async for chunk in stream_response.stream:
                            yield f'data: {json.dumps({"message": _to_dict(chunk)}, separators=_JSON_SEPARATORS)}\n\n'

                        result = await stream_response.response
                        yield f'data: {json.dumps({"result": _to_dict(result)}, separators=_JSON_SEPARATORS)}\n\n'
                    except Exception as e:
                        log_served_failure(adapter_logger=logger, error=e, where='stream')
                        yield served_stream_error_event(error=e)

                iter = _iter_over_async(async_gen(), loop)
                return iter
            else:
                try:
                    response = await flow.run(input_data.get('data'), context=action_context, init=init)
                    return {'result': _to_dict(response.response)}
                except Exception as e:
                    log_served_failure(adapter_logger=logger, error=e, where='run')
                    return _error_response(e)

        return handler

    return decorator
