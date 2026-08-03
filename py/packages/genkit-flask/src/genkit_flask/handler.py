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
from asyncio import AbstractEventLoop
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterable
from typing import Any, Protocol, TypeAlias, TypeVar

from flask import Response, request
from pydantic import BaseModel

from genkit import Genkit, GenkitError
from genkit._core._action import Action
from genkit.plugin_api import (
    ContextProvider,
    RequestData,
    get_callable_json,
)

# Compact JSON (no spaces) for smaller wire payload.
_JSON_SEPARATORS = (',', ':')


def _to_dict(obj: Any) -> Any:  # noqa: ANN401
    """Convert object to dict if it's a Pydantic model, otherwise return as-is."""
    return obj.model_dump() if isinstance(obj, BaseModel) else obj


T = TypeVar('T')


def _create_loop() -> AbstractEventLoop:
    """Return a live asyncio event loop for sync Flask streaming bridges.

    Never reuse a closed loop: warm Cloud Run / sync hosts often close the
    previous request's loop (#4925), and ``asyncio.get_event_loop()`` can still
    return that closed object.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


class _LoopRunner(Protocol):
    def run_until_complete(self, coro: Awaitable[Any]) -> Any: ...


class _LazyLoop:
    """Resolve a live event loop when sync streaming iteration actually starts.

    Flask may run the async ``handler`` on a temporary loop (e.g. asgiref
    ``async_to_sync``) that is closed as soon as ``handler()`` returns — before
    the returned streaming iterable is consumed. Resolving the loop lazily
    avoids ``RuntimeError: Event loop is closed`` (#4925).
    """

    def run_until_complete(self, coro: Awaitable[Any]) -> Any:
        return _create_loop().run_until_complete(coro)


def _iter_over_async(ait: AsyncIterable[T], loop: _LoopRunner) -> Iterable[T]:
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
    def __init__(self) -> None:
        super().__init__(request=request)
        self.method = request.method

        self.headers = {}
        for key, value in request.headers:
            self.headers[key.lower()] = value

        input_data = request.get_json()
        self.input = input_data.get('data') if input_data else None


def genkit_flask_handler(
    ai: Genkit,
    context_provider: ContextProvider | None = None,
) -> Callable[[Action], Callable[..., Awaitable[FlaskRouteReturn]]]:
    """A decorator for serving Genkit flows via a flask sever.

    ```python
    from genkit_flask import genkit_flask_handler

    app = Flask(__name__)


    @app.post('/chat')
    @genkit_flask_handler(ai)
    @ai.flow()
    async def say_hi(name: str, ctx):
        return await ai.generate(
            on_chunk=ctx.send_chunk,
            prompt=f'tell a medium sized joke about {name}',
        )
    ```

    """

    def decorator(flow: Action) -> Callable[..., Awaitable[FlaskRouteReturn]]:
        if not isinstance(flow, Action):
            raise GenkitError(status='INVALID_ARGUMENT', message='must apply @genkit_flask_handler on a @flow')

        async def handler() -> FlaskRouteReturn:
            input_data = request.get_json()
            if 'data' not in input_data:
                return Response(status=400, response='flow request must be wrapped in {"data": data} object')

            request_data = _FlaskRequestData()
            context = None
            action_context: dict[str, object] | None = None
            if context_provider:
                context = context_provider(request_data)
                if asyncio.iscoroutine(context):
                    context = await context
                if isinstance(context, dict):
                    action_context = context

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
                        ex = e
                        if isinstance(ex, GenkitError):
                            ex = ex.cause
                        yield f'data: {json.dumps({"error": get_callable_json(ex)}, separators=_JSON_SEPARATORS)}\n\n'

                # Lazily resolve the loop at iteration time — Flask may close the
                # temporary async-runner loop before consuming this iterable (#4925).
                iter = _iter_over_async(async_gen(), _LazyLoop())
                return iter
            else:
                try:
                    response = await flow.run(input_data.get('data'), context=action_context, init=init)
                    return {'result': _to_dict(response.response)}
                except Exception as e:
                    ex = e
                    if isinstance(ex, GenkitError):
                        ex = ex.cause
                    return Response(
                        status=500,
                        response=json.dumps(get_callable_json(ex), separators=_JSON_SEPARATORS),
                    )

        return handler

    return decorator
