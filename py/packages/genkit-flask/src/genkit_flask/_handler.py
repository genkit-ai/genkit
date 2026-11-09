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
import contextlib
import json
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from typing import Any, cast

from flask import Response, request, stream_with_context
from werkzeug.exceptions import HTTPException

from genkit import Genkit, GenkitError, PublicError
from genkit.plugin_api import Action
from genkit.web import (
    ContextProvider,
    RequestData,
    error_json,
    error_sse,
    error_status,
    read_body,
    result_json,
    to_sse,
    wants_stream,
)

logger = logging.getLogger(__name__)  # noqa: TID251


def _log_failure(*, error: Exception, flow: str, where: str) -> None:
    """Log a served-action failure to the host web server's stdlib logger, naming the action.

    Intentionally emits to stdlib ``logging`` rather than ``genkit.get_logger``
    because route failures happen at the HTTP boundary outside the flow span,
    where WSGI/Gunicorn handlers, Sentry ``LoggingIntegration``, and ``pytest``
    ``caplog`` listen. 5xx errors include the traceback.
    """
    extra = {'genkit_action': flow}
    if error_status(error) >= 500:
        logger.exception('served action %r %s failed', flow, where, extra=extra)
    else:
        logger.warning('served action %r %s failed: %s', flow, where, error, extra=extra)


def _error_response(error: Exception, status: int | None = None) -> Response:
    return Response(
        status=error_status(error) if status is None else status,
        response=error_json(error),
        mimetype='application/json',
    )


def _parse_request_json() -> object:
    """Parse the Flask body as JSON, or raise a 400 PublicError."""
    try:
        raw = request.get_data()
        return json.loads(raw.decode('utf-8')) if raw else {}
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise PublicError('INVALID_ARGUMENT', 'request body must be valid JSON') from err


def _iter_over_async(agen: AsyncGenerator[str, None]) -> Generator[str, None, None]:
    """Drive ``agen`` on a private event loop owned by this response.

    WSGI servers iterate the body on a worker thread after the async view has
    returned and asgiref has closed the view's loop. Each response gets its own
    loop, driven from whichever thread iterates it, so concurrent streams never
    share a loop (a shared loop fails with "This event loop is already
    running"). Tasks the flow starts copy the iterating thread's context, which
    ``stream_with_context`` has pushed, so ``flask.g`` and ``flask.request``
    work inside the flow.
    """
    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                item = loop.run_until_complete(anext(agen))
            except StopAsyncIteration:
                break
            yield item
    finally:
        # Werkzeug closes this iterator when the client hangs up. The async side
        # only runs while we drive the loop, so close it here or the flow never
        # hears about it, then reap what it left behind before the loop goes.
        try:
            loop.run_until_complete(agen.aclose())
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.wait(pending))
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()


class _FlaskRequestData(RequestData):
    def __init__(self, input_data: dict[str, Any]) -> None:
        super().__init__(
            request=request,
            method=request.method,
            headers=request.headers.items(),
            input=input_data.get('data'),
        )


def genkit_flask_handler(
    ai: Genkit,
    context_provider: ContextProvider | None = None,
) -> Callable[[Action], Callable[..., Awaitable[Response]]]:
    """A decorator for serving Genkit flows via a Flask server.

    The returned view function is named after the flow (dots replaced with
    underscores) so multiple handlers in one Flask app do not collide on
    the default endpoint name. Pass ``endpoint=`` to Flask's route decorator
    (e.g., ``@app.post('/chat', endpoint='custom_chat')``) to override it.

    ```python
    from genkit import FlowContext
    from genkit_flask import genkit_flask_handler

    app = Flask(__name__)


    @app.post('/chat')
    @genkit_flask_handler(ai)
    @ai.flow()
    async def say_hi(name: str, ctx: FlowContext) -> str:
        stream = ai.generate_stream(prompt=f'tell a joke about {name}')
        async for chunk in stream.stream:
            if chunk.text:
                ctx.send_chunk(chunk.text)
        res = await stream.response
        return res.text
    ```

    """

    def decorator(flow: Action) -> Callable[..., Awaitable[Response]]:
        if not isinstance(flow, Action):
            raise GenkitError(status='INVALID_ARGUMENT', message='must apply @genkit_flask_handler on a @flow')

        async def handler() -> Response:
            try:
                input_data = _parse_request_json()
            except PublicError as e:
                _log_failure(error=e, flow=flow.name, where='run')
                return _error_response(e)
            try:
                action_input = read_body(input_data)
            except PublicError as e:
                return _error_response(e)
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
                except HTTPException:
                    # The app's own abort(401) passes through.
                    raise
                except Exception as e:
                    _log_failure(error=e, flow=flow.name, where='context provider')
                    return _error_response(e)

            init = input_data.get('init')
            if wants_stream(accept=request_data.headers.get('accept'), stream=request.args.get('stream')):

                async def async_gen() -> AsyncGenerator[str, None]:
                    try:
                        stream_response = flow.stream(input=action_input, context=action_context, init=init)
                    except Exception as e:
                        _log_failure(error=e, flow=flow.name, where='stream')
                        yield error_sse(e)
                        return
                    # closing this generator has to cancel the flow before the
                    # loop stops; nothing runs the loop afterwards to collect it.
                    async with contextlib.aclosing(to_sse(stream_response)) as lines:
                        async for line in lines:
                            yield line

                # pyrefly can't solve Flask's AnyStr overload for a str generator.
                sse_body = stream_with_context(_iter_over_async(async_gen()))  # pyrefly: ignore[no-matching-overload]
                return Response(sse_body, mimetype='text/event-stream')
            try:
                response = await flow.run(input=action_input, context=action_context, init=init)
                return Response(response=result_json(response.response), mimetype='application/json')
            except Exception as e:
                _log_failure(error=e, flow=flow.name, where='run')
                return _error_response(e)

        # Flask's default endpoint is the view function's __name__. Name the
        # view after the flow, as if @app.post had decorated the flow itself,
        # so two handlers in one app don't both register as 'handler'.
        # Flask rejects dots in view names.
        view_name = flow.name.replace('.', '_')
        handler.__name__ = view_name
        handler.__qualname__ = view_name
        return handler

    return decorator
