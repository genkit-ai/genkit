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

"""Genkit's SSE wire frames for a streaming handle."""

import json
import logging
from collections.abc import AsyncIterator, Awaitable
from typing import Protocol

from pydantic_core import to_jsonable_python

from genkit._ai._agents._client import AgentChunk, AgentResponse
from genkit._ai._prompt import ModelStreamResponse
from genkit._core._error import GenkitError, Interrupt, TaskGroupError, error_body, error_status, interrupt_escape_error
from genkit._core._model import ModelResponse
from genkit._core._typing import FinishReason

# Compact JSON, the same bytes the FastAPI, Flask and Django adapters send.
_JSON_SEPARATORS = (',', ':')

_logger = logging.getLogger(__name__)


class StreamHandle(Protocol):
    """Anything with chunks on ``.stream`` and a result on ``.response``.

    ``flow.stream()``, ``ai.generate_stream()``, ``prompt.stream()`` and
    ``chat.send_stream()`` all return one.
    """

    @property
    def stream(self) -> AsyncIterator[object]:
        """Chunks, in order."""
        ...

    @property
    def response(self) -> Awaitable[object]:
        """The final result, once the stream ends."""
        ...


def _frame_value(obj: object) -> object:
    """JSON-ready frame value: camelCase aliases, no null model fields, nested models encoded."""
    # Agent turns stream client-side wrappers; the wire carries the shapes they
    # wrap (AgentStreamChunk / AgentOutput), the same ones serve_agent sends.
    # Gated on the wrapper types, so a user model with a `raw` field goes out whole.
    if isinstance(obj, (AgentChunk, AgentResponse)) and obj.raw is not None:
        obj = obj.raw
    return to_jsonable_python(obj, by_alias=True, exclude_none=True)


def _frame(key: str, value: object) -> str:
    return f'data: {json.dumps({key: value}, separators=_JSON_SEPARATORS)}\n\n'


async def to_sse(handle: StreamHandle) -> AsyncIterator[str]:
    r"""Frame a streaming handle as Genkit SSE lines.

    Yields ``data: {"message": ...}`` for each chunk, then ``data: {"result": ...}``.
    If the run fails, it yields one ``data: {"error": ...}`` instead of the rest
    and logs the failure with Python's logging module (5xx with a traceback,
    4xx as a warning). Only a ``PublicError``'s text reaches the client; anything else goes out as
    ``Internal Error``. These are the frames ``serve_flow``,
    ``handle_genkit_request`` and ``serve_agent`` send.

    The result goes out as the run returned it. ``ai.generate_stream()``,
    ``prompt.stream()``, and flows returning a whole ``ModelResponse`` send its
    ``request`` too (system prompt, tools, docs, history), so stream a flow
    returning ``response.text`` or ``response.output`` from a served route.

    ``.response`` is read only after the stream drains, so a handle that fails
    or is never iterated doesn't leave an un-awaited result behind.

    Example:
        ```python
        from fastapi import Depends
        from fastapi.responses import StreamingResponse
        from genkit.web import to_sse


        @app.post('/deploy/stream')
        async def deploy(service: str, user: User = Depends(current_user)):
            # 1. FastAPI owns the route: typed params, Depends, OpenAPI
            stream = deploy_service.stream(service, context={'user_id': user.id})

            # 2. Genkit owns the wire
            return StreamingResponse(to_sse(stream), media_type='text/event-stream')
            # => data: {"message":"[u_42] building payments"}
            #    data: {"result":"https://payments.run.app"}
        ```

    Args:
        handle: The streaming handle to frame.

    Yields:
        One SSE ``data:`` line per frame, each ending in a blank line.
    """
    try:
        async for chunk in handle.stream:
            yield _frame('message', _frame_value(chunk))
        result = await handle.response
        # ModelStreamResponse boxes turn failures onto .response instead of raising
        # from .stream; re-raise here so SSE callers get the error frame and log.
        if (
            isinstance(handle, ModelStreamResponse)
            and isinstance(result, ModelResponse)
            and result.finish_reason == FinishReason.FAILED
        ):
            raise (
                result._failure.to_exception()
                if result._failure is not None
                else GenkitError(
                    status='INTERNAL',
                    message=result.finish_message or 'Model generation failed',
                )
            )
        yield _frame('result', _frame_value(result))
    except (Exception, Interrupt, TaskGroupError) as exc:
        failure = interrupt_escape_error(exc)
        if failure is not None:
            e: Exception = failure
        elif isinstance(exc, Exception):
            e = exc
        else:
            raise
        action_name = getattr(handle, '_action_name', None)
        if isinstance(action_name, str) and action_name:
            extra = {'genkit_action': action_name}
            if error_status(e) >= 500:
                _logger.exception('served action %r stream failed', action_name, extra=extra)
            else:
                _logger.warning('served action %r stream failed: %s', action_name, e, extra=extra)
        elif error_status(e) >= 500:
            _logger.exception('served stream failed')
        else:
            _logger.warning('served stream failed: %s', e)
        yield _frame('error', error_body(e))
