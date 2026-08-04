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
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable
from typing import Any, TypeAlias, TypeVar

from flask import Blueprint, Response, request
from pydantic import BaseModel

from genkit import Genkit, GenkitError
from genkit._core._action import Action, ActionKind
from genkit.agent import Agent, SessionSnapshot
from genkit.plugin_api import (
    ContextProvider,
    RequestData,
    extract_action_input,
    get_callable_json,
    parse_abort_input,
    parse_snapshot_lookup_input,
)

# Compact JSON (no spaces) for smaller wire payload.
_JSON_SEPARATORS = (',', ':')


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
    def __init__(self) -> None:
        super().__init__(request=request)
        self.method = request.method

        self.headers = {}
        for key, value in request.headers:
            self.headers[key.lower()] = value

        input_data = request.get_json(silent=True)
        self.input = input_data.get('data') if isinstance(input_data, dict) else None


def genkit_flask_handler(
    ai: Genkit | None = None,
    context_provider: ContextProvider | None = None,
) -> Callable[[Action], Callable[..., Any]]:
    """A decorator for serving Genkit flows via a flask server."""
    loop = _create_loop()

    def decorator(flow: Action) -> Callable[..., Any]:
        if not isinstance(flow, Action):
            raise GenkitError(status='INVALID_ARGUMENT', message='must apply @genkit_flask_handler on a @flow')

        async def handler() -> FlaskRouteReturn:
            try:
                raw_body = request.get_json(silent=True)
                body = (
                    raw_body
                    if isinstance(raw_body, dict)
                    else ({})
                    if raw_body is None and not request.data
                    else raw_body
                )
                if not isinstance(body, dict):
                    raise GenkitError(status='INVALID_ARGUMENT', message='Action request must be a JSON object')
                input_val = extract_action_input(body)
            except GenkitError as err:
                ex = err.cause if err.cause is not None else err
                return Response(
                    status=400,
                    response=json.dumps(get_callable_json(ex), separators=_JSON_SEPARATORS),
                    content_type='application/json',
                )

            request_data = _FlaskRequestData()
            action_context: dict[str, object] | None = None
            if context_provider:
                try:
                    context = context_provider(request_data)
                    if asyncio.iscoroutine(context):
                        context = await context
                    if isinstance(context, dict):
                        action_context = context
                except Exception as e:
                    ex = e.cause if isinstance(e, GenkitError) and e.cause is not None else e
                    return Response(
                        status=500,
                        response=json.dumps(get_callable_json(ex), separators=_JSON_SEPARATORS),
                        content_type='application/json',
                    )

            accept = request_data.headers.get('accept', '')
            stream = 'text/event-stream' in accept or request.args.get('stream') == 'true'
            init = body.get('init') if isinstance(body, dict) else None
            if stream:

                async def async_gen() -> AsyncIterator[str]:
                    try:
                        stream_response = flow.stream(input_val, context=action_context, init=init)
                        async for chunk in stream_response.stream:
                            yield f'data: {json.dumps({"message": _to_dict(chunk)}, separators=_JSON_SEPARATORS)}\n\n'

                        result = await stream_response.response
                        yield f'data: {json.dumps({"result": _to_dict(result)}, separators=_JSON_SEPARATORS)}\n\n'
                    except Exception as e:
                        ex = e.cause if isinstance(e, GenkitError) and e.cause is not None else e
                        yield f'data: {json.dumps({"error": get_callable_json(ex)}, separators=_JSON_SEPARATORS)}\n\n'

                iter = _iter_over_async(async_gen(), loop)
                return iter
            else:
                try:
                    response = await flow.run(input_val, context=action_context, init=init)
                    if response.response is None and flow.kind == ActionKind.AGENT_SNAPSHOT:
                        return Response(status=404)
                    return {'result': _to_dict(response.response)}
                except Exception as e:
                    ex = e.cause if isinstance(e, GenkitError) and e.cause is not None else e
                    return Response(
                        status=500,
                        response=json.dumps(get_callable_json(ex), separators=_JSON_SEPARATORS),
                        content_type='application/json',
                    )

        return handler

    return decorator


def serve_flow(
    flow: Action,
    *,
    base_path: str | None = None,
    context_provider: ContextProvider | None = None,
) -> Blueprint:
    """Build a Flask Blueprint serving a single flow over HTTP."""
    resolved_base_path = f'/{flow.name}' if base_path is None else base_path
    bp = Blueprint(f'genkit_flow_{flow.name}', __name__)
    bp.add_url_rule(
        resolved_base_path,
        endpoint=flow.name,
        view_func=genkit_flask_handler(None, context_provider=context_provider)(flow),  # type: ignore[arg-type]
        methods=['POST'],
    )
    return bp


def serve_agent(
    agent: Agent[Any],
    *,
    base_path: str | None = None,
    context_provider: ContextProvider | None = None,
) -> Blueprint:
    """Build a Flask Blueprint serving an agent and its snapshot/abort endpoints over HTTP."""
    resolved_base_path = f'/{agent.name}' if base_path is None else base_path
    bp = Blueprint(f'genkit_agent_{agent.name}', __name__)

    bp.add_url_rule(
        resolved_base_path,
        endpoint=f'{agent.name}_turn',
        view_func=genkit_flask_handler(None, context_provider=context_provider)(agent),  # type: ignore[arg-type]
        methods=['POST'],
    )

    if agent.store is not None:

        async def snapshot_fn(input_val: dict[str, Any] | str | None = None) -> SessionSnapshot | None:
            sid, sess_id = parse_snapshot_lookup_input(input_val)
            return await agent.get_snapshot_data(snapshot_id=sid, session_id=sess_id)

        async def abort_fn(input_val: dict[str, Any] | str | None = None) -> dict[str, object]:
            snapshot_id = parse_abort_input(input_val)
            status = await agent.abort_snapshot_data(snapshot_id)
            return {'snapshotId': snapshot_id, 'status': str(status) if status else None}

        snapshot_action = Action(
            kind=ActionKind.AGENT_SNAPSHOT,
            name=f'{agent.name}_snapshot',
            fn=snapshot_fn,
            description=f'Gets snapshot data for {agent.name}',
        )
        abort_action = Action(
            kind=ActionKind.AGENT_ABORT,
            name=f'{agent.name}_abort',
            fn=abort_fn,
            description=f'Aborts {agent.name} agent by snapshotId',
        )

        bp.add_url_rule(
            f'{resolved_base_path}/getSnapshot',
            endpoint=f'{agent.name}_getSnapshot',
            view_func=genkit_flask_handler(None, context_provider=context_provider)(snapshot_action),  # type: ignore[arg-type]
            methods=['POST'],
        )
        bp.add_url_rule(
            f'{resolved_base_path}/abort',
            endpoint=f'{agent.name}_abort',
            view_func=genkit_flask_handler(None, context_provider=context_provider)(abort_action),  # type: ignore[arg-type]
            methods=['POST'],
        )

    return bp
