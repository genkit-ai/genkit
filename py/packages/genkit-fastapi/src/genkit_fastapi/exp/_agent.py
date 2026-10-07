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

"""``serve_agent`` and its snapshot/abort input parsing. Public via ``genkit_fastapi.exp``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from fastapi import APIRouter
from pydantic import BaseModel

from genkit import PublicError
from genkit.exp.agent import Agent, SessionSnapshot
from genkit.plugin_api import Action, ActionKind

from ..handler import _mount_action

StateT = TypeVar('StateT', bound=BaseModel)


def extract_agent_input(body: dict[str, Any]) -> object:
    """Read the agent wire shapes: ``message``, top-level snapshot/session ids, or ``data``/``input``."""
    if 'data' in body:
        return body['data']
    if 'input' in body:
        return body['input']
    if 'message' in body:
        return {'message': {'role': 'user', 'content': [{'text': str(body['message'])}]}}
    if 'snapshotId' in body or 'sessionId' in body:
        return body
    if not body:
        return None
    raise PublicError(
        'INVALID_ARGUMENT',
        'Action request must be wrapped in {"data": ...} object',
    )


def resolve_session_init(body: dict[str, Any], query_params: Mapping[str, str]) -> object:
    """Resolve per-run init, injecting session_id from ``?session_id=`` / ``?thread_id=``."""
    init = body.get('init')
    query_session_id = query_params.get('session_id') or query_params.get('thread_id')
    if not query_session_id:
        return init
    if isinstance(init, dict) and not init.get('session_id') and not init.get('sessionId'):
        return {**init, 'session_id': query_session_id}
    if init is None:
        return {'session_id': query_session_id}
    return init


def _parse_snapshot_lookup_input(input_val: dict[str, Any] | str | None) -> tuple[str | None, str | None]:
    """Parse snapshot lookup params from payload dict or bare snapshot ID string."""
    if isinstance(input_val, str):
        return input_val, None
    if isinstance(input_val, dict):
        sid = input_val.get('snapshotId') or input_val.get('snapshot_id')
        sess_id = input_val.get('sessionId') or input_val.get('session_id')
        if bool(sid) == bool(sess_id):
            raise PublicError(
                'INVALID_ARGUMENT',
                (
                    "getSnapshot requires exactly one of 'snapshotId' (or 'snapshot_id') "
                    "or 'sessionId' (or 'session_id')."
                ),
            )
        return sid, sess_id
    raise PublicError(
        'INVALID_ARGUMENT',
        'getSnapshot input must be a dictionary or snapshot ID string.',
    )


def _parse_abort_input(input_val: dict[str, Any] | str | None) -> str:
    """Parse snapshot ID from payload dict or bare snapshot ID string."""
    if isinstance(input_val, str):
        return input_val
    if isinstance(input_val, dict):
        sid = input_val.get('snapshotId') or input_val.get('snapshot_id')
        if sid:
            return sid
    raise PublicError(
        'INVALID_ARGUMENT',
        "abort requires 'snapshotId' (or 'snapshot_id') in input.",
    )


def serve_agent(
    agent: Agent[StateT],
    *,
    base_path: str | None = None,
    context_dependency: Callable[..., Any] | None = None,
) -> APIRouter:
    """Build an APIRouter serving an agent and its snapshot/abort endpoints over HTTP.

    Mount the returned router like any other::

        app.include_router(serve_agent(weather_agent), prefix='/api')

    Args:
        agent: The agent to serve.
        base_path: Route path. Defaults to /<agent name>.
        context_dependency: A FastAPI dependency whose resolved value becomes the
            action context, applied to the turn, getSnapshot, and abort routes.
            Use this to reuse existing ``Depends``-based auth / resources.

    Returns:
        An APIRouter with the turn route plus snapshot/abort endpoints.
    """
    resolved_base_path = f'/{agent.name}' if base_path is None else base_path
    router = APIRouter(tags=[agent.name])

    _mount_action(
        router,
        resolved_base_path,
        agent,
        context_dependency=context_dependency,
        extract_input=extract_agent_input,
        resolve_init=resolve_session_init,
    )

    if agent.store is not None:

        async def snapshot_fn(input_val: dict[str, Any] | str | None = None) -> SessionSnapshot | None:
            sid, sess_id = _parse_snapshot_lookup_input(input_val)
            return await agent.get_snapshot_data(snapshot_id=sid, session_id=sess_id)

        async def abort_fn(input_val: dict[str, Any] | str | None = None) -> dict[str, object]:
            snapshot_id = _parse_abort_input(input_val)
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

        _mount_action(
            router,
            f'{resolved_base_path}/getSnapshot',
            snapshot_action,
            context_dependency=context_dependency,
            extract_input=extract_agent_input,
            empty_status=404,
        )
        _mount_action(
            router,
            f'{resolved_base_path}/abort',
            abort_action,
            context_dependency=context_dependency,
            extract_input=extract_agent_input,
        )

    return router
