# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for serve_agent in genkit_fastapi.exp."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

# serve_agent needs the agent subsystem; skip the whole module where it isn't built.
_genkit_agent = pytest.importorskip('genkit.exp.agent', reason='agents API not available')
if not hasattr(_genkit_agent, 'InMemorySessionStore'):
    pytest.skip('agents API not available', allow_module_level=True)
InMemorySessionStore = _genkit_agent.InMemorySessionStore
AgentInit = _genkit_agent.AgentInit

from genkit_fastapi import handle_genkit_request  # noqa: E402
from genkit_fastapi.exp import serve_agent  # noqa: E402

from genkit._ai._agents._client import error_from_http  # noqa: E402
from genkit._core._error import RuntimeErrorReason  # noqa: E402
from genkit._core._model import (  # noqa: E402
    Message,
    ModelResponse,
    ModelResponseChunk as ModelResponseChunkModel,
    Part,
)
from genkit._core._typing import FinishReason, Role  # noqa: E402
from genkit.exp import Genkit  # noqa: E402
from genkit.testing import define_scripted_model  # noqa: E402


def build_agent(name: str, *, server_managed: bool = True) -> Any:
    """A prompt agent whose model replies with a fixed line; server-backed unless told otherwise."""
    ai = Genkit()
    define_scripted_model(
        ai,
        name='scriptedModel',
        responses=[
            ModelResponse(
                finish_reason=FinishReason.STOP,
                message=Message(role=Role.MODEL, content=[Part.from_text('Hi there!')]),
            )
        ],
        chunks=[[ModelResponseChunkModel(role=Role.MODEL, content=[Part.from_text('Hi there!')])]],
    )
    ai.define_prompt(name=name, model='scriptedModel', system='You echo things.')
    return ai.define_prompt_agent(name=name, store=InMemorySessionStore() if server_managed else None)


def sse_events(text: str) -> list[dict[str, Any]]:
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('data: '):
            records.append(json.loads(line[6:].strip()))
        elif line.startswith('error: '):
            records.append(json.loads(line[7:].strip()))
    return records


def client(agent: Any, **kwargs: Any) -> TestClient:
    """Mount ``agent`` under ``/api`` and return a test client."""
    app = FastAPI()
    app.include_router(serve_agent(agent, base_path='/chat', **kwargs), prefix='/api')
    return TestClient(app)


def test_turn_streams_sse_and_final_result() -> None:
    """A turn returns SSE events and a final {"result": <AgentOutput>}."""
    client_obj = client(build_agent('echoAgent'))

    response = client_obj.post('/api/chat?stream=true', json={'message': 'Hi'})

    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/event-stream')
    records = sse_events(response.text)
    assert 'result' in records[-1]
    # The reply text lands in the settled AgentOutput.
    assert 'Hi there!' in json.dumps(records[-1]['result'])


def test_base_path_defaults_to_agent_name() -> None:
    """Omitting base_path mounts the turn route at /<agent name>."""
    app = FastAPI()
    app.include_router(serve_agent(build_agent('weatherAgent')), prefix='/api')  # no base_path
    client_obj = TestClient(app)

    response = client_obj.post('/api/weatherAgent', json={'message': 'Hi'})

    assert response.status_code == 200
    assert 'Hi there!' in json.dumps(response.json()['result'])


def test_turn_shorthand_matches_wire_format() -> None:
    """The {"input": ..., "init": ...} wire shape works the same as the shorthand."""
    client_obj = client(build_agent('wireAgent'))

    body = {'input': {'message': {'role': 'user', 'content': [{'text': 'Hi'}]}}, 'init': {}}
    response = client_obj.post('/api/chat', json=body)

    assert response.status_code == 200
    assert 'Hi there!' in json.dumps(response.json()['result'])


def test_get_snapshot_missing_returns_404() -> None:
    """getSnapshot for an unknown snapshot id returns 404."""
    client_obj = client(build_agent('snapAgent'))

    response = client_obj.post('/api/chat/getSnapshot', json={'snapshotId': 'does-not-exist'})

    assert response.status_code == 404


def test_context_dependency_gates_the_turn() -> None:
    """A context_dependency that raises stops the turn before it streams."""

    async def deny() -> dict[str, object]:
        raise HTTPException(status_code=401, detail='no token')

    client_obj = client(build_agent('depAuthAgent'), context_dependency=deny)

    response = client_obj.post('/api/chat', json={'message': 'Hi'})

    assert response.status_code == 401


def test_context_dependency_allows_the_turn() -> None:
    """A resolved context_dependency lets the turn run and stream normally."""

    async def allow() -> dict[str, object]:
        return {'uid': 'user-123'}

    client_obj = client(build_agent('depOkAgent'), context_dependency=allow)

    response = client_obj.post('/api/chat?stream=true', json={'message': 'Hi'})

    assert response.status_code == 200
    assert 'Hi there!' in json.dumps(sse_events(response.text)[-1]['result'])


def test_serve_agent_message_body_starts_a_turn() -> None:
    """POST {"message": "hi"} to serve_agent starts a turn."""
    client_obj = client(build_agent('msgAgent'))

    response = client_obj.post('/api/chat', json={'message': 'hi'})

    assert response.status_code == 200
    assert 'Hi there!' in json.dumps(response.json()['result'])


def test_serve_agent_session_id_query_param_continues_session() -> None:
    """POST /chat?session_id=s1 runs the turn in session s1."""
    client_obj = client(build_agent('sessionAgent'))

    response = client_obj.post('/api/chat?session_id=s1', json={'message': 'Hi'})

    assert response.status_code == 200
    result = response.json()['result']
    assert result['sessionId'] == 's1'
    assert 'Hi there!' in json.dumps(result)

    snap = client_obj.post('/api/chat/getSnapshot', json={'sessionId': 's1'})
    assert snap.status_code == 200
    assert snap.json()['result']['sessionId'] == 's1'


def test_handle_genkit_request_agent_route_with_data_envelope_runs_turn() -> None:
    """A custom route that calls handle_genkit_request with {"data": ...} runs a turn."""
    agent = build_agent('handRolledAgent')
    app = FastAPI()

    @app.post('/custom', response_model=None)
    async def custom(request: Request) -> object:
        # A real app would build this context and init from its own Depends params.
        return await handle_genkit_request(
            request,
            action=agent,
            context={'uid': 'user-123'},
            init=AgentInit(session_id='session-789'),
        )

    client_obj = TestClient(app)

    response = client_obj.post(
        '/custom',
        json={'data': {'message': {'role': 'user', 'content': [{'text': 'Hi'}]}}},
    )

    assert response.status_code == 200
    result = response.json()['result']
    assert result['sessionId'] == 'session-789'
    assert 'Hi there!' in json.dumps(result)


@pytest.mark.parametrize(
    'path, body, message',
    [
        pytest.param('/api/chat', {'foo': 'bar'}, 'Action request must be wrapped in {"data": ...} object', id='turn'),
        pytest.param(
            '/api/chat/getSnapshot',
            {'snapshotId': 's1', 'sessionId': 'x1'},
            "getSnapshot requires exactly one of 'snapshotId' (or 'snapshot_id') or 'sessionId' (or 'session_id').",
            id='get-snapshot',
        ),
        pytest.param(
            '/api/chat/abort', {'data': {}}, "abort requires 'snapshotId' (or 'snapshot_id') in input.", id='abort'
        ),
    ],
)
def test_serve_agent_bad_input_returns_400_with_its_message(path: str, body: dict[str, Any], message: str) -> None:
    """Agent-route input errors are fixed adapter text, so the caller sees what to fix."""
    response = client(build_agent('badInputAgent')).post(path, json=body)

    assert response.status_code == 400
    assert response.json() == {'message': message, 'status': 'INVALID_ARGUMENT'}


@pytest.mark.parametrize(
    'server_managed, init, message, reason',
    [
        pytest.param(
            True,
            {'state': {'custom': {'table': 4}}},
            "Cannot send 'state' to agent 'initAgent': this agent uses a server-managed store. "
            "Send 'snapshotId' or 'sessionId' instead.",
            None,
            id='state-to-server-managed',
        ),
        pytest.param(
            False,
            {'snapshotId': 'snap-1'},
            "Cannot use 'snapshotId' with agent 'initAgent': this agent has no store configured "
            "(client-managed state). Send 'state' instead.",
            RuntimeErrorReason.SESSION_STORE_NOT_CONFIGURED,
            id='snapshot-id-without-store',
        ),
    ],
)
def test_serve_agent_init_mismatch_returns_agent_init_error(
    server_managed: bool, init: dict[str, Any], message: str, reason: RuntimeErrorReason | None
) -> None:
    """AgentInitError is a PublicError, so a remote client sees the status, message, and reason it would in-process."""
    response = client(build_agent('initAgent', server_managed=server_managed)).post(
        '/api/chat', json={'input': {'message': {'role': 'user', 'content': [{'text': 'Hi'}]}}, 'init': init}
    )

    assert response.status_code == 400
    expected: dict[str, Any] = {'message': message, 'status': 'FAILED_PRECONDITION'}
    if reason is not None:
        expected['details'] = {'reason': reason.value}
    assert response.json() == expected

    err = error_from_http(status_code=response.status_code, body=response.text)
    assert err.status == 'FAILED_PRECONDITION'
    assert err.reason is reason
