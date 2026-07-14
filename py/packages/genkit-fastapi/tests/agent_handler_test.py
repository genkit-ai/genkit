# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for serve_agent in genkit_fastapi."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

# serve_agent needs the agent subsystem; skip the whole module where it isn't built.
_genkit_agent = pytest.importorskip('genkit.agent', reason='agents API not available')
if not hasattr(_genkit_agent, 'InMemorySessionStore'):
    pytest.skip('agents API not available', allow_module_level=True)
InMemorySessionStore = _genkit_agent.InMemorySessionStore
AgentInit = _genkit_agent.AgentInit

from genkit_fastapi import handle_genkit_request, serve_agent  # noqa: E402

from genkit import Genkit  # noqa: E402
from genkit._ai._testing import define_programmable_model  # noqa: E402
from genkit._core._model import Message, ModelResponse, ModelResponseChunk as ModelResponseChunkModel  # noqa: E402
from genkit._core._typing import FinishReason, Part, Role, TextPart  # noqa: E402


def build_agent(name: str) -> Any:
    """A server-backed prompt agent whose model replies with a fixed line."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    ai.define_prompt(name=name, model='programmableModel', system='You echo things.')
    agent = ai.define_prompt_agent(name=name, store=InMemorySessionStore())

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='Hi there!'))]),
        )
    )
    pm.chunks = [[ModelResponseChunkModel(role=Role.MODEL, content=[Part(root=TextPart(text='Hi there!'))])]]
    return agent


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


def test_handle_genkit_request_powers_a_hand_rolled_route() -> None:
    """The public primitive serves the wire format from a custom endpoint."""
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

    response = client_obj.post('/custom', json={'message': 'Hi'})

    assert response.status_code == 200
    assert 'Hi there!' in json.dumps(response.json()['result'])
