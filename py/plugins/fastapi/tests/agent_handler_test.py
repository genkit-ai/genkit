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

"""Tests for serve_agent."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

# serve_agent needs the agent subsystem; skip the whole module where it isn't built.
_genkit_agent = pytest.importorskip('genkit.agent', reason='agents API not available')
if not hasattr(_genkit_agent, 'InMemorySessionStore'):
    pytest.skip('agents API not available', allow_module_level=True)
InMemorySessionStore = _genkit_agent.InMemorySessionStore

from genkit import Genkit  # noqa: E402
from genkit._ai._testing import define_programmable_model  # noqa: E402
from genkit._core._model import Message, ModelResponse, ModelResponseChunk as ModelResponseChunkModel  # noqa: E402
from genkit._core._typing import FinishReason, Part, Role, TextPart  # noqa: E402
from genkit.plugins.fastapi import serve_agent  # noqa: E402


def _build_agent(name: str) -> Any:
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


def _ndjson_lines(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _client(agent: Any, **kwargs: Any) -> TestClient:
    """Mount ``agent`` under ``/api`` and return a test client."""
    app = FastAPI()
    app.include_router(serve_agent(agent, base_path='/chat', **kwargs), prefix='/api')
    return TestClient(app)


def test_turn_streams_ndjson_and_final_result() -> None:
    """A turn returns newline-delimited chunks and a final {"result": <AgentOutput>}."""
    client = _client(_build_agent('echoAgent'))

    response = client.post('/api/chat', json={'message': 'Hi'})

    assert response.status_code == 200
    records = _ndjson_lines(response.text)
    assert 'result' in records[-1]
    # The reply text lands in the settled AgentOutput.
    assert 'Hi there!' in json.dumps(records[-1]['result'])


def test_base_path_defaults_to_agent_name() -> None:
    """Omitting base_path mounts the turn route at /<agent name>."""
    app = FastAPI()
    app.include_router(serve_agent(_build_agent('weatherAgent')), prefix='/api')  # no base_path
    client = TestClient(app)

    response = client.post('/api/weatherAgent', json={'message': 'Hi'})

    assert response.status_code == 200
    assert 'Hi there!' in json.dumps(_ndjson_lines(response.text)[-1]['result'])


def test_turn_shorthand_matches_wire_format() -> None:
    """The {"input": ..., "init": ...} wire shape works the same as the shorthand."""
    client = _client(_build_agent('wireAgent'))

    body = {'input': {'message': {'role': 'user', 'content': [{'text': 'Hi'}]}}, 'init': {}}
    response = client.post('/api/chat', json=body)

    assert response.status_code == 200
    assert 'Hi there!' in json.dumps(_ndjson_lines(response.text)[-1]['result'])


def test_context_provider_rejects_before_stream() -> None:
    """A context_provider that raises stops the turn with a normal error response."""

    def deny(_request_data: Any) -> dict[str, object]:
        raise HTTPException(status_code=401, detail='no token')

    client = _client(_build_agent('authAgent'), context_provider=deny)

    response = client.post('/api/chat', json={'message': 'Hi'})

    assert response.status_code == 401


def test_context_provider_receives_request() -> None:
    """The provider sees the incoming request headers before the turn runs."""
    seen: dict[str, Any] = {}

    def provide(request_data: Any) -> dict[str, object]:
        seen['authorization'] = request_data.request.headers.get('authorization')
        return {'uid': 'user-123'}

    client = _client(_build_agent('ctxAgent'), context_provider=provide)

    response = client.post('/api/chat', json={'message': 'Hi'}, headers={'Authorization': 'Bearer abc'})

    assert response.status_code == 200
    assert seen['authorization'] == 'Bearer abc'


def test_get_snapshot_missing_returns_404() -> None:
    """getSnapshot for an unknown snapshot id returns 404."""
    client = _client(_build_agent('snapAgent'))

    response = client.post('/api/chat/getSnapshot', json={'snapshotId': 'does-not-exist'})

    assert response.status_code == 404
