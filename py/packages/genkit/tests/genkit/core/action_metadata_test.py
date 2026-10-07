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

"""What a plugin puts on ActionMetadata is what the Dev UI lists in /api/actions."""

from typing import Any, cast

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel, ValidationError

from genkit import Genkit, ModelResponse
from genkit._core._reflection import create_reflection_asgi_app
from genkit.embedder import EmbedRequest, EmbedResponse, embedder_action_metadata
from genkit.evaluator import EvalFnResponse, EvalRequest, evaluator_action_metadata
from genkit.model import ModelRequest, model_action_metadata
from genkit.plugin_api import Action, ActionKind, ActionMetadata, Plugin, to_json_schema

REQUEST_SCHEMA = {'type': 'object', 'properties': {'prompt': {'type': 'string'}}, 'required': ['prompt']}
RESPONSE_SCHEMA = {'type': 'object', 'properties': {'text': {'type': 'string'}}}


class ListingPlugin(Plugin):
    """Lists the rows it's given and resolves nothing."""

    name = 'acme'

    def __init__(self, rows: list[ActionMetadata]) -> None:
        self.rows = rows

    async def init(self) -> list[Action]:
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        return None

    async def list_actions(self) -> list[ActionMetadata]:
        return self.rows


async def api_actions(ai: Genkit) -> dict[str, Any]:
    app = create_reflection_asgi_app(ai.registry)
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        response = await client.get('/api/actions')
    assert response.status_code == 200
    return response.json()


@pytest.mark.asyncio
async def test_action_metadata_input_schema_is_what_reflection_lists() -> None:
    """A plugin row with `input_schema=S` shows `inputSchema == S` in /api/actions."""
    row = ActionMetadata(action_type=ActionKind.MODEL, name='acme/m', input_schema=REQUEST_SCHEMA)

    listed = await api_actions(Genkit(plugins=[ListingPlugin([row])]))

    assert listed['/model/acme/m']['inputSchema'] == REQUEST_SCHEMA
    assert 'outputSchema' not in listed['/model/acme/m']


@pytest.mark.asyncio
async def test_action_metadata_output_schema_is_what_reflection_lists() -> None:
    """A plugin row with `output_schema=O` shows `outputSchema == O` in /api/actions."""
    row = ActionMetadata(action_type=ActionKind.MODEL, name='acme/m', output_schema=RESPONSE_SCHEMA)

    listed = await api_actions(Genkit(plugins=[ListingPlugin([row])]))

    assert listed['/model/acme/m']['outputSchema'] == RESPONSE_SCHEMA
    assert 'inputSchema' not in listed['/model/acme/m']


def test_action_metadata_input_json_schema_keyword_raises_validation_error() -> None:
    """`ActionMetadata(input_json_schema=...)` fails at construction instead of being dropped."""
    old_keyword: dict[str, Any] = {'input_json_schema': REQUEST_SCHEMA}

    with pytest.raises(ValidationError, match='input_json_schema'):
        ActionMetadata(action_type=ActionKind.MODEL, name='acme/m', **old_keyword)


def test_action_metadata_output_json_schema_keyword_raises_validation_error() -> None:
    """`ActionMetadata(output_json_schema=...)` fails at construction instead of being dropped."""
    old_keyword: dict[str, Any] = {'output_json_schema': RESPONSE_SCHEMA}

    with pytest.raises(ValidationError, match='output_json_schema'):
        ActionMetadata(action_type=ActionKind.MODEL, name='acme/m', **old_keyword)


def test_action_metadata_input_schema_class_raises_validation_error() -> None:
    """`ActionMetadata(input_schema=SomeModel)` fails; the field takes a JSON schema dict."""

    class Prompt(BaseModel):
        prompt: str

    with pytest.raises(ValidationError, match='input_schema'):
        ActionMetadata(action_type=ActionKind.MODEL, name='acme/m', input_schema=cast(Any, Prompt))


@pytest.mark.asyncio
async def test_listed_model_embedder_evaluator_send_request_schema() -> None:
    """Rows from the model/embedder/evaluator metadata helpers carry their request schema to /api/actions."""
    model_row = model_action_metadata(name='acme/m')
    embedder_row = embedder_action_metadata(name='acme/e')
    evaluator_row = evaluator_action_metadata(name='acme/ev')

    listed = await api_actions(Genkit(plugins=[ListingPlugin([model_row, embedder_row, evaluator_row])]))

    assert model_row.input_schema == to_json_schema(ModelRequest)
    assert model_row.output_schema == to_json_schema(ModelResponse)
    assert listed['/model/acme/m']['inputSchema'] == to_json_schema(ModelRequest)
    assert listed['/model/acme/m']['outputSchema'] == to_json_schema(ModelResponse)

    assert embedder_row.input_schema == to_json_schema(EmbedRequest)
    assert embedder_row.output_schema == to_json_schema(EmbedResponse)
    assert listed['/embedder/acme/e']['inputSchema'] == to_json_schema(EmbedRequest)
    assert listed['/embedder/acme/e']['outputSchema'] == to_json_schema(EmbedResponse)

    assert evaluator_row.input_schema == to_json_schema(EvalRequest)
    assert evaluator_row.output_schema == to_json_schema(list[EvalFnResponse])
    assert listed['/evaluator/acme/ev']['inputSchema'] == to_json_schema(EvalRequest)
    assert listed['/evaluator/acme/ev']['outputSchema'] == to_json_schema(list[EvalFnResponse])


@pytest.mark.asyncio
async def test_flow_input_schema_in_reflection_unchanged() -> None:
    """A typed flow lists its input type's schema; a flow with no input lists no inputSchema."""
    ai = Genkit()

    class Topic(BaseModel):
        name: str

    @ai.flow()
    async def outline(topic: Topic) -> str:
        return topic.name

    @ai.flow()
    async def ping() -> str:
        return 'pong'

    listed = await api_actions(ai)

    assert listed['/flow/outline']['inputSchema'] == {
        'properties': {'name': {'title': 'Name', 'type': 'string'}},
        'required': ['name'],
        'title': 'Topic',
        'type': 'object',
    }
    assert listed['/flow/outline']['outputSchema'] == {'type': 'string'}
    assert 'inputSchema' not in listed['/flow/ping']
    assert listed['/flow/ping']['outputSchema'] == {'type': 'string'}
