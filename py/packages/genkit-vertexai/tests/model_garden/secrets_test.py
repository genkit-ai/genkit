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

"""A per-request key in context.secrets on a Model Garden model."""

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, patch

import pytest
from genkit_vertexai.model_garden import ModelGarden
from genkit_vertexai.model_garden.model_garden import ModelGardenModel

from genkit import ActionRunContext, Genkit, GenkitError, Message, ModelResponse, Part, Role
from genkit.model import ModelRequest

CLAUDE_MODEL = 'modelgarden/anthropic/claude-sonnet-4-5@20250929'


def model_garden_app() -> Genkit:
    return Genkit(plugins=[ModelGarden(project='my-project', location='us-east5')])


@pytest.mark.asyncio
@pytest.mark.parametrize('key_name', ['api_key', 'apiKey'])
async def test_generate_model_garden_claude_secrets_api_key_fails_invalid_argument(key_name: str) -> None:
    """A key in `context.secrets` on Model Garden Claude fails with INVALID_ARGUMENT and never builds a client."""
    ai = model_garden_app()

    with patch('genkit_vertexai.model_garden.anthropic.AsyncAnthropicVertex') as vertex_client:
        response = await ai.generate(model=CLAUDE_MODEL, prompt='hi', context={'secrets': {key_name: 'tenant-key'}})

    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert 'Google Cloud credentials' in response.error.message
    vertex_client.assert_not_called()


def openai_compatible_handler() -> Callable[[ModelRequest, ActionRunContext], Awaitable[ModelResponse]]:
    return ModelGardenModel(
        model='meta/llama-3.1-405b-instruct-maas', location='us-east5', project='my-project'
    ).to_openai_compatible_model()


def hi_request() -> ModelRequest:
    return ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('hi')])], config={})


@pytest.mark.asyncio
async def test_generate_model_garden_openai_compatible_secrets_api_key_raises_invalid_argument() -> None:
    """The same key on a Model Garden OpenAI-compatible model raises INVALID_ARGUMENT and never fetches a token."""
    generate = openai_compatible_handler()

    with patch.object(ModelGardenModel, 'create_client') as create_client:
        with pytest.raises(GenkitError) as exc_info:
            await generate(hi_request(), ActionRunContext(context={'secrets': {'api_key': 'tenant-key'}}))

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'Google Cloud credentials' in str(exc_info.value)
    create_client.assert_not_called()


@pytest.mark.asyncio
async def test_generate_model_garden_secrets_without_api_key_runs_on_google_credentials() -> None:
    """Other entries in `context.secrets` don't block the call; it goes on to the app's Google credentials."""
    generate = openai_compatible_handler()
    create_client = AsyncMock(side_effect=RuntimeError('reached the client'))

    with patch.object(ModelGardenModel, 'create_client', create_client):
        with pytest.raises(RuntimeError, match='reached the client'):
            await generate(hi_request(), ActionRunContext(context={'secrets': {'db_password': 'x'}}))

    create_client.assert_called_once()
