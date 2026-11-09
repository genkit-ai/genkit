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

"""Context caching: no model allowlist, code_execution conflict, context_cache=False."""

import pytest
from genkit_google_genai import GeminiConfig
from genkit_google_genai._models._context_caching._utils import validate_context_cache_request
from genkit_google_genai._models._gemini import GeminiModel
from google.genai import types as genai_types
from pytest_mock import MockerFixture

from genkit import GenkitError, Message, Part, Role
from genkit.model import ModelContext, ModelRequest

_MENU = 'Tasting menu: 1. Smoked salmon tartine 2. Pesto gnocchi 3. Olive oil cake. ' * 50


def _cached_request(config: GeminiConfig | None = None) -> ModelRequest:
    return ModelRequest(
        messages=[
            Message(role=Role.USER, content=[Part.from_text(_MENU)], metadata={'cache': {'ttlSeconds': 300}}),
            Message(role=Role.USER, content=[Part.from_text('Which dish is nut-free?')]),
        ],
        config=config,
    )


@pytest.mark.parametrize('model', ['gemini-flash-latest', 'gemini-pro-latest', 'gemini-3.8-flash', 'gemini-2.5-flash'])
def test_any_model_passes_validation(model: str) -> None:
    """The API decides which models can cache; the plugin no longer blocks live models."""
    assert validate_context_cache_request(_cached_request(), model) is True


def test_code_execution_conflicts_with_cache() -> None:
    """The commented-out code_execution check is back."""
    with pytest.raises(GenkitError) as raised:
        validate_context_cache_request(_cached_request(GeminiConfig(code_execution=True)), 'gemini-flash-latest')
    assert raised.value.status == 'INVALID_ARGUMENT'
    assert 'code execution' in raised.value.message


@pytest.mark.asyncio
async def test_context_cache_false_ignores_cache_markers(mocker: MockerFixture) -> None:
    """context_cache=False sends every message inline and never touches the caches API."""
    client = mocker.AsyncMock()
    client.vertexai = False
    client.aio.models.generate_content.return_value = genai_types.GenerateContentResponse(
        candidates=[genai_types.Candidate(content=genai_types.Content(parts=[genai_types.Part(text='Olive oil cake.')]))]
    )
    gemini = GeminiModel('gemini-flash-latest', client)

    await gemini.generate(_cached_request(GeminiConfig(context_cache=False)), ModelContext())

    client.aio.caches.list.assert_not_called()
    client.aio.caches.create.assert_not_called()
    kwargs = client.aio.models.generate_content.call_args.kwargs
    assert len(kwargs['contents']) == 2
    assert kwargs['config'] is None or kwargs['config'].cached_content is None
