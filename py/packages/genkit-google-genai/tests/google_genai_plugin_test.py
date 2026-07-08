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

"""Tests for Google GenAI plugin."""

import asyncio
import os
import queue
import threading
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genkit_google_genai import (
    EmbeddingTaskType,
    GeminiConfigSchema,
    GeminiEmbeddingModels,
    GoogleAI,
    GoogleAIGeminiVersion,
    VertexAI,
    VertexAIGeminiVersion,
    VertexEmbeddingModels,
)
from genkit_google_genai.google import (
    GOOGLEAI_PLUGIN_NAME,
    VERTEXAI_PLUGIN_NAME,
    GenaiModels,
    googleai_name,
    vertexai_name,
)

from genkit import ActionKind


def test_googleai_name() -> None:
    """Test googleai_name helper function."""
    assert googleai_name('gemini-2.0-flash') == 'googleai/gemini-2.0-flash'
    assert googleai_name('gemini-embedding-001') == 'googleai/gemini-embedding-001'


def test_vertexai_name() -> None:
    """Test vertexai_name helper function."""
    assert vertexai_name('gemini-2.0-flash') == 'vertexai/gemini-2.0-flash'
    assert vertexai_name('imagen-3.0-generate-001') == 'vertexai/imagen-3.0-generate-001'


def test_plugin_names() -> None:
    """Test plugin name constants."""
    assert GOOGLEAI_PLUGIN_NAME == 'googleai'
    assert VERTEXAI_PLUGIN_NAME == 'vertexai'


def test_googleai_initialization_with_api_key() -> None:
    """Test GoogleAI plugin initializes with API key parameter."""
    with patch('genkit_google_genai.google.genai.client.Client'):
        plugin = GoogleAI(api_key='test-key')
        assert plugin.name == 'googleai'
        assert plugin._vertexai is False


def test_googleai_initialization_from_env() -> None:
    """Test GoogleAI plugin reads API key from environment."""
    with patch.dict(os.environ, {'GEMINI_API_KEY': 'env-key'}):
        with patch('genkit_google_genai.google.genai.client.Client'):
            plugin = GoogleAI()
            assert plugin.name == 'googleai'


def test_googleai_initialization_without_api_key() -> None:
    """Test GoogleAI plugin raises error without API key."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            GoogleAI()
        assert 'GEMINI_API_KEY environment variable not set' in str(exc_info.value)
        assert 'Obtain an API key from Google AI Studio' in str(exc_info.value)
        assert 'https://aistudio.google.com/app/apikey' in str(exc_info.value)
        assert 'https://genkit.dev/docs/python/integrations/google-genai/' in str(exc_info.value)


def test_vertexai_initialization() -> None:
    """Test VertexAI plugin initializes correctly."""
    with patch('genkit_google_genai.google.genai.client.Client'):
        plugin = VertexAI(project='test-project', location='us-central1')
        assert plugin.name == 'vertexai'
        assert plugin._vertexai is True


def test_vertexai_initialization_from_env() -> None:
    """Test VertexAI plugin reads project from environment."""
    with patch.dict(os.environ, {'GCLOUD_PROJECT': 'env-project'}):
        with patch('genkit_google_genai.google.genai.client.Client'):
            plugin = VertexAI()
            assert plugin.name == 'vertexai'


@patch('genkit_google_genai.google.genai.client.Client')
@pytest.mark.asyncio
async def test_googleai_runtime_clients_are_loop_local(mock_client_ctor: MagicMock) -> None:
    """GoogleAI runtime clients should be cached per event loop."""
    created: list[MagicMock] = []

    def _new_client(*args: object, **kwargs: object) -> MagicMock:
        client = MagicMock(name=f'client-{len(created)}')
        created.append(client)
        return client

    mock_client_ctor.side_effect = _new_client

    plugin = GoogleAI(api_key='test-key')
    first = plugin._runtime_client()
    second = plugin._runtime_client()
    assert first is second

    q: queue.Queue[MagicMock] = queue.Queue()

    def _other_thread() -> None:
        async def _get_client() -> MagicMock:
            return plugin._runtime_client()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            q.put(loop.run_until_complete(_get_client()))
        finally:
            loop.close()

    t = threading.Thread(target=_other_thread, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive()
    other_loop_client = q.get_nowait()

    assert other_loop_client is not first


def test_genai_models_container() -> None:
    """Test GenaiModels container initialization."""
    models = GenaiModels()
    assert models.gemini == []
    assert models.imagen == []
    assert models.embedders == []
    assert models.veo == []


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_googleai_resolve_model(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Test GoogleAI plugin resolves model actions."""
    mock_list_models.return_value = GenaiModels()

    plugin = GoogleAI(api_key='test-key')
    action = await plugin.resolve(ActionKind.MODEL, 'googleai/gemini-2.0-flash')

    assert action is not None
    assert action.kind == ActionKind.MODEL
    assert action.name == 'googleai/gemini-2.0-flash'


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_googleai_resolve_imagen_model(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Test GoogleAI plugin resolves Imagen image generation models."""
    mock_list_models.return_value = GenaiModels()

    plugin = GoogleAI(api_key='test-key')
    action = await plugin.resolve(ActionKind.MODEL, 'googleai/imagen-3.0-generate-002')

    assert action is not None
    assert action.kind == ActionKind.MODEL
    assert action.name == 'googleai/imagen-3.0-generate-002'


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_googleai_init_registers_imagen_models(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Test GoogleAI init registers Imagen models from dynamic discovery."""
    models = GenaiModels()
    models.imagen = ['imagen-3.0-generate-002']
    mock_list_models.return_value = models

    plugin = GoogleAI(api_key='test-key')
    actions = await plugin.init()

    imagen_actions = [a for a in actions if 'imagen' in a.name]
    assert len(imagen_actions) == 1
    assert imagen_actions[0].name == 'googleai/imagen-3.0-generate-002'
    assert imagen_actions[0].kind == ActionKind.MODEL


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_googleai_list_actions_includes_imagen(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Test GoogleAI list_actions includes Imagen models."""
    models = GenaiModels()
    models.imagen = ['imagen-3.0-generate-002']
    mock_list_models.return_value = models

    plugin = GoogleAI(api_key='test-key')
    actions_list = await plugin.list_actions()

    imagen_actions = [a for a in actions_list if 'imagen' in a.name]
    assert len(imagen_actions) == 1
    assert imagen_actions[0].name == 'googleai/imagen-3.0-generate-002'


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_googleai_resolve_embedder(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Test GoogleAI plugin resolves embedder actions."""
    mock_list_models.return_value = GenaiModels()

    plugin = GoogleAI(api_key='test-key')
    action = await plugin.resolve(ActionKind.EMBEDDER, 'googleai/gemini-embedding-001')

    assert action is not None
    assert action.kind == ActionKind.EMBEDDER
    assert action.name == 'googleai/gemini-embedding-001'


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_googleai_resolve_non_model_returns_none(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Test GoogleAI plugin returns None for unsupported action kinds."""
    mock_list_models.return_value = GenaiModels()

    plugin = GoogleAI(api_key='test-key')
    action = await plugin.resolve(ActionKind.PROMPT, 'some-prompt')
    assert action is None


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_vertexai_resolve_model(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Test VertexAI plugin resolves model actions."""
    mock_list_models.return_value = GenaiModels()

    plugin = VertexAI(project='test-project')
    action = await plugin.resolve(ActionKind.MODEL, 'vertexai/gemini-2.0-flash')

    assert action is not None
    assert action.kind == ActionKind.MODEL
    assert action.name == 'vertexai/gemini-2.0-flash'


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_vertexai_resolve_embedder(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Test VertexAI plugin resolves embedder actions."""
    mock_list_models.return_value = GenaiModels()

    plugin = VertexAI(project='test-project')
    action = await plugin.resolve(ActionKind.EMBEDDER, 'vertexai/gemini-embedding-001')

    assert action is not None
    assert action.kind == ActionKind.EMBEDDER
    assert action.name == 'vertexai/gemini-embedding-001'


def test_embedding_task_types() -> None:
    """Test EmbeddingTaskType enum values."""
    assert EmbeddingTaskType.RETRIEVAL_QUERY is not None
    assert EmbeddingTaskType.RETRIEVAL_DOCUMENT is not None
    assert EmbeddingTaskType.SEMANTIC_SIMILARITY is not None
    assert EmbeddingTaskType.CLASSIFICATION is not None
    assert EmbeddingTaskType.CLUSTERING is not None


def test_gemini_embedding_models_enum() -> None:
    """Test GeminiEmbeddingModels enum has values."""
    # Check that the enum has at least one value
    assert len(list(GeminiEmbeddingModels)) > 0


def test_vertex_embedding_models_enum() -> None:
    """Test VertexEmbeddingModels enum has values."""
    # Check that the enum has at least one value
    assert len(list(VertexEmbeddingModels)) > 0


def test_googleai_gemini_version_enum() -> None:
    """Test GoogleAIGeminiVersion enum has values."""
    # Check that the enum has at least one value
    assert len(list(GoogleAIGeminiVersion)) > 0


def test_vertexai_gemini_version_enum() -> None:
    """Test VertexAIGeminiVersion enum has values."""
    # Check that the enum has at least one value
    assert len(list(VertexAIGeminiVersion)) > 0


def test_gemini_config_schema() -> None:
    """Test GeminiConfigSchema can be instantiated."""
    config = GeminiConfigSchema(temperature=0.7, max_output_tokens=1000)
    assert config.temperature == 0.7
    assert config.max_output_tokens == 1000


def test_gemini_config_schema_defaults() -> None:
    """Test GeminiConfigSchema has proper defaults."""
    config = GeminiConfigSchema()
    # All fields should be optional with None defaults
    assert config.temperature is None
    assert config.max_output_tokens is None


def test_to_finish_reason_mapping() -> None:
    """Gemini finish reason names map onto Genkit's FinishReason."""
    from genkit_google_genai.models.gemini import _to_finish_reason

    from genkit.model import FinishReason

    assert _to_finish_reason('STOP') == FinishReason.STOP
    assert _to_finish_reason('MAX_TOKENS') == FinishReason.LENGTH
    assert _to_finish_reason('SAFETY') == FinishReason.BLOCKED
    assert _to_finish_reason('PROHIBITED_CONTENT') == FinishReason.BLOCKED
    assert _to_finish_reason(None) == FinishReason.OTHER
    assert _to_finish_reason('SOMETHING_NEW') == FinishReason.OTHER


def test_usage_from_metadata() -> None:
    """Token usage is read off a google-genai usage_metadata block."""
    from genkit_google_genai.models.gemini import _usage_from_metadata

    assert _usage_from_metadata(None).input_tokens is None

    md = MagicMock(
        prompt_token_count=5,
        candidates_token_count=7,
        total_token_count=12,
        thoughts_token_count=0,
        cached_content_token_count=0,
    )
    usage = _usage_from_metadata(md)
    assert usage.input_tokens == 5
    assert usage.output_tokens == 7
    assert usage.total_tokens == 12


@pytest.mark.asyncio
async def test_streaming_generate_propagates_finish_reason_and_usage() -> None:
    """A streamed turn reports the finish reason and usage that ride on the trailing chunks.

    Without this the model would stream content but return no finish reason, so
    agents (which always stream) could never tell a turn actually completed.
    """
    from genkit_google_genai.models.gemini import GeminiModel

    from genkit.model import FinishReason

    # google-genai delivers the finish reason + cumulative usage on a trailing chunk,
    # after the content chunks.
    content_chunk = MagicMock(candidates=None, usage_metadata=None)
    final_chunk = MagicMock()
    final_chunk.candidates = [MagicMock()]
    final_chunk.candidates[0].finish_reason.name = 'STOP'
    final_chunk.usage_metadata = MagicMock(
        prompt_token_count=5,
        candidates_token_count=7,
        total_token_count=12,
        thoughts_token_count=0,
        cached_content_token_count=0,
    )

    async def fake_stream() -> AsyncIterator[MagicMock]:
        for chunk in (content_chunk, final_chunk):
            yield chunk

    async def generate_content_stream(**_kwargs: object) -> AsyncIterator[MagicMock]:
        return fake_stream()

    client = MagicMock()
    client.aio.models.generate_content_stream = generate_content_stream

    ctx = MagicMock()
    model = GeminiModel('gemini-2.0-flash', client=client)

    with (
        patch.object(GeminiModel, '_contents_from_response', new=AsyncMock(return_value=[])),
        patch(
            'genkit_google_genai.models.gemini.resolve_vertex_model_name',
            return_value='gemini-2.0-flash',
        ),
    ):
        resp = await model._streaming_generate([], None, ctx, 'gemini-2.0-flash', client=client)

    assert resp.finish_reason == FinishReason.STOP
    assert resp.usage is not None
    assert resp.usage.input_tokens == 5
    assert resp.usage.output_tokens == 7
    assert resp.usage.total_tokens == 12
