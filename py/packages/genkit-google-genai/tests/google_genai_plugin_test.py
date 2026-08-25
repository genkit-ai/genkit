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
from typing import cast, get_args, get_type_hints
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
    _list_genai_models,
    googleai_name,
    vertexai_name,
)
from genkit_google_genai.models.gemini import (
    GeminiImageConfigSchema,
    GeminiTtsConfigSchema,
    GemmaConfigSchema,
)
from genkit_google_genai.models.imagen import ImagenConfigSchema
from genkit_google_genai.models.veo import VeoConfigSchema, VeoModel

from genkit import ActionKind, GenkitError, Message, ModelRequest, Part, Role, TextPart
from genkit.model import Operation
from genkit.plugin_api import Action, to_json_schema


def _custom_options(action: Action) -> object:
    """Return the advertised config schema from an action's model metadata."""
    model_meta = cast('dict[str, object]', action.metadata['model'])
    return model_meta['customOptions']


def _request_config_type(action: Action) -> type:
    """Return the ModelRequest[T] config parameter from an action fn."""
    hints = get_type_hints(action._fn)  # noqa: SLF001
    request_type = hints['request']
    args = get_args(request_type)
    if args:
        return args[0]
    metadata = getattr(request_type, '__pydantic_generic_metadata__', None) or {}
    args = metadata.get('args') or ()
    assert args, f'expected ModelRequest[T], got {request_type!r}'
    return args[0]


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
    assert _custom_options(action) == to_json_schema(ImagenConfigSchema)


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_googleai_resolve_gemini_image_is_not_imagen(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Native Gemini image models must not route through Imagen."""
    mock_list_models.return_value = GenaiModels()

    plugin = GoogleAI(api_key='test-key')
    action = await plugin.resolve(ActionKind.MODEL, 'googleai/gemini-2.5-flash-image')

    assert action is not None
    assert _custom_options(action) == to_json_schema(GeminiImageConfigSchema)
    assert _request_config_type(action) is GeminiImageConfigSchema


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('model_name', 'config_type'),
    [
        ('googleai/gemini-2.0-flash', GeminiConfigSchema),
        ('googleai/gemini-2.5-flash-preview-tts', GeminiTtsConfigSchema),
        ('googleai/gemini-2.5-flash-image', GeminiImageConfigSchema),
        ('googleai/gemma-3-12b-it', GemmaConfigSchema),
        ('googleai/imagen-3.0-generate-002', ImagenConfigSchema),
    ],
)
async def test_googleai_resolve_types_family_config(
    mock_list_models: MagicMock,
    mock_client: MagicMock,
    model_name: str,
    config_type: type,
) -> None:
    """Each family action opts into ModelRequest[FamilyConfig]."""
    mock_list_models.return_value = GenaiModels()

    plugin = GoogleAI(api_key='test-key')
    action = await plugin.resolve(ActionKind.MODEL, model_name)

    assert action is not None
    assert _request_config_type(action) is config_type


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('model_name', 'config_type'),
    [
        ('vertexai/gemini-2.0-flash', GeminiConfigSchema),
        ('vertexai/gemini-2.5-flash-preview-tts', GeminiTtsConfigSchema),
        ('vertexai/gemini-2.5-flash-image', GeminiImageConfigSchema),
        ('vertexai/gemma-3-12b-it', GemmaConfigSchema),
        ('vertexai/imagen-3.0-generate-002', ImagenConfigSchema),
    ],
)
async def test_vertexai_resolve_types_family_config(
    mock_list_models: MagicMock,
    mock_client: MagicMock,
    model_name: str,
    config_type: type,
) -> None:
    """Vertex family actions opt into ModelRequest[FamilyConfig] the same way."""
    mock_list_models.return_value = GenaiModels()

    plugin = VertexAI(project='test-project')
    action = await plugin.resolve(ActionKind.MODEL, model_name)

    assert action is not None
    assert _request_config_type(action) is config_type


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_vertexai_gemma_action_accepts_temperature_3(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Gemma's schema accepts temperature=3.0; falling through to Gemini would reject it."""
    mock_list_models.return_value = GenaiModels()

    plugin = VertexAI(project='test-project')
    action = await plugin.resolve(ActionKind.MODEL, 'vertexai/gemma-3-12b-it')
    assert action is not None

    with patch('genkit_google_genai.google.GeminiModel.generate', new_callable=AsyncMock) as mock_generate:
        await action.run({
            'messages': [{'role': 'user', 'content': [{'text': 'hi'}]}],
            'config': {'temperature': 3.0},
        })
        called = mock_generate.await_args
        assert called is not None
        request = called.args[0]
        assert request.config.temperature == 3.0


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_veo_start_types_family_config(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Veo start is ModelRequest[VeoConfigSchema] so Action keeps aspectRatio / durationSeconds."""
    mock_list_models.return_value = GenaiModels()

    for plugin, name in (
        (GoogleAI(api_key='test-key'), 'googleai/veo-3.0-generate-001'),
        (VertexAI(project='test-project'), 'vertexai/veo-3.0-generate-001'),
    ):
        action = await plugin.resolve(ActionKind.BACKGROUND_MODEL, name)
        assert action is not None
        assert _request_config_type(action) is VeoConfigSchema

        with patch('genkit_google_genai.google.VeoModel.start', new_callable=AsyncMock) as mock_start:
            await action.run({
                'messages': [{'role': 'user', 'content': [{'text': 'a cat walking'}]}],
                'config': {'aspectRatio': '16:9', 'durationSeconds': 5},
            })
            called = mock_start.await_args
            assert called is not None
            request = called.args[0]
            assert isinstance(request.config, VeoConfigSchema)
            assert request.config.aspect_ratio == '16:9'
            assert request.config.duration_seconds == 5


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_veo_action_run_dumps_leftover_and_stamps(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Action.run camelCase + leftover reaches generate_videos; start stamps the action key."""
    mock_list_models.return_value = GenaiModels()
    op = MagicMock()
    op.name = 'operations/1'
    op.done = False
    mock_client.return_value.aio.models.generate_videos = AsyncMock(return_value=op)

    plugin = VertexAI(project='test-project')
    action = await plugin.resolve(ActionKind.BACKGROUND_MODEL, 'vertexai/veo-3.0-generate-001')
    assert action is not None

    started = await action.run({
        'messages': [{'role': 'user', 'content': [{'text': 'a cat walking'}]}],
        'config': {'aspectRatio': '16:9', 'durationSeconds': 5, 'fooBar': 1},
    })

    called = mock_client.return_value.aio.models.generate_videos.await_args
    assert called is not None
    cfg = called.kwargs['config']
    assert cfg.aspect_ratio == '16:9'
    assert cfg.duration_seconds == 5
    assert cfg.http_options is not None
    assert cfg.http_options.extra_body == {'parameters': {'fooBar': 1}}
    assert started.response.action == '/background-model/vertexai/veo-3.0-generate-001'


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_veo_action_run_rejects_bad_duration(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Action rejects durationSeconds='nope' before generate_videos."""
    mock_list_models.return_value = GenaiModels()

    plugin = VertexAI(project='test-project')
    action = await plugin.resolve(ActionKind.BACKGROUND_MODEL, 'vertexai/veo-3.0-generate-001')
    assert action is not None

    with pytest.raises(GenkitError) as exc_info:
        await action.run({
            'messages': [{'role': 'user', 'content': [{'text': 'a clip'}]}],
            'config': {'durationSeconds': 'nope'},
        })

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    mock_client.return_value.aio.models.generate_videos.assert_not_called()


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_veo_check_is_typed(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Check is Operation in, Operation out — no config to coerce."""
    mock_list_models.return_value = GenaiModels()

    plugin = VertexAI(project='test-project')
    action = await plugin.resolve(ActionKind.CHECK_OPERATION, 'vertexai/veo-3.0-generate-001/check')
    assert action is not None
    hints = get_type_hints(action._fn)  # noqa: SLF001
    assert hints['op'] is Operation
    assert hints['return'] is Operation


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
@pytest.mark.parametrize(
    'model_id',
    [
        'virtual-try-on-001',
        'imagegeneration@006',
        'imagetext@001',
        'lyria-002',
        'deep-research-pro-preview',
        'gemini-embedding-001',
        'models/deep-research-pro-preview',
        'publishers/google/models/deep-research-pro-preview',
    ],
)
async def test_vertexai_unroutable_ids_fail_closed(
    mock_list_models: MagicMock, mock_client: MagicMock, model_id: str
) -> None:
    """Ids with no generate path here resolve to nothing, not to Gemini."""
    mock_list_models.return_value = GenaiModels()

    plugin = VertexAI(project='test-project')
    action = await plugin.resolve(ActionKind.MODEL, f'vertexai/{model_id}')

    assert action is None


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
@pytest.mark.parametrize(
    'model_id',
    [
        'virtual-try-on-001',
        'imagegeneration@006',
        'imagetext@001',
        'lyria-002',
        'deep-research-pro-preview',
        'gemini-embedding-001',
        'models/deep-research-pro-preview',
        'publishers/google/models/deep-research-pro-preview',
    ],
)
async def test_googleai_unroutable_ids_fail_closed(
    mock_list_models: MagicMock, mock_client: MagicMock, model_id: str
) -> None:
    """Ids with no generate path here resolve to nothing, not to Gemini."""
    mock_list_models.return_value = GenaiModels()

    plugin = GoogleAI(api_key='test-key')
    action = await plugin.resolve(ActionKind.MODEL, f'googleai/{model_id}')

    assert action is None


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_googleai_resolve_veo_as_model_returns_none(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Veo is background-only; resolving it as MODEL must not build a Gemini action."""
    mock_list_models.return_value = GenaiModels()

    plugin = GoogleAI(api_key='test-key')
    action = await plugin.resolve(ActionKind.MODEL, 'googleai/veo-3.0-generate-001')

    assert action is None


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_vertexai_resolve_veo_as_model_returns_none(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Veo is background-only; resolving it as MODEL must not build a Gemini action."""
    mock_list_models.return_value = GenaiModels()

    plugin = VertexAI(project='test-project')
    action = await plugin.resolve(ActionKind.MODEL, 'vertexai/veo-3.0-generate-001')

    assert action is None


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_vertexai_resolve_veo_background_model(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Vertex Veo resolves as a background model with a check action."""
    mock_list_models.return_value = GenaiModels()

    plugin = VertexAI(project='test-project')
    start = await plugin.resolve(ActionKind.BACKGROUND_MODEL, 'vertexai/veo-3.0-generate-001')
    check = await plugin.resolve(ActionKind.CHECK_OPERATION, 'vertexai/veo-3.0-generate-001/check')

    assert start is not None
    assert start.kind == ActionKind.BACKGROUND_MODEL
    assert start.name == 'vertexai/veo-3.0-generate-001'
    model_meta = cast('dict[str, object]', start.metadata['model'])
    supports = cast('dict[str, object]', model_meta['supports'])
    assert supports['longRunning'] is True
    assert check is not None
    assert check.kind == ActionKind.CHECK_OPERATION
    assert check.name == 'vertexai/veo-3.0-generate-001/check'


@patch('genkit_google_genai.google.create_vertex_evaluators')
@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_vertexai_init_registers_veo_as_background(
    mock_list_models: MagicMock, mock_client: MagicMock, mock_evaluators: MagicMock
) -> None:
    """Vertex init registers Veo start/check, never a blocking MODEL action."""
    models = GenaiModels()
    models.veo = ['veo-3.0-generate-001']
    mock_list_models.return_value = models
    mock_evaluators.return_value = []

    plugin = VertexAI(project='test-project')
    actions = await plugin.init()

    veo_actions = [a for a in actions if 'veo' in a.name]
    assert {a.kind for a in veo_actions} == {ActionKind.BACKGROUND_MODEL, ActionKind.CHECK_OPERATION}
    assert not any(a.kind == ActionKind.MODEL for a in veo_actions)


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_list_actions_advertises_veo_as_background(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Both plugins list Veo with the kind that resolve() actually serves."""
    models = GenaiModels()
    models.veo = ['veo-3.0-generate-001']
    mock_list_models.return_value = models

    googleai_actions = await GoogleAI(api_key='test-key').list_actions()
    vertexai_actions = await VertexAI(project='test-project').list_actions()

    for actions, plugin_name in ((googleai_actions, 'googleai'), (vertexai_actions, 'vertexai')):
        veo_entries = [a for a in actions if 'veo' in a.name]
        assert len(veo_entries) == 1
        assert veo_entries[0].name == f'{plugin_name}/veo-3.0-generate-001'
        assert veo_entries[0].action_type == ActionKind.BACKGROUND_MODEL


def test_list_genai_models_vertex_skips_substring_veo_and_retired_image() -> None:
    """Discovery buckets on the ``veo-`` prefix, not a ``veo`` substring."""

    def _model(name: str) -> MagicMock:
        item = MagicMock()
        item.name = name
        item.supported_actions = None
        item.description = ''
        return item

    client = MagicMock()
    client.models.list.return_value = [
        _model('publishers/google/models/gemini-2.5-flash'),
        _model('publishers/google/models/veo-3.0-generate-001'),
        _model('publishers/google/models/braveo-lab'),
        _model('publishers/google/models/imagegeneration@006'),
        _model('publishers/google/models/virtual-try-on-001'),
        _model('publishers/google/models/imagetext@001'),
    ]
    catalog = _list_genai_models(client, is_vertex=True)
    assert catalog.veo == ['veo-3.0-generate-001']
    assert catalog.imagen == []
    assert 'imagetext@001' not in catalog.gemini
    assert 'braveo-lab' not in catalog.gemini


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_veo_start_stamps_background_action_key(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Start and check stamp ``/background-model/{name}`` so a later check can resolve."""
    mock_list_models.return_value = GenaiModels()
    plugin = VertexAI(project='test-project')
    start = await plugin.resolve(ActionKind.BACKGROUND_MODEL, 'vertexai/veo-3.0-generate-001')
    check = await plugin.resolve(ActionKind.CHECK_OPERATION, 'vertexai/veo-3.0-generate-001/check')
    assert start is not None
    assert check is not None

    request = ModelRequest(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='a clip'))])])
    with patch.object(VeoModel, 'start', new=AsyncMock(return_value=Operation(id='ops/1'))):
        started = await start.run(request)
    assert started.response.action == '/background-model/vertexai/veo-3.0-generate-001'

    with patch.object(VeoModel, 'check', new=AsyncMock(return_value=Operation(id='ops/1'))):
        checked = await check.run(Operation(id='ops/1'))
    assert checked.response.action == '/background-model/vertexai/veo-3.0-generate-001'


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
