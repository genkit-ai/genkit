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

"""Unit-Tests for GoogleAI & VertexAI plugin."""

import sys  # noqa
import os

import unittest
from unittest.mock import MagicMock, patch, ANY

from google.auth.credentials import Credentials
from dataclasses import dataclass

from google.genai.types import HttpOptions

import pytest
from genkit.ai import Genkit, GENKIT_CLIENT_HEADER
from genkit.core.registry import ActionKind
from genkit.plugins.google_genai import GoogleAI, VertexAI
from genkit.plugins.google_genai.google import googleai_name, vertexai_name
from genkit.plugins.google_genai.google import _inject_attribution_headers
from genkit.plugins.google_genai.models.gemini import (
    DEFAULT_SUPPORTS_MODEL,
    GeminiModel,
    SUPPORTED_MODELS,
    GeminiConfigSchema,
)
from genkit.plugins.google_genai.models.imagen import (
    SUPPORTED_MODELS as IMAGE_SUPPORTED_MODELS,
    DEFAULT_IMAGE_SUPPORT,
)
from genkit.types import (
    GenerateRequest,
    Message,
    ModelInfo,
    Part,
    Role,
    TextPart,
)
from google import genai


@pytest.fixture
@patch('google.genai.client.Client')
def googleai_plugin_instance(client: MagicMock) -> GoogleAI:
    """GoogleAI fixture."""
    api_key = 'test_api_key'
    return GoogleAI(api_key=api_key)


class TestGoogleAIInit(unittest.TestCase):
    """Test cases for __init__ plugin."""

    @patch('google.genai.client.Client')
    def test_init_with_api_key(self, mock_genai_client: MagicMock) -> None:
        """Test using api_key parameter."""
        api_key = 'test_api_key'
        plugin = GoogleAI(api_key=api_key)
        mock_genai_client.assert_called_once_with(
            vertexai=False,
            api_key=api_key,
            credentials=None,
            debug_config=None,
            http_options=_inject_attribution_headers(),
        )
        self.assertIsInstance(plugin, GoogleAI)
        self.assertFalse(plugin._vertexai)
        self.assertIsInstance(plugin._client, MagicMock)

    @patch('google.genai.client.Client')
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'env_api_key'})
    def test_init_from_env_var(self, mock_genai_client: MagicMock) -> None:
        """Test using env var for api_key."""
        plugin = GoogleAI()
        mock_genai_client.assert_called_once_with(
            vertexai=False,
            api_key='env_api_key',
            credentials=None,
            debug_config=None,
            http_options=_inject_attribution_headers(),
        )
        self.assertIsInstance(plugin, GoogleAI)
        self.assertFalse(plugin._vertexai)
        self.assertIsInstance(plugin._client, MagicMock)

    @patch('google.genai.client.Client')
    def test_init_with_credentials(self, mock_genai_client: MagicMock) -> None:
        """Test using credentials parameter."""
        mock_credentials = MagicMock(spec=Credentials)
        plugin = GoogleAI(credentials=mock_credentials)
        mock_genai_client.assert_called_once_with(
            vertexai=False,
            api_key=ANY,
            credentials=mock_credentials,
            debug_config=None,
            http_options=_inject_attribution_headers(),
        )
        self.assertIsInstance(plugin, GoogleAI)
        self.assertFalse(plugin._vertexai)
        self.assertIsInstance(plugin._client, MagicMock)

    def test_init_raises_value_error_no_api_key(self) -> None:
        """Test using credentials parameter."""
        with (
            patch.dict(os.environ, {'GEMINI_API_KEY': ''}, clear=True),
            self.assertRaisesRegex(
                ValueError,
                'Gemini api key should be passed in plugin params or as a GEMINI_API_KEY environment variable',
            ),
        ):
            GoogleAI()


@patch('google.genai.client.Client')
@pytest.mark.asyncio
async def test_googleai_initialize(mock_client_cls: MagicMock) -> None:
    """Unit tests for GoogleAI.init method."""
    mock_client = mock_client_cls.return_value

    m1 = MagicMock()
    m1.name = 'models/gemini-pro'
    m1.supported_actions = ['generateContent']
    m1.description = ' Gemini Pro '

    m2 = MagicMock()
    m2.name = 'models/gemini-embedding-001'
    m2.supported_actions = ['embedContent']
    m2.description = ' Embedding '

    mock_client.models.list.return_value = [m1, m2]

    api_key = 'test_api_key'
    plugin = GoogleAI(api_key=api_key)
    # Ensure usage of mock
    plugin._client = mock_client

    await plugin.init()
    result = await plugin.list_actions()

    # init returns known models and embedders
    assert len(result) > 0, 'Should initialize with known models and embedders'
    assert all(hasattr(action, 'kind') for action in result), 'All actions should have a kind'
    assert all(hasattr(action, 'name') for action in result), 'All actions should have a name'
    assert all(action.name.startswith('googleai/') for action in result), (
        "All actions should be namespaced with 'googleai/'"
    )

    # Verify we have both models and embedders
    model_actions = [a for a in result if a.kind == ActionKind.MODEL]
    embedder_actions = [a for a in result if a.kind == ActionKind.EMBEDDER]
    assert len(model_actions) > 0, 'Should have at least one model'
    assert len(embedder_actions) > 0, 'Should have at least one embedder'


@patch('genkit.plugins.google_genai.GoogleAI._resolve_model')
@pytest.mark.asyncio
async def test_googleai_resolve_action_model(
    mock_resolve_action: MagicMock, googleai_plugin_instance: GoogleAI
) -> None:
    """Test resolve action for model."""
    plugin = googleai_plugin_instance

    await plugin.resolve(action_type=ActionKind.MODEL, name='lazaro-model')
    mock_resolve_action.assert_called_once_with('lazaro-model')


@patch('genkit.plugins.google_genai.GoogleAI._resolve_embedder')
@pytest.mark.asyncio
async def test_googleai_resolve_action_embedder(
    mock_resolve_action: MagicMock, googleai_plugin_instance: GoogleAI
) -> None:
    """Test resolve action for embedder."""
    plugin = googleai_plugin_instance

    await plugin.resolve(action_type=ActionKind.EMBEDDER, name='lazaro-model')
    mock_resolve_action.assert_called_once_with('lazaro-model')


@patch('genkit.plugins.google_genai.models.gemini.google_model_info')
@pytest.mark.parametrize(
    'model_name, expected_model_name, key',
    [
        (
            'gemini-pro-deluxe-max',
            'googleai/gemini-pro-deluxe-max',
            'gemini-pro-deluxe-max',
        ),
        (
            'googleai/gemini-pro-deluxe-max',
            'googleai/gemini-pro-deluxe-max',
            'gemini-pro-deluxe-max',
        ),
    ],
)
def test_googleai__resolve_model(
    mock_google_model_info: MagicMock,
    model_name: str,
    expected_model_name: str,
    key: str,
    googleai_plugin_instance: GoogleAI,
) -> None:
    """Tests for GoogleAI._resolve_model method."""
    plugin = googleai_plugin_instance

    mock_google_model_info.return_value = ModelInfo(
        label=f'Google AI - {model_name}',
        supports=DEFAULT_SUPPORTS_MODEL,
    )

    action = plugin._resolve_model(name=expected_model_name)

    assert action is not None
    assert action.kind == ActionKind.MODEL
    assert action.name == expected_model_name
    assert key in SUPPORTED_MODELS


@pytest.mark.parametrize(
    'model_name, expected_model_name, clean_name',
    [
        ('gemini-pro-deluxe-max', 'googleai/gemini-pro-deluxe-max', 'gemini-pro-deluxe-max'),
        ('googleai/gemini-pro-deluxe-max', 'googleai/gemini-pro-deluxe-max', 'gemini-pro-deluxe-max'),
    ],
)
def test_googleai__resolve_embedder(
    model_name: str,
    expected_model_name: str,
    clean_name: str,
    googleai_plugin_instance: GoogleAI,
) -> None:
    """Tests for GoogleAI._resolve_embedder method."""
    plugin = googleai_plugin_instance

    action = plugin._resolve_embedder(name=expected_model_name)

    assert action is not None
    assert action.kind == ActionKind.EMBEDDER
    assert action.name == expected_model_name


@pytest.mark.asyncio
async def test_googleai_list_actions(googleai_plugin_instance: GoogleAI) -> None:
    """Unit test for list actions."""

    @dataclass
    class MockModel:
        supported_actions: list[str]
        name: str
        description: str = ''

    models_return_value = [
        MockModel(supported_actions=['generateContent'], name='models/gemini-pro'),
        MockModel(supported_actions=['embedContent'], name='models/gemini-embedding-001'),
        MockModel(supported_actions=['generateContent'], name='models/gemini-2.0-flash-tts'),  # TTS
    ]

    mock_client = MagicMock()
    mock_client.models.list.return_value = models_return_value
    googleai_plugin_instance._client = mock_client

    result = await googleai_plugin_instance.list_actions()

    # Check Gemini Pro
    action1 = next(a for a in result if a.name == googleai_name('gemini-pro'))
    assert action1 is not None

    # Check Embedder
    action2 = next(a for a in result if a.name == googleai_name('gemini-embedding-001'))
    assert action2 is not None
    assert action2.kind == ActionKind.EMBEDDER

    # Check TTS
    action3 = next(a for a in result if a.name == googleai_name('gemini-2.0-flash-tts'))
    assert action3 is not None
    # from genkit.plugins.google_genai.models.gemini import GeminiTtsConfigSchema, GeminiConfigSchema
    # assert action3.config_schema == GeminiTtsConfigSchema
    # assert action1.config_schema == GeminiConfigSchema


@pytest.mark.asyncio
async def test_googleai_list_known_models(googleai_plugin_instance: GoogleAI) -> None:
    """Unit test for list known models."""

    @dataclass
    class MockModel:
        supported_actions: list[str]
        name: str
        description: str = ''

    models_return_value = [
        MockModel(supported_actions=['generateContent'], name='models/gemini-pro'),
        MockModel(supported_actions=['embedContent'], name='models/gemini-embedding-001'),
        MockModel(supported_actions=['generateContent'], name='models/gemini-2.0-flash-tts'),  # TTS
    ]

    mock_client = MagicMock()
    mock_client.models.list.return_value = models_return_value
    googleai_plugin_instance._client = mock_client

    result = googleai_plugin_instance._list_known_models()

    # Check Gemini Pro
    action1 = next(a for a in result if a.name == googleai_name('gemini-pro'))
    assert action1 is not None

    # Check TTS
    action3 = next(a for a in result if a.name == googleai_name('gemini-2.0-flash-tts'))
    assert action3 is not None


@pytest.mark.asyncio
async def test_googleai_list_known_veo_models(googleai_plugin_instance: GoogleAI) -> None:
    """Unit test for list known veo models."""

    @dataclass
    class MockModel:
        supported_actions: list[str]
        name: str
        description: str = ''

    models_return_value = [
        MockModel(supported_actions=['generateVideos'], name='models/veo-2.0-generate-001'),
    ]

    mock_client = MagicMock()
    mock_client.models.list.return_value = models_return_value
    googleai_plugin_instance._client = mock_client

    result = googleai_plugin_instance._list_known_veo_models()

    # Check Veo
    action1 = next(a for a in result if a.name == googleai_name('veo-2.0-generate-001'))
    assert action1 is not None


@pytest.mark.asyncio
async def test_googleai_list_known_embedders(googleai_plugin_instance: GoogleAI) -> None:
    """Unit test for list known embedders."""

    @dataclass
    class MockModel:
        supported_actions: list[str]
        name: str
        description: str = ''

    models_return_value = [
        MockModel(supported_actions=['embedContent'], name='models/gemini-embedding-001'),
    ]

    mock_client = MagicMock()
    mock_client.models.list.return_value = models_return_value
    googleai_plugin_instance._client = mock_client

    result = googleai_plugin_instance._list_known_embedders()

    # Check Embedder
    action1 = next(a for a in result if a.name == googleai_name('gemini-embedding-001'))
    assert action1 is not None


@pytest.mark.parametrize(
    'input_options, expected_headers',
    [
        (
            None,
            {
                'x-goog-api-client': GENKIT_CLIENT_HEADER,
                'user-agent': GENKIT_CLIENT_HEADER,
            },
        ),
        (
            {},
            {
                'x-goog-api-client': GENKIT_CLIENT_HEADER,
                'user-agent': GENKIT_CLIENT_HEADER,
            },
        ),
        (
            {'headers': {'existing-header': 'value'}},
            {
                'existing-header': 'value',
                'x-goog-api-client': GENKIT_CLIENT_HEADER,
                'user-agent': GENKIT_CLIENT_HEADER,
            },
        ),
        (
            {'headers': {'x-goog-api-client': 'initial-client'}},
            {
                'x-goog-api-client': f'initial-client {GENKIT_CLIENT_HEADER}',
                'user-agent': GENKIT_CLIENT_HEADER,
            },
        ),
        (
            {'headers': {'user-agent': 'initial-agent'}},
            {
                'x-goog-api-client': GENKIT_CLIENT_HEADER,
                'user-agent': f'initial-agent {GENKIT_CLIENT_HEADER}',
            },
        ),
        (
            {'headers': {'x-goog-api-client': 'old', 'user-agent': 'old-ua'}},
            {
                'x-goog-api-client': f'old {GENKIT_CLIENT_HEADER}',
                'user-agent': f'old-ua {GENKIT_CLIENT_HEADER}',
            },
        ),
        (
            HttpOptions(),
            {
                'x-goog-api-client': GENKIT_CLIENT_HEADER,
                'user-agent': GENKIT_CLIENT_HEADER,
            },
        ),
        (
            HttpOptions(headers={'other': 'value'}),
            {
                'other': 'value',
                'x-goog-api-client': GENKIT_CLIENT_HEADER,
                'user-agent': GENKIT_CLIENT_HEADER,
            },
        ),
        (
            HttpOptions(headers={'x-goog-api-client': 'initial'}),
            {
                'x-goog-api-client': f'initial {GENKIT_CLIENT_HEADER}',
                'user-agent': GENKIT_CLIENT_HEADER,
            },
        ),
        (
            HttpOptions(headers={'user-agent': 'initial-u'}),
            {
                'x-goog-api-client': GENKIT_CLIENT_HEADER,
                'user-agent': f'initial-u {GENKIT_CLIENT_HEADER}',
            },
        ),
        (
            HttpOptions(headers={'x-goog-api-client': 'pre', 'user-agent': 'pre-u'}),
            {
                'x-goog-api-client': f'pre {GENKIT_CLIENT_HEADER}',
                'user-agent': f'pre-u {GENKIT_CLIENT_HEADER}',
            },
        ),
        (
            HttpOptions(timeout=10),
            {
                'x-goog-api-client': GENKIT_CLIENT_HEADER,
                'user-agent': GENKIT_CLIENT_HEADER,
            },
        ),
        (
            {'timeout': 5},
            {
                'x-goog-api-client': GENKIT_CLIENT_HEADER,
                'user-agent': GENKIT_CLIENT_HEADER,
            },
        ),
        (
            {'headers': {'one': '1'}},
            {
                'one': '1',
                'x-goog-api-client': GENKIT_CLIENT_HEADER,
                'user-agent': GENKIT_CLIENT_HEADER,
            },
        ),
    ],
)
def test_inject_attribution_headers(
    input_options: HttpOptions | dict[str, object] | None, expected_headers: dict[str, str]
) -> None:
    """Tests the _inject_attribution_headers function with various inputs."""
    result = _inject_attribution_headers(input_options)  # type: ignore
    assert isinstance(result, HttpOptions)
    assert result.headers == expected_headers


class TestVertexAIInit(unittest.TestCase):
    """Test cases for VertexAI.__init__ plugin."""

    @patch('google.genai.client.Client')
    @patch.dict(os.environ, {'GCLOUD_PROJECT': 'project'})
    def test_init_with_api_key(self, mock_genai_client: MagicMock) -> None:
        """Test using api_key parameter."""
        api_key = 'test_api_key'
        plugin = VertexAI(api_key=api_key)
        mock_genai_client.assert_called_once_with(
            vertexai=True,
            api_key=api_key,
            credentials=None,
            project='project',
            location='us-central1',
            debug_config=None,
            http_options=_inject_attribution_headers(),
        )
        self.assertIsInstance(plugin, VertexAI)
        self.assertTrue(plugin._vertexai)
        self.assertIsInstance(plugin._client, MagicMock)

    @patch('google.genai.client.Client')
    @patch.dict(os.environ, {'GCLOUD_PROJECT': 'project'})
    def test_init_with_credentials(self, mock_genai_client: MagicMock) -> None:
        """Test using credentials parameter."""
        mock_credentials = MagicMock(spec=Credentials)
        plugin = VertexAI(credentials=mock_credentials)
        mock_genai_client.assert_called_once_with(
            vertexai=True,
            api_key=None,
            credentials=mock_credentials,
            project='project',
            location='us-central1',
            debug_config=None,
            http_options=_inject_attribution_headers(),
        )
        self.assertIsInstance(plugin, VertexAI)
        self.assertTrue(plugin._vertexai)
        self.assertIsInstance(plugin._client, MagicMock)

    @patch('google.genai.client.Client')
    def test_init_with_all(self, mock_genai_client: MagicMock) -> None:
        """Test using credentials parameter."""
        mock_credentials = MagicMock(spec=Credentials)
        api_key = 'test_api_key'
        plugin = VertexAI(
            credentials=mock_credentials,
            api_key=api_key,
            project='project',
            location='location',
        )
        mock_genai_client.assert_called_once_with(
            vertexai=True,
            api_key=api_key,
            credentials=mock_credentials,
            project='project',
            location='location',
            debug_config=None,
            http_options=_inject_attribution_headers(),
        )
        self.assertIsInstance(plugin, VertexAI)
        self.assertTrue(plugin._vertexai)
        self.assertIsInstance(plugin._client, MagicMock)


@pytest.fixture
@patch('google.genai.client.Client')
def vertexai_plugin_instance(client: MagicMock) -> VertexAI:
    """VertexAI fixture."""
    return VertexAI(project='test-project', location='us-central1')


@pytest.mark.asyncio
async def test_vertexai_initialize(vertexai_plugin_instance: VertexAI) -> None:
    """Unit tests for VertexAI.init method."""
    plugin = vertexai_plugin_instance

    # Configure mock client to return models
    m1 = MagicMock()
    m1.name = 'publishers/google/models/gemini-1.5-flash'
    m1.supported_actions = ['generateContent']

    m2 = MagicMock()
    m2.name = 'publishers/google/models/gemini-embedding-001'
    m2.supported_actions = ['embedContent']

    plugin._client.models.list.return_value = [m1, m2]  # type: ignore

    await plugin.init()

    # init returns known models and embedders in internal registry, but list_actions returns them list
    result = await plugin.list_actions()

    assert len(result) > 0, 'Should initialize with known models and embedders'
    assert all(hasattr(action, 'kind') for action in result), 'All actions should have a kind'

    # ... (rest of test unchanged)

    assert all(hasattr(action, 'name') for action in result), 'All actions should have a name'
    assert all(action.name.startswith('vertexai/') for action in result), (
        "All actions should be namespaced with 'vertexai/'"
    )

    # Verify we have both models and embedders
    model_actions = [a for a in result if a.kind == ActionKind.MODEL]
    embedder_actions = [a for a in result if a.kind == ActionKind.EMBEDDER]
    assert len(model_actions) > 0, 'Should have at least one model'
    assert len(embedder_actions) > 0, 'Should have at least one embedder'


@patch('genkit.plugins.google_genai.VertexAI._resolve_model')
@pytest.mark.asyncio
async def test_vertexai_resolve_action_model(
    mock_resolve_action: MagicMock, vertexai_plugin_instance: VertexAI
) -> None:
    """Test resolve action for model."""
    plugin = vertexai_plugin_instance

    await plugin.resolve(action_type=ActionKind.MODEL, name='lazaro-model')
    mock_resolve_action.assert_called_once_with('lazaro-model')


@patch('genkit.plugins.google_genai.VertexAI._resolve_embedder')
@pytest.mark.asyncio
async def test_vertexai_resolve_action_embedder(
    mock_resolve_action: MagicMock, vertexai_plugin_instance: VertexAI
) -> None:
    """Test resolve action for embedder."""
    plugin = vertexai_plugin_instance

    await plugin.resolve(action_type=ActionKind.EMBEDDER, name='lazaro-model')
    mock_resolve_action.assert_called_once_with('lazaro-model')


@patch(
    'genkit.plugins.google_genai.models.gemini.google_model_info',
    new_callable=MagicMock,
)
@patch(
    'genkit.plugins.google_genai.models.imagen.vertexai_image_model_info',
    new_callable=MagicMock,
)
@pytest.mark.parametrize(
    'model_name, expected_model_name, key, image',
    [
        (
            'gemini-pro-deluxe-max',
            'vertexai/gemini-pro-deluxe-max',
            'gemini-pro-deluxe-max',
            False,
        ),
        (
            'vertexai/gemini-pro-deluxe-max',
            'vertexai/gemini-pro-deluxe-max',
            'gemini-pro-deluxe-max',
            False,
        ),
        (
            'vertexai/image-gemini-pro-deluxe-max',
            'vertexai/image-gemini-pro-deluxe-max',
            'image-gemini-pro-deluxe-max',
            True,
        ),
        (
            'image-gemini-pro-deluxe-max',
            'vertexai/image-gemini-pro-deluxe-max',
            'image-gemini-pro-deluxe-max',
            True,
        ),
        (
            'gemini-pro-deluxe-max-image',
            'vertexai/gemini-pro-deluxe-max-image',
            'gemini-pro-deluxe-max-image',
            False,
        ),
    ],
)
def test_vertexai__resolve_model(
    mock_google_model_info: MagicMock,
    mock_vertexai_image_model_info: MagicMock,
    model_name: str,
    expected_model_name: str,
    key: str,
    image: bool,
    vertexai_plugin_instance: VertexAI,
) -> None:
    """Tests for VertexAI._resolve_model method."""
    plugin = vertexai_plugin_instance
    MagicMock(spec=Genkit)

    mock_google_model_info.return_value = ModelInfo(
        label=f'Google AI - {model_name}',
        supports=DEFAULT_SUPPORTS_MODEL,
    )

    mock_vertexai_image_model_info.return_value = ModelInfo(
        label=f'Vertex AI - {model_name}',
        supports=DEFAULT_IMAGE_SUPPORT,
    )

    action = plugin._resolve_model(name=expected_model_name)

    assert action is not None
    assert action.kind == ActionKind.MODEL
    assert action.name == expected_model_name

    if image:
        assert key in IMAGE_SUPPORTED_MODELS
    else:
        assert key in SUPPORTED_MODELS


@pytest.mark.parametrize(
    'model_name, expected_model_name, clean_name',
    [
        (
            'gemini-pro-deluxe-max',
            'vertexai/gemini-pro-deluxe-max',
            'gemini-pro-deluxe-max',
        ),
        (
            'vertexai/gemini-pro-deluxe-max',
            'vertexai/gemini-pro-deluxe-max',
            'gemini-pro-deluxe-max',
        ),
    ],
)
def test_vertexai__resolve_embedder(
    model_name: str,
    expected_model_name: str,
    clean_name: str,
    vertexai_plugin_instance: VertexAI,
) -> None:
    """Tests for VertexAI._resolve_embedder method."""
    plugin = vertexai_plugin_instance

    action = plugin._resolve_embedder(name=expected_model_name)

    assert action is not None
    assert action.kind == ActionKind.EMBEDDER
    assert action.name == expected_model_name


@pytest.mark.asyncio
async def test_vertexai_list_actions(vertexai_plugin_instance: VertexAI) -> None:
    """Unit test for list actions."""

    @dataclass
    class MockModel:
        name: str
        description: str = ''

    [
        MockModel(name='publishers/google/models/gemini-1.5-flash'),
        MockModel(name='publishers/google/models/gemini-embedding-001'),
        MockModel(name='publishers/google/models/imagen-3.0-generate-001'),
        MockModel(name='publishers/google/models/veo-2.0-generate-001'),
    ]

    mock_client = MagicMock()
    # Create sophisticated mocks that have supported_actions
    m1 = MagicMock()
    m1.name = 'publishers/google/models/gemini-1.5-flash'
    m1.supported_actions = ['generateContent']
    m1.description = 'Gemini model'

    m2 = MagicMock()
    m2.name = 'publishers/google/models/gemini-embedding-001'
    m2.supported_actions = ['embedContent']
    m2.description = 'Embedder'

    m3 = MagicMock()
    m3.name = 'publishers/google/models/imagen-3.0-generate-001'
    m3.supported_actions = ['predict']  # Imagen uses predict
    m3.description = 'Imagen'

    m4 = MagicMock()
    m4.name = 'publishers/google/models/veo-2.0-generate-001'
    m4.supported_actions = ['generateVideos']  # Veo uses generateVideos
    m4.description = 'Veo'

    mock_client.models.list.return_value = [m1, m2, m3, m4]
    vertexai_plugin_instance._client = mock_client

    result = await vertexai_plugin_instance.list_actions()

    # Verify Gemini
    action1 = next(a for a in result if a.name == vertexai_name('gemini-1.5-flash'))
    assert action1 is not None

    # Verify Embedder
    action2 = next(a for a in result if a.name == vertexai_name('gemini-embedding-001'))
    assert action2 is not None

    # Verify Imagen
    action3 = next(a for a in result if a.name == vertexai_name('imagen-3.0-generate-001'))
    assert action3 is not None
    assert action3.kind == ActionKind.MODEL

    # Verify Veo
    action4 = next(a for a in result if a.name == vertexai_name('veo-2.0-generate-001'))
    assert action4 is not None
    # from genkit.plugins.google_genai.models.veo import VeoConfigSchema
    # assert action4.config_schema == VeoConfigSchema


@pytest.mark.asyncio
async def test_googleai_resolve_background_model(googleai_plugin_instance: GoogleAI) -> None:
    """Test resolve action for background model."""
    plugin = googleai_plugin_instance

    action = await plugin.resolve(action_type=ActionKind.BACKGROUND_MODEL, name=googleai_name('veo-2.0-generate-001'))
    assert action is not None
    assert action.kind == ActionKind.BACKGROUND_MODEL
    assert action.name == googleai_name('veo-2.0-generate-001')


@pytest.mark.asyncio
async def test_googleai_resolve_check_operation(googleai_plugin_instance: GoogleAI) -> None:
    """Test resolve action for check operation."""
    plugin = googleai_plugin_instance

    action = await plugin.resolve(
        action_type=ActionKind.CHECK_OPERATION, name=googleai_name('veo-2.0-generate-001/check')
    )
    assert action is not None
    assert action.kind == ActionKind.CHECK_OPERATION
    assert action.name == googleai_name('veo-2.0-generate-001/check')


@pytest.mark.asyncio
async def test_vertexai_list_known_models(vertexai_plugin_instance: VertexAI) -> None:
    """Unit test for list known models."""

    @dataclass
    class MockModel:
        name: str
        description: str = ''

    [
        MockModel(name='publishers/google/models/gemini-1.5-flash'),
        MockModel(name='publishers/google/models/gemini-embedding-001'),
        MockModel(name='publishers/google/models/imagen-3.0-generate-001'),
        MockModel(name='publishers/google/models/veo-2.0-generate-001'),
    ]

    mock_client = MagicMock()
    # Create sophisticated mocks that have supported_actions
    m1 = MagicMock()
    m1.name = 'publishers/google/models/gemini-1.5-flash'
    m1.supported_actions = ['generateContent']
    m1.description = 'Gemini model'

    m2 = MagicMock()
    m2.name = 'publishers/google/models/gemini-embedding-001'
    m2.supported_actions = ['embedContent']
    m2.description = 'Embedder'

    m3 = MagicMock()
    m3.name = 'publishers/google/models/imagen-3.0-generate-001'
    m3.supported_actions = ['predict']
    m3.description = 'Imagen'

    m4 = MagicMock()
    m4.name = 'publishers/google/models/veo-2.0-generate-001'
    m4.supported_actions = ['generateVideos']
    m4.description = 'Veo'

    mock_client.models.list.return_value = [m1, m2, m3, m4]
    vertexai_plugin_instance._client = mock_client

    result = vertexai_plugin_instance._list_known_models()

    # Verify Gemini
    action1 = next(a for a in result if a.name == vertexai_name('gemini-1.5-flash'))
    assert action1 is not None

    # Verify Imagen
    action3 = next(a for a in result if a.name == vertexai_name('imagen-3.0-generate-001'))
    assert action3 is not None

    # Verify Veo
    action4 = next(a for a in result if a.name == vertexai_name('veo-2.0-generate-001'))
    assert action4 is not None


@pytest.mark.asyncio
async def test_vertexai_list_known_embedders(vertexai_plugin_instance: VertexAI) -> None:
    """Unit test for list known embedders."""

    @dataclass
    class MockModel:
        name: str
        description: str = ''

    [
        MockModel(name='publishers/google/models/gemini-1.5-flash'),
        MockModel(name='publishers/google/models/gemini-embedding-001'),
        MockModel(name='publishers/google/models/imagen-3.0-generate-001'),
        MockModel(name='publishers/google/models/veo-2.0-generate-001'),
    ]

    mock_client = MagicMock()
    # Create sophisticated mocks that have supported_actions
    m1 = MagicMock()
    m1.name = 'publishers/google/models/gemini-1.5-flash'
    m1.supported_actions = ['generateContent']
    m1.description = 'Gemini model'

    m2 = MagicMock()
    m2.name = 'publishers/google/models/gemini-embedding-001'
    m2.supported_actions = ['embedContent']
    m2.description = 'Embedder'

    m3 = MagicMock()
    m3.name = 'publishers/google/models/imagen-3.0-generate-001'
    m3.supported_actions = ['predict']
    m3.description = 'Imagen'

    m4 = MagicMock()
    m4.name = 'publishers/google/models/veo-2.0-generate-001'
    m4.supported_actions = ['generateVideos']
    m4.description = 'Veo'

    mock_client.models.list.return_value = [m1, m2, m3, m4]
    vertexai_plugin_instance._client = mock_client

    result = vertexai_plugin_instance._list_known_embedders()

    # Verify Embedder
    action2 = next(a for a in result if a.name == vertexai_name('gemini-embedding-001'))
    assert action2 is not None


@pytest.mark.asyncio
async def test_vertexai_resolve_evaluator(vertexai_plugin_instance: VertexAI) -> None:
    """Test resolve action for evaluator."""
    plugin = vertexai_plugin_instance

    action = await plugin.resolve(action_type=ActionKind.EVALUATOR, name=vertexai_name('fluency'))
    assert action is not None
    assert action.kind == ActionKind.EVALUATOR
    assert action.name == vertexai_name('fluency')


def test_config_schema_extra_fields() -> None:
    """Test that config schema accepts extra fields (dynamic config)."""
    # Validation should succeed with unknown field by using model_validate for dynamic fields
    # to avoid static type checker errors on constructor
    config_data = {'temperature': 0.5, 'new_experimental_param': 'test'}
    config = GeminiConfigSchema.model_validate(config_data)

    assert config.temperature == 0.5
    # Access dynamic fields via getattr or __dict__ to make type checker happy
    assert config.new_experimental_param == 'test'  # type: ignore
    assert config.model_dump()['new_experimental_param'] == 'test'


@pytest.mark.asyncio
async def test_system_prompt_handling() -> None:
    """Test that system prompts are correctly extracted to config."""
    mock_client = MagicMock(spec=genai.Client)
    model = GeminiModel(version='gemini-1.5-flash', client=mock_client)

    request = GenerateRequest(
        messages=[
            Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='You are a helpful assistant'))]),
            Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))]),
        ],
        config=None,
    )

    cfg = await model._genkit_to_googleai_cfg(request)

    assert cfg is not None
    assert cfg.system_instruction is not None
    assert cfg.system_instruction.parts is not None  # type: ignore
    assert len(cfg.system_instruction.parts) == 1  # type: ignore
    assert cfg.system_instruction.parts[0].text == 'You are a helpful assistant'  # type: ignore
