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

"""Conftest for ollama plugin."""

from collections.abc import Generator
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import ollama as ollama_api
import pytest

from genkit import Genkit
from genkit.plugins.ollama.constants import OllamaAPITypes
from genkit.plugins.ollama.models import ModelDefinition
from genkit.plugins.ollama.plugin_api import Ollama


@pytest.fixture
def ollama_model() -> str:
    """Ollama model to use for testing."""
    return 'ollama/gemma2:latest'


@pytest.fixture
def chat_model_plugin(ollama_model: str) -> Ollama:
    """Chat model plugin parameters."""
    return Ollama(
        models=[
            ModelDefinition(
                name=ollama_model.split('/')[-1],
                api_type=OllamaAPITypes.CHAT,
            )
        ],
    )


@pytest.fixture
def genkit_veneer_chat_model(
    mock_ollama_api_async_client: MagicMock,
    ollama_model: str,
    chat_model_plugin: Ollama,
) -> Genkit:
    """Genkit veneer chat model.

    Args:
        mock_ollama_api_async_client: Mock for ollama async client (ensures it's set up first).
        ollama_model: Ollama model to use for testing.
        chat_model_plugin: Chat model plugin parameters.

    Returns:
        Genkit veneer chat model.
    """
    return Genkit(
        plugins=[chat_model_plugin],
        model=ollama_model,
    )


@pytest.fixture
def generate_model_plugin(ollama_model: str) -> Ollama:
    """Generate model plugin parameters.

    Args:
        ollama_model: Ollama model to use for testing.

    Returns:
        Generate model plugin parameters.
    """
    return Ollama(
        models=[
            ModelDefinition(
                name=ollama_model.split('/')[-1],
                api_type=OllamaAPITypes.GENERATE,
            )
        ],
    )


@pytest.fixture
def genkit_veneer_generate_model(
    mock_ollama_api_async_client: MagicMock,
    ollama_model: str,
    generate_model_plugin: Ollama,
) -> Genkit:
    """Genkit veneer generate model.

    Args:
        mock_ollama_api_async_client: Mock for ollama async client (ensures it's set up first).
        ollama_model: Ollama model to use for testing.
        generate_model_plugin: Generate model plugin parameters.

    Returns:
        Genkit veneer generate model.
    """
    return Genkit(
        plugins=[generate_model_plugin],
        model=ollama_model,
    )


@pytest.fixture
def mock_ollama_api_client() -> Generator[MagicMock | AsyncMock, None, None]:
    """Mock the ollama API client."""
    with mock.patch.object(ollama_api, 'Client') as mock_ollama_client:
        yield mock_ollama_client


@pytest.fixture
def mock_ollama_api_async_client() -> Generator[MagicMock | AsyncMock, None, None]:
    """Mock the ollama API async client."""
    with mock.patch.object(ollama_api, 'AsyncClient') as mock_ollama_async_client:
        # Create an AsyncMock instance with async methods
        client_instance = AsyncMock()
        client_instance.chat = AsyncMock()
        client_instance.generate = AsyncMock()
        client_instance.embed = AsyncMock()
        mock_ollama_async_client.return_value = client_instance
        yield mock_ollama_async_client


@pytest.fixture
@patch('ollama.AsyncClient')
def ollama_plugin_instance(ollama_async_client: MagicMock) -> Ollama:
    """Common instance of ollama plugin."""
    return Ollama()
