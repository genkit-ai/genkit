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

"""Unit tests for the CF Workers AI plugin (Cloudflare Workers AI).

These tests verify the plugin initialization, model registration,
and request/response handling.
"""

from unittest.mock import MagicMock, patch

import pytest

from genkit.core.action.types import ActionKind
from genkit.plugins.cloudflare_workers_ai.models.model import CfModel
from genkit.plugins.cloudflare_workers_ai.models.model_info import (
    SUPPORTED_CF_MODELS,
    SUPPORTED_EMBEDDING_MODELS,
    get_model_info,
)
from genkit.plugins.cloudflare_workers_ai.plugin import (
    CLOUDFLARE_WORKERS_AI_PLUGIN_NAME,
    CloudflareWorkersAI,
    cloudflare_model,
    cloudflare_name,
)
from genkit.plugins.cloudflare_workers_ai.typing import CloudflareConfig, CloudflareEmbedConfig
from genkit.types import (
    GenerateRequest,
    GenerationCommonConfig,
    Message,
    Part,
    Role,
    TextPart,
    ToolRequestPart,
)


class TestCloudflareWorkersAIPluginInit:
    """Tests for CloudflareWorkersAI plugin initialization."""

    def test_plugin_name(self) -> None:
        """Plugin name should be 'cloudflare-workers-ai'."""
        assert CLOUDFLARE_WORKERS_AI_PLUGIN_NAME == 'cloudflare-workers-ai'

    def test_cloudflare_name_helper(self) -> None:
        """cloudflare_name should create qualified model names."""
        result = cloudflare_name('@cf/meta/llama-3.1-8b-instruct')
        assert result == 'cloudflare-workers-ai/@cf/meta/llama-3.1-8b-instruct'

    def test_cloudflare_model_alias(self) -> None:
        """cloudflare_model should be an alias for cloudflare_name."""
        assert cloudflare_model == cloudflare_name

    def test_init_without_credentials_raises(self) -> None:
        """Plugin should raise ValueError without credentials."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match='account ID is required'):
                CloudflareWorkersAI()

    def test_init_without_token_raises(self) -> None:
        """Plugin should raise ValueError without API token."""
        with patch.dict('os.environ', {'CLOUDFLARE_ACCOUNT_ID': 'test-id'}, clear=True):
            with pytest.raises(ValueError, match='API token is required'):
                CloudflareWorkersAI()

    def test_init_with_env_vars(self) -> None:
        """Plugin should initialize with environment variables."""
        with patch.dict(
            'os.environ',
            {
                'CLOUDFLARE_ACCOUNT_ID': 'test-account-id',
                'CLOUDFLARE_API_TOKEN': 'test-token',
            },
        ):
            plugin = CloudflareWorkersAI()
            assert plugin._account_id == 'test-account-id'
            assert plugin._api_token == 'test-token'

    def test_init_with_explicit_params(self) -> None:
        """Plugin should accept explicit parameters."""
        plugin = CloudflareWorkersAI(
            account_id='explicit-id',
            api_token='explicit-token',
        )
        assert plugin._account_id == 'explicit-id'
        assert plugin._api_token == 'explicit-token'

    def test_init_with_custom_models(self) -> None:
        """Plugin should accept custom model list."""
        custom_models = ['@cf/meta/llama-3.1-8b-instruct']
        plugin = CloudflareWorkersAI(
            account_id='test-id',
            api_token='test-token',
            models=custom_models,
        )
        assert plugin._models == custom_models


class TestModelInfo:
    """Tests for model information registry."""

    def test_supported_models_not_empty(self) -> None:
        """SUPPORTED_CF_MODELS should contain models."""
        assert len(SUPPORTED_CF_MODELS) > 0

    def test_supported_embeddings_not_empty(self) -> None:
        """SUPPORTED_EMBEDDING_MODELS should contain models."""
        assert len(SUPPORTED_EMBEDDING_MODELS) > 0

    def test_llama_model_exists(self) -> None:
        """Llama 3.1 8B should be in supported models."""
        assert '@cf/meta/llama-3.1-8b-instruct' in SUPPORTED_CF_MODELS

    def test_bge_embedder_exists(self) -> None:
        """BGE base embedder should be in supported models."""
        assert '@cf/baai/bge-base-en-v1.5' in SUPPORTED_EMBEDDING_MODELS

    def test_get_model_info_known(self) -> None:
        """get_model_info should return info for known models."""
        info = get_model_info('@cf/meta/llama-3.1-8b-instruct')
        assert info.label is not None
        assert 'Llama' in info.label

    def test_get_model_info_unknown(self) -> None:
        """get_model_info should return default for unknown models."""
        info = get_model_info('@cf/unknown/model')
        assert info.label is not None
        assert 'unknown' in info.label.lower() or 'Cloudflare' in info.label


class TestCloudflareConfig:
    """Tests for CloudflareConfig schema."""

    def test_default_config(self) -> None:
        """Default config should have None values."""
        config = CloudflareConfig()
        assert config.temperature is None
        assert config.max_output_tokens is None
        assert config.top_k is None
        assert config.seed is None

    def test_config_with_values(self) -> None:
        """Config should accept valid values."""
        config = CloudflareConfig(
            temperature=0.7,
            max_output_tokens=1024,
            top_k=40,
            seed=42,
            repetition_penalty=1.1,
        )
        assert config.temperature == 0.7
        assert config.max_output_tokens == 1024
        assert config.top_k == 40
        assert config.seed == 42
        assert config.repetition_penalty == 1.1

    def test_config_top_k_bounds(self) -> None:
        """top_k should be validated (1-50)."""
        # Valid
        config = CloudflareConfig(top_k=1)
        assert config.top_k == 1

        config = CloudflareConfig(top_k=50)
        assert config.top_k == 50

    def test_embed_config(self) -> None:
        """CloudflareEmbedConfig should accept pooling parameter."""
        # Default pooling is None
        config = CloudflareEmbedConfig()
        assert config.pooling is None

        # Accept 'mean' pooling
        config = CloudflareEmbedConfig(pooling='mean')
        assert config.pooling == 'mean'

        # Accept 'cls' pooling
        config = CloudflareEmbedConfig(pooling='cls')
        assert config.pooling == 'cls'


class TestCfModel:
    """Tests for CfModel class."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock httpx client."""
        return MagicMock()

    @pytest.fixture
    def model(self, mock_client: MagicMock) -> CfModel:
        """Create a CfModel instance."""
        return CfModel(
            model_id='@cf/meta/llama-3.1-8b-instruct',
            account_id='test-account',
            client=mock_client,
        )

    def test_model_initialization(self, model: CfModel) -> None:
        """Model should initialize with correct attributes."""
        assert model.model_id == '@cf/meta/llama-3.1-8b-instruct'
        assert model.account_id == 'test-account'

    def test_get_api_url(self, model: CfModel) -> None:
        """_get_api_url should return correct URL."""
        url = model._get_api_url()
        expected = 'https://api.cloudflare.com/client/v4/accounts/test-account/ai/run/@cf/meta/llama-3.1-8b-instruct'
        assert url == expected

    def test_normalize_config_none(self, model: CfModel) -> None:
        """_normalize_config should handle None."""
        config = model._normalize_config(None)
        assert isinstance(config, CloudflareConfig)

    def test_normalize_config_cf_config(self, model: CfModel) -> None:
        """_normalize_config should pass through CloudflareConfig."""
        original = CloudflareConfig(temperature=0.5)
        config = model._normalize_config(original)
        assert config is original

    def test_normalize_config_common_config(self, model: CfModel) -> None:
        """_normalize_config should convert GenerationCommonConfig."""
        common = GenerationCommonConfig(temperature=0.7, max_output_tokens=512)
        config = model._normalize_config(common)
        assert isinstance(config, CloudflareConfig)
        assert config.temperature == 0.7
        assert config.max_output_tokens == 512

    def test_normalize_config_dict(self, model: CfModel) -> None:
        """_normalize_config should handle dict with camelCase keys."""
        d = {'temperature': 0.8, 'maxOutputTokens': 256, 'topK': 40}
        config = model._normalize_config(d)
        assert isinstance(config, CloudflareConfig)
        assert config.temperature == 0.8
        assert config.max_output_tokens == 256
        assert config.top_k == 40

    def test_to_cloudflare_role_user(self, model: CfModel) -> None:
        """_to_cloudflare_role should map user role."""
        result = model._to_cloudflare_role(Role.USER)
        assert result == 'user'

    def test_to_cloudflare_role_model(self, model: CfModel) -> None:
        """_to_cloudflare_role should map model to assistant."""
        result = model._to_cloudflare_role(Role.MODEL)
        assert result == 'assistant'

    def test_to_cloudflare_role_system(self, model: CfModel) -> None:
        """_to_cloudflare_role should map system role."""
        result = model._to_cloudflare_role(Role.SYSTEM)
        assert result == 'system'


class TestResponseParsing:
    """Tests for response parsing logic."""

    @pytest.fixture
    def model(self) -> CfModel:
        """Create a CfModel instance for testing."""
        mock_client = MagicMock()
        return CfModel(
            model_id='@cf/meta/llama-3.1-8b-instruct',
            account_id='test-account',
            client=mock_client,
        )

    def test_parse_simple_response(self, model: CfModel) -> None:
        """_parse_response should handle simple text response."""
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='Hello'))],
                )
            ]
        )

        data = {
            'result': {
                'response': 'Hello! How can I help you?',
                'usage': {
                    'prompt_tokens': 5,
                    'completion_tokens': 10,
                    'total_tokens': 15,
                },
            }
        }

        response = model._parse_response(data, request)

        # Explicit None checks for type narrowing
        assert response.message is not None
        assert response.message.role == Role.MODEL
        assert len(response.message.content) == 1

        text_part = response.message.content[0].root
        assert isinstance(text_part, TextPart)
        assert text_part.text == 'Hello! How can I help you?'

        assert response.usage is not None
        assert response.usage.input_tokens == 5
        assert response.usage.output_tokens == 10

    def test_parse_response_with_tool_calls(self, model: CfModel) -> None:
        """_parse_response should handle tool call responses."""
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='What is the weather?'))],
                )
            ]
        )

        data = {
            'result': {
                'response': '',
                'tool_calls': [
                    {
                        'name': 'get_weather',
                        'arguments': {'location': 'NYC'},
                    }
                ],
            }
        }

        response = model._parse_response(data, request)

        # Explicit None checks for type narrowing
        assert response.message is not None

        # Should have tool request in content
        assert len(response.message.content) >= 1
        tool_part = response.message.content[0]

        # Narrow the type to ToolRequestPart
        assert isinstance(tool_part.root, ToolRequestPart)
        assert tool_part.root.tool_request.name == 'get_weather'

    def test_parse_response_with_openai_nested_tool_calls(self, model: CfModel) -> None:
        """_parse_response should handle OpenAI-compatible nested tool call format.

        Regression test: Cloudflare returns tool calls in OpenAI format with
        a nested ``function`` object containing ``name`` and ``arguments``
        (as a JSON string).  Previously the parser read ``tool_call['name']``
        which was always empty.
        """
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='What is the weather?'))],
                )
            ]
        )

        data = {
            'result': {
                'tool_calls': [
                    {
                        'id': 'chatcmpl-tool-abc123',
                        'type': 'function',
                        'function': {
                            'name': 'weather',
                            'arguments': '{"city": "New York"}',
                        },
                    }
                ],
                'usage': {
                    'prompt_tokens': 279,
                    'completion_tokens': 17,
                    'total_tokens': 296,
                },
            },
        }

        response = model._parse_response(data, request)

        assert response.message is not None
        assert len(response.message.content) == 1

        tool_part = response.message.content[0]
        assert isinstance(tool_part.root, ToolRequestPart)
        assert tool_part.root.tool_request.name == 'weather'
        assert tool_part.root.tool_request.input == {'city': 'New York'}

    def test_parse_response_with_multiple_nested_tool_calls(self, model: CfModel) -> None:
        """_parse_response should handle multiple OpenAI-format tool calls."""
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='Weather in NYC and LA'))],
                )
            ]
        )

        data = {
            'result': {
                'tool_calls': [
                    {
                        'id': 'call-1',
                        'type': 'function',
                        'function': {
                            'name': 'weather',
                            'arguments': '{"city": "New York"}',
                        },
                    },
                    {
                        'id': 'call-2',
                        'type': 'function',
                        'function': {
                            'name': 'weather',
                            'arguments': '{"city": "Los Angeles"}',
                        },
                    },
                ],
            },
        }

        response = model._parse_response(data, request)

        assert response.message is not None
        assert len(response.message.content) == 2

        part0 = response.message.content[0]
        assert isinstance(part0.root, ToolRequestPart)
        assert part0.root.tool_request.name == 'weather'
        assert part0.root.tool_request.input == {'city': 'New York'}

        part1 = response.message.content[1]
        assert isinstance(part1.root, ToolRequestPart)
        assert part1.root.tool_request.name == 'weather'
        assert part1.root.tool_request.input == {'city': 'Los Angeles'}

    def test_parse_response_with_flat_tool_calls(self, model: CfModel) -> None:
        """_parse_response should still handle flat tool call format (backwards compat)."""
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='What is the weather?'))],
                )
            ]
        )

        data = {
            'result': {
                'tool_calls': [
                    {
                        'name': 'get_weather',
                        'arguments': {'location': 'NYC'},
                    }
                ],
            }
        }

        response = model._parse_response(data, request)

        assert response.message is not None
        assert len(response.message.content) == 1

        tool_part = response.message.content[0]
        assert isinstance(tool_part.root, ToolRequestPart)
        assert tool_part.root.tool_request.name == 'get_weather'
        assert tool_part.root.tool_request.input == {'location': 'NYC'}

    def test_parse_response_tool_call_malformed_json_args(self, model: CfModel) -> None:
        """_parse_response should handle malformed JSON in arguments gracefully."""
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='test'))],
                )
            ]
        )

        data = {
            'result': {
                'tool_calls': [
                    {
                        'id': 'call-1',
                        'type': 'function',
                        'function': {
                            'name': 'weather',
                            'arguments': '{invalid json',
                        },
                    }
                ],
            }
        }

        response = model._parse_response(data, request)

        assert response.message is not None
        assert len(response.message.content) == 1

        tool_part = response.message.content[0]
        assert isinstance(tool_part.root, ToolRequestPart)
        assert tool_part.root.tool_request.name == 'weather'
        # Malformed JSON falls back to empty dict
        assert tool_part.root.tool_request.input == {}


class TestClientCaching:
    """Tests for per-event-loop client caching."""

    @pytest.fixture(autouse=True)
    def clear_cache(self) -> None:
        """Clear the HTTP client cache before each test for isolation."""
        from genkit.core.http_client import clear_client_cache

        clear_client_cache()

    @pytest.mark.asyncio
    async def test_client_cached_per_event_loop(self) -> None:
        """Client should be cached and reused within the same event loop."""
        plugin = CloudflareWorkersAI(
            account_id='test-id',
            api_token='test-token',
        )

        # Get client twice in the same event loop
        client1 = plugin._get_client()
        client2 = plugin._get_client()

        # Should be the same instance (cached)
        assert client1 is client2

        # Clean up
        await client1.aclose()

    @pytest.mark.asyncio
    async def test_client_has_correct_headers(self) -> None:
        """Client should have authorization headers configured."""
        plugin = CloudflareWorkersAI(
            account_id='test-id',
            api_token='my-secret-token',
        )

        client = plugin._get_client()

        assert 'Authorization' in client.headers
        assert client.headers['Authorization'] == 'Bearer my-secret-token'
        assert client.headers['Content-Type'] == 'application/json'

        # Clean up
        await client.aclose()

    @pytest.mark.asyncio
    async def test_closed_client_gets_replaced(self) -> None:
        """A closed client should be replaced with a new one."""
        plugin = CloudflareWorkersAI(
            account_id='test-id',
            api_token='test-token',
        )

        # Get client and close it
        client1 = plugin._get_client()
        await client1.aclose()

        # Get client again - should be a new instance
        client2 = plugin._get_client()

        assert client1 is not client2
        assert not client2.is_closed

        # Clean up
        await client2.aclose()


class TestPluginActions:
    """Tests for plugin action methods."""

    @pytest.mark.asyncio
    async def test_init_returns_empty_list(self) -> None:
        """init() should return empty list (using lazy loading)."""
        plugin = CloudflareWorkersAI(
            account_id='test-id',
            api_token='test-token',
            models=['@cf/meta/llama-3.1-8b-instruct'],
            embedders=[],
        )

        result = await plugin.init()
        assert result == []

    @pytest.mark.asyncio
    async def test_resolve_returns_model_action(self) -> None:
        """resolve() should return Action for model."""
        plugin = CloudflareWorkersAI(
            account_id='test-id',
            api_token='test-token',
            models=['@cf/meta/llama-3.1-8b-instruct'],
            embedders=[],
        )

        action = await plugin.resolve(ActionKind.MODEL, 'cloudflare-workers-ai/@cf/meta/llama-3.1-8b-instruct')

        assert action is not None
        assert action.kind == ActionKind.MODEL
        assert action.name == 'cloudflare-workers-ai/@cf/meta/llama-3.1-8b-instruct'

    @pytest.mark.asyncio
    async def test_resolve_returns_embedder_action(self) -> None:
        """resolve() should return Action for embedder."""
        plugin = CloudflareWorkersAI(
            account_id='test-id',
            api_token='test-token',
            models=[],
            embedders=['@cf/baai/bge-base-en-v1.5'],
        )

        action = await plugin.resolve(ActionKind.EMBEDDER, 'cloudflare-workers-ai/@cf/baai/bge-base-en-v1.5')

        assert action is not None
        assert action.kind == ActionKind.EMBEDDER
        assert action.name == 'cloudflare-workers-ai/@cf/baai/bge-base-en-v1.5'

    @pytest.mark.asyncio
    async def test_resolve_returns_none_for_unknown(self) -> None:
        """resolve() should return None for unknown action types."""
        plugin = CloudflareWorkersAI(
            account_id='test-id',
            api_token='test-token',
        )

        action = await plugin.resolve(ActionKind.TOOL, 'some-tool')

        assert action is None

    @pytest.mark.asyncio
    async def test_list_actions(self) -> None:
        """list_actions() should return available actions."""
        plugin = CloudflareWorkersAI(
            account_id='test-id',
            api_token='test-token',
            models=['@cf/meta/llama-3.1-8b-instruct'],
            embedders=['@cf/baai/bge-base-en-v1.5'],
        )

        actions = await plugin.list_actions()

        assert len(actions) == 2
        names = [a.name for a in actions]
        assert 'cloudflare-workers-ai/@cf/meta/llama-3.1-8b-instruct' in names
        assert 'cloudflare-workers-ai/@cf/baai/bge-base-en-v1.5' in names
