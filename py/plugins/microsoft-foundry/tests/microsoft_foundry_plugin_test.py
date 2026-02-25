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

"""Microsoft Foundry plugin tests.

Tests for the Microsoft Foundry plugin following the patterns from other Genkit plugins.

See: https://ai.azure.com/catalog/models

This module includes:
- Plugin initialization tests
- Model and embedder resolution tests
- Config schema validation tests
- Generate and embed functionality tests (with mocked client)
- Request/response conversion tests
"""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import AsyncAzureOpenAI

from genkit.core.registry import ActionKind
from genkit.plugins.microsoft_foundry import (
    MICROSOFT_FOUNDRY_PLUGIN_NAME,
    MicrosoftFoundry,
    MicrosoftFoundryConfig,
    gpt4o,
    microsoft_foundry_model,
)
from genkit.plugins.microsoft_foundry.models.converters import (
    extract_text,
    normalize_config,
    to_openai_role,
)
from genkit.plugins.microsoft_foundry.models.model import MicrosoftFoundryModel
from genkit.plugins.microsoft_foundry.models.model_info import get_model_info
from genkit.plugins.microsoft_foundry.plugin import _sanitize_credential
from genkit.types import (
    GenerateRequest,
    GenerateResponseChunk,
    Message,
    OutputConfig,
    Part,
    Role,
    TextPart,
    ToolRequest,
    ToolRequestPart,
)


def test_plugin_name() -> None:
    """Test that plugin name is correct."""
    assert MICROSOFT_FOUNDRY_PLUGIN_NAME == 'microsoft-foundry'


def test_plugin_init() -> None:
    """Test plugin initialization with API key."""
    plugin = MicrosoftFoundry(
        api_key='test-key',
        endpoint='https://test.openai.azure.com/',
    )
    assert plugin.name == 'microsoft-foundry'


def test_microsoft_foundry_model_helper() -> None:
    """Test the microsoft_foundry_model helper function."""
    assert microsoft_foundry_model('gpt-4o') == 'microsoft-foundry/gpt-4o'
    assert microsoft_foundry_model('gpt-4') == 'microsoft-foundry/gpt-4'
    assert microsoft_foundry_model('gpt-3.5-turbo') == 'microsoft-foundry/gpt-3.5-turbo'
    # Test with other provider models
    assert microsoft_foundry_model('DeepSeek-V3.2') == 'microsoft-foundry/DeepSeek-V3.2'
    assert microsoft_foundry_model('claude-opus-4-5') == 'microsoft-foundry/claude-opus-4-5'


def test_predefined_model_refs() -> None:
    """Test pre-defined model reference constants."""
    assert gpt4o == 'microsoft-foundry/gpt-4o'


def test_config_schema() -> None:
    """Test MicrosoftFoundryConfig schema."""
    config = MicrosoftFoundryConfig(
        temperature=0.7,
        max_tokens=100,
        frequency_penalty=0.5,
        presence_penalty=0.5,
    )
    assert config.temperature == 0.7
    assert config.max_tokens == 100
    assert config.frequency_penalty == 0.5
    assert config.presence_penalty == 0.5


def test_config_schema_with_aliases() -> None:
    """Test MicrosoftFoundryConfig with camelCase aliases."""
    config = MicrosoftFoundryConfig.model_validate({
        'maxTokens': 200,
        'topP': 0.9,
        'frequencyPenalty': 0.3,
        'visualDetailLevel': 'high',
    })
    assert config.max_tokens == 200
    assert config.top_p == 0.9
    assert config.frequency_penalty == 0.3
    assert config.visual_detail_level == 'high'


@pytest.mark.asyncio
async def test_resolve_model() -> None:
    """Test resolving a model action."""
    plugin = MicrosoftFoundry(
        api_key='test-key',
        endpoint='https://test.openai.azure.com/',
    )
    action = await plugin.resolve(ActionKind.MODEL, 'microsoft-foundry/gpt-4o')
    assert action is not None
    assert action.name == 'microsoft-foundry/gpt-4o'
    assert action.kind == ActionKind.MODEL


@pytest.mark.asyncio
async def test_resolve_embedder() -> None:
    """Test resolving an embedder action."""
    plugin = MicrosoftFoundry(
        api_key='test-key',
        endpoint='https://test.openai.azure.com/',
    )
    action = await plugin.resolve(ActionKind.EMBEDDER, 'microsoft-foundry/text-embedding-3-small')
    assert action is not None
    assert action.name == 'microsoft-foundry/text-embedding-3-small'
    assert action.kind == ActionKind.EMBEDDER


@pytest.mark.asyncio
async def test_list_actions() -> None:
    """Test listing all available actions."""
    plugin = MicrosoftFoundry(
        api_key='test-key',
        endpoint='https://test.openai.azure.com/',
    )
    actions = await plugin.list_actions()
    assert len(actions) > 0

    # Check for expected models
    action_names = [a.name for a in actions]
    assert 'microsoft-foundry/gpt-4o' in action_names
    assert 'microsoft-foundry/gpt-4o-mini' in action_names
    assert 'microsoft-foundry/gpt-4' in action_names

    # Check for embedders
    assert 'microsoft-foundry/text-embedding-3-small' in action_names
    assert 'microsoft-foundry/text-embedding-3-large' in action_names


@pytest.mark.asyncio
async def test_init_registers_actions() -> None:
    """Test that init() registers all supported actions."""
    plugin = MicrosoftFoundry(
        api_key='test-key',
        endpoint='https://test.openai.azure.com/',
    )
    actions = await plugin.init()
    assert len(actions) > 0

    # Should include both models and embedders
    model_actions = [a for a in actions if a.kind == ActionKind.MODEL]
    embedder_actions = [a for a in actions if a.kind == ActionKind.EMBEDDER]

    assert len(model_actions) > 0
    assert len(embedder_actions) > 0


class TestMicrosoftFoundryModel:
    """Tests for MicrosoftFoundryModel generation logic."""

    def test_normalize_config_with_none(self) -> None:
        """Test config normalization with None input."""
        config = normalize_config(None)
        assert isinstance(config, MicrosoftFoundryConfig)

    def test_normalize_config_with_microsoft_foundry_config(self) -> None:
        """Test config normalization with MicrosoftFoundryConfig input."""
        input_config = MicrosoftFoundryConfig(temperature=0.5, max_tokens=100)
        config = normalize_config(input_config)
        assert config.temperature == 0.5
        assert config.max_tokens == 100

    def test_normalize_config_with_dict(self) -> None:
        """Test config normalization with dict input (camelCase keys)."""
        input_config = {'temperature': 0.8, 'maxTokens': 200, 'topP': 0.9}
        config = normalize_config(input_config)
        assert config.temperature == 0.8
        assert config.max_tokens == 200
        assert config.top_p == 0.9

    def test_to_openai_role_conversion(self) -> None:
        """Test role conversion from Genkit to OpenAI format."""
        assert to_openai_role(Role.USER) == 'user'
        assert to_openai_role(Role.MODEL) == 'assistant'
        assert to_openai_role(Role.SYSTEM) == 'system'
        assert to_openai_role(Role.TOOL) == 'tool'
        # Test string roles
        assert to_openai_role('user') == 'user'
        assert to_openai_role('model') == 'assistant'

    def test_extract_text_from_message(self) -> None:
        """Test text extraction from a message."""
        msg = Message(
            role=Role.USER,
            content=[
                Part(root=TextPart(text='Hello ')),
                Part(root=TextPart(text='world!')),
            ],
        )
        text = extract_text(msg)
        assert text == 'Hello world!'

    @pytest.mark.asyncio
    async def test_generate_basic_request(self) -> None:
        """Test basic generation with mocked client."""
        mock_client = AsyncMock()

        # Mock the chat completion response
        mock_choice = MagicMock()
        mock_choice.message.content = 'Hello! How can I help you?'
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = 'stop'

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 8
        mock_response.usage.total_tokens = 18

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        model = MicrosoftFoundryModel(model_name='gpt-4o', client=mock_client)

        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='Hello'))],
                )
            ]
        )

        response = await model.generate(request, ctx=None)

        assert response is not None
        assert response.message is not None
        assert len(response.message.content) > 0
        assert response.usage is not None
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 8

    def test_build_request_body_json_schema_format(self) -> None:
        """Test that structured output uses json_schema format with schema."""
        mock_client = AsyncMock()
        model = MicrosoftFoundryModel(model_name='gpt-4o', client=mock_client)

        schema = {
            'title': 'RpgCharacter',
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'backstory': {'type': 'string'},
            },
            'required': ['name', 'backstory'],
        }

        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='Generate a character'))],
                )
            ],
            output=OutputConfig(format='json', schema=schema),
        )

        config = normalize_config(None)
        body = model._build_request_body(request, config)

        # Must use json_schema format when schema is provided
        assert body['response_format']['type'] == 'json_schema'
        assert body['response_format']['json_schema']['name'] == 'RpgCharacter'
        assert body['response_format']['json_schema']['strict'] is True
        assert 'schema' in body['response_format']['json_schema']

    def test_build_request_body_json_object_without_schema(self) -> None:
        """Test that JSON mode without schema uses json_object format."""
        mock_client = AsyncMock()
        model = MicrosoftFoundryModel(model_name='gpt-4o', client=mock_client)

        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='Give me JSON'))],
                )
            ],
            output=OutputConfig(format='json'),
        )

        config = normalize_config(None)
        body = model._build_request_body(request, config)

        assert body['response_format'] == {'type': 'json_object'}

    @pytest.mark.asyncio
    async def test_streaming_tool_request_emits_chunk(self) -> None:
        """Tool request chunks are emitted via ctx.send_chunk during streaming."""
        mock_client = AsyncMock()

        # Build mock streaming chunks for a tool call.
        def _make_chunk(
            *,
            tool_calls: list[MagicMock] | None = None,
            content: str | None = None,
            has_choices: bool = True,
            usage: MagicMock | None = None,
        ) -> MagicMock:
            """Create a mock ChatCompletionChunk."""
            chunk = MagicMock()
            if has_choices:
                delta = MagicMock()
                delta.content = content
                delta.tool_calls = tool_calls
                choice = MagicMock()
                choice.delta = delta
                choice.index = 0
                chunk.choices = [choice]
            else:
                chunk.choices = []
            chunk.usage = usage
            return chunk

        tc_start = MagicMock()
        tc_start.index = 0
        tc_start.id = 'call_abc'
        tc_start.function = MagicMock()
        tc_start.function.name = 'weather'
        tc_start.function.arguments = '{"city":'

        tc_cont = MagicMock()
        tc_cont.index = 0
        tc_cont.id = None
        tc_cont.function = MagicMock()
        tc_cont.function.name = None
        tc_cont.function.arguments = ' "London"}'

        async def _stream() -> AsyncIterator[MagicMock]:
            yield _make_chunk(tool_calls=[tc_start])
            yield _make_chunk(tool_calls=[tc_cont])
            yield _make_chunk(has_choices=False)

        mock_client.chat.completions.create = AsyncMock(return_value=_stream())

        model = MicrosoftFoundryModel(model_name='gpt-4o', client=mock_client)
        request = GenerateRequest(
            messages=[
                Message(role=Role.USER, content=[Part(root=TextPart(text='Weather?'))]),
            ],
            tools=[],
        )

        chunks: list[GenerateResponseChunk] = []
        ctx = MagicMock()
        ctx.is_streaming = True
        ctx.send_chunk = MagicMock(side_effect=lambda c: chunks.append(c))

        response = await model._generate_streaming(
            model._build_request_body(request, normalize_config(None)),
            ctx,
            request,
        )

        # Final response should contain the tool request.
        assert response.message is not None
        tool_parts = [p for p in response.message.content if isinstance(p.root, ToolRequestPart)]
        assert len(tool_parts) == 1
        tr = tool_parts[0].root.tool_request
        assert tr is not None
        assert isinstance(tr, ToolRequest)
        assert tr.name == 'weather'
        assert tr.ref == 'call_abc'
        assert tr.input == {'city': 'London'}

        # A tool request chunk must have been emitted.
        tool_chunks = [c for c in chunks if any(isinstance(p.root, ToolRequestPart) for p in c.content)]
        assert len(tool_chunks) == 1, f'Expected 1 tool request chunk, got {len(tool_chunks)}'


class TestMicrosoftFoundryEmbed:
    """Tests for embedding functionality."""

    @pytest.mark.asyncio
    async def test_embed_action_created(self) -> None:
        """Test that embedder action is created correctly."""
        plugin = MicrosoftFoundry(
            api_key='test-key',
            endpoint='https://test.openai.azure.com/',
        )
        action = await plugin.resolve(ActionKind.EMBEDDER, 'microsoft-foundry/text-embedding-3-small')

        assert action is not None
        assert action.kind == ActionKind.EMBEDDER
        assert 'text-embedding-3-small' in action.name


class TestMicrosoftFoundryModelInfo:
    """Tests for model info and capabilities."""

    def test_get_model_info_known_model(self) -> None:
        """Test getting info for a known model."""
        info = get_model_info('gpt-4o')
        assert info is not None
        assert info.label is not None
        assert 'gpt-4o' in info.label.lower() or 'Microsoft Foundry' in info.label

    def test_get_model_info_unknown_model(self) -> None:
        """Test getting info for an unknown model returns default."""
        info = get_model_info('unknown-model-xyz')
        assert info is not None
        assert info.label is not None
        assert 'unknown-model-xyz' in info.label


class TestSanitizeCredential:
    """Tests for the _sanitize_credential() defense against copy-paste Unicode artifacts.

    Credentials copied from web UIs (e.g., Azure Portal) often contain
    invisible Unicode characters like zero-width spaces (U+200B) that cause
    ``UnicodeEncodeError: 'ascii' codec can't encode character`` failures
    deep inside HTTP transport layers.
    """

    def test_none_returns_none(self) -> None:
        """None input passes through unchanged."""
        assert _sanitize_credential(None) is None

    def test_clean_string_unchanged(self) -> None:
        """Clean ASCII strings pass through unchanged."""
        assert _sanitize_credential('https://test.openai.azure.com/') == 'https://test.openai.azure.com/'

    def test_strips_zero_width_space(self) -> None:
        """Zero-width spaces (U+200B) are removed."""
        assert _sanitize_credential('abc\u200bdef') == 'abcdef'

    def test_strips_bom(self) -> None:
        """Byte-order marks (U+FEFF) are removed."""
        assert _sanitize_credential('\ufeffhttps://example.com') == 'https://example.com'

    def test_strips_non_breaking_space(self) -> None:
        """Non-breaking spaces (U+00A0) are removed."""
        assert _sanitize_credential('key\u00a0value') == 'keyvalue'

    def test_strips_multiple_invisible_chars(self) -> None:
        """All types of invisible characters are removed in one pass."""
        dirty = '\u200bhttps://\u200ctest\u200d.\u200eopenai\u200f.azure.com/\ufeff'
        assert _sanitize_credential(dirty) == 'https://test.openai.azure.com/'

    def test_strips_whitespace(self) -> None:
        """Leading and trailing whitespace is stripped."""
        assert _sanitize_credential('  https://test.com  ') == 'https://test.com'

    def test_strips_whitespace_and_invisible_chars(self) -> None:
        """Both whitespace and invisible characters are handled together."""
        assert _sanitize_credential(' \u200b api-key \u200b ') == 'api-key'

    def test_init_sanitizes_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Plugin __init__ strips invisible chars from environment variables."""
        monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'test-key\u200b')
        monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://test.openai.azure.com/\u200b')
        monkeypatch.setenv('AZURE_OPENAI_API_VERSION', '2024-10-21\u200b')

        plugin = MicrosoftFoundry()

        # The client should have been created with sanitized credentials.
        # We verify this by checking the attributes on the created client.
        assert isinstance(plugin._openai_client, AsyncAzureOpenAI)
        assert plugin._openai_client.api_key == 'test-key'
        assert str(plugin._openai_client._azure_endpoint) == 'https://test.openai.azure.com/'
        assert plugin._openai_client._api_version == '2024-10-21'
