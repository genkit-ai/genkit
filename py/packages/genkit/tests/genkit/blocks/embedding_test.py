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

"""Tests for the action module."""

from collections.abc import Callable
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from genkit.ai._aio import Genkit
from genkit.blocks.document import Document
from genkit.blocks.embedding import (
    EmbedderOptions,
    EmbedderSupports,
    create_embedder_ref,
    embedder_action_metadata,
)
from genkit.core.action import Action, ActionMetadata
from genkit.core.action.types import ActionResponse
from genkit.core.schema import to_json_schema
from genkit.core.typing import Embedding, EmbedRequest, EmbedResponse


def test_embedder_action_metadata() -> None:
    """Test for embedder_action_metadata with basic options."""
    options = EmbedderOptions(label='Test Embedder', dimensions=128)
    action_metadata = embedder_action_metadata(
        name='test_model',
        options=options,
    )

    assert isinstance(action_metadata, ActionMetadata)
    assert action_metadata.input_json_schema is not None
    assert action_metadata.output_json_schema is not None
    assert action_metadata.metadata == {
        'embedder': {
            'label': options.label,
            'dimensions': options.dimensions,
            'customOptions': None,
        }
    }


def test_embedder_action_metadata_with_supports_and_config_schema() -> None:
    """Test for embedder_action_metadata with supports and config_schema."""

    class CustomConfig(BaseModel):
        param1: str
        param2: int

    options = EmbedderOptions(
        label='Advanced Embedder',
        dimensions=256,
        supports=EmbedderSupports(input=['text', 'image']),
        config_schema=to_json_schema(CustomConfig),
    )
    action_metadata = embedder_action_metadata(
        name='advanced_model',
        options=options,
    )
    assert isinstance(action_metadata, ActionMetadata)
    assert action_metadata.metadata is not None
    metadata = cast(dict[str, Any], action_metadata.metadata)
    embedder_meta = cast(dict[str, Any], metadata['embedder'])
    assert embedder_meta['label'] == 'Advanced Embedder'
    assert embedder_meta['dimensions'] == options.dimensions
    assert embedder_meta['supports'] == {
        'input': ['text', 'image'],
    }
    assert embedder_meta['customOptions'] == {
        'title': 'CustomConfig',
        'type': 'object',
        'properties': {
            'param1': {'title': 'Param1', 'type': 'string'},
            'param2': {'title': 'Param2', 'type': 'integer'},
        },
        'required': ['param1', 'param2'],
    }


def test_embedder_action_metadata_no_options() -> None:
    """Test embedder_action_metadata when no options are provided."""
    action_metadata = embedder_action_metadata(name='default_model')
    assert isinstance(action_metadata, ActionMetadata)
    assert action_metadata.metadata == {'embedder': {'customOptions': None, 'dimensions': None}}


def test_create_embedder_ref_basic() -> None:
    """Test basic creation of EmbedderRef."""
    ref = create_embedder_ref('my-embedder')
    assert ref.name == 'my-embedder'
    assert ref.config is None
    assert ref.version is None


def test_create_embedder_ref_with_config() -> None:
    """Test creation of EmbedderRef with configuration."""
    config = {'temperature': 0.5, 'max_tokens': 100}
    ref = create_embedder_ref('configured-embedder', config=config)
    assert ref.name == 'configured-embedder'
    assert ref.config == config
    assert ref.version is None


def test_create_embedder_ref_with_version() -> None:
    """Test creation of EmbedderRef with a version."""
    ref = create_embedder_ref('versioned-embedder', version='v1.0')
    assert ref.name == 'versioned-embedder'
    assert ref.config is None
    assert ref.version == 'v1.0'


def test_create_embedder_ref_with_config_and_version() -> None:
    """Test creation of EmbedderRef with both config and version."""
    config = {'task_type': 'retrieval'}
    ref = create_embedder_ref('full-embedder', config=config, version='beta')
    assert ref.name == 'full-embedder'
    assert ref.config == config
    assert ref.version == 'beta'


class MockGenkitRegistry:
    """A mock registry to simulate action lookup."""

    def __init__(self) -> None:
        """Initialize the MockGenkitRegistry."""
        self.actions = {}

    def register_action(
        self,
        name: str,
        kind: str,
        fn: Callable[..., Any],
        metadata: dict[str, object] | None,
        description: str | None,
    ) -> Any:  # noqa: ANN401
        """Register a mock action.

        Note: Returns Any because we return MagicMock objects that have
        mock-specific attributes like assert_called_once and call_args.
        """
        mock_action = MagicMock(spec=Action)
        mock_action.name = name
        mock_action.kind = kind
        mock_action.metadata = metadata
        mock_action.description = description

        async def mock_arun_side_effect(request: object, *args: object, **kwargs: object) -> ActionResponse:
            # Call the actual (fake) embedder function directly
            embed_response = await fn(request)
            return ActionResponse(response=embed_response, trace_id='mock_trace_id')

        mock_action.arun = AsyncMock(side_effect=mock_arun_side_effect)
        self.actions[kind, name] = mock_action
        return mock_action

    async def resolve_action(self, kind: str, name: str) -> Any:  # noqa: ANN401
        """Async action resolution for new plugin API.

        Note: Returns Any because actions are MagicMock objects.
        """
        return self.actions.get((kind, name))

    async def resolve_embedder(self, name: str) -> Any:  # noqa: ANN401
        """Typed embedder resolution.

        Note: Returns Any because actions are MagicMock objects.
        """
        return self.actions.get(('embedder', name))


@pytest.fixture
def mock_genkit_instance() -> tuple[Genkit, MockGenkitRegistry]:
    """Fixture for a Genkit instance with a mock registry."""
    registry = MockGenkitRegistry()
    genkit_instance = Genkit()
    genkit_instance.registry = registry  # type: ignore[assignment]
    return genkit_instance, registry


@pytest.mark.asyncio
async def test_embed_with_embedder_ref(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry],
) -> None:
    """Test the embed method using EmbedderRef."""
    genkit_instance, registry = mock_genkit_instance

    async def fake_embedder_fn(request: EmbedRequest) -> EmbedResponse:
        return EmbedResponse(embeddings=[Embedding(embedding=[1.0, 2.0, 3.0])])

    embedder_options = EmbedderOptions(
        label='Fake Embedder',
        dimensions=3,
        supports=EmbedderSupports(input=['text']),
        config_schema={'type': 'object', 'properties': {'param': {'type': 'string'}}},
    )
    registry.register_action(
        name='my-plugin/my-embedder',
        kind='embedder',
        fn=fake_embedder_fn,
        metadata=embedder_action_metadata('my-plugin/my-embedder', options=embedder_options).metadata,
        description='A fake embedder for testing',
    )
    embedder_ref = create_embedder_ref('my-plugin/my-embedder', config={'param': 'value'}, version='v1')

    content = Document.from_text('hello world')

    response = await genkit_instance.embed(embedder=embedder_ref, content=content, options={'additional_option': True})

    assert response[0].embedding == [1.0, 2.0, 3.0]

    embed_action = await registry.resolve_action('embedder', 'my-plugin/my-embedder')
    assert embed_action is not None
    embed_action.arun.assert_called_once()

    called_request = embed_action.arun.call_args[0][0]
    assert isinstance(called_request, EmbedRequest)
    assert called_request.input == [content]
    # Check if config from EmbedderRef and options are merged correctly
    assert called_request.options == {'param': 'value', 'additional_option': True, 'version': 'v1'}


@pytest.mark.asyncio
async def test_embed_with_string_name_and_options(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry],
) -> None:
    """Test the embed method using a string name for embedder and options."""
    genkit_instance, registry = mock_genkit_instance

    async def fake_embedder_fn(request: EmbedRequest) -> EmbedResponse:
        return EmbedResponse(embeddings=[Embedding(embedding=[4.0, 5.0, 6.0])])

    embedder_options = EmbedderOptions(label='Another Fake', dimensions=3)
    registry.register_action(
        name='another-embedder',
        kind='embedder',
        fn=fake_embedder_fn,
        metadata=embedder_action_metadata('another-embedder', options=embedder_options).metadata,
        description='Another fake embedder',
    )

    content = 'test text'

    response = await genkit_instance.embed(
        embedder='another-embedder', content=content, options={'custom_setting': 'high'}
    )

    assert response[0].embedding == [4.0, 5.0, 6.0]
    embed_action = await registry.resolve_action('embedder', 'another-embedder')
    called_request = embed_action.arun.call_args[0][0]
    assert called_request.options == {'custom_setting': 'high'}


@pytest.mark.asyncio
async def test_embed_missing_embedder_raises_error(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry],
) -> None:
    """Test that embedding with a missing embedder raises an error."""
    genkit_instance, _ = mock_genkit_instance
    content = 'some text'

    with pytest.raises(ValueError, match='Embedder must be specified as a string name or an EmbedderRef.'):
        await genkit_instance.embed(content=content)


@pytest.mark.asyncio
async def test_embed_many(mock_genkit_instance: tuple[Genkit, MockGenkitRegistry]) -> None:
    """Test the embed_many method."""
    genkit_instance, registry = mock_genkit_instance

    async def fake_embedder_fn(request: EmbedRequest) -> EmbedResponse:
        return EmbedResponse(embeddings=[Embedding(embedding=[1.0, 1.1]), Embedding(embedding=[2.0, 2.1])])

    registry.register_action(
        name='multi-embedder',
        kind='embedder',
        fn=fake_embedder_fn,
        metadata=embedder_action_metadata('multi-embedder').metadata,
        description='A multi embedder for testing',
    )

    content = ['text1', 'text2']
    response = await genkit_instance.embed_many(embedder='multi-embedder', content=content)

    assert len(response) == 2
    assert response[0].embedding == [1.0, 1.1]
    assert response[1].embedding == [2.0, 2.1]

    embed_action = await registry.resolve_action('embedder', 'multi-embedder')
    called_request = embed_action.arun.call_args[0][0]
    assert called_request.input == [Document.from_text('text1'), Document.from_text('text2')]


# --- Tests for _resolve_embedder_name helper ---


def test_resolve_embedder_name_with_string() -> None:
    """Test _resolve_embedder_name returns name when given a string."""
    genkit_instance = Genkit()
    result = genkit_instance._resolve_embedder_name('my-embedder')
    assert result == 'my-embedder'


def test_resolve_embedder_name_with_embedder_ref() -> None:
    """Test _resolve_embedder_name extracts name from EmbedderRef."""
    genkit_instance = Genkit()
    ref = create_embedder_ref('ref-embedder', config={'key': 'value'}, version='v1')
    result = genkit_instance._resolve_embedder_name(ref)
    assert result == 'ref-embedder'


def test_resolve_embedder_name_with_none_raises_error() -> None:
    """Test _resolve_embedder_name raises ValueError when given None."""
    genkit_instance = Genkit()
    with pytest.raises(ValueError, match='Embedder must be specified as a string name or an EmbedderRef.'):
        genkit_instance._resolve_embedder_name(None)


def test_resolve_embedder_name_with_invalid_type_raises_error() -> None:
    """Test _resolve_embedder_name raises ValueError for invalid types."""
    genkit_instance = Genkit()
    with pytest.raises(ValueError, match='Embedder must be specified as a string name or an EmbedderRef.'):
        genkit_instance._resolve_embedder_name(123)  # type: ignore[arg-type]
