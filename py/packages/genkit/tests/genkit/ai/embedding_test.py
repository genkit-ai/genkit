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
from pydantic import BaseModel, ValidationError

from genkit import Document, Genkit, GenkitError
from genkit._ai._embedding import (
    EmbedderInfo,
    EmbedderRef,
    EmbedderSupports,
    create_embedder_ref,
    embedder,
    embedder_action_metadata,
)
from genkit._core._action import Action, ActionResponse
from genkit._core._schema import to_json_schema
from genkit._core._typing import ActionMetadata, Embedding, EmbedResponse
from genkit.embedder import EmbedRequest


def test_embedder_action_metadata() -> None:
    """Test for embedder_action_metadata with a catalog card."""
    info = EmbedderInfo(label='Test Embedder', dimensions=128)
    action_metadata = embedder_action_metadata(
        name='test_model',
        info=info,
    )

    assert isinstance(action_metadata, ActionMetadata)
    assert action_metadata.input_json_schema is not None
    assert action_metadata.output_json_schema is not None
    assert action_metadata.metadata == {
        'embedder': {
            'label': info.label,
            'dimensions': info.dimensions,
            'customOptions': None,
        }
    }


def test_embedder_action_metadata_with_supports_and_config_schema() -> None:
    """Test for embedder_action_metadata with supports and config_schema."""

    class CustomConfig(BaseModel):
        param1: str
        param2: int

    info = EmbedderInfo(
        label='Advanced Embedder',
        dimensions=256,
        supports=EmbedderSupports(input=['text', 'image']),
        config_schema=to_json_schema(CustomConfig),
    )
    action_metadata = embedder_action_metadata(
        name='advanced_model',
        info=info,
    )
    assert isinstance(action_metadata, ActionMetadata)
    assert action_metadata.metadata is not None
    metadata = action_metadata.metadata
    embedder_meta = cast(dict[str, Any], metadata['embedder'])
    assert embedder_meta['label'] == 'Advanced Embedder'
    assert embedder_meta['dimensions'] == info.dimensions
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


@pytest.mark.asyncio
async def test_embedder_factory_does_not_register() -> None:
    """Plugin resolve builds via embedder(); the registry is what registers."""

    async def embed_fn(request: EmbedRequest) -> EmbedResponse:
        return EmbedResponse(embeddings=[Embedding(embedding=[1.0])])

    ai = Genkit()
    action = embedder('text-plugin-style', embed_fn)

    assert await ai.registry.resolve_action(action.kind, action.name) is None

    ai.registry.register_action_from_instance(action)
    resolved = await ai.registry.resolve_action(action.kind, action.name)
    assert resolved is action


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


def test_create_embedder_ref_with_positional_config_raises_type_error() -> None:
    """create_embedder_ref(name, {...}) raises TypeError; settings go in config=."""
    with pytest.raises(TypeError):
        create_embedder_ref('e', {'task': 'retrieval'})  # type: ignore[misc]


def test_create_embedder_ref_with_positional_version_raises_type_error() -> None:
    """create_embedder_ref(name, config, version) raises TypeError; version= is named."""
    with pytest.raises(TypeError):
        create_embedder_ref('e', None, 'v1')  # type: ignore[misc]


@pytest.mark.parametrize(
    'build',
    [
        lambda: create_embedder_ref('e', config=cast(Any, 'v1')),
        lambda: EmbedderRef(name='e', config=cast(Any, 'v1')),
    ],
    ids=['create_embedder_ref', 'EmbedderRef'],
)
def test_embedder_ref_with_non_dict_config_raises_validation_error(build: Callable[[], EmbedderRef]) -> None:
    """A non-dict config raises instead of being silently dropped by ai.embed."""
    with pytest.raises(ValidationError):
        build()


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

        mock_action.run = AsyncMock(side_effect=mock_arun_side_effect)
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

    embedder_info = EmbedderInfo(
        label='Fake Embedder',
        dimensions=3,
        supports=EmbedderSupports(input=['text']),
        config_schema={'type': 'object', 'properties': {'param': {'type': 'string'}}},
    )
    registry.register_action(
        name='my-plugin/my-embedder',
        kind='embedder',
        fn=fake_embedder_fn,
        metadata=embedder_action_metadata('my-plugin/my-embedder', info=embedder_info).metadata,
        description='A fake embedder for testing',
    )
    embedder_ref = create_embedder_ref('my-plugin/my-embedder', config={'param': 'value'}, version='v1')

    content = Document.from_text('hello world')

    response = await genkit_instance.embed(embedder=embedder_ref, content=content, config={'additional_option': True})

    assert response[0].embedding == [1.0, 2.0, 3.0]

    embed_action = await registry.resolve_action('embedder', 'my-plugin/my-embedder')
    assert embed_action is not None
    embed_action.run.assert_called_once()

    called_request = embed_action.run.call_args[0][0]
    assert isinstance(called_request, EmbedRequest)
    assert called_request.input == [content]
    # ref config, version, and call config all arrive as request.options
    assert called_request.options == {'param': 'value', 'additional_option': True, 'version': 'v1'}


@pytest.mark.asyncio
async def test_create_embedder_ref_config_keyword_reaches_the_embedder(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry],
) -> None:
    """create_embedder_ref(name, config={...}) arrives at the embedder as request.options."""
    genkit_instance, registry = mock_genkit_instance

    async def fake_embedder_fn(request: EmbedRequest) -> EmbedResponse:
        return EmbedResponse(embeddings=[Embedding(embedding=[1.0])])

    registry.register_action(
        name='kw-embedder',
        kind='embedder',
        fn=fake_embedder_fn,
        metadata=embedder_action_metadata('kw-embedder').metadata,
        description='A fake embedder for testing',
    )
    ref = create_embedder_ref('kw-embedder', config={'task': 'retrieval'})

    response = await genkit_instance.embed(embedder=ref, content='hello')

    assert response[0].embedding == [1.0]
    embed_action = await registry.resolve_action('embedder', 'kw-embedder')
    called_request = embed_action.run.call_args[0][0]
    assert isinstance(called_request, EmbedRequest)
    assert called_request.options == {'task': 'retrieval'}


@pytest.mark.asyncio
async def test_embed_config_reaches_embedder_as_options(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry],
) -> None:
    """ai.embed(config={...}) arrives at the embedder as request.options."""
    genkit_instance, registry = mock_genkit_instance

    async def fake_embedder_fn(request: EmbedRequest) -> EmbedResponse:
        return EmbedResponse(embeddings=[Embedding(embedding=[4.0, 5.0, 6.0])])

    embedder_info = EmbedderInfo(label='Another Fake', dimensions=3)
    registry.register_action(
        name='another-embedder',
        kind='embedder',
        fn=fake_embedder_fn,
        metadata=embedder_action_metadata('another-embedder', info=embedder_info).metadata,
        description='Another fake embedder',
    )

    content = 'test text'

    response = await genkit_instance.embed(
        embedder='another-embedder', content=content, config={'custom_setting': 'high'}
    )

    assert response[0].embedding == [4.0, 5.0, 6.0]
    embed_action = await registry.resolve_action('embedder', 'another-embedder')
    called_request = embed_action.run.call_args[0][0]
    assert called_request.options == {'custom_setting': 'high'}


@pytest.mark.asyncio
async def test_embed_missing_embedder_raises_error(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry],
) -> None:
    """ai.embed without embedder= fails at the call, before anything runs."""
    genkit_instance, _ = mock_genkit_instance
    content = 'some text'

    with pytest.raises(TypeError, match='embedder'):
        await genkit_instance.embed(content=content)  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]


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
    called_request = embed_action.run.call_args[0][0]
    assert called_request.input == [Document.from_text('text1'), Document.from_text('text2')]


@pytest.mark.asyncio
async def test_embed_many_with_embedder_ref_merges_config_the_same_as_embed(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry],
) -> None:
    """embed_many with an EmbedderRef merges ref config, version, and call config like embed."""
    genkit_instance, registry = mock_genkit_instance

    async def fake_embedder_fn(request: EmbedRequest) -> EmbedResponse:
        return EmbedResponse(embeddings=[Embedding(embedding=[1.0]), Embedding(embedding=[2.0])])

    registry.register_action(
        name='my-plugin/my-embedder',
        kind='embedder',
        fn=fake_embedder_fn,
        metadata=embedder_action_metadata('my-plugin/my-embedder').metadata,
        description='A fake embedder for testing',
    )
    embedder_ref = create_embedder_ref('my-plugin/my-embedder', config={'param': 'value'}, version='v1')
    content = [Document.from_text('one'), Document.from_text('two')]

    response = await genkit_instance.embed_many(embedder=embedder_ref, content=content, config={'extra': True})

    assert [item.embedding for item in response] == [[1.0], [2.0]]
    embed_action = await registry.resolve_action('embedder', 'my-plugin/my-embedder')
    called_request = embed_action.run.call_args[0][0]
    assert isinstance(called_request, EmbedRequest)
    assert called_request.input == content
    assert called_request.options == {'param': 'value', 'version': 'v1', 'extra': True}


@pytest.mark.asyncio
async def test_embed_many_call_config_wins_over_embedder_ref_config(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry],
) -> None:
    """embed_many config= wins over the same key on the EmbedderRef."""
    genkit_instance, registry = mock_genkit_instance

    async def fake_embedder_fn(request: EmbedRequest) -> EmbedResponse:
        return EmbedResponse(embeddings=[Embedding(embedding=[1.0])])

    registry.register_action(
        name='override-embedder',
        kind='embedder',
        fn=fake_embedder_fn,
        metadata=embedder_action_metadata('override-embedder').metadata,
        description='A fake embedder for testing',
    )
    embedder_ref = create_embedder_ref('override-embedder', config={'param': 'from_ref'})

    response = await genkit_instance.embed_many(
        embedder=embedder_ref,
        content=['hello'],
        config={'param': 'override'},
    )

    assert response[0].embedding == [1.0]
    embed_action = await registry.resolve_action('embedder', 'override-embedder')
    called_request = embed_action.run.call_args[0][0]
    assert called_request.options == {'param': 'override'}


@pytest.mark.asyncio
async def test_embed_many_does_not_change_the_embedder_ref_config(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry],
) -> None:
    """embed_many leaves the EmbedderRef config dict unchanged."""
    genkit_instance, registry = mock_genkit_instance

    async def fake_embedder_fn(request: EmbedRequest) -> EmbedResponse:
        return EmbedResponse(embeddings=[Embedding(embedding=[1.0])])

    registry.register_action(
        name='stable-embedder',
        kind='embedder',
        fn=fake_embedder_fn,
        metadata=embedder_action_metadata('stable-embedder').metadata,
        description='A fake embedder for testing',
    )
    config = {'param': 'value'}
    embedder_ref = create_embedder_ref('stable-embedder', config=config, version='v1')

    await genkit_instance.embed_many(
        embedder=embedder_ref,
        content=['hello'],
        config={'extra': True},
    )

    assert embedder_ref.config == {'param': 'value'}
    assert config == {'param': 'value'}


@pytest.mark.asyncio
async def test_embed_many_config_reaches_embedder_as_options(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry],
) -> None:
    """ai.embed_many(config={...}) with a string name arrives as request.options."""
    genkit_instance, registry = mock_genkit_instance

    async def fake_embedder_fn(request: EmbedRequest) -> EmbedResponse:
        return EmbedResponse(embeddings=[Embedding(embedding=[1.0]), Embedding(embedding=[2.0])])

    registry.register_action(
        name='plain-embedder',
        kind='embedder',
        fn=fake_embedder_fn,
        metadata=embedder_action_metadata('plain-embedder').metadata,
        description='A fake embedder for testing',
    )

    await genkit_instance.embed_many(embedder='plain-embedder', content=['a', 'b'], config={'dim': 3})

    embed_action = await registry.resolve_action('embedder', 'plain-embedder')
    called_request = embed_action.run.call_args[0][0]
    assert called_request.options == {'dim': 3}


@pytest.mark.asyncio
async def test_embed_with_no_config_sends_none_options(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry],
) -> None:
    """With no ref config, no version, and no config=, the embedder gets options=None like a Dev UI run."""
    genkit_instance, registry = mock_genkit_instance

    async def fake_embedder_fn(request: EmbedRequest) -> EmbedResponse:
        return EmbedResponse(embeddings=[Embedding(embedding=[1.0])])

    registry.register_action(
        name='bare-embedder',
        kind='embedder',
        fn=fake_embedder_fn,
        metadata=embedder_action_metadata('bare-embedder').metadata,
        description='A fake embedder for testing',
    )

    await genkit_instance.embed(embedder='bare-embedder', content='hi')
    await genkit_instance.embed_many(embedder='bare-embedder', content=['hi'])

    embed_action = await registry.resolve_action('embedder', 'bare-embedder')
    assert [call.args[0].options for call in embed_action.run.call_args_list] == [None, None]


@pytest.mark.asyncio
async def test_embed_unknown_embedder_raises_not_found() -> None:
    """ai.embed with an embedder name nobody registered raises GenkitError NOT_FOUND naming it."""
    ai = Genkit()

    with pytest.raises(GenkitError) as exc_info:
        await ai.embed(embedder='nope/missing', content='hi')

    assert exc_info.value.status == 'NOT_FOUND'
    assert 'nope/missing' in str(exc_info.value)


@pytest.mark.asyncio
async def test_embed_many_unknown_embedder_raises_not_found() -> None:
    """ai.embed_many with an embedder name nobody registered raises GenkitError NOT_FOUND naming it."""
    ai = Genkit()

    with pytest.raises(GenkitError) as exc_info:
        await ai.embed_many(embedder='nope/missing', content=['hi'])

    assert exc_info.value.status == 'NOT_FOUND'
    assert 'nope/missing' in str(exc_info.value)


@pytest.mark.asyncio
async def test_embed_document_with_metadata_raises_type_error(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry],
) -> None:
    """A Document carries its own metadata, so metadata= next to it raises instead of being dropped."""
    genkit_instance, _ = mock_genkit_instance

    with pytest.raises(TypeError, match='set it on the Document'):
        await genkit_instance.embed(
            embedder='any-embedder',
            content=Document.from_text('hi'),
            metadata={'source': 'faq'},
        )


@pytest.mark.asyncio
async def test_embed_many_document_with_metadata_raises_type_error(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry],
) -> None:
    """embed_many follows embed: metadata= with a Document in the list raises."""
    genkit_instance, _ = mock_genkit_instance

    with pytest.raises(TypeError, match='set it on the Document'):
        await genkit_instance.embed_many(
            embedder='any-embedder',
            content=[Document.from_text('hi')],
            metadata={'source': 'faq'},
        )


@pytest.mark.asyncio
async def test_embed_string_with_metadata_attaches_it_to_the_document(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry],
) -> None:
    """metadata= with string content still lands on the Document the embedder sees."""
    genkit_instance, registry = mock_genkit_instance

    async def fake_embedder_fn(request: EmbedRequest) -> EmbedResponse:
        return EmbedResponse(embeddings=[Embedding(embedding=[1.0])])

    registry.register_action(
        name='meta-embedder',
        kind='embedder',
        fn=fake_embedder_fn,
        metadata=embedder_action_metadata('meta-embedder').metadata,
        description='An embedder that records its request',
    )

    await genkit_instance.embed(embedder='meta-embedder', content='hi', metadata={'source': 'faq'})

    embed_action = await registry.resolve_action('embedder', 'meta-embedder')
    called_request = embed_action.run.call_args[0][0]
    assert called_request.input == [Document.from_text('hi', {'source': 'faq'})]


@pytest.mark.asyncio
async def test_embed_many_mixed_list_with_metadata_raises_type_error(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry],
) -> None:
    """A Document anywhere in the list rejects metadata=, not just in the first slot."""
    genkit_instance, _ = mock_genkit_instance
    mixed: list[Any] = ['Nut-free kitchen.', Document.from_text('Open until 10pm.')]

    with pytest.raises(TypeError, match='set it on the Document'):
        await genkit_instance.embed_many(embedder='any-embedder', content=mixed, metadata={'source': 'faq'})


@pytest.mark.asyncio
async def test_embed_document_with_empty_metadata_raises_type_error(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry],
) -> None:
    """metadata={} counts as passed: only None means no metadata."""
    genkit_instance, _ = mock_genkit_instance

    with pytest.raises(TypeError, match='set it on the Document'):
        await genkit_instance.embed(embedder='any-embedder', content=Document.from_text('hi'), metadata={})


@pytest.mark.asyncio
async def test_embed_missing_content_raises_type_error(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry],
) -> None:
    """ai.embed without content= fails at the call, before anything runs."""
    genkit_instance, _ = mock_genkit_instance

    with pytest.raises(TypeError, match='content'):
        await genkit_instance.embed(embedder='any-embedder')  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]


@pytest.mark.asyncio
async def test_embed_many_missing_content_raises_type_error(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry],
) -> None:
    """ai.embed_many without content= fails at the call, before anything runs."""
    genkit_instance, _ = mock_genkit_instance

    with pytest.raises(TypeError, match='content'):
        await genkit_instance.embed_many(embedder='any-embedder')  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]


@pytest.mark.asyncio
async def test_embed_many_missing_embedder_raises_type_error(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry],
) -> None:
    """ai.embed_many without embedder= fails at the call, before anything runs."""
    genkit_instance, _ = mock_genkit_instance

    with pytest.raises(TypeError, match='embedder'):
        await genkit_instance.embed_many(content=['hi'])  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]


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
