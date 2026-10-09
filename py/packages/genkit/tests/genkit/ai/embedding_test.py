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
def test_embedder_ref_with_non_dict_config_raises(build: Callable[[], EmbedderRef]) -> None:
    """A non-dict config raises instead of being silently dropped by ai.embed."""
    with pytest.raises(GenkitError, match='config must be a mapping when config_schema is not set, got str'):
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
async def test_embed_with_no_config_sends_empty_dict_options(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry],
) -> None:
    """With no ref config, no version, and no config=, the embedder gets options={}."""
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
    assert [call.args[0].options for call in embed_action.run.call_args_list] == [{}, {}]


def test_embed_request_options_none_or_missing_becomes_empty_dict() -> None:
    """EmbedRequest built in code or parsed from the wire (Dev UI sends options: null) always carries a dict."""
    docs = [Document.from_text('hi')]

    assert EmbedRequest(input=docs).options == {}
    assert EmbedRequest(input=docs, options=None).options == {}  # type: ignore[arg-type] - wire null
    assert EmbedRequest.model_validate({'input': [{'content': [{'text': 'hi'}]}], 'options': None}).options == {}


@pytest.mark.parametrize('bad_options', [[], [('dimensions', 256)], 'x'], ids=['empty_list', 'pairs', 'str'])
def test_embed_request_non_mapping_options_raises(bad_options: object) -> None:
    """EmbedRequest.options must be a mapping, same as ModelRequest.config; a list or str fails validation."""
    with pytest.raises(ValidationError, match='options must be a mapping'):
        EmbedRequest(input=[Document.from_text('hi')], options=bad_options)  # type: ignore[arg-type] - bad wire value


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_config', [[], [('dimensions', 256)], 'x'], ids=['empty_list', 'pairs', 'str'])
async def test_embed_with_non_mapping_config_raises_invalid_argument(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry], bad_config: object
) -> None:
    """ai.embed config= must be a dict or a BaseModel, same as generate; a list or str raises."""
    genkit_instance, registry = mock_genkit_instance

    async def fake_embedder_fn(request: EmbedRequest) -> EmbedResponse:
        return EmbedResponse(embeddings=[Embedding(embedding=[0.0])])

    registry.register_action(
        name='strict-embedder',
        kind='embedder',
        fn=fake_embedder_fn,
        metadata=embedder_action_metadata('strict-embedder').metadata,
        description='A fake embedder for testing',
    )

    with pytest.raises(GenkitError) as exc_info:
        await genkit_instance.embed(embedder='strict-embedder', content='hi', config=bad_config)  # type: ignore[arg-type]

    assert exc_info.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_embed_with_basemodel_config_sends_dict_options(
    mock_genkit_instance: tuple[Genkit, MockGenkitRegistry],
) -> None:
    """A BaseModel config= is dumped, so the untyped embedder still gets a dict."""

    class CrmEmbedConfig(BaseModel):
        dimensions: int = 768

    genkit_instance, registry = mock_genkit_instance

    async def fake_embedder_fn(request: EmbedRequest) -> EmbedResponse:
        return EmbedResponse(embeddings=[Embedding(embedding=[0.0])])

    registry.register_action(
        name='crm-embedder',
        kind='embedder',
        fn=fake_embedder_fn,
        metadata=embedder_action_metadata('crm-embedder').metadata,
        description='A fake embedder for testing',
    )

    await genkit_instance.embed(embedder='crm-embedder', content='hi', config=CrmEmbedConfig(dimensions=256))

    embed_action = await registry.resolve_action('embedder', 'crm-embedder')
    assert embed_action.run.call_args.args[0].options == {'dimensions': 256}


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


class _CrmEmbedConfig(BaseModel):
    dimensions: int = 768
    task_type: str = 'RETRIEVAL_DOCUMENT'


def _recording_embedder(ai: Genkit, seen: list[dict[str, Any]]) -> None:
    async def crm_embedder(request: EmbedRequest) -> EmbedResponse:
        seen.append(request.options)
        return EmbedResponse(embeddings=[Embedding(embedding=[0.0]) for _ in request.input])

    ai.define_embedder(name='crm-embedder', fn=crm_embedder)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('config', 'expected'),
    [
        (_CrmEmbedConfig(), {}),
        (_CrmEmbedConfig(task_type='QUESTION_ANSWERING'), {'task_type': 'QUESTION_ANSWERING'}),
    ],
)
async def test_embed_with_basemodel_config_sends_only_set_fields(
    config: _CrmEmbedConfig, expected: dict[str, object]
) -> None:
    """An untyped embedder gets the fields the caller set, same as an untyped model fn's config."""
    ai = Genkit()
    seen: list[dict[str, Any]] = []
    _recording_embedder(ai, seen)

    await ai.embed(embedder='crm-embedder', content='hi', config=config)

    assert seen == [expected]


@pytest.mark.asyncio
async def test_embed_ref_config_survives_unset_basemodel_field() -> None:
    """ref < call per field: a BaseModel call config doesn't clobber ref keys it left unset."""
    ai = Genkit()
    seen: list[dict[str, Any]] = []
    _recording_embedder(ai, seen)
    ref = create_embedder_ref('crm-embedder', config={'dimensions': 256})

    await ai.embed(embedder=ref, content='hi', config=_CrmEmbedConfig(task_type='QUESTION_ANSWERING'))

    assert seen == [{'dimensions': 256, 'task_type': 'QUESTION_ANSWERING'}]


def test_embed_request_basemodel_options_becomes_dict_of_set_fields() -> None:
    """EmbedRequest takes a BaseModel the same way ai.embed(config=...) does."""
    request = EmbedRequest(input=[], options=_CrmEmbedConfig(dimensions=256))  # type: ignore[arg-type]

    assert request.options == {'dimensions': 256}


def test_embed_request_options_schema_is_an_object() -> None:
    """The embedder input schema says options is an object, not Any."""
    schema = to_json_schema(EmbedRequest)

    assert schema['properties']['options']['type'] == 'object'
