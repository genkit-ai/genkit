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

"""Tests for the Bedrock typed ref constructors."""

import json
from collections.abc import Callable
from typing import Any, cast, get_args

import pytest
from genkit_amazon_bedrock import Bedrock, BedrockConfig, StabilityImageConfig
from genkit_amazon_bedrock._embedders import EMBEDDER_INFO, KnownBedrockEmbedder
from genkit_amazon_bedrock._model_info import (
    MODEL_CAPABILITIES,
    KnownBedrockText,
    KnownStabilityImage,
)
from genkit_amazon_bedrock._transport import BedrockTransport
from pydantic import BaseModel

from genkit import Genkit, GenkitError
from genkit.embedder import EmbedderRef
from genkit.model import ModelRef
from genkit.plugin_api import ActionKind

RefConstructor = Callable[..., ModelRef[Any]]

CONSTRUCTORS: list[tuple[str, RefConstructor, str, type[BaseModel]]] = [
    ('text', Bedrock.text, 'amazon.nova-lite-v1:0', BedrockConfig),
    ('stability', Bedrock.stability, 'stability.sd3-5-large-v1:0', StabilityImageConfig),
]
IDS = [c[0] for c in CONSTRUCTORS]


class FakeTransport:
    """Stands in for BedrockTransport; records InvokeModel and Converse calls."""

    def __init__(self) -> None:
        self.invoke_calls: list[dict[str, Any]] = []
        self.converse_calls: list[dict[str, Any]] = []

    async def ensure_client(self) -> None:
        return None

    async def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.converse_calls.append(kwargs)
        return {'output': {'message': {'role': 'assistant', 'content': [{'text': 'ok'}]}}, 'stopReason': 'end_turn'}

    async def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        self.invoke_calls.append(kwargs)
        if 'embed' in kwargs['modelId']:
            return {'embedding': [0.1, 0.2, 0.3], 'inputTextTokenCount': 4}
        return {'images': ['img'], 'finish_reasons': [None]}

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(call['body']) for call in self.invoke_calls]


def bedrock_app() -> tuple[Genkit, FakeTransport]:
    plugin = Bedrock(region='us-east-1')
    transport = FakeTransport()
    plugin._transport = cast(BedrockTransport, transport)  # noqa: SLF001
    return Genkit(plugins=[plugin]), transport


@pytest.mark.parametrize(('method', 'constructor', 'model_id', 'schema'), CONSTRUCTORS, ids=IDS)
def test_each_constructor_namespaces_the_id_and_binds_its_class(
    method: str, constructor: RefConstructor, model_id: str, schema: type[BaseModel]
) -> None:
    ref = constructor(model_id)

    assert isinstance(ref, ModelRef)
    assert ref.name == f'bedrock/{model_id}'
    assert ref.config_schema is schema
    assert ref.config is None


@pytest.mark.parametrize(('method', 'constructor', 'model_id', 'schema'), CONSTRUCTORS, ids=IDS)
def test_a_prefixed_id_is_not_stamped_twice(
    method: str, constructor: RefConstructor, model_id: str, schema: type[BaseModel]
) -> None:
    assert constructor(f'bedrock/{model_id}').name == f'bedrock/{model_id}'


@pytest.mark.parametrize(('method', 'constructor', 'model_id', 'schema'), CONSTRUCTORS, ids=IDS)
def test_a_config_passed_at_construction_rides_on_the_ref(
    method: str, constructor: RefConstructor, model_id: str, schema: type[BaseModel]
) -> None:
    config = schema(seed=7) if 'seed' in schema.model_fields else schema(max_output_tokens=256)
    ref = constructor(model_id, config=config)

    assert ref.config == config


@pytest.mark.parametrize(
    ('constructor', 'model_id', 'expected_name'),
    [
        (Bedrock.stability, 'us.stability.stable-image-ultra-v1:1', 'bedrock/us.stability.stable-image-ultra-v1:1'),
        (
            Bedrock.text,
            'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
            'bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0',
        ),
        (
            Bedrock.stability,
            'arn:aws:bedrock:us-west-2::foundation-model/stability.sd3-5-large-v1:0',
            'bedrock/arn:aws:bedrock:us-west-2::foundation-model/stability.sd3-5-large-v1:0',
        ),
        # An opaque application inference profile carries no model ID, so it
        # routes to Converse at runtime, and only text() takes it.
        (
            Bedrock.text,
            'arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123',
            'bedrock/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123',
        ),
        # Unknown text IDs stay usable, as resolve() takes them.
        (Bedrock.text, 'acme.brand-new-model-v1:0', 'bedrock/acme.brand-new-model-v1:0'),
    ],
)
def test_profiles_and_arns_route_like_resolve(constructor: RefConstructor, model_id: str, expected_name: str) -> None:
    assert constructor(model_id).name == expected_name


@pytest.mark.parametrize(
    ('constructor', 'model_id', 'needle'),
    [
        (Bedrock.text, 'stability.sd3-5-large-v1:0', 'use Bedrock.stability()'),
        (Bedrock.embedder, 'us.stability.stable-image-core-v1:1', 'use Bedrock.stability()'),
        # Retired on Bedrock and untyped: the hint names the raw ID and extra.
        (Bedrock.text, 'amazon.nova-canvas-v1:0', "pass 'bedrock/amazon.nova-canvas-v1:0'"),
        (Bedrock.stability, 'amazon.titan-image-generator-v2:0', "config['extra']"),
        (Bedrock.stability, 'amazon.nova-lite-v1:0', 'text models take Bedrock.text()'),
        (Bedrock.text, 'stability.stable-image-inpaint-v1:0', 'image editing/upscale service'),
        (Bedrock.stability, 'stability.stable-image-inpaint-v1:0', 'image editing/upscale service'),
        (Bedrock.stability, 'us.stability.stable-outpaint-v1:0', 'image editing/upscale service'),
        (Bedrock.embedder, 'stability.stable-creative-upscale-v1:0', 'image editing/upscale service'),
        (
            Bedrock.text,
            'arn:aws:bedrock:us-west-2::foundation-model/stability.stable-style-transfer-v1:0',
            'image editing/upscale service',
        ),
        (Bedrock.text, 'amazon.titan-embed-text-v2:0', 'use Bedrock.embedder()'),
        (Bedrock.stability, 'cohere.embed-english-v3', 'use Bedrock.embedder()'),
        (Bedrock.embedder, 'amazon.nova-lite-v1:0', 'use Bedrock.text()'),
        (Bedrock.embedder, 'amazon.nova-canvas-v1:0', 'retired Amazon image model'),
        (Bedrock.embedder, 'cohere.rerank-v3-5:0', 'no rerank action'),
        (Bedrock.stability, 'cohere.rerank-v3-5:0', 'no rerank action'),
        (
            Bedrock.stability,
            'arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc',
            'Bedrock.text()',
        ),
    ],
)
def test_an_id_from_another_family_is_refused(constructor: RefConstructor, model_id: str, needle: str) -> None:
    with pytest.raises(GenkitError) as exc_info:
        constructor(model_id)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert needle in str(exc_info.value)


@pytest.mark.parametrize('name', [None, 123, '', 'bedrock/'])
def test_a_missing_or_non_string_name_is_refused(name: object) -> None:
    with pytest.raises(GenkitError) as exc_info:
        Bedrock.stability(name)  # type: ignore[arg-type]
    assert exc_info.value.status == 'INVALID_ARGUMENT'


@pytest.mark.parametrize(('method', 'constructor', 'model_id', 'schema'), CONSTRUCTORS, ids=IDS)
@pytest.mark.asyncio
async def test_a_ref_agrees_with_the_registered_action(
    method: str, constructor: RefConstructor, model_id: str, schema: type[BaseModel]
) -> None:
    # Refs and runtime registration share routing, so they can't disagree.
    action = await Bedrock(region='us-east-1').resolve(ActionKind.MODEL, model_id)

    assert action is not None
    assert action.config_schema is constructor(model_id).config_schema


@pytest.mark.parametrize(
    ('literal', 'constructor'),
    [
        (KnownBedrockText, Bedrock.text),
        (KnownStabilityImage, Bedrock.stability),
    ],
)
def test_every_known_id_is_accepted_by_its_constructor(literal: object, constructor: RefConstructor) -> None:
    for model_id in get_args(literal):
        assert constructor(model_id).name == f'bedrock/{model_id}'


def test_known_text_ids_match_the_catalog() -> None:
    assert set(get_args(KnownBedrockText)) == set(MODEL_CAPABILITIES)


def test_known_embedder_ids_match_the_catalog() -> None:
    assert set(get_args(KnownBedrockEmbedder)) == set(EMBEDDER_INFO)


@pytest.mark.parametrize(
    ('model_id', 'expected_name'),
    [
        ('amazon.titan-embed-text-v2:0', 'bedrock/amazon.titan-embed-text-v2:0'),
        ('bedrock/cohere.embed-english-v3', 'bedrock/cohere.embed-english-v3'),
        ('us.cohere.embed-v4:0', 'bedrock/us.cohere.embed-v4:0'),
    ],
)
def test_embedder_returns_an_embedder_ref(model_id: str, expected_name: str) -> None:
    # An EmbedderRef, never a ModelRef: an embedding ID can't reach generate().
    ref = Bedrock.embedder(model_id)

    assert isinstance(ref, EmbedderRef)
    assert ref.name == expected_name
    assert ref.config is None
    assert Bedrock(region='us-east-1').embedder(model_id) == ref


def test_every_known_embedder_is_accepted() -> None:
    for model_id in get_args(KnownBedrockEmbedder):
        assert Bedrock.embedder(model_id).name == f'bedrock/{model_id}'


@pytest.mark.asyncio
async def test_embed_with_an_embedder_ref_reaches_invoke_model() -> None:
    ai, transport = bedrock_app()

    embeddings = await ai.embed(embedder=Bedrock.embedder('amazon.titan-embed-text-v2:0'), content='Smoked salmon')

    assert embeddings[0].embedding == [0.1, 0.2, 0.3]
    assert transport.invoke_calls[0]['modelId'] == 'amazon.titan-embed-text-v2:0'
    assert json.loads(transport.invoke_calls[0]['body'])['inputText'] == 'Smoked salmon'


@pytest.mark.asyncio
async def test_generate_with_a_retired_amazon_id_sends_extra_on_invoke_model() -> None:
    # No typed ref; the raw ID still routes to InvokeModel and extra is the body.
    ai, transport = bedrock_app()

    response = await ai.generate(
        model='bedrock/amazon.nova-canvas-v1:0',
        prompt='A dim sum spread on a marble table.',
        config={'extra': {'imageGenerationConfig': {'width': 512, 'height': 512}}},
    )

    assert response.error is None
    assert transport.invoke_calls[0]['modelId'] == 'amazon.nova-canvas-v1:0'
    assert transport.bodies() == [
        {
            'taskType': 'TEXT_IMAGE',
            'textToImageParams': {'text': 'A dim sum spread on a marble table.'},
            'imageGenerationConfig': {'width': 512, 'height': 512},
        }
    ]


@pytest.mark.asyncio
async def test_a_refs_config_is_applied_under_the_call_config() -> None:
    ai, transport = bedrock_app()
    sd = Bedrock.stability('stability.sd3-5-large-v1:0', config=StabilityImageConfig(seed=42, output_format='jpeg'))

    response = await ai.generate(model=sd, prompt='A food truck at dusk.', config={'aspect_ratio': '16:9'})

    assert response.error is None
    # The ref's seed and format stay; the call adds aspect_ratio.
    assert transport.bodies() == [
        {'prompt': 'A food truck at dusk.', 'output_format': 'jpeg', 'seed': 42, 'aspect_ratio': '16:9'}
    ]
    assert response.media is not None
    assert response.media.content_type == 'image/jpeg'


@pytest.mark.asyncio
async def test_a_wrong_family_config_on_a_ref_fails_before_the_call() -> None:
    ai, transport = bedrock_app()
    # Type checkers already reject this call; Any checks the runtime guard too.
    wrong_family: Any = BedrockConfig(max_output_tokens=64)

    with pytest.raises(GenkitError) as exc_info:
        await ai.generate(
            model=Bedrock.stability('stability.sd3-5-large-v1:0'),
            prompt='A bowl of ramen.',
            config=wrong_family,
        )

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert transport.invoke_calls == []


@pytest.mark.asyncio
async def test_generate_with_a_text_ref_reaches_converse_with_its_config() -> None:
    ai, transport = bedrock_app()
    nova = Bedrock.text('amazon.nova-lite-v1:0', config=BedrockConfig(max_output_tokens=64))

    response = await ai.generate(model=nova, prompt='Suggest a soup.')

    assert response.text == 'ok'
    assert transport.converse_calls[0]['modelId'] == 'amazon.nova-lite-v1:0'
    assert transport.converse_calls[0]['inferenceConfig']['maxTokens'] == 64
