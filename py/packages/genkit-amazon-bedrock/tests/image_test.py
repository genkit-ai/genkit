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

"""Tests for Bedrock image generation (no AWS involved)."""

import json
from typing import Any

import pytest
from botocore.exceptions import ClientError, NoCredentialsError
from genkit_amazon_bedrock import StabilityImageConfig
from genkit_amazon_bedrock._config import AmazonImageConfig
from genkit_amazon_bedrock._image import BedrockImageModel, image_config_class, is_image_model
from pydantic import ConfigDict

from genkit import FinishReason, GenkitError, Media, Message, ModelResponse, Part, Role
from genkit.model import ModelConfig, ModelContext, ModelRequest
from genkit.plugin_api import to_json_schema

TITAN_IMAGE = 'amazon.titan-image-generator-v1'
NOVA_CANVAS = 'amazon.nova-canvas-v1:0'
SD3 = 'stability.sd3-5-large-v1:0'
STABLE_CORE = 'stability.stable-image-core-v1:1'
OPAQUE_ARN = 'arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123opaque'

PROMPT = 'a serene mountain landscape'

PNG_B64 = 'iVBORw0KGgoAAAANSUhEUg=='
PNG_DATA_URL = f'data:image/png;base64,{PNG_B64}'


class FakeInvokeTransport:
    """Stands in for BedrockTransport; records the InvokeModel kwargs."""

    def __init__(self, response: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.response = response if response is not None else {}
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def bodies(self) -> list[dict[str, Any]]:
        """The parsed request bodies, in call order."""
        return [json.loads(call['body']) for call in self.calls]

    async def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class ForbiddenTransport:
    """Fails the test if the model reaches the wire at all."""

    async def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError(f'no InvokeModel call expected, got {kwargs}')


def text_part(text: str) -> Part:
    return Part.from_text(text)


def media_part(url: str = PNG_DATA_URL) -> Part:
    return Part.from_media(url, content_type='image/png')


def user_message(*parts: Part) -> Message:
    return Message(role=Role.USER, content=list(parts))


def image_request(prompt: str = PROMPT, config: Any = None) -> ModelRequest:  # noqa: ANN401
    request = ModelRequest(messages=[user_message(text_part(prompt))])
    # Assigned rather than passed to the constructor: ModelRequest coerces a
    # mapping into GenerationCommonConfig, and these tests pin what the plugin
    # does with the config value it is actually handed.
    request.config = config
    return request


def amazon_response(*images: str, error: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {'images': list(images)}
    if error is not None:
        payload['error'] = error
    return payload


def stability_response(*images: str, finish_reasons: list[Any] | None = None) -> dict[str, Any]:
    # seeds ride along on every real response and are ignored.
    payload: dict[str, Any] = {'images': list(images), 'seeds': [0] * len(images)}
    if finish_reasons is not None:
        payload['finish_reasons'] = finish_reasons
    return payload


async def generate(
    model_id: str,
    transport: Any,  # noqa: ANN401
    request: ModelRequest | None = None,
    ctx: ModelContext | None = None,
) -> ModelResponse:
    model = BedrockImageModel(model_id=model_id, transport=transport)
    return await model.generate(request if request is not None else image_request(), ctx)


def media(response: ModelResponse) -> list[Media]:
    """The media payloads of a response, in order."""
    assert response.message is not None
    return [part.media for part in response.message.content if part.media is not None]


# ---- Classification ---------------------------------------------------------


@pytest.mark.parametrize(
    ('model_id', 'routable'),
    [
        ('amazon.titan-image-generator-v1', True),
        ('amazon.titan-image-generator-v2:0', True),
        ('amazon.nova-canvas-v1:0', True),
        ('stability.sd3-5-large-v1:0', True),
        ('stability.sd3-large-v1:0', True),
        ('stability.stable-image-core-v1:1', True),
        ('stability.stable-image-ultra-v1:1', True),
        ('us.amazon.nova-canvas-v1:0', True),
        ('us.stability.sd3-5-large-v1:0', True),
        # Bedrock's own IDs are lowercase, but a custom-model ARN carries a
        # caller-chosen resource name, so matching is case-insensitive.
        ('AMAZON.NOVA-CANVAS-V1:0', True),
        ('US.stability.SD3-5-large-v1:0', True),
        ('arn:aws:bedrock:us-east-1:123456789012:custom-model/amazon.nova-canvas-v1:0/Nova-Canvas-Tuned', True),
        # An embedder: 'titan-embed-image' is not 'titan-image'.
        ('amazon.titan-embed-image-v1', False),
        # Legacy SDXL (text_prompts/artifacts) is deliberately not ported.
        ('stability.stable-diffusion-xl-v1', False),
        # A bare 'stable-image' pattern would swallow these editing services,
        # which are not text-to-image and take different request bodies. Some
        # are inference-profile-only in a given region, hence the us. form.
        ('stability.stable-image-inpaint-v1:0', False),
        ('us.stability.stable-image-erase-object-v1:0', False),
        ('stability.stable-image-remove-background-v1:0', False),
        ('stability.stable-image-search-replace-v1:0', False),
        ('stability.stable-image-style-guide-v1:0', False),
        ('stability.stable-image-control-sketch-v1:0', False),
        ('amazon.nova-lite-v1:0', False),
        ('anthropic.claude-sonnet-4-5-20250929-v1:0', False),
    ],
)
def test_image_model_routing(model_id: str, routable: bool) -> None:
    assert is_image_model(model_id) is routable


@pytest.mark.asyncio
async def test_profile_prefixed_id_routes_but_the_wire_keeps_it() -> None:
    transport = FakeInvokeTransport(amazon_response('nova-image'))
    await generate(f'us.{NOVA_CANVAS}', transport)

    call = transport.calls[0]
    assert call['modelId'] == f'us.{NOVA_CANVAS}'
    assert call['contentType'] == 'application/json'
    assert call['accept'] == 'application/json'


@pytest.mark.parametrize(
    'model_id',
    ['stability.stable-diffusion-xl-v1', OPAQUE_ARN, 'stability.stable-image-inpaint-v1:0'],
)
@pytest.mark.asyncio
async def test_unsupported_model_fails_before_any_call(model_id: str) -> None:
    with pytest.raises(GenkitError, match='unsupported image generation model') as excinfo:
        await generate(model_id, ForbiddenTransport())
    assert excinfo.value.status == 'UNIMPLEMENTED'
    assert model_id in str(excinfo.value)


# ---- Prompt extraction ------------------------------------------------------


@pytest.mark.asyncio
async def test_the_last_user_message_wins_and_its_text_parts_concatenate() -> None:
    transport = FakeInvokeTransport(amazon_response('titan-image'))
    request = ModelRequest(
        messages=[
            user_message(text_part('old prompt')),
            Message(role=Role.MODEL, content=[text_part('ignore model')]),
            user_message(text_part('new prompt'), text_part(' detail')),
        ]
    )
    await generate(TITAN_IMAGE, transport, request)

    # Concatenated with no separator.
    assert transport.bodies()[0]['textToImageParams'] == {'text': 'new prompt detail'}


@pytest.mark.asyncio
async def test_a_media_only_user_message_falls_back_to_an_older_one() -> None:
    transport = FakeInvokeTransport(amazon_response('titan-image'))
    request = ModelRequest(messages=[user_message(text_part('older prompt')), user_message(media_part())])
    await generate(TITAN_IMAGE, transport, request)

    assert transport.bodies()[0]['textToImageParams'] == {'text': 'older prompt'}


@pytest.mark.asyncio
async def test_missing_prompt_fails_before_any_call() -> None:
    request = ModelRequest(
        messages=[
            Message(role=Role.SYSTEM, content=[text_part('system text')]),
            user_message(media_part()),
        ]
    )
    with pytest.raises(GenkitError, match='no text prompt') as excinfo:
        await generate(TITAN_IMAGE, ForbiddenTransport(), request)
    assert excinfo.value.status == 'INVALID_ARGUMENT'


# ---- Retired Amazon request bodies ------------------------------------------
# Titan Image and Nova Canvas are retired on Bedrock and get no typed config.
# The raw ID still routes, and extra carries every body field past the prompt.


@pytest.mark.parametrize('model_id', [TITAN_IMAGE, NOVA_CANVAS])
@pytest.mark.asyncio
async def test_retired_amazon_models_send_only_the_prompt_fields(model_id: str) -> None:
    transport = FakeInvokeTransport(amazon_response('amazon-image'))
    await generate(model_id, transport, image_request('a mountain'))

    # Exact equality: no imageGenerationConfig unless extra sends one.
    assert transport.bodies() == [{'taskType': 'TEXT_IMAGE', 'textToImageParams': {'text': 'a mountain'}}]


@pytest.mark.parametrize('model_id', [TITAN_IMAGE, NOVA_CANVAS, f'us.{NOVA_CANVAS}'])
@pytest.mark.asyncio
async def test_retired_amazon_models_take_body_fields_through_extra(model_id: str) -> None:
    transport = FakeInvokeTransport(amazon_response('amazon-image'))
    extra: dict[str, Any] = {
        'textToImageParams': {'negativeText': 'blurry'},
        'imageGenerationConfig': {'numberOfImages': 2, 'seed': 7},
    }
    await generate(model_id, transport, image_request(config={'extra': extra}))

    assert transport.bodies() == [
        {
            'taskType': 'TEXT_IMAGE',
            'textToImageParams': {'text': PROMPT, 'negativeText': 'blurry'},
            'imageGenerationConfig': {'numberOfImages': 2, 'seed': 7},
        }
    ]


@pytest.mark.parametrize('key', ['imageGenerationConfig', 'image_generation_config', 'seed', 'temperature'])
@pytest.mark.parametrize('model_id', [TITAN_IMAGE, NOVA_CANVAS])
@pytest.mark.asyncio
async def test_a_top_level_key_on_a_retired_amazon_model_is_rejected(model_id: str, key: str) -> None:
    # Only extra is declared; ai.generate's own check names extra in the message.
    with pytest.raises(GenkitError, match='invalid config') as excinfo:
        await generate(model_id, ForbiddenTransport(), image_request(config={key: 7}))

    assert excinfo.value.status == 'INVALID_ARGUMENT'
    assert key in str(excinfo.value)


@pytest.mark.parametrize(
    ('model_id', 'config'),
    [
        (SD3, {'aspectRatio': '16:9', 'style_preset': 'photographic'}),
        (SD3, {'width': 512}),
        (SD3, {'numberOfImages': 2}),
        # Genkit's generic knobs: the image configs extend GenkitConfig, not ModelConfig.
        (SD3, {'temperature': 0.9}),
        (STABLE_CORE, {'topP': 0.5}),
        (SD3, {'max_output_tokens': 100}),
        (SD3, {'stop_sequences': ['x']}),
    ],
    ids=[
        'stability_unlisted',
        'stability_amazon_key',
        'stability_amazon_wire_key',
        'stability_temperature',
        'stability_top_p',
        'stability_max_output_tokens',
        'stability_stop_sequences',
    ],
)
@pytest.mark.asyncio
async def test_an_unknown_key_is_rejected_before_any_call(model_id: str, config: dict[str, Any]) -> None:
    with pytest.raises(GenkitError, match='invalid config') as excinfo:
        await generate(model_id, ForbiddenTransport(), image_request(config=config))
    assert excinfo.value.status == 'INVALID_ARGUMENT'


@pytest.mark.parametrize(
    ('model_id', 'config'),
    [
        (SD3, {'aspect_ratio': '4:3'}),
        (SD3, {'output_format': 'gif'}),
        (SD3, {'seed': -1}),
        (STABLE_CORE, {'seed': 4294967296}),
    ],
)
@pytest.mark.asyncio
async def test_an_out_of_range_value_is_rejected_before_any_call(model_id: str, config: dict[str, Any]) -> None:
    with pytest.raises(GenkitError, match='invalid config') as excinfo:
        await generate(model_id, ForbiddenTransport(), image_request(config=config))
    assert excinfo.value.status == 'INVALID_ARGUMENT'


# ---- Stability request bodies -----------------------------------------------


@pytest.mark.asyncio
async def test_stability_sends_the_documented_defaults() -> None:
    transport = FakeInvokeTransport(stability_response('modern-image', finish_reasons=['SUCCESS']))
    await generate(SD3, transport, image_request('a reef'))

    # Exact equality: the legacy text_prompts array must not be sent.
    assert transport.bodies() == [{'prompt': 'a reef', 'output_format': 'png'}]


@pytest.mark.asyncio
async def test_stability_accepts_core_and_ultra_max_seed() -> None:
    # Core and Ultra allow up to 4294967295 (SD 3.5 caps one lower at 4294967294).
    transport = FakeInvokeTransport(stability_response('modern-image', finish_reasons=['SUCCESS']))
    await generate(STABLE_CORE, transport, image_request('a reef', config=StabilityImageConfig(seed=4294967295)))

    assert transport.bodies() == [{'prompt': 'a reef', 'output_format': 'png', 'seed': 4294967295}]


@pytest.mark.parametrize(
    'config',
    [
        StabilityImageConfig(aspect_ratio='16:9', seed=42, negative_prompt='blurry', output_format='jpeg'),
        {'aspect_ratio': '16:9', 'seed': 42, 'negative_prompt': 'blurry', 'output_format': 'jpeg'},
        {'aspectRatio': '16:9', 'seed': 42, 'negativePrompt': 'blurry', 'outputFormat': 'jpeg'},
    ],
    ids=['typed', 'snake_dict', 'camel_dict'],
)
@pytest.mark.asyncio
async def test_stability_sends_declared_fields_at_the_top_level(config: Any) -> None:  # noqa: ANN401
    transport = FakeInvokeTransport(stability_response('modern-image', finish_reasons=['SUCCESS']))
    await generate(STABLE_CORE, transport, image_request('a reef', config=config))

    # Stability's wire names are snake_case, whatever spelling the caller used.
    assert transport.bodies() == [
        {'prompt': 'a reef', 'output_format': 'jpeg', 'aspect_ratio': '16:9', 'seed': 42, 'negative_prompt': 'blurry'}
    ]


@pytest.mark.asyncio
async def test_stability_mime_follows_the_requested_output_format() -> None:
    transport = FakeInvokeTransport(stability_response('modern-image', finish_reasons=['SUCCESS']))
    response = await generate(SD3, transport, image_request(config={'output_format': 'jpeg'}))

    assert [item.content_type for item in media(response)] == ['image/jpeg']
    assert [item.url for item in media(response)] == ['data:image/jpeg;base64,modern-image']


@pytest.mark.asyncio
async def test_stability_mime_lowercases_while_the_wire_keeps_the_casing() -> None:
    # The declared field takes lowercase only; extra is sent as-is.
    transport = FakeInvokeTransport(stability_response('modern-image', finish_reasons=['SUCCESS']))
    response = await generate(SD3, transport, image_request(config={'extra': {'output_format': 'JPEG'}}))

    assert transport.bodies()[0]['output_format'] == 'JPEG'
    assert [item.content_type for item in media(response)] == ['image/jpeg']


# ---- Stability finish reasons -----------------------------------------------


@pytest.mark.parametrize('images', [['blocked-image'], []])
@pytest.mark.asyncio
async def test_a_non_success_finish_reason_is_internal(images: list[str]) -> None:
    # Checked before the images: a filtered result still returns a slot, and an
    # empty array must still report the reason rather than "no images".
    transport = FakeInvokeTransport({'images': images, 'finish_reasons': ['CONTENT_FILTERED']})
    with pytest.raises(GenkitError, match='CONTENT_FILTERED') as excinfo:
        await generate(SD3, transport)
    assert excinfo.value.status == 'INTERNAL'


@pytest.mark.parametrize('reason', [None, 'SUCCESS', ''])
@pytest.mark.asyncio
async def test_success_shaped_finish_reasons_pass(reason: Any) -> None:  # noqa: ANN401
    # JSON null is the success value here; '' is what a terse service sends.
    transport = FakeInvokeTransport(stability_response('modern-image', finish_reasons=[reason]))
    assert len(media(await generate(SD3, transport))) == 1


# ---- Responses --------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_carries_one_media_part_per_image() -> None:
    transport = FakeInvokeTransport(amazon_response('titan-image-1', 'titan-image-2'))
    request = image_request()
    response = await generate(TITAN_IMAGE, transport, request)

    assert [item.url for item in media(response)] == [
        'data:image/png;base64,titan-image-1',
        'data:image/png;base64,titan-image-2',
    ]
    assert [item.content_type for item in media(response)] == ['image/png', 'image/png']
    assert response.finish_reason is FinishReason.STOP
    assert response.message is not None and response.message.role is Role.MODEL
    # Genkit backfills an empty usage object; image generation reports no tokens.
    assert response.usage is not None
    assert (response.usage.input_tokens, response.usage.output_tokens, response.usage.total_tokens) == (
        None,
        None,
        None,
    )
    assert response.request == request


@pytest.mark.asyncio
async def test_a_streaming_context_receives_no_chunks() -> None:
    transport = FakeInvokeTransport(amazon_response('titan-image'))
    chunks: list[Any] = []
    response = await generate(TITAN_IMAGE, transport, ctx=ModelContext(streaming_callback=chunks.append))

    assert chunks == []
    assert len(media(response)) == 1


@pytest.mark.parametrize(
    ('model_id', 'payload'),
    [
        (TITAN_IMAGE, {'images': []}),
        (TITAN_IMAGE, {}),
        (NOVA_CANVAS, {'images': None}),
        (SD3, {'images': [], 'finish_reasons': ['SUCCESS']}),
        (SD3, {}),
    ],
)
@pytest.mark.asyncio
async def test_no_usable_image_is_internal(model_id: str, payload: dict[str, Any]) -> None:
    with pytest.raises(GenkitError, match='no images generated') as excinfo:
        await generate(model_id, FakeInvokeTransport(payload))
    assert excinfo.value.status == 'INTERNAL'


@pytest.mark.asyncio
async def test_an_in_band_error_surfaces_the_service_text() -> None:
    # Amazon reports a blocked prompt on an HTTP 200; "no images generated"
    # would throw away the only usable diagnostic.
    transport = FakeInvokeTransport(amazon_response(error='prompt blocked by content filter'))
    with pytest.raises(GenkitError, match='prompt blocked by content filter') as excinfo:
        await generate(NOVA_CANVAS, transport)
    assert excinfo.value.status == 'INTERNAL'


@pytest.mark.asyncio
async def test_images_alongside_an_error_are_partial_success() -> None:
    # Nova Canvas drops individually filtered images this way; the rest stand.
    transport = FakeInvokeTransport(amazon_response('nova-image', error='1 image was filtered'))
    response = await generate(NOVA_CANVAS, transport)

    assert [item.url for item in media(response)] == ['data:image/png;base64,nova-image']


@pytest.mark.asyncio
async def test_blank_base64_strings_are_skipped() -> None:
    transport = FakeInvokeTransport(amazon_response('', 'titan-image', ''))
    response = await generate(TITAN_IMAGE, transport)

    assert [item.url for item in media(response)] == ['data:image/png;base64,titan-image']


@pytest.mark.parametrize('model_id', [TITAN_IMAGE, SD3])
@pytest.mark.asyncio
async def test_an_all_blank_image_array_is_internal(model_id: str) -> None:
    with pytest.raises(GenkitError, match='no images generated') as excinfo:
        await generate(model_id, FakeInvokeTransport({'images': ['', '']}))
    assert excinfo.value.status == 'INTERNAL'


# ---- Config handling --------------------------------------------------------


@pytest.mark.parametrize(
    ('model_id', 'config_class', 'overrides'),
    [
        (SD3, StabilityImageConfig, {'seed': 7, 'extra': {'mode': 'text-to-image'}}),
        (NOVA_CANVAS, AmazonImageConfig, {'extra': {'textToImageParams': {'negativeText': 'blurry'}}}),
    ],
)
@pytest.mark.asyncio
async def test_a_pydantic_config_and_the_equivalent_dict_agree(
    model_id: str, config_class: type[StabilityImageConfig] | type[AmazonImageConfig], overrides: dict[str, Any]
) -> None:
    typed = FakeInvokeTransport(stability_response('image', finish_reasons=['SUCCESS']))
    plain = FakeInvokeTransport(stability_response('image', finish_reasons=['SUCCESS']))

    await generate(model_id, typed, image_request(config=config_class.model_validate(overrides)))
    await generate(model_id, plain, image_request(config=overrides))

    assert typed.bodies() == plain.bodies()


@pytest.mark.asyncio
async def test_extra_overrides_a_colliding_typed_field() -> None:
    # A colliding leaf in extra wins, as in #6644; siblings keep the typed values.
    transport = FakeInvokeTransport(stability_response('modern-image', finish_reasons=['SUCCESS']))
    config = StabilityImageConfig(seed=7, aspect_ratio='16:9', extra={'seed': 9})
    await generate(SD3, transport, image_request(config=config))

    assert transport.bodies() == [{'prompt': PROMPT, 'output_format': 'png', 'seed': 9, 'aspect_ratio': '16:9'}]


@pytest.mark.asyncio
async def test_a_generic_config_model_and_the_equivalent_dict_agree() -> None:
    # A generic config object and a raw mapping are both legal inputs, so the
    # two paths must build the same body.
    class LooseConfig(ModelConfig):
        model_config = ConfigDict(extra='allow')

    typed = FakeInvokeTransport(stability_response('modern-image', finish_reasons=['SUCCESS']))
    plain = FakeInvokeTransport(stability_response('modern-image', finish_reasons=['SUCCESS']))
    overrides: dict[str, Any] = {'seed': 42}

    await generate(SD3, typed, image_request(config=LooseConfig.model_validate(overrides)))
    await generate(SD3, plain, image_request(config=overrides))

    assert typed.bodies() == plain.bodies()
    assert typed.bodies() == [{'prompt': PROMPT, 'output_format': 'png', 'seed': 42}]


@pytest.mark.asyncio
async def test_a_generic_config_model_carrying_a_genkit_knob_is_rejected() -> None:
    with pytest.raises(GenkitError, match='invalid config') as excinfo:
        await generate(SD3, ForbiddenTransport(), image_request(config=ModelConfig(temperature=0.9)))
    assert excinfo.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_a_non_mapping_config_is_rejected_before_any_call() -> None:
    with pytest.raises(GenkitError, match='unexpected config type') as excinfo:
        await generate(TITAN_IMAGE, ForbiddenTransport(), image_request(config=42))
    assert excinfo.value.status == 'INVALID_ARGUMENT'


# ---- Error mapping ----------------------------------------------------------


@pytest.mark.asyncio
async def test_client_errors_map_to_genkit_statuses() -> None:
    error = ClientError({'Error': {'Code': 'ThrottlingException', 'Message': 'boom'}}, 'InvokeModel')
    with pytest.raises(GenkitError, match='invoke model failed') as excinfo:
        await generate(TITAN_IMAGE, FakeInvokeTransport(error=error))
    assert excinfo.value.status == 'RESOURCE_EXHAUSTED'


@pytest.mark.asyncio
async def test_botocore_errors_map_to_genkit_statuses() -> None:
    with pytest.raises(GenkitError, match='invoke model failed') as excinfo:
        await generate(SD3, FakeInvokeTransport(error=NoCredentialsError()))
    assert excinfo.value.status == 'UNAUTHENTICATED'


# ---- Config schema ----------------------------------------------------------


@pytest.mark.parametrize(
    ('model_id', 'config_class'),
    [
        (TITAN_IMAGE, AmazonImageConfig),
        (NOVA_CANVAS, AmazonImageConfig),
        (f'us.{NOVA_CANVAS}', AmazonImageConfig),
        (SD3, StabilityImageConfig),
        (STABLE_CORE, StabilityImageConfig),
        ('amazon.nova-lite-v1:0', None),
    ],
)
def test_image_config_class_follows_the_family(model_id: str, config_class: type | None) -> None:
    assert image_config_class(model_id) is config_class


@pytest.mark.parametrize(
    ('config_class', 'expected'),
    [
        # Retired models: extra only.
        (AmazonImageConfig, set()),
        (StabilityImageConfig, {'aspectRatio', 'seed', 'negativePrompt', 'outputFormat'}),
    ],
)
def test_each_image_schema_is_strict_and_declares_its_wire_fields(config_class: type, expected: set[str]) -> None:
    # Strict, so a typo raises instead of being dropped; the Dev UI renders these as a form.
    schema = to_json_schema(config_class)

    assert schema.get('additionalProperties') is False
    assert set(schema['properties']) == expected | {'extra'}
    # GenkitConfig base: no temperature/topP/... fields in the Dev UI form.
    assert 'temperature' not in schema['properties']
