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

"""What ai.generate sends to Bedrock for config['extra'] and context.secrets."""

import json
from typing import Any, cast

import pytest
from genkit_amazon_bedrock import Bedrock, BedrockConfig, NovaCanvasConfig, StabilityImageConfig, TitanImageConfig
from genkit_amazon_bedrock._transport import BedrockTransport
from pydantic import ValidationError

from genkit import Genkit, GenkitError

CHAT_MODEL = 'bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0'
IMAGE_MODEL = 'bedrock/stability.sd3-5-large-v1:0'
NOVA_CANVAS = 'bedrock/amazon.nova-canvas-v1:0'
TITAN_IMAGE = 'bedrock/amazon.titan-image-generator-v2:0'
THINKING = {'thinking': {'type': 'enabled', 'budget_tokens': 2048}}


class FakeTransport:
    """Stands in for BedrockTransport; records every call it would send."""

    def __init__(self) -> None:
        self.converse_calls: list[dict[str, Any]] = []
        self.invoke_calls: list[dict[str, Any]] = []

    async def ensure_client(self) -> None:
        return None

    async def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.converse_calls.append(kwargs)
        return {
            'output': {'message': {'role': 'assistant', 'content': [{'text': 'ok'}]}},
            'stopReason': 'end_turn',
        }

    async def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        self.invoke_calls.append(kwargs)
        return {'images': ['img'], 'finish_reasons': ['SUCCESS']}


def bedrock_app() -> tuple[Genkit, FakeTransport]:
    plugin = Bedrock(
        region='us-east-1',
        models=[
            'stability.sd3-5-large-v1:0',
            'amazon.nova-canvas-v1:0',
        ],
    )
    transport = FakeTransport()
    plugin._transport = cast(BedrockTransport, transport)  # noqa: SLF001
    return Genkit(plugins=[plugin]), transport


@pytest.mark.asyncio
async def test_generate_bedrock_extra_is_sent_as_additional_model_request_fields() -> None:
    """`config={'extra': {...}}` reaches Converse as `additionalModelRequestFields={...}`."""
    ai, transport = bedrock_app()

    await ai.generate(model=CHAT_MODEL, prompt='hi', config={'extra': THINKING})

    assert transport.converse_calls[0]['additionalModelRequestFields'] == THINKING


@pytest.mark.asyncio
async def test_generate_bedrock_config_class_extra_is_sent_as_additional_model_request_fields() -> None:
    """`config=BedrockConfig(extra={...})` sends the same request."""
    ai, transport = bedrock_app()

    await ai.generate(model=CHAT_MODEL, prompt='hi', config=BedrockConfig(extra=THINKING))

    assert transport.converse_calls[0]['additionalModelRequestFields'] == THINKING


@pytest.mark.asyncio
@pytest.mark.parametrize('old_name', ['additionalModelRequestFields', 'additional_model_request_fields'])
async def test_generate_bedrock_old_additional_fields_name_raises_and_points_to_extra(old_name: str) -> None:
    """The old name, in either spelling, raises INVALID_ARGUMENT saying to use `extra`, and sends nothing."""
    ai, transport = bedrock_app()

    with pytest.raises(GenkitError) as exc_info:
        await ai.generate(model=CHAT_MODEL, prompt='hi', config={old_name: THINKING})

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert old_name in str(exc_info.value)
    assert "config['extra']" in str(exc_info.value)
    assert transport.converse_calls == []


@pytest.mark.asyncio
async def test_generate_bedrock_image_extra_is_merged_into_request_body() -> None:
    """`config={'extra': {'style_preset': ...}}` on an image model puts `style_preset` in the request body."""
    ai, transport = bedrock_app()

    await ai.generate(model=IMAGE_MODEL, prompt='a reef', config={'extra': {'style_preset': 'photographic'}})

    body = json.loads(transport.invoke_calls[0]['body'])
    assert body['style_preset'] == 'photographic'
    assert 'extra' not in body


@pytest.mark.asyncio
async def test_generate_bedrock_image_extra_key_overrides_built_field() -> None:
    """An `extra` key the plugin also builds (`output_format`) is sent with the caller's value."""
    ai, transport = bedrock_app()

    response = await ai.generate(model=IMAGE_MODEL, prompt='a reef', config={'extra': {'output_format': 'jpeg'}})

    body = json.loads(transport.invoke_calls[0]['body'])
    assert body['output_format'] == 'jpeg'
    assert response.media[0].content_type == 'image/jpeg'


@pytest.mark.asyncio
async def test_generate_bedrock_image_extra_deep_merges_nested_object() -> None:
    """`extra={'imageGenerationConfig': {'seed': 7}}` sets `seed` and keeps the other defaults."""
    ai, transport = bedrock_app()

    await ai.generate(
        model=NOVA_CANVAS,
        prompt='a reef',
        config={'extra': {'imageGenerationConfig': {'seed': 7}}},
    )

    body = json.loads(transport.invoke_calls[0]['body'])
    assert body['imageGenerationConfig'] == {
        'numberOfImages': 1,
        'height': 1024,
        'width': 1024,
        'cfgScale': 8.0,
        'seed': 7,
        'quality': 'standard',
    }
    assert body['taskType'] == 'TEXT_IMAGE'


@pytest.mark.asyncio
async def test_generate_bedrock_image_extra_adds_negative_text_next_to_prompt() -> None:
    """`extra={'textToImageParams': {'negativeText': ...}}` on Nova Canvas keeps the prompt text."""
    ai, transport = bedrock_app()

    await ai.generate(
        model=NOVA_CANVAS,
        prompt='a reef',
        config={'extra': {'textToImageParams': {'negativeText': 'blurry'}}},
    )

    body = json.loads(transport.invoke_calls[0]['body'])
    assert body['textToImageParams'] == {'text': 'a reef', 'negativeText': 'blurry'}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('model', 'extra', 'field'),
    [
        (IMAGE_MODEL, {'prompt': 'a shipwreck'}, 'prompt'),
        (NOVA_CANVAS, {'taskType': 'INPAINTING'}, 'taskType'),
        (NOVA_CANVAS, {'textToImageParams': {'text': 'a shipwreck'}}, 'textToImageParams.text'),
        (NOVA_CANVAS, {'textToImageParams': 'a shipwreck'}, 'textToImageParams'),
        (TITAN_IMAGE, {'taskType': 'INPAINTING'}, 'taskType'),
        (TITAN_IMAGE, {'textToImageParams': {'text': 'a shipwreck'}}, 'textToImageParams.text'),
    ],
    ids=['stability_prompt', 'nova_task_type', 'nova_text', 'nova_params_not_mapping', 'titan_task_type', 'titan_text'],
)
async def test_generate_bedrock_image_extra_prompt_field_fails_invalid_argument(
    model: str, extra: dict[str, Any], field: str
) -> None:
    """An `extra` key that would overwrite the prompt fails with INVALID_ARGUMENT and sends nothing."""
    ai, transport = bedrock_app()

    response = await ai.generate(model=model, prompt='a reef', config={'extra': extra})

    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert repr(field) in response.error.message
    assert transport.invoke_calls == []


@pytest.mark.asyncio
async def test_generate_bedrock_image_extra_not_a_mapping_fails_invalid_argument() -> None:
    """`config={'extra': 'oops'}` fails with INVALID_ARGUMENT before the call, and sends nothing."""
    ai, transport = bedrock_app()

    with pytest.raises(GenkitError) as exc_info:
        await ai.generate(model=IMAGE_MODEL, prompt='a reef', config={'extra': 'oops'})

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert "'extra'" in str(exc_info.value)
    assert transport.invoke_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('model', 'config', 'expected'),
    [
        (
            NOVA_CANVAS,
            NovaCanvasConfig(seed=7, width=512, height=512, quality='premium', number_of_images=2, cfg_scale=6.5),
            {
                'taskType': 'TEXT_IMAGE',
                'textToImageParams': {'text': 'a reef'},
                'imageGenerationConfig': {
                    'numberOfImages': 2,
                    'height': 512,
                    'width': 512,
                    'cfgScale': 6.5,
                    'seed': 7,
                    'quality': 'premium',
                },
            },
        ),
        (
            TITAN_IMAGE,
            {'seed': 7, 'width': 512, 'height': 512},
            {
                'taskType': 'TEXT_IMAGE',
                'textToImageParams': {'text': 'a reef'},
                'imageGenerationConfig': {'numberOfImages': 1, 'height': 512, 'width': 512, 'cfgScale': 8.0, 'seed': 7},
            },
        ),
        (
            IMAGE_MODEL,
            StabilityImageConfig(aspect_ratio='16:9', seed=42, negative_prompt='blurry', output_format='jpeg'),
            {
                'prompt': 'a reef',
                'output_format': 'jpeg',
                'aspect_ratio': '16:9',
                'seed': 42,
                'negative_prompt': 'blurry',
            },
        ),
    ],
    ids=['nova_typed', 'titan_dict', 'stability_typed'],
)
async def test_generate_bedrock_image_config_reaches_the_wire(
    model: str, config: Any, expected: dict[str, Any]
) -> None:  # noqa: ANN401
    """Typed and dict configs map to each family's wire body."""
    ai, transport = bedrock_app()

    response = await ai.generate(model=model, prompt='a reef', config=config)

    assert response.error is None
    assert json.loads(transport.invoke_calls[0]['body']) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('model', 'config', 'needle'),
    [
        (IMAGE_MODEL, {'aspect_ratio': '16:9', 'style_preset': 'photographic'}, "'style_preset'"),
        (NOVA_CANVAS, {'imageGenerationConfg': {'seed': 7}}, "'imageGenerationConfg'"),
        (TITAN_IMAGE, {'number_of_image': 2}, "'number_of_image'"),
        (IMAGE_MODEL, {'aspect_ratio': '16:9', 'temperature': 0.4}, "'temperature'"),
        (NOVA_CANVAS, {'maxOutputTokens': 10}, "'maxOutputTokens'"),
    ],
    ids=['stability_unlisted_field', 'nova_typo', 'titan_typo', 'stability_temperature', 'nova_max_output_tokens'],
)
async def test_generate_bedrock_image_unknown_key_fails_and_points_to_extra(
    model: str, config: dict[str, Any], needle: str
) -> None:
    """An undeclared image config key raises INVALID_ARGUMENT naming it and `extra`, and sends nothing."""
    ai, transport = bedrock_app()

    with pytest.raises(GenkitError) as exc_info:
        await ai.generate(model=model, prompt='a reef', config=config)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert needle in str(exc_info.value)
    assert "config['extra']" in str(exc_info.value)
    assert transport.invoke_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize('old_name', ['imageGenerationConfig', 'image_generation_config'])
@pytest.mark.parametrize('model', [NOVA_CANVAS, TITAN_IMAGE])
async def test_generate_bedrock_image_old_nested_config_raises_and_names_flat_fields(model: str, old_name: str) -> None:
    """The old nested `imageGenerationConfig`, in either spelling, raises INVALID_ARGUMENT naming the flat fields."""
    ai, transport = bedrock_app()

    with pytest.raises(GenkitError) as exc_info:
        await ai.generate(model=model, prompt='a reef', config={old_name: {'seed': 7}})

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert f'{old_name} was flattened' in str(exc_info.value)
    assert 'number_of_images=' in str(exc_info.value)
    assert transport.invoke_calls == []


@pytest.mark.asyncio
async def test_generate_bedrock_image_config_class_of_another_family_fails() -> None:
    """`NovaCanvasConfig` on a Titan model is rejected; each family registers its own class."""
    ai, transport = bedrock_app()

    with pytest.raises(GenkitError) as exc_info:
        await ai.generate(model=TITAN_IMAGE, prompt='a reef', config=NovaCanvasConfig(seed=7))

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'TitanImageConfig' in str(exc_info.value)
    assert transport.invoke_calls == []


def test_titan_image_config_rejects_the_old_nested_key_at_construction() -> None:
    """`TitanImageConfig(imageGenerationConfig=...)` raises the same INVALID_ARGUMENT as the dict form."""
    with pytest.raises(GenkitError, match='imageGenerationConfig was flattened'):
        TitanImageConfig.model_validate({'imageGenerationConfig': {'seed': 7}})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('model', 'expected_body'),
    [
        (IMAGE_MODEL, {'prompt': 'a reef', 'output_format': 'png'}),
        (
            NOVA_CANVAS,
            {
                'taskType': 'TEXT_IMAGE',
                'textToImageParams': {'text': 'a reef'},
                'imageGenerationConfig': {
                    'numberOfImages': 1,
                    'height': 1024,
                    'width': 1024,
                    'cfgScale': 8.0,
                    'seed': 0,
                    'quality': 'standard',
                },
            },
        ),
    ],
    ids=['stability', 'nova_canvas'],
)
async def test_generate_bedrock_image_without_extra_body_unchanged(model: str, expected_body: dict[str, Any]) -> None:
    """An image call with no `extra` sends exactly the default body for its family."""
    ai, transport = bedrock_app()

    await ai.generate(model=model, prompt='a reef')

    assert json.loads(transport.invoke_calls[0]['body']) == expected_body


def test_bedrock_config_with_old_additional_fields_name_raises_validation_error() -> None:
    """`BedrockConfig(additional_model_request_fields=...)` is a ValidationError."""
    with pytest.raises(ValidationError):
        BedrockConfig(additional_model_request_fields=THINKING)  # type: ignore[call-arg]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'context',
    [
        None,
        {'secrets': {'db_password': 'x'}},
        {'secrets': {'api_key': 'sk-openai'}},
        {'secrets': {'apiKey': 'sk-openai'}},
    ],
    ids=['no_context', 'other_secrets', 'api_key', 'apiKey'],
)
@pytest.mark.parametrize('model', [CHAT_MODEL, IMAGE_MODEL], ids=['chat', 'image'])
async def test_generate_bedrock_ignores_secrets_api_key(model: str, context: dict[str, Any] | None) -> None:
    """Bedrock runs on the app's AWS credentials; a `secrets.api_key` meant for another provider doesn't fail it."""
    ai, transport = bedrock_app()

    response = await ai.generate(model=model, prompt='hi', context=context)

    assert response.error is None
    assert len(transport.converse_calls) + len(transport.invoke_calls) == 1
