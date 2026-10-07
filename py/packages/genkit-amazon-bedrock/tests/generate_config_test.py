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
from genkit_amazon_bedrock import Bedrock, BedrockConfig, ModelDefinition
from genkit_amazon_bedrock.transport import BedrockTransport
from pydantic import ValidationError

from genkit import Genkit, GenkitError

CHAT_MODEL = 'bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0'
IMAGE_MODEL = 'bedrock/stability.sd3-5-large-v1:0'
NOVA_CANVAS = 'bedrock/amazon.nova-canvas-v1:0'
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
            ModelDefinition(name='stability.sd3-5-large-v1:0', type='image'),
            ModelDefinition(name='amazon.nova-canvas-v1:0', type='image'),
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
async def test_generate_bedrock_image_extra_replaces_top_level_key() -> None:
    """`extra={'imageGenerationConfig': {'seed': 7}}` replaces that whole key; nested fields are not merged."""
    ai, transport = bedrock_app()

    await ai.generate(
        model=NOVA_CANVAS,
        prompt='a reef',
        config={'extra': {'imageGenerationConfig': {'seed': 7}}},
    )

    body = json.loads(transport.invoke_calls[0]['body'])
    assert body['imageGenerationConfig'] == {'seed': 7}
    assert body['taskType'] == 'TEXT_IMAGE'


@pytest.mark.asyncio
async def test_generate_bedrock_image_undeclared_key_is_accepted() -> None:
    """`config={'aspect_ratio': '16:9'}` on a Stability model is sent; image config does not reject unknown keys."""
    ai, transport = bedrock_app()

    await ai.generate(model=IMAGE_MODEL, prompt='a reef', config={'aspect_ratio': '16:9'})

    body = json.loads(transport.invoke_calls[0]['body'])
    assert body['aspect_ratio'] == '16:9'


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
@pytest.mark.parametrize('key_name', ['api_key', 'apiKey'])
async def test_generate_bedrock_secrets_api_key_fails_invalid_argument(key_name: str) -> None:
    """A per-request key in `context.secrets` fails with INVALID_ARGUMENT on `.error` and sends nothing to AWS."""
    ai, transport = bedrock_app()

    response = await ai.generate(model=CHAT_MODEL, prompt='hi', context={'secrets': {key_name: 'tenant-key'}})

    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert 'AWS credentials' in response.error.message
    assert transport.converse_calls == []


@pytest.mark.asyncio
async def test_generate_bedrock_image_secrets_api_key_fails_invalid_argument() -> None:
    """The same key on a Bedrock image model fails the same way and sends nothing."""
    ai, transport = bedrock_app()

    response = await ai.generate(model=IMAGE_MODEL, prompt='a reef', context={'secrets': {'api_key': 'tenant-key'}})

    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert transport.invoke_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'context',
    [None, {'secrets': {'db_password': 'x'}}],
    ids=['no_context', 'secrets_without_api_key'],
)
async def test_generate_bedrock_without_secrets_api_key_uses_aws_credentials(context: dict[str, Any] | None) -> None:
    """No key in `context.secrets` runs on the app's AWS credentials as before."""
    ai, transport = bedrock_app()

    response = await ai.generate(model=CHAT_MODEL, prompt='hi', context=context)

    assert response.error is None
    assert response.text == 'ok'
    assert len(transport.converse_calls) == 1
