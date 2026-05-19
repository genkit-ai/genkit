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


"""Tests for the Imagen model implementation."""

import base64

import pytest
from google import genai
from pytest_mock import MockerFixture

from genkit import (
    ActionRunContext,
    MediaPart,
    Message,
    ModelRequest,
    ModelResponse,
    Part,
    Role,
    TextPart,
)
from genkit.plugin_api import to_json_schema
from genkit.plugins.google_genai.models.imagen import (
    DEFAULT_NUMBER_OF_IMAGES,
    ImagenConfigSchema,
    ImagenModel,
    ImagenVersion,
)


@pytest.mark.asyncio
@pytest.mark.parametrize('version', [x for x in ImagenVersion])
async def test_generate_media_response(mocker: MockerFixture, version: ImagenVersion) -> None:
    """Test generate method for media responses."""
    request_text = 'response question'
    response_byte_string = b'\x89PNG\r\n\x1a\n'
    response_mimetype = 'image/png'

    request = ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(root=TextPart(text=request_text)),
                ],
            ),
        ],
    )

    response_images = genai.types.GenerateImagesResponse(
        generated_images=[
            genai.types.GeneratedImage(
                image=genai.types.Image(image_bytes=response_byte_string, mime_type=response_mimetype)
            )
        ]
    )

    googleai_client_mock = mocker.AsyncMock()
    googleai_client_mock.aio.models.generate_images.return_value = response_images

    imagen = ImagenModel(version, googleai_client_mock)

    ctx = ActionRunContext()
    response = await imagen.generate(request, ctx)

    googleai_client_mock.assert_has_calls([
        mocker.call.aio.models.generate_images(model=version, prompt=request_text, config=mocker.ANY)
    ])
    config = googleai_client_mock.aio.models.generate_images.call_args.kwargs['config']
    assert config['number_of_images'] == DEFAULT_NUMBER_OF_IMAGES
    assert isinstance(response, ModelResponse)
    assert response.message is not None
    content = response.message.content[0]
    assert isinstance(content.root, MediaPart)

    assert content.root.media.content_type == response_mimetype

    # Verify the data URL contains the correct base64-encoded content
    # Data URLs have format: data:<mimetype>;base64,<data>
    data_url = content.root.media.url
    assert data_url.startswith(f'data:{response_mimetype};base64,')
    encoded_data = data_url.split(',', 1)[1]
    assert base64.b64decode(encoded_data) == response_byte_string


def test_imagen_config_schema_exposes_supported_options() -> None:
    """Test Imagen config options are visible in action metadata."""
    schema = to_json_schema(ImagenConfigSchema)
    properties = schema['properties']
    number_of_images = properties['numberOfImages']

    assert set(properties) == {
        'numberOfImages',
        'imageSize',
        'aspectRatio',
        'personGeneration',
    }
    assert number_of_images['default'] == DEFAULT_NUMBER_OF_IMAGES
    assert number_of_images['anyOf'][0]['maximum'] == 4


def test_imagen_config_preserves_requested_image_count(mocker: MockerFixture) -> None:
    """Test explicit image count is not overwritten by default."""
    request = ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(root=TextPart(text='draw a fox')),
                ],
            ),
        ],
        config={'number_of_images': 2},
    )
    googleai_client_mock = mocker.AsyncMock()
    imagen = ImagenModel(ImagenVersion.IMAGEN4, googleai_client_mock)

    config = imagen._get_config(request)

    assert config['number_of_images'] == 2


def test_imagen_config_rejects_unsupported_options(mocker: MockerFixture) -> None:
    """Test Imagen only accepts the supported Google AI options."""
    request = ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(root=TextPart(text='draw a fox')),
                ],
            ),
        ],
        config={'safetyFilterLevel': 'BLOCK_LOW_AND_ABOVE'},
    )
    googleai_client_mock = mocker.AsyncMock()
    imagen = ImagenModel(ImagenVersion.IMAGEN4, googleai_client_mock)

    with pytest.raises(ValueError, match='configuration dictionary is invalid'):
        imagen._get_config(request)
