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
from unittest.mock import MagicMock

import pytest
from genkit_google_genai.models.imagen import ImagenConfigSchema, ImagenModel, ImagenVersion
from google import genai
from pydantic import ValidationError
from pytest_mock import MockerFixture

from genkit import (
    ActionRunContext,
    GenkitError,
    MediaPart,
    Message,
    ModelRequest,
    ModelResponse,
    Part,
    Role,
    TextPart,
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
        mocker.call.aio.models.generate_images(model=version, prompt=request_text, config=None)
    ])
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


def test_imagen_unknown_extra_rides_on_extra_body() -> None:
    """Leftover keys ride on extra_body so a newly supported field still reaches the API."""
    imagen = ImagenModel(ImagenVersion.IMAGEN3, MagicMock())
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='a cat'))])],
        config=ImagenConfigSchema.model_validate({'fooBar': 1}),
    )

    cfg = imagen._get_config(request)

    assert cfg is not None
    assert cfg.http_options is not None
    assert cfg.http_options.extra_body == {'parameters': {'fooBar': 1}}


def test_imagen_rejects_raw_dicts() -> None:
    """A dict at the dump leaf means Action never produced the family instance."""
    imagen = ImagenModel(ImagenVersion.IMAGEN3, MagicMock())
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='a cat'))])],
        config={'number_of_images': 1},  # type: ignore[arg-type]
    )

    with pytest.raises(GenkitError) as exc_info:
        imagen._get_config(request)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert imagen._version in str(exc_info.value)


def test_imagen_invalid_sdk_field_is_invalid_argument() -> None:
    """SDK type errors on an untyped-but-known key become a named INVALID_ARGUMENT."""
    imagen = ImagenModel(ImagenVersion.IMAGEN3, MagicMock())
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='a cat'))])],
        config=ImagenConfigSchema.model_validate({'http_options': 'nope'}),
    )

    with pytest.raises(GenkitError) as exc_info:
        imagen._get_config(request)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'http_options' in str(exc_info.value)


def test_imagen_bad_typed_field_is_rejected_by_the_schema() -> None:
    """Typed fields are checked before the request is built."""
    with pytest.raises(ValidationError):
        ImagenConfigSchema.model_validate({'numberOfImages': 'nope'})


def test_imagen_typed_config_reaches_generate_images_config() -> None:
    """Every typed Imagen field lands on the SDK config rather than extra_body."""
    imagen = ImagenModel(ImagenVersion.IMAGEN3, MagicMock())
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='a cat'))])],
        config=ImagenConfigSchema.model_validate({
            'numberOfImages': 2,
            'aspectRatio': '16:9',
            'negativePrompt': 'blurry',
            'guidanceScale': 12.5,
            'seed': 7,
            'safetyFilterLevel': 'BLOCK_ONLY_HIGH',
            'personGeneration': 'allow_adult',
            'includeSafetyAttributes': True,
            'includeRaiReason': True,
            'language': 'en',
            'outputMimeType': 'image/jpeg',
            'outputCompressionQuality': 80,
            'addWatermark': False,
            'outputGcsUri': 'gs://bucket/prefix',
            'labels': {'team': 'ads'},
            'imageSize': '2K',
            'enhancePrompt': True,
        }),
    )

    cfg = imagen._get_config(request)

    assert cfg is not None
    assert cfg.number_of_images == 2
    assert cfg.aspect_ratio == '16:9'
    assert cfg.negative_prompt == 'blurry'
    assert cfg.guidance_scale == 12.5
    assert cfg.seed == 7
    assert cfg.safety_filter_level == 'BLOCK_ONLY_HIGH'
    assert cfg.person_generation == 'ALLOW_ADULT'
    assert cfg.include_safety_attributes is True
    assert cfg.include_rai_reason is True
    assert cfg.language == 'en'
    assert cfg.output_mime_type == 'image/jpeg'
    assert cfg.output_compression_quality == 80
    assert cfg.add_watermark is False
    assert cfg.output_gcs_uri == 'gs://bucket/prefix'
    assert cfg.labels == {'team': 'ads'}
    assert cfg.image_size == '2K'
    assert cfg.enhance_prompt is True
    assert cfg.http_options is None


def test_imagen_config_accepts_snake_case_keys() -> None:
    """Typed fields populate by field name as well as by alias."""
    imagen = ImagenModel(ImagenVersion.IMAGEN3, MagicMock())
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='a cat'))])],
        config=ImagenConfigSchema.model_validate({'number_of_images': 3, 'aspect_ratio': '9:16'}),
    )

    cfg = imagen._get_config(request)

    assert cfg is not None
    assert cfg.number_of_images == 3
    assert cfg.aspect_ratio == '9:16'
