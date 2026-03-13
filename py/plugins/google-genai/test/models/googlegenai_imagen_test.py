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

from genkit.ai import ActionRunContext
from genkit.plugins.google_genai.models.imagen import ImagenModel, ImagenVersion
from genkit.types import (
    GenerateRequest,
    GenerateResponse,
    MediaPart,
    Message,
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

    request = GenerateRequest(
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
    assert isinstance(response, GenerateResponse)
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
