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

"""Tests for OpenAI-compatible image generation model."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from genkit.plugins.compat_oai.models.image import (
    SUPPORTED_IMAGE_MODELS,
    OpenAIImageModel,
    _extract_prompt_text,
    _to_generate_response,
    _to_image_generate_params,
)
from genkit.types import (
    GenerateRequest,
    MediaPart,
    Message,
    Part,
    Role,
    TextPart,
)


class TestExtractPromptText:
    """Tests for extracting text from GenerateRequest messages."""

    def test_extracts_text_from_first_message(self) -> None:
        """Verify text extraction from a simple single-message request."""
        request = GenerateRequest(
            messages=[
                Message(role=Role.USER, content=[Part(root=TextPart(text='a sunset'))]),
            ],
        )
        got = _extract_prompt_text(request)
        assert got == 'a sunset'

    def test_raises_on_empty_messages(self) -> None:
        """Verify ValueError when messages list is empty."""
        request = GenerateRequest(messages=[])
        with pytest.raises(ValueError, match='No messages found'):
            _extract_prompt_text(request)

    def test_raises_on_no_text_content(self) -> None:
        """Verify ValueError when message has no text parts."""
        request = GenerateRequest(
            messages=[
                Message(role=Role.USER, content=[]),
            ],
        )
        with pytest.raises(ValueError, match='No text content found'):
            _extract_prompt_text(request)


class TestToImageGenerateParams:
    """Tests for converting GenerateRequest to OpenAI image params."""

    def test_basic_params(self) -> None:
        """Verify required params are set with correct defaults."""
        request = GenerateRequest(
            messages=[
                Message(role=Role.USER, content=[Part(root=TextPart(text='a cat'))]),
            ],
        )
        got = _to_image_generate_params('dall-e-3', request)
        assert got['model'] == 'dall-e-3'
        assert got['prompt'] == 'a cat'
        assert got['response_format'] == 'b64_json'

    def test_config_passthrough(self) -> None:
        """Verify image-specific config options pass through."""
        request = GenerateRequest(
            messages=[
                Message(role=Role.USER, content=[Part(root=TextPart(text='a dog'))]),
            ],
            config={'size': '1024x1024', 'quality': 'hd', 'n': 2},
        )
        got = _to_image_generate_params('dall-e-3', request)
        assert got.get('size') == '1024x1024'
        assert got.get('quality') == 'hd'
        assert got.get('n') == 2

    def test_strips_standard_genai_config(self) -> None:
        """Verify standard GenAI keys are stripped from params."""
        request = GenerateRequest(
            messages=[
                Message(role=Role.USER, content=[Part(root=TextPart(text='test'))]),
            ],
            config={'temperature': 0.5, 'topK': 40, 'topP': 0.9},
        )
        got = _to_image_generate_params('dall-e-3', request)
        assert 'temperature' not in got
        assert 'topK' not in got
        assert 'topP' not in got

    def test_version_override(self) -> None:
        """Verify model version override via config."""
        request = GenerateRequest(
            messages=[
                Message(role=Role.USER, content=[Part(root=TextPart(text='test'))]),
            ],
            config={'version': 'dall-e-3-custom'},
        )
        got = _to_image_generate_params('dall-e-3', request)
        assert got['model'] == 'dall-e-3-custom'


class TestToGenerateResponse:
    """Tests for converting OpenAI ImagesResponse to GenerateResponse."""

    def test_empty_data(self) -> None:
        """Verify empty image data produces empty content."""
        mock_result = MagicMock()
        mock_result.data = []
        got = _to_generate_response(mock_result)
        assert got.message is not None
        assert len(got.message.content) == 0

    def test_url_response(self) -> None:
        """Verify URL-based image response is preserved."""
        mock_image = MagicMock()
        mock_image.url = 'https://example.com/image.png'
        mock_image.b64_json = None
        mock_result = MagicMock()
        mock_result.data = [mock_image]

        got = _to_generate_response(mock_result)
        assert got.message is not None
        assert len(got.message.content) == 1

        part = got.message.content[0].root
        assert isinstance(part, MediaPart)
        assert str(part.media.url) == 'https://example.com/image.png'

    def test_b64_response(self) -> None:
        """Verify base64-encoded image is wrapped in a data URI."""
        mock_image = MagicMock()
        mock_image.url = None
        mock_image.b64_json = 'aGVsbG8='
        mock_result = MagicMock()
        mock_result.data = [mock_image]

        got = _to_generate_response(mock_result)
        assert got.message is not None
        part = got.message.content[0].root
        assert isinstance(part, MediaPart)
        assert str(part.media.url) == 'data:image/png;base64,aGVsbG8='

    def test_multiple_images(self) -> None:
        """Verify multiple images produce multiple content parts."""
        images = []
        for i in range(3):
            img = MagicMock()
            img.url = f'https://example.com/{i}.png'
            img.b64_json = None
            images.append(img)
        mock_result = MagicMock()
        mock_result.data = images

        got = _to_generate_response(mock_result)
        assert got.message is not None
        assert len(got.message.content) == 3


class TestSupportedImageModels:
    """Tests that the model info registry is correct."""

    def test_dall_e_3_in_registry(self) -> None:
        """Verify DALL-E 3 is registered."""
        assert 'dall-e-3' in SUPPORTED_IMAGE_MODELS

    def test_gpt_image_1_in_registry(self) -> None:
        """Verify GPT-Image-1 is registered."""
        assert 'gpt-image-1' in SUPPORTED_IMAGE_MODELS

    def test_image_models_support_media_output(self) -> None:
        """Verify all image models declare 'media' output support."""
        for name, info in SUPPORTED_IMAGE_MODELS.items():
            assert info.supports is not None, f'{name} has no supports metadata'
            assert 'media' in (info.supports.output or []), f"{name} should support 'media' output"


class TestOpenAIImageModel:
    """Tests for the OpenAIImageModel class."""

    @pytest.mark.asyncio
    async def test_generate_calls_client(self) -> None:
        """Verify generate() calls client.images.generate and returns media."""
        mock_image = MagicMock()
        mock_image.url = 'https://example.com/generated.png'
        mock_image.b64_json = None
        mock_response = MagicMock()
        mock_response.data = [mock_image]

        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        model = OpenAIImageModel('dall-e-3', mock_client)
        request = GenerateRequest(
            messages=[
                Message(role=Role.USER, content=[Part(root=TextPart(text='a mountain'))]),
            ],
        )

        ctx = MagicMock()
        got = await model.generate(request, ctx)

        mock_client.images.generate.assert_called_once()
        call_kwargs = mock_client.images.generate.call_args
        assert call_kwargs.kwargs.get('model') == 'dall-e-3'
        assert call_kwargs.kwargs.get('prompt') == 'a mountain'

        assert got.message is not None
        assert len(got.message.content) == 1
