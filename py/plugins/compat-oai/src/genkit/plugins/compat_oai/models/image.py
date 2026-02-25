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

"""OpenAI-compatible image generation model for Genkit.

Provides image generation capabilities via the OpenAI Images API,
supporting models like DALL-E 3 and GPT-Image-1.

Data Flow::

    ┌─────────────────────────────────────────────────────────────────────┐
    │  GenerateRequest (text prompt)                                      │
    │         │                                                           │
    │         ▼                                                           │
    │  to_image_generate_params()  ──►  ImageGenerateParams               │
    │         │                                                           │
    │         ▼                                                           │
    │  client.images.generate()                                           │
    │         │                                                           │
    │         ▼                                                           │
    │  to_generate_response()  ──►  GenerateResponse (media parts)        │
    └─────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI
from openai.types.images_response import ImagesResponse

from genkit.ai import ActionRunContext
from genkit.core.typing import FinishReason
from genkit.plugins.compat_oai.models.utils import _extract_text, extract_config_dict
from genkit.types import (
    GenerateRequest,
    GenerateResponse,
    Media,
    MediaPart,
    Message,
    ModelInfo,
    Part,
    Role,
    Supports,
)

# Supported image generation models with their metadata.
SUPPORTED_IMAGE_MODELS: dict[str, ModelInfo] = {
    'dall-e-3': ModelInfo(
        label='OpenAI - DALL-E 3',
        supports=Supports(
            media=False,
            output=['media'],
            multiturn=False,
            system_role=False,
            tools=False,
        ),
    ),
    'gpt-image-1': ModelInfo(
        label='OpenAI - GPT Image 1',
        supports=Supports(
            media=False,
            output=['media'],
            multiturn=False,
            system_role=False,
            tools=False,
        ),
    ),
}


# Re-export _extract_text as _extract_prompt_text for backward compatibility.
_extract_prompt_text = _extract_text


def _to_image_generate_params(
    model_name: str,
    request: GenerateRequest,
) -> dict[str, Any]:
    """Convert a GenerateRequest into OpenAI image generation parameters.

    Extracts the text prompt and maps Genkit config options to OpenAI's
    image generation API parameters.

    Args:
        model_name: The OpenAI model name (e.g., 'dall-e-3').
        request: The Genkit generate request.

    Returns:
        A dictionary of parameters for client.images.generate().
    """
    prompt = _extract_prompt_text(request)
    config = extract_config_dict(request)

    # Start with required params.
    params: dict[str, Any] = {
        'model': config.pop('version', None) or model_name,
        'prompt': prompt,
        'response_format': config.pop('response_format', 'b64_json'),
    }

    # Strip standard GenAI config keys that don't apply to image generation.
    for key in ('temperature', 'maxOutputTokens', 'stopSequences', 'topK', 'topP'):
        config.pop(key, None)

    # Pass remaining config through (size, quality, style, n, etc.).
    params.update(config)

    # Remove None values.
    return {k: v for k, v in params.items() if v is not None}


def _to_generate_response(result: ImagesResponse) -> GenerateResponse:
    """Convert an OpenAI ImagesResponse to a Genkit GenerateResponse.

    Each generated image becomes a media part in the response message.

    Args:
        result: The OpenAI images.generate() response object.

    Returns:
        A GenerateResponse with media parts for each generated image.
    """
    images = result.data
    if not images:
        return GenerateResponse(
            message=Message(role=Role.MODEL, content=[]),
            finish_reason=FinishReason.STOP,
        )

    content: list[Part] = []
    for image in images:
        url = image.url
        if not url and image.b64_json:
            url = f'data:image/png;base64,{image.b64_json}'

        if url:
            content.append(Part(root=MediaPart(media=Media(content_type='image/png', url=url))))

    return GenerateResponse(
        message=Message(role=Role.MODEL, content=content),
        finish_reason=FinishReason.STOP,
    )


class OpenAIImageModel:
    """Handles image generation via the OpenAI Images API.

    Args:
        model_name: The image model to use (e.g., 'dall-e-3').
        client: An async OpenAI client instance.
    """

    def __init__(self, model_name: str, client: AsyncOpenAI) -> None:
        """Initialize the image model.

        Args:
            model_name: The image model to use (e.g., 'dall-e-3').
            client: An async OpenAI client instance.
        """
        self._model_name = model_name
        self._client = client

    @property
    def name(self) -> str:
        """The name of the image model."""
        return self._model_name

    async def generate(self, request: GenerateRequest, ctx: ActionRunContext) -> GenerateResponse:
        """Generate images from the request.

        Args:
            request: The generate request containing the text prompt.
            ctx: The action run context.

        Returns:
            A GenerateResponse containing generated image media parts.
        """
        params = _to_image_generate_params(self._model_name, request)
        result = await self._client.images.generate(**params)
        return _to_generate_response(result)
