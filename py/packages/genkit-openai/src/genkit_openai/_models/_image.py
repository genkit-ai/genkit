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
"""

from __future__ import annotations

from typing import Any, Literal

from openai import APIError, AsyncOpenAI
from openai.types.images_response import ImagesResponse
from pydantic import Field

from genkit import ActionRunContext, FinishReason, Message, ModelResponse, Part, Role
from genkit.model import ModelConfig, ModelInfo, ModelRequest, Supports
from genkit_openai._models._utils import _extract_text, extract_config_dict, pop_extra_body, reraise_openai_error


class OpenAIDalleConfig(ModelConfig):
    """Settings the DALL-E image endpoint reads from config."""

    size: str | None = None
    quality: Literal['standard', 'hd'] | None = None
    style: Literal['vivid', 'natural'] | None = None
    n: int | None = None
    user: str | None = None
    response_format: str | None = None


class OpenAIGptImageConfig(ModelConfig):
    """Settings the GPT Image endpoint reads from config."""

    size: Literal['1024x1024', '1536x1024', '1024x1536', 'auto'] | None = None
    style: Literal['vivid', 'natural'] | None = None
    user: str | None = None
    n: int | None = Field(default=None, ge=1, le=10)
    quality: Literal['low', 'medium', 'high'] | None = Field(
        default=None,
        json_schema_extra={'enum': ['low', 'medium', 'high']},
    )
    background: Literal['transparent', 'opaque', 'auto'] | None = None
    moderation: Literal['low', 'auto'] | None = None
    output_compression: int | None = Field(default=None, ge=1, le=100)
    output_format: Literal['png', 'jpeg', 'webp'] | None = None


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
    request: ModelRequest,
) -> dict[str, Any]:
    """Convert a ModelRequest into OpenAI image generation parameters.

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
    extra_body = pop_extra_body(config, managed=('prompt', 'model'), label='openai image')

    # Start with required params.
    effective_model = config.pop('version', None) or model_name
    params: dict[str, Any] = {
        'model': effective_model,
        'prompt': prompt,
    }

    # GPT Image 1 rejects the response_format parameter; its API always
    # returns base64 image data. DALL-E models retain the existing default.
    if 'gpt-image' not in effective_model.lower():
        params['response_format'] = config.pop('response_format', 'b64_json')
    else:
        config.pop('response_format', None)

    # Strip standard GenAI config keys that don't apply to image generation.
    for key in ('temperature', 'max_output_tokens', 'stop_sequences', 'top_k', 'top_p'):
        config.pop(key, None)

    # Pass remaining config through (size, quality, style, n, etc.).
    params.update(config)
    if extra_body:
        params['extra_body'] = extra_body

    # Remove None values.
    return {k: v for k, v in params.items() if v is not None}


def _to_generate_response(result: ImagesResponse) -> ModelResponse:
    """Convert an OpenAI ImagesResponse to a Genkit ModelResponse.

    Each generated image becomes a media part in the response message.

    Args:
        result: The OpenAI images.generate() response object.

    Returns:
        A ModelResponse with media parts for each generated image.
    """
    images = result.data
    if not images:
        return ModelResponse(
            message=Message(role=Role.MODEL, content=[]),
            finish_reason=FinishReason.STOP,
        )

    content: list[Part] = []
    for image in images:
        url = image.url
        if not url and image.b64_json:
            url = f'data:image/png;base64,{image.b64_json}'

        if url:
            content.append(Part.from_media(url, content_type='image/png'))

    return ModelResponse(
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

    async def generate(self, request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        """Generate images from the request.

        Args:
            request: The generate request containing the text prompt.
            ctx: The action run context.

        Returns:
            A ModelResponse containing generated image media parts.
        """
        try:
            params = _to_image_generate_params(self._model_name, request)
            result = await self._client.images.generate(**params)
            return _to_generate_response(result)
        except (APIError, ValueError) as e:
            reraise_openai_error(e)
