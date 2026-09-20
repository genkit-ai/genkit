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

"""Google GenAI Imagen models."""

from __future__ import annotations

import logging
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, ConfigDict, Field

from genkit import (
    Candidate,
    GenerateResponseChunk,
    Message,
    ModelResponse,
    Part,
    Role,
)
from genkit.model import ModelInfo, ModelRequest, Supports
from genkit.plugin_api import ActionRunContext, wrap_http_error
from genkit.telemetry import SpanContext, run_in_new_span
from genkit_google_genai.models._sdk_config import (
    attach_leftovers,
    dump_family_config,
    read_dumped_field,
)

logger = logging.getLogger(__name__)

IMAGEN_3 = 'imagen-3.0-generate-002'
SUPPORTED_IMAGEN_MODELS = {IMAGEN_3: 'Imagen 3'}


class ImagenConfig(BaseModel):
    """Configuration options for Imagen models."""

    model_config = ConfigDict(extra='allow')

    number_of_images: int | None = Field(default=None, description='Number of images to generate (1-4).')
    aspect_ratio: str | None = Field(default=None, description='Aspect ratio: 1:1, 3:4, 4:3, 9:16, or 16:9.')
    output_mime_type: str | None = Field(default=None, description='Output MIME type: image/jpeg or image/png.')
    person_generation: str | None = Field(
        default=None,
        description='Person generation setting: DONT_ALLOW, ALLOW_ADULT, or ALLOW_ALL.',
    )
    safety_filter_level: str | None = Field(
        default=None,
        description='Safety filter level: BLOCK_LOW_AND_ABOVE, BLOCK_MEDIUM_AND_ABOVE, or BLOCK_ONLY_HIGH.',
    )


def _to_imagen_config(config: ImagenConfig | dict[str, Any] | None) -> types.GenerateImagesConfig:
    dumped = dump_family_config(config)
    kwargs: dict[str, Any] = {}
    for dest, src in (
        ('number_of_images', 'number_of_images'),
        ('aspect_ratio', 'aspect_ratio'),
        ('output_mime_type', 'output_mime_type'),
        ('person_generation', 'person_generation'),
        ('safety_filter_level', 'safety_filter_level'),
    ):
        val = read_dumped_field(dumped, src)
        if val is not None:
            kwargs[dest] = val
    c = types.GenerateImagesConfig(**kwargs)
    attach_leftovers(c, dumped)
    return c


def create_imagen_model(
    client: genai.Client,
    model_name: str,
) -> tuple[ModelInfo, Any]:
    """Create an Imagen model action and its metadata."""
    info = ModelInfo(
        label=SUPPORTED_IMAGEN_MODELS.get(model_name, model_name),
        supports=Supports(
            multiturn=False,
            media=False,
            tools=False,
            system_role=False,
            output=['media'],
        ),
    )

    async def imagen_runner(
        request: ModelRequest,
        ctx: ActionRunContext[GenerateResponseChunk] | None = None,
    ) -> ModelResponse:
        prompt = ''
        for m in request.messages:
            for p in m.content:
                if p.text:
                    prompt += p.text

        imagen_config = None
        if request.config:
            if isinstance(request.config, ImagenConfig):
                imagen_config = request.config
            elif isinstance(request.config, dict):
                imagen_config = ImagenConfig.model_validate(request.config)

        sdk_config = _to_imagen_config(imagen_config)

        try:
            async def _generate(span: SpanContext) -> Any:
                span.set_metadata({'client': 'genai'})
                return await client.aio.models.generate_images(
                    model=model_name,
                    prompt=prompt,
                    config=sdk_config,
                )

            res = await run_in_new_span(model_name, _generate, action_type='model')
        except APIError as exc:
            raise wrap_http_error(exc) from exc

        candidates = []
        for img in res.generated_images:
            part = Part.from_bytes(
                img.image.image_bytes,
                content_type=img.image.mime_type or 'image/png',
            )
            msg = Message(role=Role.MODEL, content=[part])
            candidates.append(Candidate(index=len(candidates), message=msg))

        return ModelResponse(candidates=candidates)

    return info, imagen_runner
