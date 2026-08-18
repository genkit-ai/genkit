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

"""Imagen model implementation for Google GenAI plugin."""

import base64
import sys

if sys.version_info < (3, 11):
    from strenum import StrEnum
else:
    from enum import StrEnum

import json
from functools import cached_property
from typing import Any, Literal, TypeAlias

from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, ConfigDict, ValidationError

from genkit import (
    Media,
    MediaPart,
    Message,
    ModelInfo,
    ModelRequest,
    ModelResponse,
    Part,
    Role,
    Supports,
    TextPart,
)
from genkit.plugin_api import ActionRunContext, tracer
from genkit_google_genai.models._sdk_config import (
    attach_leftovers,
    dump_family_config,
    sdk_config_error,
    split_sdk_fields,
)


def _to_dict(obj: Any) -> Any:  # noqa: ANN401
    """Convert object to dict if it's a Pydantic model, otherwise return as-is."""
    return obj.model_dump() if isinstance(obj, BaseModel) else obj


class ImagenVersion(StrEnum):
    """Supported text-to-image models."""

    IMAGEN3 = 'imagen-3.0-generate-002'
    IMAGEN3_FAST = 'imagen-3.0-fast-generate-001'


# Quote autocomplete needs a Literal. The enum above is the catalog; a test
# requires these members and the enum values to be the same set.
KnownImagen: TypeAlias = Literal[
    'imagen-3.0-generate-002',
    'imagen-3.0-fast-generate-001',
]


SUPPORTED_MODELS = {
    ImagenVersion.IMAGEN3: ModelInfo(
        label='Vertex AI - Imagen3',
        supports=Supports(
            media=True,
            multiturn=False,
            tools=False,
            system_role=True,
            output=['media'],
        ),
    ),
    ImagenVersion.IMAGEN3_FAST: ModelInfo(
        label='Vertex AI - Imagen3 Fast',
        supports=Supports(
            media=False,
            multiturn=False,
            tools=False,
            system_role=True,
            output=['media'],
        ),
    ),
}

DEFAULT_IMAGE_SUPPORT = Supports(
    media=True,
    multiturn=False,
    tools=False,
    system_role=True,
    output=['media'],
)


def is_imagen_model_name(name: str) -> bool:
    """Return True if ``name`` is an Imagen model.

    Imagen ids start with ``imagen-`` on the local name after stripping the
    plugin / ``models/`` prefix. Gemini native image (``gemini-…-image``) is
    not Imagen.
    """
    return name.split('/')[-1].lower().startswith('imagen-')


def is_unsupported_image_model_name(name: str) -> bool:
    """Return True for image ids that must not route anywhere.

    ``imagegeneration@*`` and ``imagetext@*`` were shut down by Google in
    June 2026, and ``virtual-try-on-*`` needs a person+product image request
    shape this plugin does not implement. Letting any of those fall through
    to the Gemini default would answer with the wrong model, so callers
    treat these ids as not-a-model instead.
    """
    local = name.split('/')[-1].lower()
    return local.startswith('imagegeneration@') or local.startswith('imagetext@') or local.startswith('virtual-try-on-')


def vertexai_image_model_info(
    version: str,
) -> ModelInfo:
    """Generates a ModelInfo object.

    This function tries to get the best ModelInfo Supports
    for the given version.

    Args:
        version: Version of the model.

    Returns:
        ModelInfo object.
    """
    return ModelInfo(
        label=f'Vertex AI - {version}',
        supports=DEFAULT_IMAGE_SUPPORT,
    )


class ImagenConfigSchema(BaseModel):
    """Imagen Config Schema."""

    model_config = ConfigDict(extra='allow')


class ImagenModel:
    """Imagen text-to-image model."""

    def __init__(self, version: str | ImagenVersion, client: genai.Client) -> None:
        """Initialize Imagen model.

        Args:
            version: Imagen version
            client: Google AI client
        """
        self._version = version
        self._client = client

    def _build_prompt(self, request: ModelRequest) -> str:
        """Build prompt request from Genkit request.

        Args:
            request: Genkit request.

        Returns:
            prompt for Imagen
        """
        prompt = []
        for message in request.messages:
            for part in message.content:
                if isinstance(part.root, TextPart):
                    prompt.append(part.root.text)
                else:
                    raise ValueError('Non-text messages are not supported')
        return ' '.join(prompt)

    async def generate(self, request: ModelRequest, _: ActionRunContext) -> ModelResponse:
        """Handle a generation request.

        Args:
            request: The generation request containing messages and parameters.
            _: action context

        Returns:
            The model's response to the generation request.
        """
        prompt = self._build_prompt(request)
        config = self._get_config(request)
        if request.tools:
            raise ValueError('Tools are not supported for this model.')

        with tracer.start_as_current_span('generate_images') as span:
            span.set_attribute(
                'genkit:input',
                json.dumps({
                    'config': _to_dict(config),
                    'contents': prompt,
                    'model': self._version,
                }),
            )
            response = await self._client.aio.models.generate_images(model=self._version, prompt=prompt, config=config)
            span.set_attribute('genkit:output', json.dumps(_to_dict(response), default=str))

        content = self._contents_from_response(response)

        return ModelResponse(
            message=Message(
                content=content,
                role=Role.MODEL,
            )
        )

    def _get_config(self, request: ModelRequest) -> genai_types.GenerateImagesConfig | None:
        dumped = dump_family_config(
            config=request.config,
            expected_type=ImagenConfigSchema,
            action_name=self._version,
        )
        if not dumped:
            return None

        known, leftovers = split_sdk_fields(dumped, genai_types.GenerateImagesConfig)
        try:
            cfg = genai_types.GenerateImagesConfig(**known) if known else genai_types.GenerateImagesConfig()
        except ValidationError as e:
            raise sdk_config_error(action_name=self._version, error=e) from e
        return attach_leftovers(cfg, leftovers, nest='parameters')

    def _contents_from_response(self, response: genai_types.GenerateImagesResponse) -> list:
        """Retrieve contents from google-genai response.

        Args:
            response: google-genai response.

        Returns:
            list of generated contents.
        """
        content = []
        if response.generated_images:
            for image in response.generated_images:
                if image.image and image.image.image_bytes:
                    b64_data = base64.b64encode(image.image.image_bytes).decode('utf-8')
                    content.append(
                        Part(
                            root=MediaPart(
                                media=Media(
                                    url=f'data:{image.image.mime_type};base64,{b64_data}',
                                    content_type=image.image.mime_type,
                                )
                            )
                        )
                    )

        return content

    @cached_property
    def metadata(self) -> dict:
        """Model metadata.

        Returns:
            model metadata.
        """
        supports = {}
        if self._version in SUPPORTED_MODELS:
            model_supports = SUPPORTED_MODELS[self._version].supports  # pyright: ignore[reportArgumentType]
            if model_supports:
                supports = model_supports.model_dump(by_alias=True)
        else:
            model_supports = vertexai_image_model_info(self._version).supports
            if model_supports:
                supports = model_supports.model_dump(by_alias=True)

        return {'model': {'supports': supports}}
