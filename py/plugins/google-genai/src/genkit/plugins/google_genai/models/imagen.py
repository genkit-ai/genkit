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

"""Imagen image generation models for Google GenAI plugin.

Imagen is Google's image generation model family that creates images from text
prompts. It supports both Google AI and Vertex AI backends.

Architecture:
    ```
    ┌──────────────────────────────────────────────────────────────────────┐
    │                     Imagen Image Generation Flow                      │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                       │
    │   Input                    Model                     Output           │
    │   ┌─────────┐             ┌─────────┐             ┌─────────┐        │
    │   │ Text    │ ─predict──► │ Imagen  │ ──────────► │ Image   │        │
    │   │ Prompt  │             │ Model   │             │ (bytes) │        │
    │   └─────────┘             └─────────┘             └─────────┘        │
    │                                                                       │
    └──────────────────────────────────────────────────────────────────────┘
    ```

Note:
    Imagen models may be discovered dynamically through the SDK models.list()
    API. Known Google AI Imagen models are also registered explicitly because
    models.list() does not always surface them in every environment.

Supported Models:
    Google AI:
        - imagen-4.0-generate-001
        - imagen-4.0-fast-generate-001
        - imagen-4.0-ultra-generate-001

    Vertex AI:
        - Imagen models returned by models.list()
"""

import base64
import json
import sys

if sys.version_info < (3, 11):
    from strenum import StrEnum
else:
    from enum import StrEnum

from functools import cached_property
from typing import Annotated, Any

from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, WithJsonSchema

from genkit import (
    GenkitError,
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


def _to_dict(obj: Any) -> Any:  # noqa: ANN401
    """Convert object to dict if it's a Pydantic model, otherwise return as-is."""
    return obj.model_dump(mode='json') if isinstance(obj, BaseModel) else obj


class ImagenVersion(StrEnum):
    """Supported text-to-image models."""

    IMAGEN3 = 'imagen-3.0-generate-002'
    IMAGEN3_FAST = 'imagen-3.0-fast-generate-001'
    IMAGEN2 = 'imagegeneration@006'
    IMAGEN4 = 'imagen-4.0-generate-001'
    IMAGEN4_FAST = 'imagen-4.0-fast-generate-001'
    IMAGEN4_ULTRA = 'imagen-4.0-ultra-generate-001'


# Imagen models available on the Google AI (Gemini API) backend. Hardcoded
# here because the SDK's client.models.list() does not always surface them on
# every environment, and they must be visible in Init / list_actions to be
# selectable from user code.
GOOGLEAI_KNOWN_IMAGEN_MODELS: tuple[str, ...] = (
    ImagenVersion.IMAGEN4,
    ImagenVersion.IMAGEN4_FAST,
    ImagenVersion.IMAGEN4_ULTRA,
)


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
    ImagenVersion.IMAGEN2: ModelInfo(
        label='Vertex AI - Imagen2',
        supports=Supports(
            media=False,
            multiturn=False,
            tools=False,
            system_role=True,
            output=['media'],
        ),
    ),
    ImagenVersion.IMAGEN4: ModelInfo(
        label='Google AI - Imagen 4',
        supports=Supports(
            media=True,
            multiturn=False,
            tools=False,
            system_role=True,
            output=['media'],
        ),
    ),
    ImagenVersion.IMAGEN4_FAST: ModelInfo(
        label='Google AI - Imagen 4 Fast',
        supports=Supports(
            media=True,
            multiturn=False,
            tools=False,
            system_role=True,
            output=['media'],
        ),
    ),
    ImagenVersion.IMAGEN4_ULTRA: ModelInfo(
        label='Google AI - Imagen 4 Ultra',
        supports=Supports(
            media=True,
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


class ImagenAspectRatio(StrEnum):
    """Imagen output aspect ratios."""

    RATIO_1_1 = '1:1'
    RATIO_3_4 = '3:4'
    RATIO_4_3 = '4:3'
    RATIO_9_16 = '9:16'
    RATIO_16_9 = '16:9'


class ImagenImageSize(StrEnum):
    """Imagen output image sizes."""

    SIZE_1K = '1K'
    SIZE_2K = '2K'


class ImagenPersonGeneration(StrEnum):
    """Controls whether Imagen may generate images of people."""

    DONT_ALLOW = 'dont_allow'
    ALLOW_ADULT = 'allow_adult'
    ALLOW_ALL = 'allow_all'


def is_imagen_model(name: str) -> bool:
    """Return True when the model name refers to an Imagen image model."""
    lower = name.lower()
    return lower.startswith('imagen-') or lower.startswith('imagegeneration')


def vertexai_image_model_info(
    version: str,
) -> ModelInfo:
    """Generates a ModelInfo object for the Vertex AI backend.

    Args:
        version: Version of the model.

    Returns:
        ModelInfo object.
    """
    return ModelInfo(
        label=f'Vertex AI - {version}',
        supports=DEFAULT_IMAGE_SUPPORT,
    )


def googleai_image_model_info(
    version: str,
) -> ModelInfo:
    """Generates a ModelInfo object for the Google AI (Gemini API) backend.

    Args:
        version: Version of the model.

    Returns:
        ModelInfo object with a Google AI-prefixed label.
    """
    return ModelInfo(
        label=f'Google AI - {version}',
        supports=DEFAULT_IMAGE_SUPPORT,
    )


class ImagenConfigSchema(BaseModel):
    """Imagen Config Schema."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)

    number_of_images: int | None = Field(
        None,  # Don't silently change default behavior
        alias='numberOfImages',
        ge=1,
        le=4,
        description='Number of images to generate.',
    )
    aspect_ratio: str | None = Field(
        None,
        alias='aspectRatio',
        description=(
            'Aspect ratio of the generated images (e.g. "1:1", "16:9", "9:16"). '
            'Any ratio string supported by the model may be used.'
        ),
    )
    person_generation: Annotated[
        ImagenPersonGeneration | None,
        WithJsonSchema({
            'type': 'string',
            'enum': [mode.value for mode in ImagenPersonGeneration],
            'description': 'Allows generation of people by the model.',
        }),
    ] = Field(None, alias='personGeneration')
    image_size: Annotated[
        ImagenImageSize | None,
        WithJsonSchema({
            'type': 'string',
            'enum': [size.value for size in ImagenImageSize],
            'description': 'Largest dimension of the generated image.',
        }),
    ] = Field(None, alias='imageSize')


# Alias for backwards compatibility with __init__.py exports.
ImagenConfig = ImagenConfigSchema


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
                    raise GenkitError(
                        status='INVALID_ARGUMENT',
                        message='Non-text messages are not supported for Imagen models.',
                    )
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
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message='Tools are not supported for Imagen models.',
            )

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

    def _get_config(self, request: ModelRequest) -> genai_types.GenerateImagesConfigOrDict | None:
        if request.config is None:
            return None

        request_config = self._normalize_config(request.config)
        ta = TypeAdapter(genai_types.GenerateImagesConfigOrDict)
        try:
            request_config = ImagenConfigSchema.model_validate(request_config).model_dump(
                mode='json',
                exclude_none=True,
            )
            if not request_config:
                return None
            return ta.validate_python(request_config)
        except ValidationError as e:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message='The configuration dictionary is invalid. Refer to the documentation for available fields.',
                cause=e,
            ) from e

    def _normalize_config(self, config: object) -> dict[str, object]:
        """Normalize supported config inputs without adding SDK defaults."""
        if isinstance(config, BaseModel):
            request_config = config.model_dump(exclude_none=True)
        elif isinstance(config, dict):
            request_config = dict(config)
        else:
            validated_config = TypeAdapter(genai_types.GenerateImagesConfigOrDict).validate_python(config)
            request_config = _to_dict(validated_config)

        return request_config

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
        """Get model metadata.

        Returns:
            model metadata.
        """
        supports = {}
        if self._version in SUPPORTED_MODELS:
            model_supports = SUPPORTED_MODELS[self._version].supports  # pyrefly: ignore[bad-index]
            if model_supports:
                supports = model_supports.model_dump(by_alias=True)
        else:
            model_supports = vertexai_image_model_info(self._version).supports
            if model_supports:
                supports = model_supports.model_dump(by_alias=True)

        return {'model': {'supports': supports}}
