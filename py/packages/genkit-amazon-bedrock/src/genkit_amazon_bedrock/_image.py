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

"""Bedrock image generation implementation (InvokeModel API).

Image models predate Converse and have no unified request shape: the Amazon
families nest their options under ``imageGenerationConfig`` while the modern
Stability models take flat top-level fields, so routing by model ID is
unavoidable.

The legacy SDXL shape (``stable-diffusion``, ``text_prompts``/``artifacts``) is
not supported, because every model of that shape is end-of-life on Bedrock. The
response MIME type follows the requested ``output_format`` rather than a
hardcoded ``image/png``, which would mislabel a jpeg or webp response.
"""

import json
from typing import Any, Literal, cast

from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, ValidationError

from genkit import FinishReason, GenkitError, Message, ModelResponse, Part, Role, get_logger
from genkit.model import ModelContext, ModelRequest
from genkit_amazon_bedrock._config import ImageConfig, NovaCanvasConfig, StabilityImageConfig, TitanImageConfig
from genkit_amazon_bedrock._embedders import InvokeModelTransport
from genkit_amazon_bedrock._model_info import strip_inference_profile_prefix
from genkit_amazon_bedrock._models import _from_botocore_error, _from_client_error

logger = get_logger(__name__)

ImageFamily = Literal['titan_image', 'nova_canvas', 'stability']

# Substring matches over the base model ID, most specific first.
_IMAGE_FAMILY_PATTERNS: tuple[tuple[str, ImageFamily], ...] = (
    ('titan-image', 'titan_image'),
    ('nova-canvas', 'nova_canvas'),
    # Not bare 'stable-image': that would swallow the stability.stable-image-*
    # editing services (inpaint, erase-object, search-replace, ...) which are
    # not text-to-image and take entirely different request bodies.
    ('sd3-', 'stability'),
    ('stable-image-core', 'stability'),
    ('stable-image-ultra', 'stability'),
)

# Amazon Titan Image and Nova Canvas only ever return PNG.
_AMAZON_IMAGE_MIME = 'image/png'

_NO_IMAGES_MESSAGE = 'bedrock image: no images generated'

_FAMILY_CONFIG: dict[ImageFamily, type[ImageConfig]] = {
    'titan_image': TitanImageConfig,
    'nova_canvas': NovaCanvasConfig,
    'stability': StabilityImageConfig,
}

# Every declared field but extra reaches the body; extra is merged separately.
_STABILITY_IMAGE_FIELDS: set[str] = set(StabilityImageConfig.model_fields) - {'extra'}


def image_family(model_id: str) -> ImageFamily | None:
    """Routes a model ID to its image family, or None if it is not one."""
    base_id = strip_inference_profile_prefix(model_id)
    for pattern, family in _IMAGE_FAMILY_PATTERNS:
        if pattern in base_id:
            return family
    return None


def _managed_extra_field(extra: dict[str, Any], family: ImageFamily) -> str | None:
    """The ``extra`` key that would overwrite a field Genkit builds from the prompt, if any.

    Stability takes the prompt as ``prompt``. Titan Image and Nova Canvas take
    it as ``textToImageParams.text`` under ``taskType: TEXT_IMAGE``; other
    ``textToImageParams`` entries (``negativeText``) are the caller's.
    """
    if family == 'stability':
        return 'prompt' if 'prompt' in extra else None
    if 'taskType' in extra:
        return 'taskType'
    if 'textToImageParams' in extra:
        params = extra['textToImageParams']
        if not isinstance(params, dict):
            # A non-mapping would replace the whole object, prompt included.
            return 'textToImageParams'
        if 'text' in params:
            return 'textToImageParams.text'
    return None


def _deep_merge(body: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Merges ``extra`` over ``body``; nested objects merge key by key, other values replace."""
    merged = dict(body)
    for key, value in extra.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(cast(dict[str, Any], current), cast(dict[str, Any], value))
        else:
            merged[key] = value
    return merged


def merge_image_extra(body: dict[str, Any], extra: dict[str, Any], family: ImageFamily) -> dict[str, Any]:
    """Merges ``config.extra`` into a built InvokeModel image body.

    Keys are Bedrock wire names, matched exactly. Nested objects merge key by
    key, so ``{'imageGenerationConfig': {'seed': 7}}`` keeps the other
    defaults, and a colliding leaf takes the caller's value. Fields Genkit
    builds from the prompt are rejected rather than overwritten.

    Args:
        body: The body built from the prompt and declared config.
        extra: The caller's ``extra`` mapping.
        family: The image family the body was built for.

    Returns:
        A new body; ``body`` is not modified.

    Raises:
        GenkitError: INVALID_ARGUMENT when ``extra`` names a prompt field.
    """
    field = _managed_extra_field(extra, family)
    if field is not None:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=(
                f'bedrock image: extra field {field!r} is built by Genkit from the request '
                'and cannot be set from config'
            ),
        )
    return _deep_merge(body, extra)


def is_image_model(model_id: str) -> bool:
    """Reports whether a Bedrock model ID names a text-to-image model.

    Args:
        model_id: Bedrock model ID, inference-profile ID, or ARN.

    Returns:
        True when the ID routes to a known image family.
    """
    return image_family(model_id) is not None


def image_prompt(request: ModelRequest[Any]) -> str:
    """Extracts the image prompt from the most recent user message.

    Messages are scanned newest first, only user messages count, and their text
    parts are concatenated without a separator. A user message with no text is
    skipped rather than
    ending the scan. Non-text parts are ignored; these models are text-to-image
    only.

    Args:
        request: The Genkit model request.

    Returns:
        The prompt text, or an empty string when there is none.
    """
    for message in reversed(request.messages):
        if message.role != Role.USER:
            continue
        prompt = ''.join(part.text for part in message.content if part.text is not None)
        if prompt:
            return prompt
    return ''


def build_amazon_image_body(prompt: str, config: TitanImageConfig | NovaCanvasConfig) -> dict[str, Any]:
    """Builds the InvokeModel body for Titan Image and Nova Canvas.

    The declared fields are nested under ``imageGenerationConfig`` by their
    wire names, over the defaults. ``quality`` defaults only for Nova Canvas;
    Titan Image sends it only when set. ``extra`` is merged over the result by
    the caller.

    Args:
        prompt: The text prompt.
        config: The validated per-call config.

    Returns:
        The request body to marshal.
    """
    image_generation_config: dict[str, Any] = {
        'numberOfImages': 1,
        'height': 1024,
        'width': 1024,
        'cfgScale': 8.0,
        'seed': 0,
    }
    if isinstance(config, NovaCanvasConfig):
        image_generation_config['quality'] = 'standard'
    # Read the fields off the config's own class: Titan and Nova share this builder.
    fields = set(type(config).model_fields) - {'extra'}
    image_generation_config.update(config.model_dump(by_alias=True, exclude_none=True, include=fields))

    return {
        'taskType': 'TEXT_IMAGE',
        'textToImageParams': {'text': prompt},
        'imageGenerationConfig': image_generation_config,
    }


def build_stability_image_body(prompt: str, config: StabilityImageConfig) -> dict[str, Any]:
    """Builds the InvokeModel body for the modern Stability models.

    The declared fields are sent at the top level under their field names,
    which are Stability's wire names, over the defaults.

    Args:
        prompt: The text prompt.
        config: The validated per-call config.

    Returns:
        The request body to marshal.
    """
    body: dict[str, Any] = {'prompt': prompt, 'output_format': 'png'}
    body.update(config.model_dump(exclude_none=True, include=_STABILITY_IMAGE_FIELDS))
    return body


def image_config_class(model_id: str) -> type[ImageConfig] | None:
    """The config class an image model takes, or None if the ID is not an image model.

    Args:
        model_id: Bedrock model ID, inference-profile ID, or ARN.

    Returns:
        The per-family config class.
    """
    family = image_family(model_id)
    return _FAMILY_CONFIG[family] if family is not None else None


def _validate_image_config(config: object, schema: type[ImageConfig]) -> ImageConfig:
    """Coerces the request config into the family's config class.

    Genkit has already checked the config against ``schema`` before the
    handler runs; this is the conversion, and the check for direct calls that
    skip the framework.

    Args:
        config: The raw ``request.config`` value.
        schema: The family's config class.

    Returns:
        A ``schema`` instance; defaults when no config was given.

    Raises:
        GenkitError: INVALID_ARGUMENT for an unsupported config type or a
            config that does not fit ``schema``.
    """
    if config is None:
        return schema()
    if isinstance(config, schema):
        return config
    if isinstance(config, BaseModel):
        raw: object = config.model_dump(by_alias=True, exclude_none=True)
    elif isinstance(config, dict):
        raw = config
    else:
        raise GenkitError(
            message=f'bedrock image: unexpected config type {type(config).__name__}, want a mapping or pydantic model',
            status='INVALID_ARGUMENT',
        )
    try:
        return schema.model_validate(raw)
    except ValidationError as error:
        raise GenkitError(message=f'bedrock image: invalid config: {error}', status='INVALID_ARGUMENT') from error


def _response_images(payload: dict[str, Any]) -> list[Any]:
    """Reads the ``images`` array off an InvokeModel response body."""
    images = payload.get('images')
    return images if isinstance(images, list) else []


def _media_parts(images: list[Any], mime: str) -> list[Part]:
    """Wraps base64 image strings as Genkit media parts, skipping blanks."""
    return [
        Part.from_media(f'data:{mime};base64,{image}', content_type=mime)
        for image in images
        if isinstance(image, str) and image
    ]


class BedrockImageModel:
    """Handles a generate call for one Bedrock image model."""

    def __init__(self, model_id: str, transport: InvokeModelTransport) -> None:
        """Initializes the image model handler.

        Args:
            model_id: Bedrock model ID, inference-profile ID, or ARN, sent to
                the InvokeModel API verbatim.
            transport: The shared transport seam owning the boto3 client.
        """
        self._model_id = model_id
        self._transport = transport

    async def generate(self, request: ModelRequest[Any], ctx: ModelContext | None = None) -> ModelResponse:
        """Runs an InvokeModel call and returns the generated images.

        Args:
            request: The Genkit model request.
            ctx: Action run context, accepted for signature parity and never
                used: image generation emits no chunks, only a final response.

        Returns:
            A ModelResponse carrying one media part per generated image.

        Raises:
            GenkitError: UNIMPLEMENTED for a model this plugin cannot generate
                images with, INVALID_ARGUMENT for a request carrying no prompt
                or an unusable config, INTERNAL when the service returns no
                usable image, and the mapped AWS status for a failed call.
        """
        # ctx is never used: there is nothing to stream, only a final response.
        family = image_family(self._model_id)
        if family is None:
            raise GenkitError(
                message=f'bedrock image: unsupported image generation model {self._model_id!r}',
                status='UNIMPLEMENTED',
            )
        prompt = image_prompt(request)
        if not prompt:
            raise GenkitError(
                message='bedrock image: no text prompt found for image generation',
                status='INVALID_ARGUMENT',
            )
        config = _validate_image_config(request.config, _FAMILY_CONFIG[family])
        logger.debug('Bedrock image request', model=self._model_id, family=family)

        if isinstance(config, StabilityImageConfig):
            body = build_stability_image_body(prompt, config)
        else:
            body = build_amazon_image_body(prompt, config)
        # extra is the caller's escape hatch for fields the plugin doesn't
        # model, e.g. Nova Canvas textToImageParams.negativeText.
        body = merge_image_extra(body, config.extra or {}, family)
        payload = await self._invoke(body)

        if family == 'stability':
            # extra may set output_format in any casing; only the MIME is lowered.
            output_format = str(body.get('output_format') or 'png').lower()
            mime = f'image/{output_format}'
            images = self._stability_images(payload)
        else:
            mime = _AMAZON_IMAGE_MIME
            images = self._amazon_images(payload)

        parts = _media_parts(images, mime)
        if not parts:
            raise GenkitError(message=_NO_IMAGES_MESSAGE, status='INTERNAL')
        logger.debug('Bedrock image response', model=self._model_id, images=len(parts), mime=mime)
        return ModelResponse(
            message=Message(role=Role.MODEL, content=parts),
            # Image generation always stops on its own and reports no tokens.
            finish_reason=FinishReason.STOP,
            request=request,
        )

    def _amazon_images(self, payload: dict[str, Any]) -> list[Any]:
        """Reads the images off a Titan Image or Nova Canvas response.

        Raises:
            GenkitError: INTERNAL when no image came back, carrying the
                response's in-band ``error`` text when there is one. Images
                and an error together mean partial success: Nova Canvas drops
                individually content-filtered images that way, so the images
                win.
        """
        images = _response_images(payload)
        if any(isinstance(image, str) and image for image in images):
            return images
        # In-band failure on an HTTP 200, as with Titan multimodal embeddings.
        error = payload.get('error')
        message = f'bedrock image: {error}' if isinstance(error, str) and error else _NO_IMAGES_MESSAGE
        raise GenkitError(message=message, status='INTERNAL')

    def _stability_images(self, payload: dict[str, Any]) -> list[Any]:
        """Reads the images off a modern Stability response.

        ``seeds`` is ignored. ``finish_reasons`` is checked before the images,
        so a refusal is reported by its reason rather than as a missing image.
        JSON ``null`` is the success value on that field.

        Raises:
            GenkitError: INTERNAL for a non-success finish reason, or when no
                image came back.
        """
        for reason in payload.get('finish_reasons') or []:
            if reason is not None and reason not in ('', 'SUCCESS'):
                raise GenkitError(
                    message=f'bedrock image: image generation finished with reason: {reason}',
                    status='INTERNAL',
                )
        images = _response_images(payload)
        if not images:
            raise GenkitError(message=_NO_IMAGES_MESSAGE, status='INTERNAL')
        return images

    async def _invoke(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            return await self._transport.invoke_model(
                modelId=self._model_id,
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json',
            )
        except ClientError as error:
            raise _from_client_error(error, 'invoke model') from error
        except BotoCoreError as error:
            raise _from_botocore_error(error, 'invoke model') from error
