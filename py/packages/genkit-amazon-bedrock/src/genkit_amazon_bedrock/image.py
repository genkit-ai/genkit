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

import structlog
from botocore.exceptions import BotoCoreError, ClientError

from genkit import (
    FinishReason,
    Media,
    MediaPart,
    Message,
    ModelRequest,
    ModelResponse,
    Part,
    Role,
    TextPart,
)
from genkit.plugin_api import ActionRunContext, GenkitError, ModelConfig
from genkit_amazon_bedrock.embedders import InvokeModelTransport
from genkit_amazon_bedrock.model_info import strip_inference_profile_prefix
from genkit_amazon_bedrock.models import _from_botocore_error, _from_client_error

logger = structlog.get_logger(__name__)

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

# Both spellings of Genkit's own generation knobs: none is a Bedrock image
# parameter, and apiKey on an outbound body would leak a credential.
_GENKIT_CONFIG_KEYS: frozenset[str] = frozenset(
    key for name, field in ModelConfig.model_fields.items() for key in (name, field.alias) if key is not None
)


def _image_family(model_id: str) -> ImageFamily | None:
    """Routes a model ID to its image family, or None if it is not one."""
    base_id = strip_inference_profile_prefix(model_id)
    for pattern, family in _IMAGE_FAMILY_PATTERNS:
        if pattern in base_id:
            return family
    return None


def is_image_model(model_id: str) -> bool:
    """Reports whether a Bedrock model ID names a text-to-image model.

    Args:
        model_id: Bedrock model ID, inference-profile ID, or ARN.

    Returns:
        True when the ID routes to a known image family.
    """
    return _image_family(model_id) is not None


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
        prompt = ''.join(part.root.text for part in message.content if isinstance(part.root, TextPart))
        if prompt:
            return prompt
    return ''


def build_amazon_image_body(prompt: str, config: dict[str, Any], *, include_quality: bool) -> dict[str, Any]:
    """Builds the InvokeModel body for Titan Image and Nova Canvas.

    Only ``imageGenerationConfig`` is honoured, and only when it is a mapping;
    every other config key is dropped. Its entries are
    merged key-by-key over the defaults. ``image_generation_config`` is accepted
    as an alternative spelling, as BedrockConfig accepts both casings, and the
    camelCase key wins when a caller passes both.

    Args:
        prompt: The text prompt.
        config: The normalized per-call config.
        include_quality: Whether to default ``quality``; Nova Canvas takes it,
            Titan Image does not.

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
    if include_quality:
        image_generation_config['quality'] = 'standard'
    # Keyed on presence, not value, so the camelCase wire name wins over both.
    key = 'imageGenerationConfig' if 'imageGenerationConfig' in config else 'image_generation_config'
    overrides = config.get(key)
    if isinstance(overrides, dict):
        image_generation_config.update(overrides)

    return {
        'taskType': 'TEXT_IMAGE',
        'textToImageParams': {'text': prompt},
        'imageGenerationConfig': image_generation_config,
    }


def build_stability_image_body(prompt: str, config: dict[str, Any]) -> dict[str, Any]:
    """Builds the InvokeModel body for the modern Stability models.

    The whole config is merged over the defaults at the top level, so a caller
    can override ``prompt`` and ``output_format`` as well as add flat fields
    like ``aspect_ratio`` or ``seed``. Genkit's generic generation knobs are
    already gone by here; see ``_normalize_image_config``.

    Args:
        prompt: The text prompt.
        config: The normalized per-call config.

    Returns:
        The request body to marshal.
    """
    body: dict[str, Any] = {'prompt': prompt, 'output_format': 'png'}
    body.update(config)
    return body


def _image_config_dict(config: dict[str, Any]) -> dict[str, Any]:
    """Copies a config dict, dropping Genkit's generic generation knobs."""
    return {key: value for key, value in config.items() if key not in _GENKIT_CONFIG_KEYS}


def _normalize_image_config(config: Any) -> dict[str, Any]:  # noqa: ANN401
    """Coerces the request config into a plain dict.

    Deliberately not ``converters.normalize_config``: that returns a
    BedrockConfig rather than a plain dict, and its declared Converse fields
    (``maxTokens``, ``toolChoice``, ...) would end up on an image body.

    Genkit's own generation knobs (``temperature``, ``apiKey``, and the rest of
    the common config) are dropped in either spelling: the framework coerces
    every config into ModelConfig, and Bedrock's image APIs take none of them.

    Args:
        config: The raw ``request.config`` value.

    Returns:
        The config as a mutable dict, minus Genkit's generic knobs; empty when
        none was given.

    Raises:
        GenkitError: INVALID_ARGUMENT for unsupported config types. Failing
            loudly beats sending a body that quietly ignores the caller's
            settings.
    """
    if config is None:
        return {}
    if isinstance(config, dict):
        return _image_config_dict(config)
    dump = getattr(config, 'model_dump', None)
    if callable(dump):
        return _image_config_dict(cast(dict[str, Any], dump(exclude_none=True)))
    raise GenkitError(
        message=f'bedrock image: unexpected config type {type(config).__name__}, want a mapping or pydantic model',
        status='INVALID_ARGUMENT',
    )


def _response_images(payload: dict[str, Any]) -> list[Any]:
    """Reads the ``images`` array off an InvokeModel response body."""
    images = payload.get('images')
    return images if isinstance(images, list) else []


def _media_parts(images: list[Any], mime: str) -> list[Part]:
    """Wraps base64 image strings as Genkit media parts, skipping blanks."""
    return [
        Part(root=MediaPart(media=Media(url=f'data:{mime};base64,{image}', content_type=mime)))
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

    async def generate(self, request: ModelRequest[Any], ctx: ActionRunContext | None = None) -> ModelResponse:
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
        family = _image_family(self._model_id)
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
        config = _normalize_image_config(request.config)
        logger.debug('Bedrock image request', model=self._model_id, family=family)

        if family == 'stability':
            body = build_stability_image_body(prompt, config)
            payload = await self._invoke(body)
            # The wire body keeps the caller's casing; only the MIME is lowered.
            output_format = str(body.get('output_format') or 'png').lower()
            mime = f'image/{output_format}'
            images = self._stability_images(payload)
        else:
            body = build_amazon_image_body(prompt, config, include_quality=family == 'nova_canvas')
            payload = await self._invoke(body)
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
