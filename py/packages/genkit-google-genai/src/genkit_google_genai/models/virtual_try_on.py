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

"""Virtual Try-On image model for the Google GenAI plugin.

Virtual Try-On dresses a person in one or more products and returns the
composited images. It is a Vertex AI model; there is no Gemini API backend
for it.

Both inputs arrive as media parts on the last message, tagged so the model
knows which is which::

    Part(MediaPart(media=Media(url=person_url), metadata={'type': 'personImage'}))
    Part(MediaPart(media=Media(url=shirt_url), metadata={'type': 'productImage'}))

Each url is either a ``data:`` url, a bare base64 payload, or a ``gs://``
Cloud Storage path.
"""

import base64
import binascii
import sys
from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

if sys.version_info < (3, 11):
    from strenum import StrEnum
else:
    from enum import StrEnum

from google import genai
from google.genai import types as genai_types
from google.genai.errors import APIError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
)
from genkit.model import FinishReason, get_basic_usage_stats
from genkit.plugin_api import ActionRunContext, wrap_http_error
from genkit_google_genai.constants import is_multi_regional_location, multi_regional_base_url
from genkit_google_genai.models._sdk_config import (
    attach_leftovers,
    dump_family_config,
    sdk_config_error,
    split_sdk_fields,
)
from genkit_google_genai.models._secrets import context_api_key, reject_request_config_api_key


class VirtualTryOnVersion(StrEnum):
    """Supported Virtual Try-On models."""

    VIRTUAL_TRY_ON_001 = 'virtual-try-on-001'


# Quote autocomplete needs a Literal. The enum above is the catalog; a test
# requires these members and the enum values to be the same set.
KnownVirtualTryOn: TypeAlias = Literal['virtual-try-on-001']

# The Vertex catalog does not report a usable action for this family, so the
# plugin advertises and registers it from this list instead of discovery.
VERTEX_KNOWN_VIRTUAL_TRY_ON: tuple[str, ...] = (VirtualTryOnVersion.VIRTUAL_TRY_ON_001,)

PERSON_IMAGE_TYPE = 'personImage'
PRODUCT_IMAGE_TYPE = 'productImage'


def is_virtual_try_on_model(name: str) -> bool:
    """Check if a model name is a Virtual Try-On model.

    Args:
        name: The model name to check.

    Returns:
        True if this is a Virtual Try-On model name.
    """
    return name.split('/')[-1].lower().startswith('virtual-try-on-')


class VirtualTryOnOutputOptions(BaseModel):
    """Encoding of the returned images."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)
    mime_type: str | None = Field(default=None, alias='mimeType', description='MIME type of the returned images.')
    compression_quality: int | None = Field(
        default=None,
        alias='compressionQuality',
        ge=0,
        le=100,
        description='Compression quality for lossy output formats.',
    )


class VirtualTryOnConfig(BaseModel):
    """Virtual Try-On Config Schema."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)
    sample_count: int | None = Field(
        default=None, alias='sampleCount', ge=1, description='Number of images to generate.'
    )
    seed: int | None = Field(default=None, description='Random seed for the image generation.')
    base_steps: int | None = Field(
        default=None, alias='baseSteps', ge=1, le=100, description='Number of denoising steps.'
    )
    person_generation: str | None = Field(
        default=None,
        alias='personGeneration',
        description='Control if/how images of people are generated: "dont_allow", "allow_adult" or "allow_all".',
    )
    safety_setting: str | None = Field(
        default=None,
        alias='safetySetting',
        description='Safety filter level: "block_most", "block_some", "block_few" or "block_fewest".',
    )
    storage_uri: str | None = Field(
        default=None, alias='storageUri', description='Cloud Storage URI to store the generated images.'
    )
    output_options: VirtualTryOnOutputOptions | None = Field(
        default=None, alias='outputOptions', description='Encoding of the returned images.'
    )
    base_url: str | None = Field(default=None, alias='baseUrl', description='Override the API endpoint for this call.')
    api_version: str | None = Field(
        default=None, alias='apiVersion', description='Override the API version for this call.'
    )
    location: str | None = Field(default=None, description='Override the Vertex AI location for this call.')


DEFAULT_VIRTUAL_TRY_ON_SUPPORT = Supports(
    media=True,
    multiturn=False,
    tools=False,
    system_role=True,
    output=['media'],
)

# Config keys the typed SDK request spells differently.
_SDK_FIELD_NAMES = {
    'sample_count': 'number_of_images',
    'storage_uri': 'output_gcs_uri',
}

# Keys sent at their wire spelling through extra_body. The SDK's typed fields
# for these differ: person_generation case-folds the value onto its enum, and
# safety_filter_level is a different enum with its own vocabulary.
_VERBATIM_FIELD_NAMES = {
    'person_generation': 'personGeneration',
    'safety_setting': 'safetySetting',
}

# Config keys that pick the client rather than shape the request. They are
# read off the config before it is mapped, so they never reach the predict body.
_CLIENT_OPTION_KEYS = frozenset({'base_url', 'baseUrl', 'api_version', 'apiVersion', 'location'})

# Accept the URL-safe base64 alphabet as well as the standard one; padding
# is restored before decoding.
_BASE64_URLSAFE = str.maketrans('-_', '+/')


def virtual_try_on_model_info(version: str) -> ModelInfo:
    """Get model info for a Virtual Try-On model.

    Args:
        version: The Virtual Try-On model version.

    Returns:
        ModelInfo for the Virtual Try-On model.
    """
    return ModelInfo(
        label=f'Vertex AI - {version}',
        supports=DEFAULT_VIRTUAL_TRY_ON_SUPPORT,
    )


def _to_image(url: str) -> genai_types.Image:
    """Turn one media url into an SDK image reference.

    Cloud Storage paths are sent by reference; everything else is sent
    inline. ``mime_type`` is left unset; the API infers it from the bytes.
    """
    if url.startswith('gs://'):
        return genai_types.Image(gcs_uri=url)
    if url.startswith(('http://', 'https://')):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message='Virtual Try-On does not support http(s) URIs. Please specify a Cloud Storage URI.',
        )
    payload = url.partition(',')[2] if url.startswith('data:') else url
    payload = ''.join(payload.split()).translate(_BASE64_URLSAFE)
    payload += '=' * (-len(payload) % 4)
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message='Virtual Try-On image url is not a data url, a Cloud Storage URI, or base64 image data.',
        ) from e
    if not data:
        raise GenkitError(status='INVALID_ARGUMENT', message='Virtual Try-On image url carries no image data.')
    return genai_types.Image(image_bytes=data)


def _tagged_media_urls(request: ModelRequest, part_type: str) -> list[str]:
    """Media urls on the last message tagged with ``metadata['type']``.

    Only the last message is read: earlier turns are conversation history,
    and a single request dresses one person.
    """
    if not request.messages:
        return []
    urls: list[str] = []
    for part in request.messages[-1].content:
        root = part.root
        if not isinstance(root, MediaPart):
            continue
        metadata = root.metadata
        if isinstance(metadata, dict) and metadata.get('type') == part_type:
            urls.append(root.media.url)
    return urls


def _missing_media_error(part_type: str) -> GenkitError:
    """Name the tag the caller left off the last message."""
    return GenkitError(
        status='INVALID_ARGUMENT',
        message=f"Virtual Try-On needs a media part on the last message tagged metadata={{'type': '{part_type}'}}.",
    )


def _to_source(request: ModelRequest) -> genai_types.RecontextImageSource:
    """Build the person/product inputs from the tagged media parts."""
    person_urls = _tagged_media_urls(request, PERSON_IMAGE_TYPE)
    product_urls = _tagged_media_urls(request, PRODUCT_IMAGE_TYPE)
    if not person_urls:
        raise _missing_media_error(PERSON_IMAGE_TYPE)
    if not product_urls:
        raise _missing_media_error(PRODUCT_IMAGE_TYPE)
    return genai_types.RecontextImageSource(
        person_image=_to_image(person_urls[0]),
        product_images=[genai_types.ProductImage(product_image=_to_image(url)) for url in product_urls],
    )


def _to_sdk_config(dumped: dict[str, Any] | None, *, action_name: str) -> genai_types.RecontextImageConfig | None:
    """Map the dumped family config onto the typed SDK request config."""
    if not dumped:
        return None
    dumped = {key: value for key, value in dumped.items() if key not in _CLIENT_OPTION_KEYS}
    if not dumped:
        return None

    options: dict[str, Any] = dumped.pop('output_options', None) or {}
    mime_type = options.pop('mime_type', None)
    compression_quality = options.pop('compression_quality', None)
    if mime_type is not None:
        dumped['output_mime_type'] = mime_type
    if compression_quality is not None:
        dumped['output_compression_quality'] = compression_quality
    if options:
        dumped['outputOptions'] = options

    for key, renamed in (*_SDK_FIELD_NAMES.items(), *_VERBATIM_FIELD_NAMES.items()):
        if key in dumped:
            dumped[renamed] = dumped.pop(key)

    known, leftovers = split_sdk_fields(dumped, genai_types.RecontextImageConfig)
    try:
        cfg = genai_types.RecontextImageConfig(**known)
    except ValidationError as e:
        raise sdk_config_error(action_name=action_name, error=e) from e
    return attach_leftovers(cfg, leftovers, nest='parameters')


def _media_part(image: genai_types.Image | None) -> Part | None:
    """One generated image becomes a viewable media part."""
    if image is None:
        return None
    mime = image.mime_type or 'image/png'
    if image.gcs_uri:
        url = image.gcs_uri
    elif image.image_bytes:
        url = f'data:{mime};base64,{base64.b64encode(image.image_bytes).decode("ascii")}'
    else:
        return None
    return Part(MediaPart(media=Media(url=url, content_type=mime)))


def _from_recontext_response(
    response: genai_types.RecontextImageResponse,
    request: ModelRequest,
) -> ModelResponse:
    """Put every generated image on one model message.

    Samples the API filtered out are named in ``finish_message``.
    """
    parts: list[Part] = []
    filtered: list[str] = []
    for generated in response.generated_images or []:
        part = _media_part(generated.image)
        if part is None:
            # A filtered sample still comes back, carrying a reason and an
            # empty image.
            if generated.rai_filtered_reason:
                filtered.append(str(generated.rai_filtered_reason))
            continue
        parts.append(part)

    # mode='json' so the decoded image bytes the SDK hands back land in raw as
    # base64 text; a bytes value here makes the whole response unserializable.
    raw = response.model_dump(by_alias=True, exclude_none=True, mode='json')
    message = Message(role=Role.MODEL, content=parts)
    if not parts:
        return ModelResponse(
            message=message,
            finish_reason=FinishReason.BLOCKED,
            finish_message='; '.join(filtered) or 'Model returned no predictions. Possibly due to content filters.',
            raw=raw,
        )

    usage = get_basic_usage_stats(input_=request.messages, response=message)
    usage.custom = {'generations': len(parts)}
    return ModelResponse(
        message=message,
        finish_reason=FinishReason.STOP,
        finish_message='; '.join(filtered) or None,
        usage=usage,
        raw=raw,
    )


class VirtualTryOnModel:
    """Virtual Try-On image model runner."""

    def __init__(
        self,
        name: str,
        client: genai.Client,
        client_kwargs: dict[str, Any] | None = None,
        base_url_pinned: bool = False,
    ) -> None:
        """Initialize Virtual Try-On model runner.

        Args:
            name: The full model name.
            client: The GenAI client.
            client_kwargs: The plugin-level kwargs the client was constructed
                from. Used when a call overrides the key or endpoint.
            base_url_pinned: Whether the plugin was constructed with an
                explicit ``base_url``. A location override then keeps it.
        """
        self._name = name
        self._client = client
        self._model_id = name.split('/')[-1]
        self._client_kwargs = client_kwargs
        self._base_url_pinned = base_url_pinned

    def _client_for_context(
        self,
        ctx: ActionRunContext,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> genai.Client:
        """Plugin client, or a request-scoped one when secrets/config are set."""
        context = ctx.context
        api_key = context_api_key(context)
        context_config = context.get('config')
        context_config = context_config if isinstance(context_config, dict) else {}
        request_config = config or {}
        base_url = (
            request_config.get('base_url')
            or request_config.get('baseUrl')
            or context_config.get('base_url')
            or context_config.get('baseUrl')
        )
        api_version = (
            request_config.get('api_version')
            or request_config.get('apiVersion')
            or context_config.get('api_version')
            or context_config.get('apiVersion')
        )
        location = request_config.get('location') or context_config.get('location')
        if api_key is None and not base_url and not api_version and not location:
            return self._client

        kwargs = dict(self._client_kwargs or {})
        # The model is Vertex-only; a request-scoped client stays on that backend.
        kwargs.setdefault('vertexai', True)
        plugin_opts = kwargs.get('http_options')
        opts = plugin_opts.model_copy(deep=True) if plugin_opts is not None else genai_types.HttpOptions()
        if api_key is not None:
            kwargs['api_key'] = api_key
            # The SDK rejects api_key with credentials, project, or location.
            kwargs['credentials'] = None
            kwargs.pop('project', None)
            kwargs.pop('location', None)
            location = None
            # Express keys are not a regional Vertex host. Keep only an
            # explicit base_url from this call.
            if not base_url:
                opts.base_url = None
        if location:
            kwargs['location'] = location
            if not base_url and not self._base_url_pinned:
                opts.base_url = multi_regional_base_url(location) if is_multi_regional_location(location) else None
        if base_url:
            opts.base_url = base_url
        if api_version:
            opts.api_version = api_version
        kwargs['http_options'] = opts
        try:
            return genai.Client(**kwargs)
        except Exception as e:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f'Failed to create google-genai client: {e}',
            ) from e

    async def generate(self, request: ModelRequest[VirtualTryOnConfig], ctx: ActionRunContext) -> ModelResponse:
        """Generate try-on images for the tagged person and product media.

        Every image comes back in the one response; there is nothing to stream.

        Args:
            request: The model request carrying the tagged media parts and config.
            ctx: The action run context.

        Returns:
            ModelResponse with one media part per generated image.
        """
        reject_request_config_api_key(request.config)
        if not self._client.vertexai:
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message=f'{self._name}: Virtual Try-On is only served by the Vertex AI backend.',
            )

        source = _to_source(request)
        dumped = dump_family_config(config=request.config, expected_type=VirtualTryOnConfig, action_name=self._name)
        client = self._client_for_context(ctx, config=dumped)
        config = _to_sdk_config(dumped, action_name=self._name)
        try:
            response = await client.aio.models.recontext_image(
                model=self._model_id,
                source=source,
                config=config,
            )
        except APIError as e:
            raise wrap_http_error(e, status_code=e.code, message=e.message or str(e)) from e

        return _from_recontext_response(response, request)
