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

"""Virtual Try-On image editing model for Google Vertex AI plugin.

Virtual Try-On takes a photo of a person and one or more photos of products
(garments) and returns an image of the person wearing the product. It is
available through Vertex AI only.

Callers identify which image is which by setting a ``type`` key in the
``ai.Part`` metadata — ``personImage`` for the model's photo and
``productImage`` for each garment. The string values match the Go and JS
conventions so the same request shape works across runtimes.
"""

import base64
import json
import sys
import urllib.parse
from typing import Any

if sys.version_info < (3, 11):
    from strenum import StrEnum
else:
    from enum import StrEnum

from google import genai
from google.genai.errors import ClientError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from genkit import (
    FinishReason,
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
from genkit.plugin_api import ActionRunContext, tracer
from genkit.plugins.google_genai.models.utils import client_error_to_genkit_status

_VIRTUAL_TRY_ON_UNAVAILABLE_ERRORS = (ConnectionError, TimeoutError, OSError)


def _virtual_try_on_request_error_status(error: Exception) -> str:
    """Map non-ClientError request failures to a Genkit status name."""
    if isinstance(error, _VIRTUAL_TRY_ON_UNAVAILABLE_ERRORS):
        return 'UNAVAILABLE'
    return 'INTERNAL'


class VirtualTryOnVersion(StrEnum):
    """Supported virtual try-on image editing models."""

    VIRTUAL_TRY_ON_001 = 'virtual-try-on-001'


# Virtual Try-On models available on the Vertex AI backend. Hardcoded because
# client.models.list() does not always surface them, and they must be visible
# in the plugin's Init / list_actions output to be selectable.
VERTEXAI_KNOWN_VIRTUAL_TRY_ON_MODELS: tuple[str, ...] = (VirtualTryOnVersion.VIRTUAL_TRY_ON_001,)


# Metadata keys used to tag input images. Users attach
# ``metadata={'type': PART_METADATA_TYPE_PERSON_IMAGE}`` to the person photo
# and ``{'type': PART_METADATA_TYPE_PRODUCT_IMAGE}`` to each garment.
PART_METADATA_TYPE_PERSON_IMAGE = 'personImage'
PART_METADATA_TYPE_PRODUCT_IMAGE = 'productImage'


def is_virtual_try_on_model(name: str) -> bool:
    """Check whether a model name denotes a virtual try-on model."""
    return name.startswith('virtual-try-on-')


class VirtualTryOnOutputOptions(BaseModel):
    """Output image format controls."""

    mime_type: str | None = Field(default=None, alias='mimeType')
    compression_quality: int | None = Field(default=None, alias='compressionQuality')

    model_config = ConfigDict(populate_by_name=True, extra='allow')


def _inline_virtual_try_on_config_schema(schema: dict[str, Any], _: type[Any]) -> None:
    """Inline nested outputOptions so the model config UI can render it."""
    properties = schema.get('properties')
    if not isinstance(properties, dict):
        return

    properties['outputOptions'] = {
        'additionalProperties': True,
        'default': None,
        'description': 'Output image format controls.',
        'properties': {
            'mimeType': {
                'anyOf': [{'type': 'string'}, {'type': 'null'}],
                'default': None,
                'title': 'Mime type',
            },
            'compressionQuality': {
                'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                'default': None,
                'title': 'Compression quality',
            },
        },
        'title': 'Output options',
        'type': 'object',
    }


class VirtualTryOnConfig(BaseModel):
    """Configuration for a virtual-try-on request.

    Mirrors the JS ``ImagenTryOnConfigSchema`` in
    ``js/plugins/google-genai/src/vertexai/imagen.ts``.
    """

    sample_count: int | None = Field(default=None, ge=1, alias='sampleCount')
    storage_uri: str | None = Field(default=None, alias='storageUri')
    seed: int | None = Field(default=None)
    base_steps: int | None = Field(default=None, ge=1, le=100, alias='baseSteps')
    safety_setting: str | None = Field(default=None, alias='safetySetting')
    person_generation: str | None = Field(default=None, alias='personGeneration')
    add_watermark: bool | None = Field(default=None, alias='addWatermark')
    enhance_prompt: bool | None = Field(default=None, alias='enhancePrompt')
    output_options: VirtualTryOnOutputOptions | None = Field(default=None, alias='outputOptions')

    model_config = ConfigDict(
        populate_by_name=True,
        extra='allow',
        json_schema_extra=_inline_virtual_try_on_config_schema,
    )


VIRTUAL_TRY_ON_SUPPORTS = Supports(
    media=True,
    multiturn=False,
    tools=False,
    system_role=True,
    output=['media'],
)


VIRTUAL_TRY_ON_MODEL_INFO = ModelInfo(
    label='Vertex AI - Virtual Try-On',
    supports=VIRTUAL_TRY_ON_SUPPORTS,
)


def virtual_try_on_model_info(version: str) -> ModelInfo:
    """ModelInfo for a virtual try-on model."""
    if version == VirtualTryOnVersion.VIRTUAL_TRY_ON_001:
        return ModelInfo(label='Vertex AI - Virtual Try-On 001', supports=VIRTUAL_TRY_ON_SUPPORTS)
    return ModelInfo(label=f'Vertex AI - {version}', supports=VIRTUAL_TRY_ON_SUPPORTS)


def _extract_media_by_type(request: ModelRequest, part_type: str) -> list[dict[str, Any]]:
    """Collect input images tagged with the given metadata type.

    Handles both data URIs (extracting base64 payload) and ``gs://`` URIs
    (passed through as ``gcsUri``).
    """
    out: list[dict[str, Any]] = []
    for message in request.messages:
        for part in message.content:
            root = part.root
            # Only media parts can carry image input
            if not isinstance(root, MediaPart):
                continue
            metadata = getattr(root, 'metadata', None) or {}
            if metadata.get('type') != part_type:
                continue
            url = root.media.url or ''
            if url.startswith('gs://'):
                out.append({'image': {'gcsUri': url}})
                continue
            # data:<mime>[;base64],<payload>
            if url.startswith('data:'):
                prefix, _, payload = url.partition(',')
                if not payload:
                    continue
                if ';base64' in prefix:
                    out.append({'image': {'bytesBase64Encoded': payload}})
                else:
                    # Non-base64 data URIs are rare; decode percent-encoding
                    # and re-encode as base64 for the Vertex API.
                    decoded = urllib.parse.unquote_to_bytes(payload)
                    out.append({'image': {'bytesBase64Encoded': base64.b64encode(decoded).decode('ascii')}})
    return out


def _to_virtual_try_on_request(request: ModelRequest, config: VirtualTryOnConfig | dict | None) -> dict[str, Any]:
    """Build the Vertex predict request body for virtual try-on."""
    persons = _extract_media_by_type(request, PART_METADATA_TYPE_PERSON_IMAGE)
    products = _extract_media_by_type(request, PART_METADATA_TYPE_PRODUCT_IMAGE)
    if not persons:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f"virtual try-on requires a media part with metadata.type='{PART_METADATA_TYPE_PERSON_IMAGE}'",
        )
    if not products:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=(
                'virtual try-on requires at least one media part with '
                f"metadata.type='{PART_METADATA_TYPE_PRODUCT_IMAGE}'"
            ),
        )

    instance: dict[str, Any] = {
        'personImage': persons[0],
        'productImages': products,
    }

    parameters: dict[str, Any] = {}
    if isinstance(config, VirtualTryOnConfig):
        # Drop legacy location overrides if callers still pass them as extra
        # config; location is not a Vertex predict request parameter.
        parameters = config.model_dump(by_alias=True, exclude_none=True, exclude={'location'})
    elif isinstance(config, dict):
        parameters = {k: v for k, v in config.items() if k not in ('location', 'Location') and v is not None}

    return {
        'instances': [instance],
        'parameters': parameters,
    }


def _parse_virtual_try_on_response_body(body: object) -> dict[str, Any]:
    """Decode and validate the top-level Vertex predict response."""
    try:
        response = json.loads(body) if isinstance(body, (bytes, bytearray, str)) else body
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise GenkitError(
            status='INTERNAL',
            message=f'virtual try-on returned a malformed response body: {str(e)}',
            cause=e,
        ) from e

    if not isinstance(response, dict):
        raise GenkitError(
            status='INTERNAL',
            message=f'virtual try-on returned an unexpected response body: {type(response).__name__}',
        )

    predictions = response.get('predictions')
    if predictions is not None and not isinstance(predictions, list):
        raise GenkitError(
            status='INTERNAL',
            message='virtual try-on returned an unexpected predictions value',
        )

    return response


def _from_virtual_try_on_response(response: dict[str, Any]) -> list[Part]:
    """Convert a virtual-try-on predict response to MediaParts."""
    content: list[Part] = []
    for prediction in response.get('predictions') or []:
        if not isinstance(prediction, dict):
            raise GenkitError(
                status='INTERNAL',
                message='virtual try-on returned an unexpected prediction value',
            )
        b64 = prediction.get('bytesBase64Encoded')
        if not isinstance(b64, str) or not b64:
            raise GenkitError(
                status='INTERNAL',
                message='virtual try-on returned a prediction without image data',
            )
        mime_type = prediction.get('mimeType')
        if mime_type is not None and not isinstance(mime_type, str):
            raise GenkitError(
                status='INTERNAL',
                message='virtual try-on returned an unexpected mimeType value',
            )
        if not mime_type:
            mime_type = 'image/png'
        content.append(
            Part(
                root=MediaPart(
                    media=Media(
                        url=f'data:{mime_type};base64,{b64}',
                        content_type=mime_type,
                    )
                )
            )
        )
    return content


class VirtualTryOnModel:
    """Virtual-try-on model driver."""

    def __init__(self, version: str | VirtualTryOnVersion, client: genai.Client) -> None:
        """Initialize the virtual try-on model.

        Args:
            version: The virtual try-on model version.
            client: The configured Vertex AI genai.Client.
        """
        self._version = version
        self._client = client

    def _get_config(self, request: ModelRequest) -> VirtualTryOnConfig | dict | None:
        if request.config is None:
            return None
        if isinstance(request.config, VirtualTryOnConfig):
            return request.config
        try:
            if isinstance(request.config, dict):
                return VirtualTryOnConfig.model_validate(request.config)
            # Fall back to generic dict coercion for BaseModel instances
            if isinstance(request.config, BaseModel):
                return VirtualTryOnConfig.model_validate(request.config.model_dump())
        except ValidationError as e:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message='The virtual try-on configuration is invalid',
                cause=e,
            ) from e
        return None

    async def generate(self, request: ModelRequest, _: ActionRunContext) -> ModelResponse:
        """Handle a virtual-try-on generation request."""
        api_client = getattr(self._client, '_api_client', None)
        if api_client is None or not getattr(api_client, 'vertexai', False):
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message='Virtual Try-On is only available through the Vertex AI backend',
            )

        config = self._get_config(request)
        payload = _to_virtual_try_on_request(request, config)
        path = f'publishers/google/models/{self._version}:predict'

        with tracer.start_as_current_span('virtual_try_on_predict') as span:
            span.set_attribute(
                'genkit:input',
                json.dumps({'model': self._version, 'body': payload}, default=str),
            )
            try:
                http_response = await api_client.async_request('POST', path, payload)
            except ClientError as e:
                raise GenkitError(
                    status=client_error_to_genkit_status(e),
                    message=e.message or 'Unknown error',
                    cause=e,
                ) from e
            except GenkitError:
                raise
            except Exception as e:
                raise GenkitError(
                    status=_virtual_try_on_request_error_status(e),
                    message=f'virtual try-on request failed: {type(e).__name__}: {str(e)}',
                    cause=e,
                ) from e
            body = http_response.body if hasattr(http_response, 'body') else http_response
            response = _parse_virtual_try_on_response_body(body)
            span.set_attribute('genkit:output', json.dumps(response, default=str))

        if 'predictions' not in response:
            raise GenkitError(
                status='INTERNAL',
                message='virtual try-on returned a response without predictions',
            )

        if not response['predictions']:
            # Vertex returning zero predictions almost always means safety
            # filters blocked the output — surface it as a blocked response.
            return ModelResponse(
                message=Message(role=Role.MODEL, content=[]),
                finish_reason=FinishReason.BLOCKED,
                finish_message='virtual try-on: no predictions returned (likely content-filtered)',
            )

        return ModelResponse(
            message=Message(role=Role.MODEL, content=_from_virtual_try_on_response(response)),
        )
