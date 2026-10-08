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

"""Veo video generation model for Google GenAI plugin.

Veo is Google's video generation model that creates videos from text prompts.
"""

import base64
from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

from google import genai
from google.genai import types as genai_types
from google.genai.errors import APIError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic.alias_generators import to_camel

from genkit import ActionRunContext, FinishReason, GenkitError, Message, ModelResponse, Operation, Part, Role
from genkit.model import ModelInfo, ModelRequest, OperationError, Supports
from genkit.plugin_api import wrap_http_error
from genkit_google_genai._auth import GOOGLE_AUTH_ERRORS, raise_auth_error
from genkit_google_genai._constants import is_multi_regional_location, multi_regional_base_url
from genkit_google_genai._models._sdk_config import (
    VEO_MANAGED_BODY_FIELDS,
    attach_config_extra,
    dump_family_config,
    keep_client_extra_body,
    sdk_config_error,
)
from genkit_google_genai._models._secrets import context_api_key, misplaced_key_error

# Quote autocomplete needs a Literal, so this alias is the Veo catalog.
# ``veo_model`` takes ``KnownVeo | str`` so unlisted ids still work.
KnownVeo: TypeAlias = Literal[
    'veo-2.0-generate-001',
    'veo-2.0-generate-exp',
    'veo-3.0-generate-001',
    'veo-3.0-fast-generate-001',
    'veo-3.1-generate-preview',
    'veo-3.1-fast-generate-preview',
    'veo-3.1-generate-001',
    'veo-3.1-fast-generate-001',
]

# Sent as genai_types.VideoCompressionQuality and genai_types.ImageResizeMode.
# googlegenai_gemini_test.py pins both to the SDK enums.
VideoCompressionQuality: TypeAlias = Literal['OPTIMIZED', 'LOSSLESS']
ImageResizeMode: TypeAlias = Literal['CROP', 'PAD']


def is_veo_model(name: str) -> bool:
    """Check if a model name is a Veo model.

    Args:
        name: The model name to check.

    Returns:
        True if this is a Veo model name.
    """
    return name.split('/')[-1].lower().startswith('veo-')


class VeoConfig(BaseModel):
    """Veo Config Schema."""

    model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True, alias_generator=to_camel)
    number_of_videos: int | None = Field(default=None)
    generate_audio: bool | None = Field(default=None)
    fps: int | None = Field(default=None)
    output_gcs_uri: str | None = Field(default=None)
    pubsub_topic: str | None = Field(default=None)
    compression_quality: VideoCompressionQuality | None = Field(default=None)
    resize_mode: ImageResizeMode | None = Field(default=None)
    labels: dict[str, str] | None = Field(default=None)
    last_frame: dict[str, Any] | None = Field(default=None)
    reference_images: list[dict[str, Any]] | None = Field(default=None)
    mask: dict[str, Any] | None = Field(default=None)
    webhook_config: dict[str, Any] | None = Field(default=None)
    extra: dict[str, Any] | None = Field(
        default=None,
        description=(
            'Provider fields this class does not declare, in API wire names, merged into the top level of the '
            "request body after everything else (for example {'parameters': {...}}). Nested objects merge key "
            'by key. Not checked; do not put API keys here.'
        ),
    )
    negative_prompt: str | None = Field(default=None, description='Negative prompt for video generation.')
    aspect_ratio: str | None = Field(
        default=None, description='Desired aspect ratio of the output video (e.g. "16:9").'
    )
    person_generation: str | None = Field(default=None, description='Person generation mode.')
    duration_seconds: int | None = Field(default=None, description='Length of video in seconds.')
    resolution: str | None = Field(default=None, description='Desired output resolution (e.g. "720p").')
    seed: int | None = Field(default=None, description='Random seed for deterministic generation.')
    enhance_prompt: bool | None = Field(default=None, description='Enable prompt enhancement.')
    base_url: str | None = Field(default=None, description='Override the API endpoint for this call.')
    api_version: str | None = Field(default=None, description='Override the API version for this call.')
    location: str | None = Field(default=None, description='Override the Vertex AI location for this call.')

    @model_validator(mode='before')
    @classmethod
    def _api_key_belongs_in_secrets(cls, data: Any) -> Any:  # noqa: ANN401
        """Point a key in config or extra at context.secrets, not the generic unknown-key error."""
        if isinstance(data, Mapping):
            extra = data.get('extra')
            for bag in (data, extra if isinstance(extra, Mapping) else {}):
                if bag.get('api_key') is not None or bag.get('apiKey') is not None:
                    raise misplaced_key_error()
        return data


DEFAULT_VEO_SUPPORT = Supports(
    media=True,
    multiturn=False,
    tools=False,
    system_role=True,
    output=['media'],
    long_running=True,
)

_CLIENT_OPTION_KEYS = frozenset({'base_url', 'baseUrl', 'api_version', 'apiVersion', 'location'})


def veo_model_info(version: str) -> ModelInfo:
    """Get model info for a Veo model.

    Args:
        version: The Veo model version.

    Returns:
        ModelInfo for the Veo model.
    """
    return ModelInfo(
        label=f'Google AI - {version}',
        supports=DEFAULT_VEO_SUPPORT,
    )


def _extract_text(request: ModelRequest) -> str:
    """Extract text prompt from request messages.

    Args:
        request: The model request containing messages.

    Returns:
        The combined text prompt.
    """
    prompt_parts = [
        str(part.text)
        for message in request.messages or []
        for part in message.content
        if part.text is not None and part.text
    ]
    return ' '.join(prompt_parts)


def _sniff_video_mime(uri: str | None) -> str:
    if uri:
        lower = uri.lower()
        if lower.endswith('.webm'):
            return 'video/webm'
        if lower.endswith('.mov'):
            return 'video/quicktime'
    return 'video/mp4'


def _media_part(*, video: genai_types.Video | None) -> Part | None:
    """One SDK video becomes a playable media part.

    Studio Veo finishes with a download URL. Vertex often finishes with
    inline ``video_bytes`` (or a GCS path in ``uri``) and no HTTP URL.
    Either way the caller should see ``media.url``.
    """
    if video is None:
        return None
    mime = video.mime_type or _sniff_video_mime(video.uri)
    if video.uri:
        url = video.uri
    elif video.video_bytes:
        b64 = base64.b64encode(video.video_bytes).decode('ascii')
        url = f'data:{mime};base64,{b64}'
    else:
        return None
    return Part.from_media(url, content_type=mime)


def _operation_error_message(*, error: Any) -> str:  # noqa: ANN401
    if isinstance(error, dict):
        message = error.get('message')
    else:
        message = getattr(error, 'message', None) or str(error)
    return str(message) if message else 'Unknown error'


def _from_veo_operation(*, api_op: genai_types.GenerateVideosOperation) -> Operation:
    """Turn a GenerateVideosOperation into the Genkit ticket.

    ``output`` is a ModelResponse so pollers read
    ``operation.output.media[0].url`` the same way they read a still off
    ``generate()``.
    """
    op = Operation(id=api_op.name or '', done=bool(api_op.done))
    if api_op.error:
        op.error = OperationError(message=_operation_error_message(error=api_op.error))
        return op

    response = api_op.response
    if response is None:
        return op

    content: list[Part] = []
    for generated in response.generated_videos or []:
        part = _media_part(video=generated.video)
        if part is not None:
            content.append(part)

    raw_payload: dict[str, Any] | None = None
    if hasattr(response, 'model_dump'):
        raw_payload = response.model_dump(by_alias=True, exclude_none=True)
    elif isinstance(response, dict):
        raw_payload = response

    if content:
        op.output = ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=content),
            raw=raw_payload,
        )
        return op

    # Vertex can mark the job done and still return no videos when RAI
    # dropped every sample. Name that, don't hand back an empty success.
    if api_op.done and response.rai_media_filtered_count:
        reasons = [str(reason) for reason in (response.rai_media_filtered_reasons or []) if reason]
        op.error = OperationError(
            message='; '.join(reasons) or 'All generated videos were filtered out by safety filters.'
        )
        return op

    if api_op.done and not content and not op.error:
        op.error = OperationError(message='Operation completed but returned no playable media.')
    return op


class VeoModel:
    """Veo video generation model runner."""

    def __init__(
        self,
        name: str,
        client: genai.Client,
        client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize Veo model runner.

        Args:
            name: The full model name.
            client: The GenAI client.
            client_kwargs: The plugin-level kwargs the client was constructed
                from. Used when a call overrides the key or endpoint.
        """
        self._name = name
        self._client = client
        self._model_id = name.split('/')[-1]
        self._client_kwargs = client_kwargs

    def _client_for_context(
        self,
        ctx: ActionRunContext,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> genai.Client:
        """Plugin client, or a request-scoped one when secrets/config are set.

        The ticket is just an id. A per-request key or endpoint has to be
        handed in again on start and on every poll.
        """
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
        is_vertex = bool(getattr(self._client, 'vertexai', False) or (self._client_kwargs or {}).get('vertexai'))
        if location and not is_vertex:
            # Location is a Vertex concept; ignore it on the Gemini API backend.
            location = None
        if api_key is None and not base_url and not api_version and not location:
            return self._client

        kwargs = dict(self._client_kwargs or {})
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
            # explicit config.base_url from this call.
            if not base_url:
                opts.base_url = None
        if location:
            kwargs['location'] = location
            if not base_url:
                opts.base_url = multi_regional_base_url(location) if is_multi_regional_location(location) else None
        if base_url:
            opts.base_url = base_url
        if api_version:
            opts.api_version = api_version
        kwargs['http_options'] = opts
        try:
            return genai.Client(**kwargs)
        except GOOGLE_AUTH_ERRORS as e:
            raise_auth_error(e)
        except (ValueError, TypeError) as e:
            # The SDK rejects bad override combinations (api_key with project, say).
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message='Failed to create google-genai client',
                cause=e,
            ) from e

    async def start(self, request: ModelRequest[VeoConfig], ctx: ActionRunContext) -> Operation:
        """Start a video generation operation.

        Args:
            request: The model request containing prompt and config.
            ctx: The action run context.

        Returns:
            Operation representing the started video generation job.
        """
        prompt = _extract_text(request)
        config = self._get_config(request)

        dumped = dump_family_config(
            config=request.config,
            expected_type=VeoConfig,
            action_name=self._name,
        )

        try:
            response: genai_types.GenerateVideosOperation = await self._client_for_context(
                ctx, config=dumped
            ).aio.models.generate_videos(
                model=self._model_id,
                prompt=prompt,
                config=config,
            )
        except APIError as e:
            raise wrap_http_error(e, status_code=e.code, message=e.message or str(e)) from e
        except GOOGLE_AUTH_ERRORS as e:
            raise_auth_error(e)

        return _from_veo_operation(api_op=response)

    async def check(self, operation: Operation, ctx: ActionRunContext) -> Operation:
        """Check the status of a video generation operation.

        Args:
            operation: The operation to check.
            ctx: Run context. Pass secrets again when start used a
                per-request key.

        Returns:
            Updated Operation with current status.
        """
        # operations.get polls by the SDK object's .name, not a
        # constructor arg — that's the same ticket start() returned.
        op_request = genai_types.GenerateVideosOperation()
        op_request.name = operation.id
        try:
            response: genai_types.GenerateVideosOperation = await self._client_for_context(ctx).aio.operations.get(
                operation=op_request
            )
        except APIError as e:
            raise wrap_http_error(e, status_code=e.code, message=e.message or str(e)) from e
        except GOOGLE_AUTH_ERRORS as e:
            raise_auth_error(e)

        return _from_veo_operation(api_op=response)

    def _get_config(self, request: ModelRequest) -> genai_types.GenerateVideosConfig | None:
        dumped = dump_family_config(
            config=request.config,
            expected_type=VeoConfig,
            action_name=self._name,
        )
        if not dumped:
            return None
        for key in _CLIENT_OPTION_KEYS:
            dumped.pop(key, None)
        if not dumped:
            return None

        # Every other declared VeoConfig field is a GenerateVideosConfig field
        # (veo_test.py pins this), so the rest goes to the SDK type as-is.
        extra = dumped.pop('extra', None)
        try:
            cfg = genai_types.GenerateVideosConfig(**dumped)
        except ValidationError as e:
            raise sdk_config_error(action_name=self._name, error=e) from e

        cfg = attach_config_extra(cfg, extra, action_name=self._name, managed_body_fields=VEO_MANAGED_BODY_FIELDS)
        return keep_client_extra_body(cfg, (self._client_kwargs or {}).get('http_options'))

    @property
    def metadata(self) -> dict:
        """Model metadata."""
        return {'model': {'supports': DEFAULT_VEO_SUPPORT.model_dump(by_alias=True)}}
