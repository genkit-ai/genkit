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
import sys
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
from genkit.model import Error, Operation
from genkit.plugin_api import ActionRunContext, wrap_http_error
from genkit_google_genai.models._sdk_config import (
    attach_leftovers,
    dump_family_config,
    sdk_config_error,
    split_sdk_fields,
)


class VeoVersion(StrEnum):
    """Supported Veo video generation models.

    Note: Models are discovered dynamically. This enum provides convenience
    constants for commonly used Veo models.
    """

    VEO_2_0 = 'veo-2.0-generate-001'
    VEO_2_0_EXP = 'veo-2.0-generate-exp'
    VEO_3_0 = 'veo-3.0-generate-001'
    VEO_3_0_FAST = 'veo-3.0-fast-generate-001'
    VEO_3_1_PREVIEW = 'veo-3.1-generate-preview'
    VEO_3_1_FAST_PREVIEW = 'veo-3.1-fast-generate-preview'
    VEO_3_1 = 'veo-3.1-generate-001'
    VEO_3_1_FAST = 'veo-3.1-fast-generate-001'


# Quote autocomplete needs a Literal. The enum above is the catalog; a test
# requires these members and the enum values to be the same set.
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


def is_veo_model(name: str) -> bool:
    """Check if a model name is a Veo model.

    Args:
        name: The model name to check.

    Returns:
        True if this is a Veo model name.
    """
    return name.split('/')[-1].lower().startswith('veo-')


class VeoConfigSchema(BaseModel):
    """Veo Config Schema."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)
    negative_prompt: str | None = Field(
        default=None, alias='negativePrompt', description='Negative prompt for video generation.'
    )
    aspect_ratio: str | None = Field(
        default=None, alias='aspectRatio', description='Desired aspect ratio of the output video (e.g. "16:9").'
    )
    person_generation: str | None = Field(default=None, alias='personGeneration', description='Person generation mode.')
    duration_seconds: int | None = Field(
        default=None, alias='durationSeconds', description='Length of video in seconds.'
    )
    resolution: str | None = Field(default=None, description='Desired output resolution (e.g. "720p").')
    seed: int | None = Field(default=None, description='Random seed for deterministic generation.')
    enhance_prompt: bool | None = Field(default=None, alias='enhancePrompt', description='Enable prompt enhancement.')


# Alias for backwards compatibility with __init__.py exports
VeoConfig = VeoConfigSchema


DEFAULT_VEO_SUPPORT = Supports(
    media=True,
    multiturn=False,
    tools=False,
    system_role=True,
    output=['media'],
    long_running=True,
)


def veo_model_info(version: str) -> ModelInfo:
    """Get model info for a Veo model.

    Args:
        version: The Veo model version.

    Returns:
        ModelInfo describing the model's capabilities.
    """
    return ModelInfo(
        label=f'Google AI - {version}',
        supports=DEFAULT_VEO_SUPPORT,
    )


def _extract_text(request: ModelRequest) -> str:
    """Extract text prompt from a ModelRequest.

    Args:
        request: The generation request.

    Returns:
        The text prompt string.
    """
    prompt_parts = [
        str(part.root.text)
        for message in request.messages or []
        for part in message.content
        if hasattr(part.root, 'text') and part.root.text
    ]
    return ' '.join(prompt_parts)


def _media_part(*, video: genai_types.Video | None) -> Part | None:
    """One SDK video becomes a playable media part.

    Studio Veo finishes with a download URL. Vertex often finishes with
    inline ``video_bytes`` (or a GCS path in ``uri``) and no HTTP URL.
    Either way the caller should see ``media.url``.
    """
    if video is None:
        return None
    mime = video.mime_type or 'video/mp4'
    if video.uri:
        url = video.uri
    elif video.video_bytes:
        b64 = base64.b64encode(video.video_bytes).decode('ascii')
        url = f'data:{mime};base64,{b64}'
    else:
        return None
    return Part(MediaPart(media=Media(url=url, content_type=mime)))


def _operation_error_message(*, error: dict[str, Any]) -> str:
    message = error.get('message')
    return str(message) if message else 'Unknown error'


def _from_veo_operation(*, api_op: genai_types.GenerateVideosOperation) -> Operation:
    """Turn a GenerateVideosOperation into the Genkit ticket.

    ``output`` is a ModelResponse so pollers read
    ``operation.output.media[0].url`` the same way they read a still off
    ``generate()``.
    """
    op = Operation(id=api_op.name or '', done=bool(api_op.done))
    if api_op.error:
        op.error = Error(message=_operation_error_message(error=api_op.error))
        return op

    response = api_op.response
    if response is None:
        return op

    content: list[Part] = []
    for generated in response.generated_videos or []:
        part = _media_part(video=generated.video)
        if part is not None:
            content.append(part)

    if content:
        op.output = ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=content),
        )
        return op

    # Vertex can mark the job done and still return no videos when RAI
    # dropped every sample. Name that, don't hand back an empty success.
    if api_op.done and response.rai_media_filtered_count:
        reasons = [str(reason) for reason in (response.rai_media_filtered_reasons or []) if reason]
        op.error = Error(message='; '.join(reasons) or 'All generated videos were filtered out by safety filters.')
    return op


class VeoModel:
    """Veo video generation model.

    Veo runs as a background operation: callers start a generation and
    poll it. There is no blocking generate path.
    """

    def __init__(self, version: str, client: genai.Client) -> None:
        """Initialize Veo model.

        Args:
            version: The Veo model version.
            client: The Google GenAI client.
        """
        self._version = version
        self._client = client

    async def start(self, request: ModelRequest[VeoConfigSchema], ctx: ActionRunContext) -> Operation:
        """Start a video generation operation (background model pattern for GoogleAI).

        Args:
            request: The generation request.
            ctx: The action run context.

        Returns:
            An Operation with the job ID.
        """
        if request.tools:
            raise GenkitError(status='UNIMPLEMENTED', message='Tools are not supported for this model.')

        prompt = _extract_text(request)
        if not prompt:
            raise GenkitError(status='INVALID_ARGUMENT', message='Veo requires a text prompt')

        try:
            response = await self._client.aio.models.generate_videos(
                model=self._version,
                prompt=prompt,
                config=self._get_config(request),
            )
        except APIError as e:
            raise wrap_http_error(e, status_code=e.code, message=e.message or str(e)) from e

        return _from_veo_operation(api_op=response)

    async def check(self, operation: Operation) -> Operation:
        """Check the status of a video generation operation.

        Args:
            operation: The operation to check.

        Returns:
            Updated Operation with current status.
        """
        # operations.get polls by the SDK object's .name, not a
        # constructor arg — that's the same ticket start() returned.
        op_request = genai_types.GenerateVideosOperation()
        op_request.name = operation.id
        try:
            response: genai_types.GenerateVideosOperation = await self._client.aio.operations.get(operation=op_request)
        except APIError as e:
            raise wrap_http_error(e, status_code=e.code, message=e.message or str(e)) from e

        return _from_veo_operation(api_op=response)

    def _get_config(self, request: ModelRequest) -> genai_types.GenerateVideosConfig | None:
        dumped = dump_family_config(
            config=request.config,
            expected_type=VeoConfigSchema,
            action_name=self._version,
        )
        if not dumped:
            return None

        known, leftovers = split_sdk_fields(dumped, genai_types.GenerateVideosConfig)
        try:
            cfg = genai_types.GenerateVideosConfig(**known) if known else genai_types.GenerateVideosConfig()
        except ValidationError as e:
            raise sdk_config_error(action_name=self._version, error=e) from e
        return attach_leftovers(cfg, leftovers, nest='parameters')

    @property
    def metadata(self) -> dict:
        """Model metadata."""
        return {'model': {'supports': DEFAULT_VEO_SUPPORT.model_dump(by_alias=True)}}
