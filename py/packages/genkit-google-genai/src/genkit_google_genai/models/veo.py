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

"""Veo video generation via the GenAI generate_videos long-running API."""

from __future__ import annotations

from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, ConfigDict, Field

from genkit import (
    FinishReason,
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
from genkit._core._background import define_background_model
from genkit._core._compat import StrEnum
from genkit._core._registry import Registry
from genkit.model import BackgroundAction, Error, Operation
from genkit.plugin_api import ActionRunContext
from genkit_google_genai.models.interactions_utils import extract_version


class VeoVersion(StrEnum):
    """Commonly used Veo model version identifiers."""

    VEO_2_0 = 'veo-2.0-generate-001'
    VEO_2_0_EXP = 'veo-2.0-generate-exp'
    VEO_3_0 = 'veo-3.0-generate-001'
    VEO_3_0_FAST = 'veo-3.0-fast-generate-001'
    VEO_3_1_PREVIEW = 'veo-3.1-generate-preview'
    VEO_3_1_FAST_PREVIEW = 'veo-3.1-fast-generate-preview'
    VEO_3_1 = 'veo-3.1-generate-001'
    VEO_3_1_FAST = 'veo-3.1-fast-generate-001'


class VeoConfig(BaseModel):
    """Veo video generation configuration."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)
    negative_prompt: str | None = Field(default=None, alias='negativePrompt')
    aspect_ratio: str | None = Field(default=None, alias='aspectRatio')
    person_generation: str | None = Field(default=None, alias='personGeneration')
    duration_seconds: int | None = Field(default=None, alias='durationSeconds')
    resolution: str | None = None
    seed: int | None = None
    enhance_prompt: bool | None = Field(default=None, alias='enhancePrompt')


DEFAULT_VEO_SUPPORT = Supports(
    media=True,
    multiturn=False,
    tools=False,
    system_role=True,
    output=['media'],
    long_running=True,
)


def is_veo_model(name: str) -> bool:
    """Return True when the model name belongs to the Veo family."""
    return name.lower().startswith('veo')


def veo_model_info(version: str) -> ModelInfo:
    """Return capability metadata for a Veo model."""
    clean = extract_version(version)
    return ModelInfo(
        label=f'Google AI - {clean}',
        supports=DEFAULT_VEO_SUPPORT,
    )


def to_veo_parameters(config: VeoConfig) -> genai_types.GenerateVideosConfig:
    """Convert VeoConfig into the SDK GenerateVideosConfig."""
    return genai_types.GenerateVideosConfig.model_validate(config.model_dump(exclude_none=True))


def extract_text_prompt(request: ModelRequest[VeoConfig]) -> str:
    """Join text parts from the request into a single prompt string."""
    parts = [
        part.root.text
        for message in request.messages or []
        for part in message.content
        if isinstance(part.root, TextPart) and part.root.text
    ]
    return ' '.join(parts)


def video_parts_from_uris(uris: list[str]) -> list[Part]:
    """Build model message parts for generated video URIs."""
    return [Part(root=MediaPart(media=Media(url=uri, content_type='video/mp4'))) for uri in uris]


def extract_video_uris(response: genai_types.GenerateVideosResponse) -> list[str]:
    """Extract video URIs from a GenerateVideosResponse."""
    uris: list[str] = []
    for item in response.generated_videos or []:
        if item.video and item.video.uri:
            uris.append(item.video.uri)
    return uris


def model_response_from_veo(
    response: genai_types.GenerateVideosResponse,
) -> ModelResponse[genai_types.GenerateVideosResponse]:
    """Build a ModelResponse from a completed GenerateVideosResponse."""
    return ModelResponse[genai_types.GenerateVideosResponse](
        finish_reason=FinishReason.STOP,
        message=Message(
            role=Role.MODEL,
            content=video_parts_from_uris(extract_video_uris(response)),
        ),
        raw=response.model_dump(exclude_none=True),
    )


def from_veo_operation(operation: genai_types.GenerateVideosOperation) -> Operation:
    """Convert a GenerateVideosOperation into a Genkit Operation."""
    # LRO can omit or null `done` while still running — treat that as pending.
    op = Operation(
        id=operation.name or '',
        done=bool(operation.done),
    )
    if operation.error:
        op.error = Error(message=str(operation.error.get('message', 'Unknown error')))
        return op

    response = operation.response or operation.result
    if response is not None and extract_video_uris(response):
        output: ModelResponse[genai_types.GenerateVideosResponse] = model_response_from_veo(response)
        op.output = output
    return op


def create_veo_background_action(name: str, client: genai.Client) -> BackgroundAction:
    """Build a Veo background model: start returns an Operation; check refreshes it once."""
    version = extract_version(name)
    info = veo_model_info(version)

    async def start(request: ModelRequest[VeoConfig], _: ActionRunContext) -> Operation:
        if request.tools:
            raise ValueError('Tools are not supported for this model.')
        prompt = extract_text_prompt(request)
        if not prompt:
            raise ValueError('Veo requires a text prompt')
        config = request.config or VeoConfig()
        sdk_op = await client.aio.models.generate_videos(
            model=version,
            prompt=prompt,
            config=to_veo_parameters(config),
        )
        return from_veo_operation(sdk_op)

    async def check(operation: Operation) -> Operation:
        sdk_op = await client.aio.operations.get(
            operation=genai_types.GenerateVideosOperation.model_validate({'name': operation.id}),
        )
        return from_veo_operation(sdk_op)

    return define_background_model(
        registry=Registry(),
        name=name,
        start=start,
        check=check,
        cancel=None,
        label=info.label or name,
        info=info,
        config_schema=VeoConfig,
    )
