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
It uses the background model pattern because video generation is a long-running
operation that can take 30 seconds to several minutes.

Architecture:
    ```
    ┌──────────────────────────────────────────────────────────────────────┐
    │                      Veo Video Generation Flow                        │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                       │
    │   1. START                 2. POLL                  3. COMPLETE       │
    │   ┌─────────┐             ┌─────────┐             ┌─────────┐        │
    │   │ Prompt  │ ─predict──► │Operation│ ─getOp()──► │  Video  │        │
    │   │  +cfg   │  LongRun    │  (name) │    ...      │  (URI)  │        │
    │   └─────────┘             └────┬────┘             └─────────┘        │
    │                                                                       │
    └──────────────────────────────────────────────────────────────────────┘
    ```

Note:
    Veo models are discovered dynamically via the Google GenAI SDK's models.list() API.
    Any model with 'generateVideos' in supported_actions or 'veo' in the name is treated
    as a Veo model.

Example:
    >>> from genkit import Genkit
    >>> from genkit.plugins.google_genai import GoogleAI
    >>>
    >>> ai = Genkit(plugins=[GoogleAI()])
    >>>
    >>> # Start video generation
    >>> response = await ai.generate(
    ...     model='googleai/veo-2.0-generate-001',
    ...     prompt='A cat playing piano in a jazz club',
    ... )
    >>>
    >>> # Poll until complete
    >>> operation = response.operation
    >>> while not operation.done:
    ...     await asyncio.sleep(5)
    ...     operation = await ai.check_operation(operation)
    >>>
    >>> # Get the video URL
    >>> print(operation.output)

See Also:
    - https://ai.google.dev/gemini-api/docs/video
    - JS implementation: js/plugins/google-genai/src/googleai/veo.ts
"""

import asyncio
import sys
from typing import Any, cast

if sys.version_info < (3, 11):
    from strenum import StrEnum
else:
    from enum import StrEnum

from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, ConfigDict, Field

from genkit.core.action import ActionRunContext
from genkit.core.tracing import tracer
from genkit.core.typing import (
    Error,
    GenerateRequest,
    GenerateResponse,
    ModelInfo,
    Operation,
    Supports,
)
from genkit.types import (
    Media,
    MediaPart,
    Message,
    Part,
    Role,
    TextPart,
)


class VeoVersion(StrEnum):
    """Supported Veo video generation models.

    Note: Models are discovered dynamically. This enum provides convenience
    constants for commonly used Veo models.
    """

    VEO_2_0 = 'veo-2.0-generate-001'
    VEO_2_0_EXP = 'veo-2.0-generate-exp'
    VEO_3_1 = 'veo-3.1-generate-001'
    VEO_3_1_FAST = 'veo-3.1-fast-generate-001'


def is_veo_model(name: str) -> bool:
    """Check if a model name is a Veo model.

    Args:
        name: The model name to check.

    Returns:
        True if this is a Veo model name.
    """
    return name.lower().startswith('veo')


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
    enhance_prompt: bool | None = Field(default=None, alias='enhancePrompt', description='Enable prompt enhancement.')


# Alias for backwards compatibility with __init__.py exports
VeoConfig = VeoConfigSchema


DEFAULT_VEO_SUPPORT = Supports(
    media=True,
    multiturn=False,
    tools=False,
    system_role=True,
    output=['media'],
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


def _extract_text(request: GenerateRequest) -> str:
    """Extract text prompt from a GenerateRequest.

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


def _to_veo_parameters(config: Any) -> dict[str, Any]:  # noqa: ANN401
    """Convert config to Veo API parameters.

    Args:
        config: The model configuration (VeoConfigSchema or dict).

    Returns:
        Dictionary of Veo API parameters.
    """
    if config is None:
        return {}

    if isinstance(config, VeoConfigSchema):
        params = config.model_dump(by_alias=True, exclude_none=True)
    elif isinstance(config, dict):
        params = {k: v for k, v in config.items() if v is not None}
    else:
        return {}

    return params


def _from_veo_operation(api_op: dict[str, Any]) -> Operation:
    """Convert Veo API operation to Genkit Operation.

    The ``response`` value in ``api_op`` may be either:

    * A plain dict (from the ``start`` method, or legacy REST responses).
    * A ``GenerateVideosResponse`` Pydantic model (from the ``check`` method,
      which stores the SDK object directly).

    This function handles both cases when extracting video URIs.

    Args:
        api_op: The raw API operation response dict.

    Returns:
        A Genkit Operation object.
    """
    op = Operation(
        id=api_op.get('name', ''),
        done=api_op.get('done', False),
    )

    # Handle error
    if api_op.get('error'):
        op.error = Error(message=api_op['error'].get('message', 'Unknown error'))
        return op

    # Handle response with generated videos.
    response = api_op.get('response')
    if response is None:
        return op

    # Extract video URIs — response may be a Pydantic model or a dict.
    uris: list[str] = []
    if hasattr(response, 'generated_videos'):
        # Pydantic GenerateVideosResponse from the SDK (check path).
        for gv in response.generated_videos or []:
            if gv.video and gv.video.uri:
                uris.append(gv.video.uri)
    elif isinstance(response, dict):
        # Plain dict (start path or legacy REST).
        video_response = response.get('generateVideoResponse', {})
        for sample in video_response.get('generatedSamples', []):
            video = sample.get('video', {})
            uri = video.get('uri')
            if uri:
                uris.append(uri)

    if uris:
        content = [{'media': {'url': uri}} for uri in uris]
        op.output = {
            'finishReason': 'stop',
            'message': {
                'role': 'model',
                'content': content,
            },
        }

    return op


class VeoModel:
    """Veo video generation model.

    This class implements both the standard model interface (for Vertex AI)
    and the background model pattern (for GoogleAI) for Veo video generation.
    """

    def __init__(self, version: str, client: genai.Client) -> None:
        """Initialize Veo model.

        Args:
            version: The Veo model version.
            client: The Google GenAI client.
        """
        self._version = version
        self._client = client

    def _build_prompt(self, request: GenerateRequest) -> str:
        """Build prompt request from Genkit request."""
        prompt = []
        for message in request.messages:
            for part in message.content:
                if isinstance(part.root, TextPart):
                    prompt.append(part.root.text)
                else:
                    # TODO(#4363): Support image input if Veo supports it (e.g. for image-to-video)
                    # For now, strict text text-to-video
                    pass
        return ' '.join(prompt)

    async def generate(self, request: GenerateRequest, _: ActionRunContext) -> GenerateResponse:
        """Handle a generation request (synchronous/blocking mode for Vertex AI).

        Args:
            request: The generation request.
            _: action context

        Returns:
            The model's response.
        """
        if request.tools:
            raise ValueError('Tools are not supported for this model.')

        prompt = self._build_prompt(request)
        config = self._get_config(request)

        with tracer.start_as_current_span('generate_videos'):
            operation = await self._client.aio.models.generate_videos(model=self._version, prompt=prompt, config=config)

            # Handling LRO. Using cast(Any) to avoid strict type definition issues for operation.result()
            op = cast(Any, operation)
            if hasattr(op, 'result'):
                # Check if result is a coroutine (awaitable) or direct value
                res = op.result()
                if asyncio.iscoroutine(res):
                    response = await res
                else:
                    response = res
            else:
                response = op

            content = self._contents_from_response(cast(genai_types.GenerateVideosResponse, response))

        return GenerateResponse(
            message=Message(
                content=content,
                role=Role.MODEL,
            )
        )

    async def start(self, request: GenerateRequest, ctx: ActionRunContext) -> Operation:
        """Start a video generation operation (background model pattern for GoogleAI).

        Args:
            request: The generation request.
            ctx: The action run context.

        Returns:
            An Operation with the job ID.
        """
        if request.tools:
            raise ValueError('Tools are not supported for this model.')

        prompt = _extract_text(request)
        if not prompt:
            raise ValueError('Veo requires a text prompt')

        # Call the generateVideos API
        response = await self._client.aio.models.generate_videos(
            model=self._version,
            prompt=prompt,
            # pyrefly: ignore[bad-argument-type] - config dict matches GenerateVideosConfigDict
            config=request.config if isinstance(request.config, dict) else None,
        )

        # Convert to Operation
        return _from_veo_operation({
            'name': response.name if hasattr(response, 'name') else str(response),
            'done': getattr(response, 'done', False),
        })

    async def check(self, operation: Operation) -> Operation:
        """Check the status of a video generation operation.

        Args:
            operation: The operation to check.

        Returns:
            Updated Operation with current status.
        """
        # Get the operation status using the public operations.get() API
        # See: https://ai.google.dev/gemini-api/docs/video
        # Create a GenerateVideosOperation object from the operation ID
        op_request = genai_types.GenerateVideosOperation.model_validate({'name': operation.id})
        response = await self._client.aio.operations.get(operation=op_request)

        # Convert response to dict for processing
        op_dict = {
            'name': getattr(response, 'name', operation.id),
            'done': getattr(response, 'done', False),
        }

        if hasattr(response, 'error') and response.error:
            op_dict['error'] = {'message': str(response.error)}

        if hasattr(response, 'response') and response.response:
            op_dict['response'] = response.response

        return _from_veo_operation(op_dict)

    def _get_config(self, request: GenerateRequest) -> genai_types.GenerateVideosConfigOrDict | None:
        cfg = None
        if request.config:
            # Simple cast/validate
            cfg = request.config
        return cfg

    def _contents_from_response(self, response: genai_types.GenerateVideosResponse) -> list[Part]:
        content = []
        if response.generated_videos:
            for video in response.generated_videos:
                # Video URI is typically in video.video.uri
                if video.video and video.video.uri:
                    uri = video.video.uri
                    content.append(
                        Part(
                            root=MediaPart(
                                media=Media(
                                    url=uri,
                                    content_type='video/mp4',
                                )
                            )
                        )
                    )
        return content

    @property
    def metadata(self) -> dict:
        """Model metadata."""
        return {'model': {'supports': DEFAULT_VEO_SUPPORT.model_dump(by_alias=True)}}
