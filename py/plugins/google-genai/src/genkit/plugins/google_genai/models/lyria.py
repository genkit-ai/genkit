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

"""Lyria audio generation model for Google Vertex AI plugin.

Lyria is Google's music and audio generation model that creates audio from
text prompts. It's available through Vertex AI only (not Google AI).

Architecture:
    ```
    ┌──────────────────────────────────────────────────────────────────────┐
    │                      Lyria Audio Generation Flow                      │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                       │
    │   Input                    Model                     Output           │
    │   ┌─────────┐             ┌─────────┐             ┌─────────┐        │
    │   │ Text    │ ─predict──► │ Lyria   │ ──────────► │  Audio  │        │
    │   │ Prompt  │             │ Model   │             │  (WAV)  │        │
    │   └─────────┘             └─────────┘             └─────────┘        │
    │                                                                       │
    └──────────────────────────────────────────────────────────────────────┘
    ```

Supported Models:
    +----------------------+--------------------------------------------------+
    | Model                | Description                                      |
    +----------------------+--------------------------------------------------+
    | lyria-002            | Lyria 002 - Audio generation from text           |
    +----------------------+--------------------------------------------------+

Example:
    >>> from genkit import Genkit
    >>> from genkit.plugins.google_genai import VertexAI
    >>>
    >>> ai = Genkit(plugins=[VertexAI(project='my-project')])
    >>>
    >>> # Generate audio
    >>> response = await ai.generate(
    ...     model='vertexai/lyria-002',
    ...     prompt='A peaceful piano melody with gentle rain sounds',
    ... )
    >>>
    >>> # Response contains audio as base64-encoded WAV
    >>> audio_content = response.message.content[0]
    >>> print(audio_content.media.content_type)  # 'audio/wav'

See Also:
    - Vertex AI Audio: https://cloud.google.com/vertex-ai/docs/generative-ai/audio
    - JS implementation: js/plugins/google-genai/src/vertexai/lyria.ts
"""

import sys

if sys.version_info < (3, 11):
    from strenum import StrEnum
else:
    from enum import StrEnum

from typing import Any

from pydantic import BaseModel, Field

from genkit import ModelInfo, Supports


class LyriaVersion(StrEnum):
    """Supported Lyria audio generation models."""

    LYRIA_002 = 'lyria-002'


# Known Lyria models
KNOWN_LYRIA_MODELS = {
    LyriaVersion.LYRIA_002,
}


def is_lyria_model(name: str) -> bool:
    """Check if a model name is a Lyria model.

    Args:
        name: The model name to check.

    Returns:
        True if this is a Lyria model name.
    """
    return name.startswith('lyria-')


class LyriaConfig(BaseModel):
    """Configuration options for Lyria audio generation.

    Attributes:
        negative_prompt: Text describing what to avoid in the audio.
        seed: Random seed for reproducible generation.
        sample_count: Number of audio samples to generate (default: 1).
        location: Must be 'global' for Lyria. Override if plugin uses different region.
    """

    negative_prompt: str | None = Field(default=None, alias='negativePrompt')
    seed: int | None = Field(default=None)
    sample_count: int | None = Field(default=None, ge=1, alias='sampleCount')
    location: str | None = Field(default=None)

    model_config = {'populate_by_name': True}


LYRIA_MODEL_INFO = ModelInfo(
    label='Vertex AI - Lyria',
    supports=Supports(
        media=True,
        multiturn=False,
        tools=False,
        system_role=False,
        output=['media'],
    ),
)


def lyria_model_info(version: str) -> ModelInfo:
    """Get model info for a Lyria model.

    Args:
        version: The Lyria model version.

    Returns:
        ModelInfo describing the model's capabilities.
    """
    return ModelInfo(
        label=f'Vertex AI - {version}',
        supports=LYRIA_MODEL_INFO.supports,
    )


def _extract_text(messages: list[Any]) -> str:
    """Extract text prompt from messages.

    Args:
        messages: The message list from a ModelRequest.

    Returns:
        The text prompt string.
    """
    if not messages:
        return ''
    for message in messages:
        for part in message.content:
            if hasattr(part.root, 'text') and part.root.text:
                return str(part.root.text)
    return ''


def _to_lyria_instances(prompt: str, config: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """Convert config to Lyria API instances.

    Args:
        prompt: The text prompt.
        config: The model configuration (LyriaConfig or dict).

    Returns:
        List of Lyria instance dictionaries.
    """
    instance: dict[str, Any] = {'prompt': prompt}

    if config is None:
        return [instance]

    if isinstance(config, LyriaConfig):
        if config.negative_prompt:
            instance['negativePrompt'] = config.negative_prompt
        if config.seed is not None:
            instance['seed'] = config.seed
    elif isinstance(config, dict):
        if 'negativePrompt' in config or 'negative_prompt' in config:
            instance['negativePrompt'] = config.get('negativePrompt') or config.get('negative_prompt')
        if 'seed' in config:
            instance['seed'] = config['seed']

    return [instance]


def _to_lyria_parameters(config: Any) -> dict[str, Any]:  # noqa: ANN401
    """Convert config to Lyria API parameters.

    Args:
        config: The model configuration (LyriaConfig or dict).

    Returns:
        Dictionary of Lyria API parameters.
    """
    if config is None:
        return {'sampleCount': 1}

    if isinstance(config, LyriaConfig):
        return {'sampleCount': config.sample_count or 1}
    elif isinstance(config, dict):
        return {'sampleCount': config.get('sampleCount') or config.get('sample_count') or 1}

    return {'sampleCount': 1}


def _from_lyria_prediction(prediction: dict[str, Any], index: int) -> dict[str, Any]:
    """Convert a Lyria prediction to a candidate.

    Args:
        prediction: The raw prediction from Lyria API.
        index: The candidate index.

    Returns:
        A candidate data dictionary.
    """
    b64data = prediction.get('bytesBase64Encoded', '')
    mime_type = prediction.get('mimeType', 'audio/wav')

    return {
        'index': index,
        'finishReason': 'stop',
        'message': {
            'role': 'model',
            'content': [
                {
                    'media': {
                        'url': f'data:{mime_type};base64,{b64data}',
                        'contentType': mime_type,
                    },
                },
            ],
        },
    }
