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

"""OpenAI-compatible audio models for Genkit (TTS and STT).

Provides text-to-speech (TTS) and speech-to-text (STT / transcription)
capabilities via the OpenAI Audio API.

Supported TTS models: tts-1, tts-1-hd, gpt-4o-mini-tts
Supported STT models: gpt-4o-transcribe, gpt-4o-mini-transcribe, whisper-1

Data Flow (TTS)::

    ┌─────────────────────────────────────────────────────────────────────┐
    │  GenerateRequest (text input)                                       │
    │         │                                                           │
    │         ▼                                                           │
    │  to_tts_params()  ──►  SpeechCreateParams                           │
    │         │                                                           │
    │         ▼                                                           │
    │  client.audio.speech.create()                                       │
    │         │                                                           │
    │         ▼                                                           │
    │  to_tts_response()  ──►  GenerateResponse (audio media part)        │
    └─────────────────────────────────────────────────────────────────────┘

Data Flow (STT)::

    ┌─────────────────────────────────────────────────────────────────────┐
    │  GenerateRequest (audio media input)                                │
    │         │                                                           │
    │         ▼                                                           │
    │  to_stt_params()  ──►  TranscriptionCreateParams                    │
    │         │                                                           │
    │         ▼                                                           │
    │  client.audio.transcriptions.create()                               │
    │         │                                                           │
    │         ▼                                                           │
    │  to_stt_response()  ──►  GenerateResponse (text part)               │
    └─────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import base64
from typing import Any

from openai import AsyncOpenAI
from openai._legacy_response import HttpxBinaryResponseContent
from openai.types.audio import Transcription

from genkit.ai import ActionRunContext
from genkit.core.typing import FinishReason
from genkit.plugins.compat_oai.models.utils import (
    _extract_media,
    _extract_text,
    _find_text,
    decode_data_uri_bytes,
    extract_config_dict,
)
from genkit.types import (
    GenerateRequest,
    GenerateResponse,
    Media,
    MediaPart,
    Message,
    ModelInfo,
    Part,
    Role,
    Supports,
    TextPart,
)

# Maps audio response formats to their MIME types.
RESPONSE_FORMAT_MEDIA_TYPES: dict[str, str] = {
    'mp3': 'audio/mpeg',
    'opus': 'audio/opus',
    'aac': 'audio/aac',
    'flac': 'audio/flac',
    'wav': 'audio/wav',
    'pcm': 'audio/L16',
}

# Maps content types to file extensions for STT input filenames.
_CONTENT_TYPE_TO_EXTENSION: dict[str, str] = {
    'audio/mpeg': 'mp3',
    'audio/mp3': 'mp3',
    'audio/wav': 'wav',
    'audio/ogg': 'ogg',
    'audio/flac': 'flac',
    'audio/webm': 'webm',
    'audio/mp4': 'mp4',
}

# Supported TTS models with their metadata.
SUPPORTED_TTS_MODELS: dict[str, ModelInfo] = {
    'tts-1': ModelInfo(
        label='OpenAI - TTS 1',
        supports=Supports(
            media=False,
            output=['media'],
            multiturn=False,
            system_role=False,
            tools=False,
        ),
    ),
    'tts-1-hd': ModelInfo(
        label='OpenAI - TTS 1 HD',
        supports=Supports(
            media=False,
            output=['media'],
            multiturn=False,
            system_role=False,
            tools=False,
        ),
    ),
    'gpt-4o-mini-tts': ModelInfo(
        label='OpenAI - GPT-4o Mini TTS',
        supports=Supports(
            media=False,
            output=['media'],
            multiturn=False,
            system_role=False,
            tools=False,
        ),
    ),
}

# Supported STT / transcription models with their metadata.
SUPPORTED_STT_MODELS: dict[str, ModelInfo] = {
    'gpt-4o-transcribe': ModelInfo(
        label='OpenAI - GPT-4o Transcribe',
        supports=Supports(
            media=True,
            output=['text', 'json'],
            multiturn=False,
            system_role=False,
            tools=False,
        ),
    ),
    'gpt-4o-mini-transcribe': ModelInfo(
        label='OpenAI - GPT-4o Mini Transcribe',
        supports=Supports(
            media=True,
            output=['text', 'json'],
            multiturn=False,
            system_role=False,
            tools=False,
        ),
    ),
    'whisper-1': ModelInfo(
        label='OpenAI - Whisper 1',
        supports=Supports(
            media=True,
            output=['text', 'json'],
            multiturn=False,
            system_role=False,
            tools=False,
        ),
    ),
}


def _to_tts_params(
    model_name: str,
    request: GenerateRequest,
) -> dict[str, Any]:
    """Convert a GenerateRequest into OpenAI TTS parameters.

    Args:
        model_name: The TTS model name (e.g., 'tts-1').
        request: The Genkit generate request.

    Returns:
        A dictionary of parameters for client.audio.speech.create().
    """
    text = _extract_text(request)
    config = extract_config_dict(request)

    params: dict[str, Any] = {
        'model': config.pop('version', None) or model_name,
        'input': text,
        'voice': config.pop('voice', 'alloy'),
    }

    # Optional TTS-specific params.
    for key in ('speed', 'response_format', 'instructions'):
        if key in config:
            params[key] = config.pop(key)

    # Strip standard GenAI config keys.
    for key in ('temperature', 'maxOutputTokens', 'stopSequences', 'topK', 'topP'):
        config.pop(key, None)

    return {k: v for k, v in params.items() if v is not None}


def _to_tts_response(
    response: HttpxBinaryResponseContent,
    response_format: str = 'mp3',
) -> GenerateResponse:
    """Convert an OpenAI speech response to a Genkit GenerateResponse.

    The response body is read as bytes and encoded as a base64 data URI.

    Args:
        response: The raw HTTP response from client.audio.speech.create().
        response_format: The audio format used (determines MIME type).

    Returns:
        A GenerateResponse with a media part containing the audio data.
    """
    # The response from speech.create() is an HttpxBinaryResponseContent
    # which supports .read() to get raw bytes.
    audio_bytes = response.read()
    media_type = RESPONSE_FORMAT_MEDIA_TYPES.get(response_format, 'audio/mpeg')
    b64_data = base64.b64encode(audio_bytes).decode('ascii')

    return GenerateResponse(
        message=Message(
            role=Role.MODEL,
            content=[
                Part(
                    root=MediaPart(
                        media=Media(
                            content_type=media_type,
                            url=f'data:{media_type};base64,{b64_data}',
                        )
                    )
                )
            ],
        ),
        finish_reason=FinishReason.STOP,
    )


def _to_stt_params(
    model_name: str,
    request: GenerateRequest,
) -> dict[str, Any]:
    """Convert a GenerateRequest into OpenAI transcription parameters.

    Extracts the audio media from the first message and converts it into
    a file-like object suitable for the transcriptions API.

    Args:
        model_name: The STT model name (e.g., 'whisper-1').
        request: The Genkit generate request.

    Returns:
        A dictionary of parameters for client.audio.transcriptions.create().
    """
    media_url, content_type = _extract_media(request)
    config = extract_config_dict(request)

    audio_bytes = decode_data_uri_bytes(media_url)

    ext = _CONTENT_TYPE_TO_EXTENSION.get(content_type, 'mp3')

    params: dict[str, Any] = {
        'model': config.pop('version', None) or model_name,
        'file': (f'input.{ext}', audio_bytes, content_type or 'audio/mpeg'),
    }

    prompt_text = _find_text(request)
    if prompt_text:
        params['prompt'] = prompt_text

    # Temperature if provided.
    if temp := config.pop('temperature', None):
        params['temperature'] = temp

    # Optional STT-specific params.
    for key in ('language', 'timestamp_granularities'):
        if key in config:
            params[key] = config.pop(key)

    # Determine response format: config override > output format > default.
    response_format = config.pop('response_format', None)
    if not response_format and request.output and request.output.format in ('json', 'text'):
        response_format = request.output.format
    params['response_format'] = response_format or 'text'

    # Strip standard GenAI config keys.
    for key in ('maxOutputTokens', 'stopSequences', 'topK', 'topP'):
        config.pop(key, None)

    return {k: v for k, v in params.items() if v is not None}


def _to_stt_response(result: Transcription | str) -> GenerateResponse:
    """Convert an OpenAI transcription result to a Genkit GenerateResponse.

    Handles the full union of types returned by transcriptions.create().
    All non-str result types (Transcription, TranscriptionVerbose,
    TranscriptionDiarized) have a .text attribute.

    Args:
        result: The transcription result (either a Transcription-like
            object with a .text attribute, or a plain string).

    Returns:
        A GenerateResponse with a text part containing the transcription.
    """
    if isinstance(result, str):
        text = result
    elif hasattr(result, 'text'):
        text = result.text
    else:
        text = str(result)
    return GenerateResponse(
        message=Message(
            role=Role.MODEL,
            content=[Part(root=TextPart(text=text))],
        ),
        finish_reason=FinishReason.STOP,
    )


class OpenAITTSModel:
    """Handles text-to-speech via the OpenAI Audio API.

    Args:
        model_name: The TTS model to use (e.g., 'tts-1').
        client: An async OpenAI client instance.
    """

    def __init__(self, model_name: str, client: AsyncOpenAI) -> None:
        """Initialize the TTS model.

        Args:
            model_name: The TTS model to use (e.g., 'tts-1').
            client: An async OpenAI client instance.
        """
        self._model_name = model_name
        self._client = client

    @property
    def name(self) -> str:
        """The name of the TTS model."""
        return self._model_name

    async def generate(self, request: GenerateRequest, ctx: ActionRunContext) -> GenerateResponse:
        """Generate speech audio from the request.

        Args:
            request: The generate request containing the text input.
            ctx: The action run context.

        Returns:
            A GenerateResponse containing audio media parts.
        """
        params = _to_tts_params(self._model_name, request)
        response_format = params.get('response_format', 'mp3')
        result = await self._client.audio.speech.create(**params)
        return _to_tts_response(result, response_format)


class OpenAISTTModel:
    """Handles speech-to-text (transcription) via the OpenAI Audio API.

    Args:
        model_name: The STT model to use (e.g., 'whisper-1').
        client: An async OpenAI client instance.
    """

    def __init__(self, model_name: str, client: AsyncOpenAI) -> None:
        """Initialize the STT model.

        Args:
            model_name: The STT model to use (e.g., 'whisper-1').
            client: An async OpenAI client instance.
        """
        self._model_name = model_name
        self._client = client

    @property
    def name(self) -> str:
        """The name of the STT model."""
        return self._model_name

    async def generate(self, request: GenerateRequest, ctx: ActionRunContext) -> GenerateResponse:
        """Transcribe audio from the request.

        Args:
            request: The generate request containing audio media input.
            ctx: The action run context.

        Returns:
            A GenerateResponse containing the transcribed text.
        """
        params = _to_stt_params(self._model_name, request)
        result = await self._client.audio.transcriptions.create(
            **params,
            stream=False,
        )
        # transcriptions.create(stream=False) returns a union of
        # Transcription | TranscriptionVerbose | TranscriptionDiarized | str.
        # _to_stt_response handles all of these via isinstance/hasattr checks.
        return _to_stt_response(result)
