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

"""Tests for OpenAI-compatible audio models (TTS and STT)."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from genkit.plugins.compat_oai.models.audio import (
    SUPPORTED_STT_MODELS,
    SUPPORTED_TTS_MODELS,
    OpenAISTTModel,
    OpenAITTSModel,
    _extract_media,
    _extract_text,
    _to_stt_params,
    _to_stt_response,
    _to_tts_params,
    _to_tts_response,
)
from genkit.types import (
    GenerateRequest,
    Media,
    MediaPart,
    Message,
    Part,
    Role,
    TextPart,
)


class TestExtractText:
    """Tests for extracting text from GenerateRequest."""

    def test_extracts_text(self) -> None:
        """Verify text extraction from a simple request."""
        request = GenerateRequest(
            messages=[
                Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))]),
            ],
        )
        got = _extract_text(request)
        assert got == 'Hello'

    def test_raises_on_empty(self) -> None:
        """Verify ValueError when messages list is empty."""
        request = GenerateRequest(messages=[])
        with pytest.raises(ValueError, match='No messages found'):
            _extract_text(request)


class TestExtractMedia:
    """Tests for extracting media URLs from GenerateRequest."""

    def test_extracts_media_url(self) -> None:
        """Verify media URL and content type extraction."""
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        Part(
                            root=MediaPart(
                                media=Media(
                                    content_type='audio/mpeg',
                                    url='data:audio/mpeg;base64,dGVzdA==',
                                )
                            )
                        ),
                    ],
                ),
            ],
        )
        url, content_type = _extract_media(request)
        assert content_type == 'audio/mpeg'
        assert 'base64' in url

    def test_raises_on_no_media(self) -> None:
        """Verify ValueError when no media content is found."""
        request = GenerateRequest(
            messages=[
                Message(role=Role.USER, content=[Part(root=TextPart(text='no media'))]),
            ],
        )
        with pytest.raises(ValueError, match='No media content found'):
            _extract_media(request)


class TestToTTSParams:
    """Tests for converting GenerateRequest to TTS params."""

    def test_basic_params(self) -> None:
        """Verify required TTS params with defaults."""
        request = GenerateRequest(
            messages=[
                Message(role=Role.USER, content=[Part(root=TextPart(text='Say hello'))]),
            ],
        )
        got = _to_tts_params('tts-1', request)
        assert got['model'] == 'tts-1'
        assert got['input'] == 'Say hello'
        assert got['voice'] == 'alloy'

    def test_custom_voice(self) -> None:
        """Verify custom voice config is applied."""
        request = GenerateRequest(
            messages=[
                Message(role=Role.USER, content=[Part(root=TextPart(text='test'))]),
            ],
            config={'voice': 'nova'},
        )
        got = _to_tts_params('tts-1', request)
        assert got['voice'] == 'nova'

    def test_strips_standard_config(self) -> None:
        """Verify standard GenAI keys are stripped."""
        request = GenerateRequest(
            messages=[
                Message(role=Role.USER, content=[Part(root=TextPart(text='test'))]),
            ],
            config={'temperature': 0.5, 'topK': 40},
        )
        got = _to_tts_params('tts-1', request)
        assert 'temperature' not in got
        assert 'topK' not in got


class TestToTTSResponse:
    """Tests for converting speech response to GenerateResponse."""

    def test_converts_audio_to_media_part(self) -> None:
        """Verify audio bytes are encoded as base64 data URI."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'fake audio data'

        got = _to_tts_response(mock_response, 'mp3')
        assert got.message is not None
        assert len(got.message.content) == 1

        part = got.message.content[0].root
        assert isinstance(part, MediaPart)
        assert part.media.content_type == 'audio/mpeg'
        assert str(part.media.url).startswith('data:audio/mpeg;base64,')

    def test_opus_format(self) -> None:
        """Verify opus format uses correct MIME type."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'opus data'

        got = _to_tts_response(mock_response, 'opus')
        assert got.message is not None
        part = got.message.content[0].root
        assert isinstance(part, MediaPart)
        assert part.media.content_type == 'audio/opus'


class TestToSTTParams:
    """Tests for converting GenerateRequest to STT params."""

    def test_basic_params(self) -> None:
        """Verify required STT params from audio media input."""
        audio_data = base64.b64encode(b'fake audio').decode('ascii')
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        Part(
                            root=MediaPart(
                                media=Media(
                                    content_type='audio/mpeg',
                                    url=f'data:audio/mpeg;base64,{audio_data}',
                                )
                            )
                        ),
                    ],
                ),
            ],
        )
        got = _to_stt_params('whisper-1', request)
        assert got['model'] == 'whisper-1'
        assert 'file' in got

    def test_with_prompt_context(self) -> None:
        """Verify prompt text is included when present alongside media."""
        audio_data = base64.b64encode(b'fake audio').decode('ascii')
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        Part(root=TextPart(text='Transcribe this meeting')),
                        Part(
                            root=MediaPart(
                                media=Media(
                                    content_type='audio/mpeg',
                                    url=f'data:audio/mpeg;base64,{audio_data}',
                                )
                            )
                        ),
                    ],
                ),
            ],
        )
        got = _to_stt_params('whisper-1', request)
        assert got.get('prompt') == 'Transcribe this meeting'


class TestToSTTResponse:
    """Tests for converting transcription result to GenerateResponse."""

    def test_transcription_object(self) -> None:
        """Verify Transcription object is converted to text part."""
        mock_result = MagicMock()
        mock_result.text = 'Hello world'

        got = _to_stt_response(mock_result)
        assert got.message is not None
        part = got.message.content[0].root
        assert isinstance(part, TextPart)
        assert part.text == 'Hello world'

    def test_string_result(self) -> None:
        """Verify plain string result is wrapped as text part."""
        got = _to_stt_response('Plain text')
        assert got.message is not None
        part = got.message.content[0].root
        assert isinstance(part, TextPart)
        assert part.text == 'Plain text'


class TestModelRegistries:
    """Tests for model info registries."""

    def test_tts_models(self) -> None:
        """Verify all expected TTS models are registered."""
        for name in ('tts-1', 'tts-1-hd', 'gpt-4o-mini-tts'):
            assert name in SUPPORTED_TTS_MODELS, f'{name!r} not in SUPPORTED_TTS_MODELS'

    def test_stt_models(self) -> None:
        """Verify all expected STT models are registered."""
        for name in ('gpt-4o-transcribe', 'gpt-4o-mini-transcribe', 'whisper-1'):
            assert name in SUPPORTED_STT_MODELS, f'{name!r} not in SUPPORTED_STT_MODELS'

    def test_tts_models_support_media_output(self) -> None:
        """Verify all TTS models declare media output support."""
        for name, info in SUPPORTED_TTS_MODELS.items():
            assert info.supports is not None, f'{name} has no supports'
            assert 'media' in (info.supports.output or []), f"{name} should support 'media' output"

    def test_stt_models_support_media_input(self) -> None:
        """Verify all STT models declare media input support."""
        for name, info in SUPPORTED_STT_MODELS.items():
            assert info.supports is not None, f'{name} has no supports'
            assert info.supports.media, f'{name} should support media input'


class TestOpenAITTSModel:
    """Tests for the OpenAITTSModel class."""

    @pytest.mark.asyncio
    async def test_generate_calls_speech_create(self) -> None:
        """Verify generate() calls client.audio.speech.create."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'audio bytes'

        mock_client = AsyncMock()
        mock_client.audio.speech.create = AsyncMock(return_value=mock_response)

        model = OpenAITTSModel('tts-1', mock_client)
        request = GenerateRequest(
            messages=[
                Message(role=Role.USER, content=[Part(root=TextPart(text='Say hello'))]),
            ],
        )

        ctx = MagicMock()
        got = await model.generate(request, ctx)

        mock_client.audio.speech.create.assert_called_once()
        assert got.message is not None
        assert len(got.message.content) == 1

        part = got.message.content[0].root
        assert isinstance(part, MediaPart)


class TestOpenAISTTModel:
    """Tests for the OpenAISTTModel class."""

    @pytest.mark.asyncio
    async def test_generate_calls_transcription_create(self) -> None:
        """Verify generate() calls client.audio.transcriptions.create."""
        mock_result = MagicMock()
        mock_result.text = 'Transcribed text'

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_result)

        model = OpenAISTTModel('whisper-1', mock_client)
        audio_data = base64.b64encode(b'fake audio').decode('ascii')
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        Part(
                            root=MediaPart(
                                media=Media(
                                    content_type='audio/mpeg',
                                    url=f'data:audio/mpeg;base64,{audio_data}',
                                )
                            )
                        ),
                    ],
                ),
            ],
        )

        ctx = MagicMock()
        got = await model.generate(request, ctx)

        mock_client.audio.transcriptions.create.assert_called_once()
        assert got.message is not None

        part = got.message.content[0].root
        assert isinstance(part, TextPart)
        assert part.text == 'Transcribed text'
