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

"""Tests for Veo video generation model helpers and lifecycle."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from genkit_google_genai.models.veo import (
    VeoConfig,
    VeoModel,
    VeoVersion,
    _from_veo_operation,
    is_veo_model,
)
from google.genai import types as genai_types
from google.genai.errors import APIError

from genkit import (
    ActionRunContext,
    FinishReason,
    GenkitError,
    Message,
    ModelRequest,
    ModelResponse,
    Part,
    Role,
    TextPart,
)
from genkit.model import Operation


def _sdk_op(
    *,
    name: str,
    done: bool = False,
    error: Any = None,
    response: genai_types.GenerateVideosResponse | None = None,
) -> genai_types.GenerateVideosOperation:
    op = genai_types.GenerateVideosOperation(response=response)
    op.name = name
    op.done = done
    op.error = error
    return op


def _media(output: object) -> list:
    assert isinstance(output, ModelResponse)
    assert output.finish_reason == FinishReason.STOP
    return output.media


def _text_request(*, config: object | None = None) -> ModelRequest:
    return ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(TextPart(text='a cat walking'))])],
        config=config,  # type: ignore[arg-type]
    )


class TestIsVeoModel:
    """Tests for is_veo_model."""

    def test_veo_model_name(self) -> None:
        """Veo model names are recognized."""
        assert is_veo_model('veo-2.0-generate-001') is True

    def test_veo_uppercase(self) -> None:
        """Case-insensitive matching works."""
        assert is_veo_model('VEO-2.0-generate-001') is True

    def test_non_veo_model(self) -> None:
        """Non-Veo model names are rejected."""
        assert is_veo_model('gemini-2.0-flash') is False

    def test_namespaced_veo_model(self) -> None:
        """Plugin prefixes are stripped before the ``veo-`` check."""
        assert is_veo_model('googleai/veo-3.0-generate-001') is True

    def test_substring_veo_is_rejected(self) -> None:
        """A bare ``veo`` substring is not enough; the id has to start with ``veo-``."""
        assert is_veo_model('devotional-hymn') is False


class TestVeoVersion:
    """Tests for VeoVersion enum convenience constants."""

    @pytest.mark.parametrize(
        'version',
        [
            VeoVersion.VEO_3_1_PREVIEW,
            VeoVersion.VEO_3_1_FAST_PREVIEW,
            VeoVersion.VEO_3_0,
            VeoVersion.VEO_3_0_FAST,
        ],
    )
    def test_new_googleai_models_are_recognized(self, version: VeoVersion) -> None:
        """New Veo 3.0/3.1 model constants map to valid Veo names."""
        assert is_veo_model(version.value) is True


class TestFromVeoOperation:
    """``_from_veo_operation`` reads the SDK operation and resolves playable media."""

    def test_pending_operation_leaves_output_and_error_none(self) -> None:
        """An in-progress operation has no response — output and error stay None."""
        op = _from_veo_operation(api_op=_sdk_op(name='operations/123', done=False))
        assert op.id == 'operations/123'
        assert op.done is False
        assert op.output is None
        assert op.error is None

    def test_sdk_object_error_populates_op_error_message(self) -> None:
        """An SDK error object with a message attribute populates op.error.message."""
        error_obj = MagicMock()
        error_obj.message = 'Quota exceeded from SDK object'
        op = _from_veo_operation(
            api_op=_sdk_op(name='operations/456', done=True, error=error_obj),
        )
        assert op.id == 'operations/456'
        assert op.done is True
        assert op.error is not None
        assert op.error.message == 'Quota exceeded from SDK object'
        assert op.output is None

    def test_dict_error_populates_op_error_message(self) -> None:
        """A dictionary error payload populates op.error.message."""
        op = _from_veo_operation(
            api_op=_sdk_op(name='operations/456', done=True, error={'message': 'Quota exceeded'}),
        )
        assert op.id == 'operations/456'
        assert op.done is True
        assert op.error is not None
        assert op.error.message == 'Quota exceeded'
        assert op.output is None

    def test_string_error_populates_op_error_message(self) -> None:
        """A raw string error payload populates op.error.message."""
        op = _from_veo_operation(
            api_op=_sdk_op(name='operations/456', done=True, error='Rate limit exceeded'),
        )
        assert op.id == 'operations/456'
        assert op.done is True
        assert op.error is not None
        assert op.error.message == 'Rate limit exceeded'
        assert op.output is None

    def test_finished_operation_resolves_http_urls_to_model_response(self) -> None:
        """AI Studio HTTP download links are wrapped into ModelResponse with finish_reason=STOP."""
        op = _from_veo_operation(
            api_op=_sdk_op(
                name='operations/789',
                done=True,
                response=genai_types.GenerateVideosResponse(
                    generated_videos=[
                        genai_types.GeneratedVideo(video=genai_types.Video(uri='https://example.com/v1.mp4')),
                        genai_types.GeneratedVideo(video=genai_types.Video(uri='https://example.com/v2.mp4')),
                    ],
                ),
            ),
        )
        assert op.done is True
        assert [part.url for part in _media(op.output)] == [
            'https://example.com/v1.mp4',
            'https://example.com/v2.mp4',
        ]

    def test_finished_operation_resolves_gcs_uris_to_model_response(self) -> None:
        """Vertex gs:// bucket paths are preserved in media.url with content_type."""
        op = _from_veo_operation(
            api_op=_sdk_op(
                name='operations/gcs',
                done=True,
                response=genai_types.GenerateVideosResponse(
                    generated_videos=[
                        genai_types.GeneratedVideo(
                            video=genai_types.Video(uri='gs://bucket/clip.mp4', mime_type='video/mp4'),
                        ),
                    ],
                ),
            ),
        )
        media = _media(op.output)[0]
        assert media.url == 'gs://bucket/clip.mp4'
        assert media.content_type == 'video/mp4'

    def test_finished_operation_encodes_inline_bytes_to_base64_data_uri(self) -> None:
        """Vertex inline video_bytes are encoded to base64 data URIs."""
        op = _from_veo_operation(
            api_op=_sdk_op(
                name='operations/bytes',
                done=True,
                response=genai_types.GenerateVideosResponse(
                    generated_videos=[
                        genai_types.GeneratedVideo(
                            video=genai_types.Video(video_bytes=b'\x00\x00', mime_type='video/mp4'),
                        ),
                    ],
                ),
            ),
        )
        media = _media(op.output)[0]
        assert media.content_type == 'video/mp4'
        assert media.url == 'data:video/mp4;base64,AAA='

    def test_finished_operation_sniffs_mime_type_from_uri_extension(self) -> None:
        """URI file extensions (.webm, .mov) are sniffed when mime_type is omitted."""
        op = _from_veo_operation(
            api_op=_sdk_op(
                name='operations/webm',
                done=True,
                response=genai_types.GenerateVideosResponse(
                    generated_videos=[
                        genai_types.GeneratedVideo(video=genai_types.Video(uri='gs://bucket/clip.webm')),
                    ],
                ),
            ),
        )
        media = _media(op.output)[0]
        assert media.url == 'gs://bucket/clip.webm'
        assert media.content_type == 'video/webm'

    def test_finished_operation_attaches_sdk_response_to_raw_field(self) -> None:
        """The complete SDK response is attached to op.output.raw."""
        sdk_response = genai_types.GenerateVideosResponse(
            generated_videos=[
                genai_types.GeneratedVideo(video=genai_types.Video(uri='https://example.com/v1.mp4')),
            ],
            rai_media_filtered_count=0,
        )
        op = _from_veo_operation(
            api_op=_sdk_op(name='operations/raw', done=True, response=sdk_response),
        )
        assert isinstance(op.output, ModelResponse)
        assert op.output.raw is not None
        assert 'generatedVideos' in op.output.raw or 'generated_videos' in op.output.raw

    def test_empty_video_bytes_surfaces_empty_media_error(self) -> None:
        """An operation completing with empty bytes (b'') and no URI surfaces an explicit error."""
        op = _from_veo_operation(
            api_op=_sdk_op(
                name='operations/empty-bytes',
                done=True,
                response=genai_types.GenerateVideosResponse(
                    generated_videos=[
                        genai_types.GeneratedVideo(video=genai_types.Video(video_bytes=b'')),
                    ],
                ),
            ),
        )
        assert op.done is True
        assert op.output is None
        assert op.error is not None
        assert op.error.message == 'Operation completed but returned no playable media.'

    def test_response_none_stays_pending(self) -> None:
        """No response yet is a pending ticket, not a crash."""
        op = _from_veo_operation(api_op=_sdk_op(name='operations/null', done=False, response=None))
        assert op.output is None


class TestVeoSafetyFilters:
    """Safety (RAI) filtering contract tests."""

    def test_total_rai_filter_sets_op_error_with_backend_reasons(self) -> None:
        """A finished job with no videos and a RAI count populates op.error with filter reasons."""
        op = _from_veo_operation(
            api_op=_sdk_op(
                name='operations/rai',
                done=True,
                response=genai_types.GenerateVideosResponse(
                    generated_videos=[],
                    rai_media_filtered_count=1,
                    rai_media_filtered_reasons=['1 videos were filtered out.'],
                ),
            ),
        )
        assert op.output is None
        assert op.error is not None
        assert op.error.message == '1 videos were filtered out.'

    def test_total_rai_filter_falls_back_to_default_safety_message_when_reasons_empty(self) -> None:
        """A finished job with RAI count but empty reasons uses the default safety message."""
        op = _from_veo_operation(
            api_op=_sdk_op(
                name='operations/rai-empty',
                done=True,
                response=genai_types.GenerateVideosResponse(
                    generated_videos=[],
                    rai_media_filtered_count=2,
                    rai_media_filtered_reasons=[],
                ),
            ),
        )
        assert op.output is None
        assert op.error is not None
        assert op.error.message == 'All generated videos were filtered out by safety filters.'

    def test_partial_rai_filter_returns_valid_videos_and_preserves_filter_count_in_raw(self) -> None:
        """Partial RAI filtering returns surviving videos with finish_reason=STOP and preserves raw count."""
        op = _from_veo_operation(
            api_op=_sdk_op(
                name='operations/partial-rai',
                done=True,
                response=genai_types.GenerateVideosResponse(
                    generated_videos=[
                        genai_types.GeneratedVideo(video=genai_types.Video(uri='https://example.com/ok.mp4')),
                    ],
                    rai_media_filtered_count=1,
                    rai_media_filtered_reasons=['1 videos were filtered out.'],
                ),
            ),
        )
        assert op.done is True
        assert op.error is None
        assert len(_media(op.output)) == 1
        assert _media(op.output)[0].url == 'https://example.com/ok.mp4'
        assert isinstance(op.output, ModelResponse)
        assert op.output.raw is not None


class TestVeoModelLifecycle:
    """Model execution and polling lifecycle tests."""

    @pytest.mark.asyncio
    async def test_start_passes_generate_videos_config_and_returns_ticket(self) -> None:
        """A typed Veo config dumps aspectRatio / durationSeconds onto generate_videos."""
        client = MagicMock()
        client.aio.models.generate_videos = AsyncMock(return_value=_sdk_op(name='operations/1', done=False))
        veo = VeoModel('veo-3.0-generate-001', client)
        request = _text_request(
            config=VeoConfig.model_validate({'aspectRatio': '16:9', 'durationSeconds': 5, 'fooBar': 1}),
        )

        await veo.start(request, ActionRunContext())

        called = client.aio.models.generate_videos.await_args
        assert called is not None
        cfg = called.kwargs['config']
        assert cfg.aspect_ratio == '16:9'
        assert cfg.duration_seconds == 5
        assert cfg.http_options is not None
        assert cfg.http_options.extra_body == {'parameters': {'fooBar': 1}}

    @pytest.mark.asyncio
    async def test_start_no_config_sends_none(self) -> None:
        """No config is a valid start; generate_videos gets no knobs."""
        client = MagicMock()
        client.aio.models.generate_videos = AsyncMock(return_value=_sdk_op(name='operations/1', done=False))
        veo = VeoModel('veo-3.0-generate-001', client)

        await veo.start(_text_request(), ActionRunContext())

        called = client.aio.models.generate_videos.await_args
        assert called is not None
        assert called.kwargs['config'] is None

    @pytest.mark.asyncio
    async def test_check_polls_operation_by_sdk_name_and_returns_updated_operation(self) -> None:
        """Polling calls operations.get using the SDK GenerateVideosOperation name."""
        client = MagicMock()
        client.aio.operations.get = AsyncMock(
            return_value=_sdk_op(
                name='operations/123',
                done=True,
                response=genai_types.GenerateVideosResponse(
                    generated_videos=[
                        genai_types.GeneratedVideo(video=genai_types.Video(uri='https://example.com/done.mp4')),
                    ],
                ),
            ),
        )
        model = VeoModel('veo-3.0-generate-001', client)
        updated = await model.check(Operation(id='operations/123'))

        assert updated.done is True
        assert _media(updated.output)[0].url == 'https://example.com/done.mp4'

    @pytest.mark.asyncio
    async def test_check_wraps_api_error_into_genkit_error(self) -> None:
        """A 503 on the poll must stay retryable UNAVAILABLE, not collapse to INTERNAL."""
        client = MagicMock()
        client.aio.operations.get = AsyncMock(side_effect=APIError(503, {'error': {'message': 'overloaded'}}))
        model = VeoModel('veo-3.0-generate-001', client)

        with pytest.raises(GenkitError) as raised:
            await model.check(Operation(id='operations/abc'))
        assert raised.value.status == 'UNAVAILABLE'

    def test_invalid_sdk_field_raises_invalid_argument(self) -> None:
        """SDK type errors become a named INVALID_ARGUMENT."""
        veo = VeoModel('veo-3.0-generate-001', MagicMock())
        request = _text_request(config=VeoConfig.model_construct(duration_seconds='nope'))

        with pytest.raises(GenkitError) as exc_info:
            veo._get_config(request)

        assert exc_info.value.status == 'INVALID_ARGUMENT'
        assert 'duration_seconds' in str(exc_info.value)
