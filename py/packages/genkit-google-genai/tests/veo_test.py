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

"""Tests for Veo video generation model helpers.

``_from_veo_operation`` reads a ``GenerateVideosOperation`` and puts a
``ModelResponse`` on the ticket so callers can follow ``media.url``.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from genkit_google_genai.models.veo import (
    VeoConfigSchema,
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
    error: dict[str, object] | None = None,
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


@pytest.mark.asyncio
async def test_veo_start_dumps_aspect_ratio_and_duration() -> None:
    """A typed Veo config dumps aspectRatio / durationSeconds onto the SDK request."""
    client = MagicMock()
    client.aio.models.generate_videos = AsyncMock(return_value=_sdk_op(name='operations/1', done=False))
    veo = VeoModel('veo-3.0-generate-001', client)
    request = _text_request(
        config=VeoConfigSchema.model_validate({'aspectRatio': '16:9', 'durationSeconds': 5, 'fooBar': 1}),
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
async def test_veo_start_no_config_sends_none() -> None:
    """No config is a valid start; generate_videos gets no knobs."""
    client = MagicMock()
    client.aio.models.generate_videos = AsyncMock(return_value=_sdk_op(name='operations/1', done=False))
    veo = VeoModel('veo-3.0-generate-001', client)

    await veo.start(_text_request(), ActionRunContext())

    called = client.aio.models.generate_videos.await_args
    assert called is not None
    assert called.kwargs['config'] is None


def test_veo_rejects_raw_dicts() -> None:
    """A dict at the dump leaf means Action never produced the family instance."""
    veo = VeoModel('veo-3.0-generate-001', MagicMock())

    with pytest.raises(GenkitError) as exc_info:
        veo._get_config(_text_request(config={'aspectRatio': '16:9'}))

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert veo._version in str(exc_info.value)


def test_veo_invalid_sdk_field_is_invalid_argument() -> None:
    """SDK type errors become a named INVALID_ARGUMENT."""
    veo = VeoModel('veo-3.0-generate-001', MagicMock())
    request = _text_request(config=VeoConfigSchema.model_construct(duration_seconds='nope'))

    with pytest.raises(GenkitError) as exc_info:
        veo._get_config(request)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'duration_seconds' in str(exc_info.value)


class TestFromVeoOperation:
    """``_from_veo_operation`` reads the SDK operation, not a REST dump."""

    def test_pending_operation(self) -> None:
        """An in-progress operation has no response — output stays None."""
        op = _from_veo_operation(api_op=_sdk_op(name='operations/123', done=False))
        assert op.id == 'operations/123'
        assert op.done is False
        assert op.output is None
        assert op.error is None

    def test_error_operation(self) -> None:
        """An operation with an error populates op.error."""
        op = _from_veo_operation(
            api_op=_sdk_op(name='operations/456', done=True, error={'message': 'Quota exceeded'}),
        )
        assert op.id == 'operations/456'
        assert op.done is True
        assert op.error is not None
        assert op.error.message == 'Quota exceeded'
        assert op.output is None

    def test_response_with_videos(self) -> None:
        """A finished operation puts a ModelResponse with media urls on output."""
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

    def test_empty_videos(self) -> None:
        """A finished response with no generated_videos produces no output."""
        op = _from_veo_operation(
            api_op=_sdk_op(
                name='operations/empty',
                done=True,
                response=genai_types.GenerateVideosResponse(generated_videos=[]),
            ),
        )
        assert op.done is True
        assert op.output is None

    def test_response_none(self) -> None:
        """No response yet is a pending ticket, not a crash."""
        op = _from_veo_operation(api_op=_sdk_op(name='operations/null', done=False, response=None))
        assert op.output is None

    def test_inline_video_bytes(self) -> None:
        """Vertex often returns inline bytes and no download URL."""
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

    def test_gcs_uri(self) -> None:
        """Vertex can leave the clip on GCS in ``Video.uri``."""
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

    def test_done_with_rai_filter_is_an_error(self) -> None:
        """A finished job with no videos and a RAI count is not a success."""
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


@pytest.mark.asyncio
async def test_check_classifies_503_as_unavailable() -> None:
    """A 503 on the poll must stay retryable, not collapse to INTERNAL."""
    client = MagicMock()
    client.aio.operations.get = AsyncMock(side_effect=APIError(503, {'error': {'message': 'overloaded'}}))
    model = VeoModel('veo-3.0-generate-001', client)

    with pytest.raises(GenkitError) as raised:
        await model.check(Operation(id='operations/abc'))
    assert raised.value.status == 'UNAVAILABLE'
