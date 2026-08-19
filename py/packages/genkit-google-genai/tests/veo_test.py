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

Verifies _from_veo_operation handles both dict-based responses (from the
start path) and Pydantic GenerateVideosResponse objects (from the check
path where the SDK returns a model instance).
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

from genkit import ActionRunContext, GenkitError, Message, ModelRequest, Part, Role, TextPart


def _text_request(*, config: object | None = None) -> ModelRequest:
    return ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='a cat walking'))])],
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
    op = MagicMock()
    op.name = 'operations/1'
    op.done = False
    client.aio.models.generate_videos = AsyncMock(return_value=op)
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
    op = MagicMock()
    op.name = 'operations/1'
    op.done = False
    client.aio.models.generate_videos = AsyncMock(return_value=op)
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
    """Tests for _from_veo_operation.

    This function must handle two shapes for the 'response' value:

    1. A plain dict — returned by the start() path or legacy REST.
    2. A GenerateVideosResponse Pydantic model — returned by the check()
       path where the SDK object is stored directly.

    Regression: before the fix, case 2 raised
    ``AttributeError: 'GenerateVideosResponse' object has no attribute 'get'``
    because the code unconditionally called ``.get()`` on the response.
    """

    def test_pending_operation(self) -> None:
        """An in-progress operation has no response — output stays None."""
        op = _from_veo_operation({
            'name': 'operations/123',
            'done': False,
        })
        assert op.id == 'operations/123'
        assert op.done is False
        assert op.output is None
        assert op.error is None

    def test_error_operation(self) -> None:
        """An operation with an error populates op.error."""
        op = _from_veo_operation({
            'name': 'operations/456',
            'done': True,
            'error': {'message': 'Quota exceeded'},
        })
        assert op.id == 'operations/456'
        assert op.done is True
        assert op.error is not None
        assert op.error.message == 'Quota exceeded'
        assert op.output is None

    def test_dict_response_with_videos(self) -> None:
        """Dict-shaped response extracts video URIs (start path)."""
        op = _from_veo_operation({
            'name': 'operations/789',
            'done': True,
            'response': {
                'generateVideoResponse': {
                    'generatedSamples': [
                        {'video': {'uri': 'https://example.com/v1.mp4'}},
                        {'video': {'uri': 'https://example.com/v2.mp4'}},
                    ]
                }
            },
        })
        assert op.done is True
        assert op.output is not None
        assert op.output['finishReason'] == 'stop'
        content = op.output['message']['content']
        assert len(content) == 2
        assert content[0]['media']['url'] == 'https://example.com/v1.mp4'
        assert content[1]['media']['url'] == 'https://example.com/v2.mp4'

    def test_pydantic_response_with_videos(self) -> None:
        """Pydantic GenerateVideosResponse extracts video URIs (check path).

        This is the regression case — previously this raised AttributeError.
        """
        pydantic_response = genai_types.GenerateVideosResponse(
            generated_videos=[
                genai_types.GeneratedVideo(
                    video=genai_types.Video(
                        uri='https://example.com/video_a.mp4',
                    ),
                ),
                genai_types.GeneratedVideo(
                    video=genai_types.Video(
                        uri='https://example.com/video_b.mp4',
                    ),
                ),
            ],
        )
        op = _from_veo_operation({
            'name': 'models/veo-2.0-generate-001/operations/abc',
            'done': True,
            'response': pydantic_response,
        })
        assert op.done is True
        assert op.output is not None
        assert op.output['finishReason'] == 'stop'
        content = op.output['message']['content']
        assert len(content) == 2
        assert content[0]['media']['url'] == 'https://example.com/video_a.mp4'
        assert content[1]['media']['url'] == 'https://example.com/video_b.mp4'

    def test_pydantic_response_empty_videos(self) -> None:
        """Pydantic response with no generated_videos produces no output."""
        pydantic_response = genai_types.GenerateVideosResponse(
            generated_videos=[],
        )
        op = _from_veo_operation({
            'name': 'operations/empty',
            'done': True,
            'response': pydantic_response,
        })
        assert op.done is True
        assert op.output is None

    def test_response_none_explicit(self) -> None:
        """Explicit None response is handled (no crash)."""
        op = _from_veo_operation({
            'name': 'operations/null',
            'done': False,
            'response': None,
        })
        assert op.output is None

    def test_dict_response_no_videos(self) -> None:
        """Dict response with empty generatedSamples produces no output."""
        op = _from_veo_operation({
            'name': 'operations/empty-dict',
            'done': True,
            'response': {'generateVideoResponse': {'generatedSamples': []}},
        })
        assert op.done is True
        assert op.output is None
