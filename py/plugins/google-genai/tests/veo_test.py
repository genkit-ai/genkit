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

from google.genai import types as genai_types

from genkit.plugins.google_genai.models.veo import (
    VeoConfigSchema,
    _from_veo_operation,
    _to_veo_parameters,
    is_veo_model,
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


class TestToVeoParameters:
    """Tests for _to_veo_parameters."""

    def test_none_config(self) -> None:
        """None config returns empty dict."""
        assert _to_veo_parameters(None) == {}

    def test_dict_config(self) -> None:
        """Dict config filters out None values."""
        config = {'aspect_ratio': '16:9', 'duration_seconds': 5, 'empty': None}
        result = _to_veo_parameters(config)
        assert result == {'aspect_ratio': '16:9', 'duration_seconds': 5}

    def test_schema_config(self) -> None:
        """VeoConfigSchema is converted with camelCase keys."""
        config = VeoConfigSchema(aspect_ratio='16:9', duration_seconds=5)
        result = _to_veo_parameters(config)
        assert result['aspectRatio'] == '16:9'
        assert result['durationSeconds'] == 5


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
