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

"""Tests for Veo video generation model helpers and background actions."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genkit_google_genai import GoogleAI
from genkit_google_genai.google import GenaiModels
from genkit_google_genai.models.veo import (
    VeoConfig,
    VeoVersion,
    from_veo_operation,
    is_veo_model,
    to_veo_parameters,
)
from google.genai import types as genai_types

from genkit import FinishReason, Genkit, ModelResponse


def veo_operation(**fields: Any) -> genai_types.GenerateVideosOperation:
    """Build an SDK video operation the way the API delivers one."""
    return genai_types.GenerateVideosOperation.model_validate(fields)


class TestIsVeoModel:
    """Tests for is_veo_model."""

    def test_veo_model_name(self) -> None:
        """Veo model names are recognized."""
        assert is_veo_model('veo-2.0-generate-001') is True

    def test_veo_uppercase(self) -> None:
        """Case-insensitive matching works."""
        assert is_veo_model('VEO-2.0-generate-001') is True

    def test_non_veo_model_name(self) -> None:
        """Non-Veo models return False."""
        assert is_veo_model('gemini-2.0-flash') is False
        assert is_veo_model('imagen-3.0-generate-001') is False


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


class TestToVeoParameters:
    """Tests for to_veo_parameters."""

    def test_empty_config(self) -> None:
        """Empty VeoConfig becomes an empty GenerateVideosConfig."""
        result = to_veo_parameters(VeoConfig())
        assert isinstance(result, genai_types.GenerateVideosConfig)
        assert result.aspect_ratio is None

    def test_schema_config(self) -> None:
        """VeoConfig maps onto GenerateVideosConfig fields."""
        config = VeoConfig(aspect_ratio='16:9', duration_seconds=5)
        result = to_veo_parameters(config)
        assert result.aspect_ratio == '16:9'
        assert result.duration_seconds == 5

    def test_schema_config_includes_new_fields(self) -> None:
        """VeoConfig includes newer Veo parameters."""
        config = VeoConfig(resolution='1080p', seed=7)
        result = to_veo_parameters(config)
        assert result.resolution == '1080p'
        assert result.seed == 7


class TestFromVeoOperation:
    """Tests for from_veo_operation with typed GenerateVideosOperation."""

    def test_pending_operation(self) -> None:
        """An in-progress operation has no response — output stays None."""
        op = from_veo_operation(veo_operation(name='operations/123', done=False))
        assert op.id == 'operations/123'
        assert op.done is False
        assert op.output is None
        assert op.error is None

    def test_pending_operation_null_done_normalized(self) -> None:
        """API null/omitted done is pending, not a missing flag."""
        for sdk_op in (
            veo_operation(name='operations/null-done', done=None),
            veo_operation(name='operations/omitted-done'),
        ):
            op = from_veo_operation(sdk_op)
            assert op.done is False
            assert op.output is None

    def test_error_operation(self) -> None:
        """An operation with an error populates op.error."""
        op = from_veo_operation(
            veo_operation(
                name='operations/456',
                done=True,
                error={'message': 'Quota exceeded'},
            )
        )
        assert op.id == 'operations/456'
        assert op.done is True
        assert op.error is not None
        assert op.error.message == 'Quota exceeded'
        assert op.output is None

    def test_pydantic_response_with_videos(self) -> None:
        """GenerateVideosResponse extracts video URIs (check path)."""
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
        op = from_veo_operation(
            veo_operation(
                name='models/veo-2.0-generate-001/operations/abc',
                done=True,
                response=pydantic_response,
            )
        )
        assert op.done is True
        assert isinstance(op.output, ModelResponse)
        assert op.output.finish_reason == FinishReason.STOP
        content = op.output.message.content if op.output.message else []
        assert len(content) == 2
        media0 = content[0].root.media
        media1 = content[1].root.media
        assert media0 is not None and media0.url == 'https://example.com/video_a.mp4'
        assert media1 is not None and media1.url == 'https://example.com/video_b.mp4'

    def test_pydantic_response_empty_videos(self) -> None:
        """Response with no generated_videos produces no output."""
        op = from_veo_operation(
            veo_operation(
                name='operations/empty',
                done=True,
                response=genai_types.GenerateVideosResponse(generated_videos=[]),
            )
        )
        assert op.done is True
        assert op.output is None

    def test_response_none_explicit(self) -> None:
        """Explicit None response is handled (no crash)."""
        op = from_veo_operation(
            veo_operation(
                name='operations/null',
                done=False,
                response=None,
            )
        )
        assert op.output is None


def _mock_veo_client(start_done: bool = False) -> MagicMock:
    """Build a mocked GenAI client for Veo background-model tests."""
    client = MagicMock()
    start_op = veo_operation(
        name='operations/veo-start',
        done=start_done,
    )
    completed_response = genai_types.GenerateVideosResponse(
        generated_videos=[
            genai_types.GeneratedVideo(
                video=genai_types.Video(uri='https://example.com/generated.mp4'),
            ),
        ],
    )
    check_op = veo_operation(
        name='operations/veo-start',
        done=True,
        response=completed_response,
    )

    client.aio.models.generate_videos = AsyncMock(return_value=start_op)
    client.aio.operations.get = AsyncMock(return_value=check_op)
    return client


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_veo_generate_returns_operation(mock_list_models: MagicMock, mock_client_ctor: MagicMock) -> None:
    """generate() on a Veo model returns an Operation to poll."""
    models = GenaiModels()
    models.veo = ['veo-2.0-generate-001']
    mock_list_models.return_value = models
    mock_client_ctor.return_value = _mock_veo_client()

    ai = Genkit(plugins=[GoogleAI(api_key='test-key')])
    response = await ai.generate(
        model='googleai/veo-2.0-generate-001',
        prompt='a cat surfing',
    )

    assert response.operation is not None
    assert response.operation.id == 'operations/veo-start'
    assert response.operation.done is False
    assert response.operation.action == '/background-model/googleai/veo-2.0-generate-001'
    assert response.message is None


@patch('genkit_google_genai.google.genai.client.Client')
@patch('genkit_google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_veo_generate_operation_poll_loop(mock_list_models: MagicMock, mock_client_ctor: MagicMock) -> None:
    """generate_operation + check_operation poll Veo to a ModelResponse output."""
    models = GenaiModels()
    models.veo = ['veo-2.0-generate-001']
    mock_list_models.return_value = models
    mock_client_ctor.return_value = _mock_veo_client()

    ai = Genkit(plugins=[GoogleAI(api_key='test-key')])
    operation = await ai.generate_operation(
        model='googleai/veo-2.0-generate-001',
        prompt='a cat surfing',
    )

    assert operation.id == 'operations/veo-start'
    assert operation.done is False

    operation = await ai.check_operation(operation)

    assert operation.done is True
    assert isinstance(operation.output, ModelResponse)
    assert operation.output.finish_reason == FinishReason.STOP
    content = operation.output.message.content if operation.output.message else []
    assert len(content) == 1
    media = content[0].root.media
    assert media is not None and media.url == 'https://example.com/generated.mp4'
