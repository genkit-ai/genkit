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

"""Unit tests for Gemini finish-reason mapping and empty-image messaging."""

from types import SimpleNamespace

import pytest
from genkit_google_genai.models.gemini import (
    SUPPORTED_MODELS,
    _finish_message_for_image_response,
    _to_finish_reason,
    is_image_model,
)

from genkit import FinishReason, Media, MediaPart, Part, TextPart


@pytest.mark.parametrize(
    ('reason', 'expected'),
    [
        ('STOP', FinishReason.STOP),
        ('MAX_TOKENS', FinishReason.LENGTH),
        ('SAFETY', FinishReason.BLOCKED),
        ('IMAGE_SAFETY', FinishReason.BLOCKED),
        ('NO_IMAGE', FinishReason.OTHER),
        ('IMAGE_OTHER', FinishReason.OTHER),
        ('OTHER', FinishReason.OTHER),
        ('TOTALLY_MADE_UP', FinishReason.UNKNOWN),
        (None, FinishReason.UNKNOWN),
        (SimpleNamespace(name='no_image'), FinishReason.OTHER),
    ],
)
def test_to_finish_reason(reason: object, expected: FinishReason) -> None:
    assert _to_finish_reason(reason) == expected


def test_finish_message_for_missing_image() -> None:
    empty = [Part(root=TextPart(text=''))]
    msg = _finish_message_for_image_response(fr_name='NO_IMAGE', content=empty)
    assert msg is not None
    assert 'No image was returned' in msg

    with_media = [
        Part(root=MediaPart(media=Media(url='data:image/png;base64,xx', content_type='image/png'))),
    ]
    assert _finish_message_for_image_response(fr_name='STOP', content=with_media) is None


def test_stale_flash_image_preview_not_in_supported_models() -> None:
    assert 'gemini-2.5-flash-image-preview' not in SUPPORTED_MODELS
    assert is_image_model('gemini-2.5-flash-image-preview')
