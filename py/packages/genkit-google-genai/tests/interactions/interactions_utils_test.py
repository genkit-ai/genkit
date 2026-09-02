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

"""Tests for Interactions shared helpers."""

from __future__ import annotations

import pytest
from genkit_google_genai.models.interactions_utils import (
    extract_version,
    partition_keys,
    require_interaction_steps,
)

from genkit import GenkitError


def test_extract_version_strips_all_pasted_prefixes() -> None:
    assert extract_version('antigravity-preview-05-2026') == 'antigravity-preview-05-2026'
    assert extract_version('googleai/antigravity-preview-05-2026') == 'antigravity-preview-05-2026'
    assert extract_version('models/googleai/antigravity-preview-05-2026') == 'antigravity-preview-05-2026'


def test_partition_keys_is_non_mutating() -> None:
    payload = {
        'thinking_summaries': 'auto',
        'google_search': True,
        'store': True,
        'extra': 1,
    }
    agent, tools, create, rest = partition_keys(
        payload,
        ('thinking_summaries',),
        ('google_search',),
        ('store', 'response_modalities'),
    )

    assert agent == {'thinking_summaries': 'auto'}
    assert tools == {'google_search': True}
    assert create == {'store': True}
    assert rest == {'extra': 1}
    # Original dump is untouched.
    assert payload == {
        'thinking_summaries': 'auto',
        'google_search': True,
        'store': True,
        'extra': 1,
    }


def test_require_interaction_steps_rejects_empty() -> None:
    with pytest.raises(GenkitError, match='Missing input') as exc_info:
        require_interaction_steps([])
    assert exc_info.value.status == 'INVALID_ARGUMENT'


def test_require_interaction_steps_passes_through() -> None:
    steps = [{'type': 'user_input', 'content': [{'type': 'text', 'text': 'hi'}]}]
    assert require_interaction_steps(steps) is steps
