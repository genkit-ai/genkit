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

from types import SimpleNamespace

import pytest
from genkit_google_genai.models.interactions_utils import (
    map_genai_error,
    partition_keys,
    require_interaction_steps,
)
from google.genai.errors import APIError

from genkit import GenkitError


def test_map_genai_error_maps_rate_limit_and_retry_after() -> None:
    response = SimpleNamespace(headers={'retry-after': '1.5'})
    error = APIError(429, {'error': {'message': 'slow down', 'status': 'RESOURCE_EXHAUSTED'}}, response=response)

    mapped = map_genai_error(error)

    assert isinstance(mapped, GenkitError)
    assert mapped.status == 'RESOURCE_EXHAUSTED'
    assert mapped.original_message == 'slow down'
    assert mapped.response_metadata is not None
    assert mapped.response_metadata.get('retry_after_ms') == 1500.0


def test_map_genai_error_maps_unauthenticated() -> None:
    error = APIError(401, {'error': {'message': 'bad key'}})
    mapped = map_genai_error(error)
    assert mapped.status == 'UNAUTHENTICATED'


def test_map_genai_error_maps_gaos_status_code() -> None:
    """Interactions gaos errors aren't google.genai.errors.APIError but carry status_code."""
    error = SimpleNamespace(status_code=400, message='Missing input.')
    mapped = map_genai_error(error)
    assert mapped.status == 'INVALID_ARGUMENT'
    assert mapped.original_message == 'Missing input.'


def test_map_genai_error_tolerates_non_numeric_code() -> None:
    """APIError.code comes from response JSON, so it may be a non-numeric string."""
    error = APIError('boom', {'error': {'message': 'weird proxy error', 'code': 'boom'}})
    mapped = map_genai_error(error)
    assert mapped.status == 'UNKNOWN'


def test_map_genai_error_maps_gaos_not_found() -> None:
    error = SimpleNamespace(status_code=404, message="Did you mean 'lyria-3-pro-preview'?")
    mapped = map_genai_error(error)
    assert mapped.status == 'NOT_FOUND'


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
