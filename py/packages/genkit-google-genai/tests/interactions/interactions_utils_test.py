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


def test_api_key_for_context_prefers_tenant_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    from genkit_google_genai.models.interactions_utils import api_key_for_context

    monkeypatch.setenv('GEMINI_API_KEY', 'env-key')
    context = {'secrets': {'api_key': 'tenant-key'}}
    assert api_key_for_context(context, 'plugin-key') == 'tenant-key'


def test_api_key_for_context_falls_back_to_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    from genkit_google_genai.models.interactions_utils import api_key_for_context

    monkeypatch.setenv('GEMINI_API_KEY', 'env-key')
    assert api_key_for_context({}, 'plugin-key') == 'plugin-key'


def test_api_key_for_context_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from genkit_google_genai.models.interactions_utils import api_key_for_context

    monkeypatch.setenv('GEMINI_API_KEY', 'env-key')
    assert api_key_for_context({}, None) == 'env-key'


def test_api_key_for_context_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from genkit_google_genai.models.interactions_utils import api_key_for_context

    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
    monkeypatch.delenv('GOOGLE_GENAI_API_KEY', raising=False)
    with pytest.raises(GenkitError) as exc_info:
        api_key_for_context({}, None)
    assert exc_info.value.status == 'FAILED_PRECONDITION'


def test_client_overrides_from_config_reads_object() -> None:
    from types import SimpleNamespace

    from genkit_google_genai.models.interactions_utils import client_overrides_from_config

    cfg = SimpleNamespace(base_url='https://custom.api', api_version='v1', timeout=5000.0, custom_headers={'h': 'v'})
    opts = client_overrides_from_config(cfg)
    assert opts.base_url == 'https://custom.api'
    assert opts.api_version == 'v1'
    assert opts.timeout == 5000.0
    assert opts.custom_headers == {'h': 'v'}


def test_steps_with_folded_system_instruction_prepends_system() -> None:
    from genkit_google_genai.models.interactions_utils import steps_with_folded_system_instruction

    from genkit import Message, Part, Role

    messages = [
        Message(role=Role.SYSTEM, content=[Part.from_text('Be helpful.')]),
        Message(role=Role.USER, content=[Part.from_text('Hello!')]),
    ]
    steps = steps_with_folded_system_instruction(messages)
    assert len(steps) == 2
    assert steps[0] == {'type': 'user_input', 'content': [{'type': 'text', 'text': 'Be helpful.'}]}
    assert steps[1] == {'type': 'user_input', 'content': [{'type': 'text', 'text': 'Hello!'}]}


def test_model_name_predicates_support_namespaces() -> None:
    from genkit_google_genai.models.interactions_registry import (
        is_antigravity_model_name,
        is_deep_research_model_name,
        is_lyria_model_name,
    )

    # Bare
    assert is_antigravity_model_name('antigravity-preview-05-2026')
    assert is_deep_research_model_name('deep-research-pro-preview-12-2025')
    assert is_lyria_model_name('lyria-3-clip-preview')

    # Namespaced
    assert is_antigravity_model_name('googleai/antigravity-preview-05-2026')
    assert is_deep_research_model_name('googleai/deep-research-pro-preview-12-2025')
    assert is_lyria_model_name('googleai/lyria-3-clip-preview')

    # Full action path
    assert is_antigravity_model_name('models/googleai/antigravity-preview-05-2026')
    assert is_deep_research_model_name('models/googleai/deep-research-pro-preview-12-2025')
    assert is_lyria_model_name('models/googleai/lyria-3-clip-preview')

    # Negative
    assert not is_antigravity_model_name('gemini-2.5-flash')
    assert not is_deep_research_model_name('gemini-2.5-flash')
    assert not is_lyria_model_name('gemini-2.5-flash')
    assert not is_antigravity_model_name(None)
    assert not is_deep_research_model_name(None)
    assert not is_lyria_model_name(None)
