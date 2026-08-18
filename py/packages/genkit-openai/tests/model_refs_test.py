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

"""Tests for the OpenAI.gpt_model typed ref constructor."""

from typing import get_args, get_type_hints

import pytest
from genkit_openai import KnownGpt, OpenAI, OpenAIConfig, openai_model
from genkit_openai.models.model_info import SUPPORTED_OPENAI_MODELS

from genkit import GenkitError
from genkit.model import ModelRef


def test_openai_model_still_returns_str() -> None:
    """openai_model stays a string helper, not a ModelRef."""
    assert openai_model('gpt-4o') == 'openai/gpt-4o'
    assert isinstance(openai_model('gpt-4o'), str)


def test_gpt_model_returns_model_ref() -> None:
    """gpt_model namespaces the name and binds OpenAIConfig."""
    ref = OpenAI.gpt_model('gpt-4o')

    assert isinstance(ref, ModelRef)
    assert ref.name == 'openai/gpt-4o'
    assert ref.config_schema is OpenAIConfig


def test_gpt_model_strips_own_prefix() -> None:
    """An already-namespaced name is not double-prefixed."""
    assert OpenAI.gpt_model('openai/gpt-4o').name == 'openai/gpt-4o'


@pytest.mark.parametrize(
    'bad_id',
    [
        'gpt-image-1',  # image endpoint takes a different request shape
        'dall-e-3',
        'tts-1',
        'whisper-1',
        'gpt-4o-transcribe',
        'text-embedding-3-small',  # embedders never take chat config
    ],
)
def test_gpt_model_is_chat_only(bad_id: str) -> None:
    """Non-chat ids are refused so chat-only keys can't hit the wrong endpoint."""
    with pytest.raises(GenkitError) as exc_info:
        OpenAI.gpt_model(bad_id)
    assert exc_info.value.status == 'INVALID_ARGUMENT'


def test_gpt_model_requires_a_name() -> None:
    """A bare prefix with no id left is an invalid argument."""
    with pytest.raises(GenkitError) as exc_info:
        OpenAI.gpt_model('openai/')
    assert exc_info.value.status == 'INVALID_ARGUMENT'


def test_known_gpt_matches_catalog() -> None:
    """Quote autocomplete and the chat catalog are the same set of ids."""
    assert set(get_args(KnownGpt)) == set(SUPPORTED_OPENAI_MODELS)


def test_create_model_action_types_openai_config() -> None:
    """Chat actions opt into ModelRequest[OpenAIConfig]."""
    plugin = OpenAI(api_key='test-key')
    action = plugin._create_model_action('openai/gpt-4o')

    hints = get_type_hints(action._fn)  # noqa: SLF001
    request_type = hints['request']
    args = get_args(request_type) or (getattr(request_type, '__pydantic_generic_metadata__', {}) or {}).get('args')
    assert args and args[0] is OpenAIConfig
