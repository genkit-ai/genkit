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
from unittest.mock import MagicMock

import pytest
from genkit_openai import KnownGpt, OpenAI, OpenAIConfig, openai_model
from genkit_openai.models.model import OpenAIModel
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


def test_gpt_model_keeps_foreign_prefix() -> None:
    """A Vertex or Azure paste is a different name, not remapped onto OpenAI."""
    assert OpenAI.gpt_model('vertexai/gpt-4o').name == 'openai/vertexai/gpt-4o'
    assert OpenAI.gpt_model('azure/gpt-4o').name == 'openai/azure/gpt-4o'


def test_gpt_model_carries_config() -> None:
    """A default config passed at construction survives into the ref."""
    ref = OpenAI.gpt_model('gpt-4o', config=OpenAIConfig(temperature=0.2))
    assert ref.config is not None
    assert ref.config.temperature == 0.2


def test_gpt_model_allows_unknown_chat_ids() -> None:
    """A brand-new chat id works before this plugin learns its name."""
    assert OpenAI.gpt_model('gpt-next-99').name == 'openai/gpt-next-99'


@pytest.mark.parametrize(
    'bad_id',
    [
        'gpt-image-1',  # image endpoint takes a different request shape
        'dall-e-3',
        'tts-1',
        'gpt-4o-mini-tts',  # catalog speech id that looks like a chat model
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


def test_gpt_model_reject_names_the_string_path() -> None:
    """The error names the id and the plain-string way to still use it."""
    with pytest.raises(GenkitError) as exc_info:
        OpenAI.gpt_model('dall-e-3')
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    message = str(exc_info.value)
    assert 'an image model' in message
    assert "openai_model('dall-e-3')" in message


def test_gpt_model_requires_a_name() -> None:
    """A bare prefix with no id left is an invalid argument."""
    with pytest.raises(GenkitError) as exc_info:
        OpenAI.gpt_model('openai/')
    assert exc_info.value.status == 'INVALID_ARGUMENT'


def test_gpt_model_rejects_non_string_name() -> None:
    """A non-string must not become a name via str() (None → 'None')."""
    with pytest.raises(GenkitError) as exc_info:
        OpenAI.gpt_model(None)  # type: ignore[arg-type]
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'must be a string' in str(exc_info.value)

    with pytest.raises(GenkitError) as exc_info:
        OpenAI.gpt_model(123)  # type: ignore[arg-type]
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'must be a string' in str(exc_info.value)


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


@pytest.mark.asyncio
async def test_create_model_action_camel_case_lands_on_the_wire() -> None:
    """Dev UI camelCase binds, then create() gets OpenAI snake_case names."""
    plugin = OpenAI(api_key='test-key')
    action = plugin._create_model_action('openai/gpt-4o')
    validated = action._validate_input(  # noqa: SLF001
        {
            'messages': [{'role': 'user', 'content': [{'text': 'hi'}]}],
            'config': {
                'frequencyPenalty': 0.5,
                'maxOutputTokens': 256,
                'stopSequences': ['END'],
                'topP': 0.9,
                'apiKey': 'should-not-leak',
            },
        }
    )
    assert validated is not None
    body = await OpenAIModel(model='gpt-4o', client=MagicMock())._get_openai_request_config(validated)
    assert body['frequency_penalty'] == 0.5
    assert body['top_p'] == 0.9
    assert body['stop'] == ['END']
    assert 'max_output_tokens' not in body
    assert 'maxOutputTokens' not in body
    assert 'frequencyPenalty' not in body
    assert 'stop_sequences' not in body
    assert 'api_key' not in body
    assert 'apiKey' not in body
