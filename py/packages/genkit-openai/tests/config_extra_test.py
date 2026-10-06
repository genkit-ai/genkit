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

"""`OpenAIConfig` rejects typos; `extra` goes out as `extra_body`."""

import pytest
from genkit_openai.models.audio import _to_tts_params
from genkit_openai.models.image import _to_image_generate_params
from genkit_openai.models.model import _openai_create_kwargs
from genkit_openai.typing import OpenAIConfig
from pydantic import ValidationError

from genkit import GenkitError, Message, Part, Role
from genkit.model import ModelRequest


def test_unknown_key_raises() -> None:
    """`OpenAIConfig(temprature=0.2)` fails instead of riding to the wire."""
    with pytest.raises(ValidationError, match='temprature'):
        OpenAIConfig.model_validate({'temprature': 0.2})


def test_extra_goes_out_as_extra_body_not_a_kwarg() -> None:
    """`extra` becomes `extra_body`; no literal `extra` kwarg reaches create()."""
    kwargs = _openai_create_kwargs(config=OpenAIConfig(temperature=0.2, extra={'enable_search': True}))

    assert kwargs['extra_body'] == {'enable_search': True}
    assert 'extra' not in kwargs
    assert kwargs['temperature'] == 0.2


def test_extra_colliding_key_is_kept_for_sdk_merge() -> None:
    """A key in both places is sent twice; the SDK merges `extra_body` last, so it wins."""
    kwargs = _openai_create_kwargs(config=OpenAIConfig(temperature=0.2, extra={'temperature': 0.9}))

    assert kwargs['temperature'] == 0.2
    assert kwargs['extra_body'] == {'temperature': 0.9}


def test_camel_case_dict_extra_is_read() -> None:
    """A Dev UI dict `{'extra': {...}}` validates into the field."""
    config = OpenAIConfig.model_validate({'maxTokens': 5, 'extra': {'user_tier': 'pro'}})

    assert _openai_create_kwargs(config=config)['extra_body'] == {'user_tier': 'pro'}


@pytest.mark.parametrize('field', ['messages', 'tools', 'tool_choice', 'response_format', 'stream'])
def test_extra_cannot_set_genkit_built_fields(field: str) -> None:
    """Fields Genkit builds from the request are rejected, not overwritten."""
    with pytest.raises(GenkitError) as err:
        _openai_create_kwargs(config=OpenAIConfig(extra={field: []}))

    assert err.value.status == 'INVALID_ARGUMENT'
    assert repr(field) in str(err.value)


def test_no_extra_sends_no_extra_body() -> None:
    """An empty or unset `extra` adds nothing."""
    assert 'extra_body' not in _openai_create_kwargs(config=OpenAIConfig(extra={}))
    assert 'extra_body' not in _openai_create_kwargs(config=OpenAIConfig())


def _text_request(config: dict) -> ModelRequest:
    return ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('a red bicycle')])], config=config)


def test_image_extra_goes_out_as_extra_body() -> None:
    """The images endpoint gets `extra` as `extra_body`, not a literal `extra` kwarg."""
    params = _to_image_generate_params('dall-e-3', _text_request({'size': '1024x1024', 'extra': {'style': 'vivid'}}))

    assert params['extra_body'] == {'style': 'vivid'}
    assert 'extra' not in params


def test_tts_extra_goes_out_as_extra_body() -> None:
    """The speech endpoint gets `extra` as `extra_body` instead of dropping it."""
    params = _to_tts_params('tts-1', _text_request({'extra': {'stream_format': 'sse'}}))

    assert params['extra_body'] == {'stream_format': 'sse'}


def test_image_extra_cannot_set_prompt() -> None:
    """The prompt comes from the messages; `extra` can't replace it."""
    with pytest.raises(GenkitError, match="'prompt'"):
        _to_image_generate_params('dall-e-3', _text_request({'extra': {'prompt': 'something else'}}))
