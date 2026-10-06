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

"""`AnthropicConfig` rejects typos; `extra` is merged into `extra_body` last."""

from unittest.mock import MagicMock

import pytest
from genkit_anthropic.config import BETA_ONLY_KEYS, AnthropicConfig
from genkit_anthropic.models import AnthropicModel
from pydantic import ValidationError

from genkit import GenkitError, Message, Part, Role
from genkit.model import ModelRequest


def _request(config: AnthropicConfig) -> ModelRequest:
    return ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('hi')])], config=config)


def _params(config: AnthropicConfig) -> dict:
    model = AnthropicModel(model_name='claude-sonnet-4-5', client=MagicMock())
    return model._build_params(_request(config))


def test_unknown_key_raises() -> None:
    """`{'temprature': 0.2}` fails instead of riding to the wire."""
    with pytest.raises(ValidationError, match='temprature'):
        AnthropicConfig.model_validate({'temprature': 0.2})


def test_extra_lands_in_extra_body_not_as_a_kwarg() -> None:
    """`extra={'service_tier': 'auto'}` goes through `extra_body`; no literal `extra` kwarg."""
    params = _params(AnthropicConfig(extra={'service_tier': 'auto'}))

    assert params['extra_body'] == {'service_tier': 'auto'}
    assert 'extra' not in params


def test_extra_colliding_key_wins_through_extra_body() -> None:
    """A key in both the declared field and `extra` is sent from `extra`, since the SDK merges it last."""
    params = _params(AnthropicConfig(temperature=0.2, extra={'temperature': 0.9}))

    assert params['temperature'] == 0.2
    assert params['extra_body']['temperature'] == 0.9


@pytest.mark.parametrize('field', ['model', 'messages', 'system', 'tools', 'tool_choice', 'stream', 'output_config'])
def test_extra_cannot_set_genkit_built_fields(field: str) -> None:
    """Fields Genkit builds from the request are rejected, not overwritten."""
    with pytest.raises(GenkitError) as err:
        _params(AnthropicConfig(extra={field: []}))

    assert err.value.status == 'INVALID_ARGUMENT'
    assert repr(field) in str(err.value)


def test_beta_only_key_in_extra_selects_beta_surface() -> None:
    """A beta-only body field in `extra` still routes to the beta API."""
    beta_key = sorted(BETA_ONLY_KEYS)[0]
    config = AnthropicConfig(extra={beta_key: {}})

    assert config.beta_only_fields() == {beta_key}


def test_beta_only_key_in_extra_rejected_on_stable_surface() -> None:
    """`apiVersion='stable'` plus a beta-only key in `extra` fails at validation."""
    beta_key = sorted(BETA_ONLY_KEYS)[0]

    with pytest.raises(ValidationError, match='beta'):
        AnthropicConfig(api_version='stable', extra={beta_key: {}})
