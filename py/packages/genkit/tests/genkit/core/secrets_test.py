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

"""Rules every model plugin applies to a per-request key on context.secrets."""

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from genkit import GenkitError
from genkit.model import ModelConfig
from genkit.plugin_api import SECRETS_HINT, context_api_key, reject_config_api_key


@pytest.mark.parametrize(
    'context,expected',
    [
        pytest.param({'secrets': {'api_key': 'sk-tenant'}}, 'sk-tenant', id='api_key'),
        pytest.param({'secrets': {'apiKey': 'sk-tenant'}}, 'sk-tenant', id='js-spelling'),
        pytest.param({'secrets': {'api_key': 'sk-a', 'apiKey': 'sk-b'}}, 'sk-a', id='api_key-wins'),
        pytest.param({'secrets': {'api_key': '  sk-tenant\n'}}, 'sk-tenant', id='trimmed'),
        pytest.param({'secrets': {'db_password': 'x'}}, None, id='other-secrets-only'),
        pytest.param({'secrets': {'api_key': None}}, None, id='null-key'),
        pytest.param({'secrets': {}}, None, id='empty-secrets'),
        pytest.param({'secrets': None}, None, id='null-secrets'),
        pytest.param({'api_key': 'app-key', 'apiKey': 'app-key'}, None, id='top-level-key-ignored'),
        pytest.param({}, None, id='empty-context'),
        pytest.param(None, None, id='no-context'),
    ],
)
def test_context_api_key_reads_only_context_secrets(context: dict[str, Any] | None, expected: str | None) -> None:
    """The key comes from context.secrets; anything else runs on the plugin's key."""
    assert context_api_key(context) == expected


@pytest.mark.parametrize(
    'context,reason',
    [
        pytest.param({'secrets': 'sk-tenant'}, 'context.secrets must be a dict', id='secrets-not-a-dict'),
        pytest.param({'secrets': {'api_key': 42}}, 'context.secrets.api_key must be a string', id='not-a-string'),
        pytest.param({'secrets': {'apiKey': '   '}}, 'context.secrets.apiKey is blank', id='blank'),
        pytest.param({'secrets': {'api_key': ''}}, 'context.secrets.api_key is blank', id='empty-string'),
        pytest.param({'secrets': {'api_key': 'sk-ten ant'}}, 'invalid whitespace', id='inner-space'),
        pytest.param({'secrets': {'api_key': 'sk-ten\0ant'}}, 'control characters', id='nul'),
        pytest.param({'config': {'api_key': 'sk-tenant'}}, 'not config', id='poll-config-key'),
    ],
)
def test_context_api_key_unusable_key_raises_invalid_argument(context: dict[str, Any], reason: str) -> None:
    """A key that is set but unusable raises instead of falling back to the plugin's key."""
    with pytest.raises(GenkitError) as raised:
        context_api_key(context)

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert reason in str(raised.value)
    assert SECRETS_HINT in str(raised.value)


class _LooseConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

    extra: dict[str, Any] | None = None


@pytest.mark.parametrize(
    'config',
    [
        pytest.param({'api_key': 'sk-tenant'}, id='dict'),
        pytest.param({'apiKey': 'sk-tenant'}, id='dict-js-spelling'),
        pytest.param({'extra': {'api_key': 'sk-tenant'}}, id='dict-extra'),
        pytest.param(ModelConfig(api_key='sk-tenant'), id='model-field'),
        pytest.param(ModelConfig(extra={'apiKey': 'sk-tenant'}), id='model-extra-field'),
        pytest.param(ModelConfig.model_validate({'apiKey': 'sk-tenant'}), id='model-js-spelling'),
        pytest.param(_LooseConfig.model_validate({'api_key': 'sk-tenant'}), id='undeclared-field'),
        pytest.param(_LooseConfig(extra={'api_key': 'sk-tenant'}), id='non-modelconfig-extra'),
    ],
)
def test_reject_config_api_key_raises_on_key_in_config_or_extra(config: object) -> None:
    """A key on config, its undeclared fields, or inside config.extra raises naming context.secrets."""
    with pytest.raises(GenkitError) as raised:
        reject_config_api_key(config)

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert SECRETS_HINT in str(raised.value)
    assert 'sk-tenant' not in str(raised.value)


@pytest.mark.parametrize(
    'config',
    [
        pytest.param(None, id='none'),
        pytest.param({'temperature': 0.2, 'extra': {'reasoning': {'effort': 'low'}}}, id='dict'),
        pytest.param(ModelConfig(temperature=0.2), id='model'),
        pytest.param({'extra': None}, id='null-extra'),
    ],
)
def test_reject_config_api_key_passes_config_without_key(config: object) -> None:
    """Config without a key passes."""
    reject_config_api_key(config)
