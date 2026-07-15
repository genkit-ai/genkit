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

"""Tests for the typed Anthropic config schema."""

import pytest
from genkit_anthropic.config import AnthropicConfig, ThinkingConfig
from pydantic import ValidationError

from genkit.plugin_api import to_json_schema

# --- thinking ---------------------------------------------------------------


def test_thinking_enabled_requires_budget() -> None:
    with pytest.raises(ValidationError):
        ThinkingConfig.model_validate({'enabled': True})


def test_thinking_sdk_native_enabled_requires_budget() -> None:
    with pytest.raises(ValidationError):
        ThinkingConfig.model_validate({'type': 'enabled'})


def test_thinking_enabled_and_adaptive_mutually_exclusive() -> None:
    with pytest.raises(ValidationError):
        ThinkingConfig.model_validate({'enabled': True, 'budgetTokens': 2048, 'adaptive': True})


def test_thinking_budget_below_minimum_raises() -> None:
    with pytest.raises(ValidationError):
        ThinkingConfig.model_validate({'enabled': True, 'budgetTokens': 512})


def test_thinking_enabled_budget_tokens_must_be_integer() -> None:
    with pytest.raises(ValidationError):
        ThinkingConfig.model_validate({'enabled': True, 'budgetTokens': 2048.5})


def test_thinking_budget_only_must_be_integer() -> None:
    with pytest.raises(ValidationError):
        ThinkingConfig.model_validate({'budgetTokens': 2048.5})


def test_thinking_budget_tokens_alias_accepted() -> None:
    cfg = ThinkingConfig.model_validate({'enabled': True, 'budgetTokens': 2048})
    assert cfg.budget_tokens == 2048


def test_thinking_adaptive_with_display_valid() -> None:
    cfg = ThinkingConfig.model_validate({'adaptive': True, 'display': 'summarized'})
    assert cfg.adaptive is True
    assert cfg.display == 'summarized'


def test_thinking_adaptive_allows_fractional_ignored_budget() -> None:
    cfg = ThinkingConfig.model_validate({'adaptive': True, 'budgetTokens': 2048.5})
    assert cfg.budget_tokens == 2048.5


# --- output_config ----------------------------------------------------------


def test_output_config_task_budget_below_minimum_raises() -> None:
    with pytest.raises(ValidationError):
        AnthropicConfig.model_validate({'output_config': {'task_budget': {'total': 10000}}})


def test_output_config_task_budget_type_defaults_to_tokens() -> None:
    cfg = AnthropicConfig.model_validate({'output_config': {'task_budget': {'total': 20000}}})
    assert cfg.output_config is not None
    assert cfg.output_config.task_budget is not None
    assert cfg.output_config.task_budget.type == 'tokens'


def test_output_config_effort_literal_enforced() -> None:
    with pytest.raises(ValidationError):
        AnthropicConfig.model_validate({'output_config': {'effort': 'extreme'}})


def test_output_config_effort_max_valid_and_advertised() -> None:
    cfg = AnthropicConfig.model_validate({'output_config': {'effort': 'max'}})
    assert cfg.output_config is not None
    assert cfg.output_config.effort == 'max'

    schema = to_json_schema(AnthropicConfig)
    assert 'max' in schema['properties']['output_config']['properties']['effort']['enum']


# --- tool_choice ------------------------------------------------------------


def test_tool_choice_tool_requires_name() -> None:
    with pytest.raises(ValidationError):
        AnthropicConfig.model_validate({'tool_choice': {'type': 'tool'}})


@pytest.mark.parametrize(
    'tool_choice',
    [
        {'type': 'auto'},
        {'type': 'any'},
        {'type': 'tool', 'name': 'get_weather'},
        {'type': 'none'},
    ],
)
def test_tool_choice_variants_valid(tool_choice: dict) -> None:
    cfg = AnthropicConfig.model_validate({'tool_choice': tool_choice})
    assert cfg.tool_choice is not None
    assert cfg.tool_choice.type == tool_choice['type']


# --- top level --------------------------------------------------------------


def test_api_version_literal_and_alias() -> None:
    cfg = AnthropicConfig.model_validate({'apiVersion': 'beta'})
    assert cfg.api_version == 'beta'
    with pytest.raises(ValidationError):
        AnthropicConfig.model_validate({'apiVersion': 'nightly'})


def test_stable_api_version_with_betas_raises() -> None:
    with pytest.raises(ValidationError):
        AnthropicConfig.model_validate({'apiVersion': 'stable', 'betas': ['token-efficient-tools-2025']})


def test_beta_api_version_with_betas_valid() -> None:
    cfg = AnthropicConfig.model_validate({'apiVersion': 'beta', 'betas': ['token-efficient-tools-2025']})
    assert cfg.betas == ['token-efficient-tools-2025']


def test_unknown_extras_survive_validate_dump() -> None:
    cfg = AnthropicConfig.model_validate({'temperature': 0.5, 'foo_bar': 'baz'})
    dumped = cfg.model_dump(exclude_none=True, by_alias=False)
    assert dumped['foo_bar'] == 'baz'


def test_base_max_output_tokens_alias() -> None:
    cfg = AnthropicConfig.model_validate({'maxOutputTokens': 256})
    assert cfg.max_output_tokens == 256


# --- JSON-schema parity (alias-drift guard) ---------------------------------


def test_json_schema_advertises_js_shaped_keys() -> None:
    schema = to_json_schema(AnthropicConfig)
    props = schema['properties']

    # Advertised common and Anthropic-specific keys.
    for key in (
        'apiKey',
        'apiVersion',
        'betas',
        'maxOutputTokens',
        'tool_choice',
        'metadata',
        'thinking',
        'output_config',
    ):
        assert key in props, f'missing advertised key {key!r}'

    assert props['maxOutputTokens']['type'] == 'number'
    assert props['maxOutputTokens']['title'] == 'Max output tokens'
    assert props['apiKey']['description'] == 'Overrides the plugin-configured Anthropic API key for this request.'
    assert props['apiVersion']['description'] == 'Selects the Anthropic API surface for this request.'
    assert props['betas']['description'] == 'Anthropic beta feature headers to enable for this request.'
    assert props['tool_choice']['type'] == 'object'
    assert props['tool_choice']['properties']['type']['enum'] == ['auto', 'any', 'tool', 'none']
    assert 'oneOf' not in props['tool_choice']

    # snake_case keys must NOT drift to camelCase.
    assert 'toolChoice' not in props
    assert 'outputConfig' not in props

    # Nested snake_case/camelCase keys are preserved.
    text = str(schema)
    assert 'budgetTokens' in text  # thinking.budgetTokens (camelCase)
    assert 'task_budget' in text  # output_config.task_budget (snake_case)
    assert 'user_id' in text  # metadata.user_id (snake_case)
    assert '$defs' not in schema
    assert '$ref' not in text
