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
from genkit_anthropic import AnthropicConfig
from pydantic import ValidationError

from genkit import GenkitError
from genkit.plugin_api import to_json_schema

# --- thinking ---------------------------------------------------------------


def test_thinking_enabled_requires_budget() -> None:
    with pytest.raises(ValidationError, match="thinking_budget is required when thinking is 'enabled'"):
        AnthropicConfig.model_validate({'thinking': 'enabled'})


def test_thinking_disabled_rejects_budget() -> None:
    with pytest.raises(ValidationError, match="thinking_budget can't be set when thinking is 'disabled'"):
        AnthropicConfig(thinking='disabled', thinking_budget=2048)


def test_thinking_display_alone_raises() -> None:
    """A display with no mode would send nothing, so it fails instead of being dropped."""
    with pytest.raises(ValidationError, match='thinking_display needs thinking= or thinking_budget='):
        AnthropicConfig(thinking_display='summarized')


def test_thinking_budget_below_minimum_raises() -> None:
    with pytest.raises(ValidationError):
        AnthropicConfig.model_validate({'thinking': 'enabled', 'thinkingBudget': 512})


def test_thinking_budget_must_be_integer() -> None:
    with pytest.raises(ValidationError):
        AnthropicConfig.model_validate({'thinkingBudget': 2048.5})


def test_thinking_mode_typo_raises() -> None:
    """`thinking='adaptiv'` fails before the request is sent."""
    with pytest.raises(ValidationError):
        AnthropicConfig.model_validate({'thinking': 'adaptiv'})


@pytest.mark.parametrize(
    ('raw', 'mode'),
    [
        ({'thinkingBudget': 2048}, 'enabled'),
        ({'thinking': 'enabled', 'thinkingBudget': 2048}, 'enabled'),
        ({'thinking': 'adaptive'}, 'adaptive'),
        # Adaptive accepts and ignores a budget, as before.
        ({'thinking': 'adaptive', 'thinkingBudget': 2048}, 'adaptive'),
        ({'thinking': 'disabled'}, 'disabled'),
        ({}, None),
    ],
)
def test_thinking_mode(raw: dict, mode: str | None) -> None:
    """A budget alone means enabled; an explicit mode wins."""
    assert AnthropicConfig.model_validate(raw).thinking_mode() == mode


def test_thinking_fields_accept_both_spellings() -> None:
    by_alias = AnthropicConfig.model_validate({
        'thinking': 'adaptive',
        'thinkingBudget': 2048,
        'thinkingDisplay': 'omitted',
    })
    by_name = AnthropicConfig(thinking='adaptive', thinking_budget=2048, thinking_display='omitted')
    assert by_alias == by_name


@pytest.mark.parametrize(
    ('raw', 'message'),
    [
        ({'thinking': {'type': 'enabled', 'budget_tokens': 2048}}, 'thinking was flattened'),
        ({'output_config': {'effort': 'high'}}, 'output_config was flattened; use effort= and task_budget='),
        ({'outputConfig': {'effort': 'high'}}, 'outputConfig was flattened; use effort= and task_budget='),
        ({'metadata': {'user_id': 'guest-42'}}, 'metadata was flattened; use user_id='),
    ],
)
def test_nested_keys_name_their_flat_replacement(raw: dict, message: str) -> None:
    """The old nested shape raises a pointed error instead of the generic unknown-key one."""
    with pytest.raises(GenkitError, match=message) as exc_info:
        AnthropicConfig.model_validate(raw)
    assert exc_info.value.status == 'INVALID_ARGUMENT'


# --- effort, task_budget, user_id -------------------------------------------


def test_task_budget_below_minimum_raises() -> None:
    with pytest.raises(ValidationError):
        AnthropicConfig.model_validate({'taskBudget': 10000})


def test_effort_literal_enforced() -> None:
    with pytest.raises(ValidationError):
        AnthropicConfig.model_validate({'effort': 'extreme'})


def test_effort_max_valid_and_advertised() -> None:
    assert AnthropicConfig.model_validate({'effort': 'max'}).effort == 'max'

    schema = to_json_schema(AnthropicConfig)
    assert 'max' in schema['properties']['effort']['enum']


def test_user_id_accepts_both_spellings() -> None:
    assert AnthropicConfig.model_validate({'userId': 'guest-42'}).user_id == 'guest-42'
    assert AnthropicConfig(user_id='guest-42').user_id == 'guest-42'


# --- tool_choice ------------------------------------------------------------


@pytest.mark.parametrize('key', ['tool_choice', 'toolChoice'])
def test_tool_choice_key_points_to_generate_option(key: str) -> None:
    """The removed config key names its replacement instead of the generic unknown-key error."""
    with pytest.raises(GenkitError, match=r'ai\.generate\(tool_choice='):
        AnthropicConfig.model_validate({key: {'type': 'auto'}})


def test_disable_parallel_tool_use_accepts_both_spellings() -> None:
    assert AnthropicConfig.model_validate({'disableParallelToolUse': True}).disable_parallel_tool_use is True
    assert AnthropicConfig(disable_parallel_tool_use=True).disable_parallel_tool_use is True


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


def test_unknown_top_level_key_raises() -> None:
    with pytest.raises(ValidationError, match='foo_bar'):
        AnthropicConfig.model_validate({'temperature': 0.5, 'foo_bar': 'baz'})


def test_extra_survives_validate_dump() -> None:
    cfg = AnthropicConfig.model_validate({'temperature': 0.5, 'extra': {'foo_bar': 'baz'}})
    dumped = cfg.model_dump(exclude_none=True, by_alias=False)
    assert dumped['extra'] == {'foo_bar': 'baz'}


def test_base_max_output_tokens_alias() -> None:
    cfg = AnthropicConfig.model_validate({'maxOutputTokens': 256})
    assert cfg.max_output_tokens == 256


# --- JSON-schema parity (alias-drift guard) ---------------------------------


def test_json_schema_advertises_js_shaped_keys() -> None:
    schema = to_json_schema(AnthropicConfig)
    props = schema['properties']

    # Advertised common and Anthropic-specific keys.
    for key in (
        'apiVersion',
        'betas',
        'maxOutputTokens',
        'disableParallelToolUse',
        'thinking',
        'thinkingBudget',
        'thinkingDisplay',
        'effort',
        'taskBudget',
        'userId',
    ):
        assert key in props, f'missing advertised key {key!r}'

    assert props['maxOutputTokens']['type'] == 'number'
    assert props['maxOutputTokens']['title'] == 'Max output tokens'
    assert props['apiVersion']['description'] == 'Selects the Anthropic API surface for this request.'
    assert (
        props['betas']['description']
        == 'Anthropic beta feature headers to enable for this request. An empty list suppresses the defaults.'
    )
    assert props['disableParallelToolUse']['type'] == 'boolean'

    # Tool choice is the generate option, not a config key.
    assert 'tool_choice' not in props
    assert 'toolChoice' not in props
    # The nested provider objects are built by the plugin, not advertised.
    for key in ('output_config', 'outputConfig', 'metadata'):
        assert key not in props
    assert props['thinking']['enum'] == ['enabled', 'adaptive', 'disabled']
    assert props['thinkingDisplay']['enum'] == ['summarized', 'omitted']
    assert props['effort']['enum'] == ['low', 'medium', 'high', 'xhigh', 'max']
    text = str(schema)
    assert '$defs' not in schema
    assert '$ref' not in text


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        ({'extra': {'speed': 'fast'}}, {'speed'}),
        ({'betas': ['x']}, {'betas'}),
        # Setting a beta-only feature at all is intent, even when the value is empty.
        ({'extra': {'mcp_servers': []}}, {'mcp_servers'}),
        # An empty betas list requests no beta headers, so it does not select the surface.
        ({'betas': []}, set()),
        ({'taskBudget': 20000}, {'task_budget'}),
        ({'effort': 'high'}, set()),
        ({'temperature': 0.5}, set()),
        ({'extra': {'future_option': 'x'}}, set()),
    ],
)
def test_beta_only_fields_detection(raw: dict, expected: set[str]) -> None:
    """Only beta-only request fields select the beta surface."""
    assert AnthropicConfig.model_validate(raw).beta_only_fields() == expected


@pytest.mark.parametrize(
    'raw',
    [
        {'apiVersion': 'stable', 'betas': ['x']},
        {'apiVersion': 'stable', 'extra': {'speed': 'fast'}},
        {'apiVersion': 'stable', 'taskBudget': 20000},
    ],
)
def test_beta_only_fields_rejected_on_stable_surface(raw: dict) -> None:
    """An explicit stable apiVersion is never silently overridden."""
    with pytest.raises(ValidationError, match='require the beta API surface'):
        AnthropicConfig.model_validate(raw)


@pytest.mark.parametrize(
    'raw',
    [
        {'apiVersion': 'beta', 'extra': {'speed': 'fast'}},
        {'extra': {'speed': 'fast'}},
        {'apiVersion': 'stable', 'temperature': 0.5},
    ],
)
def test_beta_only_fields_allowed_without_explicit_stable(raw: dict) -> None:
    """Beta-only fields are accepted unless stable is explicitly requested."""
    assert AnthropicConfig.model_validate(raw) is not None


def test_dev_ui_schema_lists_every_key_the_config_accepts() -> None:
    """With `additionalProperties: false`, the Dev UI form rejects any key the schema leaves out."""
    props = to_json_schema(AnthropicConfig)['properties']
    accepted = {field.alias or name for name, field in AnthropicConfig.model_fields.items()}
    assert accepted <= set(props)
