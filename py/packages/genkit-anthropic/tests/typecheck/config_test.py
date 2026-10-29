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

"""Typed construction of AnthropicConfig and its nested settings.

pyright, pyrefly and ty must accept these snake_case kwargs with no
suppressions. The runtime asserts pin the wire dump.
"""

from genkit_anthropic import (
    AnthropicConfig,
    AutoToolChoice,
    OutputConfig,
    RequestMetadata,
    TaskBudget,
    ThinkingConfig,
)


def test_anthropic_config_snake_case_kwargs() -> None:
    """Snake_case kwargs type-check; inherited fields dump camelCase, Anthropic ones keep their wire names."""
    config = AnthropicConfig(
        temperature=0.5,
        max_output_tokens=2000,
        thinking=ThinkingConfig(enabled=True, budget_tokens=2048),
        output_config=OutputConfig(effort='high', task_budget=TaskBudget(total=20000)),
        tool_choice=AutoToolChoice(type='auto', disable_parallel_tool_use=True),
        metadata=RequestMetadata(user_id='diner-42'),
        api_version='beta',
        betas=['extended-cache-ttl-2025-04-11'],
    )

    assert config.model_dump(by_alias=True, exclude_none=True, mode='json') == {
        'temperature': 0.5,
        'maxOutputTokens': 2000,
        'thinking': {'enabled': True, 'budgetTokens': 2048},
        'output_config': {'effort': 'high', 'task_budget': {'type': 'tokens', 'total': 20000}},
        'tool_choice': {'type': 'auto', 'disable_parallel_tool_use': True},
        'metadata': {'user_id': 'diner-42'},
        'apiVersion': 'beta',
        'betas': ['extended-cache-ttl-2025-04-11'],
    }


def test_thinking_config_accepts_both_spellings() -> None:
    """budget_tokens= and the budgetTokens wire key build the same ThinkingConfig."""
    typed = ThinkingConfig(enabled=True, budget_tokens=2048)

    assert ThinkingConfig.model_validate({'enabled': True, 'budgetTokens': 2048}) == typed
    assert ThinkingConfig.model_validate({'enabled': True, 'budget_tokens': 2048}) == typed
