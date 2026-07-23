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

"""Tests for claude_model family helper."""

from genkit_anthropic import AnthropicConfig, claude_model


def test_claude_model_stamps_namespace_and_schema() -> None:
    ref = claude_model('claude-sonnet-4-5')
    assert ref.name == 'anthropic/claude-sonnet-4-5'
    assert ref.config_schema is AnthropicConfig


def test_claude_model_unknown_version_still_typed_schema() -> None:
    ref = claude_model('claude-future-99')
    assert ref.name == 'anthropic/claude-future-99'
    assert ref.config_schema is AnthropicConfig


def test_claude_model_idempotent_namespace() -> None:
    ref = claude_model('anthropic/claude-sonnet-4-5')
    assert ref.name == 'anthropic/claude-sonnet-4-5'


def test_claude_model_stamps_default_config() -> None:
    config = AnthropicConfig(temperature=0.7)
    ref = claude_model('claude-haiku-4-5', config=config)
    assert ref.config is config
    assert ref.config is not None
    assert ref.config.temperature == 0.7
