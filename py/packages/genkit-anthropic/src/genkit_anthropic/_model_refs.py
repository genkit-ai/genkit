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

"""Typed model references for Anthropic Claude models."""

from typing import Literal

from genkit.model import ModelRef, model_ref
from genkit_anthropic.config import AnthropicConfig

KnownClaude = Literal[
    'claude-sonnet-4',
    'claude-opus-4',
    'claude-sonnet-4-5',
    'claude-sonnet-4-6',
    'claude-sonnet-5',
    'claude-haiku-4-5',
    'claude-opus-4-1',
    'claude-opus-4-5',
    'claude-opus-4-6',
    'claude-opus-4-7',
    'claude-opus-4-8',
    'claude-fable-5',
]


def claude_model(
    name: KnownClaude | str,
    *,
    config: AnthropicConfig | None = None,
) -> ModelRef[AnthropicConfig]:
    """Return a typed reference to a Claude model."""
    return model_ref(
        name,
        config_schema=AnthropicConfig,
        namespace='anthropic',
        config=config,
    )
