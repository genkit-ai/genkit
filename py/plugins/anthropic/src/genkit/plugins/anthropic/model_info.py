# Copyright 2025 Google LLC
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

"""Anthropic Models for Genkit."""

from genkit.types import (
    Constrained,
    ModelInfo,
    Supports,
)

# Model definitions
CLAUDE_3_HAIKU = ModelInfo(
    label='Anthropic - Claude 3 Haiku',
    versions=['claude-3-haiku-20240307'],
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        system_role=True,
    ),
)

CLAUDE_3_5_HAIKU = ModelInfo(
    label='Anthropic - Claude 3.5 Haiku',
    versions=['claude-3-5-haiku-20241022'],
    supports=Supports(
        multiturn=True,
        media=False,
        tools=True,
        system_role=True,
    ),
)


CLAUDE_3_5_SONNET = ModelInfo(
    label='Anthropic - Claude 3.5 Sonnet',
    versions=['claude-3-5-sonnet-20241022', 'claude-3-5-sonnet-20240620'],
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        system_role=True,
        output=['text', 'json'],
    ),
)

CLAUDE_SONNET_4 = ModelInfo(
    label='Anthropic - Claude Sonnet 4',
    versions=['claude-sonnet-4-20250514'],
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        system_role=True,
    ),
)

CLAUDE_OPUS_4 = ModelInfo(
    label='Anthropic - Claude Opus 4',
    versions=['claude-opus-4-20250514'],
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        system_role=True,
    ),
)

CLAUDE_SONNET_4_5 = ModelInfo(
    label='Anthropic - Claude Sonnet 4.5',
    versions=['claude-sonnet-4-5-20250929'],
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        system_role=True,
        output=['text', 'json'],
        constrained=Constrained.ALL,
    ),
)

CLAUDE_HAIKU_4_5 = ModelInfo(
    label='Anthropic - Claude Haiku 4.5',
    versions=['claude-haiku-4-5-20251001'],
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        system_role=True,
        output=['text', 'json'],
        constrained=Constrained.ALL,
    ),
)

CLAUDE_OPUS_4_1 = ModelInfo(
    label='Anthropic - Claude Opus 4.1',
    versions=['claude-opus-4-1-20250805'],
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        system_role=True,
        output=['text', 'json'],
        constrained=Constrained.ALL,
    ),
)

CLAUDE_OPUS_4_5 = ModelInfo(
    label='Anthropic - Claude Opus 4.5',
    versions=['claude-opus-4-5-20251101'],
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        system_role=True,
        output=['text', 'json'],
        constrained=Constrained.ALL,
    ),
)

# Source: https://docs.anthropic.com/en/docs/about-claude/models
# Released: February 5, 2026. Most capable model — excels in coding, agents,
# and enterprise workflows. Supports 1M context window (beta).
CLAUDE_OPUS_4_6 = ModelInfo(
    label='Anthropic - Claude Opus 4.6',
    versions=['claude-opus-4-6'],
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        system_role=True,
        output=['text', 'json'],
        constrained=Constrained.ALL,
    ),
)

SUPPORTED_ANTHROPIC_MODELS: dict[str, ModelInfo] = {
    'claude-3-haiku': CLAUDE_3_HAIKU,
    'claude-3-5-haiku': CLAUDE_3_5_HAIKU,
    'claude-3-5-sonnet': CLAUDE_3_5_SONNET,
    'claude-sonnet-4': CLAUDE_SONNET_4,
    'claude-opus-4': CLAUDE_OPUS_4,
    'claude-sonnet-4-5': CLAUDE_SONNET_4_5,
    'claude-haiku-4-5': CLAUDE_HAIKU_4_5,
    'claude-opus-4-1': CLAUDE_OPUS_4_1,
    'claude-opus-4-5': CLAUDE_OPUS_4_5,
    'claude-opus-4-6': CLAUDE_OPUS_4_6,
}

DEFAULT_SUPPORTS = Supports(
    multiturn=True,
    media=True,
    tools=True,
    system_role=True,
    output=['text'],
)


def get_model_info(name: str) -> ModelInfo:
    """Get model info for a given model name.

    Args:
        name: Model name.

    Returns:
        Model information.
    """
    return SUPPORTED_ANTHROPIC_MODELS.get(
        name,
        ModelInfo(
            label=f'Anthropic - {name}',
            supports=DEFAULT_SUPPORTS,
        ),
    )
