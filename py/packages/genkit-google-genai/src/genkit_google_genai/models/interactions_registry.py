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

"""Curated catalogs of Google AI Interactions models.

When Google ships a new Interactions model version, add it here — not in the
individual action modules. Plugin registration, list_actions, and resolve all
read from this file.
"""

from __future__ import annotations

from collections.abc import Mapping

from genkit import ModelInfo, Supports
from genkit._core._compat import StrEnum
from genkit_google_genai.models.interactions_utils import extract_version


def _model_info(
    version: str,
    *,
    fallback: ModelInfo,
    known: Mapping[str, ModelInfo] | None = None,
) -> ModelInfo:
    """Build ModelInfo with a Google AI label and family (or per-version) supports."""
    clean = extract_version(version)
    entry = known.get(clean) if known else None
    return ModelInfo(label=f'Google AI - {clean}', supports=(entry or fallback).supports)


# ---------------------------------------------------------------------------
# Lyria
# ---------------------------------------------------------------------------

LYRIA_INFO = ModelInfo(
    label='Google AI - Lyria',
    supports=Supports(
        multiturn=False,
        media=True,
        tools=False,
        tool_choice=False,
        system_role=False,
        output=['media', 'text'],
    ),
)


class LyriaVersion(StrEnum):
    """Lyria model version identifiers."""

    LYRIA_3_CLIP = 'lyria-3-clip-preview'
    LYRIA_3_PRO = 'lyria-3-pro-preview'


KNOWN_LYRIA_MODELS: tuple[LyriaVersion, ...] = (
    LyriaVersion.LYRIA_3_CLIP,
    LyriaVersion.LYRIA_3_PRO,
)


def is_lyria_model_name(name: str | None) -> bool:
    """Return True when the model name belongs to the Lyria family."""
    return bool(name and name.startswith('lyria-'))


def lyria_model_info(version: str) -> ModelInfo:
    """Return capability metadata for an Interactions Lyria model."""
    return _model_info(version, fallback=LYRIA_INFO)


def list_known_lyria_models() -> list[str]:
    """Return statically known Interactions Lyria model names."""
    return [str(version) for version in KNOWN_LYRIA_MODELS]


# ---------------------------------------------------------------------------
# Antigravity
# ---------------------------------------------------------------------------

ANTIGRAVITY_INFO = ModelInfo(
    label='Google AI - Antigravity',
    supports=Supports(
        multiturn=True,
        media=True,
        tools=False,
        tool_choice=False,
        system_role=False,
        output=['text'],
    ),
)

KNOWN_ANTIGRAVITY_MODELS: dict[str, ModelInfo] = {
    'antigravity-preview-05-2026': ANTIGRAVITY_INFO,
}


def is_antigravity_model_name(name: str | None) -> bool:
    """Return True when the model name belongs to the Antigravity family."""
    return bool(name and name.startswith('antigravity-'))


def antigravity_model_info(version: str) -> ModelInfo:
    """Return capability metadata for an Antigravity model."""
    return _model_info(version, known=KNOWN_ANTIGRAVITY_MODELS, fallback=ANTIGRAVITY_INFO)


def list_known_antigravity_models() -> list[str]:
    """Return statically known Antigravity model names."""
    return list(KNOWN_ANTIGRAVITY_MODELS.keys())


# ---------------------------------------------------------------------------
# Deep Research
# ---------------------------------------------------------------------------

DEEP_RESEARCH_INFO = ModelInfo(
    label='Google AI - Deep Research',
    supports=Supports(
        multiturn=True,
        media=False,
        tools=False,
        tool_choice=False,
        system_role=False,
        output=['text'],
        long_running=True,
    ),
)

ADVANCED_DEEP_RESEARCH_INFO = ModelInfo(
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=False,
        system_role=False,
        output=['text', 'media'],
        long_running=True,
    ),
)

KNOWN_DEEP_RESEARCH_MODELS: dict[str, ModelInfo] = {
    'deep-research-pro-preview-12-2025': DEEP_RESEARCH_INFO,
    'deep-research-preview-04-2026': ADVANCED_DEEP_RESEARCH_INFO,
    'deep-research-max-preview-04-2026': ADVANCED_DEEP_RESEARCH_INFO,
}


def is_deep_research_model_name(name: str | None) -> bool:
    """Return True when the model name belongs to the Deep Research family."""
    return bool(name and name.startswith('deep-research-'))


def deep_research_model_info(version: str) -> ModelInfo:
    """Return capability metadata for a Deep Research model."""
    return _model_info(version, known=KNOWN_DEEP_RESEARCH_MODELS, fallback=DEEP_RESEARCH_INFO)


def list_known_deep_research_models() -> list[str]:
    """Return statically known Deep Research model names."""
    return list(KNOWN_DEEP_RESEARCH_MODELS.keys())
