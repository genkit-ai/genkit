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

"""Every model a first-party plugin ships declares capabilities Supports accepts."""

import importlib
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from genkit.model import ModelInfo, Supports


def _supports_in(value: Any, seen: set[int]) -> Iterator[Supports]:
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, Supports):
        yield value
    elif isinstance(value, ModelInfo) and value.supports is not None:
        yield value.supports
    elif isinstance(value, dict):
        for item in value.values():
            yield from _supports_in(item, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _supports_in(item, seen)


def _catalog(module_name: str) -> Callable[[], list[Supports]]:
    def collect() -> list[Supports]:
        module = importlib.import_module(module_name)
        seen: set[int] = set()
        return [s for value in vars(module).values() for s in _supports_in(value, seen)]

    return collect


def _bedrock() -> list[Supports]:
    from genkit_amazon_bedrock.model_info import MODEL_CAPABILITIES, get_model_info

    infos = [get_model_info(name) for name in MODEL_CAPABILITIES]
    infos.append(get_model_info('amazon.nova-canvas-v1:0', 'image'))
    return [info.supports for info in infos if info.supports is not None]


def _vertex_model_garden_anthropic() -> list[Supports]:
    from genkit_vertexai.model_garden.anthropic import AnthropicModelGarden

    info = AnthropicModelGarden('anthropic/claude-sonnet-4', 'us-east5', 'project').get_model_info()
    return [info.supports] if info.supports is not None else []


SOURCES = {
    'amazon-bedrock': _bedrock,
    'anthropic': _catalog('genkit_anthropic.model_info'),
    'google-genai-gemini': _catalog('genkit_google_genai.models.gemini'),
    'google-genai-veo': _catalog('genkit_google_genai.models.veo'),
    'openai': _catalog('genkit_openai.models.model_info'),
    'vertexai-model-garden-anthropic': _vertex_model_garden_anthropic,
}


@pytest.mark.parametrize('source', SOURCES)
def test_every_first_party_model_info_validates(source: str) -> None:
    """Every `Supports` a first-party plugin builds validates under the current rules."""
    found = SOURCES[source]()

    assert found, f'{source} has no model info to check'
    for supports in found:
        Supports.model_validate(supports.model_dump())
