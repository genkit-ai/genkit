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

"""One declaration per common config field, across every model plugin.

``ModelConfig`` declares the settings every model understands
(``temperature``, ``max_output_tokens``, ``top_k``, ``top_p``,
``stop_sequences``, ``version``). A plugin config that extends it must not
declare them again, even with tighter bounds (those go in
``common_field_bounds``), and no plugin config may declare a provider's own
name for one of them (``stop``, ``num_predict``, ``max_tokens``, ...): the
plugin maps the common field to its wire name. Two names for one setting
means a precedence rule nobody can see from the Dev UI form.

This walks every module of each model plugin, so a new config class is
checked the day it lands.
"""

import importlib
import inspect
import pkgutil
from types import ModuleType

import pytest
from pydantic import BaseModel

from genkit.model import ModelConfig
from genkit.plugin_api import GenkitConfig

PLUGIN_PACKAGES = (
    'genkit_amazon_bedrock',
    'genkit_anthropic',
    'genkit_google_genai',
    'genkit_ollama',
    'genkit_openai',
    'genkit_vertexai',
)

COMMON_FIELDS = frozenset(name for name in ModelConfig.model_fields if name != 'extra')

# Provider spellings of a common field. candidate_count is not one: it picks
# how many replies, which ModelConfig has no field for.
NATIVE_ALIASES = frozenset({
    'stop',
    'num_predict',
    'numPredict',
    'max_tokens',
    'maxTokens',
    'max_completion_tokens',
    'maxCompletionTokens',
    'max_new_tokens',
    'maxNewTokens',
    'topK',
    'topP',
    'stopSequences',
    'maxOutputTokens',
})


def _modules(package: str) -> list[ModuleType]:
    root = importlib.import_module(package)
    found = [root]
    for info in pkgutil.walk_packages(root.__path__, prefix=f'{package}.'):
        found.append(importlib.import_module(info.name))
    return found


def _config_classes() -> list[type[GenkitConfig]]:
    seen: dict[str, type[GenkitConfig]] = {}
    for package in PLUGIN_PACKAGES:
        for module in _modules(package):
            for _, value in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(value, GenkitConfig)
                    and value.__module__.startswith(package)
                    and value not in (GenkitConfig, ModelConfig)
                ):
                    seen[f'{value.__module__}.{value.__qualname__}'] = value
    return [seen[key] for key in sorted(seen)]


CONFIG_CLASSES = _config_classes()


def _own_fields(cls: type[BaseModel]) -> dict[str, type[BaseModel]]:
    """Fields declared by ``cls`` or a plugin base, keyed to the class that annotated them."""
    owners: dict[str, type[BaseModel]] = {}
    for klass in cls.__mro__:
        if klass in (ModelConfig, GenkitConfig, BaseModel) or not issubclass(klass, BaseModel):
            continue
        for name in vars(klass).get('__annotations__', {}):
            if name in cls.model_fields:
                owners.setdefault(name, klass)
    return owners


def test_guard_finds_the_plugin_configs() -> None:
    """The walk reaches the config classes it is meant to guard."""
    names = {cls.__name__ for cls in CONFIG_CLASSES}
    assert {'GeminiConfig', 'OpenAIConfig', 'OllamaConfig', 'AnthropicConfig', 'BedrockConfig'} <= names


@pytest.mark.parametrize('cls', [c for c in CONFIG_CLASSES if issubclass(c, ModelConfig)], ids=lambda c: c.__name__)
def test_model_config_subclass_does_not_redeclare_common_field(cls: type[ModelConfig]) -> None:
    """No class between a plugin config and ModelConfig annotates a common field again."""
    redeclared = {name: owner.__name__ for name, owner in _own_fields(cls).items() if name in COMMON_FIELDS}
    assert redeclared == {}, f'{cls.__name__} re-declares {redeclared}; narrow ranges with common_field_bounds'


@pytest.mark.parametrize('cls', CONFIG_CLASSES, ids=lambda c: c.__name__)
def test_config_does_not_declare_native_alias(cls: type[GenkitConfig]) -> None:
    """No plugin config declares a provider's own name for a common field, by name or alias."""
    aliased = {
        name
        for name in _own_fields(cls)
        if name in NATIVE_ALIASES or (cls.model_fields[name].alias or '') in NATIVE_ALIASES
    }
    assert aliased == set(), f'{cls.__name__} declares {sorted(aliased)}; map the common field to the wire name'


@pytest.mark.parametrize('cls', [c for c in CONFIG_CLASSES if issubclass(c, ModelConfig)], ids=lambda c: c.__name__)
def test_common_field_bounds_name_numeric_common_fields(cls: type[ModelConfig]) -> None:
    """``common_field_bounds`` only narrows numeric common fields, low before high."""
    for name, (low, high) in cls.common_field_bounds.items():
        assert name in {'temperature', 'top_p', 'top_k', 'max_output_tokens'}, name
        assert low <= high, name
