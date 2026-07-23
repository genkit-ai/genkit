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

"""Tests for ollama_model family helper."""

from genkit_ollama import OllamaConfig, ollama_model


def test_ollama_model_stamps_namespace_and_schema() -> None:
    ref = ollama_model('llama3.2')
    assert ref.name == 'ollama/llama3.2'
    assert ref.config_schema is OllamaConfig


def test_ollama_model_unknown_name_still_typed_schema() -> None:
    ref = ollama_model('some-new-local-model')
    assert ref.name == 'ollama/some-new-local-model'
    assert ref.config_schema is OllamaConfig


def test_ollama_model_idempotent_namespace() -> None:
    ref = ollama_model('ollama/mistral')
    assert ref.name == 'ollama/mistral'


def test_ollama_model_stamps_default_config() -> None:
    config = OllamaConfig(temperature=0.5)
    ref = ollama_model('mistral', config=config)
    assert ref.config is config
    assert ref.config.temperature == 0.5
