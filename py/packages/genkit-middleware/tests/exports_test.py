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

"""Public surface of ``genkit_middleware`` and the wrong-class plugin error."""

import pytest

import genkit_middleware
from genkit import Genkit
from genkit_middleware import (
    Fallback,
    FallbackConfig,
    FallbackModelEntry,
    Filesystem,
    FilesystemConfig,
    Retry,
    RetryConfig,
    Skills,
    SkillsConfig,
    ToolApproval,
    ToolApprovalConfig,
)


@pytest.mark.parametrize(
    ('middleware', 'config'),
    [
        (Retry, RetryConfig),
        (Fallback, FallbackConfig),
        (Filesystem, FilesystemConfig),
        (Skills, SkillsConfig),
        (ToolApproval, ToolApprovalConfig),
    ],
)
def test_config_classes_are_exported(middleware: type, config: type) -> None:
    assert middleware.Config is config
    assert config.__name__ in genkit_middleware.__all__


def test_fallback_model_entry_exported() -> None:
    assert 'FallbackModelEntry' in genkit_middleware.__all__
    entry = FallbackModelEntry(name='googleai/gemini-flash-lite-latest', config={'temperature': 0})
    assert FallbackConfig(models=[entry]).models == [entry]


def test_middleware_class_in_plugins_names_plugin_api() -> None:
    with pytest.raises(ValueError, match=r'genkit\.plugin_api\.Plugin') as exc:
        Genkit(plugins=[Retry()])  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
    assert 'genkit.ai.Plugin' not in str(exc.value)
    assert 'genkit_middleware.Middleware()' in str(exc.value)
