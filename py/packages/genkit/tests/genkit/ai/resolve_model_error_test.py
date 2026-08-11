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

"""Tests for generate-path model resolution errors."""

import pytest

from genkit import GenkitError
from genkit._ai._generate import resolve_parameters
from genkit._core._model import GenerateActionOptions
from genkit._core._registry import Registry


@pytest.mark.asyncio
async def test_resolve_parameters_missing_prefix_hints_plugin_namespace() -> None:
    registry = Registry()
    with pytest.raises(GenkitError) as exc_info:
        await resolve_parameters(
            registry,
            GenerateActionOptions(model='lyria-3-clip-preview', messages=[]),
        )
    assert exc_info.value.status == 'NOT_FOUND'
    assert "Failed to resolve model 'lyria-3-clip-preview'." in str(exc_info.value)
    assert 'Ensure the model name includes the plugin namespace' in str(exc_info.value)


@pytest.mark.asyncio
async def test_resolve_parameters_no_model_is_invalid_argument() -> None:
    registry = Registry()
    with pytest.raises(GenkitError) as exc_info:
        await resolve_parameters(registry, GenerateActionOptions(messages=[]))
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'No model configured' in str(exc_info.value)
