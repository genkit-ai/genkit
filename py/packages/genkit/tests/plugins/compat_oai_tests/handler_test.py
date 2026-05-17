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
"""Tests for the OpenAI Model Handler compatibility plugin."""

from unittest.mock import MagicMock

import pytest

from genkit.plugins.compat_oai.models import OpenAIModelHandler
from genkit.plugins.compat_oai.models.model_info import SUPPORTED_OPENAI_MODELS


def test_get_model_handler() -> None:
    """Test get_model_handler method returns a callable."""
    model_name = 'gpt-4'
    handler = OpenAIModelHandler.get_model_handler(model=model_name, client=MagicMock())
    assert callable(handler)


def test_get_model_handler_invalid() -> None:
    """Test get_model_handler raises ValueError for unsupported models."""
    with pytest.raises(ValueError, match="Model 'unsupported-model' is not supported."):
        OpenAIModelHandler.get_model_handler(model='unsupported-model', client=MagicMock())


def test_validate_version() -> None:
    """Test validate_version method validates supported versions."""
    model = MagicMock()
    model.name = 'gpt-4'
    SUPPORTED_OPENAI_MODELS['gpt-4'] = MagicMock(versions=['gpt-4', 'gpt-3.5-turbo'])
    handler = OpenAIModelHandler(model)

    handler._validate_version('gpt-4')  # Should not raise an error

    with pytest.raises(ValueError, match="Model version 'invalid-version' is not supported."):
        handler._validate_version('invalid-version')
