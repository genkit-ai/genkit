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


"""Test configuration for the OpenAI compatible plugin."""

import pytest

from genkit import (
    Message,
    ModelRequest,
    Part,
    Role,
    TextPart,
)
from genkit.plugins.compat_oai.typing import OpenAIConfig


@pytest.fixture
def sample_request() -> ModelRequest:
    """Fixture to create a sample ModelRequest object."""
    return ModelRequest(
        messages=[
            Message(
                role=Role.SYSTEM,
                content=[Part(root=TextPart(text='You are an assistant'))],
            ),
            Message(role=Role.USER, content=[Part(root=TextPart(text='Hello, world!'))]),
        ],
        config=OpenAIConfig(
            model='gpt-4',
            top_p=0.9,
            temperature=0.7,
            stop=['stop'],
            max_tokens=100,
        ),
    )
