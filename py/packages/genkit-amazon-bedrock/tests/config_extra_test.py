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

"""`extra` on `BedrockConfig` lands in `additionalModelRequestFields`."""

from genkit_amazon_bedrock.config import BedrockConfig
from genkit_amazon_bedrock.converters import build_converse_request

from genkit import Message, Part, Role
from genkit.model import ModelRequest


def _kwargs(config: BedrockConfig) -> dict:
    request = ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('hi')])], config=config)
    return build_converse_request('anthropic.claude-sonnet-4-5-20250929-v1:0', request)


def test_extra_lands_in_additional_model_request_fields() -> None:
    """Converse has no open top-level body, so `extra` goes to the model-native slot."""
    kwargs = _kwargs(BedrockConfig(extra={'top_k': 40}))

    assert kwargs['additionalModelRequestFields'] == {'top_k': 40}
    assert 'extra' not in kwargs


def test_no_extra_leaves_request_unchanged() -> None:
    """No `extra` means no `additionalModelRequestFields`."""
    assert 'additionalModelRequestFields' not in _kwargs(BedrockConfig(temperature=0.2))
