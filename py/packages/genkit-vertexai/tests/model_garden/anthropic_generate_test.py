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

"""Model Garden Claude rejects a per-request Anthropic key, driven through ``ai.generate``."""

import httpx
import pytest
from genkit_vertexai.model_garden import ModelGarden

from genkit import FinishReason, Genkit

MODEL = 'modelgarden/anthropic/claude-sonnet-4-6'


def _genkit() -> Genkit:
    return Genkit(plugins=[ModelGarden(project='my-project', location='us-east5')])


@pytest.mark.asyncio
async def test_generate_model_garden_claude_secrets_key_fails_invalid_argument(
    vertex_requests: list[httpx.Request],
) -> None:
    """Model Garden bills the Google Cloud project, so a per-request Anthropic key fails instead of being ignored."""
    ai = _genkit()

    response = await ai.generate(model=MODEL, prompt='hi', context={'secrets': {'api_key': 'tenant-key'}})

    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert 'context.secrets' in response.error.message
    assert 'tenant-key' not in response.error.message
    assert vertex_requests == []
