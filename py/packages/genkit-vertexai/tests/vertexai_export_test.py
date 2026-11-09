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

"""genkit_vertexai re-exports VertexAI from genkit_google_genai."""

import genkit_google_genai
import genkit_vertexai


def test_vertexai_is_the_google_genai_class() -> None:
    """One class, two import paths: plugin identity and isinstance checks agree."""
    assert genkit_vertexai.VertexAI is genkit_google_genai.VertexAI


def test_public_surface() -> None:
    """Both plugins are exported."""
    assert sorted(genkit_vertexai.__all__) == ['ModelGarden', 'VertexAI']
