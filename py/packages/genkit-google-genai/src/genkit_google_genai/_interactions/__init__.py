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

"""Google AI Interactions API types, converters, and HTTP client."""

from genkit_google_genai._interactions.client import InteractionsClient
from genkit_google_genai._interactions.converters import (
    ensure_tool_ids,
    from_interaction,
    from_interaction_content,
    from_interaction_step,
    from_interaction_sync,
    to_interaction_content,
    to_interaction_role,
    to_interaction_steps,
    to_interaction_tool,
)
from genkit_google_genai._interactions.types import (
    API_REVISION,
    CreateInteractionRequest,
    GeminiInteraction,
)

__all__ = [
    'API_REVISION',
    'CreateInteractionRequest',
    'GeminiInteraction',
    'InteractionsClient',
    'ensure_tool_ids',
    'from_interaction',
    'from_interaction_content',
    'from_interaction_step',
    'from_interaction_sync',
    'to_interaction_content',
    'to_interaction_role',
    'to_interaction_steps',
    'to_interaction_tool',
]
