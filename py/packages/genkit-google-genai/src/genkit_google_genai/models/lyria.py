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

"""Lyria family detection for Google model routing.

Lyria is Google's music and audio generation model. Google AI serves it through
Interactions (see ``interactions_lyria``); Vertex AI has no Lyria action, so the
family is recognized here to keep those ids from resolving as Gemini.
"""


def is_lyria_model(name: str) -> bool:
    """Check if a model name is a Lyria model.

    Args:
        name: The model name to check.

    Returns:
        True if this is a Lyria model name.
    """
    return name.split('/')[-1].lower().startswith('lyria-')
