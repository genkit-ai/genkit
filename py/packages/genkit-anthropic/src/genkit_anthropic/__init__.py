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

"""Anthropic plugin for Genkit.

This plugin provides integration with Anthropic's Claude models for the
Genkit framework. It registers Claude models as Genkit actions, enabling
text generation operations.

Example:
    ```python
    from genkit import Genkit
    from genkit_anthropic import Anthropic

    ai = Genkit(
        plugins=[Anthropic()],
        model='anthropic/claude-3-5-sonnet-latest',
    )
    response = await ai.generate(prompt='Hello, Claude!')
    print(response.text)
    ```

Requirements:
    - Requires the ``ANTHROPIC_API_KEY`` environment variable or explicit ``api_key``.

See Also:
    - Anthropic documentation: https://docs.anthropic.com/
"""

from genkit_anthropic.plugin import Anthropic, anthropic_name

__all__ = ['Anthropic', 'anthropic_name']
