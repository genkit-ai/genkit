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

"""Container for Genkit plugins.

Each plugin (``genkit.plugins.<name>``) ships in its own wheel
(``genkit-plugin-<name>``) but lives in this directory in source, so it
imports and resolves like any regular subpackage:

    from genkit.plugins.google_genai import GoogleAI
    from genkit.plugins.anthropic import Anthropic
"""
