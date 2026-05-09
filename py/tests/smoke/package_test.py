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

"""Smoke tests for package structure."""

from genkit.plugins.google_cloud import package_name as google_cloud_package_name
from genkit.plugins.google_genai import package_name as google_genai_package_name
from genkit.plugins.ollama import package_name as ollama_package_name
from genkit.plugins.vertex_ai import package_name as vertex_ai_package_name


def test_package_names() -> None:
    """A test that ensure that the package imports work correctly.

    This test verifies that the package imports work correctly from the
    end-user perspective.
    """
    assert google_cloud_package_name() == 'genkit.plugins.google_cloud'
    assert google_genai_package_name() == 'genkit.plugins.google_genai'
    assert ollama_package_name() == 'genkit.plugins.ollama'
    assert vertex_ai_package_name() == 'genkit.plugins.vertex_ai'
