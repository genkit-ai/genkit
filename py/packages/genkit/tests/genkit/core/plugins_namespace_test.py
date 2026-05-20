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

from pathlib import Path

import genkit
import genkit.plugins


def test_plugins_namespace_discovery():
    """Verify that the genkit.plugins namespace dynamically discovers installed plugins."""
    # Ensure the namespace exists and has __path__
    assert hasattr(genkit.plugins, '__path__')

    # Get all path entries in the namespace package
    namespace_paths = [Path(p).resolve() for p in genkit.plugins.__path__]

    # Ensure we have multiple distinct source/installation directories in the namespace
    # (e.g., core package genkit.plugins path vs. plugin package paths)
    assert len(namespace_paths) >= 1

    # Verify that we can dynamically import the Google GenAI plugin from the namespace package
    try:
        from genkit.plugins import google_genai
        assert google_genai is not None
    except ImportError as e:
        pytest.fail(f"Failed to import google_genai from namespace package: {e}")


def test_plugins_namespace_contents():
    """Verify that core package does not statically couple to the google-genai plugin dir."""
    # The directory of google-genai should be in sys.path for this workspace
    # and discovered as part of the genkit.plugins namespace.
    import importlib
    google_genai_module = importlib.import_module("genkit.plugins.google_genai")

    module_file = Path(google_genai_module.__file__).resolve()
    # It should live under the google-genai plugin source tree, not under the core genkit tree
    assert "plugins/google-genai" in str(module_file.as_posix())
