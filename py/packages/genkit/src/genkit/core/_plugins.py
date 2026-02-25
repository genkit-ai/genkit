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

"""Internal utilities for plugin discovery and package setup.

This module handles namespace package setup for the plugin system.
"""

import sys
from pathlib import Path


def extend_plugin_namespace() -> None:
    """Extend genkit.plugins namespace to discover all installed plugins.

    This enables plugins installed in different locations to be imported
    as genkit.plugins.<plugin_name>, following PEP 420 namespace packages.

    The function scans sys.path for any genkit/plugins directories and adds
    them to the genkit.plugins.__path__, allowing Python to find plugins
    from multiple installed packages.

    This is necessary because the main genkit package has an __init__.py
    (making it a regular package), so Python doesn't automatically discover
    namespace packages nested within it.

    Example:
        After this runs, the following imports work for any installed plugin:

        >>> from genkit.plugins.anthropic import Anthropic
        >>> from genkit.plugins.firebase import Firebase
        >>> from genkit.plugins.my_custom_plugin import MyPlugin
    """
    # Import genkit.plugins to initialize the namespace if needed
    if 'genkit.plugins' not in sys.modules:
        # Deferred import: initialize the namespace package only if not already loaded
        import genkit.plugins  # noqa: F401, PLC0415

    genkit_plugins = sys.modules.get('genkit.plugins')
    if genkit_plugins is None or not hasattr(genkit_plugins, '__path__'):
        return

    # Track paths we've already added to avoid duplicates
    existing_paths = set(genkit_plugins.__path__)

    # Scan sys.path for genkit/plugins directories
    for path_str in sys.path:
        path = Path(path_str)
        plugins_path = path / 'genkit' / 'plugins'

        if plugins_path.is_dir():
            plugins_path_str = str(plugins_path)
            if plugins_path_str not in existing_paths:
                genkit_plugins.__path__.append(plugins_path_str)
                existing_paths.add(plugins_path_str)


__all__ = ['extend_plugin_namespace']
