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

"""Plugin discovery and environment variable checking.

All plugin metadata (required env vars, additional model providers) is read
from :class:`conform.config.ConformConfig` so that new plugins can be added
via ``pyproject.toml`` without touching code.

Plugin paths are derived from the active :class:`conform.types.Runtime`,
making this module runtime-agnostic.
"""

from __future__ import annotations

import os
from pathlib import Path

from conform.config import ConformConfig
from conform.types import Runtime


def discover_plugins(runtime: Runtime) -> list[str]:
    """Return sorted list of plugin names that have conformance directories.

    A plugin is "available" if its subdirectory under the runtime's
    ``specs_dir`` contains a ``model-conformance.yaml`` spec file.
    """
    specs_dir = runtime.specs_dir
    if not specs_dir.is_dir():
        return []
    plugins: list[str] = []
    for child in sorted(specs_dir.iterdir()):
        if child.is_dir() and (child / 'model-conformance.yaml').exists():
            plugins.append(child.name)
    return plugins


def check_env(plugin: str, config: ConformConfig) -> list[str]:
    """Return list of missing environment variables for *plugin*.

    The required variables are read from *config.env*.
    """
    required = config.env.get(plugin, [])
    return [v for v in required if not os.environ.get(v)]


def discover_model_plugins(config: ConformConfig) -> list[str]:
    """Return plugin directory names that are model providers.

    A plugin is a model provider if it contains ``model_info.py`` anywhere
    under ``src/``, or if it appears in *config.additional_model_plugins*.
    """
    plugins_dir = config.runtime.plugins_dir
    model_plugins: set[str] = set()

    if plugins_dir.is_dir():
        for plugin_dir in sorted(plugins_dir.iterdir()):
            if not plugin_dir.is_dir():
                continue
            src_dir = plugin_dir / 'src'
            if not src_dir.exists():
                continue
            for model_info in src_dir.rglob(config.runtime.model_marker):
                if model_info.is_file():
                    model_plugins.add(plugin_dir.name)
                    break

    # Include additional model plugins that use dynamic registration.
    for name in config.additional_model_plugins:
        if (plugins_dir / name).is_dir():
            model_plugins.add(name)

    return sorted(model_plugins)


def spec_file(plugin: str, config: ConformConfig) -> Path:
    """Return the path to a plugin's conformance spec file."""
    return config.runtime.specs_dir / plugin / 'model-conformance.yaml'


def entry_point(plugin: str, config: ConformConfig) -> Path:
    """Return the path to a plugin's conformance entry point."""
    return config.runtime.specs_dir / plugin / config.runtime.entry_filename
