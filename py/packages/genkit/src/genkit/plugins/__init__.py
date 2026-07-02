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

"""Namespace package for Genkit plugins.

This package acts as a namespace allowing plugins to be discovered from
multiple installed packages. Each plugin can be imported as:

    from genkit.plugins.<plugin_name> import <PluginClass>

For example:
    from genkit.plugins.google_genai import GoogleGenai
    from genkit.plugins.anthropic import Anthropic

Because this module ships an ``__init__.py``, ``genkit.plugins`` is a regular
package, not a PEP 420 namespace. Its ``__path__`` is extended at runtime by
``genkit._core._plugins.extend_plugin_namespace`` (called from
``genkit.__init__``), which scans ``sys.path`` for ``genkit/plugins`` dirs.

``pkgutil.extend_path`` is not used: it searches ``genkit.__path__`` rather than
``sys.path``, and a single-directory ``genkit`` package does not span the other
distributions, so it would find nothing.

A native PEP 420 namespace would require ``genkit`` itself to be a pure namespace
package (no ``__init__.py``, like ``google.cloud.*``), giving up the populated
top-level ``genkit`` API. So the runtime scan stays.
"""
