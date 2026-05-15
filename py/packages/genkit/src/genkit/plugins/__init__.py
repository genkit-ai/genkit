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

This is a namespace package that allows plugins to be discovered from
multiple installed packages. Each plugin can be imported as:

    from genkit.plugins.<plugin_name> import <PluginClass>

For example:
    from genkit.plugins.google_genai import GoogleGenai
    from genkit.plugins.anthropic import Anthropic
"""
