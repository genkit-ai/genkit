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


"""Constants for context caching."""

CONTEXT_CACHE_SUPPORTED_MODELS = [
    'gemini-1.5-flash',
    'gemini-1.5-flash-001',
    'gemini-1.5-pro',
    'gemini-1.5-pro-001',
    'gemini-2.0-flash-001',
    'gemini-2.0-flash-exp',
    'gemini-3-flash-preview',
    'gemini-3-pro-preview',
]

INVALID_ARGUMENT_MESSAGES = {
    'modelVersion': (
        'Model version is required for context caching. Supported models: '
        + ', '.join(CONTEXT_CACHE_SUPPORTED_MODELS)
        + '.'
    ),
    'tools': 'Context caching cannot be used simultaneously with tools.',
    'codeExecution': 'Context caching cannot be used simultaneously with code execution.',
}

DEFAULT_TTL = 300
