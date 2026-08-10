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


"""Genkit format package. Provides implementation for various formats like json, jsonl, etc."""

from genkit._ai._formats._array import ArrayFormat
from genkit._ai._formats._enum import EnumFormat
from genkit._ai._formats._json import JsonFormat
from genkit._ai._formats._jsonl import JsonlFormat
from genkit._ai._formats._text import TextFormat
from genkit._ai._formats._types import FormatDef, Formatter


def package_name() -> str:
    """Get the fully qualified package name."""
    return 'genkit._ai._formats'


built_in_formats = [
    ArrayFormat(),
    EnumFormat(),
    JsonFormat(),
    JsonlFormat(),
    TextFormat(),
]
