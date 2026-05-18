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

"""Functions for working with schema."""

from typing import Any

from pydantic import TypeAdapter


def to_json_schema(schema: type | dict[str, Any] | str | None) -> dict[str, Any]:
    """Convert a Python type to JSON schema. Pass-through if already a dict."""
    if schema is None:
        return {'type': 'null'}
    if isinstance(schema, dict):
        return schema
    type_adapter = TypeAdapter(schema)
    return type_adapter.json_schema()
