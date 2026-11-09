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

"""Helpers for working with formatter JSON schemas."""

from __future__ import annotations

from typing import cast


def _unescape_json_pointer_token(token: str) -> str:
    """Unescape one RFC 6901 JSON pointer token."""
    return token.replace('~1', '/').replace('~0', '~')


def resolve_json_schema_refs(
    schema: dict[str, object],
    node: object,
    _expanding: frozenset[str] = frozenset(),
) -> object:
    """Resolve local ``$ref`` entries within a JSON schema node.

    Formatters often need to inspect ``items`` schemas directly, but Pydantic can
    emit array schemas that use ``$defs`` and ``$ref``. This helper expands those
    local references so formatters can work with standard generated schemas.

    A recursive model (``Category.children: list[Category]``) refers back to a
    def that is already being expanded. That ``$ref`` is left as is, and the
    root's ``$defs`` stay in the output so it still resolves.

    Example:
        ```python
        schema = {
            'type': 'array',
            'items': {'$ref': '#/$defs/Category'},
            '$defs': {
                'Category': {
                    'type': 'object',
                    'properties': {'children': {'type': 'array', 'items': {'$ref': '#/$defs/Category'}}},
                },
            },
        }
        resolved = resolve_json_schema_refs(schema, schema)
        resolved['items']['properties']['children']['items']
        # => {'$ref': '#/$defs/Category'}
        ```
    """
    if isinstance(node, dict):
        d = cast(dict[str, object], node)
        ref = d.get('$ref')
        if isinstance(ref, str) and ref.startswith('#/'):
            if ref in _expanding:
                return node
            target: object = schema
            for part in ref[2:].split('/'):
                if not isinstance(target, dict):
                    return node
                target = target.get(_unescape_json_pointer_token(part))
            if target is None:
                return node
            resolved = resolve_json_schema_refs(schema, target, _expanding | {ref})
            if not isinstance(resolved, dict):
                return resolved
            merged = {k: v for k, v in node.items() if k != '$ref'}
            return {**cast(dict[str, object], resolved), **merged}
        return {key: resolve_json_schema_refs(schema, value, _expanding) for key, value in node.items()}
    if isinstance(node, list):
        return [resolve_json_schema_refs(schema, value, _expanding) for value in node]
    return node
