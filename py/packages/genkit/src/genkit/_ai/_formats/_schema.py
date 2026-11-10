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
    schema: dict[str, object], node: object, *, _resolving: frozenset[str] = frozenset()
) -> object:
    """Resolve local ``$ref`` entries within a JSON schema node.

    Formatters often need to inspect ``items`` schemas directly, but Pydantic can
    emit array schemas that use ``$defs`` and ``$ref``. This helper expands those
    local references so formatters can work with standard generated schemas.
    References already being expanded on this branch remain references, so
    recursive models keep their constraints without infinite expansion.
    """
    if isinstance(node, dict):
        d = cast(dict[str, object], node)
        ref = d.get('$ref')
        if isinstance(ref, str) and ref.startswith('#/'):
            if ref in _resolving:
                return node
            target: object = schema
            for part in ref[2:].split('/'):
                if not isinstance(target, dict):
                    return node
                target = target.get(_unescape_json_pointer_token(part))
            if target is None:
                return node
            resolved = resolve_json_schema_refs(schema, target, _resolving=_resolving | {ref})
            if not isinstance(resolved, dict):
                return resolved
            merged = {k: v for k, v in node.items() if k != '$ref'}
            return {**cast(dict[str, object], resolved), **merged}
        return {key: resolve_json_schema_refs(schema, value, _resolving=_resolving) for key, value in node.items()}
    if isinstance(node, list):
        return [resolve_json_schema_refs(schema, value, _resolving=_resolving) for value in node]
    return node


def has_local_json_schema_refs(node: object) -> bool:
    """Whether a node still needs the original root to resolve a local reference."""
    if isinstance(node, dict):
        ref = node.get('$ref')
        return (isinstance(ref, str) and ref.startswith('#')) or any(
            has_local_json_schema_refs(value) for value in node.values()
        )
    if isinstance(node, list):
        return any(has_local_json_schema_refs(value) for value in node)
    return False
