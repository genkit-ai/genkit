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

"""Tiny RFC 6902 JSON Patch diff for streaming agent custom state."""

from __future__ import annotations

import copy
import json
from typing import Any

from genkit._core._typing import JsonPatchOperation


def _escape_token(token: str) -> str:
    return token.replace('~', '~0').replace('/', '~1')


def _is_object(value: Any) -> bool:
    return isinstance(value, dict)


def _deep_equal(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_deep_equal(x, y) for x, y in zip(a, b, strict=True))
    if _is_object(a) and _is_object(b):
        if set(a) != set(b):
            return False
        return all(_deep_equal(a[k], b[k]) for k in a)
    return a == b


def _clone(value: Any) -> Any:
    if value is None:
        return None
    try:
        return copy.deepcopy(value)
    except Exception:  # noqa: BLE001
        return json.loads(json.dumps(value))


def diff_json(from_value: Any, to_value: Any) -> list[JsonPatchOperation]:
    """Return add/remove/replace ops that transform ``from_value`` into ``to_value``."""
    patch: list[JsonPatchOperation] = []
    _diff_recursive(from_value, to_value, '', patch)
    return patch


def _diff_recursive(from_value: Any, to_value: Any, pointer: str, patch: list[JsonPatchOperation]) -> None:
    if _deep_equal(from_value, to_value):
        return

    if _is_object(from_value) and _is_object(to_value):
        keys = set(from_value) | set(to_value)
        for key in sorted(keys):
            child_pointer = f'{pointer}/{_escape_token(key)}'
            in_from = key in from_value
            in_to = key in to_value
            if in_from and not in_to:
                patch.append(JsonPatchOperation(op='remove', path=child_pointer))
            elif not in_from and in_to:
                patch.append(JsonPatchOperation(op='add', path=child_pointer, value=_clone(to_value[key])))
            else:
                _diff_recursive(from_value[key], to_value[key], child_pointer, patch)
        return

    if isinstance(from_value, list) and isinstance(to_value, list):
        min_len = min(len(from_value), len(to_value))
        for i in range(min_len):
            _diff_recursive(from_value[i], to_value[i], f'{pointer}/{i}', patch)
        if len(to_value) > len(from_value):
            for i in range(len(from_value), len(to_value)):
                patch.append(JsonPatchOperation(op='add', path=f'{pointer}/-', value=_clone(to_value[i])))
        elif len(from_value) > len(to_value):
            for i in range(len(from_value) - 1, len(to_value) - 1, -1):
                patch.append(JsonPatchOperation(op='remove', path=f'{pointer}/{i}'))
        return

    patch.append(JsonPatchOperation(op='replace', path=pointer, value=_clone(to_value)))
