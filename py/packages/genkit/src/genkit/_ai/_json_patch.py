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

from genkit._core._typing import JsonPatchOp, JsonPatchOperation


def escape_token(token: str) -> str:
    return token.replace('~', '~0').replace('/', '~1')


def is_object(value: Any) -> bool:
    return isinstance(value, dict)


def deep_equal(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(deep_equal(x, y) for x, y in zip(a, b, strict=True))
    if is_object(a) and is_object(b):
        if set(a) != set(b):
            return False
        return all(deep_equal(a[k], b[k]) for k in a)
    return a == b


def clone(value: Any) -> Any:
    if value is None:
        return None
    try:
        return copy.deepcopy(value)
    except Exception:  # noqa: BLE001
        return json.loads(json.dumps(value))


def diff_json(*, from_value: Any, to_value: Any) -> list[JsonPatchOperation]:
    """Return add/remove/replace ops that transform ``from_value`` into ``to_value``."""
    patch: list[JsonPatchOperation] = []
    diff_recursive(from_value, to_value, '', patch)
    return patch


def diff_recursive(from_value: Any, to_value: Any, pointer: str, patch: list[JsonPatchOperation]) -> None:
    if deep_equal(from_value, to_value):
        return

    if is_object(from_value) and is_object(to_value):
        keys = set(from_value) | set(to_value)
        for key in sorted(keys):
            child_pointer = f'{pointer}/{escape_token(str(key))}'
            in_from = key in from_value
            in_to = key in to_value
            if in_from and not in_to:
                patch.append(JsonPatchOperation(op=JsonPatchOp.REMOVE, path=child_pointer))
            elif not in_from and in_to:
                patch.append(JsonPatchOperation(op=JsonPatchOp.ADD, path=child_pointer, value=clone(to_value[key])))
            else:
                diff_recursive(from_value[key], to_value[key], child_pointer, patch)
        return

    if isinstance(from_value, list) and isinstance(to_value, list):
        min_len = min(len(from_value), len(to_value))
        for i in range(min_len):
            diff_recursive(from_value[i], to_value[i], f'{pointer}/{i}', patch)
        if len(to_value) > len(from_value):
            for i in range(len(from_value), len(to_value)):
                patch.append(JsonPatchOperation(op=JsonPatchOp.ADD, path=f'{pointer}/-', value=clone(to_value[i])))
        elif len(from_value) > len(to_value):
            for i in range(len(from_value) - 1, len(to_value) - 1, -1):
                patch.append(JsonPatchOperation(op=JsonPatchOp.REMOVE, path=f'{pointer}/{i}'))
        return

    patch.append(JsonPatchOperation(op=JsonPatchOp.REPLACE, path=pointer, value=clone(to_value)))


def unescape_token(token: str) -> str:
    # ~1 before ~0 so "~01" decodes to "~1" rather than "/".
    return token.replace('~1', '/').replace('~0', '~')


def parse_pointer(pointer: str) -> list[str]:
    """Split an RFC 6901 JSON Pointer into reference tokens; root ("") yields []."""
    if pointer == '':
        return []
    if not pointer.startswith('/'):
        raise ValueError(f'Invalid JSON Pointer {pointer!r}: must start with "/".')
    return [unescape_token(part) for part in pointer[1:].split('/')]


def is_container(value: Any) -> bool:
    return isinstance(value, (dict, list))


def array_index(token: str, length: int, *, allow_end: bool) -> int | None:
    """Parse an array reference token; the "-" end token resolves to length only for inserts."""
    if token == '-':
        return length if allow_end else None
    try:
        idx = int(token)
    except ValueError:
        return None
    return idx if idx >= 0 else None


def get_path(node: Any, tokens: list[str]) -> Any:
    """Read the value at tokens, returning None for any missing segment."""
    cur = node
    for token in tokens:
        if isinstance(cur, dict):
            cur = cur.get(token)
        elif isinstance(cur, list):
            idx = array_index(token, len(cur), allow_end=False)
            if idx is None or idx >= len(cur):
                return None
            cur = cur[idx]
        else:
            return None
    return cur


def set_member(node: Any, token: str, value: Any, *, is_add: bool) -> Any:
    """Set the leaf token on node, returning the (possibly new) node."""
    if isinstance(node, dict):
        node[token] = value
        return node
    if isinstance(node, list):
        if token == '-':
            node.append(value)
            return node
        idx = array_index(token, len(node), allow_end=is_add)
        if idx is None:
            return node
        if is_add:
            if idx <= len(node):
                node.insert(idx, value)
            return node
        if idx < len(node):
            node[idx] = value
        return node
    return {token: value}


def set_path(node: Any, tokens: list[str], value: Any, *, is_add: bool) -> Any:
    """Set value at tokens, creating missing intermediate objects, returning the (possibly new) node."""
    if not tokens:
        return value
    # Lenient: initialize a missing/null container so member sets still land.
    if node is None:
        node = {}
    if len(tokens) == 1:
        return set_member(node, tokens[0], value, is_add=is_add)
    token = tokens[0]
    if isinstance(node, dict):
        child = node.get(token)
        if not is_container(child):
            child = {}
        node[token] = set_path(child, tokens[1:], value, is_add=is_add)
        return node
    if isinstance(node, list):
        idx = array_index(token, len(node), allow_end=False)
        if idx is None or idx >= len(node):
            return node
        if not is_container(node[idx]):
            node[idx] = {}
        node[idx] = set_path(node[idx], tokens[1:], value, is_add=is_add)
        return node
    # Primitive where a container was expected: replace it with one.
    return set_path({}, tokens, value, is_add=is_add)


def remove_path(node: Any, tokens: list[str]) -> Any:
    """Delete the member at tokens, returning the (possibly new) node. Missing members are a no-op."""
    if not tokens:
        return None
    if node is None:
        return None
    token = tokens[0]
    if len(tokens) == 1:
        if isinstance(node, dict):
            node.pop(token, None)
            return node
        if isinstance(node, list):
            idx = array_index(token, len(node), allow_end=False)
            if idx is not None and idx < len(node):
                node.pop(idx)
            return node
        return node
    if isinstance(node, dict):
        if token in node:
            node[token] = remove_path(node[token], tokens[1:])
        return node
    if isinstance(node, list):
        idx = array_index(token, len(node), allow_end=False)
        if idx is not None and idx < len(node):
            node[idx] = remove_path(node[idx], tokens[1:])
        return node
    return node


def apply_op(doc: Any, op: JsonPatchOperation) -> Any:
    tokens = parse_pointer(op.path)
    if op.op == JsonPatchOp.ADD:
        return set_path(doc, tokens, clone(op.value), is_add=True)
    if op.op == JsonPatchOp.REPLACE:
        return set_path(doc, tokens, clone(op.value), is_add=False)
    if op.op == JsonPatchOp.REMOVE:
        return remove_path(doc, tokens)
    if op.op == JsonPatchOp.TEST:
        if not deep_equal(get_path(doc, tokens), op.value):
            raise ValueError(f'JSON Patch test failed at {op.path!r}.')
        return doc
    if op.op == JsonPatchOp.MOVE:
        from_tokens = parse_pointer(op.from_ or '')
        value = clone(get_path(doc, from_tokens))
        doc = remove_path(doc, from_tokens)
        return set_path(doc, tokens, value, is_add=True)
    if op.op == JsonPatchOp.COPY:
        from_tokens = parse_pointer(op.from_ or '')
        return set_path(doc, tokens, clone(get_path(doc, from_tokens)), is_add=True)
    raise ValueError(f'Unsupported JSON Patch op: {op.op!r}.')


def apply_json_patch(*, doc: Any, patch: list[JsonPatchOperation]) -> Any:
    """Apply RFC 6902 JSON Patch operations to a document and return the transformed copy.

    Lenient by design so a stream of deltas stays robust: an add/replace whose
    parent container is missing initializes it, and a remove/replace of a missing
    member is a no-op. ``test`` is honored (raises on mismatch); an unrecognized
    op raises rather than silently letting client state drift.
    """
    result = clone(doc)
    for op in patch:
        result = apply_op(result, op)
    return result
