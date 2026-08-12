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

"""Property and edge coverage for the JSON Patch diff/apply engine.

The store contract is: for any two JSON-serializable states,
``apply(parent, wire_round_trip(diff(parent, child))) == child`` — including
explicit nulls, multilingual strings, RFC 6901 escaping, and list resizing.
A seeded randomized sweep enforces the property; targeted cases pin the
interop-critical shapes and the op handlers the differ itself never emits.
"""

import copy
import json
import random
from typing import Any

import pytest

from genkit._ai._json_patch import apply_json_patch, diff_json
from genkit._core._typing import JsonPatchOp, JsonPatchOperation

_KEYS = ['a', 'b', '中文', 'k/slash', 't~tilde', 'x', 'messages']
_LEAVES = [None, 0, 1, 'ok', '中😀', True, False, 3.5, '']


def _rand_value(rng: random.Random, depth: int = 0) -> Any:  # noqa: ANN401
    roll = rng.random()
    if depth > 3 or roll < 0.25:
        return rng.choice(_LEAVES)
    if roll < 0.55:
        return {k: _rand_value(rng, depth + 1) for k in rng.sample(_KEYS, rng.randint(0, 4))}
    return [_rand_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]


def _mutate(rng: random.Random, value: Any, depth: int = 0) -> Any:  # noqa: ANN401
    if rng.random() < 0.3 or not isinstance(value, (dict, list)):
        return _rand_value(rng, depth)
    value = copy.deepcopy(value)
    if isinstance(value, dict):
        for key in list(value):
            if rng.random() < 0.3:
                del value[key]
            elif rng.random() < 0.5:
                value[key] = _mutate(rng, value[key], depth + 1)
        if rng.random() < 0.5:
            value[rng.choice(_KEYS)] = _rand_value(rng, depth + 1)
    else:
        for i in range(len(value)):
            if rng.random() < 0.4:
                value[i] = _mutate(rng, value[i], depth + 1)
        if rng.random() < 0.4:
            value.append(_rand_value(rng, depth + 1))
        if value and rng.random() < 0.4:
            value.pop(rng.randrange(len(value)))
    return value


def _wire_round_trip(patch: list[JsonPatchOperation]) -> list[JsonPatchOperation]:
    """Serialize the way the stores do (nulls preserved) and re-parse from JSON."""
    raw = []
    for op in patch:
        d: dict[str, Any] = {'op': op.op.value, 'path': op.path}
        if op.op in (JsonPatchOp.ADD, JsonPatchOp.REPLACE, JsonPatchOp.TEST):
            d['value'] = op.value
        if op.op in (JsonPatchOp.MOVE, JsonPatchOp.COPY):
            d['from'] = op.from_
        raw.append(d)
    parsed = json.loads(json.dumps(raw, ensure_ascii=False))
    return [JsonPatchOperation.model_validate(op) for op in parsed]


def test_json_patch_random_round_trips() -> None:
    """diff → wire → apply reproduces the target across 2000 seeded random pairs."""
    rng = random.Random(5757)
    for trial in range(2000):
        parent = _rand_value(rng)
        child = _mutate(rng, parent)
        patch = _wire_round_trip(diff_json(from_value=parent, to_value=child))
        got = apply_json_patch(doc=copy.deepcopy(parent), patch=patch)
        assert got == child, f'trial {trial}: parent={parent!r} child={child!r} got={got!r}'


@pytest.mark.parametrize(
    ('parent', 'child'),
    [
        ({'x': 1}, {'x': None}),
        ({}, {'x': None}),
        ({'a': {'b': [1, 2, 3]}}, {'a': {'b': [1]}}),
        ({'a': [1]}, {'a': [1, 2, 3]}),
        ({'k/slash': 1, 't~tilde': 2}, {'k/slash': 9}),
        (None, {'fresh': {'中': None}}),
        ({'n': 1}, {'n': True}),
        ({'n': 1}, {'n': 1.0}),
    ],
)
def test_json_patch_targeted_round_trips(parent: Any, child: Any) -> None:  # noqa: ANN401
    """Interop-critical shapes: explicit nulls, escaping, resizing, type strictness."""
    patch = _wire_round_trip(diff_json(from_value=parent, to_value=child))
    got = apply_json_patch(doc=copy.deepcopy(parent), patch=patch)
    assert got == child
    # Note: null-on-the-wire is asserted against actual stored documents in the
    # Firestore store suite; Python's lenient apply (value defaults to None)
    # would mask a dropped null here, so an equality check alone can't prove it.


def test_json_patch_apply_move_copy_and_test_ops() -> None:
    """Handlers the differ never emits still apply correctly when read from the wire."""
    doc = {'a': {'x': 1}, 'b': [1, 2]}
    patch = [
        JsonPatchOperation(op=JsonPatchOp.TEST, path='/a/x', value=1),
        JsonPatchOperation(op=JsonPatchOp.COPY, path='/c', **{'from': '/a'}),
        JsonPatchOperation(op=JsonPatchOp.MOVE, path='/d', **{'from': '/b/0'}),
    ]
    got = apply_json_patch(doc=doc, patch=patch)
    assert got == {'a': {'x': 1}, 'b': [2], 'c': {'x': 1}, 'd': 1}
    assert doc == {'a': {'x': 1}, 'b': [1, 2]}  # input untouched


def test_json_patch_test_op_mismatch_raises() -> None:
    with pytest.raises(ValueError, match='test failed'):
        apply_json_patch(
            doc={'a': 1},
            patch=[JsonPatchOperation(op=JsonPatchOp.TEST, path='/a', value=2)],
        )


def test_json_patch_lenient_apply_semantics() -> None:
    """Removes of missing members are no-ops; adds into missing parents create them."""
    got = apply_json_patch(
        doc={},
        patch=[
            JsonPatchOperation(op=JsonPatchOp.REMOVE, path='/ghost/deep'),
            JsonPatchOperation(op=JsonPatchOp.ADD, path='/made/up/path', value=7),
            JsonPatchOperation(op=JsonPatchOp.ADD, path='/arr/-', value=1),
        ],
    )
    assert got['made']['up']['path'] == 7
    assert 'ghost' not in got
    # Quirk, pinned deliberately: '-' under a *missing* parent initializes a dict,
    # so the token becomes a key rather than an append. The differ never emits
    # this shape; if this assertion ever fails, apply semantics changed.
    assert got['arr'] == {'-': 1}
