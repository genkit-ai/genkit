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

"""Compatibility layer for asyncio."""

import sys
import types
import typing
import weakref

# StrEnum - use strenum package for cross-version compatibility
# Note: StrEnum was added to stdlib in Python 3.11, but we use strenum for 3.10 compat
# override decorator - use typing_extensions for consistency across Python versions
# Note: override was added to typing in Python 3.12, but typing_extensions has it for all versions
from typing import overload as overload  # noqa: E402

if sys.version_info >= (3, 11):
    from enum import StrEnum as StrEnum  # noqa: E402
else:
    from strenum import StrEnum as StrEnum  # noqa: E402
import typing_extensions  # noqa: E402
from typing_extensions import override as override  # noqa: E402

# -----------------------------------------------------------------------------
# typing.TypedDict on Python < 3.12
#
# Pydantic raises PydanticUserError for a typing.TypedDict before 3.12: the
# stdlib class lacks introspection Pydantic relies on (generic bases,
# Required/NotRequired). typing_extensions.TypedDict has it. The swap below
# rebuilds the stdlib class as a typing_extensions one with the same name,
# module, docstring, fields, and required keys, so a user who wrote
# `class Cfg(typing.TypedDict)` gets the same validation as on 3.12+.
# -----------------------------------------------------------------------------

_REQUIRED_WRAPPERS: tuple[object, ...] = (
    typing_extensions.Required,
    typing_extensions.NotRequired,
    typing_extensions.ReadOnly,
)
# Field types and names here are only known at runtime; type checkers want
# literals for Union[...], Required[...], and the functional TypedDict form.
_dynamic_typing = typing.cast(typing.Any, typing)
_dynamic_typing_extensions = typing.cast(typing.Any, typing_extensions)
# Weak keys: a TypedDict defined inside a function shouldn't live forever.
# Cached so ModelRequest[Cfg] is the same class every time it's written.
_TYPEDDICT_COPIES: 'weakref.WeakKeyDictionary[type, type]' = weakref.WeakKeyDictionary()


def pydantic_safe_typeddicts(tp: typing.Any) -> typing.Any:  # noqa: ANN401 - any annotation in, same shape out
    """``tp`` with each ``typing.TypedDict`` in it replaced by a ``typing_extensions`` copy.

    Walks generics (``list[Cfg]``, ``Cfg | None``, ``Annotated[Cfg, ...]``)
    and TypedDict fields. Returns ``tp`` itself on Python 3.12+, when nothing
    needed swapping, or for a TypedDict whose hints can't be resolved yet
    (Pydantic then reports it as before).
    """
    if sys.version_info >= (3, 12):
        return tp
    return _swap(tp, frozenset())


def _swap(tp: typing.Any, seen: frozenset[type]) -> typing.Any:  # noqa: ANN401
    if isinstance(tp, type) and type(tp).__module__ == 'typing' and typing.is_typeddict(tp):
        return _typeddict_copy(tp, seen)
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is None or not args or origin is typing.Literal:
        return tp
    if origin is typing_extensions.Annotated:
        base = _swap(args[0], seen)
        if base is args[0]:
            return tp
        return _dynamic_typing_extensions.Annotated[(base, *args[1:])]
    new_args = tuple(_swap(a, seen) for a in args)
    if all(n is o for n, o in zip(new_args, args, strict=True)):
        return tp
    if origin is typing.Union or origin is types.UnionType:
        return _dynamic_typing.Union[new_args]
    if isinstance(tp, types.GenericAlias):
        return types.GenericAlias(origin, new_args)
    copy_with = getattr(tp, 'copy_with', None)
    return copy_with(new_args) if copy_with is not None else tp


def _strip_required(tp: typing.Any) -> typing.Any:  # noqa: ANN401
    while typing.get_origin(tp) in _REQUIRED_WRAPPERS:
        tp = typing.get_args(tp)[0]
    return tp


def _typeddict_copy(td: type, seen: frozenset[type]) -> type:
    cached = _TYPEDDICT_COPIES.get(td)
    if cached is not None:
        return cached
    if td in seen:
        # Self-referencing TypedDict: leave the inner reference alone.
        return td
    try:
        hints = typing.get_type_hints(td, include_extras=True)
    except Exception:  # noqa: BLE001 - unresolved forward ref; let Pydantic report it
        return td
    required: frozenset[str] = getattr(td, '__required_keys__', frozenset())
    inner = seen | {td}
    fields: dict[str, typing.Any] = {}
    for key, hint in hints.items():
        value = _swap(_strip_required(hint), inner)
        wrapper = _dynamic_typing_extensions.Required if key in required else _dynamic_typing_extensions.NotRequired
        fields[key] = wrapper[value]
    copy = _dynamic_typing_extensions.TypedDict(td.__name__, fields)
    copy.__module__ = td.__module__
    copy.__qualname__ = td.__qualname__
    copy.__doc__ = td.__doc__
    _TYPEDDICT_COPIES[td] = copy
    return copy
