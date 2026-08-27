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
#
# SPDX-License-Identifier: Apache-2.0

"""Synthesize an all-optional Pydantic model for streaming chunks.

``output_schema=Recipe`` cannot construct a real ``Recipe`` until every
required field has arrived. Streaming chunks get a sibling class
(``RecipePartial``) where every field is ``T | None = None``. The final
``ModelResponse.output`` still validates into the original type.

The values below are runtime type objects being assembled dynamically, so
they are deliberately typed ``Any``: the type expressions this module
builds (``Union[X, None]``, ``list[X]``) only exist at runtime.
"""

from __future__ import annotations

from types import UnionType
from typing import Annotated, Any, ForwardRef, TypeGuard, Union, cast, get_args, get_origin

from pydantic import BaseModel, ConfigDict, create_model

_partials: dict[type[BaseModel], type[BaseModel]] = {}


def _is_base_model(annotation: Any) -> TypeGuard[type[BaseModel]]:  # noqa: ANN401
    return isinstance(annotation, type) and issubclass(annotation, BaseModel) and annotation is not BaseModel


def _referenced_models(schema_type: type[BaseModel]) -> dict[str, type[BaseModel]]:
    """Collect model classes reachable from ``schema_type``'s compiled schema.

    A model rebuilt inside a function body can keep referring to sibling
    models by bare name in its field annotations; the actual classes are
    only reachable through the validator pydantic already compiled, so we
    recover them from there to resolve those names.
    """
    found: dict[str, type[BaseModel]] = {}

    def walk(node: Any) -> None:  # noqa: ANN401
        if isinstance(node, dict):
            cls = node.get('cls')
            if _is_base_model(cls):
                found.setdefault(cls.__name__, cls)
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(getattr(schema_type, '__pydantic_core_schema__', None))
    return found


def _rewrite_annotation(
    annotation: Any,  # noqa: ANN401
    building: dict[type[BaseModel], type[BaseModel] | None],
    refs: dict[str, type[BaseModel]],
) -> Any:  # noqa: ANN401
    """Rewrite nested model types into their synthesized partials."""
    if isinstance(annotation, (str, ForwardRef)):
        name = annotation if isinstance(annotation, str) else annotation.__forward_arg__
        target = refs.get(name)
        return _partial_for(target, building) if target is not None else annotation
    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        return _rewrite_annotation(args[0], building, refs) if args else annotation
    if origin in (Union, UnionType):
        # Union is rebuilt member by member so models nested anywhere in it
        # get partial treatment; typing.Union also tolerates forward refs,
        # which the | operator rejects on Python 3.10.
        rewritten = cast('Any', tuple(_rewrite_annotation(arg, building, refs) for arg in get_args(annotation)))
        return Union[rewritten]  # noqa: UP007
    if origin in (list, set, frozenset):
        args = get_args(annotation)
        if not args:
            return annotation
        inner = _rewrite_annotation(args[0], building, refs)
        if origin is list:
            return list[inner]
        if origin is set:
            return set[inner]
        return frozenset[inner]
    if origin is dict:
        args = get_args(annotation)
        if len(args) != 2:
            return annotation
        key_annotation = args[0]
        value_annotation = _rewrite_annotation(args[1], building, refs)
        return dict[key_annotation, value_annotation]
    if origin is tuple:
        args = get_args(annotation)
        if not args:
            return annotation
        rewritten = cast('Any', tuple(_rewrite_annotation(arg, building, refs) for arg in args))
        return tuple[rewritten]
    if _is_base_model(annotation):
        return _partial_for(annotation, building)
    return annotation


def _partial_for(
    schema_type: type[BaseModel],
    building: dict[type[BaseModel], type[BaseModel] | None],
) -> Any:  # noqa: ANN401
    """Return the partial for ``schema_type``, tolerating recursive schemas."""
    cached = _partials.get(schema_type)
    if cached is not None:
        return cached
    if schema_type in building:
        started = building[schema_type]
        if started is not None:
            return started
        # We looped back into a model whose partial is still being assembled
        # (e.g. a tree node whose children are nodes). Reference it by name
        # now; partial_model resolves the name once the whole graph exists.
        return ForwardRef(f'{schema_type.__name__}Partial')
    building[schema_type] = None
    refs = _referenced_models(schema_type)
    fields: dict[str, Any] = {}
    for name, info in schema_type.model_fields.items():
        annotation: Any = (
            _rewrite_annotation(info.annotation, building, refs) if info.annotation is not None else object
        )
        # Only the type is carried over. Constraints (Field(gt=0), Annotated
        # metadata) and validators are dropped on purpose: a half-streamed
        # value legitimately violates them, and enforcing them mid-stream
        # would reject data the user can already display. The final response
        # validates against the real model with everything intact.
        fields[name] = (Union[annotation, None], None)  # noqa: UP007
    model = create_model(
        f'{schema_type.__name__}Partial',
        __module__=schema_type.__module__,
        __config__=ConfigDict(extra='ignore', populate_by_name=True),
        **fields,
    )
    building[schema_type] = model
    return model


def partial_model(schema_type: type[BaseModel]) -> type[BaseModel]:
    """Return a cached sibling model with every field optional.

    The result is not a subclass of ``schema_type``. ``isinstance(chunk.output,
    Recipe)`` is false; ``type(chunk.output).__name__`` is ``RecipePartial``.
    """
    cached = _partials.get(schema_type)
    if cached is not None:
        return cached
    building: dict[type[BaseModel], type[BaseModel] | None] = {}
    result = _partial_for(schema_type, building)
    built = {source: partial for source, partial in building.items() if partial is not None}
    namespace: dict[str, Any] = {partial.__name__: partial for partial in built.values()}
    for partial in built.values():
        partial.model_rebuild(force=True, _types_namespace=namespace)
    _partials.update(built)
    return result
