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

"""Fill a Pydantic model from a half-arrived JSON object.

``output_schema=Recipe`` cannot run full validation until every required
field has arrived. Streaming chunks still want ``chunk.output.title`` the
moment that key exists, so missing fields are set to ``None`` and
constraints are skipped. The finished ``ModelResponse.output`` is what
validates into the real type.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from types import UnionType
from typing import Annotated, Any, ForwardRef, TypeGuard, Union, get_args, get_origin, get_type_hints

from pydantic import AliasChoices, BaseModel, RootModel
from pydantic.fields import FieldInfo

SEQUENCE_ORIGINS = (list, tuple, Sequence, MutableSequence)
MAPPING_ORIGINS = (dict, Mapping, MutableMapping)


def is_model(annotation: Any) -> TypeGuard[type[BaseModel]]:  # noqa: ANN401
    return (
        isinstance(annotation, type)
        and issubclass(annotation, BaseModel)
        and annotation is not BaseModel
        and not issubclass(annotation, RootModel)
    )


def unwrap_annotated(*, annotation: Any) -> Any:  # noqa: ANN401
    while get_origin(annotation) is Annotated:
        args = get_args(annotation)
        annotation = args[0] if args else annotation
    return annotation


def lookup_name(*, name: str, schema_type: type[BaseModel]) -> type[BaseModel] | None:
    # list['Post'] stores the name they wrote, not the class. The class
    # lives next to the schema on its module, or is the schema itself.
    if name == schema_type.__name__ and is_model(schema_type):
        return schema_type
    module = sys.modules.get(schema_type.__module__)
    obj = getattr(module, name, None) if module is not None else None
    return obj if is_model(obj) else None


def resolve(*, annotation: Any, schema_type: type[BaseModel]) -> Any:  # noqa: ANN401
    annotation = unwrap_annotated(annotation=annotation)
    if isinstance(annotation, ForwardRef):
        annotation = annotation.__forward_arg__
    if isinstance(annotation, str):
        found = lookup_name(name=annotation, schema_type=schema_type)
        return found if found is not None else annotation
    return annotation


def union_members(*, annotation: Any) -> tuple[Any, ...]:  # noqa: ANN401
    annotation = unwrap_annotated(annotation=annotation)
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        return get_args(annotation)
    return (annotation,)


def field_annotation(*, schema_type: type[BaseModel], name: str, info: FieldInfo) -> Any:  # noqa: ANN401
    annotation = info.annotation
    if annotation is None:
        return object
    try:
        hints = get_type_hints(schema_type, include_extras=True)
    except Exception:
        return annotation
    return hints.get(name, annotation)


def field_keys(*, name: str, info: FieldInfo) -> tuple[str, ...]:
    keys: list[str] = [name]
    if info.alias is not None:
        keys.append(info.alias)
    validation_alias = info.validation_alias
    if isinstance(validation_alias, str):
        keys.append(validation_alias)
    elif isinstance(validation_alias, AliasChoices):
        keys.extend(choice for choice in validation_alias.choices if isinstance(choice, str))
    seen: set[str] = set()
    unique: list[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return tuple(unique)


def lookup(*, data: dict[str, Any], name: str, info: FieldInfo) -> tuple[Any, bool]:
    for key in field_keys(name=name, info=info):
        if key in data:
            return data[key], True
    return None, False


def overlap(*, schema_type: type[BaseModel], data: dict[str, Any]) -> int:
    keys: set[str] = set()
    for name, info in schema_type.model_fields.items():
        keys.update(field_keys(name=name, info=info))
    return len(keys & data.keys())


def pick_model(*, annotation: Any, data: dict[str, Any]) -> type[BaseModel] | None:  # noqa: ANN401
    candidates = [member for member in union_members(annotation=annotation) if is_model(member)]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda model: overlap(schema_type=model, data=data))


def coerce(*, annotation: Any, raw: Any, schema_type: type[BaseModel]) -> Any:  # noqa: ANN401
    if raw is None:
        return None
    annotation = resolve(annotation=annotation, schema_type=schema_type)
    if isinstance(raw, dict):
        model = pick_model(annotation=annotation, data=raw)
        if model is not None:
            return construct_partial(schema_type=model, data=raw)
        origin = get_origin(annotation)
        if origin in MAPPING_ORIGINS:
            args = get_args(annotation)
            if len(args) == 2:
                return {
                    key: coerce(annotation=args[1], raw=value, schema_type=schema_type) for key, value in raw.items()
                }
        return raw
    if isinstance(raw, list):
        origin = get_origin(annotation)
        if origin in SEQUENCE_ORIGINS:
            inners = tuple(arg for arg in get_args(annotation) if arg is not Ellipsis)
            if len(inners) == 1:
                return [coerce(annotation=inners[0], raw=item, schema_type=schema_type) for item in raw]
            if inners:
                return [
                    coerce(
                        annotation=inners[index] if index < len(inners) else inners[-1],
                        raw=item,
                        schema_type=schema_type,
                    )
                    for index, item in enumerate(raw)
                ]
        return raw
    return raw


def construct_partial(*, schema_type: type[BaseModel], data: dict[str, Any]) -> BaseModel:
    """Build ``schema_type`` from streamed JSON without running validation.

    Missing fields are ``None`` so a caller can read ``chunk.output.title``
    as soon as that key exists. Nested objects become the same class with
    the same holes. ``(await sr.response).output`` is the only fully
    validated value.
    """
    values: dict[str, Any] = {}
    for name, info in schema_type.model_fields.items():
        raw, found = lookup(data=data, name=name, info=info)
        if not found:
            values[name] = None
            continue
        values[name] = coerce(
            annotation=field_annotation(schema_type=schema_type, name=name, info=info),
            raw=raw,
            schema_type=schema_type,
        )
    return schema_type.model_construct(**values)
