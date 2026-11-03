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

"""Fill a Pydantic model from a streaming JSON object.

In a JSON stream, every key before the last key of an open object and every
item before the last item of an open array is already closed on the wire, so
those subtrees validate against their real Pydantic types. Only the trailing
frontier still arriving is built with ``model_construct`` / raw values, missing
optional fields get their defaults, and missing required fields are ``None``.
``(await stream.response).output`` remains the only fully validated value.
"""

from __future__ import annotations

import sys
from collections.abc import Hashable, Mapping, MutableMapping, MutableSequence, Sequence
from contextlib import suppress
from enum import Enum
from functools import lru_cache
from types import UnionType
from typing import Annotated, Any, ForwardRef, Literal, TypeGuard, Union, get_args, get_origin, get_type_hints

from pydantic import AliasChoices, BaseModel, RootModel, TypeAdapter
from pydantic.fields import FieldInfo

SEQUENCE_ORIGINS = (list, tuple, Sequence, MutableSequence)
MAPPING_ORIGINS = (dict, Mapping, MutableMapping)
UNION_ORIGINS = (Union, UnionType)
_UNRESOLVED = object()


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
        if not args:
            break
        annotation = args[0]
    return annotation


def extract_discriminator(*, annotation: Any, info: FieldInfo | None = None) -> str | None:  # noqa: ANN401
    if info is not None and isinstance(info.discriminator, str):
        return info.discriminator
    cur = annotation
    while get_origin(cur) is Annotated:
        args = get_args(cur)
        if not args:
            break
        for meta in args[1:]:
            disc = getattr(meta, 'discriminator', None)
            if isinstance(disc, str):
                return disc
        cur = args[0]
    if get_origin(cur) in UNION_ORIGINS:
        for arg in get_args(cur):
            found = extract_discriminator(annotation=arg)
            if found is not None:
                return found
    return None


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


def union_members(*, annotation: Any, schema_type: type[BaseModel]) -> tuple[Any, ...]:  # noqa: ANN401
    annotation = resolve(annotation=annotation, schema_type=schema_type)
    origin = get_origin(annotation)
    if origin in UNION_ORIGINS:
        members: list[Any] = []
        for arg in get_args(annotation):
            members.extend(union_members(annotation=arg, schema_type=schema_type))
        return tuple(members)
    return (annotation,)


def unwrap_optional_container(*, annotation: Any, schema_type: type[BaseModel]) -> Any:  # noqa: ANN401
    annotation = resolve(annotation=annotation, schema_type=schema_type)
    if get_origin(annotation) in UNION_ORIGINS:
        non_none = [
            arg for arg in union_members(annotation=annotation, schema_type=schema_type) if arg is not type(None)
        ]
        if len(non_none) == 1:
            return non_none[0]
    return annotation


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


def lookup(*, data: dict[str, Any], name: str, info: FieldInfo) -> tuple[Any, str | None]:
    for key in field_keys(name=name, info=info):
        if key in data:
            return data[key], key
    return None, None


def default_value(*, info: FieldInfo, validated_data: dict[str, Any]) -> Any:  # noqa: ANN401
    if info.is_required():
        return None
    try:
        return info.get_default(call_default_factory=True, validated_data=validated_data)
    except Exception:
        return None


def literal_values(*, annotation: Any) -> set[Any]:  # noqa: ANN401
    annotation = unwrap_annotated(annotation=annotation)
    origin = get_origin(annotation)
    if origin in UNION_ORIGINS:
        out: set[Any] = set()
        for arg in get_args(annotation):
            out.update(literal_values(annotation=arg))
        return out
    if origin is Literal:
        values: set[Any] = set()
        for arg in get_args(annotation):
            values.add(arg)
            if isinstance(arg, Enum):
                values.add(arg.value)
        return values
    return set()


def pick_discriminated_model(
    *,
    candidates: Sequence[type[BaseModel]],
    data: dict[str, Any],
    discriminator: str,
    closed: bool,
) -> tuple[type[BaseModel] | None, bool]:
    """Match ``data`` to a discriminated union member.

    Returns ``(model, True)`` when literal tags are declared on the candidates
    (``model`` is ``None`` while the tag is missing or still a half-streamed
    prefix), or ``(None, False)`` when candidates carry no literal tags.
    """
    entries: list[tuple[type[BaseModel], str, FieldInfo, set[Any]]] = []
    all_tags: set[Any] = set()
    for candidate in candidates:
        for fname, finfo in candidate.model_fields.items():
            if discriminator in field_keys(name=fname, info=finfo):
                tags = literal_values(
                    annotation=field_annotation(schema_type=candidate, name=fname, info=finfo),
                )
                entries.append((candidate, fname, finfo, tags))
                all_tags.update(tags)
                break
    if not all_tags:
        return None, False

    open_key = None if closed or not data else next(reversed(data))
    for candidate, fname, finfo, tags in entries:
        raw_tag, matched_key = lookup(data=data, name=fname, info=finfo)
        # A dict or list tag can't match a Literal; treat it as not yet known.
        if matched_key is None or not isinstance(raw_tag, Hashable) or raw_tag not in tags:
            continue
        if (
            matched_key == open_key
            and isinstance(raw_tag, str)
            and any(isinstance(other, str) and other != raw_tag and other.startswith(raw_tag) for other in all_tags)
        ):
            return None, True
        return candidate, True
    return None, True


def overlap(*, schema_type: type[BaseModel], data: dict[str, Any]) -> int:
    keys: set[str] = set()
    for name, info in schema_type.model_fields.items():
        keys.update(field_keys(name=name, info=info))
    return len(keys & data.keys())


def pick_model(*, annotation: Any, data: dict[str, Any], schema_type: type[BaseModel]) -> type[BaseModel] | None:  # noqa: ANN401
    candidates = [
        member for member in union_members(annotation=annotation, schema_type=schema_type) if is_model(member)
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda model: overlap(schema_type=model, data=data))


@lru_cache(maxsize=256)
def _cached_adapter(annotation: Any, module_name: str) -> TypeAdapter[Any]:  # noqa: ANN401
    return TypeAdapter(annotation, module=module_name)


def validate_closed_value(*, annotation: Any, raw: Any, schema_type: type[BaseModel]) -> tuple[Any, bool]:  # noqa: ANN401
    try:
        try:
            adapter = _cached_adapter(annotation, schema_type.__module__)
        except TypeError:
            adapter = TypeAdapter(annotation, module=schema_type.__module__)
        return adapter.validate_python(raw), True
    except Exception:
        return None, False


def validate_closed_field(
    *,
    schema_type: type[BaseModel],
    name: str,
    annotation: Any,  # noqa: ANN401
    raw: Any,  # noqa: ANN401
    values: dict[str, Any],
) -> tuple[Any, bool]:
    try:
        shell = object.__new__(schema_type)
        field_map = dict.fromkeys(schema_type.model_fields)
        field_map.update(values)
        object.__setattr__(shell, '__dict__', field_map)
        object.__setattr__(shell, '__pydantic_fields_set__', set(values))
        extra = {} if schema_type.model_config.get('extra') == 'allow' else None
        object.__setattr__(shell, '__pydantic_extra__', extra)
        object.__setattr__(shell, '__pydantic_private__', None)
        validated = schema_type.__pydantic_validator__.validate_assignment(shell, name, raw)
        return getattr(validated, name), True
    except Exception:
        return validate_closed_value(annotation=annotation, raw=raw, schema_type=schema_type)


def accepts_open_scalar(*, annotation: Any, raw: Any, schema_type: type[BaseModel]) -> bool:  # noqa: ANN401
    members = [m for m in union_members(annotation=annotation, schema_type=schema_type) if m is not type(None)]
    if not members or any(m in (Any, object) for m in members):
        return True
    tags = literal_values(annotation=annotation)
    if tags and raw in tags:
        return not (
            isinstance(raw, str)
            and any(isinstance(other, str) and other != raw and other.startswith(raw) for other in tags)
        )
    if isinstance(raw, str):
        return any(m is str for m in members)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return not all((isinstance(m, type) and issubclass(m, Enum)) or get_origin(m) is Literal for m in members)
    return True


def coerce(
    *,
    annotation: Any,  # noqa: ANN401
    raw: Any,  # noqa: ANN401
    schema_type: type[BaseModel],
    closed: bool = False,
    discriminator: str | None = None,
) -> Any:  # noqa: ANN401
    if raw is None:
        return None

    disc = discriminator or extract_discriminator(annotation=annotation)
    if isinstance(raw, dict) and disc is not None:
        candidates = [
            member for member in union_members(annotation=annotation, schema_type=schema_type) if is_model(member)
        ]
        if candidates:
            matched, handled = pick_discriminated_model(
                candidates=candidates,
                data=raw,
                discriminator=disc,
                closed=closed,
            )
            if handled:
                if matched is None:
                    return _UNRESOLVED
                return construct_partial(schema_type=matched, data=raw, closed=closed)

    if closed:
        validated, ok = validate_closed_value(annotation=annotation, raw=raw, schema_type=schema_type)
        if ok:
            return validated

    container_annotation = unwrap_optional_container(annotation=annotation, schema_type=schema_type)
    if isinstance(raw, dict):
        model = pick_model(annotation=annotation, data=raw, schema_type=schema_type)
        if model is not None:
            return construct_partial(schema_type=model, data=raw, closed=closed)
        origin = get_origin(container_annotation)
        if origin in MAPPING_ORIGINS:
            args = get_args(container_annotation)
            if len(args) == 2:
                open_map_key = None if closed or not raw else next(reversed(raw))
                mapped: dict[Any, Any] = {}
                for key, value in raw.items():
                    coerced_key = coerce(annotation=args[0], raw=key, schema_type=schema_type, closed=True)
                    coerced_val = coerce(
                        annotation=args[1],
                        raw=value,
                        schema_type=schema_type,
                        closed=(closed or key != open_map_key),
                    )
                    if coerced_val is not _UNRESOLVED:
                        mapped[coerced_key] = coerced_val
                return mapped
        return raw
    if isinstance(raw, list):
        origin = get_origin(container_annotation)
        if origin in SEQUENCE_ORIGINS:
            inners = tuple(arg for arg in get_args(container_annotation) if arg is not Ellipsis)
            if inners:
                last_idx = len(raw) - 1
                items: list[Any] = []
                for index, item in enumerate(raw):
                    item_annotation = (
                        inners[0] if len(inners) == 1 else (inners[index] if index < len(inners) else inners[-1])
                    )
                    coerced_item = coerce(
                        annotation=item_annotation,
                        raw=item,
                        schema_type=schema_type,
                        closed=(closed or index < last_idx),
                    )
                    if coerced_item is not _UNRESOLVED:
                        items.append(coerced_item)
                return tuple(items) if origin is tuple else items
        return raw
    if not closed and not accepts_open_scalar(annotation=annotation, raw=raw, schema_type=schema_type):
        return _UNRESOLVED
    return raw


def construct_partial(*, schema_type: type[BaseModel], data: dict[str, Any], closed: bool = False) -> BaseModel:
    """Build ``schema_type`` from streamed JSON, validating closed subtrees.

    Optional fields default to their declared defaults and missing required
    fields are ``None``. Keys and list items before the trailing open frontier
    are validated against their real Pydantic types (falling back to
    ``model_construct`` if a model constraint fails mid-stream).
    ``(await stream.response).output`` remains the only fully validated value.
    """
    if closed:
        with suppress(Exception):
            return schema_type.model_validate(data)

    open_key = None if closed or not data else next(reversed(data))
    values: dict[str, Any] = {}
    for name, info in schema_type.model_fields.items():
        raw, matched_key = lookup(data=data, name=name, info=info)
        if matched_key is None:
            values[name] = default_value(info=info, validated_data=values)
            continue

        annotation = field_annotation(schema_type=schema_type, name=name, info=info)
        field_closed = closed or (matched_key != open_key)
        disc = extract_discriminator(annotation=annotation, info=info)

        if field_closed and raw is not None and not (isinstance(raw, dict) and disc is not None):
            validated_field, ok = validate_closed_field(
                schema_type=schema_type,
                name=name,
                annotation=annotation,
                raw=raw,
                values=values,
            )
            if ok:
                values[name] = validated_field
                continue

        coerced = coerce(
            annotation=annotation,
            raw=raw,
            schema_type=schema_type,
            closed=field_closed,
            discriminator=disc,
        )
        values[name] = default_value(info=info, validated_data=values) if coerced is _UNRESOLVED else coerced
    return schema_type.model_construct(**values)
