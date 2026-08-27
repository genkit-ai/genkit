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
builds (``X | None``, ``list[X]``) only exist at runtime.
"""

from __future__ import annotations

from functools import lru_cache
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, create_model


def _is_base_model(annotation: Any) -> bool:  # noqa: ANN401
    return isinstance(annotation, type) and issubclass(annotation, BaseModel) and annotation is not BaseModel


def _rewrite_annotation(annotation: Any) -> Any:  # noqa: ANN401
    """Rewrite nested model types into their synthesized partials."""
    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        return _rewrite_annotation(args[0]) if args else annotation
    if origin in (Union, UnionType):
        rewritten = [_rewrite_annotation(arg) if arg is not type(None) else arg for arg in get_args(annotation)]
        non_none = [arg for arg in rewritten if arg is not type(None)]
        if len(non_none) == 1:
            return non_none[0] | None
        return annotation
    if origin in (list, set, frozenset):
        args = get_args(annotation)
        if not args:
            return annotation
        inner = _rewrite_annotation(args[0])
        if origin is list:
            return list[inner]
        if origin is set:
            return set[inner]
        return frozenset[inner]
    if _is_base_model(annotation):
        return partial_model(annotation)
    return annotation


@lru_cache(maxsize=128)
def partial_model(schema_type: type[BaseModel]) -> type[BaseModel]:
    """Return a cached sibling model with every field optional.

    The result is not a subclass of ``schema_type``. ``isinstance(chunk.output,
    Recipe)`` is false; ``type(chunk.output).__name__`` is ``RecipePartial``.
    """
    fields: dict[str, Any] = {}
    for name, info in schema_type.model_fields.items():
        annotation: Any = _rewrite_annotation(info.annotation) if info.annotation is not None else object
        fields[name] = (annotation | None, None)
    return create_model(
        f'{schema_type.__name__}Partial',
        __module__=schema_type.__module__,
        __config__=ConfigDict(extra='ignore', populate_by_name=True),
        **fields,
    )
