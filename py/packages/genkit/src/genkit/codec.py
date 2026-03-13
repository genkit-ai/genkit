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

"""Encoding and decoding utilities for the Genkit framework.

This module provides functions for serializing Genkit objects to dictionaries
and JSON strings, with special handling for Pydantic models and binary data.

Overview:
    Genkit uses Pydantic models extensively for type safety. This module
    provides utilities to convert these models to dictionaries and JSON
    strings for serialization, API responses, and debugging.

Key Functions:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Function          │ Purpose                                             │
    ├───────────────────┼─────────────────────────────────────────────────────┤
    │ dump_dict()       │ Convert Pydantic model to dict (exclude_none=True)  │
    │ dump_json()       │ Convert object to JSON string with alias support    │
    │ default_serializer│ Fallback serializer for non-standard types (bytes)  │
    └───────────────────┴─────────────────────────────────────────────────────┘

Example:
    Converting Pydantic models:

    ```python
    from genkit.codec import dump_dict, dump_json
    from genkit.types import Message, Part, TextPart

    msg = Message(role='user', content=[Part(root=TextPart(text='Hello'))])

    # Convert to dict (excludes None values, uses aliases)
    msg_dict = dump_dict(msg)
    # {'role': 'user', 'content': [{'text': 'Hello'}]}

    # Convert to JSON string
    msg_json = dump_json(msg, indent=2)
    # '{"role": "user", "content": [{"text": "Hello"}]}'
    ```

Caveats:
    - Pydantic models use by_alias=True for JSON Schema compatibility
    - None values are excluded from output (exclude_none=True)
    - Binary data (bytes) is base64-encoded when serializing

See Also:
    - Pydantic documentation: https://docs.pydantic.dev/
    - genkit.core.typing: Core type definitions using Pydantic
"""

import base64
import json
from collections.abc import Callable

from pydantic import BaseModel


def dump_dict(obj: object, fallback: Callable[[object], object] | None = None) -> object:
    """Converts an object or Pydantic to a dictionary.

    If the input object is a Pydantic BaseModel, it returns a dictionary
    representation of the model, excluding fields with `None` values and using
    aliases for field names.  For any other object type, it returns the object
    unchanged.

    Args:
        obj: The object to potentially convert to a dictionary.
        fallback: A function to call when an unknown value is encountered. If not provided, error is raised.

    Returns:
        A dictionary if the input is a Pydantic BaseModel, otherwise the
        original object.

    Raises:
        ValueError: If a circular reference is detected.
    """

    def _dump(o: object, seen: set[int]) -> object:
        if isinstance(o, (list, dict)):
            obj_id = id(o)
            if obj_id in seen:
                raise ValueError('Circular reference detected')
            seen.add(obj_id)

        try:
            if isinstance(o, BaseModel):
                return o.model_dump(exclude_none=True, by_alias=True, fallback=fallback)
            elif isinstance(o, list):
                return [_dump(i, seen) for i in o]
            elif isinstance(o, dict):
                return {k: _dump(v, seen) for k, v in o.items()}
            else:
                return o
        finally:
            if isinstance(o, (list, dict)):
                seen.remove(id(o))

    return _dump(obj, set())


def default_serializer(obj: object) -> object:
    """Default serializer for objects not handled by json.dumps.

    Args:
        obj: The object to serialize.

    Returns:
        A serializable representation of the object.
    """
    if isinstance(obj, bytes):
        try:
            return base64.b64encode(obj).decode('utf-8')
        except Exception:
            return '<bytes>'
    return str(obj)


def dump_json(
    obj: object,
    indent: int | None = None,
    fallback: Callable[[object], object] | None = None,
) -> str:
    """Dumps an object to a JSON string.

    If the object is a Pydantic BaseModel, it will be dumped using the
    model_dump_json method using the by_alias flag set to True.  Otherwise, the
    object will be dumped using the json.dumps method.

    Args:
        obj: The object to dump.
        indent: The indentation level for the JSON string.
        fallback: A function to call when an unknown value is encountered. If not provided, error is raised.

    Returns:
        A JSON string.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump_json(by_alias=True, exclude_none=True, indent=indent, fallback=fallback)
    else:
        return json.dumps(obj, indent=indent, default=fallback or default_serializer)
