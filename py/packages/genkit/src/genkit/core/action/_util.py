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

"""Action utility module for defining and managing action utilities."""

import inspect
from collections.abc import Mapping
from typing import Any


def noop_streaming_callback(_chunk: Any) -> None:  # noqa: ANN401
    """A no-op streaming callback.

    This callback does nothing and is used when no streaming is desired.

    Args:
        _chunk: The chunk (unused, callback is a no-op).

    Returns:
        None.
    """
    pass


def parse_plugin_name_from_action_name(name: str) -> str | None:
    """Parses the plugin name from an action name.

    As per convention, the plugin name is optional. If present, it's the first
    part of the action name, separated by a forward slash: `pluginname/*`.

    Args:
        name: The action name string.

    Returns:
        The plugin name, or None if no plugin name is found in the action name.
    """
    tokens = name.split('/')
    if len(tokens) > 1:
        return tokens[0]
    return None


def extract_action_args_and_types(
    input_spec: inspect.FullArgSpec,
    annotations: Mapping[str, Any] | None = None,
) -> tuple[list[str], list[Any]]:
    """Extracts relevant argument names and types from a function's FullArgSpec.

    Specifically handles the case where the first argument might be 'self'
    (for methods) and determines the type hint for each argument.

    Args:
        input_spec: The FullArgSpec object obtained from
            inspect.getfullargspec().
        annotations: Optional override for type annotations. If not provided,
            uses input_spec.annotations.

    Returns:
        A tuple containing:
            - A list of argument names (potentially excluding 'self').
            - A list of corresponding argument types (using Any if no
            annotation).
    """
    arg_types = []
    action_args = input_spec.args.copy()
    resolved_annotations = annotations or input_spec.annotations

    # Special case when using a method as an action, we ignore first "self"
    # arg. (Note: The original condition `len(action_args) <= 3` is preserved
    # from the source snippet).
    if len(action_args) > 0 and len(action_args) <= 3 and action_args[0] == 'self':
        del action_args[0]

    for arg in action_args:
        arg_types.append(resolved_annotations.get(arg, Any))

    return action_args, arg_types
