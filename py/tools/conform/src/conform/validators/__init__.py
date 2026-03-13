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

"""Validator registry for model conformance tests.

Each validator is a callable that inspects a ``GenerateResponse`` and
raises ``ValidationError`` if the response does not meet expectations.

Validators are auto-registered via the :func:`register` decorator and
looked up by name at test time (e.g. ``has-tool-request:weather``).

Canonical JS source (keep in sync):
    genkit-tools/cli/src/commands/dev-test-model.ts — ``VALIDATORS``
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

__all__ = [
    'VALIDATORS',
    'ValidationError',
    'Validator',
    'get_validator',
    'register',
]


class ValidationError(Exception):
    """Raised when a validator check fails."""


@runtime_checkable
class Validator(Protocol):
    """Protocol for conformance test validators.

    A validator inspects the model's response (and optionally streaming
    chunks) and raises :class:`ValidationError` on failure.
    """

    def __call__(
        self,
        response: dict[str, Any],
        arg: str | None = None,
        chunks: list[dict[str, Any]] | None = None,
    ) -> None:
        """Validate the response.

        Args:
            response: The parsed ``GenerateResponse`` from the model.
            arg: Optional argument (e.g. the expected tool name).
            chunks: Streaming chunks (only for stream-* validators).

        Raises:
            ValidationError: If the response fails validation.
        """
        ...


# Global validator registry, keyed by validator name.
VALIDATORS: dict[str, Validator] = {}


def register(name: str) -> Callable[[Validator], Validator]:
    """Decorator to register a validator function.

    Usage::

        @register('has-tool-request')
        def has_tool_request(response, arg=None, chunks=None): ...
    """

    def decorator(fn: Validator) -> Validator:
        VALIDATORS[name] = fn
        return fn

    return decorator


def get_validator(name: str) -> Validator:
    """Look up a validator by name.

    Raises:
        KeyError: If the validator is not registered.
    """
    if name not in VALIDATORS:
        raise KeyError(f'Unknown validator: {name!r}')
    return VALIDATORS[name]


# Import submodules to trigger @register decorators.
# These MUST come AFTER the registry is defined.
import conform.validators.json  # noqa: E402, F401
import conform.validators.media  # noqa: E402, F401
import conform.validators.reasoning  # noqa: E402, F401
import conform.validators.streaming  # noqa: E402, F401
import conform.validators.text  # noqa: E402, F401
import conform.validators.tool  # noqa: E402, F401
