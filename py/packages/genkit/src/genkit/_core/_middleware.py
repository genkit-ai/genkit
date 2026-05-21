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

"""Core middleware abstractions for the Genkit generate pipeline."""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from genkit._core._action import Action
from genkit._core._logger import get_logger
from genkit._core._model import (
    GenerateActionOptions,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
)
from genkit._core._protocols import RegistryLike
from genkit._core._typing import MiddlewareDescData, Part, ToolRequestPart

logger = get_logger(__name__)


class MiddlewareValidationResult(NamedTuple):
    errored: bool
    error_message: str


_FORBIDDEN_IN_MIDDLEWARE_KEY_SEGMENT = re.compile(r'[\x00-\x1f/\\:]|\s')


def _validate_middleware_key_segment(name: str) -> MiddlewareValidationResult:
    """Validate if ``name`` is usable as a middleware registry key.

    * no ``/`` (that shape is reserved for models and other actions);
    * no whitespace, ``:``, backslashes, or control characters that
      would break registry keys or the Dev UI.

    Args:
        name: Proposed name.

    Returns:
        A MiddlewareValidationResult.
    """
    if not name or not name.strip():
        return MiddlewareValidationResult(
            errored=True,
            error_message='must be a non-empty string (not whitespace-only).',
        )
    if name != name.strip():
        return MiddlewareValidationResult(
            errored=True,
            error_message='must not have leading or trailing whitespace.',
        )
    if _FORBIDDEN_IN_MIDDLEWARE_KEY_SEGMENT.search(name):
        return MiddlewareValidationResult(
            errored=True,
            error_message=(
                'must be one path-free token: no whitespace, "/", ":", '
                r'backslashes, or control characters (for example "myorg_logging_mw").'
            ),
        )
    return MiddlewareValidationResult(errored=False, error_message='')


class MultipartToolResponse(BaseModel):
    """A tool result with optional rich content attachments.

    Return from ``wrap_tool`` to send structured output alongside extra
    parts — images, file contents, error details — that the model can
    reason about.

    The engine serializes both fields into a single ``ToolResponsePart`` on
    the wire: ``output`` becomes ``ToolResponse.output`` and ``content``
    becomes ``ToolResponse.content``. Packing them together preserves the
    LLM's one-response-per-call contract while still letting middleware
    attach rich context.

    Fields:
        output: Structured result returned to the model. May be ``None``
            when the tool only produces rich content parts.
        content: Extra ``Part`` objects (images, files, metadata) bundled
            alongside ``output`` in the same ``ToolResponsePart``.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    output: Any = None
    content: list[Part] = Field(default_factory=list)


class GenerateHookParams(BaseModel):
    """Params passed to the ``wrap_generate`` hook.

    Covers one full iteration of the tool loop: a model call plus optional tool
    resolution. ``message_index`` and ``on_chunk`` support streaming.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    options: GenerateActionOptions
    request: ModelRequest
    iteration: int
    message_index: int = 0
    on_chunk: Callable[[ModelResponseChunk], None] | None = None


class ModelHookParams(BaseModel):
    """Params passed to the ``wrap_model`` hook (each raw model API call)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    request: ModelRequest
    on_chunk: Callable[[ModelResponseChunk], None] | None = None
    context: dict[str, object] = Field(default_factory=dict)


class ToolHookParams(BaseModel):
    """Params passed to the ``wrap_tool`` hook (each individual tool execution)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    tool_request_part: ToolRequestPart
    tool: Action


@dataclass
class MiddlewareContext:
    """Per-``generate()`` services shared by every middleware in ``use=[...]``.

    ``registry`` is the call-scoped child registry (resolve tools, register
    call-local actions). ``enqueue_parts`` queues extra user message parts for
    the next turn.
    """

    registry: RegistryLike
    enqueue_parts: Callable[[list[Part]], None]


class BaseMiddleware(BaseModel):
    """BaseMiddleware is the base class that you extend to create a middleware.
    A middleware is defined by its custom configuration (backed by Pydantic),
    and a set of hooks to inject logic into the generate pipeline. 

    The base middleware has no custom configuration and noop hooks. The hooks
    that are not overriden are still called by the engine when the middleware
    is invoked. 

    To author a middleware,

    1. Subclass `BaseMiddleware` and add pydantic fields for config.

    2. Override the ``wrap_generate`` / ``wrap_model`` / ``wrap_tool`` hooks.

    3. Wrap your subclass with the ``@ai.middleware`` decorator to make it available
    in your local Dev UI.

    Example:

        @ai.middleware(name='logger')
        class Logger(BaseMiddleware):
            prefix: str = '[trace]'

            async def wrap_model(self, params, next_fn, ctx):
                t = time.monotonic()
                resp = await next_fn(params)
                log(f'{self.prefix} {time.monotonic() - t:.3f}s')
                return resp
    """

    model_config = ConfigDict(extra='forbid')

    def tools(self, ctx: MiddlewareContext) -> list[Action]:
        """Return additional tools to expose to the model for this generate call."""
        return []

    async def wrap_generate(
        self,
        params: GenerateHookParams,
        next_fn: Callable[[GenerateHookParams], Awaitable[ModelResponse]],
        ctx: MiddlewareContext,
    ) -> ModelResponse:
        """Wrap each iteration of the tool loop (model call + optional tool resolution)."""
        return await next_fn(params)

    async def wrap_model(
        self,
        params: ModelHookParams,
        next_fn: Callable[[ModelHookParams], Awaitable[ModelResponse]],
        ctx: MiddlewareContext,
    ) -> ModelResponse:
        """Wrap each model API call."""
        return await next_fn(params)

    async def wrap_tool(
        self,
        params: ToolHookParams,
        next_fn: Callable[[ToolHookParams], Awaitable[MultipartToolResponse]],
        ctx: MiddlewareContext,
    ) -> MultipartToolResponse:
        """Wrap each tool execution.

        Return a `MultipartToolResponse` to forward (or substitute) the
        tool's result.  Raise `Interrupt(metadata)` to halt this tool call
        and surface an interrupt to the caller.
        """
        return await next_fn(params)


class MiddlewareDesc(MiddlewareDescData):
    """Registered middleware descriptor: wire shape + the class to instantiate.

    Inherits wire fields from auto-generated class in _typing.py. Holds the
    actually BaseMiddleware class in a private attribute ``_cls``.
    """

    # `arbitrary_types_allowed` lets the `PrivateAttr` carry an opaque class
    # reference; parent's `alias_generator` and `extra='forbid'` settings are inherited.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # The class the registry instantiates per `generate()` call.
    _cls: type[BaseMiddleware] = PrivateAttr()

    def __init__(
        self,
        *,
        cls: type[BaseMiddleware],
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        res = _validate_middleware_key_segment(name)
        if res.errored:
            raise ValueError(f'MiddlewareDesc name {res.error_message}')
        # Fall back to the class docstring so authors get a Dev-UI-visible
        # description "for free" — same pattern actions/tools use for their
        # function docstrings. ``inspect.cleandoc`` strips the indentation
        # pydantic-style class docstrings pick up from being inside a class
        # body so the Dev UI doesn't render that leading whitespace.
        if description is None and cls.__doc__:
            description = inspect.cleandoc(cls.__doc__)
        super().__init__(
            name=name,
            description=description,
            config_schema=_derive_config_schema(cls),
            metadata=metadata,
        )
        self._cls = cls

    def __call__(self, config: dict[str, Any] | None = None) -> BaseMiddleware:
        """Return a fresh BaseMiddleware instance for this generate() call."""
        return self._cls(**(config or {}))


def _derive_config_schema(cls: type[BaseMiddleware]) -> dict[str, Any]:
    """Build a JSON Schema describing a middleware's user-facing config fields.

    The Dev UI renders a config form for each registered middleware from this
    schema.
    """
    if cls is BaseMiddleware or not cls.model_fields:
        return {
            'type': 'object',
            'properties': {},
            'additionalProperties': True,
        }
    try:
        return cls.model_json_schema()
    except Exception as e:
        # If a config field carries a type pydantic can't translate, prefer a
        # permissive empty schema over crashing the whole registration —
        # the middleware itself still works, just without form generation.
        logger.warning(
            f'Failed to derive config schema for middleware {cls.__name__}: {e}. '
            'Form generation in the Dev UI will be disabled for this middleware.',
            exc_info=True,
        )
        return {
            'type': 'object',
            'properties': {},
            'additionalProperties': True,
        }


def new_middleware(
    cls: type[BaseMiddleware],
    name: str,
    description: str | None = None,
) -> MiddlewareDesc:
    """Ergonomic helper to define a new MiddlewareDesc.

    Args:
        cls: The BaseMiddleware subclass.
        name: The registry name.
        description: Optional human-readable description.

    Returns:
        A new MiddlewareDesc instance.
    """
    return MiddlewareDesc(cls=cls, name=name, description=description)


def middleware_class_index(registry: RegistryLike) -> dict[type[BaseMiddleware], str]:
    """Reverse index from a registered class to the name it was registered under.

    Used by the generate pipeline and prompt registration to resolve inline
    ``use=[Foo(...)]`` instances back to the name their class was registered
    with via ``@ai.middleware``, ``new_middleware``, or a middleware plugin.
    Callers that loop over a ``use`` list should build this once up front so
    the total work is ``O(M + N)`` instead of ``O(M * N)`` for ``M`` inline
    middlewares and ``N`` registered ones.

    Classes that were never registered won't appear; the lookup just returns
    ``None`` and the caller decides whether to drop the entry or assign a
    synthetic id.
    """
    out: dict[type[BaseMiddleware], str] = {}
    for reg_name, value in registry.list_values('middleware').items():
        if isinstance(value, MiddlewareDesc):
            out[value._cls] = reg_name
    return out
