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

import re
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, create_model

from genkit._core._action import Action
from genkit._core._model import (
    GenerateActionOptions,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
)
from genkit._core._protocols import RegistryLike
from genkit._core._typing import MiddlewareDescData, Part, ToolRequestPart
from genkit._core._logger import get_logger

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


class BaseMiddleware(BaseModel):
    """Pydantic-backed middleware: config fields + hook overrides in one class.

    To author a middleware, 
    1. Subclass `BaseMiddleware` and add pydantic fields for config.
    2. Override the ``wrap_generate`` / ``wrap_model`` / ``wrap_tool`` hooks.
    3. Wrap your subclass with the `@ai.middleware` decorator to make it available
    in your local Dev UI.

    To use a middleware, you can either:
    1. Pass the Middleware directly into the ``use`` argument of ``ai.generate``.
    2. Reference it by name with :class:`MiddlewareRef`.

    Example:
        @ai.middleware(name='logger')
        class Logger(BaseMiddleware):
            prefix: str = '[trace]'

            async def wrap_model(self, params, next_fn):
                t = time.monotonic()
                resp = await next_fn(params)
                log(f'{self.prefix} {time.monotonic() - t:.3f}s')
                return resp

        await ai.generate(
            prompt='...',
            use=[MiddlewareRef(name='logger', config={'prefix': '[span]'})],
        )

    Inside any hook, two framework-injected attributes are guaranteed to be set:

    1. `self.registry` — the per-call registry. Use it to resolve actions
      and to inspect what else is in scope for this call. Anything you register
      through it is automatically scoped to the call and torn down at the end.
    2. `self.enqueue_parts(parts)` — queue an extra user message to be injected
      into the conversation at the start of the next generate iteration. Use it
      from a tool closure or from ``wrap_tool`` to surface error details, file
      contents, or other rich context to the model without forging a tool
      response.

    Outside a `generate()` call these attributes are `None` — they only
    become valid once the engine binds the instance to a specific call.

    Avoid storing additional mutable state on ``self`` inside your hooks.
    Each `generate()` call works on its own shallow copy of the middleware
    instance.
    """

    # ``arbitrary_types_allowed`` lets subclasses keep non-pydantic fields like
    # ``Callable`` or opaque resources without opting in per-subclass.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Class-level metadata stamped by ``@ai.middleware(...)``; the descriptor
    # and the Dev UI's reflection endpoint read these.
    # These are ClassVars, not fields, so they do not appear in ``model_dump()``
    # or ``config`` dicts passed to ``cls(**config)``.
    name: ClassVar[str] = ''
    description: ClassVar[str | None] = None

    # Framework-injected at the start of each generate() call (see the class
    # docstring). They are public fields, not PrivateAttrs, so a middleware
    # author writing ``self.`` in their IDE sees them in autocomplete and knows
    # they exist.  Annotated as required so hooks can write
    # ``self.registry.lookup_action(...)`` without a None-narrow; the runtime
    # default of ``None`` lets bare constructors like ``Retry(max_retries=3)``
    # work, with the engine rebinding before any hook fires.
    registry: RegistryLike = Field(default=None, exclude=True, repr=False)  # type: ignore[assignment]
    enqueue_parts: Callable[[list[Part]], None] = Field(default=None, exclude=True, repr=False)  # type: ignore[assignment]

    def tools(self) -> list[Action]:
        """Return additional tools to expose to the model for this generate call.

        Called once per ``generate()`` call after the engine has bound
        ``self.registry``.

        Tools are registered on a call-scoped child registry, so they
        do not pollute the root registry and are invisible to other
        concurrent ``generate()`` calls.

        Override to contribute tools dynamically. The default returns
        ``[]``.
        """
        return []

    async def wrap_generate(
        self,
        params: GenerateHookParams,
        next_fn: Callable[[GenerateHookParams], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Wrap each iteration of the tool loop (model call + optional tool resolution)."""
        return await next_fn(params)

    async def wrap_model(
        self,
        params: ModelHookParams,
        next_fn: Callable[[ModelHookParams], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Wrap each model API call."""
        return await next_fn(params)

    async def wrap_tool(
        self,
        params: ToolHookParams,
        next_fn: Callable[[ToolHookParams], Awaitable[MultipartToolResponse]],
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

    # The class the registry instantiates per `generate()` call. The engine
    # binds per-call attrs (`self.registry`, `self.enqueue_parts`) onto the
    # result before any hook fires.
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
        if description is None:
            description = cls.description
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

    Pydantic's full ``cls.model_json_schema()`` would also include the
    framework-injected ``registry`` and ``enqueue_parts`` attributes — those
    aren't config the user sets, and their types (a registry protocol, a
    callable) aren't always representable in JSON Schema. Build the schema from
    a stripped pydantic model containing only the subclass-added fields so the
    Dev UI sees just the knobs the author meant to expose.
    """
    base_fields = set(BaseMiddleware.model_fields)
    new_fields: dict[str, Any] = {
        field_name: (info.annotation, info)
        for field_name, info in cls.model_fields.items()
        if field_name not in base_fields
    }
    if not new_fields:
        return {
            'type': 'object',
            'properties': {},
            'additionalProperties': True,
        }
    try:
        stripped = create_model(  # type: ignore[call-overload]
            f'{cls.__name__}Config',
            __config__=ConfigDict(arbitrary_types_allowed=True),
            **new_fields,
        )
        return stripped.model_json_schema()
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
