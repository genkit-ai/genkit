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

"""Core middleware abstractions for the Genkit generate pipeline.

Defines :class:`BaseMiddleware` (the class authors subclass to add config fields
and hook overrides), :class:`MiddlewareDesc` (the registry descriptor used for
Dev UI name-based dispatch), plus the :func:`middleware` decorator and
:func:`new_middleware` factory for registration.

Also contains the hook parameter types (:class:`GenerateHookParams`,
:class:`ModelHookParams`, :class:`ToolHookParams`, :class:`MultipartToolResponse`)
that are passed into each hook by the engine. These live here rather than in
``_model.py`` because middleware is a concept built on top of the model layer.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, TypeVar

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

_M = TypeVar('_M', bound='type[BaseMiddleware]')

# Disallowed in middleware definition names and in ``middleware_plugin(..., namespace=...)``.
# Model/action keys use ``provider/name``; middleware stays one path-free token for the registry.
_FORBIDDEN_IN_MIDDLEWARE_KEY_SEGMENT = re.compile(r'[\x00-\x1f/\\:]|\s')


def _validate_middleware_key_segment(name: str, *, label: str) -> None:
    """Raise if ``name`` is not usable as a middleware registry key or namespace.

    Middleware definitions are stored under
    ``register_value(kind='middleware', name=...)``. The optional
    ``middleware_plugin(..., namespace='acme')`` builds keys of the form
    ``acme_logging``. The string must therefore be one segment:

    * no ``/`` (that shape is reserved for models and other actions);
    * no whitespace, ``:``, backslashes, or control characters that
      would break registry keys or the Dev UI.

    Args:
        name: Proposed name or namespace segment.
        label: Field name for error messages (e.g. ``MiddlewareDesc name``).
    """
    if not name or not name.strip():
        raise ValueError(f'{label} must be a non-empty string (not whitespace-only).')
    if name != name.strip():
        raise ValueError(f'{label} must not have leading or trailing whitespace.')
    if _FORBIDDEN_IN_MIDDLEWARE_KEY_SEGMENT.search(name):
        raise ValueError(
            f'{label} must be one path-free token: no whitespace, "/", ":", '
            r'backslashes, or control characters (for example "myorg_logging_mw").'
        )


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

    The config struct *is* the middleware — there is no separate
    "factory args" type. To author one:

    * Subclass and add pydantic fields for config.
    * Override the ``wrap_generate`` / ``wrap_model`` / ``wrap_tool`` hooks.
    * Either pass instances inline in ``use=[...]``, or register the class
      with ``Genkit.define_middleware`` (or ``middleware_plugin`` /
      ``Plugin.list_middleware``) and reference it by name with
      :class:`MiddlewareRef`.

    Inside any hook, two framework-injected attributes are guaranteed to be set:

    * ``self.registry`` — the per-call child registry. Use it to resolve actions
      and to inspect what else is in scope for this call. Anything you register
      through it is automatically scoped to the call and torn down at the end.
    * ``self.enqueue_parts(parts)`` — queue an extra user message to be injected
      into the conversation at the start of the next generate iteration. Use it
      from a tool closure or from ``wrap_tool`` to surface error details, file
      contents, or other rich context to the model without forging a tool
      response.

    Outside a ``generate()`` call these attributes are ``None`` — they only
    become valid once the engine binds the instance to a specific call.

    Example:
        @middleware(name='logger')
        class Logger(BaseMiddleware):
            prefix: str = '[trace]'

            async def wrap_model(self, params, next_fn):
                t = time.monotonic()
                resp = await next_fn(params)
                log(f'{self.prefix} {time.monotonic() - t:.3f}s')
                return resp

        # Inline (fast path, no registration):
        await ai.generate(prompt='...', use=[Logger(prefix='[span]')])

        # Registered (visible in the Dev UI, dispatched by name):
        ai.define_middleware(Logger)
        await ai.generate(
            prompt='...',
            use=[MiddlewareRef(name='logger', config={'prefix': '[span]'})],
        )

    Concurrency:
        Each ``generate()`` call works on its own shallow copy of the
        middleware instance with a freshly bound ``self.registry`` and
        ``self.enqueue_parts``, so those framework attributes are safe
        even when the same instance is reused across concurrent calls.
        Author-added state on ``self`` is *not* deep-copied — keep
        per-call state in method locals, or override ``model_copy`` if
        you need stronger isolation (same convention as Django /
        Starlette middleware).
    """

    # ``arbitrary_types_allowed`` lets subclasses keep non-pydantic fields like
    # ``Callable`` or opaque resources without opting in per-subclass.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Class-level metadata used by ``new_middleware(MyClass)`` and the Dev UI.
    # These are ClassVars, not fields, so they do not appear in ``model_dump()`` or
    # ``config`` dicts passed to factories.
    name: ClassVar[str] = ''
    description: ClassVar[str | None] = None
    middleware_config_schema: ClassVar[dict[str, Any] | None] = None
    middleware_metadata: ClassVar[dict[str, object] | None] = None

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
        ``self.registry`` and ``self.enqueue_parts``. Tool closures may
        capture ``self.enqueue_parts`` to queue extra user messages
        alongside the normal ``ToolResponsePart`` (e.g. filesystem
        error details for the next turn).

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

        Return a ``MultipartToolResponse`` to forward (or substitute) the
        tool's result.  Raise ``Interrupt(metadata)`` to halt this tool call
        and surface an interrupt to the caller — the engine attaches
        ``metadata`` to the pending ``ToolRequestPart`` exactly like an
        interrupt raised by the tool body itself.  Mirroring the tool-side
        convention means authors learn one rule: **responses are return
        values, interrupts are exceptions, everywhere**.

        Example (tool approval gate)::

            class Approval(BaseMiddleware):
                async def wrap_tool(self, params, next_fn):
                    if params.tool.name == 'transfer_money' and not approved():
                        raise Interrupt({'reason': 'requires_approval'})
                    return await next_fn(params)
        """
        return await next_fn(params)


class MiddlewareDesc(MiddlewareDescData):
    """Registered middleware descriptor: wire shape + per-process factory closure.

    Inherits the wire fields (``name``, ``description``, ``config_schema``,
    ``metadata``) from the auto-generated
    :class:`genkit._core._typing.MiddlewareDescData` schema, and adds a
    ``PrivateAttr`` factory used to mint a fresh :class:`BaseMiddleware` per
    ``generate()`` call. ``PrivateAttr`` is excluded from serialization, so
    ``model_dump(by_alias=True, exclude_none=True)`` produces the wire shape
    directly.

    Stored under ``register_value('middleware', name, desc)`` and resolved
    when ``generate()`` runs with a ``use=`` entry that references the
    descriptor by name. This follows the same hand-authored runtime-subclass
    convention as ``Message`` / ``MessageData`` and ``GenerateActionOptions`` /
    ``GenerateActionOptionsData``: the runtime class adds non-serializable
    behavior (here: the factory) on top of the pure wire schema.
    """

    # ``arbitrary_types_allowed`` lets the ``PrivateAttr`` carry an opaque ``Callable``;
    # parent's ``alias_generator`` and ``extra='forbid'`` settings are inherited.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Factory takes ``config`` and mints a fresh BaseMiddleware instance per
    # generate() call. The engine binds per-call attrs (``self.registry``,
    # ``self.enqueue_parts``) onto the result before any hook fires.
    _factory: Callable[[dict[str, Any] | None], BaseMiddleware] = PrivateAttr()

    def __init__(
        self,
        *,
        factory: Callable[[dict[str, Any] | None], BaseMiddleware],
        name: str,
        description: str | None = None,
        config_schema: object | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _validate_middleware_key_segment(name, label='MiddlewareDesc name')
        super().__init__(
            name=name,
            description=description,
            config_schema=config_schema,
            metadata=metadata,
        )
        self._factory = factory

    def __call__(self, config: dict[str, Any] | None = None) -> BaseMiddleware:
        """Return a fresh BaseMiddleware instance for this generate() call."""
        return self._factory(config)

    def with_name(self, name: str) -> MiddlewareDesc:
        """Return a copy with the same factory and metadata but a different registry name."""
        return MiddlewareDesc(
            factory=self._factory,
            name=name,
            description=self.description,
            config_schema=self.config_schema,
            metadata=self.metadata,
        )


def middleware(
    name: str,
    *,
    description: str | None = None,
    config_schema: dict[str, Any] | None = None,
    metadata: dict[str, object] | None = None,
) -> Callable[[_M], _M]:
    """Class decorator that sets registry metadata on a ``BaseMiddleware`` subclass.

    Required when registering middleware via ``new_middleware``,
    ``define_middleware``, or ``middleware_plugin``. Optional for inline-only
    use (``use=[MyClass()]``).

    Example:
        @middleware(name='latency_logger', description='Logs model call latency')
        class LatencyLogger(BaseMiddleware):
            prefix: str = '[trace]'

            async def wrap_model(self, params, next_fn): ...

    Args:
        name: Registry key. Must be a single path-free token (no ``/``,
            whitespace, ``:``, backslashes, or control characters).
        description: Human-readable description shown in the Dev UI.
        config_schema: JSON Schema for the config. Inferred from pydantic
            fields when omitted.
        metadata: Arbitrary metadata passed through to the Dev UI wire format.
    """
    _validate_middleware_key_segment(name, label='middleware name')

    def decorator(cls: _M) -> _M:
        cls.name = name  # type: ignore[attr-defined]
        cls.description = description  # type: ignore[attr-defined]
        cls.middleware_config_schema = config_schema  # type: ignore[attr-defined]
        cls.middleware_metadata = metadata  # type: ignore[attr-defined]
        return cls

    return decorator


def _derive_config_schema(cls: type[BaseMiddleware]) -> dict[str, Any]:
    """Build a JSON Schema describing a middleware's user-facing config fields.

    The Dev UI renders a config form for each registered middleware from this
    schema. Without it the form has nothing to draw and falls back to a free-text
    JSON box, so every middleware should expose one even when it has no knobs.

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
        # Empty object schema still tells the Dev UI "this middleware exists
        # and has no knobs", which renders as a no-input form rather than a
        # raw JSON editor.
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
    except Exception:
        # If a config field carries a type pydantic can't translate, prefer a
        # permissive empty schema over crashing the whole registration —
        # the middleware itself still works, just without form generation.
        return {
            'type': 'object',
            'properties': {},
            'additionalProperties': True,
        }


def new_middleware(middleware_cls: type[BaseMiddleware]) -> MiddlewareDesc:
    """Create a ``MiddlewareDesc`` from a ``BaseMiddleware`` subclass.

    Set ``name``, and optionally ``description``, ``middleware_config_schema``, and
    ``middleware_metadata`` on the class. The resulting factory instantiates the class
    with ``**(config or {})`` when a request resolves the descriptor, so the same
    pydantic fields on the class drive both the inline (``use=[Cls(...)]``) and the
    registered (``MiddlewareRef(name=..., config=...)``) paths.

    When ``middleware_config_schema`` is not set explicitly, a JSON Schema is
    derived from the subclass's pydantic fields so the Dev UI can render a
    config form without the author having to hand-write one.

    Does not register; pass the result to ``middleware_plugin([...])`` or return from
    a custom ``Plugin.list_middleware``.

    Args:
        middleware_cls: A ``BaseMiddleware`` subclass with a non-empty ``name``.

    Returns:
        A descriptor suitable for ``registry.register_value`` or ``middleware_plugin``.
    """
    reg_name = middleware_cls.name
    if not reg_name:
        raise ValueError(f'{middleware_cls.__qualname__}.name must be set for new_middleware(MyClass).')
    _validate_middleware_key_segment(str(reg_name), label=f'{middleware_cls.__qualname__}.name')

    def _factory(config: dict[str, Any] | None) -> BaseMiddleware:
        # Instantiate with the incoming config so registered use is equivalent to
        # ``use=[middleware_cls(**config)]``; empty/None config uses class defaults.
        return middleware_cls(**(config or {}))

    config_schema = middleware_cls.middleware_config_schema
    if config_schema is None:
        config_schema = _derive_config_schema(middleware_cls)

    return MiddlewareDesc(
        name=reg_name,
        factory=_factory,
        description=middleware_cls.description,
        config_schema=config_schema,
        metadata=middleware_cls.middleware_metadata,
    )
