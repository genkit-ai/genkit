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

"""Middleware hook params, default implementation, and registration types.

Two roles (they are not superclass/subclass):

- **Runtime hook class:** ``BaseMiddleware`` is a ``pydantic.BaseModel`` that subclasses
  extend with config fields and hook overrides. Pass instances directly in ``use=[...]``
  for the in-process fast path (no registration needed).
- **Registry descriptor:** ``MiddlewareDesc`` — a named bundle in the registry (wire
  metadata plus a factory callable). It does not extend ``BaseMiddleware``; calling it
  with options returns the ``BaseMiddleware`` instance for that request. Register via
  ``middleware_plugin``, ``Plugin.list_middleware``, or ``Genkit.define_middleware``;
  reference by name in ``use=`` as ``MiddlewareRef``. ``MiddlewareDesc`` extends the
  auto-generated wire schema ``MiddlewareDescData`` and adds a ``PrivateAttr`` factory,
  so ``model_dump`` produces the wire shape automatically (the factory is private
  and never serialized).

Instance contract (matches Django / Starlette middleware conventions):
- Instances may be reused across concurrent ``generate()`` calls.
- Per-call scratch state belongs in method locals, not on ``self``.
- Shared state on ``self`` (buckets, counters) is the author's concurrency responsibility.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from genkit._core._registry import Registry

from pydantic import BaseModel, ConfigDict, PrivateAttr

from genkit._core._model import (
    GenerateHookParams,
    ModelHookParams,
    ModelResponse,
    MultipartToolResponse,
    ToolHookParams,
)
from genkit._core._typing import MiddlewareDescData, Part, ToolRequestPart

if TYPE_CHECKING:
    from genkit._core._action import Action

# Disallowed in middleware definition names and in ``middleware_plugin(..., namespace=...)``.
# Model/action keys use ``provider/name``; middleware stays one path-free token for the registry.
_FORBIDDEN_IN_MIDDLEWARE_KEY_SEGMENT = re.compile(r'[\x00-\x1f/\\:]|\s')


def _validate_middleware_key_segment(name: str, *, label: str) -> None:
    """Raise if ``name`` is not usable as a single middleware registry key (or namespace).

    Middleware definitions are stored under ``register_value(kind='middleware', name=...)``.
    Optional ``middleware_plugin(..., namespace='acme')`` builds keys like ``acme_logging``.
    So the string must be one segment: no ``/`` (that shape is for models/actions), and no
    whitespace, ``:``, backslashes, or control characters that would break keys or Dev UI.

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


class BaseMiddleware(BaseModel):
    """Pydantic-backed middleware: config fields + hook overrides in one class.

    The config struct IS the middleware — there is no separate "factory args"
    type.  Subclass with pydantic fields for config, override ``wrap_*`` hooks
    for behavior, pass instances directly in ``use=[...]``, or register via
    ``Genkit.define_middleware`` (or ``middleware_plugin`` /
    ``Plugin.list_middleware``) and reference by name with ``MiddlewareRef``.

    Example:
        class Logger(BaseMiddleware):
            name: ClassVar[str] = 'logger'
            prefix: str = '[trace]'

            async def wrap_model(self, params, next_fn):
                t = time.monotonic()
                resp = await next_fn(params)
                log(f'{self.prefix} {time.monotonic() - t:.3f}s')
                return resp

        # Local fast path:
        await ai.generate(prompt='...', use=[Logger(prefix='[span]')])

        # Registered / JSON-dispatched via Dev UI:
        ai.define_middleware(Logger)
        await ai.generate(prompt='...', use=[MiddlewareRef(name='logger', config={'prefix': '[span]'})])

    Concurrency: instances may be reused across concurrent ``generate()`` calls. Put
    per-call state in method locals; shared state on ``self`` is the author's
    concurrency responsibility (matches Django / Starlette middleware conventions).
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

    # Injected at resolve time by ``resolve_middleware_from_use`` so hooks can
    # do model lookup, action resolution, etc. as ``self._registry``.
    #
    # This is the **per-call child registry**, not the root.  Lookups still see
    # everything registered on the root (children delegate to parents on miss),
    # but anything middleware *registers* through this handle — extra tools,
    # values, dynamic actions — is automatically scoped to the current
    # ``generate()`` call and discarded when the call ends.  Middleware can
    # also resolve dynamic tools contributed by other middleware in the same
    # call, since they all share this scope.
    #
    # Registered middleware receive this via their factory; inline instances
    # passed in ``use=`` are ``model_copy()``-cloned first so the caller's
    # value is never mutated (safe to reuse one instance across concurrent
    # ``generate()`` calls).
    _registry: Registry | None = PrivateAttr(default=None)

    def tools(self, enqueue_parts: Callable[[list[Part]], None] | None = None) -> list[Action]:
        """Return additional tools to expose to the model for this generate call.

        Tools are registered on a call-scoped child registry, so they do not
        pollute the root registry and are invisible to other concurrent
        ``generate()`` calls.

        Args:
            enqueue_parts: Call this with a list of ``Part`` objects to queue an
                extra user message that will be injected into the conversation at
                the start of the next generate iteration.  Tool closures can
                capture this to report errors or rich context alongside the
                normal ``ToolResponsePart`` (e.g. filesystem error details).

        Override to contribute tools dynamically.  The default returns ``[]``.
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
        next_fn: Callable[
            [ToolHookParams],
            Awaitable[tuple[MultipartToolResponse | None, ToolRequestPart | None]],
        ],
    ) -> tuple[MultipartToolResponse | None, ToolRequestPart | None]:
        """Wrap each tool execution.

        Return ``(tool_response, interrupt)``: one of the tuple elements is non-``None``.
        """
        return await next_fn(params)


class MiddlewareDesc(MiddlewareDescData):
    """Registered middleware descriptor: wire shape + per-process factory closure.

    Inherits the wire fields (``name``, ``description``, ``config_schema``, ``metadata``)
    from the auto-generated :class:`genkit._core._typing.MiddlewareDescData` schema and
    adds a ``PrivateAttr`` factory used to mint a fresh :class:`BaseMiddleware` per
    ``generate()`` call. Because ``PrivateAttr`` is excluded from serialization,
    ``model_dump(by_alias=True, exclude_none=True)`` produces the wire shape directly.

    Stored under ``register_value('middleware', name, desc)`` and resolved when
    ``generate()`` runs with ``use=`` entries that reference by name.  Same
    hand-authored runtime subclass convention as ``Message``/``MessageData``
    and ``GenerateActionOptions``/``GenerateActionOptionsData`` — the runtime
    class adds non-serializable behavior (here: the factory) on top of the
    pure wire schema.
    """

    # ``arbitrary_types_allowed`` lets the ``PrivateAttr`` carry an opaque ``Callable``;
    # parent's ``alias_generator`` and ``extra='forbid'`` settings are inherited.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Factory takes ``(config, registry)`` and mints a fresh BaseMiddleware
    # instance per generate() call.  ``registry`` is ``None`` only when the
    # factory is called outside a generate context (e.g. tests or tooling).
    _factory: Callable[[dict[str, Any] | None, Registry | None], BaseMiddleware] = PrivateAttr()

    def __init__(
        self,
        *,
        factory: Callable[[dict[str, Any] | None, Registry | None], BaseMiddleware],
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

    def __call__(
        self,
        config: dict[str, Any] | None = None,
        registry: Registry | None = None,
    ) -> BaseMiddleware:
        """Return a fresh BaseMiddleware instance for this generate() call.

        Mirrors JS ``def.instantiate({ config, ai })``: the registry is injected
        at resolution time so the instance can look up models and actions.
        """
        return self._factory(config, registry)

    def with_name(self, name: str) -> MiddlewareDesc:
        """Return a copy with the same factory and metadata but a different registry name."""
        return MiddlewareDesc(
            factory=self._factory,
            name=name,
            description=self.description,
            config_schema=self.config_schema,
            metadata=self.metadata,
        )


def new_middleware(middleware_cls: type[BaseMiddleware]) -> MiddlewareDesc:
    """Create a ``MiddlewareDesc`` from a ``BaseMiddleware`` subclass.

    Set ``name``, and optionally ``description``, ``middleware_config_schema``, and
    ``middleware_metadata`` on the class. The resulting factory instantiates the class
    with ``**(config or {})`` when a request resolves the descriptor, so the same
    pydantic fields on the class drive both the inline (``use=[Cls(...)]``) and the
    registered (``MiddlewareRef(name=..., config=...)``) paths.

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

    def _factory(config: dict[str, Any] | None, registry: Registry | None = None) -> BaseMiddleware:
        # Instantiate with the incoming config so registered use is equivalent to
        # ``use=[middleware_cls(**config)]``; empty/None config uses class defaults.
        inst = middleware_cls(**(config or {}))
        inst._registry = registry
        return inst

    return MiddlewareDesc(
        name=reg_name,
        factory=_factory,
        description=middleware_cls.description,
        config_schema=middleware_cls.middleware_config_schema,
        metadata=middleware_cls.middleware_metadata,
    )
