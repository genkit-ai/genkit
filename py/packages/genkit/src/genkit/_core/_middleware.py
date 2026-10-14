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

import asyncio
import inspect
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, NamedTuple, Protocol, cast, get_args, get_origin

from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator
from typing_extensions import TypeVar

from genkit._core._action import Action
from genkit._core._logger import get_logger
from genkit._core._model import (
    GenerateActionOptions,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
    MultipartToolResponse,
    Part,
    as_part,
)
from genkit._core._protocols import GenkitLike, RegistryLike
from genkit._core._tool import Tool
from genkit._core._typing import MiddlewareDesc

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


class _EmptyMiddlewareConfig(BaseModel):
    """Placeholder config for middleware with no user-facing knobs."""

    model_config = ConfigDict(extra='forbid')


# Covariant (config is read-only) with a BaseModel default, so bare
# ``Middleware`` means ``Middleware[BaseModel]`` and ``Sequence[Middleware]``
# accepts every subclass.
TConfig = TypeVar('TConfig', bound=BaseModel, covariant=True, default=BaseModel)


class GenerateHookParams(BaseModel):
    """Params passed to the ``wrap_generate`` hook.

    Covers one full iteration of the tool loop: a model call plus optional tool
    resolution. ``message_index`` tracks streaming position for this turn.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    options: GenerateActionOptions
    iteration: int
    message_index: int = 0


class ModelHookParams(BaseModel):
    """Params passed to the ``wrap_model`` hook (each raw model API call)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    request: ModelRequest


class ToolHookParams(BaseModel):
    """Params passed to the ``wrap_tool`` hook (each individual tool execution)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    tool_request_part: Part
    tool: Action

    @field_validator('tool_request_part', mode='before')
    @classmethod
    def _wrap_tool_request_part(cls, v: object) -> object:
        if v is None:
            return v
        part = as_part(v)
        if part.tool_request is None:
            raise ValueError('wrap_tool needs a tool request part')
        return part


@dataclass
class MiddlewareContext:
    """Per-``generate()`` runtime services shared by every middleware in ``use=[...]``.

    ``ai`` is the app's ``Genkit`` instance, so ``await ctx.ai.lookup_model(name)``,
    ``await ctx.ai.lookup_tool(name)`` and ``ctx.ai.lookup_value(kind=..., name=...)``
    find what the app defined. Inside an agent, read the session with
    ``genkit.exp.current_session()``. Also carries a copy of the call's
    ``context=`` (auth, secrets; the same dict flows and tools read as
    ``ctx.context``), streaming hooks, and the abort signal for the whole
    generate invocation.
    """

    ai: GenkitLike
    abort_signal: asyncio.Event = field(default_factory=asyncio.Event)
    context: dict[str, object] = field(default_factory=dict)
    on_chunk: Callable[[ModelResponseChunk], None] | None = None
    telemetry_labels: dict[str, str] | None = None

    @property
    def is_streaming(self) -> bool:
        """True when the caller registered a streaming callback for this generate."""
        return self.on_chunk is not None

    def send_chunk(self, chunk: ModelResponseChunk) -> None:
        """Stream a chunk to the client when ``on_chunk`` is set."""
        if self.on_chunk is not None:
            self.on_chunk(chunk)

    def replace_on_chunk(
        self,
        on_chunk: Callable[[ModelResponseChunk], None] | None,
    ) -> Callable[[ModelResponseChunk], None] | None:
        """Swap the streaming callback; returns the previous callback."""
        previous = self.on_chunk
        self.on_chunk = on_chunk
        return previous


class Middleware(Generic[TConfig]):
    """Base class for generate middleware.

    A middleware is defined by its custom configuration (backed by Pydantic),
    and a set of hooks to inject logic into the generate pipeline.

    The base middleware has no custom configuration and noop hooks. The hooks
    that are not overridden are still called by the engine when the middleware
    is invoked.

    To author a middleware,
    1. Declare a config model, e.g. RetryConfig:

        class RetryConfig(BaseModel):
            max_retries: int = 3

    2. Extend Middleware with your config model, e.g. Retry:
        ai = Genkit()

        @ai.middleware(
            name="retry",
            description="Configures smart retry logic with exponential backoff and a jitter."
        )
        class Retry(Middleware[RetryConfig]):
            async def wrap_model(self, params, next_fn, ctx):
                for attempt in range(self.config.max_retries + 1):
                    ...

    Wrap your subclass with the ``@ai.middleware`` decorator to make it available
    in your local Dev UI.

    3. Use the Retry middleware in your `generate` call:
        ai.generate(
            ...,
            use=[Retry(max_retries=5)]
        )

        # Or alternatively, for full keyword auto-complete in your preferred IDE:
        ai.generate(
            ...,
            use=[Retry(config=RetryConfig(max_retries=5))]
        )

    ``self.config`` is set once at construction. The attribute is read-only;
    treat the config object as read-only too (Pydantic models are not frozen).
    """

    name: ClassVar[str] = ''
    description: ClassVar[str | None] = None
    Config: ClassVar[type[BaseModel]] = _EmptyMiddlewareConfig
    _config: TConfig

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init_subclass__(**kwargs)
        # Python keeps RetryConfig in Middleware[RetryConfig] for type checkers only.
        # We copy it onto cls.Config so Retry(max_retries=5) and the Dev UI form work.
        if 'Config' in cls.__dict__:
            raise TypeError(f'{cls.__name__} must not define Config; declare config as Middleware[YourConfig] instead.')
        config_cls: type[BaseModel] | None = None
        # Pull config type out of the brackets of the class declaration.
        # i.e. CustomMiddlewareConfig from CustomMiddleware(Middleware[CustomMiddlewareConfig]).
        for base in getattr(cls, '__orig_bases__', ()):
            if get_origin(base) is Middleware:
                args = get_args(base)
                if len(args) == 1:
                    arg = args[0]
                    if isinstance(arg, type) and issubclass(arg, BaseModel):
                        config_cls = arg
                break
        # Look for a config in parent classes, e.g. if a parent is Middleware[SomeConfig]
        if config_cls is None:
            for base in cls.__mro__[1:]:
                if base is Middleware:
                    break
                if issubclass(base, Middleware) and base.Config is not _EmptyMiddlewareConfig:
                    config_cls = base.Config
                    break
        # Handle the case where no config type is specified, e.g. class Logging(Middleware):
        # This means there is no user-facing configuration for the middleware.
        cls.Config = config_cls or _EmptyMiddlewareConfig

    def __init__(self, *, config: TConfig | None = None, **kwargs: Any) -> None:  # noqa: ANN401
        if config is not None:
            if kwargs:
                raise TypeError('pass either config= or keyword config fields, not both')
            if not isinstance(config, self.Config):
                raise TypeError(f'expected config type {self.Config.__name__}, got {type(config).__name__}')
            self._config = config
        else:
            self._config = cast(Any, self.Config(**kwargs))

    @property
    def config(self) -> TConfig:
        """The validated config this instance was constructed with (read-only)."""
        return self._config

    @classmethod
    def _middleware_metadata(
        cls,
        name: str | None = None,
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> _MiddlewareMetadata:
        """Build the internal Dev UI catalog descriptor for this middleware class."""
        effective_name = name if name is not None else cls.name
        effective_desc = description if description is not None else cls.description
        return _MiddlewareMetadata(
            cls=cls,
            name=effective_name,
            description=effective_desc,
            metadata=metadata,
        )

    def tools(self, ctx: MiddlewareContext) -> list[Tool]:
        """Return additional tools to expose to the model for this generate call."""
        return []

    async def wrap_generate(
        self,
        params: GenerateHookParams,
        ctx: MiddlewareContext,
        next_fn: Callable[[GenerateHookParams, MiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Wrap each iteration of the tool loop (model call + optional tool resolution)."""
        return await next_fn(params, ctx)

    async def wrap_model(
        self,
        params: ModelHookParams,
        ctx: MiddlewareContext,
        next_fn: Callable[[ModelHookParams, MiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Wrap each model API call."""
        return await next_fn(params, ctx)

    async def wrap_tool(
        self,
        params: ToolHookParams,
        ctx: MiddlewareContext,
        next_fn: Callable[[ToolHookParams, MiddlewareContext], Awaitable[MultipartToolResponse]],
    ) -> MultipartToolResponse:
        """Wrap each tool execution.

        Return a `MultipartToolResponse` to forward (or substitute) the
        tool's result.  Raise `Interrupt(metadata)` to halt this tool call
        and surface an interrupt to the caller.
        """
        return await next_fn(params, ctx)


def _copy_middleware_instance(impl: Middleware) -> Middleware:
    """Return a fresh instance with the same config; internal hook state is not copied."""
    return type(impl)(config=impl._config.model_copy(deep=True))


class MiddlewareDef(Protocol):
    """Hook contract the generate pipeline chains.

    Authors implement this by subclassing ``Middleware``. The pipeline types
    against this protocol so it only calls hooks, not constructors or config.
    """

    def tools(self, ctx: MiddlewareContext) -> list[Tool]:
        """Return additional tools to expose to the model for this generate call."""
        ...

    async def wrap_generate(
        self,
        params: GenerateHookParams,
        ctx: MiddlewareContext,
        next_fn: Callable[[GenerateHookParams, MiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Wrap each iteration of the tool loop (model call + optional tool resolution)."""
        ...

    async def wrap_model(
        self,
        params: ModelHookParams,
        ctx: MiddlewareContext,
        next_fn: Callable[[ModelHookParams, MiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Wrap each model API call."""
        ...

    async def wrap_tool(
        self,
        params: ToolHookParams,
        ctx: MiddlewareContext,
        next_fn: Callable[[ToolHookParams, MiddlewareContext], Awaitable[MultipartToolResponse]],
    ) -> MultipartToolResponse:
        """Wrap each tool execution."""
        ...


class _MiddlewareMetadata(MiddlewareDesc):
    """Internal Dev UI catalog descriptor and factory for a registered middleware class."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _cls: type[Middleware] = PrivateAttr()

    def __init__(
        self,
        *,
        cls: type[Middleware],
        name: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        effective_name = name if name is not None else getattr(cls, 'name', '')
        res = _validate_middleware_key_segment(effective_name)
        if res.errored:
            raise ValueError(f'middleware name {res.error_message}')
        effective_desc = description if description is not None else getattr(cls, 'description', None)
        if effective_desc is None and cls.__doc__:
            effective_desc = inspect.cleandoc(cls.__doc__)
        super().__init__(
            name=effective_name,
            description=effective_desc,
            config_schema=_derive_config_schema(cls),
            metadata=metadata,
        )
        self._cls = cls

    def instantiate(self, config: dict[str, Any] | None = None) -> Middleware:
        """Build a configured middleware instance for one resolve step."""
        return self._cls(**(config or {}))

    def __call__(self, config: dict[str, Any] | None = None) -> Middleware:
        return self.instantiate(config)


def _derive_config_schema(cls: type[Middleware]) -> dict[str, Any]:
    """Build a JSON Schema describing a middleware's user-facing config fields."""
    config_cls = cls.Config
    if config_cls is _EmptyMiddlewareConfig or not config_cls.model_fields:
        return {
            'type': 'object',
            'properties': {},
            'additionalProperties': True,
        }
    try:
        return config_cls.model_json_schema()
    except Exception as e:
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


def middleware_class_index(registry: RegistryLike) -> dict[type[Middleware], str]:
    """Reverse index from a registered class to the name it was registered under."""
    out: dict[type[Middleware], str] = {}
    for reg_name, value in registry.list_values('middleware').items():
        if isinstance(value, _MiddlewareMetadata):
            out[value._cls] = reg_name
    return out
