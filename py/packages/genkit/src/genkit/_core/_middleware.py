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
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, NamedTuple, Protocol, TypeVar, get_args, get_origin

from pydantic import BaseModel, ConfigDict, PrivateAttr

from genkit._core._action import Action
from genkit._core._logger import get_logger
from genkit._core._model import (
    GenerateActionOptions,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
)
from genkit._core._protocols import RegistryLike
from genkit._core._typing import MiddlewareDesc, MultipartToolResponse, ToolRequestPart

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


TConfig = TypeVar('TConfig', bound=BaseModel)


class GenerateHookParams(BaseModel):
    """Params passed to the ``wrap_generate`` hook.

    Covers one full iteration of the tool loop: a model call plus optional tool
    resolution. ``message_index`` tracks streaming position for this turn.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    options: GenerateActionOptions
    request: ModelRequest
    iteration: int
    message_index: int = 0


class ModelHookParams(BaseModel):
    """Params passed to the ``wrap_model`` hook (each raw model API call)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    request: ModelRequest


class ToolHookParams(BaseModel):
    """Params passed to the ``wrap_tool`` hook (each individual tool execution)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    tool_request_part: ToolRequestPart
    tool: Action


@dataclass
class GenerateMiddlewareContext:
    """Per-``generate()`` runtime services shared by every middleware in ``use=[...]``.

    Holds the call-scoped registry, caller-provided metadata (``custom_context``),
    and streaming hooks for the whole generate invocation. Hook ``params`` carry
    per-turn request data only.
    """

    registry: RegistryLike
    custom_context: dict[str, object] = field(default_factory=dict)
    on_chunk: Callable[[ModelResponseChunk], None] | None = None
    abort_signal: Any | None = None
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


class BaseMiddleware(Generic[TConfig]):
    """Base class for generate middleware.

    A middleware is defined by its custom configuration (backed by Pydantic),
    and a set of hooks to inject logic into the generate pipeline.

    The base middleware has no custom configuration and noop hooks. The hooks
    that are not overriden are still called by the engine when the middleware
    is invoked.

    To author a middleware,
    1. Declare a config model, e.g. RetryConfig:

        class RetryConfig(BaseModel):
            max_retries: int = 3

    2. Extend BaseMiddleware with your config model, e.g. Retry:
        ai = Genkit()

        @ai.middleware(
            name="retry",
            description="Configures smart retry logic with exponential backoff and a jitter."
        )
        class Retry(BaseMiddleware[RetryConfig]):
            async def wrap_model(self, params, next_fn, ctx):
                for attempt in range(self.config.max_retries + 1):
                    ...

    Wrap your subclass with the ``@ai.middleware`` decorator to make it available
    in your local Dev UI.

    3.Use the Retry middleware in your `generate` call:
        ai.generate(
            ...,
            use=[Retry(max_retries=5)]
        )

        # Or alternatively, for full keyword auto-complete in your preferred IDE:
        ai.generate(
            ...,
            use=[Retry(config=RetryConfig(max_retries=5))]
        )

    Keep in mind that config are not meant to be mutated from within hooks.
    """

    Config: ClassVar[type[BaseModel]] = _EmptyMiddlewareConfig
    config: TConfig

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init_subclass__(**kwargs)
        # Python keeps RetryConfig in BaseMiddleware[RetryConfig] for type checkers only.
        # We copy it onto cls.Config so Retry(max_retries=5) and the Dev UI form work.
        if 'Config' in cls.__dict__:
            raise TypeError(
                f'{cls.__name__} must not define Config; declare config as BaseMiddleware[YourConfig] instead.'
            )
        config_cls: type[BaseModel] | None = None
        # Pull config type out of the brackets of the class declaration.
        # i.e. CustomMiddlewareConfig from CustomMiddleware(BaseMiddleware[CustomMiddlewareConfig]).
        for base in getattr(cls, '__orig_bases__', ()):
            if get_origin(base) is BaseMiddleware:
                args = get_args(base)
                if len(args) == 1:
                    arg = args[0]
                    if isinstance(arg, type) and issubclass(arg, BaseModel):
                        config_cls = arg
                break
        # Look for a config in parent classes, e.g. if a parent is BaseMiddleware[SomeConfig]
        if config_cls is None:
            for base in cls.__mro__[1:]:
                if base is BaseMiddleware:
                    break
                if issubclass(base, BaseMiddleware) and base.Config is not _EmptyMiddlewareConfig:
                    config_cls = base.Config
                    break
        # Handle the case where no config type is specified, e.g. class Logging(BaseMiddleware):
        # This means there is no user-facing configuration for the middleware.
        cls.Config = config_cls or _EmptyMiddlewareConfig

    def __init__(self, *, config: TConfig | None = None, **kwargs: Any) -> None:  # noqa: ANN401
        if config is not None:
            if kwargs:
                raise TypeError('pass either config= or keyword config fields, not both')
            if not isinstance(config, self.Config):
                raise TypeError(f'expected config type {self.Config.__name__}, got {type(config).__name__}')
            self.config = config
        else:
            self.config = self.Config(**kwargs)  # type: ignore[assignment]

    def tools(self, ctx: GenerateMiddlewareContext) -> list[Action]:
        """Return additional tools to expose to the model for this generate call."""
        return []

    async def wrap_generate(
        self,
        params: GenerateHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Wrap each iteration of the tool loop (model call + optional tool resolution)."""
        return await next_fn(params, ctx)

    async def wrap_model(
        self,
        params: ModelHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Wrap each model API call."""
        return await next_fn(params, ctx)

    async def wrap_tool(
        self,
        params: ToolHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]],
    ) -> MultipartToolResponse:
        """Wrap each tool execution.

        Return a `MultipartToolResponse` to forward (or substitute) the
        tool's result.  Raise `Interrupt(metadata)` to halt this tool call
        and surface an interrupt to the caller.
        """
        return await next_fn(params, ctx)


def _copy_middleware_instance(impl: BaseMiddleware[Any]) -> BaseMiddleware[Any]:
    """Return a fresh instance with the same config; internal hook state is not copied."""
    return type(impl)(config=impl.config.model_copy(deep=True))


class MiddlewareDef(Protocol):
    """Hook contract the generate pipeline chains.

    Authors implement this by subclassing ``BaseMiddleware``. The pipeline types
    against this protocol so it only calls hooks, not constructors or config.
    """

    def tools(self, ctx: GenerateMiddlewareContext) -> list[Action]:
        """Return additional tools to expose to the model for this generate call."""
        ...

    async def wrap_generate(
        self,
        params: GenerateHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Wrap each iteration of the tool loop (model call + optional tool resolution)."""
        ...

    async def wrap_model(
        self,
        params: ModelHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Wrap each model API call."""
        ...

    async def wrap_tool(
        self,
        params: ToolHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]],
    ) -> MultipartToolResponse:
        """Wrap each tool execution."""
        ...


class GenerateMiddleware(MiddlewareDesc):
    """Registered middleware factory: wire metadata + class used to instantiate hooks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
            raise ValueError(f'GenerateMiddleware name {res.error_message}')
        if description is None and cls.__doc__:
            description = inspect.cleandoc(cls.__doc__)
        super().__init__(
            name=name,
            description=description,
            config_schema=_derive_config_schema(cls),
            metadata=metadata,
        )
        self._cls = cls

    def instantiate(self, config: dict[str, Any] | None = None) -> BaseMiddleware:
        """Build a configured middleware instance for one resolve step."""
        return self._cls(**(config or {}))

    def __call__(self, config: dict[str, Any] | None = None) -> BaseMiddleware:
        return self.instantiate(config)


def _derive_config_schema(cls: type[BaseMiddleware]) -> dict[str, Any]:
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


def new_middleware(
    cls: type[BaseMiddleware],
    name: str,
    description: str | None = None,
) -> GenerateMiddleware:
    """Ergonomic helper to define a new ``GenerateMiddleware``.

    Args:
        cls: The BaseMiddleware subclass.
        name: The registry name.
        description: Optional human-readable description.

    Returns:
        A new GenerateMiddleware instance.
    """
    return GenerateMiddleware(cls=cls, name=name, description=description)


def middleware_class_index(registry: RegistryLike) -> dict[type[BaseMiddleware], str]:
    """Reverse index from a registered class to the name it was registered under."""
    out: dict[type[BaseMiddleware], str] = {}
    for reg_name, value in registry.list_values('middleware').items():
        if isinstance(value, GenerateMiddleware):
            out[value._cls] = reg_name
    return out
