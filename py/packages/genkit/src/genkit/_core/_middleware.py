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
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, NamedTuple, Protocol, TypeVar, cast, get_args, get_origin

from pydantic import AliasChoices, BaseModel, ConfigDict, PrivateAttr, ValidationError, field_validator

from genkit._core._action import Action
from genkit._core._config_check import config_field_names
from genkit._core._error import GenkitError, RuntimeErrorReason
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


TConfig = TypeVar('TConfig', bound=BaseModel)


def _config_key_spellings(config_cls: type[BaseModel]) -> dict[str, str] | None:
    """Map every key ``config_cls`` accepts to the spelling to suggest for it.

    Returns None when the class takes keys it doesn't declare (``extra``
    set explicitly to ``'allow'`` or ``'ignore'``) or uses an alias path this
    check can't enumerate.
    """
    if config_cls.model_config.get('extra') in ('allow', 'ignore'):
        return None
    fields = config_cls.model_fields
    for info in fields.values():
        validation_alias = info.validation_alias
        if validation_alias is None or isinstance(validation_alias, str):
            continue
        if not isinstance(validation_alias, AliasChoices) or not all(
            isinstance(choice, str) for choice in validation_alias.choices
        ):
            return None
    accepts_field_name = bool(
        config_cls.model_config.get('validate_by_name') or config_cls.model_config.get('populate_by_name')
    )
    # config_field_names is the same walk model config folding uses; pydantic
    # only rejects the bare field name when it has an alias and the class
    # doesn't validate by name.
    names = config_field_names(config_cls)
    alias_keys = set(names) - set(fields)
    for name, info in fields.items():
        if name == info.alias or name == info.validation_alias:
            alias_keys.add(name)
        elif isinstance(info.validation_alias, AliasChoices) and name in info.validation_alias.choices:
            alias_keys.add(name)
    accepted = {
        key: name
        for key, name in names.items()
        if key in alias_keys
        or accepts_field_name
        or (fields[name].alias is None and fields[name].validation_alias is None)
    }
    spellings: dict[str, str] = {}
    for key, name in accepted.items():
        info = fields[name]
        # The wire, Dev UI schema and the JS/Go SDKs use the serialization alias.
        wire = info.serialization_alias
        spellings[key] = wire if wire is not None and wire in accepted else (info.alias or name)
    return spellings


def _dump_middleware_ref_config(config: BaseModel) -> dict[str, Any] | None:
    """Dump a middleware config for ``MiddlewareRef`` using keys the class validates.

    Excludes ``@computed_field``s and falls back to the validation alias or
    field name when ``serialization_alias`` is output-only.
    """
    cfg_cls = type(config)
    computed = set(cfg_cls.model_computed_fields)
    dumped = config.model_dump(exclude_none=True, by_alias=True, mode='json', exclude=computed)
    spellings = _config_key_spellings(cfg_cls)
    if spellings is None or all(k in spellings for k in dumped):
        return dumped or None
    raw = config.model_dump(exclude_none=True, by_alias=False, mode='json', exclude=computed)
    out: dict[str, Any] = {}
    for name, val in raw.items():
        info = cfg_cls.model_fields.get(name)
        if info is None:
            out[name] = val
            continue
        candidates: list[str] = []
        if info.serialization_alias:
            candidates.append(info.serialization_alias)
        if info.alias:
            candidates.append(info.alias)
        if isinstance(info.validation_alias, str):
            candidates.append(info.validation_alias)
        elif isinstance(info.validation_alias, AliasChoices):
            candidates.extend(c for c in info.validation_alias.choices if isinstance(c, str))
        candidates.append(name)
        key = next((c for c in candidates if c in spellings), name)
        out[key] = val
    return out or None


def _unknown_config_keys(config_cls: type[BaseModel], keys: Iterable[str]) -> tuple[list[str], list[str]]:
    """Keys ``config_cls`` would drop, and the keys it accepts, for an error message.

    A plain BaseModel ignores unknown keys, so ``maxRetries`` against a
    ``max_retries`` field would run with the default and no error.
    """
    spellings = _config_key_spellings(config_cls)
    if spellings is None:
        return [], []
    unknown = sorted(k for k in keys if k not in spellings)
    return unknown, sorted(set(spellings.values()))


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
class GenerateMiddlewareContext:
    """Per-``generate()`` runtime services shared by every middleware in ``use=[...]``.

    ``ai`` is the app's ``Genkit`` instance, so ``await ctx.ai.lookup_model(name)``,
    ``await ctx.ai.lookup_tool(name)`` and ``ctx.ai.lookup_value(kind=..., name=...)``
    find what the app defined. Inside an agent, read the session with
    ``genkit.exp.current_session()``. Also carries caller-provided metadata
    (``custom_context``), streaming hooks, and the abort signal for the whole
    generate invocation.
    """

    ai: GenkitLike
    abort_signal: asyncio.Event = field(default_factory=asyncio.Event)
    custom_context: dict[str, object] = field(default_factory=dict)
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


class BaseMiddleware(Generic[TConfig]):
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

       Any pydantic BaseModel works. Its keys (field names, or aliases if it
       declares them) are the config keys in .prompt files, the Dev UI form
       and traces. A key the class doesn't declare raises instead of being
       dropped (a ValidationError, as for a GenkitConfig), unless the class sets
       ``extra='allow'`` or ``'ignore'``.

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
            unknown, _ = _unknown_config_keys(self.Config, kwargs)
            if unknown:
                # The same error a GenkitConfig raises, so the base class doesn't change what callers catch.
                raise ValidationError.from_exception_data(
                    self.Config.__name__,
                    [{'type': 'extra_forbidden', 'loc': (key,), 'input': kwargs[key]} for key in unknown],
                )
            self.config = cast(Any, self.Config(**kwargs))

    def tools(self, ctx: GenerateMiddlewareContext) -> list[Tool]:
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

    def tools(self, ctx: GenerateMiddlewareContext) -> list[Tool]:
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
        """Build a configured middleware instance for one resolve step.

        ``config`` comes from a .prompt file, the Dev UI or a trace, so an
        unknown key is INVALID_ARGUMENT naming the middleware and its keys.
        """
        config = config or {}
        unknown, accepted = _unknown_config_keys(self._cls.Config, config)
        if unknown:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=(
                    f'Middleware "{self.name}" got unknown config key(s) {", ".join(unknown)}; '
                    f'accepted keys: {", ".join(accepted) or "none"}'
                ),
                source='genkit.generate',
                reason=RuntimeErrorReason.INVALID_INPUT,
            )
        try:
            return self._cls(**config)
        except ValidationError as e:
            # Wrong type, a bound, or a missing required field.
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f'Middleware "{self.name}" got invalid config',
                source='genkit.generate',
                reason=RuntimeErrorReason.INVALID_INPUT,
                cause=e,
            ) from e

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


def middleware_class_index(registry: RegistryLike) -> dict[type[BaseMiddleware], str]:
    """Reverse index from a registered class to the name it was registered under."""
    out: dict[type[BaseMiddleware], str] = {}
    for reg_name, value in registry.list_values('middleware').items():
        if isinstance(value, GenerateMiddleware):
            out[value._cls] = reg_name
    return out
