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

"""Generate action."""

import asyncio
import contextlib
import copy
import secrets
import time
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

from genkit._ai._agents._session import get_current_session
from genkit._ai._formats._types import FormatDef, Formatter
from genkit._ai._messages import inject_instructions
from genkit._ai._model import (
    Message,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
    resolve_model_name,
    text_from_content,
)
from genkit._ai._resource import ResourceArgument, ResourceInput, find_matching_resource, resolve_resources
from genkit._ai._tools import (
    ORIGINAL_OUTPUT_SCHEMA_KEY,
    Interrupt,
    Tool,
    as_multipart_tool_response,
    dump_tool_metadata,
    dump_tool_output,
    normalize_pending_content,
    parts_to_wire,
    restart_interrupt_error,
    run_tool_after_restart,
    run_tool_request,
)
from genkit._core._action import (
    GENKIT_DYNAMIC_ACTION_PROVIDER_ATTR,
    Action,
    ActionKind,
    ActionRunContext,
    create_action_key,
    parse_action_key,
    parse_dap_qualified_name,
)
from genkit._core._background import (
    _ensure_operation,
    missing_operation_error,
    stamp_operation_action,
)
from genkit._core._error import GenkitError, GenkitRuntimeError, PublicError, RuntimeErrorReason
from genkit._core._logger import get_logger, is_debug_enabled
from genkit._core._middleware import (
    BaseMiddleware,
    GenerateHookParams,
    GenerateMiddleware,
    GenerateMiddlewareContext,
    MiddlewareDef,
    ModelHookParams,
    ToolHookParams,
    _copy_middleware_instance,
    middleware_class_index,
)
from genkit._core._model import (
    ABNORMAL_FINISH_REASONS,
    Document,
    GenerateActionOptions,
    MultipartToolResponse,
    OutputConfig,
    Part,
    as_message,
)
from genkit._core._protocols import RegistryLike, SessionLike
from genkit._core._registry import Registry
from genkit._core._schema import check_output_schema
from genkit._core._tracing import SpanMetadata, run_in_new_span
from genkit._core._typing import (
    Error,
    FinishReason,
    GenerateActionOutputConfig,
    MiddlewareRef,
    Operation,
    Role,
    ToolDefinition,
    ToolRequest,
    ToolResponse,
)

DEFAULT_MAX_TURNS = 50

logger = get_logger(__name__)

HookParamsT = TypeVar('HookParamsT')
HookResultT = TypeVar('HookResultT')
HookWrap = Callable[
    [
        HookParamsT,
        GenerateMiddlewareContext,
        Callable[[HookParamsT, GenerateMiddlewareContext], Awaitable[HookResultT]],
    ],
    Awaitable[HookResultT],
]


class StreamingCallbackError(Exception):
    """Carries a caller callback failure through generate's failure handling."""

    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.cause = cause


def streaming_callback_cause(*, exc: BaseException) -> Exception | None:
    """Find the caller exception carried through action error wrappers."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, StreamingCallbackError):
            return current.cause
        if isinstance(current, GenkitError) and current.cause is not None:
            current = current.cause
        else:
            current = current.__cause__
    return None


class ModelContractError(GenkitError):
    """A model action returned a value its registered kind cannot use."""


def log_output_parse(
    *,
    model: str | None,
    finish_reason: FinishReason | None,
    finish_message: str | None,
    formatter: Formatter[Any, Any] | None,
    message: Message | None,
) -> None:
    """Warn on an abnormal finish; debug when the formatter cannot parse."""
    if formatter is None:
        return
    if finish_reason in ABNORMAL_FINISH_REASONS:
        logger.warning(
            'model finished abnormally, skipping output parsing',
            model=model,
            finishReason=finish_reason,
            finishMessage=finish_message,
        )
        return
    if message is None or not is_debug_enabled(logger):
        return
    try:
        formatter.parse_message(message)
    except Exception as e:
        logger.debug(
            'model output does not match the expected schema',
            model=model,
            error=e,
        )


def middleware_name(mw: MiddlewareDef) -> str:
    """Class name is what shows up on hook log records."""
    return type(mw).__name__


def hook_finished(
    *,
    name: str,
    hook: str,
    start: float,
    next_called: bool,
    error: str | None,
    extra: dict[str, object],
) -> dict[str, object]:
    """Attributes for the ``middleware hook finished`` record."""
    ms = max(0, round((time.monotonic() - start) * 1000))
    out: dict[str, object] = {
        'middleware': name,
        'hook': hook,
        'duration': f'{ms}ms',
        **extra,
    }
    if not next_called:
        out['short_circuited'] = True
    if error is not None:
        out['error'] = error
    return out


async def run_logged_hook(
    *,
    mw: MiddlewareDef,
    hook: str,
    params: HookParamsT,
    ctx: GenerateMiddlewareContext,
    wrap: HookWrap[HookParamsT, HookResultT],
    inner: Callable[[HookParamsT, GenerateMiddlewareContext], Awaitable[HookResultT]],
    extra: dict[str, object] | None = None,
) -> HookResultT:
    """Run one middleware hook, with started/finished records when debug is on."""
    if not is_debug_enabled(logger):
        return await wrap(params, ctx, inner)
    attrs = extra or {}
    name = middleware_name(mw)
    logger.debug('middleware hook started', middleware=name, hook=hook, **attrs)
    start = time.monotonic()
    seen = {'next_called': False, 'error': None}

    async def tracked(tp: HookParamsT, tc: GenerateMiddlewareContext) -> HookResultT:
        seen['next_called'] = True
        return await inner(tp, tc)

    try:
        return await wrap(params, ctx, tracked)
    except BaseException as e:
        seen['error'] = str(e) or type(e).__name__
        raise
    finally:
        logger.debug(
            'middleware hook finished',
            **hook_finished(
                name=name,
                hook=hook,
                start=start,
                next_called=seen['next_called'],
                error=seen['error'],
                extra=attrs,
            ),
        )


class ScopedGenkitView:
    """A GenkitLike view over the call-scoped registry for one generate invocation.

    Middleware reads ``ctx.ai.registry`` expecting the per-call child registry
    (with this call's middleware/tool registrations), not the global one, so we
    hand it this thin wrapper instead of the full Genkit veneer.
    """

    def __init__(self, reg: RegistryLike) -> None:
        self.registry: RegistryLike = reg

    def current_session(self) -> SessionLike | None:
        return get_current_session()


def register_middleware(
    registry: Registry,
    use: Sequence[BaseMiddleware | MiddlewareRef] | None,
) -> list[MiddlewareRef] | None:
    """Normalize ``use=`` to ``MiddlewareRef`` entries (name + config only).

    Inline ``BaseMiddleware`` instances are not stored on the registry. Their
    config is serialized onto the ref and, when the class is not registered on
    a parent registry, a ``GenerateMiddleware`` is registered on this layer so
    ``resolve_middleware_from_use`` can build a fresh instance per ``generate()``.
    """
    if use is None:
        return None
    refs: list[MiddlewareRef] = []
    # Track how many times each name appears so duplicates get unique suffixes.
    name_counts: dict[str, int] = {}
    # Build the class→name index once so resolving the use list is O(M+N).
    cls_index = middleware_class_index(registry)
    for i, entry in enumerate(use):
        if isinstance(entry, BaseMiddleware):
            # Prefer the registered name so traces show ``concise_reply_mw``
            # instead of an opaque id. For an unregistered ``use=[Foo()]``
            # passed inline, fall back to a synthetic id that can't collide
            # with any globally registered middleware.
            mw_cls = type(entry)
            registered = cls_index.get(mw_cls)
            base_name = registered or f'dynamic-middleware-{i}-{secrets.token_hex(5)}'
            count = name_counts.get(base_name, 0)
            name_counts[base_name] = count + 1
            reg_name = base_name if count == 0 else f'{base_name}__{count}'
            if registered is None and registry.lookup_value('middleware', reg_name) is None:
                registry.register_value(
                    'middleware',
                    reg_name,
                    GenerateMiddleware(cls=mw_cls, name=reg_name),
                )
            config = cast(BaseModel, entry.config).model_dump(exclude_none=True, mode='json') or None
            refs.append(MiddlewareRef(name=reg_name, config=config))
        else:
            refs.append(entry)
    return refs


def resolve_middleware_from_use(
    registry: Registry,
    use: Sequence[MiddlewareRef] | None,
) -> list[BaseMiddleware]:
    """Resolve ``MiddlewareRef`` entries to fresh ``BaseMiddleware`` instances.

    Each ref is instantiated from the registered ``GenerateMiddleware`` and
    ``ref.config`` (same path for Dev UI, dotprompt, and inline ``use=[Mw(...)]``).
    """
    if not use:
        return []
    out: list[BaseMiddleware] = []
    for entry in use:
        defn = registry.lookup_value('middleware', entry.name)
        if defn is None:
            raise GenkitError(
                status='NOT_FOUND',
                message=(
                    f'A middleware with the name "{entry.name}" cannot be found. '
                    'Register it via @ai.middleware(...), a middleware plugin, or pass '
                    'a BaseMiddleware instance in use= so the framework can normalize it.'
                ),
                source='genkit.generate',
                reason=RuntimeErrorReason.INVALID_INPUT,
            )
        if not isinstance(defn, GenerateMiddleware):
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=(
                    f'Middleware "{entry.name}" is registered with the wrong type '
                    f'({type(defn).__name__}). Expected GenerateMiddleware from '
                    '@ai.middleware(...), a middleware plugin, or inline use= normalization.'
                ),
                source='genkit.generate',
                reason=RuntimeErrorReason.INVALID_INPUT,
            )
        cfg = entry.config if isinstance(entry.config, dict) else None
        out.append(defn.instantiate(cfg))
    return out


@dataclass
class MiddlewarePipeline:
    """Holds the middleware chain and the shared context for a single generate call."""

    middleware: list[MiddlewareDef]
    ctx: GenerateMiddlewareContext


def prepare_middleware(
    middleware: list[BaseMiddleware],
    *,
    ctx: GenerateMiddlewareContext,
) -> MiddlewarePipeline:
    """Return per-call middleware defs sharing one ``GenerateMiddlewareContext``."""
    return MiddlewarePipeline(
        middleware=[_copy_middleware_instance(mw) for mw in middleware],
        ctx=ctx,
    )


def hook_wrap(mw: MiddlewareDef, hook: str) -> HookWrap[HookParamsT, HookResultT]:
    if hook == 'generate':
        wrap = mw.wrap_generate
    elif hook == 'model':
        wrap = mw.wrap_model
    elif hook == 'tool':
        wrap = mw.wrap_tool
    else:
        raise ValueError(f'unknown middleware hook {hook!r}')
    # Same (params, ctx, next) shape on every hook; params type differs.
    return cast(HookWrap[HookParamsT, HookResultT], wrap)


async def dispatch_hooks(
    *,
    middleware: list[MiddlewareDef],
    hook: str,
    params: HookParamsT,
    ctx: GenerateMiddlewareContext,
    next_fn: Callable[[HookParamsT, GenerateMiddlewareContext], Awaitable[HookResultT]],
    extra: Callable[[HookParamsT], dict[str, object] | None] | None = None,
    after_result: Callable[[HookResultT], None] | None = None,
) -> HookResultT:
    """Run wrap_{hook} outside-in, then next_fn.

    ``after_result`` runs after next_fn and after each hook that returns.
    A hook that raises leaves the last successful result in place.
    """

    def with_after_result(
        fn: Callable[[HookParamsT, GenerateMiddlewareContext], Awaitable[HookResultT]],
    ) -> Callable[[HookParamsT, GenerateMiddlewareContext], Awaitable[HookResultT]]:
        if after_result is None:
            return fn
        callback = after_result

        async def stamped(p: HookParamsT, c: GenerateMiddlewareContext) -> HookResultT:
            result = await fn(p, c)
            callback(result)
            return result

        return stamped

    runner = with_after_result(next_fn)
    for mw in reversed(middleware):
        wrap = hook_wrap(mw, hook)

        async def run_next(
            p: HookParamsT,
            c: GenerateMiddlewareContext,
            _mw: MiddlewareDef = mw,
            _inner: Callable[[HookParamsT, GenerateMiddlewareContext], Awaitable[HookResultT]] = runner,
            _wrap: HookWrap[HookParamsT, HookResultT] = wrap,
        ) -> HookResultT:
            return await run_logged_hook(
                mw=_mw,
                hook=hook,
                params=p,
                ctx=c,
                wrap=_wrap,
                inner=_inner,
                extra=extra(p) if extra is not None else None,
            )

        runner = with_after_result(run_next)
    return await runner(params, ctx)


async def dispatch_tool(
    *,
    middleware: list[MiddlewareDef],
    params: ToolHookParams,
    ctx: GenerateMiddlewareContext,
    next_fn: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]],
) -> MultipartToolResponse:
    """Chain wrap_tool middleware and call next_fn."""
    return await dispatch_hooks(
        middleware=middleware,
        hook='tool',
        params=params,
        ctx=ctx,
        next_fn=next_fn,
        extra=lambda p: {'tool': p.tool.name},
    )


async def expand_wildcard_tools(registry: Registry, tool_names: list[str]) -> list[str]:
    """Bind ``provider:tool/…`` selectors to ``/tool.v2/<name>`` catalog keys.

    People write ``mcp:tool/echo`` or ``mcp:tool/*``. We resolve the ``tool``
    bucket, register each Action on ``registry`` (the generate child), and
    return the same key a local tool uses.
    """
    expanded: list[str] = []
    for name in tool_names:
        qualified = parse_dap_qualified_name(name)
        if qualified is None or qualified.inner_kind != 'tool':
            expanded.append(name)
            continue

        provider_action = await registry.resolve_action(
            ActionKind.DYNAMIC_ACTION_PROVIDER,
            qualified.provider,
        )
        if provider_action is None:
            expanded.append(name)
            continue

        dap = getattr(provider_action, GENKIT_DYNAMIC_ACTION_PROVIDER_ATTR, None)
        if dap is None:
            expanded.append(name)
            continue

        metas = await dap.list_action_metadata('tool', qualified.inner_name)
        if not metas:
            expanded.append(name)
            continue
        for meta in metas:
            tool_name = meta.get('name')
            if not tool_name:
                continue
            action = await dap.get_action('tool', str(tool_name))
            if action is None:
                continue
            registry.register_action_from_instance(action)
            expanded.append(create_action_key(ActionKind.TOOL, action.name))

    return expanded


def tools_to_action_names(
    tools: Sequence[str | Tool] | None,
) -> list[str] | None:
    """Normalize tool arguments to registry names for GenerateActionOptions.

    Each item may be a tool name (``str``) or a Tool returned by
    Genkit.tool().
    """
    if tools is None:
        return None
    names: list[str] = []
    for t in tools:
        if isinstance(t, str):
            names.append(t)
        else:
            names.append(t.name)
    return names


async def register_tools(registry: Registry, tools: Sequence[str | Tool] | None) -> None:
    """Creates a child registry and ensures that all tools are registered.

    Supports dynamically defined tools that are only passed in at call time
    and never actually registered.
    """
    if not tools:
        return
    for t in tools:
        if not isinstance(t, Tool):
            continue
        # If the same action is already reachable through the parent chain,
        # skip — re-registering would either no-op or trigger a duplicate.
        resolved = await registry.resolve_action(ActionKind.TOOL, t.name)
        if resolved is t.action():
            continue
        registry.register_action_from_instance(t.action())


CONTEXT_PREFACE = '\n\nUse the following information to complete your task:\n\n'


def last_user_message(*, messages: list[Message]) -> Message | None:
    """Find the last user message in a list."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == 'user':
            return messages[i]
    return None


def context_item_template(d: Document, index: int) -> str:
    """Render a document as a citation line for context injection."""
    out = '- '
    ref = (d.metadata and (d.metadata.get('ref') or d.metadata.get('id'))) or index
    out += f'[{ref}]: '
    out += text_from_content(d.content) + '\n'
    return out


def augment_with_context(
    request: ModelRequest,
    *,
    preface: str | None = CONTEXT_PREFACE,
    item_template: Callable[[Document, int], str] | None = None,
    citation_key: str | None = None,
) -> ModelRequest:
    """Return a deepcopy of ``request`` with ``request.docs`` injected as a context part on the last user message.

    No-op (returns ``request`` unchanged) when there are no docs, no user message, or the last user message
    already has a non-pending ``purpose: 'context'`` part.
    """
    if not request.docs:
        return request

    user_message = last_user_message(messages=request.messages)
    if user_message is None:
        return request

    # Find any existing context part in the last user message
    context_idx = -1
    for i, part in enumerate(user_message.content):
        metadata = part.metadata or {}
        if metadata.get('purpose') == 'context':
            context_idx = i
            break

    # If context already exists, only proceed if it is a pending placeholder
    if context_idx >= 0:
        meta = user_message.content[context_idx].metadata or {}
        if not meta.get('pending'):
            return request

    # Render all documents as a single formatted text string
    template = item_template or context_item_template
    rendered_docs = []
    for i, doc_data in enumerate(request.docs):
        doc = Document(content=doc_data.content, metadata=doc_data.metadata)
        if citation_key and doc.metadata:
            doc.metadata['ref'] = doc.metadata.get(citation_key, i)
        rendered_docs.append(template(doc, i))

    text_content = (preface or '') + ''.join(rendered_docs) + '\n'
    text_part = Part.from_text(text_content, metadata={'purpose': 'context'})

    # Safe-mutation via deep copy
    new_req = copy.deepcopy(request)
    new_user = last_user_message(messages=new_req.messages)
    assert new_user is not None

    if context_idx >= 0:
        new_user.content[context_idx] = text_part
    else:
        new_user.content.append(text_part)

    return new_req


def raise_if_aborted(abort_signal: asyncio.Event) -> None:
    if abort_signal.is_set():
        raise GenkitError(status='ABORTED', message='Generation aborted.')


def define_generate_action(registry: Registry) -> None:
    """Register the generation action triggered by the Dev UI."""

    async def generate_action_fn(
        input: GenerateActionOptions,
        ctx: ActionRunContext,
    ) -> ModelResponse:
        on_chunk = cast(Callable[[ModelResponseChunk], None], ctx.streaming_callback) if ctx.is_streaming else None
        return await run_generate(
            registry=registry,
            options=input,
            abort_signal=ctx.abort_signal,
            on_chunk=on_chunk,
            context=dict(ctx.context),
        )

    _ = registry.register_action(
        kind=ActionKind.UTIL,
        name='generate',
        fn=generate_action_fn,
    )


async def generate_action(
    registry: Registry,
    options: GenerateActionOptions,
    on_chunk: Callable[[ModelResponseChunk], None] | None = None,
    message_index: int = 0,
    current_turn: int = 0,
    context: dict[str, Any] | None = None,
    abort_signal: asyncio.Event | None = None,
) -> ModelResponse:
    """Open the user-facing ``generate`` span and delegate to the engine.

    Thin wrapper so in-process callers get a trace span named ``generate``
    around the whole call.  The registered ``/util/generate`` action skips
    this wrapper because the action runtime already opens its own span.
    """
    span_name = 'generate'
    span_metadata = SpanMetadata(name=span_name, type='util', input=options)
    with run_in_new_span(span_metadata) as span:
        result = await run_generate(
            registry=registry,
            options=options,
            abort_signal=abort_signal,
            on_chunk=on_chunk,
            message_index=message_index,
            current_turn=current_turn,
            context=context,
        )
        with contextlib.suppress(Exception):
            span.set_attribute('genkit:output', result.model_dump_json(by_alias=True, exclude_none=True))
        if result.error is not None:
            span_metadata.state = 'error'
        return result


async def run_generate(
    *,
    registry: Registry,
    options: GenerateActionOptions,
    on_chunk: Callable[[ModelResponseChunk], None] | None = None,
    message_index: int = 0,
    current_turn: int = 0,
    context: dict[str, Any] | None = None,
    abort_signal: asyncio.Event | None = None,
) -> ModelResponse:
    """Resolve ``options.use`` and run the generation.

    Core generate business logic. `ai.generate` veneer and the registered
    `/util/generate` action funnel through here.
    """
    # Shallow-copy the wire-shape struct so per-field updates below (and any
    # future mutations) don't leak back to the caller's ``options``.
    # Empty messages is a valid start (model speaks first); normalize None here
    # so the rest of generate can treat the field as a list.
    options = options.model_copy(
        update={'messages': list(options.messages or [])},
    )
    if options.max_turns is not None and options.max_turns < 0:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'max turns cannot be negative, got {options.max_turns}',
            reason=RuntimeErrorReason.INVALID_INPUT,
        )
    registry = registry if registry.is_child else registry.new_child()

    if options.tools:
        options.tools = await expand_wildcard_tools(registry, options.tools)

    middleware = resolve_middleware_from_use(registry, options.use)
    caller_on_chunk = on_chunk

    def send_caller_chunk(chunk: ModelResponseChunk) -> None:
        if caller_on_chunk is None:
            return
        try:
            caller_on_chunk(chunk)
        except Exception as exc:
            raise StreamingCallbackError(exc) from exc

    ctx = GenerateMiddlewareContext(
        ai=ScopedGenkitView(registry),
        custom_context=dict(context or {}),
        on_chunk=send_caller_chunk if caller_on_chunk is not None else None,
        abort_signal=abort_signal if abort_signal is not None else asyncio.Event(),
    )

    mw_pipeline: MiddlewarePipeline | None = None
    if middleware:
        mw_pipeline = prepare_middleware(middleware, ctx=ctx)
        mw_tools: list[Action[Any, Any, Any, Any]] = []
        for mw in mw_pipeline.middleware:
            mw_tools.extend(mw.tools(mw_pipeline.ctx))

        if mw_tools:
            existing = list(options.tools) if options.tools else []
            declared = set(existing)
            contributed_names: list[str] = []
            for t in mw_tools:
                name = t.name
                if name in declared or name in contributed_names:
                    raise GenkitError(
                        status='INVALID_ARGUMENT',
                        message=(f"tool '{name}' is contributed by middleware but already declared elsewhere"),
                        reason=RuntimeErrorReason.INVALID_INPUT,
                    )
                registry.register_action_from_instance(t)
                contributed_names.append(name)
            options = options.model_copy()
            options.tools = existing + contributed_names
    else:
        mw_pipeline = MiddlewarePipeline(middleware=[], ctx=ctx)

    call = GenerateRun(messages=list(options.messages or []))

    if is_debug_enabled(logger):
        resolved: dict[str, object] = {
            'model': options.model,
            'messages': len(options.messages),
            'tools': len(options.tools or []),
            'max_turns': options.max_turns,
            'streaming': on_chunk is not None,
        }
        resolved['format'] = options.output.format if options.output else None
        resolved['constrained'] = options.output.constrained if options.output else None
        if middleware:
            resolved['middleware'] = [middleware_name(m) for m in middleware]
        logger.debug('generate request resolved', **resolved)

    try:
        return await run_wrap_generate(
            registry=registry,
            options=options,
            mw_pipeline=mw_pipeline,
            message_index=message_index,
            current_turn=current_turn,
            call=call,
        )
    except Exception as exc:
        if streaming_callback_cause(exc=exc) is None:
            raise
        return box_from_exc(
            response=ModelResponse(),
            messages=call.messages,
            exc=exc,
            caller_stopped=False,
        )


class ChunkAccumulator:
    """Tracks role and message-index state across a streaming turn's chunks.

    The message index it lands on is what seeds the next turn, so the counter
    the streaming callback bumps is the same one the tool loop reads to keep
    saved history numbered consistently.
    """

    def __init__(
        self,
        message_index: int,
        formatter: Formatter[Any, Any] | None,
        schema_type: type[BaseModel] | None = None,
    ) -> None:
        self.message_index = message_index
        self.formatter = formatter
        self.schema_type = schema_type
        self.chunk_role: Role = Role.MODEL
        self.prev_chunks: list[ModelResponseChunk[Any]] = []
        self._chunk_parser: Callable[[ModelResponseChunk[Any]], Any | None] | None = (
            formatter.parse_chunk if formatter is not None else None
        )

    def make(self, *, role: Role, chunk: ModelResponseChunk[Any]) -> ModelResponseChunk[Any]:
        """Wrap a raw chunk with metadata and track message index changes."""
        if role != self.chunk_role and len(self.prev_chunks) > 0:
            self.message_index += 1

        self.chunk_role = role

        prev_to_send = copy.copy(self.prev_chunks)
        self.prev_chunks.append(chunk)

        return ModelResponseChunk(
            chunk,
            index=self.message_index,
            previous_chunks=prev_to_send,
            chunk_parser=self._chunk_parser,
            schema_type=self.schema_type,
        )

    def stream_chunk(
        self,
        *,
        chunk: ModelResponseChunk[Any],
        role: Role,
        ctx: GenerateMiddlewareContext,
    ) -> None:
        """Send one framework-wrapped chunk through the current stream chain."""
        if ctx.on_chunk is None:
            return
        ctx.on_chunk(self.make(role=role, chunk=chunk))

    @contextlib.contextmanager
    def intercept_model_stream(
        self,
        ctx: GenerateMiddlewareContext,
        *,
        role: Role,
    ) -> Generator[None, None, None]:
        """Wrap raw model tokens for one model call, then restore the prior callback."""
        downstream = ctx.on_chunk
        if downstream is None:
            yield
            return

        def handler(chunk: ModelResponseChunk[Any]) -> None:
            if downstream is not None:
                downstream(self.make(role=role, chunk=chunk))

        previous = ctx.replace_on_chunk(handler)
        try:
            yield
        finally:
            ctx.replace_on_chunk(previous)


def box_background_start(
    *,
    raw: object,
    request: ModelRequest,
    name: str,
    latency_ms: float | None = None,
) -> ModelResponse:
    """Turn a start() Operation into the ModelResponse wrap_model reads.

    Timing comes from Action.run, not from the ticket.
    """
    op = _ensure_operation(response=raw, name=name)
    stamp_operation_action(operation=op, name=name)
    return ModelResponse(operation=op, request=request, latency_ms=latency_ms)


def require_model_response(*, raw: object, name: str) -> ModelResponse:
    """A chat model returns a ModelResponse, not a dict or a job handle."""
    if isinstance(raw, Operation) or (isinstance(raw, ModelResponse) and raw.operation is not None):
        raise ModelContractError(
            status='FAILED_PRECONDITION',
            message=(
                f"Model '{name}' is a regular model that returns a response immediately. "
                'Use define_background_model for background models that return operations.'
            ),
        )
    return as_model_response(raw=raw, name=name)


def as_model_response(*, raw: object, name: str) -> ModelResponse:
    """A hook or model returns a ModelResponse they can read, not a dict."""
    if not isinstance(raw, ModelResponse):
        raise ModelContractError(
            status='FAILED_PRECONDITION',
            message=f"Model '{name}' did not return a ModelResponse.",
        )
    walked = ModelResponse(
        message=raw.message,
        error=raw.error,
        finish_reason=raw.finish_reason,
        finish_message=raw.finish_message,
        latency_ms=raw.latency_ms,
        usage=raw.usage,
        custom=raw.custom,
        raw=raw.raw,
        request=raw.request,
        operation=raw.operation,
        candidates=raw.candidates,
    )
    walked._message_parser = raw._message_parser
    walked._schema_type = raw._schema_type
    return walked


@dataclass
class ResolvedTurn:
    """The model and tools they named, looked up before any hook runs."""

    model: Action
    tools: list[Action]
    formatter: Formatter[Any, Any] | None = None


@dataclass
class GenerateRun:
    """One generate call.

    ``messages`` and ``output`` last the whole call. ``ticket`` and
    ``last_response`` are this wrap_generate — cleared when the next
    turn starts. ``remember`` feeds the box if a later hook raises.
    """

    messages: list[Message]
    ticket: ModelResponse | None = None
    last_response: ModelResponse | None = None
    output: GenerateActionOutputConfig | None = None

    def set_messages(self, messages: list[Message]) -> None:
        self.messages = list(messages)

    def remember(self, result: ModelResponse) -> None:
        # after_result runs once per hook. A later hook that rewrites this
        # turn replaces that message so history is not two model turns in a row.
        if (
            result.message is not None
            and self.last_response is not None
            and self.last_response.message is not None
            and self.messages
            and self.messages[-1] == self.last_response.message
        ):
            self.messages = [*self.messages[:-1], result.message]
        else:
            self.messages = history_with_closed_turn(messages=self.messages, result=result)
        self.last_response = result


def history_with_closed_turn(*, messages: list[Message], result: ModelResponse) -> list[Message]:
    """If this response closed a model turn, that turn is on the history they can resend."""
    if result.message is None:
        return list(messages)
    if messages and messages[-1] == result.message:
        return list(messages)
    return [*messages, result.message]


def request_with_messages(*, request: ModelRequest, messages: list[Message]) -> ModelRequest:
    return request.model_copy(update={'messages': list(messages)})


def output_config_from(out: GenerateActionOutputConfig) -> OutputConfig:
    return OutputConfig(
        format=out.format,
        # pyrefly: ignore[unexpected-keyword] - populate_by_name accepts the field name
        json_schema=out.json_schema,
        constrained=out.constrained,
        content_type=out.content_type,
    )


def attach_resendable_history(response: ModelResponse, messages: list[Message]) -> ModelResponse:
    """The closed history they resend, not the request the model saw."""
    if response.request is not None:
        response.request = request_with_messages(request=response.request, messages=messages)
    return response


def box_dead_turn(
    *,
    response: ModelResponse,
    messages: list[Message],
    finish_reason: FinishReason,
    finish_message: str,
    error: GenkitRuntimeError,
) -> ModelResponse:
    """Stop before this turn closed: only completed rounds stay.

    The unanswered model call is dropped so the caller can send the
    history again.
    """
    # Hooks and the model may still hold the response they returned, so
    # finish_reason, error, and a cleared message go on a copy.
    out = response.model_copy()
    out.finish_reason = finish_reason
    out.finish_message = finish_message
    out.error = error
    out.message = None
    if out.operation is not None and out.operation.error is None:
        # The ticket already started. The failure why lives on the
        # handle so check/cancel is not a clean start.
        out.operation = out.operation.model_copy(update={'error': Error(message=finish_message)})
    if out.request is None:
        out.request = ModelRequest(messages=list(messages))
    return attach_resendable_history(out, messages)


INTERNAL_FINISH_MESSAGE = 'internal error'


def public_error(exc: BaseException) -> PublicError | None:
    if isinstance(exc, PublicError):
        return exc
    if isinstance(exc, GenkitError) and isinstance(exc.cause, PublicError):
        return exc.cause
    return None


def boxed_finish_message(*, exc: BaseException, pipe_failed: bool) -> str:
    # The string on a returned response is what a flow can put in a 200.
    # PublicError is how a tool author publishes that sentence.
    if pipe_failed:
        if isinstance(exc, GenkitError):
            return exc.original_message or type(exc).__name__
        return str(exc) or type(exc).__name__
    published = public_error(exc)
    if published is not None:
        return published.original_message or INTERNAL_FINISH_MESSAGE
    if isinstance(exc, GenkitError) and (
        exc.cause is None or isinstance(exc.cause, GenkitError) or exc.status != 'INTERNAL'
    ):
        return exc.original_message or INTERNAL_FINISH_MESSAGE
    return INTERNAL_FINISH_MESSAGE


def raise_if_foreign_cancel(*, exc: BaseException, abort_signal: asyncio.Event) -> None:
    # abort_signal is our own stop. A wait_for / task.cancel() has to
    # surface as CancelledError so the deadline still works.
    if isinstance(exc, asyncio.CancelledError) and not abort_signal.is_set():
        raise


def box_from_exc(
    *,
    response: ModelResponse,
    messages: list[Message],
    exc: BaseException,
    caller_stopped: bool,
    reason: RuntimeErrorReason | None = None,
) -> ModelResponse:
    """Box a failure after generate has entered: closed history, unanswered turn dropped."""
    callback_cause = streaming_callback_cause(exc=exc)
    pipe_failed = False
    if callback_cause is not None:
        # The stream pipe is framework plumbing, not a tool or model
        # sentinel. Keep the sink's wording; drop any loop reason.
        exc = callback_cause
        reason = None
        pipe_failed = True
    finish_message = boxed_finish_message(exc=exc, pipe_failed=pipe_failed)
    if caller_stopped:
        finish_message = 'Generation aborted.'
        status = 'CANCELLED'
        details = exc.details if isinstance(exc, GenkitError) else None
    elif pipe_failed:
        status = 'INTERNAL'
        details = None
    elif isinstance(exc, GenkitError):
        status = exc.status
        details = exc.details
    else:
        status = 'INTERNAL'
        details = None
    if reason is not None and not caller_stopped:
        details = dict(details) if isinstance(details, Mapping) else {}
        details['reason'] = reason.value
    return box_dead_turn(
        response=response,
        messages=messages,
        finish_reason=FinishReason.ABORTED if caller_stopped else FinishReason.FAILED,
        finish_message=finish_message,
        error=GenkitRuntimeError(status=status, message=finish_message, details=details),
    )


def box_if_hook_dropped_ticket(
    *,
    ticket: ModelResponse,
    after_hooks: ModelResponse,
    messages: list[Message],
    name: str,
) -> ModelResponse | None:
    """start() already billed a ticket. Dropping it orphans the job.

    Keep the handle so they can still check or cancel.
    """
    if ticket.operation is not None and after_hooks.operation is None:
        return box_from_exc(
            response=ticket,
            messages=messages,
            exc=missing_operation_error(name=name),
            caller_stopped=False,
        )
    return None


@dataclass(frozen=True)
class ContinueTurn:
    """A closed tool round. wrap_generate for the next turn uses these options."""

    options: GenerateActionOptions
    messages: list[Message]
    message_index: int


def message_with_output_meta(*, message: Message, output: GenerateActionOutputConfig) -> Message:
    """Format metadata so the Dev UI can render formatted JSON vs plain text."""
    if not (output.content_type or output.format):
        return message
    generate_output: dict[str, str] = {}
    if output.content_type:
        generate_output['contentType'] = output.content_type
    if output.format:
        generate_output['format'] = output.format
    existing_meta = dict(message.metadata) if isinstance(message.metadata, dict) else {}
    generate_meta = existing_meta.get('generate')
    if not isinstance(generate_meta, dict):
        generate_meta = {}
    generate_meta['output'] = generate_output
    existing_meta['generate'] = generate_meta
    return message.model_copy(update={'metadata': existing_meta})


def tool_requests_on(message: Message) -> list[Part]:
    return [part for part in message.content if part.tool_request is not None]


def log_model_responded(
    *,
    model: str | None,
    turn: int,
    response: ModelResponse,
    tool_requests: int,
) -> None:
    if not is_debug_enabled(logger):
        return
    responded: dict[str, object] = {
        'model': model,
        'turn': turn,
        'finish_reason': response.finish_reason,
        'tool_requests': tool_requests,
    }
    if response.usage is not None:
        responded['input_tokens'] = response.usage.input_tokens
        responded['output_tokens'] = response.usage.output_tokens
    logger.debug('model responded', **responded)


async def resolve_door(
    *,
    registry: Registry,
    options: GenerateActionOptions,
    abort_signal: asyncio.Event,
) -> tuple[GenerateActionOptions, ResolvedTurn]:
    """Look up the model and tools they named. Raises if a chat cannot start."""
    turn_model, turn_tools, format_def = await resolve_parameters(registry=registry, options=options)
    if turn_model.kind == ActionKind.BACKGROUND_MODEL and options.resume is not None:
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message=(
                f"Cannot resume background model '{turn_model.name}'; "
                'a background start cannot satisfy an interrupted tool turn'
            ),
            reason=RuntimeErrorReason.INVALID_RESUME,
        )
    options, formatter = apply_format(options=options, format_def=format_def)
    if options.resources:
        options = await apply_resources(registry=registry, options=options, abort_signal=abort_signal)
    assert_valid_tool_names(turn_tools)
    return options, ResolvedTurn(model=turn_model, tools=turn_tools, formatter=formatter)


async def run_wrap_generate(
    *,
    registry: Registry,
    options: GenerateActionOptions,
    mw_pipeline: MiddlewarePipeline,
    message_index: int,
    current_turn: int,
    call: GenerateRun,
    resolved: ResolvedTurn | None = None,
) -> ModelResponse:
    """One wrap_generate. Door on the first call; generate_turn inside.

    Raise before a request exists. After that they always get a ModelResponse.
    ``ModelResponse.messages`` is closed history. The unanswered turn is
    dropped. If wrap_generate raises, the box uses ``call.last_response`` and
    ``call.messages``.
    """
    ctx = mw_pipeline.ctx
    call.set_messages(options.messages or [])
    if ctx.abort_signal.is_set():
        return box_dead_turn(
            response=ModelResponse(),
            messages=call.messages,
            finish_reason=FinishReason.ABORTED,
            finish_message='Generation aborted.',
            error=GenkitRuntimeError(status='CANCELLED', message='Generation aborted.'),
        )

    if resolved is None:
        options, resolved = await resolve_door(
            registry=registry,
            options=options,
            abort_signal=ctx.abort_signal,
        )
        call.set_messages(options.messages or [])

    call.ticket = None
    call.last_response = None
    call.output = options.output

    async def run_turn(
        params: GenerateHookParams,
        ctx: GenerateMiddlewareContext,
    ) -> ModelResponse:
        return await generate_turn(
            params=params,
            ctx=ctx,
            registry=registry,
            resolved=resolved,
            call=call,
            mw_pipeline=mw_pipeline,
            current_turn=current_turn,
        )

    def remember_response(result: object) -> None:
        if isinstance(result, ModelResponse):
            call.remember(result)

    try:
        response = as_model_response(
            raw=await dispatch_hooks(
                middleware=mw_pipeline.middleware,
                hook='generate',
                params=GenerateHookParams(
                    options=options,
                    iteration=current_turn,
                    message_index=message_index,
                ),
                ctx=ctx,
                next_fn=run_turn,
                extra=lambda p: {'iteration': p.iteration},
                after_result=remember_response,
            ),
            name=resolved.model.name,
        )
    except (Exception, asyncio.CancelledError) as exc:
        raise_if_foreign_cancel(exc=exc, abort_signal=ctx.abort_signal)
        return box_from_exc(
            response=call.last_response if call.last_response is not None else ModelResponse(),
            messages=call.messages,
            exc=exc,
            caller_stopped=ctx.abort_signal.is_set(),
        )
    dropped = (
        box_if_hook_dropped_ticket(
            ticket=call.ticket,
            after_hooks=response,
            messages=call.messages,
            name=resolved.model.name,
        )
        if call.ticket is not None
        else None
    )
    if dropped is not None:
        return dropped
    return stamp_output(
        response=response,
        options=options,
        call=call,
        formatter=resolved.formatter,
    )


async def generate_turn(
    *,
    params: GenerateHookParams,
    ctx: GenerateMiddlewareContext,
    registry: Registry,
    resolved: ResolvedTurn,
    call: GenerateRun,
    mw_pipeline: MiddlewarePipeline,
    current_turn: int,
) -> ModelResponse:
    """Resume if they asked, call the model, run tools or stop.

    A closed tool round calls run_wrap_generate again with the same door Actions.
    """
    options = params.options
    call.output = options.output

    options, paused, resumed_tool_message = await resolve_resume_options(
        options=options,
        mw_pipeline=mw_pipeline,
        tools=resolved.tools,
    )
    if paused:
        # The restart paused again. They can answer it the same
        # way as the first interrupt.
        return paused
    call.set_messages(options.messages or [])

    chunks = ChunkAccumulator(
        params.message_index,
        resolved.formatter,
        schema_type=options.output.schema_type if options.output else None,
    )
    if resumed_tool_message:
        chunks.stream_chunk(
            chunk=ModelResponseChunk(
                role=resumed_tool_message.role,
                content=resumed_tool_message.content,
            ),
            role=Role.TOOL,
            ctx=mw_pipeline.ctx,
        )

    try:
        response = await call_model(
            options=options,
            resolved=resolved,
            call=call,
            chunks=chunks,
            ctx=ctx,
            mw_pipeline=mw_pipeline,
            current_turn=current_turn,
        )
    except (Exception, asyncio.CancelledError) as exc:
        raise_if_foreign_cancel(exc=exc, abort_signal=ctx.abort_signal)
        return box_from_exc(
            response=call.ticket if call.ticket is not None else ModelResponse(),
            messages=call.messages,
            exc=exc,
            caller_stopped=ctx.abort_signal.is_set(),
        )
    dropped = (
        box_if_hook_dropped_ticket(
            ticket=call.ticket,
            after_hooks=response,
            messages=call.messages,
            name=resolved.model.name,
        )
        if call.ticket is not None
        else None
    )
    if dropped is not None:
        return dropped

    done = stop_after_model(
        response=response,
        options=options,
        current_turn=current_turn,
        formatter=resolved.formatter,
    )
    if done is not None:
        return done

    generated_msg = response.message
    assert generated_msg is not None
    after_tools = await run_tools_or_stop(
        response=response,
        generated_msg=generated_msg,
        tool_requests=tool_requests_on(generated_msg),
        options=options,
        resolved=resolved,
        registry=registry,
        ctx=ctx,
        mw_pipeline=mw_pipeline,
        current_turn=current_turn,
        chunks=chunks,
        call=call,
    )
    if isinstance(after_tools, ModelResponse):
        return after_tools
    # Tools already ran. This is the conversation if a later pipe fails.
    call.set_messages(after_tools.messages)
    return await run_wrap_generate(
        registry=registry,
        options=after_tools.options,
        mw_pipeline=mw_pipeline,
        current_turn=current_turn + 1,
        message_index=after_tools.message_index,
        call=call,
        resolved=resolved,
    )


async def call_model(
    *,
    options: GenerateActionOptions,
    resolved: ResolvedTurn,
    call: GenerateRun,
    chunks: ChunkAccumulator,
    ctx: GenerateMiddlewareContext,
    mw_pipeline: MiddlewarePipeline,
    current_turn: int,
) -> ModelResponse:
    """wrap_model, then the action they named.

    A background start stamps ``call.ticket`` at Action.run — that is when
    the job is billed.
    """
    turn_model = resolved.model
    request = await to_model_request(options=options, tools=resolved.tools, model=turn_model)
    if request.docs:
        request = augment_with_context(request)

    async def run_action(params: ModelHookParams, c: GenerateMiddlewareContext) -> ModelResponse:
        if is_debug_enabled(logger):
            logger.debug(
                'calling model',
                model=options.model,
                turn=current_turn,
                messages=len(params.request.messages),
            )
        result = await turn_model.run(
            input=params.request,
            context=c.custom_context,
            on_chunk=c.on_chunk,
            abort_signal=c.abort_signal,
        )
        raw = result.response
        if turn_model.kind == ActionKind.BACKGROUND_MODEL:
            call.ticket = box_background_start(
                raw=raw,
                request=params.request,
                name=turn_model.name,
                latency_ms=result.latency_ms,
            )
            return call.ticket
        return require_model_response(raw=raw, name=turn_model.name)

    with chunks.intercept_model_stream(ctx, role=Role.MODEL):
        response = as_model_response(
            raw=await dispatch_hooks(
                middleware=mw_pipeline.middleware,
                hook='model',
                params=ModelHookParams(request=request),
                ctx=ctx,
                next_fn=run_action,
            ),
            name=turn_model.name,
        )
    response.request = request
    formatter = resolved.formatter
    if formatter:
        parse = formatter.parse_message
        response._message_parser = lambda msg: parse(msg)
    schema_type = options.output.schema_type if options.output else None
    if schema_type:
        response._schema_type = schema_type
    return response


def stop_after_model(
    *,
    response: ModelResponse,
    options: GenerateActionOptions,
    current_turn: int,
    formatter: Formatter[Any, Any] | None,
) -> ModelResponse | None:
    """Return when this model call is the whole turn (ticket, no tools, or they asked to stop)."""
    generated_msg = response.message
    tool_requests = tool_requests_on(generated_msg) if generated_msg is not None else []
    log_output_parse(
        model=options.model,
        finish_reason=response.finish_reason,
        finish_message=response.finish_message,
        formatter=formatter,
        message=generated_msg,
    )
    response.assert_valid()

    if response.operation is not None:
        return attach_resendable_history(response, options.messages)

    if generated_msg is None:
        response.assert_valid_schema()
        log_model_responded(
            model=options.model,
            turn=current_turn,
            response=response,
            tool_requests=len(tool_requests),
        )
        return attach_resendable_history(response, options.messages)

    if options.output is not None:
        response.message = message_with_output_meta(message=generated_msg, output=options.output)
        generated_msg = response.message

    if options.return_tool_requests or len(tool_requests) == 0:
        if len(tool_requests) == 0:
            response.assert_valid_schema()
        log_model_responded(
            model=options.model,
            turn=current_turn,
            response=response,
            tool_requests=len(tool_requests),
        )
        return attach_resendable_history(response, options.messages)
    return None


async def run_tools_or_stop(
    *,
    response: ModelResponse,
    generated_msg: Message,
    tool_requests: list[Part],
    options: GenerateActionOptions,
    resolved: ResolvedTurn,
    registry: Registry,
    ctx: GenerateMiddlewareContext,
    mw_pipeline: MiddlewarePipeline,
    current_turn: int,
    chunks: ChunkAccumulator,
    call: GenerateRun,
) -> ModelResponse | ContinueTurn:
    """Run the tools the model named, or stop (cap, missing, interrupt, dead tool)."""
    max_iters = options.max_turns if options.max_turns is not None else DEFAULT_MAX_TURNS
    if current_turn + 1 > max_iters:
        # The cap is how many tool rounds they allowed, so print 5 not 5.0.
        finish_message = f'Exceeded maximum tool call iterations ({int(max_iters)})'
        log_model_responded(
            model=options.model,
            turn=current_turn,
            response=response,
            tool_requests=len(tool_requests),
        )
        return box_dead_turn(
            response=response,
            messages=list(options.messages),
            finish_reason=FinishReason.ABORTED,
            finish_message=finish_message,
            error=GenkitRuntimeError(
                status='ABORTED',
                message=finish_message,
                details={'reason': RuntimeErrorReason.MAX_TURNS_EXCEEDED.value},
            ),
        )

    if ctx.abort_signal.is_set():
        return box_dead_turn(
            response=response,
            messages=list(options.messages),
            finish_reason=FinishReason.ABORTED,
            finish_message='Generation aborted.',
            error=GenkitRuntimeError(status='CANCELLED', message='Generation aborted.'),
        )

    known_tools = tool_map_from_actions(resolved.tools)
    missing_tool = next(
        (
            p.tool_request.name
            for p in tool_requests
            if p.tool_request is not None and p.tool_request.name not in known_tools
        ),
        None,
    )
    if missing_tool is not None:
        finish_message = f'Tool {missing_tool} not found'
        log_model_responded(
            model=options.model,
            turn=current_turn,
            response=response,
            tool_requests=len(tool_requests),
        )
        return box_dead_turn(
            response=response,
            messages=list(options.messages),
            finish_reason=FinishReason.FAILED,
            finish_message=finish_message,
            error=GenkitRuntimeError(
                status='NOT_FOUND',
                message=finish_message,
                details={'reason': RuntimeErrorReason.TOOL_NOT_FOUND.value},
            ),
        )

    try:
        revised_model_msg, tool_msg = await resolve_tool_requests(
            registry=registry,
            message=generated_msg,
            mw_pipeline=mw_pipeline,
            abort_signal=ctx.abort_signal,
            tools=resolved.tools,
        )
    except (Exception, asyncio.CancelledError) as exc:
        raise_if_foreign_cancel(exc=exc, abort_signal=ctx.abort_signal)
        return box_from_exc(
            response=response,
            messages=list(options.messages),
            exc=exc,
            caller_stopped=ctx.abort_signal.is_set(),
            reason=RuntimeErrorReason.TOOL_FAILED,
        )

    if revised_model_msg:
        logger.debug(
            'generation paused by tool interrupts',
            model=options.model,
            turn=current_turn,
        )
        interrupted_resp = response.model_copy(deep=False)
        interrupted_resp.finish_reason = FinishReason.INTERRUPTED
        interrupted_resp.finish_message = 'One or more tool calls resulted in interrupts.'
        interrupted_resp.message = as_message(revised_model_msg)
        log_model_responded(
            model=options.model,
            turn=current_turn,
            response=interrupted_resp,
            tool_requests=len(tool_requests),
        )
        return attach_resendable_history(interrupted_resp, options.messages)

    log_model_responded(
        model=options.model,
        turn=current_turn,
        response=response,
        tool_requests=len(tool_requests),
    )
    next_options = copy.copy(options)
    next_messages = copy.copy(options.messages)
    next_messages.append(generated_msg)
    if tool_msg:
        next_messages.append(tool_msg)
    next_options.messages = next_messages
    next_options.model = resolved.model.name
    # Tools already ran. This is the history they resend if the tool-chunk
    # pipe fails after that closed round.
    call.set_messages(next_messages)
    if tool_msg:
        chunks.stream_chunk(
            chunk=ModelResponseChunk(
                role=tool_msg.role,
                content=tool_msg.content,
            ),
            role=Role.TOOL,
            ctx=mw_pipeline.ctx,
        )
    return ContinueTurn(
        options=next_options,
        messages=list(next_messages),
        message_index=chunks.message_index + 1,
    )


def stamp_output(
    *,
    response: ModelResponse,
    options: GenerateActionOptions,
    call: GenerateRun,
    formatter: Formatter[Any, Any] | None,
) -> ModelResponse:
    """Put format and schema on the response they get back."""
    out = call.output
    output = output_config_from(out) if out is not None else OutputConfig()
    if response.request is None:
        response.request = ModelRequest(
            messages=list(options.messages or []),
            output=output,
        )
    else:
        response.request = response.request.model_copy(update={'output': output})
    if formatter and response._message_parser is None:
        parse = formatter.parse_message
        response._message_parser = lambda msg: parse(msg)
    if out and out.schema_type:
        response._schema_type = out.schema_type
    response.assert_valid()
    response.assert_valid_schema()
    return response


def apply_format(
    *,
    options: GenerateActionOptions,
    format_def: FormatDef | None,
) -> tuple[GenerateActionOptions, Formatter[Any, Any] | None]:
    """Apply format definition to request, injecting instructions and output config."""
    if not format_def:
        return options, None

    out_request = copy.deepcopy(options)

    formatter = format_def(options.output.json_schema if options.output else None)

    # Extract instructions - handle bool | str | None type
    # Schema allows: str (custom instructions), True (use defaults), False (disable), None (default behavior)
    raw_instructions = options.output.instructions if options.output else None
    str_instructions = raw_instructions if isinstance(raw_instructions, str) else None
    instructions = resolve_instructions(formatter=formatter, instructions=str_instructions)

    should_inject = False
    if options.output and options.output.instructions is not None:
        should_inject = bool(options.output.instructions)
    elif format_def.config.default_instructions is not None:
        should_inject = format_def.config.default_instructions
    elif instructions:
        should_inject = True

    if should_inject and instructions is not None:
        out_request.messages = inject_instructions(out_request.messages, instructions)  # type: ignore[arg-type]

    # Ensure output is set before modifying its properties
    if out_request.output is None:
        return (out_request, formatter)

    if format_def.config.constrained is not None:
        out_request.output.constrained = format_def.config.constrained
    if options.output and options.output.constrained is not None:
        out_request.output.constrained = options.output.constrained

    if format_def.config.content_type is not None:
        out_request.output.content_type = format_def.config.content_type
    if format_def.config.format is not None:
        out_request.output.format = format_def.config.format

    return (out_request, formatter)


def resolve_instructions(*, formatter: Formatter[Any, Any], instructions: str | None) -> str | None:
    """Return custom instructions if provided, otherwise use formatter defaults."""
    if instructions is not None:
        return instructions
    if not formatter:
        return None  # pyright: ignore[reportUnreachable] - defensive check
    return formatter.instructions


def extract_resource_uri(*, resource: Any) -> str | None:  # noqa: ANN401
    """Extract URI from a resource object, unwrapping Pydantic structures as needed."""
    if hasattr(resource, 'uri'):
        return resource.uri
    if hasattr(resource, 'root'):
        return extract_resource_uri(resource=resource.root)
    if hasattr(resource, 'resource'):
        return extract_resource_uri(resource=resource.resource)
    if isinstance(resource, dict) and 'uri' in resource:
        return resource['uri']
    return None


async def apply_resources(
    *,
    registry: Registry,
    options: GenerateActionOptions,
    abort_signal: asyncio.Event,
) -> GenerateActionOptions:
    """Resolve and hydrate resource parts in the request messages."""
    # Quick check if any message has a resource part
    has_resource = False
    for msg in options.messages:
        for part in msg.content:
            if part.resource:
                has_resource = True
                break
        if has_resource:
            break

    if not has_resource:
        return options

    # Resolve all declared resources
    resources = []
    if options.resources:
        resources = await resolve_resources(registry, cast(list[ResourceArgument], options.resources))

    updated_messages = []
    for msg in options.messages:
        if not any(p.resource for p in msg.content):
            updated_messages.append(msg)
            continue

        updated_content = []
        for part in msg.content:
            if not part.resource:
                updated_content.append(part)
                continue

            resource_obj = part.resource

            # Extract URI from the resource object
            # The resource can be wrapped in various Pydantic structures (Resource, Resource1, etc.)
            ref_uri = extract_resource_uri(resource=resource_obj)
            if not ref_uri:
                logger.warning(
                    f'Unable to extract URI from resource part: {type(resource_obj).__name__}. '
                    + 'Resource part will be skipped.'
                )
                continue

            # Find matching resource action
            if not resources:
                raise GenkitError(
                    status='NOT_FOUND',
                    message=f'failed to find matching resource for {ref_uri}',
                    reason=RuntimeErrorReason.INVALID_INPUT,
                )

            # Normalize to ResourceInput for matching
            resource_input = ResourceInput(uri=ref_uri)
            resource_action = await find_matching_resource(registry, resources, resource_input)

            if not resource_action:
                raise GenkitError(
                    status='NOT_FOUND',
                    message=f'failed to find matching resource for {ref_uri}',
                    reason=RuntimeErrorReason.INVALID_INPUT,
                )

            # Execute the resource
            response = await resource_action.run(
                resource_input,
                on_chunk=None,
                context=None,
                abort_signal=abort_signal,
            )

            # response.response is ResourceOutput which has .content (list of Parts)
            # It usually returns a dict if coming from dynamic_resource (model_dump called)
            output_content = None
            if hasattr(response.response, 'content'):
                output_content = response.response.content
            elif isinstance(response.response, dict) and 'content' in response.response:
                output_content = response.response['content']

            if output_content:
                updated_content.extend(output_content)

        updated_messages.append(Message(role=msg.role, content=updated_content, metadata=msg.metadata))

    # Return a new request with updated messages
    new_request = options.model_copy()
    new_request.messages = updated_messages
    return new_request


def tool_short_name(*, name: str) -> str:
    """Return the last path segment of a tool name."""
    if '/' not in name:
        return name
    return name[name.rfind('/') + 1 :]


def tool_map_from_actions(tools: list[Action]) -> dict[str, Action]:
    """Index door Actions by the name the model uses and the full action name."""
    tool_dict: dict[str, Action] = {}
    for tool_action in tools:
        tool_dict[tool_action.name] = tool_action
        short = tool_short_name(name=tool_action.name)
        if short not in tool_dict:
            tool_dict[short] = tool_action
    return tool_dict


def assert_valid_tool_names(tools: list[Action]) -> None:
    """Reject overlapping model-facing tool names before the model is called.

    Two resolved tools that share the same short name (segment after the last ``/``)
    cannot both appear in one generate request.
    """
    if not tools:
        return
    seen: dict[str, str] = {}
    for tool in tools:
        short = tool_short_name(name=tool.name)
        if short in seen:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=(f"Cannot provide two tools with the same name: '{tool.name}' and '{seen[short]}'"),
                reason=RuntimeErrorReason.INVALID_INPUT,
            )
        seen[short] = tool.name


async def resolve_tools_from_options(
    registry: Registry,
    tool_names: list[str] | None,
) -> list[Action]:
    """Expand wildcards and resolve tool actions for a list of tool names."""
    if not tool_names:
        return []
    expanded = await expand_wildcard_tools(registry, tool_names)
    actions: list[Action] = []
    for t_name in expanded:
        actions.append(await resolve_tool(registry, t_name))
    return actions


async def resolve_model_action(registry: Registry, model: str | None) -> Action:
    """Look up the generate or start action for this model name."""
    name = resolve_model_name(model=model, registry=registry)
    action = await registry.resolve_model(name)
    if action is None:
        message = f"Failed to resolve model '{name}'."
        if isinstance(name, str) and '/' not in name:
            message += " Ensure the model name includes the plugin namespace (e.g., 'plugin/model')."
        raise GenkitError(
            status='NOT_FOUND',
            message=message,
            reason=RuntimeErrorReason.MODEL_NOT_FOUND,
        )
    return action


async def resolve_parameters(
    *,
    registry: Registry,
    options: GenerateActionOptions,
) -> tuple[Action, list[Action], FormatDef | None]:
    """Resolve model, tools, and format from registry for a generation request."""
    model_action = await resolve_model_action(registry, options.model)

    # Callers pick the model and tools before any hook runs. wrap_generate
    # can wrap that call; it cannot add names we have not resolved.
    tools = await resolve_tools_from_options(registry, options.tools)

    format_def: FormatDef | None = None
    if options.output and options.output.format:
        looked_up_format = registry.lookup_value('format', options.output.format)
        if looked_up_format is None:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f'Unable to resolve format {options.output.format}',
                reason=RuntimeErrorReason.INVALID_INPUT,
            )
        format_def = cast(FormatDef, looked_up_format)

    if options.output and options.output.json_schema is not None:
        json_schema = options.output.json_schema
        if hasattr(json_schema, 'model_dump'):
            json_schema = json_schema.model_dump()
        if isinstance(json_schema, dict):
            check_output_schema(json_schema)

    return (model_action, tools, format_def)


async def to_model_request(
    *,
    options: GenerateActionOptions,
    tools: list[Action],
    model: Action,
) -> ModelRequest[Any]:
    """Convert GenerateActionOptions to a ModelRequest with tool definitions."""
    # TODO(#4340): add warning when tools are not supported in ModelInfo
    # TODO(#4341): add warning when toolChoice is not supported in ModelInfo

    tool_defs = [to_tool_definition(tool) for tool in tools] if tools else []
    output = options.output
    out_schema = output.json_schema if output else None
    if out_schema is not None and hasattr(out_schema, 'model_dump'):
        out_schema = out_schema.model_dump()
    request_kwargs: dict[str, Any] = dict(
        messages=options.messages,
        config=options.config if options.config is not None else {},
        docs=options.docs if options.docs else None,
        tools=tool_defs,
        tool_choice=options.tool_choice,
        output=OutputConfig(
            format=output.format if output else None,
            # pyrefly: ignore[unexpected-keyword] - populate_by_name accepts the field name
            json_schema=out_schema,
            constrained=output.constrained if output else None,
            content_type=output.content_type if output else None,
        ),
    )
    input_class = model.input_class
    if input_class is not None and issubclass(input_class, ModelRequest) and input_class is not ModelRequest:
        try:
            # Fast path: construct the action's exact input class so validation
            # happens once, here; _validate_input then passes it through as-is.
            return input_class(**request_kwargs)
        except ValidationError:
            # Invalid input for the typed class. Fall through to the bare
            # carrier so Action._validate_input re-discovers the failure and
            # raises the proper GenkitError(INVALID_ARGUMENT) with the action
            # name — the pre-fast-path error contract, preserved exactly.
            pass
    return ModelRequest(**request_kwargs)


def to_tool_definition(tool: Action) -> ToolDefinition:
    """Convert an Action to a ToolDefinition for model requests."""
    metadata = tool.metadata or {}
    if ORIGINAL_OUTPUT_SCHEMA_KEY in metadata:
        original = metadata[ORIGINAL_OUTPUT_SCHEMA_KEY]
        output_schema = original if isinstance(original, dict) else None
    else:
        output_schema = tool.output_schema
    return ToolDefinition(
        name=tool.name,
        description=tool.description or '',
        input_schema=tool.input_schema,
        output_schema=output_schema,
    )


async def resolve_tool_requests(
    *,
    registry: Registry,
    message: Message,
    abort_signal: asyncio.Event,
    mw_pipeline: MiddlewarePipeline | None = None,
    tools: list[Action],
) -> tuple[Message | None, Message | None]:
    """Execute tool requests in a message, returning responses or interrupt info."""
    # The door Actions are what run. wrap_tool can wrap them; it cannot
    # swap the function behind a name they already passed.
    tool_dict = tool_map_from_actions(tools)

    revised_model_message = message.model_copy(deep=True)
    mw_list = mw_pipeline.middleware if mw_pipeline else []

    work: list[tuple[int, Action, Part]] = []
    for i, tool_request_part in enumerate(message.content):
        if not (isinstance(tool_request_part, Part) and tool_request_part.tool_request is not None):  # pyright: ignore[reportUnnecessaryIsInstance]
            continue

        tool_request = tool_request_part.tool_request

        if tool_request.name not in tool_dict:
            raise GenkitError(
                status='NOT_FOUND',
                message=f'Tool {tool_request.name} not found',
                reason=RuntimeErrorReason.TOOL_NOT_FOUND,
            )
        tool = tool_dict[tool_request.name]
        work.append((i, tool, tool_request_part))

    if not work:
        return (None, Message(role=Role.TOOL, content=[]))

    if is_debug_enabled(logger):
        logger.debug(
            'executing tool requests',
            tools=[trp.tool_request.name for _, _, trp in work if trp.tool_request is not None],
        )

    async def run_one_tool(tool: Action, trp: Part) -> tuple[MultipartToolResponse | None, Part | None]:
        if trp.tool_request is None:
            raise GenkitError(status='INTERNAL', message='Expected a tool request part')
        ctx = (
            mw_pipeline.ctx
            if mw_pipeline is not None
            else GenerateMiddlewareContext(
                ai=ScopedGenkitView(registry),
                abort_signal=abort_signal,
            )
        )
        raise_if_aborted(ctx.abort_signal)
        params = ToolHookParams(tool_request_part=trp, tool=tool)

        async def next_fn(p: ToolHookParams, c: GenerateMiddlewareContext) -> MultipartToolResponse:
            return await execute_tool_request(
                tool=p.tool,
                tool_request_part=p.tool_request_part,
                ctx=c,
            )

        try:
            if mw_list and mw_pipeline is not None:
                multipart = as_multipart_tool_response(
                    await dispatch_tool(
                        middleware=mw_list,
                        params=params,
                        ctx=mw_pipeline.ctx,
                        next_fn=next_fn,
                    ),
                    tool_name=trp.tool_request.name,
                )
            else:
                multipart = as_multipart_tool_response(await next_fn(params, ctx), tool_name=trp.tool_request.name)
            return (multipart, None)
        except Exception as e:
            # Interrupts (raised by the tool body or by middleware) become a
            # tool-request Part with interrupt metadata.  Any tracing span is the
            # middleware's responsibility (e.g. ToolApproval wraps its raise in
            # ``run_in_new_span`` explicitly).  Non-Interrupt exceptions are real
            # failures and propagate to ``asyncio.gather``.
            intr = interrupt_from_exc(e)
            if intr is None:
                raise
            logger.debug('tool triggered an interrupt', tool=trp.tool_request.name)
            return (None, interrupt_request_part(trp, intr))

    outs = await asyncio.gather(*[run_one_tool(tool, trp) for _, tool, trp in work])

    has_interrupts = False
    response_parts: list[Part] = []
    for (idx, _tool, tool_req_root), (multipart_resp, interrupt_part) in zip(work, outs, strict=True):
        if multipart_resp is not None:
            tool_req = tool_req_root.tool_request
            if tool_req is None:
                raise GenkitError(status='INTERNAL', message='Expected a tool request part')
            tool_response_part = Part(
                tool_response=ToolResponse(
                    name=tool_req.name,
                    ref=tool_req.ref,
                    output=multipart_resp.output,
                    content=parts_to_wire(multipart_resp.content, tool_name=tool_req.name),
                ),
                metadata=multipart_resp.metadata,
            )
            revised_model_message.content[idx] = to_pending_response(tool_req_root, tool_response_part)
            response_parts.append(tool_response_part)

        if interrupt_part:
            has_interrupts = True
            revised_model_message.content[idx] = interrupt_part

    if has_interrupts:
        return (revised_model_message, None)

    return (None, Message(role=Role.TOOL, content=response_parts))


def to_pending_response(request: Part, response: Part) -> Part:
    """Stash a completed sibling tool so resume can rebuild the same tool message.

    When another tool in the same turn interrupts, this tool already finished.
    The next model turn still needs that output — and any media — without
    running the tool again.
    """
    tool_response = response.tool_response
    if tool_response is None:
        raise GenkitError(status='INTERNAL', message='Expected a tool response part')
    metadata = dict(request.metadata) if request.metadata else {}
    metadata['pendingOutput'] = tool_response.output
    if tool_response.content:
        metadata['pendingContent'] = tool_response.content
    if response.metadata:
        metadata['pendingMetadata'] = response.metadata
    return Part(
        tool_request=request.tool_request,
        metadata=metadata,
    )


def interrupt_from_exc(exc: Exception) -> Interrupt | None:
    """If ``exc`` is (or wraps) an Interrupt exception, return that interrupt."""
    if isinstance(exc, Interrupt):
        return exc
    if isinstance(exc, GenkitError) and exc.cause is not None and isinstance(exc.cause, Interrupt):
        return exc.cause
    return None


async def execute_tool_request(
    *,
    tool: Action,
    tool_request_part: Part,
    ctx: GenerateMiddlewareContext,
) -> MultipartToolResponse:
    """Execute a tool and return its response.

    Interrupts from the tool body propagate to the caller (the engine
    stamps interrupt metadata on the tool-request Part at the top of
    ``run_one_tool``).  This keeps the contract symmetric with
    ``BaseMiddleware.wrap_tool``: responses are return values, interrupts
    are exceptions.
    """
    # run_tool_request threads custom_context/telemetry (and the abort signal) into
    # the tool. We still watch abort_signal here so a tool that ignores it gets hard
    # cancelled instead of hanging past a client abort.
    abort_signal = ctx.abort_signal
    tool_task = asyncio.create_task(run_tool_request(tool=tool, tool_request_part=tool_request_part, ctx=ctx))

    async def watch_abort() -> None:
        await abort_signal.wait()
        if not tool_task.done():
            tool_task.cancel()

    watcher_task = asyncio.create_task(watch_abort())
    try:
        tool_response = await tool_task
    except asyncio.CancelledError:
        # An outer cancel (deadline / gather teardown) is delivered to *us*, not to
        # the detached tool_task — cancel it so the tool body actually winds down
        # instead of running to completion after the caller is gone. (Idempotent on
        # the abort path, where the watcher already cancelled it.)
        tool_task.cancel()
        if abort_signal.is_set():
            raise GenkitError(status='ABORTED', message='Task aborted') from None
        raise
    finally:
        watcher_task.cancel()

    tool_req = tool_request_part.tool_request
    if tool_req is None:
        raise GenkitError(status='INTERNAL', message='Expected a tool request part')
    return as_multipart_tool_response(tool_response, tool_name=tool_req.name)


def interrupt_request_part(trp: Part, intr: Interrupt) -> Part:
    """Stamp interrupt metadata onto the tool-request Part the model already sent."""
    payload: dict[str, Any] | bool = intr.metadata if intr.metadata else True
    tool_meta = trp.metadata or {}
    return Part(
        tool_request=trp.tool_request,
        metadata={**tool_meta, 'interrupt': payload},
    )


async def resolve_tool(registry: Registry, tool_ref: str | Tool) -> Action:
    """Resolve a tool already on the registry.

    Catalog keys (``/tool.v2/name``) and bare registered names. DAP
    selectors (``mcp:tool/echo``) are bound in expand, not here.
    """
    if isinstance(tool_ref, Tool):
        return tool_ref.action()

    name = tool_ref
    if tool_ref.startswith('/'):
        try:
            kind, name = parse_action_key(tool_ref)
        except ValueError as e:
            raise GenkitError(
                status='NOT_FOUND',
                message=f'Unable to resolve tool {tool_ref}',
                reason=RuntimeErrorReason.TOOL_NOT_FOUND,
            ) from e
        if kind != ActionKind.TOOL:
            raise GenkitError(
                status='NOT_FOUND',
                message=f'Unable to resolve tool {tool_ref}',
                reason=RuntimeErrorReason.TOOL_NOT_FOUND,
            )
    elif parse_dap_qualified_name(tool_ref) is not None:
        raise GenkitError(
            status='NOT_FOUND',
            message=f'Unable to resolve tool {tool_ref}',
            reason=RuntimeErrorReason.TOOL_NOT_FOUND,
        )

    tool = await registry.resolve_action(kind=ActionKind.TOOL, name=name)
    if tool is None:
        raise GenkitError(
            status='NOT_FOUND',
            message=f'Unable to resolve tool {tool_ref}',
            reason=RuntimeErrorReason.TOOL_NOT_FOUND,
        )
    return tool


async def resolve_resume_options(
    *,
    options: GenerateActionOptions,
    mw_pipeline: MiddlewarePipeline | None = None,
    tools: list[Action],
) -> tuple[GenerateActionOptions, ModelResponse | None, Message | None]:
    """Handle resume options by resolving pending tool calls from a previous turn."""
    if not options.resume:
        return (options, None, None)

    messages = list(options.messages or [])
    last_message = messages[-1] if messages else None
    tool_requests = [p for p in last_message.content if p.tool_request is not None] if last_message else []
    if last_message is None or last_message.role != Role.MODEL or len(tool_requests) == 0:
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message=(
                "Cannot 'resume' generation unless the previous message is a model "
                'message with at least one tool request.'
            ),
            reason=RuntimeErrorReason.INVALID_RESUME,
        )

    # Build updated_content in a new list — do NOT mutate last_message.content
    # directly; the caller's options object must remain unchanged.
    updated_content = list(last_message.content)
    indexed_requests = [
        (index, part) for index, part in enumerate(last_message.content) if part.tool_request is not None
    ]
    resolved = await asyncio.gather(*[
        resolve_resumed_tool(
            options=options,
            tool_request_part=part,
            mw_pipeline=mw_pipeline,
            tools=tools,
        )
        for _, part in indexed_requests
    ])

    tool_responses = []
    has_interrupts = False
    for (index, _orig_part), (resumed_request, resumed_response) in zip(indexed_requests, resolved, strict=True):
        updated_content[index] = resumed_request
        if resumed_response is None:
            has_interrupts = True
            continue
        tool_responses.append(resumed_response)

    if has_interrupts:
        for (index, _orig), (resumed_request, resumed_response) in zip(indexed_requests, resolved, strict=True):
            if resumed_response is None:
                continue
            updated_content[index] = to_pending_response(resumed_request, resumed_response)
        interrupted = ModelResponse(
            finish_reason=FinishReason.INTERRUPTED,
            finish_message='One or more tool calls resulted in interrupts.',
            message=Message(
                role=last_message.role,
                content=updated_content,
                metadata=last_message.metadata,
            ),
            request=ModelRequest(messages=list(messages[:-1])),
        )
        return (options, interrupted, None)

    if len(tool_responses) != len(tool_requests):
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message=f'Expected {len(tool_requests)} responses, but resolved to {len(tool_responses)}',
            reason=RuntimeErrorReason.INVALID_RESUME,
        )

    tool_message = Message(
        role=Role.TOOL,
        content=tool_responses,
        metadata={'resumed': options.resume.metadata if options.resume.metadata else True},
    )

    revised_request = options.model_copy(deep=True)
    revised_request.resume = None
    # Replace the last message in the deep copy with the resolved version
    # (pending TRPs swapped for resolved ones) without touching options.
    revised_request.messages[-1] = Message(
        role=last_message.role,
        content=updated_content,
        metadata=last_message.metadata,
    )
    revised_request.messages.append(tool_message)

    return (revised_request, None, tool_message)


async def resolve_resumed_tool(
    *,
    options: GenerateActionOptions,
    tool_request_part: Part,
    mw_pipeline: MiddlewarePipeline | None = None,
    tools: list[Action],
) -> tuple[Part, Part | None]:
    """Resolve a single tool request from pending output, resume.respond, or resume.restart."""
    if tool_request_part.tool_request is None:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message='Expected a tool request part, got a different part type.',
            reason=RuntimeErrorReason.INVALID_PART,
        )

    tool_req_root = tool_request_part
    tool_req = tool_request_part.tool_request

    if tool_req_root.metadata and 'pendingOutput' in tool_req_root.metadata:
        # Strip the stash from the model TRP and rebuild the tool message so
        # resume looks like the tool already ran (output, media, metadata).
        trp_metadata = dict(tool_req_root.metadata)
        pending_output = trp_metadata.pop('pendingOutput')
        pending_content = trp_metadata.pop('pendingContent', None)
        pending_part_metadata = trp_metadata.pop('pendingMetadata', None)
        tool_name = tool_req.name
        pending_content = normalize_pending_content(pending_content, tool_name=tool_name)
        if pending_part_metadata is not None and not isinstance(pending_part_metadata, dict):
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=(
                    f'Tool {tool_name!r} pendingMetadata must be a dict, got {type(pending_part_metadata).__name__}.'
                ),
                reason=RuntimeErrorReason.INVALID_INPUT,
            )
        revised_trp = Part(
            tool_request=tool_req,
            metadata=trp_metadata if trp_metadata else None,
        )
        saved_meta = (
            dump_tool_metadata(pending_part_metadata, tool_name=tool_name)
            if isinstance(pending_part_metadata, dict)
            else None
        ) or {}
        response_metadata = {**saved_meta, 'source': 'pending'}
        return (
            revised_trp,
            Part(
                tool_response=ToolResponse(
                    name=tool_name,
                    ref=tool_req.ref,
                    output=dump_tool_output(pending_output, tool_name=tool_name),
                    content=pending_content,
                ),
                metadata=response_metadata,
            ),
        )

    # if there's a corresponding reply, append it to toolResponses
    provided_response = matching_tool_response(
        responses=(options.resume.respond if options.resume and options.resume.respond else []),
        tool_request=tool_req_root,
    )
    if provided_response:
        # remove the 'interrupt' but leave a 'resolvedInterrupt'
        metadata = dict(tool_req_root.metadata) if tool_req_root.metadata else {}
        interrupt = metadata.get('interrupt')
        if interrupt:
            del metadata['interrupt']
        return (
            Part(
                tool_request=ToolRequest(
                    name=tool_req.name,
                    ref=tool_req.ref,
                    input=tool_req.input,
                ),
                metadata={**metadata, 'resolvedInterrupt': interrupt},
            ),
            provided_response,
        )

    restart_trp = matching_restart(
        restarts=options.resume.restart if options.resume else None,
        tool_request=tool_req_root,
    )
    if restart_trp:
        tool = tool_map_from_actions(tools).get(tool_req.name)
        if tool is None:
            raise GenkitError(
                status='NOT_FOUND',
                message=f'Tool {tool_req.name} not found',
                reason=RuntimeErrorReason.TOOL_NOT_FOUND,
            )
        try:
            executed = await run_restart(
                tool=tool,
                restart_trp=restart_trp,
                mw_pipeline=mw_pipeline,
            )
        except Exception as e:
            intr = interrupt_from_exc(e)
            if intr is not None:
                return (interrupt_request_part(tool_req_root, intr), None)
            raise
        metadata = dict(tool_req_root.metadata) if tool_req_root.metadata else {}
        interrupt = metadata.get('interrupt')
        if interrupt:
            del metadata['interrupt']
        return (
            Part(
                tool_request=ToolRequest(
                    name=tool_req.name,
                    ref=tool_req.ref,
                    input=tool_req.input,
                ),
                metadata={**metadata, 'resolvedInterrupt': interrupt},
            ),
            executed,
        )

    raise GenkitError(
        status='INVALID_ARGUMENT',
        message=f"Unresolved tool request '{tool_req.name}' "
        + "was not handled by the 'resume' argument. You must supply replies or "
        + 'restarts for all interrupted tool requests.',
        reason=RuntimeErrorReason.UNRESOLVED_TOOL_REQUEST,
    )


async def run_restart(
    *,
    tool: Action,
    restart_trp: Part,
    mw_pipeline: MiddlewarePipeline | None,
) -> Part:
    """Run a restarted tool through the wrap_tool middleware chain.

    Restart paths reuse the same dispatch as fresh tool calls so middleware
    (ToolApproval, Filesystem error queueing, etc.) sees every tool execution
    regardless of whether it was triggered by the model or by a resumed
    interrupt.  Without this, a restart would silently bypass approval checks.
    """
    tool_req = restart_trp.tool_request
    if tool_req is None:
        raise GenkitError(status='INVALID_ARGUMENT', message='Expected a tool request part')
    mw_list = mw_pipeline.middleware if mw_pipeline else []
    if not mw_list or mw_pipeline is None:
        return await run_tool_after_restart(
            tool=tool,
            restart_trp=restart_trp,
            ctx=mw_pipeline.ctx if mw_pipeline is not None else None,
        )

    params = ToolHookParams(
        tool_request_part=restart_trp,
        tool=tool,
    )

    async def next_fn(p: ToolHookParams, ctx: GenerateMiddlewareContext) -> MultipartToolResponse:
        executed = await run_tool_after_restart(tool=p.tool, restart_trp=p.tool_request_part, ctx=ctx)
        if executed.tool_response is None:
            raise GenkitError(status='INTERNAL', message='Expected a tool response part')
        raw_content = executed.tool_response.content or []
        return MultipartToolResponse(
            output=executed.tool_response.output,
            content=[Part.model_validate(item) for item in raw_content] or None,
            metadata=executed.metadata,
        )

    try:
        multipart = as_multipart_tool_response(
            await dispatch_tool(
                middleware=mw_list,
                params=params,
                ctx=mw_pipeline.ctx,
                next_fn=next_fn,
            ),
            tool_name=tool_req.name,
        )
    except Exception as e:
        intr = interrupt_from_exc(e)
        if intr is not None:
            # run_tool_after_restart already logged when the tool body interrupted.
            # wrap_tool can raise Interrupt itself; that's the only interrupt path.
            if not isinstance(e, GenkitError):
                logger.debug(
                    'restarted tool triggered an interrupt',
                    tool=tool_req.name,
                )
            # wrap_tool paused again. generate turns this into the same
            # INTERRUPTED they already know how to answer.
            raise restart_interrupt_error(intr) from e
        raise

    return Part(
        tool_response=ToolResponse(
            name=tool_req.name,
            ref=tool_req.ref,
            output=multipart.output,
            content=parts_to_wire(multipart.content, tool_name=tool_req.name),
        ),
        metadata=multipart.metadata,
    )


def matching_restart(
    *,
    restarts: list[Part] | None,
    tool_request: Part,
) -> Part | None:
    """Find a restart part matching the pending request by name and ref."""
    if not restarts or tool_request.tool_request is None:
        return None
    for part in restarts:
        tr = part.tool_request
        if tr is not None and tr.name == tool_request.tool_request.name and tr.ref == tool_request.tool_request.ref:
            return part
    return None


def matching_tool_response(
    *,
    responses: list[Part],
    tool_request: Part,
) -> Part | None:
    """Find a response matching the request by name and ref."""
    if tool_request.tool_request is None:
        return None
    for part in responses:
        resp = part.tool_response
        if (
            resp is not None
            and resp.name == tool_request.tool_request.name
            and resp.ref == tool_request.tool_request.ref
        ):
            return part
    return None
