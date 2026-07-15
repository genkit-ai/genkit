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
import re
import secrets
from collections.abc import Awaitable, Callable, Generator, Sequence
from dataclasses import dataclass
from typing import Any, cast

from pydantic import BaseModel
from typing_extensions import Never

from genkit._ai._formats._types import FormatDef, Formatter
from genkit._ai._messages import inject_instructions
from genkit._ai._model import (
    Message,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
    text_from_content,
)
from genkit._ai._resource import ResourceArgument, ResourceInput, find_matching_resource, resolve_resources
from genkit._ai._tools import Interrupt, Tool, run_tool_after_restart, run_tool_request
from genkit._core._action import (
    GENKIT_DYNAMIC_ACTION_PROVIDER_ATTR,
    Action,
    ActionKind,
    ActionRunContext,
)
from genkit._core._error import GenkitError
from genkit._core._logger import get_logger
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
    Document,
    GenerateActionOptions,
)
from genkit._core._registry import Registry
from genkit._core._tracing import SpanMetadata, run_in_new_span
from genkit._core._typing import (
    FinishReason,
    MiddlewareRef,
    MultipartToolResponse,
    Part,
    Role,
    TextPart,
    ToolDefinition,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)

DEFAULT_MAX_TURNS = 5

logger = get_logger(__name__)


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
            )
        cfg = entry.config if isinstance(entry.config, dict) else None
        out.append(defn.instantiate(cfg))
    return out


@dataclass
class _GenerateMiddlewarePipeline:
    """Holds the middleware chain and the shared context for a single generate call."""

    middleware: list[MiddlewareDef]
    ctx: GenerateMiddlewareContext


def _prepare_middleware(
    middleware: list[BaseMiddleware],
    *,
    ctx: GenerateMiddlewareContext,
) -> _GenerateMiddlewarePipeline:
    """Return per-call middleware defs sharing one ``GenerateMiddlewareContext``."""
    return _GenerateMiddlewarePipeline(
        middleware=[_copy_middleware_instance(mw) for mw in middleware],
        ctx=ctx,
    )


async def dispatch_tool(
    middleware: list[MiddlewareDef],
    params: ToolHookParams,
    ctx: GenerateMiddlewareContext,
    next_fn: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]],
) -> MultipartToolResponse:
    """Chain wrap_tool middleware and call next_fn."""
    runner: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]] = next_fn
    for mw in reversed(middleware):
        _mw = mw
        _inner = runner

        async def run_next(
            p: ToolHookParams,
            c: GenerateMiddlewareContext,
            _m: MiddlewareDef = _mw,
            _i: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]] = _inner,
        ) -> MultipartToolResponse:
            return await _m.wrap_tool(p, c, _i)

        runner = run_next
    return await runner(params, ctx)


async def expand_wildcard_tools(registry: Registry, tool_names: list[str]) -> list[str]:
    """Expand DAP wildcard tool names into individual registry keys.

    A wildcard has the form ``<provider>:tool/*`` (or ``<provider>:tool/<prefix>*``).
    Each match becomes a full DAP key
    ``/dynamic-action-provider/<provider>:<actionType>/<toolName>`` so later resolution
    stays bound to that provider (no ambiguous bare-name lookup across DAPs).

    Non-wildcard names are passed through unchanged.
    """
    expanded: list[str] = []
    for name in tool_names:
        if not name.endswith('*') or ':' not in name:
            expanded.append(name)
            continue

        colon = name.index(':')
        provider_name = name[:colon]
        rest = name[colon + 1 :]  # e.g. "tool/*" or "tool/prefix*"

        provider_action = await registry.resolve_action(ActionKind.DYNAMIC_ACTION_PROVIDER, provider_name)
        if provider_action is None:
            expanded.append(name)
            continue

        dap = getattr(provider_action, GENKIT_DYNAMIC_ACTION_PROVIDER_ATTR, None)
        if dap is None:
            expanded.append(name)
            continue

        if '/' not in rest:
            expanded.append(name)
            continue

        action_type, action_pattern = rest.split('/', 1)
        metas = await dap.list_action_metadata(action_type, action_pattern)
        for meta in metas:
            tool_name = meta.get('name')
            if tool_name:
                tn = str(tool_name)
                expanded.append(f'/dynamic-action-provider/{provider_name}:{action_type}/{tn}')

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


_CONTEXT_PREFACE = '\n\nUse the following information to complete your task:\n\n'


def _last_user_message(messages: list[Message]) -> Message | None:
    """Find the last user message in a list."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == 'user':
            return messages[i]
    return None


def _context_item_template(d: Document, index: int) -> str:
    """Render a document as a citation line for context injection."""
    out = '- '
    ref = (d.metadata and (d.metadata.get('ref') or d.metadata.get('id'))) or index
    out += f'[{ref}]: '
    out += text_from_content(d.content) + '\n'
    return out


def _augment_with_context(
    request: ModelRequest,
    *,
    preface: str | None = _CONTEXT_PREFACE,
    item_template: Callable[[Document, int], str] | None = None,
    citation_key: str | None = None,
) -> ModelRequest:
    """Return a deepcopy of ``request`` with ``request.docs`` injected as a context part on the last user message.

    No-op (returns ``request`` unchanged) when there are no docs, no user message, or the last user message
    already has a non-pending ``purpose: 'context'`` part.
    """
    if not request.docs:
        return request

    user_message = _last_user_message(request.messages)
    if user_message is None:
        return request

    # Find any existing context part in the last user message
    context_idx = -1
    for i, part in enumerate(user_message.content):
        metadata = getattr(part.root, 'metadata', None) or {}
        if metadata.get('purpose') == 'context':
            context_idx = i
            break

    # If context already exists, only proceed if it is a pending placeholder
    if context_idx >= 0:
        meta = getattr(user_message.content[context_idx].root, 'metadata', None) or {}
        if not meta.get('pending'):
            return request

    # Render all documents as a single formatted text string
    template = item_template or _context_item_template
    rendered_docs = []
    for i, doc_data in enumerate(request.docs):
        doc = Document(content=doc_data.content, metadata=doc_data.metadata)
        if citation_key and doc.metadata:
            doc.metadata['ref'] = doc.metadata.get(citation_key, i)
        rendered_docs.append(template(doc, i))

    text_content = (preface or '') + ''.join(rendered_docs) + '\n'
    text_part = Part(root=TextPart(text=text_content, metadata={'purpose': 'context'}))

    # Safe-mutation via deep copy
    new_req = copy.deepcopy(request)
    new_user = _last_user_message(new_req.messages)
    assert new_user is not None

    if context_idx >= 0:
        new_user.content[context_idx] = text_part
    else:
        new_user.content.append(text_part)

    return new_req


# Matches data URIs: everything up to the first comma is the media-type +
# parameters (e.g. "data:audio/L16;codec=pcm;rate=24000;base64,").
_DATA_URI_RE = re.compile(r'data:[^,]{0,200},(?=.{100})', re.ASCII)


def _redact_data_uris(obj: Any) -> Any:  # noqa: ANN401
    """Recursively truncate long ``data:`` URIs in a serialized dict/list.

    Replaces values like ``data:image/png;base64,iVBORw0KGgo...`` with
    ``data:image/png;base64,...<12345 bytes>`` so debug logs stay readable
    when requests contain inline images or other binary media.
    """
    if isinstance(obj, str):
        m = _DATA_URI_RE.match(obj)
        if m:
            return f'{m.group()}...<{len(obj) - m.end()} bytes>'
        return obj
    if isinstance(obj, dict):
        return {k: _redact_data_uris(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_data_uris(v) for v in obj]
    return obj


def define_generate_action(registry: Registry) -> None:
    """Register the generation action triggered by the Dev UI."""

    async def generate_action_fn(
        input: GenerateActionOptions,
        ctx: ActionRunContext,
    ) -> ModelResponse:
        on_chunk = cast(Callable[[ModelResponseChunk], None], ctx.streaming_callback) if ctx.is_streaming else None
        return await generate_with_request(
            registry=registry,
            raw_request=input,
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
    raw_request: GenerateActionOptions,
    on_chunk: Callable[[ModelResponseChunk], None] | None = None,
    message_index: int = 0,
    current_turn: int = 0,
    context: dict[str, Any] | None = None,
) -> ModelResponse:
    """Open the user-facing ``generate`` span and delegate to the engine.

    Thin wrapper so in-process callers get a trace span named ``generate``
    around the whole call.  The registered ``/util/generate`` action skips
    this wrapper because the action runtime already opens its own span.
    """
    span_name = 'generate'
    with run_in_new_span(SpanMetadata(name=span_name, type='util', input=raw_request)) as span:
        result = await generate_with_request(
            registry=registry,
            raw_request=raw_request,
            on_chunk=on_chunk,
            message_index=message_index,
            current_turn=current_turn,
            context=context,
        )
        with contextlib.suppress(Exception):
            span.set_attribute('genkit:output', result.model_dump_json(by_alias=True, exclude_none=True))
        return result


async def generate_with_request(
    registry: Registry,
    raw_request: GenerateActionOptions,
    on_chunk: Callable[[ModelResponseChunk], None] | None = None,
    message_index: int = 0,
    current_turn: int = 0,
    context: dict[str, Any] | None = None,
) -> ModelResponse:
    """Resolve ``raw_request.use`` and run the generation.

    Core generate business logic. `ai.generate` veneer and the registered
    `/util/generate` action funnel through here.
    """
    # Shallow-copy the wire-shape struct so per-field updates below (and any
    # future mutations) don't leak back to the caller's ``raw_request``.
    raw_request = raw_request.model_copy()
    registry = registry if registry.is_child else registry.new_child()

    if raw_request.tools:
        raw_request.tools = await expand_wildcard_tools(registry, raw_request.tools)

    middleware = resolve_middleware_from_use(registry, raw_request.use)
    run_ctx = GenerateMiddlewareContext(
        registry=registry,
        custom_context=dict(context or {}),
        on_chunk=on_chunk,
    )

    mw_pipeline: _GenerateMiddlewarePipeline | None = None
    if middleware:
        mw_pipeline = _prepare_middleware(middleware, ctx=run_ctx)
        mw_tools: list[Action[Any, Any, Never]] = []
        for mw in mw_pipeline.middleware:
            contributed = mw.tools(mw_pipeline.ctx)
            mw_tools.extend(contributed)

        if mw_tools:
            mw_tool_names: list[str] = []
            for t in mw_tools:
                registry.register_action_from_instance(t)
                mw_tool_names.append(t.name)
            existing = list(raw_request.tools) if raw_request.tools else []
            for name in mw_tool_names:
                if name not in existing:
                    existing.append(name)
            raw_request = raw_request.model_copy()
            raw_request.tools = existing
    else:
        mw_pipeline = _GenerateMiddlewarePipeline(middleware=[], ctx=run_ctx)

    return await _generate_action_turn(
        registry=registry,
        raw_request=raw_request,
        mw_pipeline=mw_pipeline,
        message_index=message_index,
        current_turn=current_turn,
    )


class ChunkAccumulator:
    """Tracks role and message-index state across a streaming turn's chunks.

    The message index it lands on is what seeds the next turn, so the counter
    the streaming callback bumps is the same one the tool loop reads to keep
    saved history numbered consistently.
    """

    def __init__(self, message_index: int, formatter: Formatter[Any, Any] | None) -> None:
        self.message_index = message_index
        self.formatter = formatter
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


async def _generate_action_turn(
    registry: Registry,
    raw_request: GenerateActionOptions,
    mw_pipeline: _GenerateMiddlewarePipeline,
    message_index: int,
    current_turn: int,
) -> ModelResponse:
    """Run one model call plus tool resolution, then recurse for the next turn."""
    middleware = mw_pipeline.middleware
    run_ctx = mw_pipeline.ctx

    model, tools, format_def = await resolve_parameters(registry, raw_request)

    raw_request, formatter = apply_format(raw_request, format_def)

    if raw_request.resources:
        raw_request = await apply_resources(registry, raw_request)

    assert_valid_tool_names(tools)

    (
        revised_request,
        interrupted_response,
        resumed_tool_message,
    ) = await _resolve_resume_options(
        registry,
        raw_request,
        mw_pipeline=mw_pipeline,
    )

    # NOTE: in the future we should make it possible to interrupt a restart, but
    # at the moment it's too complicated because it's not clear how to return a
    # response that amends history but doesn't generate a new message, so we throw
    if interrupted_response:
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message='One or more tools triggered an interrupt during a restarted execution.',
            details={'message': interrupted_response.message},
        )
    raw_request = revised_request

    chunks = ChunkAccumulator(message_index, formatter)

    async def dispatch_generate(
        params: GenerateHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Chain wrap_generate middleware and call next_fn."""
        runner: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]] = next_fn
        for mw in reversed(middleware):
            _mw = mw
            _inner = runner

            async def run_next(
                p: GenerateHookParams,
                c: GenerateMiddlewareContext,
                _m: MiddlewareDef = _mw,
                _i: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]] = _inner,
            ) -> ModelResponse:
                return await _m.wrap_generate(p, c, _i)

            runner = run_next
        return await runner(params, ctx)

    async def dispatch_model(
        params: ModelHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Chain wrap_model middleware and call next_fn."""
        runner: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]] = next_fn
        for mw in reversed(middleware):
            _mw = mw
            _inner = runner

            async def run_next(
                params: ModelHookParams,
                c: GenerateMiddlewareContext,
                _mw: MiddlewareDef = _mw,
                _inner: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]] = _inner,
            ) -> ModelResponse:
                return await _mw.wrap_model(params, c, _inner)

            runner = cast(Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]], run_next)
        return await runner(params, ctx)

    # if resolving the 'resume' option above generated a tool message, stream it.
    if resumed_tool_message:
        chunks.stream_chunk(
            chunk=ModelResponseChunk(
                role=resumed_tool_message.role,
                content=resumed_tool_message.content,
            ),
            role=Role.TOOL,
            ctx=run_ctx,
        )

    async def run_one_iteration(
        params: GenerateHookParams,
        ctx: GenerateMiddlewareContext,
    ) -> ModelResponse:
        """Execute one turn of the generate loop (model call + optional tool resolution)."""
        chunks.message_index = params.message_index
        turn_options = params.options
        # Re-resolve and re-validate tools per turn to pick up dynamic tool
        # injections or removals from middleware (e.g. wrap_generate).
        turn_tools = await resolve_tools_from_options(registry, turn_options.tools)
        assert_valid_tool_names(turn_tools)
        request = await action_to_generate_request(turn_options, turn_tools, model)
        if request.docs:
            request = _augment_with_context(request)

        async def next_fn(params: ModelHookParams, c: GenerateMiddlewareContext) -> ModelResponse:
            return (
                await model.run(
                    input=params.request,
                    context=c.custom_context,
                    on_chunk=c.on_chunk,
                )
            ).response

        with chunks.intercept_model_stream(ctx, role=Role.MODEL):
            model_response = await dispatch_model(
                ModelHookParams(request=request),
                ctx,
                next_fn,
            )

        def message_parser(msg: Message) -> Any:  # noqa: ANN401
            if formatter is None:
                return None
            return formatter.parse_message(msg)

        # Extract schema_type for runtime Pydantic validation
        schema_type = turn_options.output.schema_type if turn_options.output else None

        # Plugin returns ModelResponse directly. Framework sets request and
        # any output format context (message_parser, schema_type) as private attrs.
        response = model_response
        response.request = request
        if formatter:
            response._message_parser = message_parser
        if schema_type:
            response._schema_type = schema_type

        logger.debug(
            'generate response',
            response=_redact_data_uris(response.model_dump()),
        )

        response.assert_valid()
        generated_msg = response.message

        if generated_msg is None:
            # No message in response, return as-is
            return response

        # Stamp output format metadata on message so the Dev UI can render formatted JSON vs plain text.
        out = turn_options.output
        if out and (out.content_type or out.format):
            generate_output: dict[str, str] = {}
            if out.content_type:
                generate_output['contentType'] = out.content_type
            if out.format:
                generate_output['format'] = out.format
            existing_meta = dict(generated_msg.metadata) if isinstance(generated_msg.metadata, dict) else {}
            generate_meta = existing_meta.get('generate')
            if not isinstance(generate_meta, dict):
                generate_meta = {}
            generate_meta['output'] = generate_output
            existing_meta['generate'] = generate_meta
            generated_msg.metadata = existing_meta

        tool_requests = [x for x in generated_msg.content if x.root.tool_request]

        if turn_options.return_tool_requests or len(tool_requests) == 0:
            if len(tool_requests) == 0:
                response.assert_valid_schema()
            return response

        max_iters = turn_options.max_turns if turn_options.max_turns is not None else DEFAULT_MAX_TURNS

        if current_turn + 1 > max_iters:
            raise GenerationResponseError(
                response=response,
                message=f'Exceeded maximum tool call iterations ({max_iters})',
                status='ABORTED',
                details={'request': request},
            )

        (
            revised_model_msg,
            tool_msg,
            transfer_preamble,
        ) = await resolve_tool_requests(
            registry,
            turn_options,
            generated_msg,
            mw_pipeline=mw_pipeline,
        )

        # if an interrupt message is returned, stop the tool loop and return a
        # response.
        if revised_model_msg:
            interrupted_resp = response.model_copy(deep=False)
            interrupted_resp.finish_reason = FinishReason.INTERRUPTED
            interrupted_resp.finish_message = 'One or more tool calls resulted in interrupts.'
            interrupted_resp.message = Message(revised_model_msg)
            return interrupted_resp

        # If the loop will continue, stream out the tool response message...
        if tool_msg:
            chunks.stream_chunk(
                chunk=ModelResponseChunk(
                    role=tool_msg.role,
                    content=tool_msg.content,
                ),
                role=Role.TOOL,
                ctx=run_ctx,
            )

        next_request = copy.copy(turn_options)
        next_messages = copy.copy(turn_options.messages)
        next_messages.append(generated_msg)
        if tool_msg:
            next_messages.append(tool_msg)
        next_request.messages = next_messages
        if transfer_preamble:
            next_request = apply_transfer_preamble(next_request, transfer_preamble)

        return await _generate_action_turn(
            registry=registry,
            raw_request=next_request,
            mw_pipeline=mw_pipeline,
            current_turn=current_turn + 1,
            message_index=chunks.message_index + 1,
        )

    generate_params = GenerateHookParams(
        options=raw_request,
        iteration=current_turn,
        message_index=chunks.message_index,
    )
    return await dispatch_generate(generate_params, run_ctx, run_one_iteration)


def apply_format(
    raw_request: GenerateActionOptions, format_def: FormatDef | None
) -> tuple[GenerateActionOptions, Formatter[Any, Any] | None]:
    """Apply format definition to request, injecting instructions and output config."""
    if not format_def:
        return raw_request, None

    out_request = copy.deepcopy(raw_request)

    formatter = format_def(raw_request.output.json_schema if raw_request.output else None)

    # Extract instructions - handle bool | str | None type
    # Schema allows: str (custom instructions), True (use defaults), False (disable), None (default behavior)
    raw_instructions = raw_request.output.instructions if raw_request.output else None
    str_instructions = raw_instructions if isinstance(raw_instructions, str) else None
    instructions = resolve_instructions(formatter, str_instructions)

    should_inject = False
    if raw_request.output and raw_request.output.instructions is not None:
        should_inject = bool(raw_request.output.instructions)
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
    if raw_request.output and raw_request.output.constrained is not None:
        out_request.output.constrained = raw_request.output.constrained

    if format_def.config.content_type is not None:
        out_request.output.content_type = format_def.config.content_type
    if format_def.config.format is not None:
        out_request.output.format = format_def.config.format

    return (out_request, formatter)


def resolve_instructions(formatter: Formatter[Any, Any], instructions_opt: str | None) -> str | None:
    """Return custom instructions if provided, otherwise use formatter defaults."""
    if instructions_opt is not None:
        # user provided instructions
        return instructions_opt
    if not formatter:
        return None  # pyright: ignore[reportUnreachable] - defensive check
    return formatter.instructions


def apply_transfer_preamble(
    next_request: GenerateActionOptions, _preamble: GenerateActionOptions
) -> GenerateActionOptions:
    """Transfer preamble settings to the next request. (TODO: not yet implemented)."""
    # TODO(#4338): implement me
    return next_request


def _extract_resource_uri(resource_obj: Any) -> str | None:  # noqa: ANN401
    """Extract URI from a resource object, unwrapping Pydantic structures as needed."""
    # Direct uri attribute (Resource1, ResourceInput, etc.)
    if hasattr(resource_obj, 'uri'):
        return resource_obj.uri

    # Unwrap RootModel structures
    if hasattr(resource_obj, 'root'):
        return _extract_resource_uri(resource_obj.root)

    # Unwrap nested resource attribute
    if hasattr(resource_obj, 'resource'):
        return _extract_resource_uri(resource_obj.resource)

    # Handle dict representation
    if isinstance(resource_obj, dict) and 'uri' in resource_obj:
        return resource_obj['uri']

    return None


async def apply_resources(registry: Registry, raw_request: GenerateActionOptions) -> GenerateActionOptions:
    """Resolve and hydrate resource parts in the request messages."""
    # Quick check if any message has a resource part
    has_resource = False
    for msg in raw_request.messages:
        for part in msg.content:
            if part.root.resource:
                has_resource = True
                break
        if has_resource:
            break

    if not has_resource:
        return raw_request

    # Resolve all declared resources
    resources = []
    if raw_request.resources:
        resources = await resolve_resources(registry, cast(list[ResourceArgument], raw_request.resources))

    updated_messages = []
    for msg in raw_request.messages:
        if not any(p.root.resource for p in msg.content):
            updated_messages.append(msg)
            continue

        updated_content = []
        for part in msg.content:
            if not part.root.resource:
                updated_content.append(part)
                continue

            resource_obj = part.root.resource

            # Extract URI from the resource object
            # The resource can be wrapped in various Pydantic structures (Resource, Resource1, etc.)
            ref_uri = _extract_resource_uri(resource_obj)
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
                )

            # Normalize to ResourceInput for matching
            resource_input = ResourceInput(uri=ref_uri)
            resource_action = await find_matching_resource(registry, resources, resource_input)

            if not resource_action:
                raise GenkitError(
                    status='NOT_FOUND',
                    message=f'failed to find matching resource for {ref_uri}',
                )

            # Execute the resource
            response = await resource_action.run(resource_input, on_chunk=None, context=None)

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
    new_request = raw_request.model_copy()
    new_request.messages = updated_messages
    return new_request


def _tool_short_name_for_model(name: str) -> str:
    """Return the last path segment of a tool name."""
    if '/' not in name:
        return name
    return name[name.rfind('/') + 1 :]


def assert_valid_tool_names(tools: list[Action[Any, Any, Any]]) -> None:
    """Reject overlapping model-facing tool names before the model is called.

    Two resolved tools that share the same short name (segment after the last ``/``)
    cannot both appear in one generate request.
    """
    if not tools:
        return
    seen: dict[str, str] = {}
    for tool in tools:
        short = _tool_short_name_for_model(tool.name)
        if short in seen:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=(f"Cannot provide two tools with the same name: '{tool.name}' and '{seen[short]}'"),
            )
        seen[short] = tool.name


async def resolve_tools_from_options(
    registry: Registry,
    tool_names: list[str] | None,
) -> list[Action[Any, Any, Any]]:
    """Expand wildcards and resolve tool actions for a list of tool names."""
    if not tool_names:
        return []
    expanded = await expand_wildcard_tools(registry, tool_names)
    actions: list[Action[Any, Any, Any]] = []
    for t_name in expanded:
        actions.append(await resolve_tool(registry, t_name))
    return actions


async def resolve_parameters(
    registry: Registry, request: GenerateActionOptions
) -> tuple[Action[Any, Any, Any], list[Action[Any, Any, Any]], FormatDef | None]:
    """Resolve model, tools, and format from registry for a generation request."""
    model = (
        request.model
        if request.model is not None
        else cast(str | None, registry.lookup_value('defaultModel', 'defaultModel'))
    )
    if not model:
        raise Exception('No model configured.')

    model_action = await registry.resolve_model(model)
    if model_action is None:
        raise Exception(f'Failed to to resolve model {model}')

    # Resolve tools up front to fail fast on invalid caller-supplied tool names or
    # duplicate short names before running side effects or middleware.
    tools = await resolve_tools_from_options(registry, request.tools)

    format_def: FormatDef | None = None
    if request.output and request.output.format:
        looked_up_format = registry.lookup_value('format', request.output.format)
        if looked_up_format is None:
            raise ValueError(f'Unable to resolve format {request.output.format}')
        format_def = cast(FormatDef, looked_up_format)

    return (model_action, tools, format_def)


async def action_to_generate_request(
    options: GenerateActionOptions, resolved_tools: list[Action[Any, Any, Any]], _model: Action[Any, Any, Any]
) -> ModelRequest[Any]:
    """Convert GenerateActionOptions to a ModelRequest with tool definitions."""
    # TODO(#4340): add warning when tools are not supported in ModelInfo
    # TODO(#4341): add warning when toolChoice is not supported in ModelInfo

    tool_defs = [to_tool_definition(tool) for tool in resolved_tools] if resolved_tools else []
    output = options.output
    out_schema = output.json_schema if output else None
    if out_schema is not None and hasattr(out_schema, 'model_dump'):
        out_schema = out_schema.model_dump()
    return ModelRequest(
        # Field validators auto-wrap MessageData -> Message and DocumentData -> Document
        messages=options.messages,  # type: ignore[arg-type]
        config=options.config if options.config is not None else {},  # type: ignore[arg-type]
        docs=options.docs if options.docs else None,  # type: ignore[arg-type]
        tools=tool_defs,
        tool_choice=options.tool_choice,
        output_format=output.format if output else None,
        output_schema=out_schema,
        output_constrained=output.constrained if output else None,
        output_content_type=output.content_type if output else None,
    )


def to_tool_definition(tool: Action) -> ToolDefinition:
    """Convert an Action to a ToolDefinition for model requests."""
    tdef = ToolDefinition(
        name=tool.name,
        description=tool.description or '',
        input_schema=tool.input_schema,
        output_schema=tool.output_schema,
    )
    return tdef


async def resolve_tool_requests(
    registry: Registry,
    request: GenerateActionOptions,
    message: Message,
    *,
    mw_pipeline: _GenerateMiddlewarePipeline | None = None,
) -> tuple[Message | None, Message | None, GenerateActionOptions | None]:
    """Execute tool requests in a message, returning responses or interrupt info."""
    # TODO(#4342): prompt transfer
    tool_dict: dict[str, Action] = {}
    if request.tools:
        for tool_name in request.tools:
            tool_action = await resolve_tool(registry, tool_name)
            tool_dict[tool_name] = tool_action
            # Model tool calls use ToolDefinition.name (short); wildcard expansion uses full DAP keys.
            short = tool_action.name
            if short not in tool_dict:
                tool_dict[short] = tool_action

    revised_model_message = message.model_copy(deep=True)
    mw_list = mw_pipeline.middleware if mw_pipeline else []

    work: list[tuple[int, Action, ToolRequestPart]] = []
    for i, tool_request_part in enumerate(message.content):
        if not (isinstance(tool_request_part, Part) and isinstance(tool_request_part.root, ToolRequestPart)):  # pyright: ignore[reportUnnecessaryIsInstance]
            continue

        tool_req_root = tool_request_part.root
        tool_request = tool_req_root.tool_request

        if tool_request.name not in tool_dict:
            raise RuntimeError(f'failed {tool_request.name} not found')
        tool = tool_dict[tool_request.name]
        work.append((i, tool, tool_req_root))

    if not work:
        return (None, Message(role=Role.TOOL, content=[]), None)

    async def _resolve_one_tool(
        tool: Action, trp: ToolRequestPart
    ) -> tuple[MultipartToolResponse | None, ToolRequestPart | None]:
        params = ToolHookParams(tool_request_part=trp, tool=tool)

        async def next_fn(p: ToolHookParams, c: GenerateMiddlewareContext) -> MultipartToolResponse:
            return await _resolve_tool_request(
                tool=p.tool,
                tool_request_part=p.tool_request_part,
                ctx=c,
            )

        try:
            if mw_list and mw_pipeline is not None:
                multipart = await dispatch_tool(mw_list, params, mw_pipeline.ctx, next_fn)
            else:
                multipart = await next_fn(
                    params, mw_pipeline.ctx if mw_pipeline else GenerateMiddlewareContext(registry=registry)
                )
            return (multipart, None)
        except Exception as e:
            # Interrupts (raised by the tool body or by middleware) become a
            # wire-shape interrupt ``ToolRequestPart``.  Any tracing span is the
            # middleware's responsibility (e.g. ToolApproval wraps its raise in
            # ``run_in_new_span`` explicitly).  Non-Interrupt exceptions are real
            # failures and propagate to ``asyncio.gather``.
            intr = _interrupt_from_tool_exc(e)
            if intr is None:
                raise
            return (None, _interrupt_request_part(trp, intr))

    outs = await asyncio.gather(*[_resolve_one_tool(tool, trp) for _, tool, trp in work])

    has_interrupts = False
    response_parts: list[Part] = []
    for (idx, _tool, tool_req_root), (multipart_resp, interrupt_part) in zip(work, outs, strict=True):
        if multipart_resp is not None:
            tool_response_part = ToolResponsePart(
                tool_response=ToolResponse(
                    name=tool_req_root.tool_request.name,
                    ref=tool_req_root.tool_request.ref,
                    output=multipart_resp.output,
                    content=[p.model_dump() for p in multipart_resp.content] if multipart_resp.content else None,
                ),
                metadata=multipart_resp.metadata,
            )
            revised_model_message.content[idx] = _to_pending_response(tool_req_root, tool_response_part)
            response_parts.append(Part(root=tool_response_part))

        if interrupt_part:
            has_interrupts = True
            revised_model_message.content[idx] = Part(root=interrupt_part)

    if has_interrupts:
        return (revised_model_message, None, None)

    return (None, Message(role=Role.TOOL, content=response_parts), None)


def _to_pending_response(request: ToolRequestPart, response: ToolResponsePart) -> Part:
    """Mark a tool request as pending with its response stored in metadata."""
    metadata = dict(request.metadata) if request.metadata else {}
    metadata['pendingOutput'] = response.tool_response.output
    # Part is a RootModel, so we pass content via 'root' parameter
    return Part(
        root=ToolRequestPart(
            tool_request=request.tool_request,
            metadata=metadata,
        )
    )


def _interrupt_from_tool_exc(exc: BaseException) -> Interrupt | None:
    """If ``exc`` is (or wraps) an Interrupt exception, return that interrupt."""
    if isinstance(exc, Interrupt):
        return exc
    if isinstance(exc, GenkitError) and exc.cause is not None and isinstance(exc.cause, Interrupt):
        return exc.cause
    return None


async def _resolve_tool_request(
    *,
    tool: Action,
    tool_request_part: ToolRequestPart,
    ctx: GenerateMiddlewareContext,
) -> MultipartToolResponse:
    """Execute a tool and return its response.

    Interrupts from the tool body propagate to the caller (the engine
    converts them to a wire ``ToolRequestPart`` at the top of
    ``_resolve_one_tool``).  This keeps the contract symmetric with
    ``BaseMiddleware.wrap_tool``: responses are return values, interrupts
    are exceptions.
    """
    tool_response = await run_tool_request(tool=tool, tool_request_part=tool_request_part, ctx=ctx)
    return MultipartToolResponse(
        output=tool_response.model_dump() if isinstance(tool_response, BaseModel) else tool_response,
    )


def _interrupt_request_part(trp: ToolRequestPart, intr: Interrupt) -> ToolRequestPart:
    """Convert an Interrupt exception into the wire-shape interrupt ToolRequestPart."""
    payload: dict[str, Any] | bool = intr.metadata if intr.metadata else True
    tool_meta = trp.metadata or {}
    return ToolRequestPart(
        tool_request=trp.tool_request,
        metadata={**tool_meta, 'interrupt': payload},
    )


async def resolve_tool(registry: Registry, tool_ref: str | Tool) -> Action:
    """Resolve a tool from a registry name or a Tool instance.

    Accepts full action keys (``/dynamic-action-provider/...``), DAP-qualified
    names (``provider:tool/name``), or plain registered tool names.

    Used when building ModelRequest (for example from to_generate_request).
    """
    if isinstance(tool_ref, Tool):
        return tool_ref.action()

    if tool_ref.startswith('/'):
        tool = await registry.resolve_action_by_key(tool_ref)
        if tool is not None:
            return tool

    tool = await registry.resolve_action(kind=ActionKind.TOOL, name=tool_ref)
    if tool is None:
        raise GenkitError(status='NOT_FOUND', message=f'Unable to resolve tool {tool_ref}')
    return tool


async def _resolve_resume_options(
    _registry: Registry,
    raw_request: GenerateActionOptions,
    *,
    mw_pipeline: _GenerateMiddlewarePipeline | None = None,
) -> tuple[GenerateActionOptions, ModelResponse | None, Message | None]:
    """Handle resume options by resolving pending tool calls from a previous turn."""
    if not raw_request.resume:
        return (raw_request, None, None)

    messages = raw_request.messages
    last_message = messages[-1]
    tool_requests = [p for p in last_message.content if p.root.tool_request]
    if not last_message or last_message.role != Role.MODEL or len(tool_requests) == 0:
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message=(
                "Cannot 'resume' generation unless the previous message is a model "
                'message with at least one tool request.'
            ),
        )

    i = 0
    tool_responses = []
    # Build updated_content in a new list — do NOT mutate last_message.content
    # directly; the caller's raw_request object must remain unchanged.
    updated_content = list(last_message.content)
    for part in last_message.content:
        if not isinstance(part.root, ToolRequestPart):
            i += 1
            continue

        resumed_request, resumed_response = await _resolve_resumed_tool_request(
            _registry,
            raw_request,
            part,
            mw_pipeline=mw_pipeline,
        )
        tool_responses.append(Part(root=resumed_response))
        updated_content[i] = Part(root=resumed_request)
        i += 1

    if len(tool_responses) != len(tool_requests):
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message=f'Expected {len(tool_requests)} responses, but resolved to {len(tool_responses)}',
        )

    tool_message = Message(
        role=Role.TOOL,
        content=tool_responses,
        metadata={'resumed': (raw_request.resume.metadata if raw_request.resume.metadata else True)},
    )

    revised_request = raw_request.model_copy(deep=True)
    revised_request.resume = None
    # Replace the last message in the deep copy with the resolved version
    # (pending TRPs swapped for resolved ones) without touching raw_request.
    revised_request.messages[-1] = Message(
        role=last_message.role,
        content=updated_content,
        metadata=last_message.metadata,
    )
    revised_request.messages.append(tool_message)

    return (revised_request, None, tool_message)


async def _resolve_resumed_tool_request(
    registry: Registry,
    raw_request: GenerateActionOptions,
    tool_request_part: Part,
    *,
    mw_pipeline: _GenerateMiddlewarePipeline | None = None,
) -> tuple[ToolRequestPart, ToolResponsePart]:
    """Resolve a single tool request from pending output, resume.respond, or resume.restart."""
    # Type narrowing: ensure we're working with a ToolRequestPart
    if not isinstance(tool_request_part.root, ToolRequestPart):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message='Expected a ToolRequestPart, got a different part type.',
        )

    tool_req_root = tool_request_part.root

    if tool_req_root.metadata and 'pendingOutput' in tool_req_root.metadata:
        # resolveResumedToolRequest: strip pendingOutput from the model TRP; reconstruct
        # output on the tool message with metadata { ...rest, source: 'pending' }.
        trp_metadata = dict(tool_req_root.metadata)
        pending_output = trp_metadata.pop('pendingOutput')
        revised_trp = ToolRequestPart(
            tool_request=tool_req_root.tool_request,
            metadata=trp_metadata if trp_metadata else None,
        )
        response_metadata = {**trp_metadata, 'source': 'pending'}
        return (
            revised_trp,
            ToolResponsePart(
                tool_response=ToolResponse(
                    name=tool_req_root.tool_request.name,
                    ref=tool_req_root.tool_request.ref,
                    output=pending_output.model_dump() if isinstance(pending_output, BaseModel) else pending_output,
                ),
                metadata=response_metadata,
            ),
        )

    # if there's a corresponding reply, append it to toolResponses
    provided_response = _find_corresponding_tool_response(
        (raw_request.resume.respond if raw_request.resume and raw_request.resume.respond else []),
        tool_req_root,
    )
    if provided_response:
        # remove the 'interrupt' but leave a 'resolvedInterrupt'
        metadata = dict(tool_req_root.metadata) if tool_req_root.metadata else {}
        interrupt = metadata.get('interrupt')
        if interrupt:
            del metadata['interrupt']
        return (
            ToolRequestPart(
                tool_request=ToolRequest(
                    name=tool_req_root.tool_request.name,
                    ref=tool_req_root.tool_request.ref,
                    input=tool_req_root.tool_request.input,
                ),
                metadata={**metadata, 'resolvedInterrupt': interrupt},
            ),
            provided_response,
        )

    restart_trp = _find_corresponding_restart(
        raw_request.resume.restart if raw_request.resume else None,
        tool_req_root,
    )
    if restart_trp:
        tool = await resolve_tool(registry, tool_req_root.tool_request.name)
        executed = await _run_restart_through_middleware(tool, restart_trp, mw_pipeline=mw_pipeline)
        metadata = dict(tool_req_root.metadata) if tool_req_root.metadata else {}
        interrupt = metadata.get('interrupt')
        if interrupt:
            del metadata['interrupt']
        return (
            ToolRequestPart(
                tool_request=ToolRequest(
                    name=tool_req_root.tool_request.name,
                    ref=tool_req_root.tool_request.ref,
                    input=tool_req_root.tool_request.input,
                ),
                metadata={**metadata, 'resolvedInterrupt': interrupt},
            ),
            executed,
        )

    raise GenkitError(
        status='INVALID_ARGUMENT',
        message=f"Unresolved tool request '{tool_req_root.tool_request.name}' "
        + "was not handled by the 'resume' argument. You must supply replies or "
        + 'restarts for all interrupted tool requests.',
    )


async def _run_restart_through_middleware(
    tool: Action,
    restart_trp: ToolRequestPart,
    *,
    mw_pipeline: _GenerateMiddlewarePipeline | None,
) -> ToolResponsePart:
    """Run a restarted tool through the wrap_tool middleware chain.

    Restart paths reuse the same dispatch as fresh tool calls so middleware
    (ToolApproval, Filesystem error queueing, etc.) sees every tool execution
    regardless of whether it was triggered by the model or by a resumed
    interrupt.  Without this, a restart would silently bypass approval checks.
    """
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

    async def next_fn(p: ToolHookParams, c: GenerateMiddlewareContext) -> MultipartToolResponse:
        executed = await run_tool_after_restart(tool=p.tool, restart_trp=p.tool_request_part, ctx=c)
        return MultipartToolResponse(
            output=executed.tool_response.output,
            content=[Part.model_validate(c) for c in (executed.tool_response.content or [])],
        )

    try:
        multipart = await dispatch_tool(mw_list, params, mw_pipeline.ctx, next_fn)
    except Exception as e:
        if _interrupt_from_tool_exc(e) is not None:
            # Re-interrupting during restart is a hard error — same as the legacy
            # run_tool_after_restart path, which raises FAILED_PRECONDITION when
            # the inner tool throws an Interrupt during restart.
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message='Tool interrupted again during a restart execution; not supported yet.',
            ) from e
        raise

    return ToolResponsePart(
        tool_response=ToolResponse(
            name=restart_trp.tool_request.name,
            ref=restart_trp.tool_request.ref,
            output=multipart.output,
            content=[p.model_dump() for p in multipart.content] if multipart.content else None,
        ),
        metadata=multipart.metadata,
    )


def _find_corresponding_restart(
    restarts: list[ToolRequestPart] | None,
    request: ToolRequestPart,
) -> ToolRequestPart | None:
    """Find a restart part matching the pending request by name and ref."""
    if not restarts:
        return None
    for trp in restarts:
        if trp.tool_request.name == request.tool_request.name and trp.tool_request.ref == request.tool_request.ref:
            return trp
    return None


def _find_corresponding_tool_response(
    responses: list[ToolResponsePart], request: ToolRequestPart
) -> ToolResponsePart | None:
    """Find a response matching the request by name and ref."""
    for p in responses:
        if p.tool_response.name == request.tool_request.name and p.tool_response.ref == request.tool_request.ref:
            return p
    return None


# TODO(#4336): extend GenkitError
class GenerationResponseError(Exception):
    # TODO(#4337): use status enum
    """Error raised when a generation request fails."""

    def __init__(
        self,
        response: ModelResponse,
        message: str,
        status: str,
        details: dict[str, Any],
    ) -> None:
        """Initialize with the failed response and error details."""
        super().__init__(message)
        self.response: ModelResponse = response
        self.message: str = message
        self.status: str = status
        self.details: dict[str, Any] = details
