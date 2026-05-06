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
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Never, cast

from pydantic import BaseModel

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
from genkit._ai._tools import Interrupt, Tool, run_tool_after_restart
from genkit._core._action import (
    GENKIT_DYNAMIC_ACTION_PROVIDER_ATTR,
    Action,
    ActionKind,
    ActionRunContext,
)
from genkit._core._error import GenkitError
from genkit._core._logger import get_logger
from genkit._core._middleware._base import BaseMiddleware, MiddlewareDesc
from genkit._core._model import (
    Document,
    GenerateActionOptions,
    GenerateHookParams,
    ModelHookParams,
    MultipartToolResponse,
    ToolHookParams,
)
from genkit._core._registry import Registry
from genkit._core._tracing import run_in_new_span
from genkit._core._typing import (
    FinishReason,
    MiddlewareRef,
    Part,
    Role,
    SpanMetadata,
    TextPart,
    ToolDefinition,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)

DEFAULT_MAX_TURNS = 5

logger = get_logger(__name__)


def split_use_entries(
    use: Sequence[BaseMiddleware | MiddlewareRef] | None,
) -> list[MiddlewareRef] | None:
    """Pick out just the ``MiddlewareRef`` entries from a veneer ``use=`` list.

    Inline ``BaseMiddleware`` instances are not serializable, so they cannot flow
    through ``GenerateActionOptions.use`` (which is the JSON wire form for the
    reflection API and JSON action dispatch). The veneer resolves the full list
    locally via :func:`resolve_middleware_from_use` and passes the resolved list as
    ``resolved_middleware=`` to :func:`generate_action`; this helper produces the
    parallel ref-only list for ``GenerateActionOptions.use`` so JSON inputs of the
    action still round-trip.
    """
    if not use:
        return None
    refs = [entry for entry in use if isinstance(entry, MiddlewareRef)]
    return refs or None


def resolve_middleware_from_use(
    registry: Registry,
    use: Sequence[BaseMiddleware | MiddlewareRef] | None,
) -> list[BaseMiddleware] | None:
    """Resolve a ``use=[...]`` list to concrete ``BaseMiddleware`` instances.

    Each entry is one of:
    - ``BaseMiddleware`` instance: used directly (inline fast path; no registry lookup).
    - ``MiddlewareRef``: looked up by ``name`` and instantiated via the registered
      ``MiddlewareDesc`` factory, which receives ``config`` directly.

    List order is preserved so inline instances and refs can be interleaved.
    """
    if not use:
        return None
    out: list[BaseMiddleware] = []
    for entry in use:
        if isinstance(entry, BaseMiddleware):
            # Clone before injecting the registry so we do not mutate the
            # caller's instance (which they may share across concurrent
            # generate() calls or across Genkit apps).  ``model_copy()`` is
            # shallow — sufficient since we only set the ``_registry``
            # PrivateAttr on the copy.  The registry passed in is the per-call
            # child registry created by ``generate_action``, so any writes the
            # middleware does through ``self._registry`` die with the call.
            inst = entry.model_copy()
            inst._registry = registry
            out.append(inst)
            continue
        # MiddlewareRef -> look up the registered definition, pass registry.
        defn = registry.lookup_value('middleware', entry.name)
        if isinstance(defn, MiddlewareDesc):
            cfg = entry.config if isinstance(entry.config, dict) else None
            out.append(defn(cfg, registry))
            continue
        raise GenkitError(
            status='NOT_FOUND',
            message=(
                f'No middleware named "{entry.name}" is registered on this app. '
                'Register descriptors with middleware_plugin([...]), Plugin.list_middleware(), '
                'or ai.define_middleware(MyMiddleware); or pass the middleware instance directly '
                'in use=[MyMiddleware(...)].'
            ),
            source='genkit.generate',
        )
    return out


async def _chain_tool_middleware(
    middleware: list[BaseMiddleware],
    params: ToolHookParams,
    next_fn: Callable[
        [ToolHookParams],
        Awaitable[tuple[MultipartToolResponse | None, ToolRequestPart | None]],
    ],
) -> tuple[MultipartToolResponse | None, ToolRequestPart | None]:
    """Run the tool middleware chain and return (multipart_response, interrupt_part)."""
    runner: Callable[
        [ToolHookParams],
        Awaitable[tuple[MultipartToolResponse | None, ToolRequestPart | None]],
    ] = next_fn
    for mw in reversed(middleware):
        _mw = mw
        _inner = runner

        async def run_next(
            p: ToolHookParams,
            *,
            _m: BaseMiddleware = _mw,
            _i: Callable[
                [ToolHookParams],
                Awaitable[tuple[MultipartToolResponse | None, ToolRequestPart | None]],
            ] = _inner,
        ) -> tuple[MultipartToolResponse | None, ToolRequestPart | None]:
            return await _m.wrap_tool(p, _i)

        runner = run_next
    return await runner(params)


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


async def registry_with_inline_tools(registry: Registry, tools: Sequence[str | Tool] | None) -> Registry:
    """Scope unregistered :class:`~genkit._ai._tools.Tool` instances for a single generate call.

    For each :class:`~genkit._ai._tools.Tool` value, if :meth:`Registry.resolve_action` for
    ``ActionKind.TOOL`` and ``tool.name`` already yields ``tool.action()`` (same instance), it is left
    alone. Otherwise the tool is registered on a short-lived child registry so this call can
    resolve it without mutating the root—same idea as Go ``GenerateWithRequest`` (``NewChild`` +
    register middleware/dynamic tools) so execution still uses one resolve path.

    Plain string tool names are unchanged.
    """
    if not tools:
        return registry

    child: Registry | None = None
    for t in tools:
        if not isinstance(t, Tool):
            continue
        resolved = await registry.resolve_action(ActionKind.TOOL, t.name)
        if resolved is t.action():
            continue
        if child is None:
            child = registry.new_child()
        child.register_action_from_instance(t.action())

    return child if child is not None else registry


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

    context_part_index = -1
    for i, part in enumerate(user_message.content):
        part_metadata = part.root.metadata if hasattr(part.root, 'metadata') else None
        if isinstance(part_metadata, dict) and part_metadata.get('purpose') == 'context':
            context_part_index = i
            break

    if context_part_index >= 0:
        existing_meta = user_message.content[context_part_index].root.metadata
        if not (isinstance(existing_meta, dict) and existing_meta.get('pending')):
            return request

    template = item_template or _context_item_template
    out = preface or ''
    for i, doc_data in enumerate(request.docs):
        doc = Document(content=doc_data.content, metadata=doc_data.metadata)
        if citation_key and doc.metadata:
            doc.metadata['ref'] = doc.metadata.get(citation_key, i)
        out += template(doc, i)
    out += '\n'

    text_part = Part(root=TextPart(text=out, metadata={'purpose': 'context'}))

    new_req = copy.deepcopy(request)
    new_user = _last_user_message(new_req.messages)
    assert new_user is not None  # mirrors the guard above; deepcopy preserves structure
    if context_part_index >= 0:
        new_user.content[context_part_index] = text_part
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
    """Registers generate action in the provided registry."""

    async def generate_action_fn(
        input: GenerateActionOptions,
        ctx: ActionRunContext,
    ) -> ModelResponse:
        on_chunk = cast(Callable[[ModelResponseChunk], None], ctx.streaming_callback) if ctx.is_streaming else None
        return await _generate_action(
            registry=registry,
            raw_request=input,
            on_chunk=on_chunk,
            context=ctx.context,
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
    resolved_middleware: list[BaseMiddleware] | None = None,
) -> ModelResponse:
    """Execute a generation request with tool calling and middleware support, wrapped in a util ``generate`` span.

    The registered ``/util/generate`` action calls :func:`_generate_action` directly,
    so reflection runs do not stack another util span on the action span.

    Middleware comes from one of two sources:

    - ``resolved_middleware``: a pre-resolved list from the veneer when the caller
      passed inline ``BaseMiddleware`` instances in ``use=`` (not serializable on the
      wire, so the veneer resolves them up-front and ``raw_request.use`` only carries
      the ``MiddlewareRef`` entries).
    - ``raw_request.use``: ``MiddlewareRef`` entries resolved here via the registry
      (used by JSON action dispatch and tests that call ``generate_action`` directly).

    When ``resolved_middleware`` is supplied, it is used as-is (it already reflects
    the full original ordering and any inline instances). Otherwise the resolver runs
    against ``raw_request.use`` alone.
    """
    span_name = 'generate'
    with run_in_new_span(
        SpanMetadata(name=span_name),
        labels={'genkit:type': 'util'},
    ) as span:
        span.set_attribute('genkit:name', span_name)
        with contextlib.suppress(Exception):
            span.set_attribute('genkit:input', raw_request.model_dump_json(by_alias=True, exclude_none=True))

        # Build a per-call child registry up front and route everything for this
        # call through it: middleware resolution, middleware-contributed tools,
        # and the recursive engine itself.  Two consequences:
        #   1. Anything middleware registers via ``self._registry`` (tools,
        #      values, etc.) is auto-cleaned when the call ends — it cannot
        #      leak into the parent or other concurrent ``generate()`` calls.
        #   2. Middleware can resolve dynamic tools contributed by other
        #      middleware in the same call, since they all share this scope.
        call_registry = registry.new_child()

        middleware = (
            resolved_middleware
            if resolved_middleware is not None
            else resolve_middleware_from_use(call_registry, raw_request.use)
        )
        # Per-call message queue shared across all recursive _generate_action
        # turns.  Middleware (and tool closures) enqueue extra user-role parts
        # here via ``enqueue_parts``; the engine drains and injects them as a
        # USER message at the start of the next wrap_generate iteration so the
        # model sees them on its very next turn.
        _queue: list[Message] = []

        def _enqueue_parts(parts: list[Part]) -> None:
            if _queue and _queue[-1].role == Role.USER:
                _queue[-1] = Message(role=Role.USER, content=list(_queue[-1].content) + list(parts))
            else:
                _queue.append(Message(role=Role.USER, content=list(parts)))

        # Register tools contributed by middleware (via ``BaseMiddleware.tools()``)
        # on the call registry so they're resolvable for this generate() only.
        # Their names are also appended to ``raw_request.tools`` so
        # ``resolve_parameters`` ships their definitions to the model and the
        # tool-loop can look them up in the tool_dict it builds from that list.
        if middleware:
            # Each middleware may return zero or more dynamic Actions from tools();
            # flatten in use= order so registration matches how hooks are chained.
            mw_tools: list[Action[Any, Any, Never]] = []
            for mw in middleware:
                contributed = mw.tools(_enqueue_parts)
                mw_tools.extend(contributed)

            if mw_tools:
                mw_tool_names: list[str] = []
                for t in mw_tools:
                    call_registry.register_action_from_instance(t)
                    mw_tool_names.append(t.name)
                existing = list(raw_request.tools) if raw_request.tools else []
                raw_request = raw_request.model_copy(
                    update={'tools': existing + mw_tool_names}
                )
        result = await _generate_action(
            call_registry, raw_request, on_chunk, message_index, current_turn, middleware, context,
            _enqueue_parts=_enqueue_parts, _queue=_queue,
        )
        with contextlib.suppress(Exception):
            span.set_attribute('genkit:output', result.model_dump_json(by_alias=True, exclude_none=True))
        return result


async def _generate_action(
    registry: Registry,
    raw_request: GenerateActionOptions,
    on_chunk: Callable[[ModelResponseChunk], None] | None = None,
    message_index: int = 0,
    current_turn: int = 0,
    middleware: list[BaseMiddleware] | None = None,
    context: dict[str, Any] | None = None,
    _enqueue_parts: Callable[[list[Part]], None] | None = None,
    _queue: list[Message] | None = None,
) -> ModelResponse:
    """Execute a generation request with tool calling and middleware support."""
    tools_in = raw_request.tools
    if tools_in:
        raw_request = raw_request.model_copy()
        raw_request.tools = await expand_wildcard_tools(registry, tools_in)

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
        middleware=middleware,
        enqueue_parts=_enqueue_parts,
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

    request = await action_to_generate_request(raw_request, tools, model)

    logger.debug('generate request', model=model.name, request=_redact_data_uris(request.model_dump()))

    prev_chunks: list[ModelResponseChunk] = []

    chunk_role: Role = Role.MODEL

    def make_chunk(role: Role, chunk: ModelResponseChunk) -> ModelResponseChunk:
        """Wrap a raw chunk with metadata and track message index changes."""
        nonlocal chunk_role, message_index

        if role != chunk_role and len(prev_chunks) > 0:
            message_index += 1

        chunk_role = role

        prev_to_send = copy.copy(prev_chunks)
        prev_chunks.append(chunk)

        def chunk_parser(chunk: ModelResponseChunk) -> Any:  # noqa: ANN401
            if formatter is None:
                return None
            return formatter.parse_chunk(chunk)

        return ModelResponseChunk(
            chunk,
            index=message_index,
            previous_chunks=prev_to_send,
            chunk_parser=chunk_parser if formatter else None,
        )

    def wrap_chunks(role: Role | None = None) -> Callable[[ModelResponseChunk], None]:
        """Return a callback that wraps chunks with the given role for streaming."""
        if role is None:
            role = Role.MODEL

        def wrapper(chunk: ModelResponseChunk) -> None:
            if on_chunk is not None:
                on_chunk(make_chunk(role, chunk))

        return wrapper

    if not middleware:
        middleware = []

    # Inject ``request.docs`` as a context part on the last user message.
    if request.docs:
        request = _augment_with_context(request)

    normalized_mw: list[BaseMiddleware] = list(middleware)

    async def dispatch_generate(
        params: GenerateHookParams,
        next_fn: Callable[[GenerateHookParams], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Chain wrap_generate middleware and call next_fn."""
        runner: Callable[[GenerateHookParams], Awaitable[ModelResponse]] = next_fn
        for mw in reversed(normalized_mw):
            _mw = mw
            _inner = runner

            async def run_next(
                p: GenerateHookParams,
                *,
                _m: BaseMiddleware = _mw,
                _i: Callable[[GenerateHookParams], Awaitable[ModelResponse]] = _inner,
            ) -> ModelResponse:
                return await _m.wrap_generate(p, _i)

            runner = run_next
        return await runner(params)

    async def dispatch_model(
        req: ModelRequest,
        chunk_callback: Callable[[ModelResponseChunk], None] | None,
    ) -> ModelResponse:
        async def run_model(params: ModelHookParams) -> ModelResponse:
            return (
                await model.run(
                    input=params.request,
                    context=params.context,
                    on_chunk=cast(Callable[[object], None], params.on_chunk) if params.on_chunk else None,
                )
            ).response

        runner: Callable[[ModelHookParams], Awaitable[ModelResponse]] = run_model
        for mw in reversed(normalized_mw):
            _mw = mw
            _inner = runner

            async def run_next(
                params: ModelHookParams,
                *,
                _mw: BaseMiddleware = _mw,
                _inner: Callable[[ModelHookParams], Awaitable[ModelResponse]] = _inner,
            ) -> ModelResponse:
                return await _mw.wrap_model(params, _inner)

            runner = cast(Callable[[ModelHookParams], Awaitable[ModelResponse]], run_next)

        return await runner(
            ModelHookParams(
                request=req,
                on_chunk=chunk_callback,
                context=context or {},
            )
        )

    # if resolving the 'resume' option above generated a tool message, stream it.
    if resumed_tool_message and on_chunk:
        wrap_chunks(Role.TOOL)(
            ModelResponseChunk(
                role=resumed_tool_message.role,
                content=resumed_tool_message.content,
            )
        )

    async def run_one_iteration(_params: GenerateHookParams) -> ModelResponse:
        """Execute one turn of the generate loop (model call + optional tool resolution)."""
        nonlocal request, message_index, chunk_role
        # Sync from params so wrap_generate middleware can reshape the request
        # by returning a model_copy(update={'request': ...}) to next_fn.
        # Without this, a middleware-modified params.request would be silently ignored.
        request = _params.request
        # Drain anything middleware queued during the previous turn's tool
        # calls and inject it as additional USER messages before the model
        # runs.  This is how a tool-side middleware (e.g. Filesystem read_file)
        # can make extra context — file contents, error notes, etc. — visible
        # to the model on the very next turn without forging a tool response.
        if _queue:
            queued = list(_queue)
            _queue.clear()
            if on_chunk:
                # Emit each queued message at the current index and advance once
                # per message.  We bypass `make_chunk` here because its role
                # tracker treats every USER chunk as a new message and would
                # double-count the role flip from MODEL to USER.
                for msg in queued:
                    msg_role = cast(Role, msg.role)
                    chunk = ModelResponseChunk(
                        role=msg_role,
                        content=msg.content,
                        index=message_index,
                        previous_chunks=list(prev_chunks),
                    )
                    prev_chunks.append(chunk)
                    on_chunk(chunk)
                    message_index += 1
                    chunk_role = msg_role
            request = request.model_copy(
                update={'messages': list(request.messages) + queued}
            )

        model_response = await dispatch_model(
            request,
            wrap_chunks() if on_chunk else None,
        )

        def message_parser(msg: Message) -> Any:  # noqa: ANN401
            if formatter is None:
                return None
            return formatter.parse_message(msg)

        # Extract schema_type for runtime Pydantic validation
        schema_type = raw_request.output.schema_type if raw_request.output else None

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
        out = raw_request.output
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

        if raw_request.return_tool_requests or len(tool_requests) == 0:
            if len(tool_requests) == 0:
                response.assert_valid_schema()
            return response

        max_iters = raw_request.max_turns if raw_request.max_turns else DEFAULT_MAX_TURNS

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
            registry, raw_request, generated_msg,
            middleware=normalized_mw, enqueue_parts=_enqueue_parts,
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
        if on_chunk and tool_msg:
            on_chunk(
                make_chunk(
                    Role.TOOL,
                    ModelResponseChunk(
                        role=tool_msg.role,
                        content=tool_msg.content,
                    ),
                )
            )

        next_request = copy.copy(raw_request)
        next_messages = copy.copy(raw_request.messages)
        next_messages.append(generated_msg)
        if tool_msg:
            next_messages.append(tool_msg)
        next_request.messages = next_messages
        if transfer_preamble:
            next_request = apply_transfer_preamble(next_request, transfer_preamble)

        # then recursively call for another loop.
        return await _generate_action(
            registry,
            raw_request=next_request,
            middleware=middleware,
            current_turn=current_turn + 1,
            message_index=message_index + 1,
            on_chunk=on_chunk,
            context=context,
            _enqueue_parts=_enqueue_parts,
            _queue=_queue,
        )

    generate_params = GenerateHookParams(
        options=raw_request,
        request=request,
        iteration=current_turn,
        message_index=message_index,
        on_chunk=on_chunk,
        enqueue_parts=_enqueue_parts,
    )
    return await dispatch_generate(generate_params, run_one_iteration)


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

    tools: list[Action[Any, Any, Any]] = []
    if request.tools:
        for tool_name in request.tools:
            try:
                tool_action = await resolve_tool(registry, tool_name)
            except GenkitError as e:
                raise Exception(f'Unable to resolve tool {tool_name}') from e
            tools.append(tool_action)

    format_def: FormatDef | None = None
    if request.output and request.output.format:
        looked_up_format = registry.lookup_value('format', request.output.format)
        if looked_up_format is None:
            raise ValueError(f'Unable to resolve format {request.output.format}')
        format_def = cast(FormatDef, looked_up_format)

    return (model_action, tools, format_def)


async def action_to_generate_request(
    options: GenerateActionOptions, resolved_tools: list[Action[Any, Any, Any]], _model: Action[Any, Any, Any]
) -> ModelRequest:
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
    middleware: list[BaseMiddleware] | None = None,
    enqueue_parts: Callable[[list[Part]], None] | None = None,
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
    mw_list = middleware or []

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
        if mw_list:
            params = ToolHookParams(tool_request_part=trp, tool=tool, enqueue_parts=enqueue_parts)

            async def next_fn(
                p: ToolHookParams,
            ) -> tuple[MultipartToolResponse | None, ToolRequestPart | None]:
                return await _resolve_tool_request(p.tool, p.tool_request_part)

            try:
                return await _chain_tool_middleware(mw_list, params, next_fn)
            except Exception as e:
                intr = _interrupt_from_tool_exc(e)
                if intr is None:
                    raise
                # Middleware raised Interrupt without calling next_fn — convert to
                # the same wire shape that _resolve_tool_request produces.  Any
                # tracing span is the middleware's responsibility (e.g. ToolApproval
                # wraps its raise in run_in_new_span explicitly).
                payload: dict[str, Any] | bool = intr.metadata if intr.metadata else True
                tool_meta = trp.metadata or {}
                return (
                    None,
                    ToolRequestPart(
                        tool_request=trp.tool_request,
                        metadata={**tool_meta, 'interrupt': payload},
                    ),
                )
        return await _resolve_tool_request(tool, trp)

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
                )
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
    tool: Action, tool_request_part: ToolRequestPart
) -> tuple[MultipartToolResponse | None, ToolRequestPart | None]:
    """Execute a tool.

    Returns ``(MultipartToolResponse, None)`` on success or ``(None, ToolRequestPart)``
    when interrupted.  The caller unpacks ``MultipartToolResponse`` into the wire
    ``ToolResponsePart`` so the 1-to-1 request/response LLM contract is preserved.
    """
    try:
        tool_response = (await tool.run(tool_request_part.tool_request.input)).response
        return (
            MultipartToolResponse(
                output=tool_response.model_dump() if isinstance(tool_response, BaseModel) else tool_response,
            ),
            None,
        )
    except Exception as e:
        intr = _interrupt_from_tool_exc(e)
        if intr is not None:
            payload: dict[str, Any] | bool = intr.metadata if intr.metadata else True
            tool_meta = tool_request_part.metadata or {}
            return (
                None,
                ToolRequestPart(
                    tool_request=tool_request_part.tool_request,
                    metadata={**tool_meta, 'interrupt': payload},
                ),
            )
        raise


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
    middleware: list[BaseMiddleware] | None = None,
    enqueue_parts: Callable[[list[Part]], None] | None = None,
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
            middleware=middleware,
            enqueue_parts=enqueue_parts,
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
    middleware: list[BaseMiddleware] | None = None,
    enqueue_parts: Callable[[list[Part]], None] | None = None,
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
        executed = await _run_restart_through_middleware(
            tool, restart_trp, middleware=middleware, enqueue_parts=enqueue_parts
        )
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
    middleware: list[BaseMiddleware] | None,
    enqueue_parts: Callable[[list[Part]], None] | None,
) -> ToolResponsePart:
    """Run a restarted tool through the wrap_tool middleware chain.

    Restart paths reuse the same dispatch as fresh tool calls so middleware
    (ToolApproval, Filesystem error queueing, etc.) sees every tool execution
    regardless of whether it was triggered by the model or by a resumed
    interrupt.  Without this, a restart would silently bypass approval checks.
    """
    mw_list = middleware or []
    if not mw_list:
        return await run_tool_after_restart(tool, restart_trp)

    params = ToolHookParams(
        tool_request_part=restart_trp, tool=tool, enqueue_parts=enqueue_parts
    )

    async def next_fn(
        p: ToolHookParams,
    ) -> tuple[MultipartToolResponse | None, ToolRequestPart | None]:
        executed = await run_tool_after_restart(p.tool, restart_trp)
        return (
            MultipartToolResponse(
                output=executed.tool_response.output,
                content=[Part.model_validate(c) for c in (executed.tool_response.content or [])],
            ),
            None,
        )

    multipart, interrupt_part = await _chain_tool_middleware(mw_list, params, next_fn)
    if interrupt_part is not None:
        # Re-interrupting during restart is a hard error — same as the legacy
        # run_tool_after_restart path, which raises FAILED_PRECONDITION when
        # the inner tool throws an Interrupt during restart.
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message='Tool interrupted again during a restart execution; not supported yet.',
        )
    if multipart is None:
        # Defensive: middleware contract requires exactly one of the two to be set.
        raise GenkitError(
            status='INTERNAL',
            message='Tool middleware returned (None, None) for a restart execution.',
        )
    return ToolResponsePart(
        tool_response=ToolResponse(
            name=restart_trp.tool_request.name,
            ref=restart_trp.tool_request.ref,
            output=multipart.output,
            content=[p.model_dump() for p in multipart.content] if multipart.content else None,
        )
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
