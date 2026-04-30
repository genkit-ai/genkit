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
import inspect
import re
from collections.abc import Callable, Sequence
from typing import Any, cast

from pydantic import BaseModel

from genkit._ai._formats._types import FormatDef, Formatter
from genkit._ai._messages import inject_instructions
from genkit._ai._middleware import augment_with_context
from genkit._ai._model import (
    Message,
    ModelMiddleware,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
)
from genkit._ai._resource import ResourceArgument, ResourceInput, find_matching_resource, resolve_resources
from genkit._ai._tools import Tool, ToolInterruptError
from genkit._core._action import (
    GENKIT_DYNAMIC_ACTION_PROVIDER_ATTR,
    Action,
    ActionKind,
    ActionRunContext,
)
from genkit._core._error import GenkitError
from genkit._core._logger import get_logger
from genkit._core._model import GenerateActionOptions
from genkit._core._registry import Registry
from genkit._core._tracing import run_in_new_span
from genkit._core._typing import (
    FinishReason,
    Part,
    Role,
    SpanMetadata,
    ToolDefinition,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)

DEFAULT_MAX_TURNS = 5

logger = get_logger(__name__)


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
    middleware: list[ModelMiddleware] | None = None,
    context: dict[str, Any] | None = None,
) -> ModelResponse:
    """Run generation with a util ``generate`` span.

    The registered ``/util/generate`` action calls `_generate_action` directly
    so reflection runs do not stack another util span on the action span.
    """
    span_name = 'generate'
    with run_in_new_span(
        SpanMetadata(name=span_name),
        labels={'genkit:type': 'util'},
    ) as span:
        span.set_attribute('genkit:name', span_name)
        with contextlib.suppress(Exception):
            span.set_attribute('genkit:input', raw_request.model_dump_json(by_alias=True, exclude_none=True))
        result = await _generate_action(
            registry, raw_request, on_chunk, message_index, current_turn, middleware, context
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
    middleware: list[ModelMiddleware] | None = None,
    context: dict[str, Any] | None = None,
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
    ) = await _resolve_resume_options(registry, raw_request)

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

    supports_context = False
    if model.metadata:
        model_info = model.metadata.get('model')
        if model_info and isinstance(model_info, dict):
            model_info_dict = cast(dict[str, object], model_info)
            supports_info = model_info_dict.get('supports')
            if supports_info and isinstance(supports_info, dict):
                supports_dict = cast(dict[str, object], supports_info)
                supports_context = bool(supports_dict.get('context'))
    # if it doesn't support contextm inject context middleware
    if raw_request.docs and not supports_context:
        middleware.append(augment_with_context())

    async def dispatch(
        index: int,
        req: ModelRequest,
        ctx: ActionRunContext,
        chunk_callback: Callable[[ModelResponseChunk], None] | None,
    ) -> ModelResponse:
        """Dispatch request through middleware chain to the model."""
        if not middleware or index == len(middleware):
            # End of the chain, call the original model action
            return (
                await model.run(
                    input=req,
                    context=ctx.context,
                    on_chunk=cast(Callable[[object], None], chunk_callback) if chunk_callback else None,
                )
            ).response

        current_middleware = middleware[index]
        n_params = len(inspect.signature(current_middleware).parameters)

        if n_params == 4:
            # Streaming middleware: (req, ctx, on_chunk, next) -> response
            async def next_fn_streaming(
                modified_req: ModelRequest | None = None,
                modified_ctx: ActionRunContext | None = None,
                modified_on_chunk: Callable[[ModelResponseChunk], None] | None = None,
            ) -> ModelResponse:
                return await dispatch(
                    index + 1,
                    modified_req if modified_req else req,
                    modified_ctx if modified_ctx else ctx,
                    modified_on_chunk if modified_on_chunk is not None else chunk_callback,
                )

            return await current_middleware(req, ctx, chunk_callback, next_fn_streaming)
        else:
            # Simple middleware: (req, ctx, next) -> response
            async def next_fn_simple(
                modified_req: ModelRequest | None = None,
                modified_ctx: ActionRunContext | None = None,
            ) -> ModelResponse:
                return await dispatch(
                    index + 1,
                    modified_req if modified_req else req,
                    modified_ctx if modified_ctx else ctx,
                    chunk_callback,
                )

            return await current_middleware(req, ctx, next_fn_simple)

    # if resolving the 'resume' option above generated a tool message, stream it.
    if resumed_tool_message and on_chunk:
        wrap_chunks(Role.TOOL)(
            ModelResponseChunk(
                role=resumed_tool_message.role,
                content=resumed_tool_message.content,
            )
        )

    model_response = await dispatch(
        0,
        request,
        ActionRunContext(context=context),
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

    logger.debug('generate response', response=_redact_data_uris(response.model_dump()))

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
    ) = await resolve_tool_requests(registry, raw_request, generated_msg)

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

    # then recursively call for another loop
    return await _generate_action(
        registry,
        raw_request=next_request,
        # middleware: middleware,
        current_turn=current_turn + 1,
        message_index=message_index + 1,
        on_chunk=on_chunk,
    )


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
    registry: Registry, request: GenerateActionOptions, message: Message
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

    outs = await asyncio.gather(*[_resolve_tool_request(tool, trp) for _, tool, trp in work])

    has_interrupts = False
    response_parts: list[Part] = []
    for (idx, _tool, tool_req_root), (tool_response_part, interrupt_part) in zip(work, outs, strict=True):
        if tool_response_part:
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
) -> tuple[ToolResponsePart | None, ToolRequestPart | None]:
    """Execute a tool.

    Returns ``(ToolResponsePart, None)`` on success or ``(None, ToolRequestPart)`` when interrupted.
    """
    try:
        tool_response = (await tool.run(tool_request_part.tool_request.input)).response
        return (
            ToolResponsePart(
                tool_response=ToolResponse(
                    name=tool_request_part.tool_request.name,
                    ref=tool_request_part.tool_request.ref,
                    output=tool_response.model_dump() if isinstance(tool_response, BaseModel) else tool_response,
                )
            ),
            None,
        )
    except Exception as e:
        intr = _interrupt_from_tool_exc(e)
        if intr is not None:
            payload: dict[str, Any] | bool = intr.metadata if intr.metadata else True
            tool_meta = tool_request_part.metadata or {}
            if not isinstance(tool_meta, dict):
                tool_meta = dict(tool_meta)
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
        return tool_ref.action

    if tool_ref.startswith('/'):
        tool = await registry.resolve_action_by_key(tool_ref)
        if tool is not None:
            return tool

    tool = await registry.resolve_action(kind=ActionKind.TOOL, name=tool_ref)
    if tool is None:
        raise GenkitError(status='NOT_FOUND', message=f'Unable to resolve tool {tool_name}')
    return tool


async def _resolve_resume_options(
    _registry: Registry, raw_request: GenerateActionOptions
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
    # Create a new list for content to avoid mutation during iteration
    updated_content = list(last_message.content)
    for part in last_message.content:
        if not isinstance(part.root, ToolRequestPart):
            i += 1
            continue

        resumed_request, resumed_response = await _resolve_resumed_tool_request(_registry, raw_request, part)
        tool_responses.append(Part(root=resumed_response))
        updated_content[i] = Part(root=resumed_request)
        i += 1
    last_message.content = updated_content

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
    revised_request.messages.append(tool_message)

    return (revised_request, None, tool_message)


async def _resolve_resumed_tool_request(
    registry: Registry, raw_request: GenerateActionOptions, tool_request_part: Part
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
        executed = await run_tool_after_restart(tool, restart_trp)
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
