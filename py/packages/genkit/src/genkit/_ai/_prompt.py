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


"""Prompt management and templating."""

import asyncio
import os
import weakref
from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Generic, TypedDict, TypeVar, cast

from dotpromptz.typing import (
    DataArgument,
    PromptFunction,
    PromptInputConfig,
    PromptMetadata,
)
from pydantic import BaseModel, ConfigDict
from typing_extensions import Unpack

from genkit._ai._generate import (
    generate_action,
    registry_with_inline_tools,
    resolve_tool,
    to_tool_definition,
    tools_to_action_names,
)
from genkit._ai._model import (
    ModelMiddleware,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
)
from genkit._ai._tools import Tool
from genkit._core._action import Action, ActionKind, ActionRunContext, StreamingCallback, create_action_key
from genkit._core._channel import Channel
from genkit._core._error import GenkitError
from genkit._core._logger import get_logger
from genkit._core._model import Document, GenerateActionOptions, Message, ModelConfig
from genkit._core._registry import Registry
from genkit._core._schema import to_json_schema
from genkit._core._typing import (
    GenerateActionOutputConfig,
    OutputConfig,
    Part,
    Resume,
    Role,
    TextPart,
    ToolChoice,
    ToolRequestPart,
    ToolResponsePart,
)

ModelStreamingCallback = StreamingCallback

logger = get_logger(__name__)

# TypeVars for generic input/output typing
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


class OutputOptions(TypedDict, total=False):
    """Output format/schema configuration for prompt generation."""

    format: str | None
    content_type: str | None
    instructions: str | None
    schema: type | dict[str, Any] | str | None
    json_schema: dict[str, Any] | None
    constrained: bool | None


def _normalize_resume_respond_parts(
    value: ToolResponsePart | list[ToolResponsePart] | None,
) -> list[ToolResponsePart] | None:
    if value is None:
        return None
    return list(value) if isinstance(value, list) else [value]


def _normalize_resume_restart_parts(
    value: ToolRequestPart | list[ToolRequestPart] | None,
) -> list[ToolRequestPart] | None:
    if value is None:
        return None
    return list(value) if isinstance(value, list) else [value]


def resume_options_to_resume(
    *,
    resume_respond: ToolResponsePart | list[ToolResponsePart] | None = None,
    resume_restart: ToolRequestPart | list[ToolRequestPart] | None = None,
    resume_metadata: dict[str, Any] | None = None,
) -> Resume | None:
    """Build wire Resume from flat keyword options (``generate`` / prompts)."""
    respond = _normalize_resume_respond_parts(resume_respond)
    restart = _normalize_resume_restart_parts(resume_restart)
    if respond is None and restart is None and resume_metadata is None:
        return None
    return Resume(respond=respond, restart=restart, metadata=resume_metadata)


class PromptGenerateOptions(TypedDict, total=False):
    """Runtime options for prompt execution (config, tools, messages, etc.)."""

    model: str | None
    config: dict[str, Any] | ModelConfig | None
    messages: list[Message] | None
    docs: list[Document] | None
    tools: Sequence[str | Tool] | None
    resources: list[str] | None
    tool_choice: ToolChoice | None
    output: OutputOptions | None
    resume: Resume | dict[str, Any] | None
    resume_respond: ToolResponsePart | list[ToolResponsePart] | None
    resume_restart: ToolRequestPart | list[ToolRequestPart] | None
    resume_metadata: dict[str, Any] | None
    return_tool_requests: bool | None
    max_turns: int | None
    on_chunk: ModelStreamingCallback | None
    use: list[ModelMiddleware] | None
    context: dict[str, Any] | None
    step_name: str | None
    metadata: dict[str, Any] | None


class ModelStreamResponse(Generic[OutputT]):
    """Response from streaming prompt execution with stream and response properties."""

    def __init__(
        self,
        channel: Channel[ModelResponseChunk, ModelResponse[OutputT]],
        response_future: asyncio.Future[ModelResponse[OutputT]],
    ) -> None:
        """Initialize with streaming channel and response future."""
        self._channel: Channel[ModelResponseChunk, ModelResponse[OutputT]] = channel
        self._response_future: asyncio.Future[ModelResponse[OutputT]] = response_future

    @property
    def stream(self) -> AsyncIterable[ModelResponseChunk]:
        """Get the async iterable of response chunks.

        Returns:
            An async iterable that yields ModelResponseChunk objects
            as they are received from the model. Each chunk contains:
            - text: The partial text generated so far
            - index: The chunk index
            - Additional metadata from the model
        """
        return self._channel

    @property
    def response(self) -> Awaitable[ModelResponse[OutputT]]:
        """Get the awaitable for the complete response.

        Returns:
            An awaitable that resolves to a ModelResponse containing:
            - text: The complete generated text
            - output: The typed output (when using Output[T])
            - messages: The full message history
            - usage: Token usage statistics
            - finish_reason: Why generation stopped (e.g., 'stop', 'length')
            - Any tool calls or interrupts from the response
        """
        return self._response_future


@dataclass
class PromptCache:
    """Model for a prompt cache."""

    user_prompt: PromptFunction[Any] | None = None
    system: PromptFunction[Any] | None = None
    messages: PromptFunction[Any] | None = None


class PromptConfig(BaseModel):
    """Model for a prompt action."""

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    variant: str | None = None
    model: str | None = None
    config: dict[str, Any] | ModelConfig | None = None
    description: str | None = None
    input_schema: type | dict[str, Any] | str | None = None
    system: str | list[Part] | None = None
    prompt: str | list[Part] | None = None
    messages: str | list[Message] | None = None
    output_format: str | None = None
    output_content_type: str | None = None
    output_instructions: str | None = None
    output_schema: type | dict[str, Any] | str | None = None
    output_constrained: bool | None = None
    max_turns: int | None = None
    return_tool_requests: bool | None = None
    metadata: dict[str, Any] | None = None
    tools: Sequence[str | Tool] | None = None
    tool_choice: ToolChoice | None = None
    use: list[ModelMiddleware] | None = None
    docs: list[Document] | None = None
    resume_respond: ToolResponsePart | list[ToolResponsePart] | None = None
    resume_restart: ToolRequestPart | list[ToolRequestPart] | None = None
    resume_metadata: dict[str, Any] | None = None
    resources: list[str] | None = None
    resume: Resume | None = None


class ExecutablePrompt(Generic[InputT, OutputT]):
    """A callable prompt with typed input/output that generates AI responses."""

    def __init__(
        self,
        registry: Registry,
        variant: str | None = None,
        model: str | None = None,
        config: dict[str, Any] | ModelConfig | None = None,
        description: str | None = None,
        input_schema: type | dict[str, Any] | str | None = None,
        system: str | list[Part] | None = None,
        prompt: str | list[Part] | None = None,
        messages: str | list[Message] | None = None,
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: str | None = None,
        output_schema: type | dict[str, Any] | str | None = None,
        output_constrained: bool | None = None,
        max_turns: int | None = None,
        return_tool_requests: bool | None = None,
        metadata: dict[str, Any] | None = None,
        tools: Sequence[str | Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        use: list[ModelMiddleware] | None = None,
        docs: list[Document] | None = None,
        resources: list[str] | None = None,
        name: str | None = None,
        ns: str | None = None,
    ) -> None:
        """Initialize prompt with configuration, templates, and schema options."""
        self._registry = registry
        self._variant = variant
        self._model = model
        self._config = config
        self._description = description
        self._input_schema = input_schema
        self._system = system
        self._prompt = prompt
        self._messages = messages
        self._output_format = output_format
        self._output_content_type = output_content_type
        self._output_instructions = output_instructions
        self._output_schema = output_schema
        self._output_constrained = output_constrained
        self._max_turns = max_turns
        self._return_tool_requests = return_tool_requests
        self._metadata = metadata
        self._tools = tools
        self._tool_choice = tool_choice
        self._use = use
        self._docs = docs
        self._resources = resources
        self._cache_prompt: PromptCache = PromptCache()
        self._name = name
        self._ns = ns
        self._prompt_action: Action | None = None

    @property
    def ref(self) -> dict[str, Any]:
        """Reference object with prompt name and metadata."""
        return {
            'name': registry_definition_key(self._name, self._variant, self._ns) if self._name else None,
            'metadata': self._metadata,
        }

    async def _ensure_resolved(self) -> None:
        if self._prompt_action or not self._name:
            return

        # Preserve Pydantic schema type if it was explicitly provided via ai.prompt(..., output=Output(schema=T))
        # The resolved prompt from .prompt file will have a dict schema, but we want to keep the Pydantic type
        # for runtime validation to get proper typed output.
        original_output_schema = self._output_schema

        resolved = await lookup_prompt(self._registry, self._name, self._variant)
        self._model = resolved._model
        self._config = resolved._config
        self._description = resolved._description
        self._input_schema = resolved._input_schema
        self._system = resolved._system
        self._prompt = resolved._prompt
        self._messages = resolved._messages
        self._output_format = resolved._output_format
        self._output_content_type = resolved._output_content_type
        self._output_instructions = resolved._output_instructions
        # Keep original Pydantic type if provided, otherwise use resolved (dict) schema
        if isinstance(original_output_schema, type) and issubclass(original_output_schema, BaseModel):
            self._output_schema = original_output_schema
        else:
            self._output_schema = resolved._output_schema
        self._output_constrained = resolved._output_constrained
        self._max_turns = resolved._max_turns
        self._return_tool_requests = resolved._return_tool_requests
        self._metadata = resolved._metadata
        self._tools = resolved._tools
        self._tool_choice = resolved._tool_choice
        self._use = resolved._use
        self._docs = resolved._docs
        self._resources = resolved._resources
        self._prompt_action = resolved._prompt_action

    async def __call__(
        self,
        input: InputT | dict[str, Any] | None = None,
        **opts: Unpack[PromptGenerateOptions],
    ) -> ModelResponse[OutputT]:
        """Execute the prompt and return the response.

        Args:
            input: Template variables for rendering.
        """
        return await self._call_impl(input, opts)  # ty: ignore[invalid-argument-type]  # ty doesn't infer Unpack[TD] as TD in function body (PEP 692 gap)

    async def _call_impl(
        self,
        input: InputT | dict[str, Any] | None,
        opts: PromptGenerateOptions,
    ) -> ModelResponse[OutputT]:
        """Execute the prompt with resolved opts. Used by __call__ and stream."""
        await self._ensure_resolved()
        on_chunk = opts.get('on_chunk')
        middleware = opts.get('use') or self._use
        context = opts.get('context')
        prompt_config = self._prompt_config_for_call(opts)
        registry = await registry_with_inline_tools(self._registry, prompt_config.tools)
        gen_options = await executable_prompt_call_to_generate_options(self, registry, prompt_config, input, opts)
        result = await generate_action(
            registry,
            gen_options,
            on_chunk=on_chunk,
            middleware=middleware,
            context=context if context else ActionRunContext._current_context(),  # pyright: ignore[reportPrivateUsage]
        )
        return cast(ModelResponse[OutputT], result)

    def _prompt_config_for_call(self, opts: PromptGenerateOptions) -> PromptConfig:
        """Merge this prompt's definition with per-call ``opts`` into a :class:`PromptConfig`."""
        output_opts = opts.get('output') or {}
        merged_config: dict[str, Any] | ModelConfig | None
        if opts.get('config') is not None:
            base = (
                self._config.model_dump(exclude_none=True)
                if isinstance(self._config, BaseModel)
                else (self._config or {})
            )
            opt_config = opts.get('config')
            override = (
                opt_config.model_dump(exclude_none=True) if isinstance(opt_config, BaseModel) else (opt_config or {})
            )
            merged_config = {**base, **override} if base or override else None
        else:
            merged_config = self._config

        merged_metadata = (
            {**(self._metadata or {}), **(opts.get('metadata') or {})} if opts.get('metadata') else self._metadata
        )

        def _or(opt_val: Any, default: Any) -> Any:  # noqa: ANN401
            return opt_val if opt_val is not None else default

        return PromptConfig(
            model=opts.get('model') or self._model,
            prompt=self._prompt,
            system=self._system,
            messages=self._messages,
            tools=opts.get('tools') or self._tools,
            return_tool_requests=_or(opts.get('return_tool_requests'), self._return_tool_requests),
            tool_choice=opts.get('tool_choice') or self._tool_choice,
            config=merged_config,
            max_turns=_or(opts.get('max_turns'), self._max_turns),
            output_format=output_opts.get('format') or self._output_format,
            output_content_type=output_opts.get('content_type') or self._output_content_type,
            output_instructions=_or(output_opts.get('instructions'), self._output_instructions),
            output_schema=output_opts.get('schema') or output_opts.get('json_schema') or self._output_schema,
            output_constrained=_or(output_opts.get('constrained'), self._output_constrained),
            input_schema=self._input_schema,
            metadata=merged_metadata,
            docs=self._docs,
            resources=opts.get('resources') or self._resources,
            resume_respond=opts.get('resume_respond'),
            resume_restart=opts.get('resume_restart'),
            resume_metadata=opts.get('resume_metadata'),
        )

    def stream(
        self,
        input: InputT | dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        **opts: Unpack[PromptGenerateOptions],
    ) -> ModelStreamResponse[OutputT]:
        """Stream the prompt execution, returning (stream, response_future)."""
        channel: Channel[ModelResponseChunk, ModelResponse[OutputT]] = Channel(timeout=timeout)
        stream_opts: PromptGenerateOptions = {
            **opts,  # ty doesn't infer Unpack[TD] as TD in function body (PEP 692 gap)
            'on_chunk': lambda c: channel.send(cast(ModelResponseChunk, c)),
        }
        resp = self._call_impl(input, stream_opts)
        response_future: asyncio.Future[ModelResponse[OutputT]] = asyncio.create_task(resp)
        channel.set_close_future(response_future)

        return ModelStreamResponse[OutputT](channel=channel, response_future=response_future)

    async def render(
        self,
        input: InputT | dict[str, Any] | None = None,
        **opts: Unpack[PromptGenerateOptions],
    ) -> GenerateActionOptions:
        """Render the prompt template without executing, returning GenerateActionOptions.

        Same keyword options as ``__call__`` (see PromptGenerateOptions).
        """
        call_opts: PromptGenerateOptions = opts  # ty: ignore[invalid-assignment]  # ty treats **opts as a plain dict here; callers are still validated against PromptGenerateOptions.
        await self._ensure_resolved()
        prompt_config = self._prompt_config_for_call(call_opts)
        registry = await registry_with_inline_tools(self._registry, prompt_config.tools)
        return await executable_prompt_call_to_generate_options(self, registry, prompt_config, input, call_opts)


def register_prompt_actions(
    registry: Registry,
    executable_prompt: ExecutablePrompt[Any, Any],
    name: str,
    variant: str | None = None,
) -> None:
    """Register PROMPT and EXECUTABLE_PROMPT actions for a prompt.

    This links the executable prompt to actions in the registry, enabling
    lookup and DevUI integration.
    """
    action_metadata: dict[str, object] = {
        'type': 'prompt',
        'source': 'programmatic',
        'prompt': {
            'name': name,
            'variant': variant or '',
        },
    }

    async def prompt_action_fn(input: Any = None) -> ModelRequest:  # noqa: ANN401
        await executable_prompt._ensure_resolved()
        call_opts: PromptGenerateOptions = {}
        prompt_config = executable_prompt._prompt_config_for_call(call_opts)
        registry = await registry_with_inline_tools(executable_prompt._registry, prompt_config.tools)
        gen_options = await executable_prompt_call_to_generate_options(
            executable_prompt, registry, prompt_config, input, call_opts
        )
        return await to_generate_request(registry, gen_options)

    async def executable_prompt_action_fn(input: Any = None) -> GenerateActionOptions:  # noqa: ANN401
        await executable_prompt._ensure_resolved()
        call_opts: PromptGenerateOptions = {}
        prompt_config = executable_prompt._prompt_config_for_call(call_opts)
        registry = await registry_with_inline_tools(executable_prompt._registry, prompt_config.tools)
        return await executable_prompt_call_to_generate_options(
            executable_prompt, registry, prompt_config, input, call_opts
        )

    action_name = registry_definition_key(name, variant)
    prompt_action = registry.register_action(
        kind=ActionKind.PROMPT,
        name=action_name,
        fn=prompt_action_fn,
        metadata=action_metadata,
    )

    executable_prompt_action = registry.register_action(
        kind=ActionKind.EXECUTABLE_PROMPT,
        name=action_name,
        fn=executable_prompt_action_fn,
        metadata=action_metadata,
    )

    # Link them
    executable_prompt._prompt_action = prompt_action  # pyright: ignore[reportPrivateUsage]
    setattr(prompt_action, '_executable_prompt', weakref.ref(executable_prompt))  # noqa: B010
    setattr(executable_prompt_action, '_executable_prompt', weakref.ref(executable_prompt))  # noqa: B010


def _resolve_output_schema(
    registry: Registry,
    output_schema: type | dict[str, Any] | str | None,
    output: GenerateActionOutputConfig,
) -> None:
    """Resolve output schema and populate the output config.

    Handles three types of output_schema:
    - str: Schema name - look up JSON schema and type from registry
    - Pydantic type: Store both JSON schema and type for runtime validation
    - dict: Raw JSON schema - convert directly

    Args:
        registry: The registry to use for schema lookups.
        output_schema: The schema to resolve (string name, Pydantic type, or dict).
        output: The output config to populate with json_schema and schema_type.
    """
    if output_schema is None:
        return

    if isinstance(output_schema, str):
        # Schema name - look up from registry
        resolved_schema = registry.lookup_schema(output_schema)
        if resolved_schema:
            output.json_schema = resolved_schema
        # Also look up the schema type for runtime validation
        schema_type = registry.lookup_schema_type(output_schema)
        if schema_type:
            output.schema_type = schema_type
    elif isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
        # Pydantic type - store both JSON schema and type
        output.json_schema = to_json_schema(output_schema)
        output.schema_type = output_schema
    else:
        # dict (raw JSON schema)
        output.json_schema = to_json_schema(output_schema)


async def to_generate_action_options(
    registry: Registry,
    options: PromptConfig,
) -> GenerateActionOptions:
    """Render ``options`` into :class:`GenerateActionOptions`.

    Pass :class:`PromptConfig` from :func:`Genkit.generate` or from
    :func:`render_prompt_config_for_executable_call` after template expansion (final ``messages``,
    merged ``docs``, optional ``resume``).

    ``registry`` should come from :func:`~genkit._ai._generate.registry_with_inline_tools` when
    ``tools`` may include unregistered :class:`~genkit._ai._tools.Tool` instances.
    """
    model = options.model or cast(str | None, registry.lookup_value('defaultModel', 'defaultModel'))
    if model is None:
        raise GenkitError(status='INVALID_ARGUMENT', message='No model configured.')

    ri: dict[str, Any] = {}
    cache = PromptCache()
    resolved_msgs: list[Message] = []
    if options.system:
        result = await render_system_prompt(registry, ri, options, cache, None)
        resolved_msgs.append(result)
    if options.messages:
        resolved_msgs.extend(await render_message_prompt(registry, ri, options, cache, None, history=None))
    if options.prompt:
        result = await render_user_prompt(registry, ri, options, cache, None)
        resolved_msgs.append(result)

    # If is schema is set but format is not explicitly set, default to
    # `json` format.
    output_format = 'json' if options.output_schema and not options.output_format else options.output_format

    output = GenerateActionOutputConfig()
    if output_format:
        output.format = output_format
    if options.output_content_type:
        output.content_type = options.output_content_type
    if options.output_instructions is not None:
        output.instructions = options.output_instructions
    _resolve_output_schema(registry, options.output_schema, output)
    if options.output_constrained is not None:
        output.constrained = options.output_constrained

    resume: Resume | None = None
    if options.resume is not None:
        resume = options.resume
    else:
        resume = resume_options_to_resume(
            resume_respond=options.resume_respond,
            resume_restart=options.resume_restart,
            resume_metadata=options.resume_metadata,
        )

    # Convert tool refs (str name or Tool object) to string names for GenerateActionOptions
    tools_refs = tools_to_action_names(options.tools)

    merged_docs = await render_docs({}, options, None)

    return GenerateActionOptions(
        model=model,
        messages=resolved_msgs,  # type: ignore[arg-type]
        config=options.config,
        tools=tools_refs,
        return_tool_requests=options.return_tool_requests,
        tool_choice=options.tool_choice,
        output=output,
        max_turns=options.max_turns,
        docs=merged_docs,  # type: ignore[arg-type]
        resume=resume,
    )


def coerce_prompt_template_input(template_input: Any) -> dict[str, Any]:  # noqa: ANN401
    """Normalize executable-prompt ``input`` to template data for rendering."""
    if template_input is None:
        return {}
    if isinstance(template_input, dict):
        return {str(k): v for k, v in template_input.items()}
    if isinstance(template_input, BaseModel):
        return template_input.model_dump()
    if hasattr(template_input, 'dict'):
        dict_func = getattr(template_input, 'dict', None)
        return cast(Callable[[], dict[str, Any]], dict_func)()
    return cast(dict[str, Any], template_input)


def resume_from_prompt_call_opts(opts: PromptGenerateOptions) -> Resume | None:
    """Normalize ``opts['resume']`` to :class:`Resume` (model instance or loose dict).

    Dict values may use a single ``respond`` / ``restart`` part instead of a list; those are
    normalized before :meth:`~genkit._core._typing.Resume.model_validate`.
    """
    raw = opts.get('resume')
    if raw is None:
        return None
    if isinstance(raw, Resume):
        resume = raw
    elif isinstance(raw, dict):
        if not raw:
            return None
        d: dict[str, Any] = {str(k): v for k, v in raw.items()}
        if 'respond' in d:
            d['respond'] = _normalize_resume_respond_parts(d['respond'])
        if 'restart' in d:
            d['restart'] = _normalize_resume_restart_parts(d['restart'])
        resume = Resume.model_validate(d)
    else:
        return None
    if resume.respond is None and resume.restart is None and resume.metadata is None:
        return None
    return resume


async def to_generate_request(registry: Registry, options: GenerateActionOptions) -> ModelRequest:
    """Convert GenerateActionOptions to ModelRequest, resolving tool names."""
    tools: list[Action] = []
    if options.tools:
        for tool_ref in options.tools:
            tools.append(await resolve_tool(registry, tool_ref))

    tool_defs = [to_tool_definition(tool) for tool in tools] if tools else []

    if not options.messages:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message='at least one message is required in generate request',
        )

    output_config = OutputConfig(
        content_type=options.output.content_type if options.output else None,
        format=options.output.format if options.output else None,
        schema_=options.output.json_schema if options.output else None,
        constrained=options.output.constrained if options.output else None,
    )
    return ModelRequest(
        # Field validators auto-wrap MessageData -> Message and DocumentData -> Document
        messages=options.messages,  # type: ignore[arg-type]
        config=options.config if options.config is not None else {},  # type: ignore[arg-type]
        docs=options.docs if options.docs else None,  # type: ignore[arg-type]
        tools=tool_defs,
        tool_choice=options.tool_choice,
        output_format=output_config.format,
        output_schema=output_config.schema_,
        output_constrained=output_config.constrained,
        output_content_type=output_config.content_type,
    )


def _normalize_prompt_arg(
    prompt: str | list[Part] | None,
) -> list[Part]:
    """Convert string/Part/list to list[Part]."""
    if not prompt:
        return []
    if isinstance(prompt, str):
        # Part is a RootModel, so we pass content via 'root' parameter
        return [Part(root=TextPart(text=prompt))]
    elif isinstance(prompt, list):
        return prompt
    elif isinstance(prompt, Part):  # pyright: ignore[reportUnnecessaryIsInstance]
        return [prompt]
    else:
        return []  # pyright: ignore[reportUnreachable] - defensive fallback


async def _render_template(
    registry: Registry,
    role: Role,
    template: str | list[Part] | None,
    input: dict[str, Any],
    input_schema: type | dict[str, Any] | str | None,
    metadata: dict[str, Any] | None,
    compiled_fn: PromptFunction[Any] | None,
    context: dict[str, Any] | None,
) -> tuple[Message, PromptFunction[Any] | None]:
    """Compile and render a prompt template, returning (message, compiled_fn)."""
    if isinstance(template, str):
        if compiled_fn is None:
            compiled_fn = await registry.dotprompt.compile(template)

        if metadata:
            context = {**(context or {}), 'state': metadata.get('state')}

        rendered_parts = cast(
            list[Part],
            await render_dotprompt_to_parts(
                context or {},
                compiled_fn,
                input,
                PromptMetadata(
                    input=PromptInputConfig(
                        schema=to_json_schema(input_schema) if input_schema else None,
                    )
                ),
            ),
        )
        return Message(role=role, content=rendered_parts), compiled_fn

    return Message(role=role, content=_normalize_prompt_arg(template)), compiled_fn


async def render_system_prompt(
    registry: Registry,
    input: dict[str, Any],
    options: PromptConfig,
    prompt_cache: PromptCache,
    context: dict[str, Any] | None = None,
) -> Message:
    """Render the system prompt."""
    msg, prompt_cache.system = await _render_template(
        registry,
        Role.SYSTEM,
        options.system,
        input,
        options.input_schema,
        options.metadata,
        prompt_cache.system,
        context,
    )
    return msg


async def render_dotprompt_to_parts(
    context: dict[str, Any],
    prompt_function: PromptFunction[Any],
    input_: dict[str, Any],
    options: PromptMetadata[Any] | None = None,
) -> list[dict[str, Any]]:
    """Execute a compiled dotprompt function and return parts as dicts."""
    # Flatten input and context for template resolution
    flattened_data = {**(context or {}), **(input_ or {})}
    rendered = await prompt_function(
        data=DataArgument[dict[str, Any]](
            input=flattened_data,
            context=context,
        ),
        options=options,
    )

    if len(rendered.messages) > 1:
        raise Exception('parts template must produce only one message')

    # Convert parts to dicts for Pydantic re-validation when creating new Message
    part_rendered: list[dict[str, Any]] = []
    for message in rendered.messages:
        for part in message.content:
            part_rendered.append(part.model_dump())

    return part_rendered


async def render_message_prompt(
    registry: Registry,
    input: dict[str, Any],
    options: PromptConfig,
    prompt_cache: PromptCache,
    context: dict[str, Any] | None = None,
    history: list[Message] | None = None,
) -> list[Message]:
    """Render a messages template (string or list) into Message objects."""
    if isinstance(options.messages, str):
        if prompt_cache.messages is None:
            prompt_cache.messages = await registry.dotprompt.compile(options.messages)

        if options.metadata:
            context = {**(context or {}), 'state': options.metadata.get('state')}

        # Convert history to dict format for template
        messages_ = None
        if history:
            messages_ = [e.model_dump() for e in history]

        # Flatten input and context for template resolution
        flattened_data = {**(context or {}), **(input or {})}
        rendered = await prompt_cache.messages(
            data=DataArgument[dict[str, Any]](
                input=flattened_data,
                context=context,
                messages=messages_,  # type: ignore[arg-type]
            ),
            options=PromptMetadata(
                input=PromptInputConfig(
                    schema=to_json_schema(options.input_schema) if options.input_schema else None,
                )
            ),
        )
        return [Message.model_validate(e.model_dump()) for e in rendered.messages]

    elif isinstance(options.messages, list):
        return [m if isinstance(m, Message) else Message.model_validate(m) for m in options.messages]

    raise TypeError(f'Unsupported type for messages: {type(options.messages)}')


async def render_user_prompt(
    registry: Registry,
    input: dict[str, Any],
    options: PromptConfig,
    prompt_cache: PromptCache,
    context: dict[str, Any] | None = None,
) -> Message:
    """Render the user prompt."""
    msg, prompt_cache.user_prompt = await _render_template(
        registry,
        Role.USER,
        options.prompt,
        input,
        options.input_schema,
        options.metadata,
        prompt_cache.user_prompt,
        context,
    )
    return msg


async def render_docs(
    input: dict[str, Any],
    options: PromptConfig,
    context: dict[str, Any] | None = None,
) -> list[Document] | None:
    """Return the docs from options (placeholder for future doc rendering)."""
    return options.docs


async def render_prompt_config_for_executable_call(
    executable_prompt: ExecutablePrompt[Any, Any],
    registry: Registry,
    prompt_config: PromptConfig,
    template_input: Any,  # noqa: ANN401
    opts: PromptGenerateOptions,
) -> PromptConfig:
    """Expand dotprompt with the call's input into one merged :class:`PromptConfig`.

    Sets final ``messages``, merged ``docs``, optional ``resume``, and clears template source fields
    before :func:`to_generate_action_options`.
    """
    ri = coerce_prompt_template_input(template_input)
    render_context = opts.get('context')
    message_history = opts.get('messages')
    cache = executable_prompt._cache_prompt
    extra_docs = opts.get('docs')

    resolved_msgs: list[Message] = []
    if prompt_config.system:
        result = await render_system_prompt(registry, ri, prompt_config, cache, render_context)
        resolved_msgs.append(result)
    if prompt_config.messages:
        resolved_msgs.extend(
            await render_message_prompt(registry, ri, prompt_config, cache, render_context, history=message_history)
        )
    elif message_history:
        resolved_msgs.extend(message_history)
    if prompt_config.prompt:
        result = await render_user_prompt(registry, ri, prompt_config, cache, render_context)
        resolved_msgs.append(result)

    merged_docs = await render_docs(ri, prompt_config, render_context)
    if extra_docs:
        merged_docs = [*merged_docs, *extra_docs] if merged_docs else list(extra_docs)

    resume = resume_from_prompt_call_opts(opts)
    return PromptConfig.model_validate({
        **prompt_config.model_dump(),
        'system': None,
        'prompt': None,
        'messages': resolved_msgs,
        'docs': merged_docs,
        'resume': resume,
    })


async def executable_prompt_call_to_generate_options(
    executable_prompt: ExecutablePrompt[Any, Any],
    registry: Registry,
    prompt_config: PromptConfig,
    template_input: Any,  # noqa: ANN401
    opts: PromptGenerateOptions,
) -> GenerateActionOptions:
    """Expand executable prompt templates, then build :class:`GenerateActionOptions`."""
    merged = await render_prompt_config_for_executable_call(
        executable_prompt, registry, prompt_config, template_input, opts
    )
    return await to_generate_action_options(registry, merged)


def registry_definition_key(name: str, variant: str | None = None, ns: str | None = None) -> str:
    """Generate a registry definition key for a prompt.

    Format: "ns/name.variant" where ns and variant are optional.

    Args:
        name: The prompt name.
        variant: Optional variant name.
        ns: Optional namespace.

    Returns:
        Registry key string.
    """
    parts = []
    if ns:
        parts.append(ns)
    parts.append(name)
    if variant:
        parts[-1] = f'{parts[-1]}.{variant}'
    return '/'.join(parts)


def registry_lookup_key(name: str, variant: str | None = None, ns: str | None = None) -> str:
    """Generate a registry lookup key for a prompt.

    Args:
        name: The prompt name.
        variant: Optional variant name.
        ns: Optional namespace.

    Returns:
        Registry lookup key string.
    """
    return f'/prompt/{registry_definition_key(name, variant, ns)}'


def define_partial(registry: Registry, name: str, source: str) -> None:
    """Define a partial template in the registry.

    Partials are reusable template fragments that can be included in other prompts.
    Files starting with `_` are treated as partials.

    Args:
        registry: The registry to register the partial in.
        name: The name of the partial.
        source: The template source code.
    """
    _ = registry.dotprompt.define_partial(name, source)
    logger.debug(f'Registered Dotprompt partial "{name}"')


def define_helper(registry: Registry, name: str, fn: Callable[..., Any]) -> None:
    """Define a Handlebars helper function in the registry.

    Args:
        registry: The registry to register the helper in.
        name: The name of the helper function.
        fn: The helper function to register.
    """
    _ = registry.dotprompt.define_helper(name, fn)
    logger.debug(f'Registered Dotprompt helper "{name}"')


def define_schema(registry: Registry, name: str, schema: type[BaseModel]) -> None:
    """Register a Pydantic schema for use in prompts.

    Schemas registered with this function can be referenced by name in
    .prompt files using the `output.schema` field.

    Args:
        registry: The registry to register the schema in.
        name: The name of the schema.
        schema: The Pydantic model class to register.

    Example:
        ```python
        from genkit._ai._prompt import define_schema

        define_schema(registry, 'Recipe', Recipe)
        ```

        Then in a .prompt file:
        ```yaml
        output:
          schema: Recipe
        ```
    """
    json_schema = to_json_schema(schema)
    registry.register_schema(name, json_schema, schema_type=schema)
    logger.debug(f'Registered schema "{name}"')


def _transform_prompt_metadata(
    raw_metadata: Any,  # noqa: ANN401
    variant: str | None,
    template: str,
    registry_key: str,
) -> dict[str, Any]:
    """Transform dotprompt metadata into the format ExecutablePrompt expects."""
    # Convert Pydantic model to dict if needed
    if hasattr(raw_metadata, 'model_dump'):
        md = raw_metadata.model_dump(by_alias=True)
    elif hasattr(raw_metadata, 'dict'):
        md = raw_metadata.dict(by_alias=True)  # pyright: ignore[reportDeprecated]
    else:
        md = cast(dict[str, Any], raw_metadata)

    # Preserve raw for accessing maxTurns, toolChoice, etc.
    if hasattr(raw_metadata, 'raw'):
        md['raw'] = raw_metadata.raw

    if variant:
        md['variant'] = variant

    # Drop description when it is explicitly null so metadata stays minimal for wire/clients.
    output = md.get('output')
    if output and isinstance(output, dict):
        schema = output.get('schema')
        if schema and isinstance(schema, dict) and schema.get('description') is None:
            schema.pop('description', None)

    input_cfg = md.get('input')
    if input_cfg and isinstance(input_cfg, dict):
        schema = input_cfg.get('schema')
        if schema and isinstance(schema, dict) and schema.get('description') is None:
            schema.pop('description', None)

    # Build metadata structure
    metadata = {
        **md,
        **(md.get('metadata', {}) if isinstance(md.get('metadata'), dict) else {}),
        'type': 'prompt',
        'prompt': {**md, 'template': template},
    }

    raw = md.get('raw')
    if raw and isinstance(raw, dict) and raw.get('metadata'):
        metadata['metadata'] = {**raw['metadata']}

    return {
        'name': registry_key,
        'model': md.get('model'),
        'config': md.get('config'),
        'tools': md.get('tools'),
        'description': md.get('description'),
        'output': {
            'jsonSchema': output.get('schema') if isinstance(output, dict) else None,
            'format': output.get('format') if isinstance(output, dict) else None,
        },
        'input': {
            'default': input_cfg.get('default') if isinstance(input_cfg, dict) else None,
            'jsonSchema': input_cfg.get('schema') if isinstance(input_cfg, dict) else None,
        },
        'metadata': metadata,
        'maxTurns': raw.get('maxTurns') if isinstance(raw, dict) else None,
        'toolChoice': raw.get('toolChoice') if isinstance(raw, dict) else None,
        'returnToolRequests': raw.get('returnToolRequests') if isinstance(raw, dict) else None,
        'messages': template,
    }


def load_prompt(registry: Registry, path: Path, filename: str, prefix: str = '', ns: str = '') -> None:
    """Load a .prompt file and register it as a lazy-loaded prompt."""
    if not filename.endswith('.prompt'):
        raise ValueError(f"Invalid prompt filename: {filename}. Must end with '.prompt'")

    base_name = filename.removesuffix('.prompt')
    name = f'{prefix}{base_name}' if prefix else base_name
    variant: str | None = None

    if '.' in name:
        parts = name.split('.')
        name = parts[0]
        variant = parts[1]

    file_path = path / (prefix.rstrip('/') + '/' + filename if prefix else filename)

    with Path(file_path).open(encoding='utf-8') as f:
        source = f.read()

    parsed_prompt = registry.dotprompt.parse(source)
    registry_key = registry_definition_key(name, variant, ns)

    # Memoized prompt instance
    _cached_prompt: ExecutablePrompt[Any, Any] | None = None

    async def create_prompt_from_file() -> ExecutablePrompt[Any, Any]:
        nonlocal _cached_prompt
        if _cached_prompt is not None:
            return _cached_prompt

        raw_metadata = await registry.dotprompt.render_metadata(parsed_prompt)
        metadata = _transform_prompt_metadata(raw_metadata, variant, parsed_prompt.template, registry_key)

        executable_prompt = ExecutablePrompt(
            registry=registry,
            variant=metadata.get('variant'),
            model=metadata.get('model'),
            config=metadata.get('config'),
            description=metadata.get('description'),
            input_schema=metadata.get('input', {}).get('jsonSchema'),
            output_schema=metadata.get('output', {}).get('jsonSchema'),
            output_constrained=True if metadata.get('output', {}).get('jsonSchema') else None,
            output_format=metadata.get('output', {}).get('format'),
            messages=metadata.get('messages'),
            max_turns=metadata.get('maxTurns'),
            tool_choice=metadata.get('toolChoice'),
            return_tool_requests=metadata.get('returnToolRequests'),
            metadata=metadata.get('metadata'),
            tools=metadata.get('tools'),
            name=name,
            ns=ns,
        )

        # Wire up action references
        definition_key = registry_definition_key(name, variant, ns)
        prompt_action = await registry.resolve_action_by_key(create_action_key(ActionKind.PROMPT, definition_key))
        exec_prompt_action = await registry.resolve_action_by_key(
            create_action_key(ActionKind.EXECUTABLE_PROMPT, definition_key)
        )

        if prompt_action and prompt_action.kind == ActionKind.PROMPT:
            executable_prompt._prompt_action = prompt_action  # pyright: ignore[reportPrivateUsage]
            setattr(prompt_action, '_executable_prompt', weakref.ref(executable_prompt))  # noqa: B010

        # Update schemas on actions for Dev UI
        for action in [prompt_action, exec_prompt_action]:
            if action:
                if metadata.get('input', {}).get('jsonSchema'):
                    action.input_schema = metadata['input']['jsonSchema']
                if metadata.get('output', {}).get('jsonSchema'):
                    action.output_schema = metadata['output']['jsonSchema']

        _cached_prompt = executable_prompt
        return executable_prompt

    action_metadata: dict[str, object] = {
        'type': 'prompt',
        'lazy': True,
        'source': 'file',
        'prompt': {'name': name, 'variant': variant or ''},
    }

    async def prompt_action_fn(input: Any = None) -> ModelRequest:  # noqa: ANN401
        prompt = await create_prompt_from_file()
        call_opts: PromptGenerateOptions = {}
        prompt_config = prompt._prompt_config_for_call(call_opts)
        registry = await registry_with_inline_tools(prompt._registry, prompt_config.tools)
        gen_options = await executable_prompt_call_to_generate_options(
            prompt, registry, prompt_config, input, call_opts
        )
        return await to_generate_request(registry, gen_options)

    async def executable_prompt_action_fn(input: Any = None) -> GenerateActionOptions:  # noqa: ANN401
        prompt = await create_prompt_from_file()
        call_opts: PromptGenerateOptions = {}
        prompt_config = prompt._prompt_config_for_call(call_opts)
        registry = await registry_with_inline_tools(prompt._registry, prompt_config.tools)
        return await executable_prompt_call_to_generate_options(prompt, registry, prompt_config, input, call_opts)

    action_name = registry_definition_key(name, variant, ns)
    prompt_action = registry.register_action(
        kind=ActionKind.PROMPT, name=action_name, fn=prompt_action_fn, metadata=action_metadata
    )
    executable_prompt_action = registry.register_action(
        kind=ActionKind.EXECUTABLE_PROMPT, name=action_name, fn=executable_prompt_action_fn, metadata=action_metadata
    )

    setattr(prompt_action, '_async_factory', create_prompt_from_file)  # noqa: B010
    setattr(executable_prompt_action, '_async_factory', create_prompt_from_file)  # noqa: B010

    logger.debug(f'Registered prompt "{registry_key}" from "{file_path}"')


def load_prompt_folder_recursively(registry: Registry, dir_path: Path, ns: str, sub_dir: str = '') -> None:
    """Recursively load all prompt files from a directory.

    Args:
        registry: The registry to register prompts in.
        dir_path: Base path to the prompts directory.
        ns: Namespace for prompts.
        sub_dir: Current subdirectory being processed (for recursion).
    """
    full_path = dir_path / sub_dir if sub_dir else dir_path

    if not full_path.exists() or not full_path.is_dir():
        return

    # Iterate through directory entries
    try:
        for entry in os.scandir(full_path):
            if entry.is_file() and entry.name.endswith('.prompt'):
                if entry.name.startswith('_'):
                    # This is a partial
                    partial_name = entry.name[1:-7]  # Remove "_" prefix and ".prompt" suffix
                    with Path(entry.path).open(encoding='utf-8') as f:
                        source = f.read()

                    # Strip frontmatter if present
                    if source.startswith('---'):
                        end_frontmatter = source.find('---', 3)
                        if end_frontmatter != -1:
                            source = source[end_frontmatter + 3 :].strip()

                    define_partial(registry, partial_name, source)
                    logger.debug(f'Registered Dotprompt partial "{partial_name}" from "{entry.path}"')
                else:
                    # This is a regular prompt
                    prefix_with_slash = f'{sub_dir}/' if sub_dir else ''
                    load_prompt(registry, dir_path, entry.name, prefix_with_slash, ns)
            elif entry.is_dir():
                # Recursively process subdirectories
                new_sub_dir = os.path.join(sub_dir, entry.name) if sub_dir else entry.name
                load_prompt_folder_recursively(registry, dir_path, ns, new_sub_dir)
    except PermissionError:
        logger.warning(f'Permission denied accessing directory: {full_path}')
    except Exception as e:
        logger.exception(f'Error loading prompts from {full_path}', exc_info=e)


def load_prompt_folder(registry: Registry, dir_path: str | Path = './prompts', ns: str = '') -> None:
    """Load all prompt files from a directory.

    This is the main entry point for loading prompts from a directory.
    It recursively processes all `.prompt` files and registers them.

    Args:
        registry: The registry to register prompts in.
        dir_path: Path to the prompts directory. Defaults to './prompts'.
        ns: Namespace for prompts. Defaults to 'dotprompt'.
    """
    path = Path(dir_path).resolve()

    if not path.exists():
        logger.warning(f'Prompt directory does not exist: {path}')
        return

    if not path.is_dir():
        logger.warning(f'Prompt path is not a directory: {path}')
        return

    load_prompt_folder_recursively(registry, path, ns, '')
    logger.info(f'Loaded prompts from directory: {path}')


async def lookup_prompt(registry: Registry, name: str, variant: str | None = None) -> ExecutablePrompt[Any, Any]:
    """Look up a prompt by name from the registry."""
    # Try without namespace first (for programmatic prompts)
    # Use create_action_key to build the full key: "/prompt/<definition_key>"
    definition_key = registry_definition_key(name, variant, None)
    lookup_key = create_action_key(ActionKind.PROMPT, definition_key)
    action = await registry.resolve_action_by_key(lookup_key)

    # If not found and no namespace was specified, try with default 'dotprompt' namespace
    # (for file-based prompts)
    if not action:
        definition_key = registry_definition_key(name, variant, 'dotprompt')
        lookup_key = create_action_key(ActionKind.PROMPT, definition_key)
        action = await registry.resolve_action_by_key(lookup_key)

    if action:
        # First check if we've stored the ExecutablePrompt directly
        prompt_ref = getattr(action, '_executable_prompt', None)
        if prompt_ref is not None:
            if isinstance(prompt_ref, weakref.ReferenceType):
                resolved = prompt_ref()
                if resolved is not None:
                    return resolved
            if isinstance(prompt_ref, ExecutablePrompt):
                return prompt_ref
        # Otherwise, create it from the factory (lazy loading)
        async_factory = getattr(action, '_async_factory', None)
        if callable(async_factory):
            # Cast to async callable - getattr returns object but we've verified it's callable
            async_factory_fn = cast(Callable[[], Awaitable[ExecutablePrompt]], async_factory)
            executable_prompt = await async_factory_fn()
            if getattr(action, '_executable_prompt', None) is None:
                setattr(action, '_executable_prompt', executable_prompt)  # noqa: B010
            return executable_prompt
        # This shouldn't happen if prompts are loaded correctly
        raise GenkitError(
            status='INTERNAL',
            message=f'Prompt action found but no ExecutablePrompt available for {name}',
        )

    variant_str = f' (variant {variant})' if variant else ''
    raise GenkitError(
        status='NOT_FOUND',
        message=f'Prompt {name}{variant_str} not found',
    )


async def prompt(
    registry: Registry,
    name: str,
    variant: str | None = None,
) -> ExecutablePrompt[Any, Any]:
    """Look up a prompt by name and optional variant."""
    return await lookup_prompt(registry, name, variant)


# Renamed — use ModelStreamResponse
