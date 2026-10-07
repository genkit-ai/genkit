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
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Generic, NamedTuple, TypedDict, TypeVar, cast

from dotpromptz.typing import (
    DataArgument,
    PromptFunction,
    PromptInputConfig,
    PromptMetadata,
)
from pydantic import BaseModel, ConfigDict
from typing_extensions import Never, Self

from genkit._ai._generate import (
    generate_action,
    register_middleware,
    register_tools,
    resolve_tools_from_options,
    to_tool_definition,
    tools_to_action_names,
)
from genkit._ai._model import (
    ModelArg,
    ModelRef,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
    assert_correct_config_class,
    check_call_config,
    check_config_dict,
    config_field_names,
    config_schema_at_define,
    normalize_config,
    resolve_call_model,
    resolve_for_generate,
)
from genkit._core._action import (
    Action,
    ActionKind,
    StreamingCallback,
    create_action_key,
    get_current_context,
)
from genkit._core._channel import Channel
from genkit._core._error import GenkitError, RuntimeErrorReason
from genkit._core._logger import get_logger
from genkit._core._middleware import BaseMiddleware, middleware_class_index
from genkit._core._model import (
    Document,
    GenerateActionOptions,
    Message,
    OutputConfig,
    Part,
    ToolChoice,
    resume_options_to_resume,
)
from genkit._core._registry import Registry
from genkit._core._schema import to_json_schema
from genkit._core._tool import Tool
from genkit._core._typing import (
    GenerateActionOutputConfig,
    MiddlewareRef,
    Role,
)

ModelStreamingCallback = StreamingCallback

logger = get_logger(__name__)

# TypeVars for generic input/output typing
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


class PromptSettings(TypedDict, total=False):
    """Prompt settings a call can replace, for that call only.

    The call's value replaces the prompt's; ``[]`` clears a list. Applied in
    ``GenerateCall.with_overrides``.
    """

    tools: Sequence[str | Tool] | None
    tool_choice: ToolChoice | None
    docs: list[Document] | None
    use: Sequence[BaseMiddleware | MiddlewareRef] | None
    max_turns: int | None
    return_tool_requests: bool | None
    resume_respond: Part | list[Part] | None
    resume_restart: Part | list[Part] | None
    resume_metadata: dict[str, Any] | None


class ModelSettings(TypedDict, total=False):
    """Which model a call runs on, and with what config.

    ``model`` replaces the prompt's; ``config`` merges over the prompt's per
    key. Applied in ``Prompt._resolve_model``.
    """

    model: ModelArg | None
    config: Mapping[str, Any] | BaseModel | None


class PromptGenerateOptions(PromptSettings, ModelSettings, total=False):
    """Everything a caller can pass to one prompt call (``__call__``, ``stream``, ``render``).

    ``None`` or omitted always means "not set". There's no output key on
    purpose: the template is written for its schema, so a prompt always
    returns the type it was defined with.
    """

    # Data for this call only. These don't change the prompt; render_call
    # passes them into the template when it renders.
    #
    # `messages` is this call's chat history. It doesn't replace the prompt's
    # own `messages`, which is a template; render_call puts the history where
    # the template says ({{history}}, or after the system message).
    messages: list[Message] | None
    # `context` is runtime state (auth, request metadata). Templates read it
    # as {{@auth}}. Omit it and render/call/stream use the enclosing flow's
    # context, the same one tools and middleware already see.
    context: dict[str, Any] | None


class ModelStreamResponse(Generic[OutputT]):
    """Response from streaming prompt execution with stream and response properties."""

    def __init__(
        self,
        channel: Channel[ModelResponseChunk[OutputT], ModelResponse[OutputT]],
        response_future: asyncio.Future[ModelResponse[OutputT]],
    ) -> None:
        """Initialize with streaming channel and response future."""
        self._channel: Channel[ModelResponseChunk[OutputT], ModelResponse[OutputT]] = channel
        self._response_future: asyncio.Future[ModelResponse[OutputT]] = response_future

    @property
    def stream(self) -> AsyncIterable[ModelResponseChunk[OutputT]]:
        """Async iterable of response chunks.

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
        """Awaitable for the complete response.

        Returns:
            An awaitable that resolves to a ModelResponse containing:
            - text: The complete generated text
            - output: The typed output, or None when the reply isn't that shape
            - messages: The full message history
            - usage: Token usage statistics
            - finish_reason: Why generation stopped (e.g., 'stop', 'length')
            - Any tool calls or interrupts from the response

        If the model fails partway through, this still resolves rather than
        raising: ``finish_reason`` is FAILED, ``error`` is set, ``text`` is
        empty, ``message`` is None, and ``messages`` ends at the last complete
        turn. The chunks already streamed are the record of what was shown.
        """
        return self._response_future

    # The natural Python expectation is `async for chunk in ai.generate_stream(...)`.
    # Delegating to the underlying channel lets that work without forcing the
    # caller to remember the extra `.stream` hop, while `.stream` and `.response`
    # remain available for cases where you want both halves explicitly.
    def __aiter__(self) -> AsyncIterator[ModelResponseChunk[OutputT]]:
        return self._channel.__aiter__()


@dataclass
class PromptCache:
    """Model for a prompt cache."""

    user_prompt: PromptFunction[Any] | None = None
    system: PromptFunction[Any] | None = None
    messages: PromptFunction[Any] | None = None


class GenerateCall(BaseModel):
    """User-shaped args for one generate. Prompts fill this; ``ai.generate`` builds it."""

    # arbitrary_types_allowed: Tool, BaseMiddleware and ModelRef are plain classes, so
    # Pydantic only isinstance-checks them.
    # extra='forbid': a misspelled field raises instead of being dropped.
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    # Adding a field: if it changes what the prompt says or returns, put it
    # under "Fixed at definition" below. Otherwise put it under "A call can
    # change these", add it to PromptSettings and to the __call__/stream/render
    # keywords; with_overrides picks it up. prompt_test fails if a field is in
    # neither group or a keyword isn't forwarded.

    # Fixed at definition: what the prompt says and what it returns.

    # PromptTemplate: the words, the input they're written against, and the
    # frontmatter `metadata` they can read ({{@state}}).
    # (On ai.generate, `messages` is the conversation instead of a template.)
    system: str | list[Part] | None = None
    prompt: str | list[Part] | None = None
    messages: str | list[Message] | None = None
    input_schema: type | dict[str, Any] | str | None = None
    metadata: dict[str, Any] | None = None

    # OutputSettings: the type that comes back. Fixed so Prompt[In, Out] holds.
    output_schema: type | dict[str, Any] | str | None = None
    output_format: str | None = None
    output_content_type: str | None = None
    output_instructions: bool | str | None = None
    output_constrained: bool | None = None

    # A call can change these (see PromptGenerateOptions).

    # PromptSettings
    tools: Sequence[str | Tool] | None = None
    tool_choice: ToolChoice | None = None
    docs: list[Document] | None = None
    use: Sequence[BaseMiddleware | MiddlewareRef] | None = None
    max_turns: int | None = None
    return_tool_requests: bool | None = None
    resume_respond: Part | list[Part] | None = None
    resume_restart: Part | list[Part] | None = None
    resume_metadata: dict[str, Any] | None = None

    # ModelSettings
    model: str | ModelRef[BaseModel] | None = None
    config: Mapping[str, Any] | BaseModel | None = None

    def with_overrides(self, opts: PromptSettings) -> Self:
        """Return a copy where every ``PromptSettings`` key the call passed replaces this one's.

        None or omitted keeps this value; [] clears it. ``opts`` may be a full
        ``PromptGenerateOptions``; keys outside ``PromptSettings`` are ignored here.
        """
        set_by_call = {k: v for k, v in opts.items() if k in PromptSettings.__optional_keys__ and v is not None}
        return self.model_copy(update=set_by_call)


class Prompt(Generic[InputT, OutputT]):
    """A callable prompt with typed input/output that generates AI responses."""

    def __init__(
        self,
        registry: Registry,
        variant: str | None = None,
        model: ModelArg | None = None,
        config: Mapping[str, Any] | BaseModel | None = None,
        description: str | None = None,
        input_schema: type | dict[str, Any] | str | None = None,
        system: str | list[Part] | None = None,
        prompt: str | list[Part] | None = None,
        messages: str | list[Message] | None = None,
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: bool | str | None = None,
        output_schema: type | dict[str, Any] | str | None = None,
        output_constrained: bool | None = None,
        max_turns: int | None = None,
        return_tool_requests: bool | None = None,
        metadata: dict[str, Any] | None = None,
        tools: Sequence[str | Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        use: Sequence[BaseMiddleware | MiddlewareRef] | None = None,
        docs: list[Document] | None = None,
        name: str | None = None,
        ns: str | None = None,
    ) -> None:
        """Initialize prompt with configuration, templates, and schema options."""
        self._registry = registry
        # Identity: how the prompt is registered and looked up. Not part of _def,
        # which only holds what goes into a generate.
        self._name = name
        self._ns = ns
        self._variant = variant
        self._description = description
        # The whole definition as one value. Per-call overrides layer over it
        # in prepare_prompt via with_overrides; nothing mutates it after define
        # except _ensure_resolved swapping in a lazily loaded definition.
        self._def = GenerateCall(
            model=model,
            config=config,
            input_schema=input_schema,
            system=system,
            prompt=prompt,
            messages=messages,
            output_format=output_format,
            output_content_type=output_content_type,
            output_instructions=output_instructions,
            output_schema=output_schema,
            output_constrained=output_constrained,
            max_turns=max_turns,
            return_tool_requests=return_tool_requests,
            metadata=metadata,
            tools=tools,
            tool_choice=tool_choice,
            use=use,
            docs=docs,
        )
        # Compiled system/messages/prompt templates, filled on first render and reused.
        self._compiled_templates: PromptCache = PromptCache()
        self._prompt_action: Action | None = None
        define_name, define_schema = config_schema_at_define(model=model, registry=registry)
        # Hop identity is what we knew at define time, not today's defaultModel.
        # Not a GenerateCall field, so _ensure_resolved copies it explicitly.
        self._defined_model_name = define_name
        assert_correct_config_class(config=config, schema=define_schema, model=define_name)

    @property
    def ref(self) -> dict[str, Any]:
        """Reference object with prompt name and metadata."""
        return {
            'name': registry_definition_key(self._name, self._variant, self._ns) if self._name else None,
            'metadata': self._def.metadata,
        }

    async def _ensure_resolved(self) -> None:
        if self._prompt_action or not self._name:
            return

        resolved = await lookup_prompt(self._registry, self._name, self._variant)
        # Keep a Pydantic output type the caller passed: the .prompt file only
        # carries a dict schema, and the type is what gives typed output at runtime.
        keep: dict[str, Any] = {}
        schema = self._def.output_schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            keep['output_schema'] = schema
        self._def = resolved._def.model_copy(update=keep)
        self._defined_model_name = resolved._defined_model_name
        self._prompt_action = resolved._prompt_action

    async def _resolve_model(self, call: GenerateCall, opts: ModelSettings) -> GenerateCall:
        """Resolve the model (the call's or the prompt's) and merge the call's config over the prompt's.

        Returns ``call`` with ``model`` set to the resolved name and ``config``
        to the merged, resolved config.
        """
        override_config = opts.get('config')
        merged_config: Mapping[str, Any] | BaseModel | None
        if override_config is not None:
            # exclude_unset semantics via normalize_config: untouched fields are
            # absent (cannot clobber defaults); an explicitly-set None survives
            # the merge and clears the lower-precedence value downstream.
            base = normalize_config(config=self._def.config)
            override = normalize_config(config=override_config)
            merged_config = {**base, **override} if base or override else None
        else:
            merged_config = self._def.config

        override_model = opts.get('model')
        resolved = await resolve_for_generate(
            model=override_model if override_model is not None else self._def.model,
            config=merged_config,
            registry=self._registry,
        )
        check_call_config(
            config=override_config,
            schema=resolved.config_schema,
            model=resolved.name,
        )
        # Re-check the stored typed config unless this call hops models.
        # A None override clears that default, so the prompt's copy of the
        # key is not checked against the model this call hits.
        if self._defined_model_name is None or self._defined_model_name == resolved.name:
            assert_correct_config_class(
                config=self._def.config,
                schema=resolved.config_schema,
                model=resolved.name,
            )
        check_config_dict(
            config=prompt_config_after_clears(
                stored=self._def.config,
                override=override_config,
                schema=resolved.config_schema,
            ),
            schema=resolved.config_schema,
            model=resolved.name,
        )
        return call.model_copy(update={'model': resolved.name, 'config': resolved.config})

    async def __call__(
        self,
        input: InputT | dict[str, Any] | None = None,
        *,
        model: ModelArg | None = None,
        config: Mapping[str, Any] | BaseModel | None = None,
        messages: list[Message] | None = None,
        tools: Sequence[str | Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        docs: list[Document] | None = None,
        use: Sequence[BaseMiddleware | MiddlewareRef] | None = None,
        max_turns: int | None = None,
        context: dict[str, Any] | None = None,
        return_tool_requests: bool | None = None,
        resume_respond: Part | list[Part] | None = None,
        resume_restart: Part | list[Part] | None = None,
        resume_metadata: dict[str, Any] | None = None,
        on_chunk: ModelStreamingCallback | None = None,
    ) -> ModelResponse[OutputT]:
        """Render the prompt with ``input`` as template variables, run it, and return the response.

        Keywords take the same names as ``ai.generate``. An omitted (or ``None``)
        keyword keeps the prompt's value; ``tools=[]``, ``use=[]`` and ``docs=[]``
        clear the prompt's list for this call. ``config`` merges per key.
        """
        opts = PromptGenerateOptions(
            model=model,
            config=config,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            docs=docs,
            use=use,
            max_turns=max_turns,
            context=context,
            return_tool_requests=return_tool_requests,
            resume_respond=resume_respond,
            resume_restart=resume_restart,
            resume_metadata=resume_metadata,
        )
        prepared = await prepare_prompt(prompt=self, input=input, opts=opts)
        result = await generate_action(
            prepared.registry,
            prepared.options,
            on_chunk=on_chunk,
            # Same context the template already rendered, so {{@auth}} and tools agree.
            context=prepared.context,
        )
        return cast(ModelResponse[OutputT], result)

    def stream(
        self,
        input: InputT | dict[str, Any] | None = None,
        *,
        model: ModelArg | None = None,
        config: Mapping[str, Any] | BaseModel | None = None,
        messages: list[Message] | None = None,
        tools: Sequence[str | Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        docs: list[Document] | None = None,
        use: Sequence[BaseMiddleware | MiddlewareRef] | None = None,
        max_turns: int | None = None,
        context: dict[str, Any] | None = None,
        return_tool_requests: bool | None = None,
        resume_respond: Part | list[Part] | None = None,
        resume_restart: Part | list[Part] | None = None,
        resume_metadata: dict[str, Any] | None = None,
    ) -> ModelStreamResponse[OutputT]:
        """Stream the prompt execution. Same keywords as ``__call__``, minus ``on_chunk``.

        Iterate the returned stream for chunks; there's no callback here so
        chunks only arrive one way.
        """
        channel: Channel[ModelResponseChunk[OutputT], ModelResponse[OutputT]] = Channel()

        # Same run path as __call__; only the chunk sink differs.
        response_future: asyncio.Future[ModelResponse[OutputT]] = asyncio.create_task(
            self(
                input,
                model=model,
                config=config,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                docs=docs,
                use=use,
                max_turns=max_turns,
                context=context,
                return_tool_requests=return_tool_requests,
                resume_respond=resume_respond,
                resume_restart=resume_restart,
                resume_metadata=resume_metadata,
                on_chunk=lambda c: channel.send(cast('ModelResponseChunk[OutputT]', c)),
            )
        )
        channel.set_close_future(response_future)

        return ModelStreamResponse[OutputT](channel=channel, response_future=response_future)

    async def render(
        self,
        input: InputT | dict[str, Any] | None = None,
        *,
        model: ModelArg | None = None,
        config: Mapping[str, Any] | BaseModel | None = None,
        messages: list[Message] | None = None,
        tools: Sequence[str | Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        docs: list[Document] | None = None,
        use: Sequence[BaseMiddleware | MiddlewareRef] | None = None,
        max_turns: int | None = None,
        context: dict[str, Any] | None = None,
        return_tool_requests: bool | None = None,
        resume_respond: Part | list[Part] | None = None,
        resume_restart: Part | list[Part] | None = None,
        resume_metadata: dict[str, Any] | None = None,
    ) -> GenerateActionOptions:
        """Render the prompt without running it. Same keywords as ``__call__``, minus ``on_chunk``."""
        opts = PromptGenerateOptions(
            model=model,
            config=config,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            docs=docs,
            use=use,
            max_turns=max_turns,
            context=context,
            return_tool_requests=return_tool_requests,
            resume_respond=resume_respond,
            resume_restart=resume_restart,
            resume_metadata=resume_metadata,
        )
        return (await prepare_prompt(prompt=self, input=input, opts=opts)).options


class PreparedPrompt(NamedTuple):
    registry: Registry
    options: GenerateActionOptions
    context: dict[str, Any] | None


async def prepare_prompt(
    *,
    prompt: Prompt[Any, Any],
    input: Any | None = None,  # noqa: ANN401
    opts: PromptGenerateOptions | None = None,
) -> PreparedPrompt:
    """Build the model request for one call of this prompt.

    Looks up the enclosing flow's context once when the call omits
    ``context=``, then that same value is what templates and the run see.
    """
    await prompt._ensure_resolved()
    call_opts: PromptGenerateOptions = opts if opts is not None else {}

    context = call_opts.get('context')
    if context is None:
        context = get_current_context()

    # The call's tools/docs/use/etc. replace the prompt's for this call.
    call = prompt._def.with_overrides(call_opts)
    # Tools and middleware passed inline (e.g. use=[Foo()]) are registered on a
    # child registry so they exist for this call only.
    registry = prompt._registry.new_child()
    await register_tools(registry, call.tools)
    if call.use is not None:
        call = call.model_copy(update={'use': register_middleware(registry, call.use)})

    call = await prompt._resolve_model(call, call_opts)

    call = await render_call(
        prompt=prompt,
        registry=registry,
        call=call,
        input=input,
        context=context,
        history=call_opts.get('messages'),
    )

    options = await to_generate_options(registry=registry, call=call)
    return PreparedPrompt(registry=registry, options=options, context=context)


def prompt_config_after_clears(
    *,
    stored: Mapping[str, Any] | BaseModel | None,
    override: object,
    schema: type[BaseModel] | None,
) -> dict[str, Any]:
    """The prompt's config minus keys this call set to None.

    None means "clear the default". The prompt still names the key, but
    this call does not send it, so it must not fail the model's check.
    """
    stored_bag = normalize_config(config=stored)
    if override is None:
        return {key: value for key, value in stored_bag.items() if value is not None}
    override_bag = normalize_config(config=override)
    names = config_field_names(schema) if schema is not None else {}
    cleared = {names.get(key, key) for key, value in override_bag.items() if value is None}
    return {key: value for key, value in stored_bag.items() if value is not None and names.get(key, key) not in cleared}


def _register_prompt_action_pair(
    registry: Registry,
    action_name: str,
    ep_factory: Callable[[], Awaitable[Prompt[Any, Any]]],
    metadata: dict[str, object],
    description: str | None = None,
) -> tuple[Action[Any, Any, Never], Action[Any, Any, Never]]:
    """Register the ``(PROMPT, EXECUTABLE_PROMPT)`` action pair for a prompt.

    Args:
        registry: Registry to register the actions on.
        action_name: Wire name (already passed through ``registry_definition_key``).
        ep_factory: Returns the ``Prompt``. Either a closure over an
            already-built instance, or a lazy factory that loads from disk.
        metadata: Wire metadata to attach to both actions (typically differs
            only in ``source``/``lazy`` between the two registration paths).
        description: Shown for both actions in the Dev UI.

    Returns:
        ``(prompt_action, executable_prompt_action)`` so callers can attach
        extra attrs (e.g. ``_async_factory`` for hot-reload on file prompts).
    """

    async def prompt_action_fn(input: Any = None) -> ModelRequest:  # noqa: ANN401
        ep = await ep_factory()
        prepared = await prepare_prompt(prompt=ep, input=input)
        return await to_prompt_model_request(registry=prepared.registry, options=prepared.options)

    async def executable_prompt_action_fn(input: Any = None) -> GenerateActionOptions:  # noqa: ANN401
        ep = await ep_factory()
        return await ep.render(input)

    prompt_action = registry.register_action(
        kind=ActionKind.PROMPT,
        name=action_name,
        fn=prompt_action_fn,
        description=description,
        metadata=metadata,
    )
    executable_prompt_action = registry.register_action(
        kind=ActionKind.EXECUTABLE_PROMPT,
        name=action_name,
        fn=executable_prompt_action_fn,
        description=description,
        metadata=metadata,
    )
    return prompt_action, executable_prompt_action


def register_prompt_actions(
    registry: Registry,
    executable_prompt: Prompt[Any, Any],
    name: str,
    variant: str | None = None,
) -> None:
    """Register PROMPT and EXECUTABLE_PROMPT actions for a prompt.

    This links the executable prompt to actions in the registry, enabling
    lookup and DevUI integration.
    """
    prompt_block: dict[str, Any] = {'name': name, 'variant': variant or ''}
    use_metadata = _use_to_wire_metadata(registry, executable_prompt._def.use)  # pyright: ignore[reportPrivateUsage]
    if use_metadata is not None:
        prompt_block['use'] = use_metadata
    metadata: dict[str, object] = {
        'type': 'prompt',
        'source': 'programmatic',
        'prompt': prompt_block,
    }

    async def _ep_factory() -> Prompt[Any, Any]:
        # Programmatic prompts hand us the already-built instance; just make
        # sure resolution finished before the action body inspects it.
        await executable_prompt._ensure_resolved()
        return executable_prompt

    action_name = registry_definition_key(name, variant)
    prompt_action, executable_prompt_action = _register_prompt_action_pair(
        registry,
        action_name,
        _ep_factory,
        metadata,
        description=executable_prompt._description,  # pyright: ignore[reportPrivateUsage]
    )

    # Link them
    executable_prompt._prompt_action = prompt_action  # pyright: ignore[reportPrivateUsage]
    setattr(prompt_action, '_executable_prompt', weakref.ref(executable_prompt))  # noqa: B010
    setattr(executable_prompt_action, '_executable_prompt', weakref.ref(executable_prompt))  # noqa: B010

    # Propagate the prompt's input/output schemas onto both actions so the Dev
    # UI Prompt Runner can render a typed form (otherwise the runner has nothing
    # to introspect and the user just sees a free-form textarea). Dotprompts do
    # the equivalent in their lazy factory after rendering frontmatter.
    input_schema = executable_prompt._def.input_schema  # pyright: ignore[reportPrivateUsage]
    if input_schema is not None:
        in_js = to_json_schema(input_schema)
        for action in (prompt_action, executable_prompt_action):
            action.input_schema = in_js
    output_schema = executable_prompt._def.output_schema  # pyright: ignore[reportPrivateUsage]
    if output_schema is not None:
        out_js = to_json_schema(output_schema)
        for action in (prompt_action, executable_prompt_action):
            action.output_schema = out_js


def resolve_output_schema(
    *,
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


async def to_generate_options(
    *,
    registry: Registry,
    call: GenerateCall,
) -> GenerateActionOptions:
    """Fold a ``GenerateCall`` into the ``options`` the engine runs.

    ``call.messages`` must already be the final list. ``system`` / ``prompt``
    and a string ``messages`` belong on the caller that renders or builds them.
    """
    if call.system is not None or call.prompt is not None or isinstance(call.messages, str):
        raise TypeError('render the prompt before building generate options')

    resolved = resolve_call_model(model=call.model, config=call.config, registry=registry)
    model = resolved.name
    default_model = registry.lookup_value('defaultModel', 'defaultModel')
    uses_ref = isinstance(call.model, ModelRef) or isinstance(default_model, ModelRef)
    config = resolved.config if uses_ref else call.config

    resolved_msgs: list[Message] = list(call.messages or [])

    # If is schema is set but format is not explicitly set, default to
    # `json` format.
    output_format = 'json' if call.output_schema and not call.output_format else call.output_format

    output = GenerateActionOutputConfig()
    if output_format:
        output.format = output_format
    if call.output_content_type:
        output.content_type = call.output_content_type
    if call.output_instructions is not None:
        output.instructions = call.output_instructions
    resolve_output_schema(registry=registry, output_schema=call.output_schema, output=output)
    if call.output_constrained is not None:
        output.constrained = call.output_constrained

    resume = resume_options_to_resume(
        resume_respond=call.resume_respond,
        resume_restart=call.resume_restart,
        resume_metadata=call.resume_metadata,
    )

    tools_refs = tools_to_action_names(call.tools)

    return GenerateActionOptions(
        model=model,
        messages=resolved_msgs,  # type: ignore[arg-type]
        config=config,
        tools=tools_refs,
        return_tool_requests=call.return_tool_requests,
        tool_choice=call.tool_choice if call.tool_choice else None,
        output=output,
        max_turns=call.max_turns,
        docs=call.docs,  # type: ignore[arg-type]
        resume=resume,
        use=call.use,  # type: ignore[arg-type]
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


async def to_prompt_model_request(*, registry: Registry, options: GenerateActionOptions) -> ModelRequest:
    """Convert GenerateActionOptions to ModelRequest, resolving tool names."""
    tools = await resolve_tools_from_options(registry, options.tools)
    tool_defs = [to_tool_definition(tool) for tool in tools] if tools else []

    output_config = OutputConfig(
        content_type=options.output.content_type if options.output else None,
        format=options.output.format if options.output else None,
        # pyrefly: ignore[unexpected-keyword] - populate_by_name accepts the field name
        json_schema=options.output.json_schema if options.output else None,
        constrained=options.output.constrained if options.output else None,
    )
    return ModelRequest(
        # Field validators auto-wrap MessageData -> Message and DocumentData -> Document
        messages=options.messages or [],  # type: ignore[arg-type]
        config=options.config if options.config is not None else {},  # type: ignore[arg-type]
        docs=options.docs if options.docs else None,  # type: ignore[arg-type]
        tools=tool_defs,
        tool_choice=options.tool_choice,
        output=output_config,
    )


def parts_from_prompt(
    prompt: str | list[Part] | None,
) -> list[Part]:
    """Convert string/Part/list to list[Part]."""
    if not prompt:
        return []
    if isinstance(prompt, str):
        return [Part.from_text(prompt)]
    elif isinstance(prompt, list):
        return prompt
    elif isinstance(prompt, Part):  # pyright: ignore[reportUnnecessaryIsInstance]
        return [prompt]
    else:
        return []  # pyright: ignore[reportUnreachable] - defensive fallback


async def render_template(
    *,
    registry: Registry,
    role: Role,
    template: str | list[Part] | None,
    input: dict[str, Any],
    input_schema: type | dict[str, Any] | str | None,
    compiled_fn: PromptFunction[Any] | None,
    context: dict[str, Any] | None,
) -> tuple[Message, PromptFunction[Any] | None]:
    """Compile and render a prompt template, returning (message, compiled_fn)."""
    if isinstance(template, str):
        if compiled_fn is None:
            compiled_fn = await registry.dotprompt.compile(template)

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

    return Message(role=role, content=parts_from_prompt(template)), compiled_fn


async def render_system_prompt(
    *,
    registry: Registry,
    input: dict[str, Any],
    call: GenerateCall,
    cache: PromptCache,
    context: dict[str, Any] | None = None,
) -> Message:
    """Render the system prompt."""
    msg, cache.system = await render_template(
        registry=registry,
        role=Role.SYSTEM,
        template=call.system,
        input=input,
        input_schema=call.input_schema,
        compiled_fn=cache.system,
        context=context,
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
    *,
    registry: Registry,
    input: dict[str, Any],
    call: GenerateCall,
    cache: PromptCache,
    context: dict[str, Any] | None = None,
    history: list[Message] | None = None,
) -> list[Message]:
    """Render a messages template (string or list) into Message objects."""
    if isinstance(call.messages, str):
        if cache.messages is None:
            cache.messages = await registry.dotprompt.compile(call.messages)

        # Convert history to dict format for template
        messages_ = None
        if history:
            messages_ = [e.model_dump() for e in history]

        # Flatten input and context for template resolution
        flattened_data = {**(context or {}), **(input or {})}
        rendered = await cache.messages(
            data=DataArgument[dict[str, Any]](
                input=flattened_data,
                context=context,
                messages=messages_,  # type: ignore[arg-type]
            ),
            options=PromptMetadata(
                input=PromptInputConfig(
                    schema=to_json_schema(call.input_schema) if call.input_schema else None,
                )
            ),
        )
        return [Message.model_validate(e.model_dump()) for e in rendered.messages]

    elif isinstance(call.messages, list):
        return [m if isinstance(m, Message) else Message.model_validate(m) for m in call.messages]

    raise TypeError(f'Unsupported type for messages: {type(call.messages)}')


async def render_user_prompt(
    *,
    registry: Registry,
    input: dict[str, Any],
    call: GenerateCall,
    cache: PromptCache,
    context: dict[str, Any] | None = None,
) -> Message:
    """Render the user prompt."""
    msg, cache.user_prompt = await render_template(
        registry=registry,
        role=Role.USER,
        template=call.prompt,
        input=input,
        input_schema=call.input_schema,
        compiled_fn=cache.user_prompt,
        context=context,
    )
    return msg


async def render_call(
    *,
    prompt: Prompt[Any, Any],
    registry: Registry,
    call: GenerateCall,
    input: Any,  # noqa: ANN401
    context: dict[str, Any] | None = None,
    history: list[Message] | None = None,
) -> GenerateCall:
    """Expand dotprompt with the call's input into one merged :class:`GenerateCall`.

    ``context`` is what templates see (``{{@auth}}``, ``{{@state}}``).
    ``history`` is this call's chat history (``messages=`` on the call).
    Sets final ``messages`` and clears template source fields, before
    :func:`to_generate_options`.
    """
    template_input = coerce_prompt_template_input(input)
    render_context = context
    # {{@state}} is written only when metadata has state; a non-empty
    # metadata bag without that key must not wipe the call's context state.
    if call.metadata and 'state' in call.metadata:
        render_context = {**(render_context or {}), 'state': call.metadata['state']}
    cache = prompt._compiled_templates

    resolved_msgs: list[Message] = []
    if call.system:
        result = await render_system_prompt(
            registry=registry, input=template_input, call=call, cache=cache, context=render_context
        )
        resolved_msgs.append(result)
    if call.messages:
        resolved_msgs.extend(
            await render_message_prompt(
                registry=registry,
                input=template_input,
                call=call,
                cache=cache,
                context=render_context,
                history=history,
            )
        )
    elif history:
        resolved_msgs.extend(history)
    if call.prompt:
        result = await render_user_prompt(
            registry=registry, input=template_input, call=call, cache=cache, context=render_context
        )
        resolved_msgs.append(result)

    # Keep the merged config bag as-is. dump/revalidate would rebuild it
    # and drop keys the plugin is about to see.
    return call.model_copy(
        update={
            'system': None,
            'prompt': None,
            'messages': resolved_msgs,
        }
    )


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


def _use_to_wire_metadata(
    registry: Registry,
    use: Sequence[BaseMiddleware | MiddlewareRef] | None,
) -> list[dict[str, Any]] | None:
    """Serialize a prompt's ``use=`` list into the wire-shape the Dev UI reads.

    Produces the ``[{name, config?}]`` list the Prompt Runner sidebar pre-fills
    from ``metadata.prompt.use``. Inline ``BaseMiddleware`` instances surface
    their configured fields so the sidebar matches what the prompt will
    actually run with. The registered name is resolved off ``registry`` so a
    class can live under multiple names without us tying it to a single
    identity. Unregistered instances — subclasses passed inline without going
    through ``@ai.middleware``, ``new_middleware``, or a middleware plugin —
    are dropped because the Dev UI has no name to address them by.
    """
    if use is None:
        return None
    out: list[dict[str, Any]] = []
    cls_index = middleware_class_index(registry)
    for entry in use:
        if isinstance(entry, MiddlewareRef):
            item: dict[str, Any] = {'name': entry.name}
            if entry.config is not None:
                item['config'] = entry.config
            out.append(item)
            continue
        if isinstance(entry, BaseMiddleware):
            name = cls_index.get(type(entry))
            if not name:
                continue
            config = entry.config.model_dump(exclude_none=True, mode='json')
            item = {'name': name}
            if config:
                item['config'] = config
            out.append(item)
    return out


def _parse_dotprompt_use(raw: Any) -> list[MiddlewareRef] | None:  # noqa: ANN401
    """Convert dotprompt frontmatter ``use`` into middleware refs.

    Each entry may be a bare string (middleware name) or a map with ``name`` and
    optional ``config``, matching the cross-SDK MiddlewareRef shape.
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'dotprompt `use` must be a list, got {type(raw).__name__}',
            reason=RuntimeErrorReason.INVALID_INPUT,
        )
    refs: list[MiddlewareRef] = []
    for i, entry in enumerate(raw):
        if isinstance(entry, str):
            if not entry:
                raise GenkitError(
                    status='INVALID_ARGUMENT',
                    message=f'dotprompt `use[{i}]` is an empty string',
                    reason=RuntimeErrorReason.INVALID_INPUT,
                )
            refs.append(MiddlewareRef(name=entry))
        elif isinstance(entry, dict):
            name = entry.get('name')
            if not isinstance(name, str) or not name:
                raise GenkitError(
                    status='INVALID_ARGUMENT',
                    message=f'dotprompt `use[{i}]` is missing required `name` field',
                    reason=RuntimeErrorReason.INVALID_INPUT,
                )
            refs.append(MiddlewareRef(name=name, config=entry.get('config')))
        else:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f'dotprompt `use[{i}]` must be a string or map, got {type(entry).__name__}',
                reason=RuntimeErrorReason.INVALID_INPUT,
            )
    return refs


def _transform_prompt_metadata(
    raw_metadata: Any,  # noqa: ANN401
    variant: str | None,
    template: str,
    registry_key: str,
    name: str,
) -> dict[str, Any]:
    """Transform dotprompt metadata into the format Prompt expects."""
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

    raw = md.get('raw')
    raw_output = raw.get('output') if isinstance(raw, dict) and isinstance(raw.get('output'), dict) else {}
    raw_use = raw.get('use') if isinstance(raw, dict) else None
    parsed_use = _parse_dotprompt_use(raw_use)

    prompt_block: dict[str, Any] = {**md, 'template': template}
    # The Dev UI keys its prompt picker off ``metadata.prompt.name`` and opens
    # the action under that same name, so this has to match the registry key
    # (filename for dotprompts, the explicit name for ``define_prompt``).
    prompt_block['name'] = name
    if parsed_use is not None:
        prompt_block['use'] = [
            ({'name': ref.name, 'config': ref.config} if ref.config is not None else {'name': ref.name})
            for ref in parsed_use
        ]

    # The Dev UI expects an array here; dotprompt leaves it null when no tools are set.
    if prompt_block.get('toolDefs') is None:
        prompt_block['toolDefs'] = []

    # Build metadata structure
    metadata: dict[str, Any] = {
        'type': 'prompt',
        'prompt': prompt_block,
    }

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
            # Fall back to raw YAML (raw_output) because dotpromptz's PromptOutputConfig
            # does not define 'instructions', causing it to be dropped from 'output'.
            'instructions': (
                output.get('instructions')
                if isinstance(output, dict) and 'instructions' in output
                else (raw_output.get('instructions') if isinstance(raw_output, dict) else None)
            ),
        },
        'input': {
            'default': input_cfg.get('default') if isinstance(input_cfg, dict) else None,
            'jsonSchema': input_cfg.get('schema') if isinstance(input_cfg, dict) else None,
        },
        'metadata': metadata,
        'maxTurns': raw.get('maxTurns') if isinstance(raw, dict) else None,
        'toolChoice': raw.get('toolChoice') if isinstance(raw, dict) else None,
        'returnToolRequests': raw.get('returnToolRequests') if isinstance(raw, dict) else None,
        'use': parsed_use,
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
    _cached_prompt: Prompt[Any, Any] | None = None

    async def create_prompt_from_file() -> Prompt[Any, Any]:
        nonlocal _cached_prompt
        if _cached_prompt is not None:
            return _cached_prompt

        raw_metadata = await registry.dotprompt.render_metadata(parsed_prompt)
        metadata = _transform_prompt_metadata(raw_metadata, variant, parsed_prompt.template, registry_key, name)

        executable_prompt = Prompt(
            registry=registry,
            variant=metadata.get('variant'),
            model=metadata.get('model'),
            config=metadata.get('config'),
            description=metadata.get('description'),
            input_schema=metadata.get('input', {}).get('jsonSchema'),
            output_schema=metadata.get('output', {}).get('jsonSchema'),
            output_constrained=True if metadata.get('output', {}).get('jsonSchema') else None,
            output_format=metadata.get('output', {}).get('format'),
            output_instructions=metadata.get('output', {}).get('instructions'),
            messages=metadata.get('messages'),
            max_turns=metadata.get('maxTurns'),
            tool_choice=metadata.get('toolChoice'),
            return_tool_requests=metadata.get('returnToolRequests'),
            metadata=metadata.get('metadata'),
            tools=metadata.get('tools'),
            use=metadata.get('use'),
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

        # Update schemas and metadata on actions for Dev UI
        for action in [prompt_action, exec_prompt_action]:
            if action:
                if metadata.get('input', {}).get('jsonSchema'):
                    action.input_schema = metadata['input']['jsonSchema']
                if metadata.get('output', {}).get('jsonSchema'):
                    action.output_schema = metadata['output']['jsonSchema']
                if metadata.get('metadata'):
                    action._metadata.update(metadata['metadata'])

        _cached_prompt = executable_prompt
        return executable_prompt

    metadata: dict[str, object] = {
        'type': 'prompt',
        'lazy': True,
        'source': 'file',
        'prompt': {'name': name, 'variant': variant or ''},
    }

    action_name = registry_definition_key(name, variant, ns)
    # Frontmatter is parsed eagerly, so the description is known before the lazy load.
    prompt_action, executable_prompt_action = _register_prompt_action_pair(
        registry, action_name, create_prompt_from_file, metadata, description=parsed_prompt.description
    )

    # File-loaded prompts expose their async factory so the tooling can
    # rebuild them on hot-reload without going back through the loader.
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


async def lookup_prompt(registry: Registry, name: str, variant: str | None = None) -> Prompt[Any, Any]:
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
        # First check if we've stored the Prompt directly
        prompt_ref = getattr(action, '_executable_prompt', None)
        if prompt_ref is not None:
            if isinstance(prompt_ref, weakref.ReferenceType):
                resolved = prompt_ref()
                if resolved is not None:
                    return resolved
            if isinstance(prompt_ref, Prompt):
                return prompt_ref
        # Otherwise, create it from the factory (lazy loading)
        async_factory = getattr(action, '_async_factory', None)
        if callable(async_factory):
            # Cast to async callable - getattr returns object but we've verified it's callable
            async_factory_fn = cast(Callable[[], Awaitable[Prompt[Any, Any]]], async_factory)
            executable_prompt = await async_factory_fn()
            if getattr(action, '_executable_prompt', None) is None:
                setattr(action, '_executable_prompt', executable_prompt)  # noqa: B010
            return executable_prompt
        # This shouldn't happen if prompts are loaded correctly
        raise GenkitError(
            status='INTERNAL',
            message=f'Prompt action found but no prompt instance available for {name}',
        )

    variant_str = f' (variant {variant})' if variant else ''
    raise GenkitError(
        status='NOT_FOUND',
        message=f'Prompt {name}{variant_str} not found',
        reason=RuntimeErrorReason.ACTION_NOT_FOUND,
    )


async def prompt(
    registry: Registry,
    name: str,
    variant: str | None = None,
) -> Prompt[Any, Any]:
    """Look up a prompt by name and optional variant."""
    return await lookup_prompt(registry, name, variant)


# Renamed — use ModelStreamResponse
