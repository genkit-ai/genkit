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

"""User-facing asyncio API for Genkit."""

from __future__ import annotations

import asyncio
import inspect
import json
import signal
import socket
import threading
import uuid
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from pathlib import Path
from typing import Any, TypeVar, cast, overload

import anyio
import uvicorn
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from pydantic import BaseModel

from genkit._ai._embedding import EmbedderFn, EmbedderOptions, EmbedderRef, define_embedder
from genkit._ai._evaluator import (
    BatchEvaluatorFn,
    EvaluatorFn,
    EvaluatorRef,
    define_batch_evaluator,
    define_evaluator,
)
from genkit._ai._formats import built_in_formats
from genkit._ai._formats._types import FormatDef
from genkit._ai._generate import define_generate_action, generate_action
from genkit._ai._model import (
    Message,
    ModelConfig,
    ModelFn,
    ModelMiddleware,
    ModelResponse,
    ModelResponseChunk,
    define_model,
)
from genkit._ai._prompt import (
    ExecutablePrompt,
    ModelStreamResponse,
    PromptConfig,
    define_helper,
    define_partial,
    define_schema,
    load_prompt_folder,
    register_prompt_actions,
    to_generate_action_options,
)
from genkit._ai._resource import (
    ResourceFn,
    ResourceOptions,
    define_resource,
)
from genkit._ai._tools import Tool, define_tool
from genkit._core._action import Action, ActionKind, ActionRunContext
from genkit._core._background import (
    BackgroundAction,
    CancelModelOpFn,
    CheckModelOpFn,
    StartModelOpFn,
    check_operation,
    define_background_model,
    lookup_background_action,
)
from genkit._core._channel import Channel, run_loop
from genkit._core._dap import (
    DapFn,
    DynamicActionProvider,
    define_dynamic_action_provider as define_dap_block,
)
from genkit._core._environment import is_dev_environment
from genkit._core._error import GenkitError
from genkit._core._logger import get_logger
from genkit._core._model import Document
from genkit._core._plugin import Plugin
from genkit._core._reflection import ReflectionServer, ServerSpec, create_reflection_asgi_app
from genkit._core._registry import Registry
from genkit._core._tracing import run_in_new_span
from genkit._core._typing import (
    BaseDataPoint,
    Embedding,
    EmbedRequest,
    EvalRequest,
    EvalResponse,
    ModelInfo,
    Operation,
    Part,
    SpanMetadata,
    ToolChoice,
)

from ._decorators import _FlowDecorator, _FlowDecoratorWithChunk
from ._runtime import RuntimeManager

logger = get_logger(__name__)

# TypeVars for generic input/output typing
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')
ChunkT = TypeVar('ChunkT')

R = TypeVar('R')
T = TypeVar('T')


def _model_supports_long_running(model_action: Action) -> bool:
    """Check if a model action supports long-running operations."""
    model_info = model_action.metadata.get('model') if model_action.metadata else None
    if not model_info:
        return False
    # Handle ModelInfo object
    if hasattr(model_info, 'supports'):
        supports = getattr(model_info, 'supports', None)
        return bool(getattr(supports, 'long_running', False)) if supports else False
    # Handle dict (cast needed because isinstance narrows too much for type checkers)
    if isinstance(model_info, dict):
        model_dict = cast(dict[str, Any], model_info)
        supports = model_dict.get('supports')
        return bool(supports.get('longRunning', False)) if isinstance(supports, dict) else False
    return False


class Genkit:
    """Genkit asyncio user-facing API."""

    def __init__(
        self,
        plugins: list[Plugin] | None = None,
        model: str | None = None,
        prompt_dir: str | Path | None = None,
        reflection_server_spec: ServerSpec | None = None,
    ) -> None:
        self.registry: Registry = Registry()
        self._reflection_server_spec: ServerSpec | None = reflection_server_spec
        self._reflection_ready = threading.Event()
        self._initialize_registry(model, plugins)
        # Ensure the default generate action is registered for async usage.
        define_generate_action(self.registry)
        # In dev mode, start the reflection server immediately in a background
        # daemon thread so it's available regardless of which web framework (or
        # none) the user chooses.
        if is_dev_environment():
            self._start_reflection_background()

        # Load prompts
        load_path = prompt_dir
        if load_path is None:
            default_prompts_path = Path('./prompts')
            if default_prompts_path.is_dir():
                load_path = default_prompts_path

        if load_path:
            load_prompt_folder(self.registry, dir_path=load_path)

    # -------------------------------------------------------------------------
    # Registry methods
    # -------------------------------------------------------------------------

    @overload
    def flow(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
        chunk_type: None = None,
    ) -> _FlowDecorator: ...

    @overload
    def flow(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
        chunk_type: type[ChunkT],
    ) -> _FlowDecoratorWithChunk[ChunkT]: ...

    def flow(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
        chunk_type: type[Any] | None = None,
    ) -> _FlowDecorator | _FlowDecoratorWithChunk[Any]:
        """Decorator to register an async function as a flow.

        Args:
            name: Optional name for the flow. Defaults to the function name.
            description: Optional description for the flow.
            chunk_type: Optional type for streaming chunks. When provided,
                the returned Action will be typed as Action[InputT, OutputT, ChunkT].

        Example:
            @ai.flow()
            async def my_flow(x: str) -> int: ...  # Action[str, int]

            @ai.flow(chunk_type=str)
            async def streaming_flow(x: int, ctx: ActionRunContext) -> str:
                ctx.send_chunk("progress")
                return "done"
            # Action[int, str, str]
        """
        if chunk_type is not None:
            return _FlowDecoratorWithChunk(self.registry, name, description, chunk_type)
        return _FlowDecorator(self.registry, name, description)

    def define_helper(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a Handlebars helper function."""
        define_helper(self.registry, name, fn)

    def define_partial(self, name: str, source: str) -> None:
        """Register a Handlebars partial template."""
        define_partial(self.registry, name, source)

    def define_schema(self, name: str, schema: type[BaseModel]) -> type[BaseModel]:
        """Register a Pydantic schema for use in prompts."""
        define_schema(self.registry, name, schema)
        return schema

    def define_json_schema(self, name: str, json_schema: dict[str, object]) -> dict[str, object]:
        """Register a JSON schema for use in prompts."""
        self.registry.register_schema(name, json_schema)
        return json_schema

    def define_dynamic_action_provider(
        self,
        name: str,
        fn: DapFn,
        *,
        description: str | None = None,
        cache_ttl_millis: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DynamicActionProvider:
        """Register a Dynamic Action Provider (DAP)."""
        return define_dap_block(
            self.registry,
            name,
            fn,
            description=description,
            cache_ttl_millis=cache_ttl_millis,
            metadata=metadata,
        )

    def tool(self, name: str | None = None, description: str | None = None) -> Callable[[Callable[..., Any]], Tool]:
        """Decorator to register a function as a tool."""

        def wrapper(func: Callable[..., Any]) -> Tool:
            return define_tool(self.registry, func, name, description)

        return wrapper

    def define_evaluator(
        self,
        *,
        name: str,
        display_name: str,
        definition: str,
        fn: EvaluatorFn[Any],
        is_billed: bool = False,
        config_schema: type[BaseModel] | dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
        description: str | None = None,
    ) -> Action:
        """Register an evaluator action."""
        return define_evaluator(
            self.registry,
            name=name,
            display_name=display_name,
            definition=definition,
            fn=fn,
            is_billed=is_billed,
            config_schema=config_schema,
            metadata=metadata,
            description=description,
        )

    def define_batch_evaluator(
        self,
        *,
        name: str,
        display_name: str,
        definition: str,
        fn: BatchEvaluatorFn[Any],
        is_billed: bool = False,
        config_schema: type[BaseModel] | dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
        description: str | None = None,
    ) -> Action:
        """Register a batch evaluator action."""
        return define_batch_evaluator(
            self.registry,
            name=name,
            display_name=display_name,
            definition=definition,
            fn=fn,
            is_billed=is_billed,
            config_schema=config_schema,
            metadata=metadata,
            description=description,
        )

    def define_model(
        self,
        name: str,
        fn: ModelFn,
        config_schema: type[BaseModel] | dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
        info: ModelInfo | None = None,
        description: str | None = None,
    ) -> Action:
        """Register a custom model action."""
        return define_model(self.registry, name, fn, config_schema, metadata, info, description)

    def define_background_model(
        self,
        name: str,
        start: StartModelOpFn,
        check: CheckModelOpFn,
        cancel: CancelModelOpFn | None = None,
        label: str | None = None,
        info: ModelInfo | None = None,
        config_schema: type[BaseModel] | dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
        description: str | None = None,
    ) -> BackgroundAction:
        """Register a background model for long-running AI operations."""
        return define_background_model(
            registry=self.registry,
            name=name,
            start=start,
            check=check,
            cancel=cancel,
            label=label,
            info=info,
            config_schema=config_schema,
            metadata=metadata,
            description=description,
        )

    def define_embedder(
        self,
        name: str,
        fn: EmbedderFn,
        options: EmbedderOptions | None = None,
        metadata: dict[str, object] | None = None,
        description: str | None = None,
    ) -> Action:
        """Register a custom embedder action."""
        return define_embedder(self.registry, name, fn, options, metadata, description)

    def define_format(self, format: FormatDef) -> None:
        """Register a custom output format."""
        self.registry.register_value('format', format.name, format)

    # Overload 1: Both input_schema and output_schema typed -> ExecutablePrompt[InputT, OutputT]
    @overload
    def define_prompt(
        self,
        name: str | None = None,
        *,
        variant: str | None = None,
        model: str | None = None,
        config: dict[str, object] | ModelConfig | None = None,
        description: str | None = None,
        system: str | list[Part] | None = None,
        prompt: str | list[Part] | None = None,
        messages: str | list[Message] | None = None,
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: str | None = None,
        output_constrained: bool | None = None,
        max_turns: int | None = None,
        return_tool_requests: bool | None = None,
        metadata: dict[str, object] | None = None,
        tools: Sequence[str | Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        use: list[ModelMiddleware] | None = None,
        docs: list[Document] | None = None,
        input_schema: type[InputT],
        output_schema: type[OutputT],
    ) -> ExecutablePrompt[InputT, OutputT]: ...

    # Overload 2: Only input_schema typed -> ExecutablePrompt[InputT, Any]
    @overload
    def define_prompt(
        self,
        name: str | None = None,
        *,
        variant: str | None = None,
        model: str | None = None,
        config: dict[str, object] | ModelConfig | None = None,
        description: str | None = None,
        system: str | list[Part] | None = None,
        prompt: str | list[Part] | None = None,
        messages: str | list[Message] | None = None,
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: str | None = None,
        output_constrained: bool | None = None,
        max_turns: int | None = None,
        return_tool_requests: bool | None = None,
        metadata: dict[str, object] | None = None,
        tools: Sequence[str | Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        use: list[ModelMiddleware] | None = None,
        docs: list[Document] | None = None,
        input_schema: type[InputT],
        output_schema: dict[str, object] | str | None = None,
    ) -> ExecutablePrompt[InputT, Any]: ...

    # Overload 3: Only output_schema typed -> ExecutablePrompt[Any, OutputT]
    @overload
    def define_prompt(
        self,
        name: str | None = None,
        *,
        variant: str | None = None,
        model: str | None = None,
        config: dict[str, object] | ModelConfig | None = None,
        description: str | None = None,
        system: str | list[Part] | None = None,
        prompt: str | list[Part] | None = None,
        messages: str | list[Message] | None = None,
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: str | None = None,
        output_constrained: bool | None = None,
        max_turns: int | None = None,
        return_tool_requests: bool | None = None,
        metadata: dict[str, object] | None = None,
        tools: Sequence[str | Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        use: list[ModelMiddleware] | None = None,
        docs: list[Document] | None = None,
        input_schema: dict[str, object] | str | None = None,
        output_schema: type[OutputT],
    ) -> ExecutablePrompt[Any, OutputT]: ...

    # Overload 4: Neither typed -> ExecutablePrompt[Any, Any]
    @overload
    def define_prompt(
        self,
        name: str | None = None,
        *,
        variant: str | None = None,
        model: str | None = None,
        config: dict[str, object] | ModelConfig | None = None,
        description: str | None = None,
        system: str | list[Part] | None = None,
        prompt: str | list[Part] | None = None,
        messages: str | list[Message] | None = None,
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: str | None = None,
        output_constrained: bool | None = None,
        max_turns: int | None = None,
        return_tool_requests: bool | None = None,
        metadata: dict[str, object] | None = None,
        tools: Sequence[str | Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        use: list[ModelMiddleware] | None = None,
        docs: list[Document] | None = None,
        input_schema: dict[str, object] | str | None = None,
        output_schema: dict[str, object] | str | None = None,
    ) -> ExecutablePrompt[Any, Any]: ...

    def define_prompt(
        self,
        name: str | None = None,
        *,
        variant: str | None = None,
        model: str | None = None,
        config: dict[str, object] | ModelConfig | None = None,
        description: str | None = None,
        system: str | list[Part] | None = None,
        prompt: str | list[Part] | None = None,
        messages: str | list[Message] | None = None,
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: str | None = None,
        output_constrained: bool | None = None,
        max_turns: int | None = None,
        return_tool_requests: bool | None = None,
        metadata: dict[str, object] | None = None,
        tools: Sequence[str | Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        use: list[ModelMiddleware] | None = None,
        docs: list[Document] | None = None,
        input_schema: type | dict[str, object] | str | None = None,
        output_schema: type | dict[str, object] | str | None = None,
    ) -> ExecutablePrompt[Any, Any]:
        """Register a prompt template."""
        executable_prompt = ExecutablePrompt(
            self.registry,
            variant=variant,
            model=model,
            config=config,
            description=description,
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
            name=name,
        )
        if name:
            register_prompt_actions(self.registry, executable_prompt, name, variant)
        return executable_prompt

    # Overload 1: Neither typed -> ExecutablePrompt[Any, Any]
    @overload
    def prompt(
        self,
        name: str,
        *,
        variant: str | None = None,
        input_schema: None = None,
        output_schema: None = None,
    ) -> ExecutablePrompt[Any, Any]: ...

    # Overload 2: Only input_schema typed
    @overload
    def prompt(
        self,
        name: str,
        *,
        variant: str | None = None,
        input_schema: type[InputT],
        output_schema: None = None,
    ) -> ExecutablePrompt[InputT, Any]: ...

    # Overload 3: Only output_schema typed
    @overload
    def prompt(
        self,
        name: str,
        *,
        variant: str | None = None,
        input_schema: None = None,
        output_schema: type[OutputT],
    ) -> ExecutablePrompt[Any, OutputT]: ...

    # Overload 4: Both input_schema and output_schema typed
    @overload
    def prompt(
        self,
        name: str,
        *,
        variant: str | None = None,
        input_schema: type[InputT],
        output_schema: type[OutputT],
    ) -> ExecutablePrompt[InputT, OutputT]: ...

    def prompt(
        self,
        name: str,
        *,
        variant: str | None = None,
        input_schema: type[InputT] | None = None,
        output_schema: type[OutputT] | None = None,
    ) -> ExecutablePrompt[InputT, OutputT] | ExecutablePrompt[Any, Any]:
        """Look up a prompt by name and optional variant."""
        return ExecutablePrompt(
            registry=self.registry,
            name=name,
            variant=variant,
            input_schema=input_schema,
            output_schema=output_schema,
        )

    def define_resource(
        self,
        *,
        fn: ResourceFn,
        name: str | None = None,
        uri: str | None = None,
        template: str | None = None,
        description: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Action:
        """Register a resource action."""
        opts: ResourceOptions = {}
        if name:
            opts['name'] = name
        if uri:
            opts['uri'] = uri
        if template:
            opts['template'] = template
        if description:
            opts['description'] = description
        if metadata:
            opts['metadata'] = metadata

        return define_resource(self.registry, opts, fn)

    # -------------------------------------------------------------------------
    # Server infrastructure methods
    # -------------------------------------------------------------------------

    def _start_reflection_background(self) -> None:
        """Start the Dev UI reflection server in a background daemon thread."""

        async def _run_server() -> None:
            sockets: list[socket.socket] | None = None
            spec = self._reflection_server_spec
            if spec is None:
                # Bind to port 0 to let OS choose available port, pass socket to uvicorn
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('127.0.0.1', 0))
                sock.listen(2048)
                host, port = sock.getsockname()
                spec = ServerSpec(scheme='http', host=host, port=port)
                self._reflection_server_spec = spec
                sockets = [sock]

            app = create_reflection_asgi_app(registry=self.registry)
            config = uvicorn.Config(app, host=spec.host, port=spec.port, loop='asyncio')
            server = ReflectionServer(config, ready=self._reflection_ready)
            async with RuntimeManager(spec, lazy_write=True) as runtime_manager:
                server_task = asyncio.create_task(server.serve(sockets=sockets))
                await asyncio.to_thread(self._reflection_ready.wait)

                if server.should_exit:
                    logger.warning(f'Reflection server at {spec.url} failed to start.')
                    return

                runtime_manager.write_runtime_file()
                await logger.ainfo(f'Genkit Dev UI reflection server running at {spec.url}')
                await server_task

        threading.Thread(
            target=lambda: asyncio.run(_run_server()),
            daemon=True,
            name='genkit-reflection-server',
        ).start()

    def _initialize_registry(self, model: str | None, plugins: list[Plugin] | None) -> None:
        """Initialize the registry with default model and plugins."""
        if model:
            self.registry.register_value('defaultModel', 'defaultModel', model)
        for fmt in built_in_formats:
            self.define_format(fmt)

        if not plugins:
            logger.warning('No plugins provided to Genkit')
        else:
            for plugin in plugins:
                if isinstance(plugin, Plugin):  # pyright: ignore[reportUnnecessaryIsInstance]
                    self.registry.register_plugin(plugin)
                else:
                    raise ValueError(f'Invalid {plugin=} provided to Genkit: must be of type `genkit.ai.Plugin`')

    def run_main(self, coro: Coroutine[Any, Any, T]) -> T | None:
        """Run the user's main coroutine, blocking in dev mode for the reflection server."""
        if not is_dev_environment():
            logger.info('Running in production mode.')
            return run_loop(coro)

        logger.info('Running in development mode.')

        async def dev_runner() -> T | None:
            user_result: T | None = None
            try:
                user_result = await coro
                logger.debug('User coroutine completed successfully.')
            except Exception:
                logger.exception('User coroutine failed')

            # Block until Ctrl+C (SIGINT handled by anyio) or SIGTERM, keeping
            # the daemon reflection thread alive.
            logger.info('Script done — Dev UI running. Press Ctrl+C to stop.')
            try:
                async with anyio.create_task_group() as tg:

                    async def _handle_sigterm(tg_: anyio.abc.TaskGroup) -> None:  # type: ignore[name-defined]
                        with anyio.open_signal_receiver(signal.SIGTERM) as sigs:
                            async for _ in sigs:
                                tg_.cancel_scope.cancel()
                                return

                    tg.start_soon(_handle_sigterm, tg)
                    await anyio.sleep_forever()
            except anyio.get_cancelled_exc_class():
                pass

            logger.info('Dev UI server stopped.')
            return user_result

        return anyio.run(dev_runner)

    # -------------------------------------------------------------------------
    # Genkit-specific methods (generation, embedding, retrieval, etc.)
    # -------------------------------------------------------------------------

    def _resolve_embedder_name(self, embedder: str | EmbedderRef | None) -> str:
        """Resolve embedder name from string or EmbedderRef."""
        if isinstance(embedder, EmbedderRef):
            return embedder.name
        elif isinstance(embedder, str):
            return embedder
        else:
            raise ValueError('Embedder must be specified as a string name or an EmbedderRef.')

    # Overload: output_schema=type[T] -> ModelResponse[T]
    @overload
    async def generate(
        self,
        *,
        model: str | None = None,
        prompt: str | list[Part] | None = None,
        system: str | list[Part] | None = None,
        messages: list[Message] | None = None,
        tools: Sequence[str | Tool] | None = None,
        return_tool_requests: bool | None = None,
        tool_choice: ToolChoice | None = None,
        tool_responses: list[Part] | None = None,
        config: dict[str, object] | ModelConfig | None = None,
        max_turns: int | None = None,
        context: dict[str, object] | None = None,
        output_schema: type[OutputT],
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: str | None = None,
        output_constrained: bool | None = None,
        use: list[ModelMiddleware] | None = None,
        docs: list[Document] | None = None,
    ) -> ModelResponse[OutputT]: ...

    # Overload: no output_schema, dict, or union -> ModelResponse[Any]
    @overload
    async def generate(
        self,
        *,
        model: str | None = None,
        prompt: str | list[Part] | None = None,
        system: str | list[Part] | None = None,
        messages: list[Message] | None = None,
        tools: Sequence[str | Tool] | None = None,
        return_tool_requests: bool | None = None,
        tool_choice: ToolChoice | None = None,
        tool_responses: list[Part] | None = None,
        config: dict[str, object] | ModelConfig | None = None,
        max_turns: int | None = None,
        context: dict[str, object] | None = None,
        output_schema: type | dict | None = None,
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: str | None = None,
        output_constrained: bool | None = None,
        use: list[ModelMiddleware] | None = None,
        docs: list[Document] | None = None,
    ) -> ModelResponse[Any]: ...

    async def generate(
        self,
        *,
        model: str | None = None,
        prompt: str | list[Part] | None = None,
        system: str | list[Part] | None = None,
        messages: list[Message] | None = None,
        tools: Sequence[str | Tool] | None = None,
        return_tool_requests: bool | None = None,
        tool_choice: ToolChoice | None = None,
        tool_responses: list[Part] | None = None,
        config: dict[str, object] | ModelConfig | None = None,
        max_turns: int | None = None,
        context: dict[str, object] | None = None,
        output_schema: type | dict | None = None,
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: str | None = None,
        output_constrained: bool | None = None,
        use: list[ModelMiddleware] | None = None,
        docs: list[Document] | None = None,
    ) -> ModelResponse[Any]:
        """Generate text or structured data using a language model.

        ``tools`` is typed as ``Sequence`` rather than ``list`` because ``Sequence``
        is covariant: ``list[Tool]`` or ``list[str]`` are both assignable to
        ``Sequence[str | Tool]``, but not to ``list[str | Tool]``.
        """
        return await generate_action(
            self.registry,
            await to_generate_action_options(
                self.registry,
                PromptConfig(
                    model=model,
                    prompt=prompt,
                    system=system,
                    messages=messages,
                    tools=tools,
                    return_tool_requests=return_tool_requests,
                    tool_choice=tool_choice,
                    tool_responses=tool_responses,
                    config=config,
                    max_turns=max_turns,
                    output_format=output_format,
                    output_content_type=output_content_type,
                    output_instructions=output_instructions,
                    output_schema=output_schema,
                    output_constrained=output_constrained,
                    docs=docs,
                ),
            ),
            middleware=use,
            context=context if context else ActionRunContext._current_context(),  # pyright: ignore[reportPrivateUsage]
        )

    # Overload: output_schema=type[T] -> ModelStreamResponse[T]
    @overload
    def generate_stream(
        self,
        *,
        model: str | None = None,
        prompt: str | list[Part] | None = None,
        system: str | list[Part] | None = None,
        messages: list[Message] | None = None,
        tools: Sequence[str | Tool] | None = None,
        return_tool_requests: bool | None = None,
        tool_choice: ToolChoice | None = None,
        config: dict[str, object] | ModelConfig | None = None,
        max_turns: int | None = None,
        context: dict[str, object] | None = None,
        output_schema: type[OutputT],
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: str | None = None,
        output_constrained: bool | None = None,
        use: list[ModelMiddleware] | None = None,
        docs: list[Document] | None = None,
        timeout: float | None = None,
    ) -> ModelStreamResponse[OutputT]: ...

    # Overload: no output_schema, dict, or union -> ModelStreamResponse[Any]
    @overload
    def generate_stream(
        self,
        *,
        model: str | None = None,
        prompt: str | list[Part] | None = None,
        system: str | list[Part] | None = None,
        messages: list[Message] | None = None,
        tools: Sequence[str | Tool] | None = None,
        return_tool_requests: bool | None = None,
        tool_choice: ToolChoice | None = None,
        config: dict[str, object] | ModelConfig | None = None,
        max_turns: int | None = None,
        context: dict[str, object] | None = None,
        output_schema: type | dict | None = None,
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: str | None = None,
        output_constrained: bool | None = None,
        use: list[ModelMiddleware] | None = None,
        docs: list[Document] | None = None,
        timeout: float | None = None,
    ) -> ModelStreamResponse[Any]: ...

    def generate_stream(
        self,
        *,
        model: str | None = None,
        prompt: str | list[Part] | None = None,
        system: str | list[Part] | None = None,
        messages: list[Message] | None = None,
        tools: Sequence[str | Tool] | None = None,
        return_tool_requests: bool | None = None,
        tool_choice: ToolChoice | None = None,
        config: dict[str, object] | ModelConfig | None = None,
        max_turns: int | None = None,
        context: dict[str, object] | None = None,
        output_schema: type | dict | None = None,
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: str | None = None,
        output_constrained: bool | None = None,
        use: list[ModelMiddleware] | None = None,
        docs: list[Document] | None = None,
        timeout: float | None = None,
    ) -> ModelStreamResponse[Any]:
        """Stream generated text, returning a ModelStreamResponse with .stream and .response."""
        channel: Channel[ModelResponseChunk, ModelResponse[Any]] = Channel(timeout=timeout)

        async def _run_generate() -> ModelResponse[Any]:
            return await generate_action(
                self.registry,
                await to_generate_action_options(
                    self.registry,
                    PromptConfig(
                        model=model,
                        prompt=prompt,
                        system=system,
                        messages=messages,
                        tools=tools,
                        return_tool_requests=return_tool_requests,
                        tool_choice=tool_choice,
                        config=config,
                        max_turns=max_turns,
                        output_format=output_format,
                        output_content_type=output_content_type,
                        output_instructions=output_instructions,
                        output_schema=output_schema,
                        output_constrained=output_constrained,
                        docs=docs,
                    ),
                ),
                on_chunk=lambda c: channel.send(c),
                middleware=use,
                context=context if context else ActionRunContext._current_context(),  # pyright: ignore[reportPrivateUsage]
            )

        response_future: asyncio.Future[ModelResponse[Any]] = asyncio.create_task(_run_generate())
        channel.set_close_future(response_future)

        return ModelStreamResponse[Any](channel=channel, response_future=response_future)

    async def embed(
        self,
        *,
        embedder: str | EmbedderRef | None = None,
        content: str | Document | None = None,
        metadata: dict[str, object] | None = None,
        options: dict[str, object] | None = None,
    ) -> list[Embedding]:
        """Generate vector embeddings for a single document or string."""
        embedder_name = self._resolve_embedder_name(embedder)
        embedder_config: dict[str, object] = {}

        # Extract config and version from EmbedderRef (not done for embed_many per JS behavior)
        if isinstance(embedder, EmbedderRef):
            embedder_config = embedder.config or {}
            if embedder.version:
                embedder_config['version'] = embedder.version  # Handle version from ref

        # Merge options passed to embed() with config from EmbedderRef
        final_options = {**(embedder_config or {}), **(options or {})}

        embed_action = await self.registry.resolve_embedder(embedder_name)
        if embed_action is None:
            raise ValueError(f'Embedder "{embedder_name}" not found')

        if content is None:
            raise ValueError('Content must be specified for embedding.')

        documents = [Document.from_text(content, metadata)] if isinstance(content, str) else [content]

        response = (
            await embed_action.run(
                EmbedRequest(
                    input=documents,  # pyright: ignore[reportArgumentType]
                    options=final_options,
                )
            )
        ).response
        return response.embeddings

    async def embed_many(
        self,
        *,
        embedder: str | EmbedderRef | None = None,
        content: list[str] | list[Document] | None = None,
        metadata: dict[str, object] | None = None,
        options: dict[str, object] | None = None,
    ) -> list[Embedding]:
        """Generate vector embeddings for multiple documents in a single batch call."""
        if content is None:
            raise ValueError('Content must be specified for embedding.')

        # Convert strings to Documents if needed
        documents: list[Document] = [
            Document.from_text(item, metadata) if isinstance(item, str) else item for item in content
        ]

        # Resolve embedder name (JS embedMany does not extract config/version from ref)
        embedder_name = self._resolve_embedder_name(embedder)

        embed_action = await self.registry.resolve_embedder(embedder_name)
        if embed_action is None:
            raise ValueError(f'Embedder "{embedder_name}" not found')

        response = (await embed_action.run(EmbedRequest(input=documents, options=options))).response  # type: ignore[arg-type]
        return response.embeddings

    async def evaluate(
        self,
        evaluator: str | EvaluatorRef | None = None,
        dataset: list[BaseDataPoint] | None = None,
        options: dict[str, object] | None = None,
        eval_run_id: str | None = None,
    ) -> EvalResponse:
        """Evaluate a dataset using the specified evaluator."""
        evaluator_name: str = ''
        evaluator_config: dict[str, object] = {}

        if isinstance(evaluator, EvaluatorRef):
            evaluator_name = evaluator.name
            evaluator_config = evaluator.config_schema or {}
        elif isinstance(evaluator, str):
            evaluator_name = evaluator
        else:
            raise ValueError('Evaluator must be specified as a string name or an EvaluatorRef.')

        final_options = {**(evaluator_config or {}), **(options or {})}

        eval_action = await self.registry.resolve_evaluator(evaluator_name)
        if eval_action is None:
            raise ValueError(f'Evaluator "{evaluator_name}" not found')

        if not eval_run_id:
            eval_run_id = str(uuid.uuid4())

        if dataset is None:
            raise ValueError('Dataset must be specified for evaluation.')

        return (
            await eval_action.run(
                EvalRequest(
                    dataset=dataset,
                    options=final_options,
                    eval_run_id=eval_run_id,
                )
            )
        ).response

    @staticmethod
    def current_context() -> dict[str, Any] | None:
        """Get the current execution context, or None if not in an action."""
        return ActionRunContext._current_context()  # pyright: ignore[reportPrivateUsage]

    async def flush_tracing(self) -> None:
        """Flush all pending trace spans to exporters."""
        provider = trace_api.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            await asyncio.to_thread(provider.force_flush)

    async def run(
        self,
        *,
        name: str,
        fn: Callable[[], Awaitable[T]],
        metadata: dict[str, Any] | None = None,
    ) -> T:
        """Run a function as a discrete traced step within a flow."""
        if not inspect.iscoroutinefunction(fn):
            raise TypeError('fn must be a coroutine function')

        span_metadata = SpanMetadata(name=name, metadata=metadata)
        with run_in_new_span(span_metadata, labels={'genkit:type': 'flowStep'}) as span:
            try:
                result = await fn()
                output = (
                    result.model_dump_json(by_alias=True, exclude_none=True)
                    if isinstance(result, BaseModel)
                    else json.dumps(result)
                )
                span.set_attribute('genkit:output', output)
                return result
            except Exception:
                # We catch all exceptions here to ensure they are captured by
                # the trace span context manager before being re-raised.
                # The run_in_new_span context manager handles recording
                # the exception details.
                raise

    async def check_operation(self, operation: Operation) -> Operation:
        """Check the status of a long-running background operation."""
        return await check_operation(self.registry, operation)

    async def cancel_operation(self, operation: Operation) -> Operation:
        """Cancel a long-running background operation."""
        if not operation.action:
            raise ValueError('Provided operation is missing original request information')

        background_action = await lookup_background_action(self.registry, operation.action)
        if background_action is None:
            raise ValueError(f'Failed to resolve background action from original request: {operation.action}')

        return await background_action.cancel(operation)

    async def generate_operation(
        self,
        *,
        model: str | None = None,
        prompt: str | list[Part] | None = None,
        system: str | list[Part] | None = None,
        messages: list[Message] | None = None,
        tools: Sequence[str | Tool] | None = None,
        return_tool_requests: bool | None = None,
        tool_choice: ToolChoice | None = None,
        config: dict[str, object] | ModelConfig | None = None,
        max_turns: int | None = None,
        context: dict[str, object] | None = None,
        output_schema: type | dict | None = None,
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: str | None = None,
        output_constrained: bool | None = None,
        use: list[ModelMiddleware] | None = None,
        docs: list[Document] | None = None,
    ) -> Operation:
        """Generate content using a long-running model, returning an Operation to poll."""
        # Resolve the model and check for long_running support
        resolved_model = model or cast(str | None, self.registry.lookup_value('defaultModel', 'defaultModel'))
        if not resolved_model:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message='No model specified for generate_operation.',
            )

        model_action = await self.registry.resolve_action(ActionKind.MODEL, resolved_model)
        if not model_action:
            raise GenkitError(
                status='NOT_FOUND',
                message=f"Model '{resolved_model}' not found.",
            )

        # Check if model supports long-running operations
        if not _model_supports_long_running(model_action):
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f"Model '{model_action.name}' does not support long running operations.",
            )

        # Call generate
        response = await self.generate(
            model=model,
            prompt=prompt,
            system=system,
            messages=messages,
            tools=tools,
            return_tool_requests=return_tool_requests,
            tool_choice=tool_choice,
            config=config,
            max_turns=max_turns,
            context=context,
            output_schema=output_schema,
            output_format=output_format,
            output_content_type=output_content_type,
            output_instructions=output_instructions,
            output_constrained=output_constrained,
            use=use,
            docs=docs,
        )

        # Extract operation from response
        if not hasattr(response, 'operation') or not response.operation:
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message=f"Model '{model_action.name}' did not return an operation.",
            )

        return response.operation
