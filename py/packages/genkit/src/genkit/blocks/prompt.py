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


"""Prompt management and templating for the Genkit framework.

This module provides the ExecutablePrompt class and related types for managing
AI prompts in Genkit. It enables defining reusable prompts with templates,
input/output schemas, tool configurations, and more.

Overview:
    Prompts in Genkit are reusable templates that can be executed against AI
    models. They encapsulate the prompt text, system instructions, model
    configuration, and other generation options. The ExecutablePrompt class
    provides a callable interface matching the JavaScript SDK.

Key Concepts:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Term                │ Description                                       │
    ├─────────────────────┼───────────────────────────────────────────────────┤
    │ ExecutablePrompt    │ A prompt that can be called like a function       │
    │ PromptGenerateOpts  │ Options to override prompt defaults at runtime    │
    │ GenerateStreamResp  │ Response object with stream and response props    │
    │ .prompt files       │ Dotprompt template files (YAML frontmatter + HBS) │
    │ Variant             │ Alternative version of a prompt (e.g., casual)    │
    └─────────────────────┴───────────────────────────────────────────────────┘

Prompt Execution Flow:
    ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
    │   Define     │ ──► │   Render    │ ──► │   Generate   │
    │   Prompt     │     │   Template  │     │   Response   │
    └──────────────┘     └─────────────┘     └──────────────┘
         │                     │                    │
         │ ai.define_prompt()  │ prompt.render()    │ model.generate()
         │ or .prompt file     │                    │
         ▼                     ▼                    ▼
    ExecutablePrompt    GenerateActionOptions   GenerateResponse

Key Operations:
    - Define prompts programmatically with `ai.define_prompt()`
    - Load prompts from .prompt files with `load_prompt_folder()`
    - Look up prompts by name with `ai.prompt()`
    - Execute prompts with `await prompt(input)` or `prompt.stream(input)`
    - Render prompts without executing with `await prompt.render(input)`
    - Override options at runtime with `opts` parameter

Example:
    Basic usage with programmatic prompt:

    ```python
    from genkit.ai import Genkit

    ai = Genkit(model='googleai/gemini-2.0-flash')

    # Define a prompt
    recipe_prompt = ai.define_prompt(
        name='recipe',
        system='You are a helpful chef.',
        prompt='Create a recipe for {{food}}.',
        config={'temperature': 0.7},
    )

    # Execute the prompt
    response = await recipe_prompt({'food': 'pizza'})
    print(response.text)

    # Override options at runtime
    response = await recipe_prompt(
        {'food': 'salad'},
        opts={
            'config': {'temperature': 0.5},  # Merged with prompt config
            'model': 'googleai/gemini-1.5-pro',  # Override model
        },
    )

    # Stream the response
    result = recipe_prompt.stream({'food': 'soup'})
    async for chunk in result.stream:
        print(chunk.text, end='')
    final = await result.response
    ```

    Using .prompt files (Dotprompt):

    ```
    # prompts/recipe.prompt
    ---
    model: googleai/gemini-2.0-flash
    config:
      temperature: 0.7
    input:
      schema:
        food: string
    ---
    You are a helpful chef.
    Create a recipe for {{food}}.
    ```

    ```python
    # Load and use the prompt
    recipe = ai.prompt('recipe')
    response = await recipe({'food': 'curry'})
    ```

Caveats:
    - Config values are MERGED (not replaced) when using opts.config
    - The `system` and `prompt` fields cannot be overridden via opts
    - Message resolvers receive opts.messages as `history`, not appended
    - Python uses snake_case (e.g., `as_tool()`) vs JS camelCase (`asTool()`)

See Also:
    - JavaScript implementation: js/ai/src/prompt.ts
    - Dotprompt documentation: https://genkit.dev/docs/dotprompt
"""

import asyncio
import os
import weakref
from collections.abc import AsyncIterable, Awaitable, Callable
from pathlib import Path
from typing import Any, ClassVar, Generic, TypedDict, TypeVar, cast, overload

from dotpromptz.typing import (
    DataArgument,
    PromptFunction,
    PromptInputConfig,
    PromptMetadata,
)
from pydantic import BaseModel, ConfigDict

from genkit.aio import Channel, ensure_async
from genkit.blocks.generate import (
    StreamingCallback as ModelStreamingCallback,
    generate_action,
    to_tool_definition,
)
from genkit.blocks.interfaces import Input, Output
from genkit.blocks.model import (
    GenerateResponseChunkWrapper,
    GenerateResponseWrapper,
    ModelMiddleware,
)
from genkit.core.action import Action, ActionRunContext, create_action_key
from genkit.core.action.types import ActionKind
from genkit.core.error import GenkitError
from genkit.core.logging import get_logger
from genkit.core.registry import Registry
from genkit.core.schema import to_json_schema
from genkit.core.typing import (
    DocumentData,
    GenerateActionOptions,
    GenerateActionOutputConfig,
    GenerateRequest,
    GenerationCommonConfig,
    Message,
    OutputConfig,
    Part,
    Resume,
    Role,
    TextPart,
    ToolChoice,
    ToolRequestPart,
    ToolResponsePart,
)

logger = get_logger(__name__)

# TypeVars for generic input/output typing
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


class OutputOptions(TypedDict, total=False):
    """Output configuration options for prompt generation.

    This matches the JavaScript OutputOptions interface and allows overriding
    output configuration when executing a prompt.

    Overview:
        OutputOptions controls how the model's response should be formatted.
        You can request JSON output, constrain it to a schema, or provide
        custom formatting instructions.

    Attributes:
        format: Output format. Common values: 'json', 'text'. Defaults to
            model's native format if not specified.
        content_type: MIME content type for the output (e.g., 'application/json').
        instructions: Instructions for formatting output. Can be:
            - True: Use default formatting instructions
            - False: Disable formatting instructions
            - str: Custom instructions to append to prompt
        schema: Output schema for structured output. Can be:
            - A Python type/class (Pydantic model or dataclass)
            - A dict representing JSON Schema
            - A string referencing a registered schema name
        json_schema: Direct JSON Schema definition for the output.
        constrained: Whether to constrain model output to the schema.
            When True, the model will be forced to output valid JSON
            matching the schema (if supported by the model).

    Example:
        ```python
        from pydantic import BaseModel


        class Recipe(BaseModel):
            name: str
            ingredients: list[str]
            steps: list[str]


        response = await prompt(
            {'food': 'pizza'},
            opts={
                'output': {
                    'format': 'json',
                    'schema': Recipe,
                    'constrained': True,
                }
            },
        )
        recipe = response.output  # Parsed Recipe object
        ```
    """

    format: str | None
    content_type: str | None
    instructions: bool | str | None
    schema: type | dict[str, Any] | str | None
    json_schema: dict[str, Any] | None
    constrained: bool | None


class ResumeOptions(TypedDict, total=False):
    """Options for resuming generation after an interrupt.

    This matches the JavaScript ResumeOptions interface and enables
    human-in-the-loop workflows where tool execution can be paused
    for user confirmation or input.

    Overview:
        When a tool is defined as an "interrupt", the model's tool call
        is not automatically executed. Instead, the response contains
        the tool request, allowing your application to:
        1. Present the tool call to a user for approval
        2. Modify the tool arguments
        3. Provide a custom response without executing the tool

        ResumeOptions is used to continue generation after handling
        the interrupt.

    Attributes:
        respond: Tool response part(s) to respond to interrupt tool requests.
            Each response must have a matching `name` (and `ref` if supplied)
            for its corresponding tool request.
        restart: Tool request part(s) to restart with additional metadata.
            This re-executes the tool with `resumed` metadata passed to
            the tool function.
        metadata: Additional metadata to annotate the created tool message
            under the "resume" key.

    Example:
        ```python
        # Define an interrupt tool for user confirmation
        @ai.tool(name='book_flight', interrupt=True)
        def book_flight(destination: str, date: str) -> str:
            # This won't be called automatically
            return f'Booked flight to {destination} on {date}'


        # First generate - gets interrupted
        response = await prompt({'request': 'Book a flight to Paris'})
        interrupt = response.interrupts[0]

        # Resume after user confirms
        response = await prompt(
            {'request': 'Book a flight to Paris'},
            opts={
                'messages': response.messages,
                'resume': {
                    'respond': book_flight.respond(interrupt, 'Confirmed: Flight booked to Paris'),
                },
            },
        )
        ```

    See Also:
        - Interrupts documentation: https://genkit.dev/docs/tool-calling#pause-agentic-loops-with-interrupts
    """

    respond: ToolResponsePart | list[ToolResponsePart] | None
    restart: ToolRequestPart | list[ToolRequestPart] | None
    metadata: dict[str, Any] | None


class PromptGenerateOptions(TypedDict, total=False):
    """Options for generating with a prompt at runtime.

    This matches the JavaScript PromptGenerateOptions type (GenerateOptions
    minus 'prompt' and 'system' fields, which are defined by the prompt).

    Overview:
        PromptGenerateOptions allows overriding a prompt's default configuration
        when executing it. Options are passed as the second argument to
        ExecutablePrompt.__call__(), render(), and stream().

        ┌─────────────────────────────────────────────────────────────────────┐
        │ Merge Behavior                                                      │
        ├─────────────────────┬───────────────────────────────────────────────┤
        │ config              │ MERGED: {...promptConfig, ...optsConfig}      │
        │ metadata            │ MERGED: {...promptMetadata, ...optsMetadata}  │
        │ All other fields    │ REPLACED: opts value overrides prompt value   │
        └─────────────────────┴───────────────────────────────────────────────┘

    Attributes:
        model: Override the model to use for generation. Can be a model name
            string like 'googleai/gemini-2.0-flash'.
        config: Model configuration options (temperature, topK, etc.).
            These are MERGED with the prompt's config, not replaced.
        messages: Conversation history for multi-turn prompts. Behavior:
            - If prompt has messages template: passed as 'history' to resolver
            - If prompt has no messages: used directly as the messages
        docs: Additional documents for RAG/grounding context.
        tools: Override the tools available for this generation.
        resources: Dynamic resources (MCP resources) to make available.
        tool_choice: Tool selection strategy:
            - 'auto': Model decides when to use tools (default)
            - 'required': Model must use at least one tool
            - 'none': Model cannot use tools
        output: Override output configuration (format, schema, etc.).
        resume: Options for resuming after an interrupt (human-in-the-loop).
        return_tool_requests: If True, return tool calls without auto-executing.
            Useful for custom tool handling or inspection.
        max_turns: Maximum tool call iterations (default: 5). Limits
            back-and-forth between model and tools.
        on_chunk: Callback function called with each response chunk during
            streaming. Signature: `(chunk: GenerateResponseChunkWrapper) -> None`
        use: Middleware to apply to this generation request.
        context: Additional context data passed to tools and sub-actions.
            Useful for passing auth info, request metadata, etc.
        step_name: Custom name for this generate call in trace views.
        metadata: Additional metadata for the generation request.

    Example:
        ```python
        # Basic override
        response = await prompt({'topic': 'AI'}, opts={'config': {'temperature': 0.9}})

        # Multi-turn conversation
        response = await prompt(
            {'question': 'What about safety?'},
            opts={
                'messages': previous_response.messages,  # Continue conversation
                'config': {'temperature': 0.5},
            },
        )


        # Streaming with callback
        def on_chunk(chunk):
            print(chunk.text, end='', flush=True)


        response = await prompt({'topic': 'Space'}, opts={'on_chunk': on_chunk})

        # Override model and tools
        response = await prompt(
            {'task': 'analyze'},
            opts={
                'model': 'googleai/gemini-1.5-pro',
                'tools': ['search', 'calculator'],
                'tool_choice': 'auto',
            },
        )
        ```

    Caveats:
        - Cannot override 'prompt' or 'system' (defined by the prompt itself)
        - Config is MERGED, not replaced - to clear a config value, set it explicitly
        - Message handling depends on whether prompt defines a messages template
    """

    model: str | None
    config: GenerationCommonConfig | dict[str, Any] | None
    messages: list[Message] | None
    docs: list[DocumentData] | None
    tools: list[str] | None
    resources: list[str] | None
    tool_choice: ToolChoice | None
    output: OutputOptions | None
    resume: ResumeOptions | None
    return_tool_requests: bool | None
    max_turns: int | None
    on_chunk: ModelStreamingCallback | None
    use: list[ModelMiddleware] | None
    context: dict[str, Any] | None
    step_name: str | None
    metadata: dict[str, Any] | None


class GenerateStreamResponse(Generic[OutputT]):
    r"""Response from a streaming prompt execution.

    This class provides a consistent interface matching the JavaScript
    GenerateStreamResponse, with both stream and response properties
    accessible simultaneously.

    When the prompt has a typed output schema, `response` returns
    `GenerateResponseWrapper[OutputT]` with typed `.output` property.

    Overview:
        When you call `prompt.stream()`, you get a GenerateStreamResponse
        that allows you to:
        1. Iterate over response chunks as they arrive (via `stream`)
        2. Await the complete response when streaming finishes (via `response`)

        This enables real-time UIs that show text as it's generated while
        still having access to the complete response for logging, analysis,
        or error handling.

        ┌─────────────────────────────────────────────────────────────────────┐
        │ Stream vs Response                                                  │
        ├─────────────────────┬───────────────────────────────────────────────┤
        │ stream              │ AsyncIterable - yields chunks as generated    │
        │ response            │ Awaitable - resolves to complete response     │
        └─────────────────────┴───────────────────────────────────────────────┘

    Attributes:
        stream: Async iterable of response chunks. Each chunk contains partial
            text and metadata as it's generated by the model.
        response: Awaitable that resolves to the complete GenerateResponseWrapper
            once streaming is finished. Contains the full text, usage stats,
            finish reason, and any tool calls.

    Example:
        Basic streaming to console:

        ```python
        result = prompt.stream({'topic': 'AI'})

        # Stream chunks to console in real-time
        async for chunk in result.stream:
            print(chunk.text, end='', flush=True)

        # Get complete response for logging
        final = await result.response
        print(f'\\nFinish reason: {final.finish_reason}')
        print(f'Token usage: {final.usage}')
        ```

        Streaming to a web response:

        ```python
        async def generate_stream(request):
            result = prompt.stream({'question': request.question})

            async def event_stream():
                async for chunk in result.stream:
                    yield f'data: {chunk.text}\\n\\n'
                yield 'data: [DONE]\\n\\n'

            return StreamingResponse(event_stream())
        ```

        Getting response without consuming stream:

        ```python
        result = prompt.stream({'topic': 'news'})

        # You can await response directly - stream is consumed internally
        final = await result.response
        print(final.text)  # Complete text
        ```

    Caveats:
        - The stream can only be consumed once
        - Awaiting `response` without consuming `stream` will still work
        - If an error occurs during streaming, it's raised when awaiting `response`

    See Also:
        - JavaScript GenerateStreamResponse: js/ai/src/generate.ts
    """

    def __init__(
        self,
        channel: Channel[GenerateResponseChunkWrapper, GenerateResponseWrapper[OutputT]],
        response_future: asyncio.Future[GenerateResponseWrapper[OutputT]],
    ) -> None:
        """Initialize the stream response.

        Args:
            channel: The channel providing response chunks. This is an async
                iterable that yields GenerateResponseChunkWrapper objects.
            response_future: Future that resolves to the complete response
                when streaming is finished.
        """
        self._channel: Channel[GenerateResponseChunkWrapper, GenerateResponseWrapper[OutputT]] = channel
        self._response_future: asyncio.Future[GenerateResponseWrapper[OutputT]] = response_future

    @property
    def stream(self) -> AsyncIterable[GenerateResponseChunkWrapper]:
        """Get the async iterable of response chunks.

        Returns:
            An async iterable that yields GenerateResponseChunkWrapper objects
            as they are received from the model. Each chunk contains:
            - text: The partial text generated so far
            - index: The chunk index
            - Additional metadata from the model
        """
        return self._channel

    @property
    def response(self) -> Awaitable[GenerateResponseWrapper[OutputT]]:
        """Get the awaitable for the complete response.

        Returns:
            An awaitable that resolves to a GenerateResponseWrapper containing:
            - text: The complete generated text
            - output: The typed output (when using Output[T])
            - messages: The full message history
            - usage: Token usage statistics
            - finish_reason: Why generation stopped (e.g., 'stop', 'length')
            - Any tool calls or interrupts from the response
        """
        return self._response_future


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
    config: GenerationCommonConfig | dict[str, Any] | None = None
    description: str | None = None
    input_schema: type | dict[str, Any] | str | None = None
    system: str | Part | list[Part] | Callable[..., Any] | None = None
    prompt: str | Part | list[Part] | Callable[..., Any] | None = None
    messages: str | list[Message] | Callable[..., Any] | None = None
    output_format: str | None = None
    output_content_type: str | None = None
    output_instructions: bool | str | None = None
    output_schema: type | dict[str, Any] | str | None = None
    output_constrained: bool | None = None
    max_turns: int | None = None
    return_tool_requests: bool | None = None
    metadata: dict[str, Any] | None = None
    tools: list[str] | None = None
    tool_choice: ToolChoice | None = None
    use: list[ModelMiddleware] | None = None
    docs: list[DocumentData] | Callable[..., Any] | None = None
    tool_responses: list[Part] | None = None
    resources: list[str] | None = None


class ExecutablePrompt(Generic[InputT, OutputT]):
    r"""A prompt that can be executed with a given input and configuration.

    This class matches the JavaScript ExecutablePrompt interface, providing
    a callable object that generates AI responses from a prompt template.

    When defined with input/output schemas via `Input[I]` and `Output[O]`,
    the prompt is typed as `ExecutablePrompt[I, O]`:
    - Input is type-checked when calling the prompt
    - Output is typed on `response.output`

    Overview:
        ExecutablePrompt is the main way to work with prompts in Genkit. It
        wraps a prompt definition (template, model, config, etc.) and provides
        methods to execute it, stream responses, or render it without execution.

        ┌─────────────────────────────────────────────────────────────────────┐
        │ ExecutablePrompt Methods                                            │
        ├─────────────────────┬───────────────────────────────────────────────┤
        │ __call__(input,opts)│ Execute prompt, return complete response      │
        │ stream(input, opts) │ Execute prompt, return streaming response     │
        │ render(input, opts) │ Render template without executing             │
        │ as_tool()           │ Convert prompt to a tool action               │
        │ ref                 │ Property with prompt name and metadata        │
        └─────────────────────┴───────────────────────────────────────────────┘

    Attributes:
        ref: A dict containing the prompt's name and metadata.

    Example:
        Basic execution:

        ```python
        # Get a prompt (from ai.define_prompt or ai.prompt)
        recipe = ai.prompt('recipe')

        # Execute with input
        response = await recipe({'food': 'pizza'})
        print(response.text)

        # Execute with options override
        response = await recipe(
            {'food': 'salad'},
            opts={
                'config': {'temperature': 0.5},
                'model': 'googleai/gemini-1.5-pro',
            },
        )
        ```

        Streaming:

        ```python
        result = recipe.stream({'food': 'soup'})

        async for chunk in result.stream:
            print(chunk.text, end='')

        final = await result.response
        print(f'\\nTokens used: {final.usage}')
        ```

        Rendering without execution:

        ```python
        # Get the GenerateActionOptions without calling the model
        options = await recipe.render({'food': 'curry'})
        print(options.messages)  # See rendered messages
        print(options.config)  # See merged config

        # Manually execute
        response = await ai.generate(options)
        ```

        Converting to a tool:

        ```python
        # Use the prompt as a tool in another prompt
        recipe_tool = await recipe.as_tool()

        response = await ai.generate(
            prompt='Suggest a healthy meal',
            tools=[recipe_tool],
        )
        ```

    Caveats:
        - Config values passed via opts are MERGED with prompt config
        - The prompt and system fields cannot be overridden at runtime
        - Lazy resolution: prompts loaded from files are resolved on first use

    See Also:
        - JavaScript ExecutablePrompt: js/ai/src/prompt.ts
        - Dotprompt: https://genkit.dev/docs/dotprompt
    """

    def __init__(
        self,
        registry: Registry,
        variant: str | None = None,
        model: str | None = None,
        config: GenerationCommonConfig | dict[str, Any] | None = None,
        description: str | None = None,
        input_schema: type | dict[str, Any] | str | None = None,
        system: str | Part | list[Part] | Callable[..., Any] | None = None,
        prompt: str | Part | list[Part] | Callable[..., Any] | None = None,
        messages: str | list[Message] | Callable[..., Any] | None = None,
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: bool | str | None = None,
        output_schema: type | dict[str, Any] | str | None = None,
        output_constrained: bool | None = None,
        max_turns: int | None = None,
        return_tool_requests: bool | None = None,
        metadata: dict[str, Any] | None = None,
        tools: list[str] | None = None,
        tool_choice: ToolChoice | None = None,
        use: list[ModelMiddleware] | None = None,
        docs: list[DocumentData] | Callable[..., Any] | None = None,
        resources: list[str] | None = None,
        _name: str | None = None,  # prompt name for action lookup
        _ns: str | None = None,  # namespace for action lookup
        _prompt_action: Action | None = None,  # reference to PROMPT action
        # TODO(#4344):
        #  docs: list[Document]):
    ) -> None:
        """Initializes an ExecutablePrompt instance.

        Args:
            registry: The registry to use for resolving models and tools.
            variant: The variant of the prompt.
            model: The model to use for generation.
            config: The generation configuration.
            description: A description of the prompt.
            input_schema: type | dict[str, Any] | str | None = None,
            system: str | Part | list[Part] | Callable | None = None,
            prompt: str | Part | list[Part] | Callable | None = None,
            messages: str | list[Message] | Callable | None = None,
            output_format: str | None = None,
            output_content_type: str | None = None,
            output_instructions: Instructions for formatting the output.
            output_schema: type | dict[str, Any] | str | None = None,
            output_constrained: Whether the output should be constrained to the output schema.
            max_turns: The maximum number of turns in a conversation.
            return_tool_requests: Whether to return tool requests.
            metadata: Metadata to associate with the prompt.
            tools: A list of tool names to use with the prompt.
            tool_choice: The tool choice strategy.
            use: A list of model middlewares to apply.
            docs: A list of documents to be used for grounding.
            resources: A list of resource URIs to be used for grounding.
        """
        self._registry: Registry = registry
        self._variant: str | None = variant
        self._model: str | None = model
        self._config: GenerationCommonConfig | dict[str, Any] | None = config
        self._description: str | None = description
        self._input_schema: type | dict[str, Any] | str | None = input_schema
        self._system: str | Part | list[Part] | Callable[..., Any] | None = system
        self._prompt: str | Part | list[Part] | Callable[..., Any] | None = prompt
        self._messages: str | list[Message] | Callable[..., Any] | None = messages
        self._output_format: str | None = output_format
        self._output_content_type: str | None = output_content_type
        self._output_instructions: bool | str | None = output_instructions
        self._output_schema: type | dict[str, Any] | str | None = output_schema
        self._output_constrained: bool | None = output_constrained
        self._max_turns: int | None = max_turns
        self._return_tool_requests: bool | None = return_tool_requests
        self._metadata: dict[str, Any] | None = metadata
        self._tools: list[str] | None = tools
        self._tool_choice: ToolChoice | None = tool_choice
        self._use: list[ModelMiddleware] | None = use
        self._docs: list[DocumentData] | Callable[..., Any] | None = docs
        self._resources: list[str] | None = resources
        self._cache_prompt: PromptCache = PromptCache()
        self._name: str | None = _name  # Store name/ns for action lookup (used by as_tool())
        self._ns: str | None = _ns
        self._prompt_action: Action | None = _prompt_action

    @property
    def ref(self) -> dict[str, Any]:
        """Returns a reference object for this prompt.

        The reference object contains the prompt's name and metadata.
        """
        return {
            'name': registry_definition_key(self._name, self._variant, self._ns) if self._name else None,
            'metadata': self._metadata,
        }

    async def _ensure_resolved(self) -> None:
        """Ensures the prompt is resolved from the registry if only a name was provided."""
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
        input: InputT | None = None,
        opts: PromptGenerateOptions | None = None,
    ) -> GenerateResponseWrapper[OutputT]:
        """Executes the prompt with the given input and configuration.

        This method matches the JavaScript ExecutablePrompt callable interface,
        accepting an optional `opts` parameter that can override the prompt's
        default configuration.

        Args:
            input: The input to the prompt template. When the prompt is defined
                with `input=Input(schema=T)`, this should be an instance of T.
            opts: Optional generation options to override prompt defaults.
                Can include: model, config, messages, docs, tools, output,
                tool_choice, return_tool_requests, max_turns, on_chunk,
                use (middleware), context, resume, and metadata.

        Returns:
            The generated response with typed output.

        Example:
            ```python
            # With typed input/output
            class RecipeInput(BaseModel):
                dish: str


            prompt = ai.define_prompt(
                name='recipe',
                input=Input(schema=RecipeInput),
                output=Output(schema=Recipe),
                prompt='Create a recipe for {dish}',
            )

            response = await prompt(RecipeInput(dish='pizza'))
            response.output.name  # Typed!
            ```
        """
        await self._ensure_resolved()
        effective_opts: PromptGenerateOptions = opts if opts else {}

        # Extract streaming callback and middleware from opts
        on_chunk = effective_opts.get('on_chunk')
        middleware = effective_opts.get('use') or self._use
        context = effective_opts.get('context')

        result = await generate_action(
            self._registry,
            await self.render(input=input, opts=effective_opts),
            on_chunk=on_chunk,
            middleware=middleware,
            context=context if context else ActionRunContext._current_context(),  # pyright: ignore[reportPrivateUsage]
        )
        # Cast to preserve the generic type parameter
        return cast(GenerateResponseWrapper[OutputT], result)

    def stream(
        self,
        input: InputT | None = None,
        opts: PromptGenerateOptions | None = None,
        *,
        timeout: float | None = None,
    ) -> GenerateStreamResponse[OutputT]:
        r"""Streams the prompt execution with the given input and configuration.

        This method matches the JavaScript ExecutablePrompt.stream() interface,
        returning a GenerateStreamResponse with both stream and response properties.

        Args:
            input: The input to the prompt template. When the prompt is defined
                with `input=Input(schema=T)`, this should be an instance of T.
            opts: Optional generation options to override prompt defaults.
                Can include: model, config, messages, docs, tools, output,
                tool_choice, return_tool_requests, max_turns, use (middleware),
                context, resume, and metadata.
            timeout: Optional timeout in seconds for the streaming operation.

        Returns:
            A GenerateStreamResponse with:
            - stream: AsyncIterable of response chunks
            - response: Awaitable that resolves to the typed complete response

        Example:
            ```python
            prompt = ai.define_prompt(
                name='story',
                input=Input(schema=StoryInput),
                output=Output(schema=Story),
                prompt='Write a story about {topic}',
            )

            # Stream the response
            result = prompt.stream(StoryInput(topic='adventure'))
            async for chunk in result.stream:
                print(chunk.text, end='')

            # Get the final typed response
            final = await result.response
            print(f'Title: {final.output.title}')
            ```
        """
        effective_opts: PromptGenerateOptions = opts if opts else {}
        channel: Channel[GenerateResponseChunkWrapper, GenerateResponseWrapper[OutputT]] = Channel(timeout=timeout)

        # Create a copy of opts with the streaming callback
        stream_opts: PromptGenerateOptions = {**effective_opts, 'on_chunk': lambda c: channel.send(c)}

        resp = self.__call__(input=input, opts=stream_opts)
        response_future: asyncio.Future[GenerateResponseWrapper[OutputT]] = asyncio.create_task(resp)
        channel.set_close_future(response_future)

        return GenerateStreamResponse[OutputT](channel=channel, response_future=response_future)

    async def render(
        self,
        input: InputT | dict[str, Any] | None = None,
        opts: PromptGenerateOptions | None = None,
    ) -> GenerateActionOptions:
        """Renders the prompt template with the given input and options.

        This method matches the JavaScript ExecutablePrompt.render() interface,
        accepting an optional `opts` parameter that can override the prompt's
        default configuration.

        Args:
            input: The input to the prompt template. Can be a typed input model
                or a dict with template variables.
            opts: Optional generation options to override prompt defaults.
                Can include: model, config, messages, docs, tools, output,
                tool_choice, return_tool_requests, max_turns, context,
                resume, and metadata.

        Returns:
            The rendered prompt as a GenerateActionOptions object, ready to
            be passed to generate().

        Example:
            ```python
            prompt = ai.define_prompt(
                name='recipe',
                input=Input(schema=RecipeInput),
                prompt='Create a recipe for {dish}',
            )

            # Render without executing
            options = await prompt.render(RecipeInput(dish='pizza'))

            # Then generate manually
            response = await ai.generate(options)
            ```
        """
        await self._ensure_resolved()
        effective_opts: PromptGenerateOptions = opts if opts else {}

        # Extract context from opts
        context = effective_opts.get('context')

        # Extract output options from opts
        output_opts = effective_opts.get('output') or {}

        # Merge config: opts.config is MERGED with prompt config (JS behavior)
        # This allows overriding specific config values while keeping others
        opts_config_value = effective_opts.get('config')
        if opts_config_value is not None:
            prompt_config = self._config or {}
            opts_config = opts_config_value or {}
            # Convert Pydantic models to dicts for merging
            if isinstance(prompt_config, BaseModel):
                prompt_config = prompt_config.model_dump(exclude_none=True)
            if isinstance(opts_config, BaseModel):
                opts_config = opts_config.model_dump(exclude_none=True)
            merged_config = (
                {
                    **(prompt_config if isinstance(prompt_config, dict) else {}),
                    **(opts_config if isinstance(opts_config, dict) else {}),
                }
                if prompt_config or opts_config
                else None
            )
        else:
            merged_config = self._config

        # Merge model: opts.model overrides prompt model
        merged_model = effective_opts.get('model') or self._model

        # Merge tools: opts.tools overrides prompt tools
        merged_tools = effective_opts.get('tools') or self._tools

        # Merge output options: opts.output overrides prompt output settings
        merged_output_format = output_opts.get('format') or self._output_format
        merged_output_content_type = output_opts.get('content_type') or self._output_content_type
        merged_output_schema = output_opts.get('schema') or output_opts.get('json_schema') or self._output_schema
        merged_output_constrained = (
            output_opts.get('constrained') if output_opts.get('constrained') is not None else self._output_constrained
        )
        merged_output_instructions = (
            output_opts.get('instructions')
            if output_opts.get('instructions') is not None
            else self._output_instructions
        )

        # Merge other options (opts values override prompt values)
        merged_tool_choice = effective_opts.get('tool_choice') or self._tool_choice
        merged_return_tool_requests = (
            effective_opts.get('return_tool_requests')
            if effective_opts.get('return_tool_requests') is not None
            else self._return_tool_requests
        )
        merged_max_turns = (
            effective_opts.get('max_turns') if effective_opts.get('max_turns') is not None else self._max_turns
        )
        merged_metadata = (
            {**(self._metadata or {}), **(effective_opts.get('metadata') or {})}
            if effective_opts.get('metadata')
            else self._metadata
        )

        # Build the merged PromptConfig
        prompt_options = PromptConfig(
            model=merged_model,
            prompt=self._prompt,
            system=self._system,
            messages=self._messages,
            tools=merged_tools,
            return_tool_requests=merged_return_tool_requests,
            tool_choice=merged_tool_choice,
            config=merged_config,
            max_turns=merged_max_turns,
            output_format=merged_output_format,
            output_content_type=merged_output_content_type,
            output_instructions=merged_output_instructions,
            output_schema=merged_output_schema,
            output_constrained=merged_output_constrained,
            input_schema=self._input_schema,
            metadata=merged_metadata,
            docs=self._docs,
            resources=effective_opts.get('resources') or self._resources,
        )

        model = prompt_options.model or self._registry.default_model
        if model is None:
            raise GenkitError(status='INVALID_ARGUMENT', message='No model configured.')

        resolved_msgs: list[Message] = []
        # Convert input to dict for render functions
        # If input is a Pydantic model, convert to dict; otherwise use as-is
        render_input: dict[str, Any]
        if input is None:
            render_input = {}
        elif isinstance(input, dict):
            # Type narrow: input is dict here, assign to dict[str, Any] typed variable
            render_input = {str(k): v for k, v in input.items()}
        elif isinstance(input, BaseModel):
            # Pydantic v2 model
            render_input = input.model_dump()
        elif hasattr(input, 'dict'):
            # Pydantic v1 model
            dict_func = getattr(input, 'dict', None)
            render_input = cast(Callable[[], dict[str, Any]], dict_func)()
        else:
            # Fallback: cast to dict (should not happen with proper typing)
            render_input = cast(dict[str, Any], input)
        # Get opts.messages for history (matching JS behavior)
        opts_messages = effective_opts.get('messages')

        # Render system prompt
        if prompt_options.system:
            result = await render_system_prompt(
                self._registry, render_input, prompt_options, self._cache_prompt, context
            )
            resolved_msgs.append(result)

        # Handle messages (matching JS behavior):
        # - If prompt has messages template: render it (opts.messages passed as history to resolvers)
        # - If prompt has no messages: use opts.messages directly
        if prompt_options.messages:
            # Prompt defines messages - render them (resolvers receive opts_messages as history)
            resolved_msgs.extend(
                await render_message_prompt(
                    self._registry,
                    render_input,
                    prompt_options,
                    self._cache_prompt,
                    context,
                    history=opts_messages,
                )
            )
        elif opts_messages:
            # Prompt has no messages template - use opts.messages directly
            resolved_msgs.extend(opts_messages)

        # Render user prompt
        if prompt_options.prompt:
            result = await render_user_prompt(self._registry, render_input, prompt_options, self._cache_prompt, context)
            resolved_msgs.append(result)

        # If schema is set but format is not explicitly set, default to 'json' format
        if prompt_options.output_schema and not prompt_options.output_format:
            output_format = 'json'
        else:
            output_format = prompt_options.output_format

        # Build output config
        output = GenerateActionOutputConfig()
        if output_format:
            output.format = output_format
        if prompt_options.output_content_type:
            output.content_type = prompt_options.output_content_type
        if prompt_options.output_instructions is not None:
            output.instructions = prompt_options.output_instructions
        _resolve_output_schema(self._registry, prompt_options.output_schema, output)
        if prompt_options.output_constrained is not None:
            output.constrained = prompt_options.output_constrained

        # Handle resume options
        resume = None
        resume_opts = effective_opts.get('resume')
        if resume_opts:
            respond = resume_opts.get('respond')
            if respond:
                resume = Resume(respond=respond) if isinstance(respond, list) else Resume(respond=[respond])

        # Merge docs: opts.docs extends prompt docs
        merged_docs = await render_docs(render_input, prompt_options, context)
        opts_docs = effective_opts.get('docs')
        if opts_docs:
            merged_docs = [*merged_docs, *opts_docs] if merged_docs else list(opts_docs)

        return GenerateActionOptions(
            model=model,
            messages=resolved_msgs,
            config=prompt_options.config,
            tools=prompt_options.tools,
            return_tool_requests=prompt_options.return_tool_requests,
            tool_choice=prompt_options.tool_choice,
            output=output,
            max_turns=prompt_options.max_turns,
            docs=merged_docs,
            resume=resume,
        )

    async def as_tool(self) -> Action:
        """Expose this prompt as a tool.

        Returns the PROMPT action, which can be used as a tool.
        """
        await self._ensure_resolved()
        # If we have a direct reference to the action, use it
        if self._prompt_action is not None:
            return self._prompt_action

        # Otherwise, try to look it up using name/variant/ns
        if self._name is None:
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message=(
                    'Prompt name not available. This prompt was not created via define_prompt_async() or load_prompt().'
                ),
            )

        lookup_key = registry_lookup_key(self._name, self._variant, self._ns)

        action = await self._registry.resolve_action_by_key(lookup_key)

        if action is None or action.kind != ActionKind.PROMPT:
            raise GenkitError(
                status='NOT_FOUND',
                message=f'PROMPT action not found for prompt "{self._name}"',
            )

        return action


# Overload 1: Both input and output typed -> ExecutablePrompt[InputT, OutputT]
@overload
def define_prompt(
    registry: Registry,
    name: str | None = None,
    variant: str | None = None,
    model: str | None = None,
    config: GenerationCommonConfig | dict[str, Any] | None = None,
    description: str | None = None,
    input_schema: type | dict[str, Any] | str | None = None,
    system: str | Part | list[Part] | Callable[..., Any] | None = None,
    prompt: str | Part | list[Part] | Callable[..., Any] | None = None,
    messages: str | list[Message] | Callable[..., Any] | None = None,
    output_format: str | None = None,
    output_content_type: str | None = None,
    output_instructions: bool | str | None = None,
    output_schema: type | dict[str, Any] | str | None = None,
    output_constrained: bool | None = None,
    max_turns: int | None = None,
    return_tool_requests: bool | None = None,
    metadata: dict[str, Any] | None = None,
    tools: list[str] | None = None,
    tool_choice: ToolChoice | None = None,
    use: list[ModelMiddleware] | None = None,
    docs: list[DocumentData] | Callable[..., Any] | None = None,
    *,
    input: 'Input[InputT]',
    output: 'Output[OutputT]',
) -> 'ExecutablePrompt[InputT, OutputT]': ...


# Overload 2: Only input typed -> ExecutablePrompt[InputT, Any]
@overload
def define_prompt(
    registry: Registry,
    name: str | None = None,
    variant: str | None = None,
    model: str | None = None,
    config: GenerationCommonConfig | dict[str, Any] | None = None,
    description: str | None = None,
    input_schema: type | dict[str, Any] | str | None = None,
    system: str | Part | list[Part] | Callable[..., Any] | None = None,
    prompt: str | Part | list[Part] | Callable[..., Any] | None = None,
    messages: str | list[Message] | Callable[..., Any] | None = None,
    output_format: str | None = None,
    output_content_type: str | None = None,
    output_instructions: bool | str | None = None,
    output_schema: type | dict[str, Any] | str | None = None,
    output_constrained: bool | None = None,
    max_turns: int | None = None,
    return_tool_requests: bool | None = None,
    metadata: dict[str, Any] | None = None,
    tools: list[str] | None = None,
    tool_choice: ToolChoice | None = None,
    use: list[ModelMiddleware] | None = None,
    docs: list[DocumentData] | Callable[..., Any] | None = None,
    *,
    input: 'Input[InputT]',
    output: None = None,
) -> 'ExecutablePrompt[InputT, Any]': ...


# Overload 3: Only output typed -> ExecutablePrompt[Any, OutputT]
@overload
def define_prompt(
    registry: Registry,
    name: str | None = None,
    variant: str | None = None,
    model: str | None = None,
    config: GenerationCommonConfig | dict[str, Any] | None = None,
    description: str | None = None,
    input_schema: type | dict[str, Any] | str | None = None,
    system: str | Part | list[Part] | Callable[..., Any] | None = None,
    prompt: str | Part | list[Part] | Callable[..., Any] | None = None,
    messages: str | list[Message] | Callable[..., Any] | None = None,
    output_format: str | None = None,
    output_content_type: str | None = None,
    output_instructions: bool | str | None = None,
    output_schema: type | dict[str, Any] | str | None = None,
    output_constrained: bool | None = None,
    max_turns: int | None = None,
    return_tool_requests: bool | None = None,
    metadata: dict[str, Any] | None = None,
    tools: list[str] | None = None,
    tool_choice: ToolChoice | None = None,
    use: list[ModelMiddleware] | None = None,
    docs: list[DocumentData] | Callable[..., Any] | None = None,
    input: None = None,
    *,
    output: 'Output[OutputT]',
) -> 'ExecutablePrompt[Any, OutputT]': ...


# Overload 4: Neither typed -> ExecutablePrompt[Any, Any]
@overload
def define_prompt(
    registry: Registry,
    name: str | None = None,
    variant: str | None = None,
    model: str | None = None,
    config: GenerationCommonConfig | dict[str, Any] | None = None,
    description: str | None = None,
    input_schema: type | dict[str, Any] | str | None = None,
    system: str | Part | list[Part] | Callable[..., Any] | None = None,
    prompt: str | Part | list[Part] | Callable[..., Any] | None = None,
    messages: str | list[Message] | Callable[..., Any] | None = None,
    output_format: str | None = None,
    output_content_type: str | None = None,
    output_instructions: bool | str | None = None,
    output_schema: type | dict[str, Any] | str | None = None,
    output_constrained: bool | None = None,
    max_turns: int | None = None,
    return_tool_requests: bool | None = None,
    metadata: dict[str, Any] | None = None,
    tools: list[str] | None = None,
    tool_choice: ToolChoice | None = None,
    use: list[ModelMiddleware] | None = None,
    docs: list[DocumentData] | Callable[..., Any] | None = None,
    input: None = None,
    output: None = None,
) -> 'ExecutablePrompt[Any, Any]': ...


# Implementation
def define_prompt(
    registry: Registry,
    name: str | None = None,
    variant: str | None = None,
    model: str | None = None,
    config: GenerationCommonConfig | dict[str, Any] | None = None,
    description: str | None = None,
    input_schema: type | dict[str, Any] | str | None = None,
    system: str | Part | list[Part] | Callable[..., Any] | None = None,
    prompt: str | Part | list[Part] | Callable[..., Any] | None = None,
    messages: str | list[Message] | Callable[..., Any] | None = None,
    output_format: str | None = None,
    output_content_type: str | None = None,
    output_instructions: bool | str | None = None,
    output_schema: type | dict[str, Any] | str | None = None,
    output_constrained: bool | None = None,
    max_turns: int | None = None,
    return_tool_requests: bool | None = None,
    metadata: dict[str, Any] | None = None,
    tools: list[str] | None = None,
    tool_choice: ToolChoice | None = None,
    use: list[ModelMiddleware] | None = None,
    docs: list[DocumentData] | Callable[..., Any] | None = None,
    input: 'Input[Any] | None' = None,
    output: 'Output[Any] | None' = None,
) -> 'ExecutablePrompt[Any, Any]':
    """Defines an executable prompt.

    Args:
        registry: The registry to use for resolving models and tools.
        name: The name of the prompt.
        variant: The variant of the prompt.
        model: The model to use for generation.
        config: The generation configuration.
        description: A description of the prompt.
        input_schema: The input schema for the prompt.
        system: The system message for the prompt.
        prompt: The user prompt.
        messages: A list of messages to include in the prompt.
        output_format: The output format.
        output_content_type: The output content type.
        output_instructions: Instructions for formatting the output.
        output_schema: The output schema (use `output` parameter for typed outputs).
        output_constrained: Whether the output should be constrained to the output schema.
        max_turns: The maximum number of turns in a conversation.
        return_tool_requests: Whether to return tool requests.
        metadata: Metadata to associate with the prompt.
        tools: A list of tool names to use with the prompt.
        tool_choice: The tool choice strategy.
        use: A list of model middlewares to apply.
        docs: A list of documents to be used for grounding.
        input: Typed input configuration using Input[T]. When provided, the
            prompt's input parameter is type-checked.
        output: Typed output configuration using Output[T]. When provided, the
            response output is typed.

    Returns:
        An ExecutablePrompt instance. When both `input=Input(schema=I)` and
        `output=Output(schema=O)` are provided, returns `ExecutablePrompt[I, O]`
        with typed input and output.

    Example:
        ```python
        from genkit import Input, Output
        from pydantic import BaseModel


        class RecipeInput(BaseModel):
            dish: str


        class Recipe(BaseModel):
            name: str
            ingredients: list[str]


        # With typed input AND output
        recipe_prompt = define_prompt(
            registry,
            name='recipe',
            prompt='Create a recipe for {dish}',
            input=Input(schema=RecipeInput),
            output=Output(schema=Recipe),
        )

        # Input is type-checked!
        response = await recipe_prompt(RecipeInput(dish='pizza'))
        response.output.name  # ✓ Typed as str
        ```
    """
    # If Input[T] is provided, extract its schema
    effective_input_schema = input_schema
    if input is not None:
        effective_input_schema = input.schema

    # If Output[T] is provided, extract its configuration
    effective_output_schema = output_schema
    effective_output_format = output_format
    effective_output_content_type = output_content_type
    effective_output_instructions = output_instructions
    effective_output_constrained = output_constrained

    if output is not None:
        effective_output_schema = output.schema
        effective_output_format = output.format if output.format else output_format
        if output.content_type is not None:
            effective_output_content_type = output.content_type
        if output.instructions is not None:
            effective_output_instructions = output.instructions
        if output.constrained is not None:
            effective_output_constrained = output.constrained

    executable_prompt: ExecutablePrompt[Any, Any] = ExecutablePrompt(
        registry,
        variant=variant,
        model=model,
        config=config,
        description=description,
        input_schema=effective_input_schema,
        system=system,
        prompt=prompt,
        messages=messages,
        output_format=effective_output_format,
        output_content_type=effective_output_content_type,
        output_instructions=effective_output_instructions,
        output_schema=effective_output_schema,
        output_constrained=effective_output_constrained,
        max_turns=max_turns,
        return_tool_requests=return_tool_requests,
        metadata=metadata,
        tools=tools,
        tool_choice=tool_choice,
        use=use,
        docs=docs,
        _name=name,
    )

    if name:
        # Register actions for this prompt
        action_metadata: dict[str, object] = {
            'type': 'prompt',
            'source': 'programmatic',
            'prompt': {
                'name': name,
                'variant': variant or '',
            },
        }

        async def prompt_action_fn(input: Any = None) -> GenerateRequest:  # noqa: ANN401
            """PROMPT action function - renders prompt and returns GenerateRequest."""
            options = await executable_prompt.render(input=input)
            return await to_generate_request(registry, options)

        async def executable_prompt_action_fn(input: Any = None) -> GenerateActionOptions:  # noqa: ANN401
            """EXECUTABLE_PROMPT action function - renders prompt and returns GenerateActionOptions."""
            return await executable_prompt.render(input=input)

        action_name = registry_definition_key(name, variant)
        prompt_action = registry.register_action(
            kind=cast(ActionKind, ActionKind.PROMPT),
            name=action_name,
            fn=prompt_action_fn,
            metadata=action_metadata,
        )

        executable_prompt_action = registry.register_action(
            kind=cast(ActionKind, ActionKind.EXECUTABLE_PROMPT),
            name=action_name,
            fn=executable_prompt_action_fn,
            metadata=action_metadata,
        )

        # Link them
        executable_prompt._prompt_action = prompt_action  # pyright: ignore[reportPrivateUsage]
        # Dynamic attributes set at runtime - these are custom attrs added to Action objects
        setattr(prompt_action, '_executable_prompt', weakref.ref(executable_prompt))  # noqa: B010
        setattr(executable_prompt_action, '_executable_prompt', weakref.ref(executable_prompt))  # noqa: B010

    return executable_prompt


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


async def to_generate_action_options(registry: Registry, options: PromptConfig) -> GenerateActionOptions:
    """Converts the given parameters to a GenerateActionOptions object.

    Args:
        registry: The registry to use for resolving models and tools.
        options: The prompt configuration.

    Returns:
        A GenerateActionOptions object.
    """
    model = options.model or registry.default_model
    if model is None:
        raise GenkitError(status='INVALID_ARGUMENT', message='No model configured.')

    cache = PromptCache()
    resolved_msgs: list[Message] = []
    # Use empty dict instead of None for render functions
    render_input: dict[str, Any] = {}
    if options.system:
        result = await render_system_prompt(registry, render_input, options, cache)
        resolved_msgs.append(result)
    if options.messages:
        resolved_msgs.extend(await render_message_prompt(registry, render_input, options, cache))
    if options.prompt:
        result = await render_user_prompt(registry, render_input, options, cache)
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

    resume = None
    if options.tool_responses:
        # Filter for only ToolResponsePart instances
        tool_response_parts = [r.root for r in options.tool_responses if isinstance(r.root, ToolResponsePart)]
        if tool_response_parts:
            resume = Resume(respond=tool_response_parts)

    return GenerateActionOptions(
        model=model,
        messages=resolved_msgs,
        config=options.config,
        tools=options.tools,
        return_tool_requests=options.return_tool_requests,
        tool_choice=options.tool_choice,
        output=output,
        max_turns=options.max_turns,
        docs=await render_docs(render_input, options),
        resume=resume,
    )


async def to_generate_request(registry: Registry, options: GenerateActionOptions) -> GenerateRequest:
    """Converts GenerateActionOptions to a GenerateRequest.

    This function resolves tool names into their respective tool definitions
    by looking them up in the provided registry. it also validates that the
    provided options contain at least one message.

    Args:
        registry: The Registry instance used to look up tool actions.
        options: The GenerateActionOptions containing the configuration,
            messages, and tool references to be converted.

    Returns:
        A GenerateRequest object populated with messages, config, resolved
        tools, and output configurations.

    Raises:
        Exception: If a tool name provided in options cannot be found in
            the registry.
        GenkitError: If the options do not contain any messages.
    """
    tools: list[Action] = []
    if options.tools:
        for tool_name in options.tools:
            tool_action = await registry.resolve_action(cast(ActionKind, ActionKind.TOOL), tool_name)
            if tool_action is None:
                raise GenkitError(status='NOT_FOUND', message=f'Unable to resolve tool {tool_name}')
            tools.append(tool_action)

    tool_defs = [to_tool_definition(tool) for tool in tools] if tools else []

    if not options.messages:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message='at least one message is required in generate request',
        )

    return GenerateRequest(
        messages=options.messages,
        config=options.config if options.config is not None else {},
        docs=options.docs,
        tools=tool_defs,
        tool_choice=options.tool_choice,
        output=OutputConfig(
            content_type=options.output.content_type if options.output else None,
            format=options.output.format if options.output else None,
            schema=options.output.json_schema if options.output else None,
            constrained=options.output.constrained if options.output else None,
        ),
    )


def _normalize_prompt_arg(
    prompt: str | Part | list[Part] | None,
) -> list[Part]:
    """Normalizes different prompt input types into a list of Parts.

    Ensures that the prompt content, whether provided as a string, a single Part,
    or a list of Parts, is consistently represented as a list of Part objects.

    Args:
        prompt: The prompt input, which can be a string, a Part, a list of Parts,
            or None.

    Returns:
        A list containing the normalized Part(s). Returns empty list if input is None
        or empty.
    """
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


async def render_system_prompt(
    registry: Registry,
    input: dict[str, Any],
    options: PromptConfig,
    prompt_cache: PromptCache,
    context: dict[str, Any] | None = None,
) -> Message:
    """Renders the system prompt for a prompt action.

    This function handles rendering system prompts by either:
    1. Processing dotprompt templates if the system prompt is a string
    2. Normalizing the system prompt into a list of Parts if it's a Part or list of Parts

    Args:
        registry: Registry instance for resolving models and tools
        input: Dictionary of input values for template rendering
        options: Configuration options for the prompt
        prompt_cache: Cache for compiled prompt templates
        context: Optional dictionary of context values for template rendering

    Returns:
        Message: A Message object containing the rendered system prompt with Role.SYSTEM

    """
    if isinstance(options.system, str):
        if prompt_cache.system is None:
            prompt_cache.system = await registry.dotprompt.compile(options.system)

        if options.metadata:
            context = {**(context or {}), 'state': options.metadata.get('state')}

        # Cast to list[Part] - Pydantic coerces dicts to Part objects at runtime
        rendered_parts = cast(
            list[Part],
            await render_dotprompt_to_parts(
                context or {},
                prompt_cache.system,
                input,
                PromptMetadata(
                    input=PromptInputConfig(
                        schema=to_json_schema(options.input_schema) if options.input_schema else None,
                    )
                ),
            ),
        )
        return Message(role=Role.SYSTEM, content=rendered_parts)

    if callable(options.system):
        resolved = await ensure_async(options.system)(input, context)
        return Message(role=Role.SYSTEM, content=_normalize_prompt_arg(resolved))

    return Message(role=Role.SYSTEM, content=_normalize_prompt_arg(options.system))


async def render_dotprompt_to_parts(
    context: dict[str, Any],
    prompt_function: PromptFunction[Any],
    input_: dict[str, Any],
    options: PromptMetadata[Any] | None = None,
) -> list[dict[str, Any]]:
    """Renders a prompt template into a list of content parts using dotprompt.

    Args:
        context: Dictionary containing context variables available to the prompt template.
        prompt_function: The compiled dotprompt function to execute.
        input_: Dictionary containing input variables for the prompt template.
        options: Optional prompt metadata configuration.

    Returns:
        A list of dictionaries representing Part objects for Pydantic re-validation.

    Raises:
        Exception: If the template produces more than one message.
    """
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
    """Render a message prompt using a given registry, input data, options, and a context.

    This function processes different types of message options (string or list) to render
    appropriate messages using a prompt registry and cache. If the `messages` option is of type
    string, the function compiles the dotprompt messages from the `registry` and applies data
    and metadata context. If the `messages` option is of type list, it either validates and
    returns the list or processes it for message rendering. The function ensures correct message
    output using the provided input, prompt configuration, and caching mechanism.

    Arguments:
        registry (Registry): The registry used to compile dotprompt messages.
        input (dict[str, Any]): The input data to render messages.
        options (PromptConfig): Configuration containing prompt options and message settings.
        prompt_cache (PromptCache): Cache to store compiled prompt results.
        context (dict[str, Any] | None): Optional additional context to be used for rendering.
            Defaults to None.
        history (list[Message] | None): Optional conversation history to be passed to message
            resolvers/templates. Matches JS behavior where opts.messages is passed as history.

    Returns:
        list[Message]: A list of rendered or validated message objects.
    """
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
                messages=messages_,  # pyright: ignore[reportArgumentType]
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

    elif callable(options.messages):
        # Pass history to resolver function (matching JS MessagesResolver signature)
        resolved = await ensure_async(options.messages)(input, {'context': context, 'history': history})
        return list(resolved) if resolved else []

    raise TypeError(f'Unsupported type for messages: {type(options.messages)}')


async def render_user_prompt(
    registry: Registry,
    input: dict[str, Any],
    options: PromptConfig,
    prompt_cache: PromptCache,
    context: dict[str, Any] | None = None,
) -> Message:
    """Asynchronously renders a user prompt based on the given input, context, and options.

    Utilizes a pre-compiled or dynamically compiled dotprompt template.

    Arguments:
        registry: The registry instance used to compile dotprompt templates.
        input: The input data used to populate the prompt.
        options: The configuration for rendering the prompt, including
            the template type and associated metadata.
        prompt_cache: A cache that stores pre-compiled prompt templates to
            optimize rendering.
        context: Optional dynamic context data to override or
            supplement in the rendering process.

    Returns:
        Message: A Message instance containing the rendered user prompt.
    """
    if isinstance(options.prompt, str):
        if prompt_cache.user_prompt is None:
            prompt_cache.user_prompt = await registry.dotprompt.compile(options.prompt)

        if options.metadata:
            context = {**(context or {}), 'state': options.metadata.get('state')}

        # Cast to list[Part] - Pydantic coerces dicts to Part objects at runtime
        rendered_parts = cast(
            list[Part],
            await render_dotprompt_to_parts(
                context or {},
                prompt_cache.user_prompt,
                input,
                PromptMetadata(
                    input=PromptInputConfig(
                        schema=to_json_schema(options.input_schema) if options.input_schema else None,
                    )
                ),
            ),
        )
        return Message(role=Role.USER, content=rendered_parts)

    if callable(options.prompt):
        resolved = await ensure_async(options.prompt)(input, context)
        return Message(role=Role.USER, content=_normalize_prompt_arg(resolved))

    return Message(role=Role.USER, content=_normalize_prompt_arg(options.prompt))


async def render_docs(
    input: dict[str, Any],
    options: PromptConfig,
    context: dict[str, Any] | None = None,
) -> list[DocumentData] | None:
    """Renders the docs for a prompt action.

    Args:
        input: Dictionary of input values.
        options: Configuration options for the prompt.
        context: Optional dictionary of context values.

    Returns:
        A list of DocumentData objects or None.
    """
    if options.docs is None:
        return None

    if callable(options.docs):
        return await ensure_async(options.docs)(input, context)

    return options.docs


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
        from genkit.blocks.prompt import define_schema

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


def load_prompt(registry: Registry, path: Path, filename: str, prefix: str = '', ns: str = '') -> None:
    """Load a single prompt file and register it in the registry.

    This function loads a .prompt file, parses it, and registers it as a lazy-loaded
    prompt that will only be fully loaded when first accessed.

    Args:
        registry: The registry to register the prompt in.
        path: Base path to the prompts directory.
        filename: Name of the prompt file (e.g., "myPrompt.prompt" or "myPrompt.variant.prompt").
        prefix: Subdirectory prefix (for namespacing).
        ns: Namespace for the prompt.
    """
    # Extract name and variant from filename
    # "myPrompt.prompt" -> name="myPrompt", variant=None
    # "myPrompt.variant.prompt" -> name="myPrompt", variant="variant"
    # "subdir/myPrompt.prompt" -> name="subdir/myPrompt", variant=None
    if not filename.endswith('.prompt'):
        raise ValueError(f"Invalid prompt filename: {filename}. Must end with '.prompt'")

    base_name = filename.removesuffix('.prompt')

    name = f'{prefix}{base_name}' if prefix else base_name
    variant: str | None = None

    # Extract variant (only takes parts[1], not all remaining parts)
    if '.' in name:
        parts = name.split('.')
        name = parts[0]
        variant = parts[1]  # Only first part after split

    # Build full file path
    # prefix may have trailing slash, so we need to handle it
    if prefix:
        # Strip trailing slash for path construction (pathlib handles it)
        prefix_clean = prefix.rstrip('/')
        file_path = path / prefix_clean / filename
    else:
        file_path = path / filename

    # Read the prompt file
    with Path(file_path).open(encoding='utf-8') as f:
        source = f.read()

    # Parse the prompt
    parsed_prompt = registry.dotprompt.parse(source)

    # Generate registry key
    registry_key = registry_definition_key(name, variant, ns)

    # Create a lazy-loaded prompt definition
    # The prompt will only be fully loaded when first accessed
    async def load_prompt_metadata() -> dict[str, Any]:
        """Lazy loader for prompt metadata."""
        prompt_metadata = await registry.dotprompt.render_metadata(parsed_prompt)

        # Convert Pydantic model to dict if needed
        prompt_metadata_dict: dict[str, Any]
        if hasattr(prompt_metadata, 'model_dump'):
            prompt_metadata_dict = prompt_metadata.model_dump(by_alias=True)
        elif hasattr(prompt_metadata, 'dict'):
            # Fallback for older Pydantic versions
            # pyrefly: ignore[deprecated] - Intentional for Pydantic v1 compatibility
            prompt_metadata_dict = prompt_metadata.dict(by_alias=True)  # pyright: ignore[reportDeprecated]
        else:
            # Already a dict - cast through object to satisfy type checker
            prompt_metadata_dict = cast(dict[str, Any], cast(object, prompt_metadata))

        # Ensure raw metadata is available (critical for lazy schema resolution)
        if hasattr(prompt_metadata, 'raw'):
            prompt_metadata_dict['raw'] = prompt_metadata.raw

        if variant:
            prompt_metadata_dict['variant'] = variant

        # Fallback for model if not present (Dotprompt issue)
        if not prompt_metadata_dict.get('model'):
            raw_model = (prompt_metadata_dict.get('raw') or {}).get('model')
            if raw_model:
                prompt_metadata_dict['model'] = raw_model

        # Clean up null descriptions
        output = prompt_metadata_dict.get('output')
        schema = None
        if output and isinstance(output, dict):
            schema = output.get('schema')
            if schema and isinstance(schema, dict) and schema.get('description') is None:
                schema.pop('description', None)

        if not schema:
            # Fallback to raw schema name if schema definition is missing
            raw_schema = (prompt_metadata_dict.get('raw') or {}).get('output', {}).get('schema')
            if isinstance(raw_schema, str):
                schema = raw_schema
                # output might be None if it wasn't in parsed config
                if not output:
                    output = {'schema': schema}
                    prompt_metadata_dict['output'] = output
                elif isinstance(output, dict):
                    output['schema'] = schema

        input_schema = prompt_metadata_dict.get('input')
        if input_schema and isinstance(input_schema, dict):
            schema = input_schema.get('schema')
            if schema and isinstance(schema, dict) and schema.get('description') is None:
                schema.pop('description', None)

        # Build metadata structure (prompt_metadata_dict is always dict[str, Any] at this point)
        metadata_inner = prompt_metadata_dict.get('metadata', {})
        base_dict = prompt_metadata_dict
        metadata = {
            **base_dict,
            **(metadata_inner if isinstance(metadata_inner, dict) else {}),
            'type': 'prompt',
            'prompt': {
                **base_dict,
                'template': parsed_prompt.template,
            },
        }

        raw = prompt_metadata_dict.get('raw')
        if raw and isinstance(raw, dict) and raw.get('metadata'):
            metadata['metadata'] = {**raw['metadata']}

        output = prompt_metadata_dict.get('output')
        input_schema = prompt_metadata_dict.get('input')
        raw = prompt_metadata_dict.get('raw')

        return {
            'name': registry_key,
            'model': prompt_metadata_dict.get('model'),
            'config': prompt_metadata_dict.get('config'),
            'tools': prompt_metadata_dict.get('tools'),
            'description': prompt_metadata_dict.get('description'),
            'output': {
                'jsonSchema': output.get('schema') if isinstance(output, dict) else None,
                'format': output.get('format') if isinstance(output, dict) else None,
            },
            'input': {
                'default': input_schema.get('default') if isinstance(input_schema, dict) else None,
                'jsonSchema': input_schema.get('schema') if isinstance(input_schema, dict) else None,
            },
            'metadata': metadata,
            'maxTurns': raw.get('maxTurns') if isinstance(raw, dict) else None,
            'toolChoice': raw.get('toolChoice') if isinstance(raw, dict) else None,
            'returnToolRequests': raw.get('returnToolRequests') if isinstance(raw, dict) else None,
            'messages': parsed_prompt.template,
        }

    # Cache the ExecutablePrompt to avoid duplicate creation when both PROMPT and
    # EXECUTABLE_PROMPT actions trigger lazy loading from the reflection API.
    _cached_prompt: ExecutablePrompt[Any, Any] | None = None

    # Create a factory function that will create the ExecutablePrompt when accessed
    # Store metadata in a closure to avoid global state
    async def create_prompt_from_file() -> ExecutablePrompt[Any, Any]:
        """Factory function to create ExecutablePrompt from file metadata (memoized)."""
        nonlocal _cached_prompt
        if _cached_prompt is not None:
            return _cached_prompt

        metadata = await load_prompt_metadata()

        # Create ExecutablePrompt from metadata
        # Pass name/ns so as_tool() can look up the action
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
            _name=name,  # Store name for action lookup
            _ns=ns,  # Store namespace for action lookup
        )

        # Store reference to PROMPT action on the ExecutablePrompt
        # Actions are already registered at this point (lazy loading happens after registration)
        definition_key = registry_definition_key(name, variant, ns)
        prompt_lookup_key = create_action_key(ActionKind.PROMPT, definition_key)
        exec_prompt_lookup_key = create_action_key(ActionKind.EXECUTABLE_PROMPT, definition_key)

        # Update PROMPT action
        prompt_action = await registry.resolve_action_by_key(prompt_lookup_key)
        if prompt_action and prompt_action.kind == ActionKind.PROMPT:
            executable_prompt._prompt_action = prompt_action  # pyright: ignore[reportPrivateUsage]
            setattr(prompt_action, '_executable_prompt', weakref.ref(executable_prompt))  # noqa: B010

        # Update EXECUTABLE_PROMPT action
        exec_prompt_action = await registry.resolve_action_by_key(exec_prompt_lookup_key)

        # Update schemas on BOTH actions for Dev UI reflection
        input_json_schema = metadata.get('input', {}).get('jsonSchema')
        output_json_schema = metadata.get('output', {}).get('jsonSchema')

        for action in [prompt_action, exec_prompt_action]:
            if action is not None:
                if input_json_schema:
                    action.input_schema = input_json_schema
                if output_json_schema:
                    action.output_schema = output_json_schema

        # Cache the result for subsequent calls
        _cached_prompt = executable_prompt
        return executable_prompt

    # Store the async factory in a way that can be accessed later
    # We'll store it in the action metadata
    action_metadata: dict[str, object] = {
        'type': 'prompt',
        'lazy': True,
        'source': 'file',
        'prompt': {
            'name': name,
            'variant': variant or '',
        },
    }

    # Create two separate action functions :
    # 1. PROMPT action - returns GenerateRequest (for rendering prompts)
    # 2. EXECUTABLE_PROMPT action - returns GenerateActionOptions (for executing prompts)

    async def prompt_action_fn(input: Any = None) -> GenerateRequest:  # noqa: ANN401
        """PROMPT action function - renders prompt and returns GenerateRequest."""
        # Load the prompt (lazy loading)
        prompt = await create_prompt_from_file()

        # Render the prompt with input to get GenerateActionOptions
        options = await prompt.render(input=input)

        # Convert GenerateActionOptions to GenerateRequest
        return await to_generate_request(registry, options)

    async def executable_prompt_action_fn(input: Any = None) -> GenerateActionOptions:  # noqa: ANN401
        """EXECUTABLE_PROMPT action function - renders prompt and returns GenerateActionOptions."""
        # Load the prompt (lazy loading)
        prompt = await create_prompt_from_file()

        # Render the prompt with input to get GenerateActionOptions
        return await prompt.render(input=input)

    # Register the PROMPT action
    # Use registry_definition_key for the action name (not registry_lookup_key)
    # The action name should be just the definition key (e.g., "dotprompt/testPrompt"),
    # not the full lookup key (e.g., "/prompt/dotprompt/testPrompt")
    action_name = registry_definition_key(name, variant, ns)
    prompt_action = registry.register_action(
        kind=cast(ActionKind, ActionKind.PROMPT),
        name=action_name,
        fn=prompt_action_fn,
        metadata=action_metadata,
    )

    # Register the EXECUTABLE_PROMPT action
    executable_prompt_action = registry.register_action(
        kind=cast(ActionKind, ActionKind.EXECUTABLE_PROMPT),
        name=action_name,
        fn=executable_prompt_action_fn,
        metadata=action_metadata,
    )

    # Store the factory function on both actions for easy access
    setattr(prompt_action, '_async_factory', create_prompt_from_file)  # noqa: B010
    setattr(executable_prompt_action, '_async_factory', create_prompt_from_file)  # noqa: B010

    # Store ExecutablePrompt reference on actions
    # This will be set when the prompt is first accessed (lazy loading)
    # We'll update it in create_prompt_from_file after the prompt is created

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
    """Look up a prompt from the registry.

    Args:
        registry: The registry to look up the prompt from.
        name: The name of the prompt.
        variant: Optional variant name.

    Returns:
        An ExecutablePrompt instance.

    Raises:
        GenkitError: If the prompt is not found.
    """
    # Try without namespace first (for programmatic prompts)
    # Use create_action_key to build the full key: "/prompt/<definition_key>"
    definition_key = registry_definition_key(name, variant, None)
    lookup_key = create_action_key(cast(ActionKind, ActionKind.PROMPT), definition_key)
    action = await registry.resolve_action_by_key(lookup_key)

    # If not found and no namespace was specified, try with default 'dotprompt' namespace
    # (for file-based prompts)
    if not action:
        definition_key = registry_definition_key(name, variant, 'dotprompt')
        lookup_key = create_action_key(cast(ActionKind, ActionKind.PROMPT), definition_key)
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
        # Fallback: try to get from metadata
        factory = action.metadata.get('_async_factory')
        if callable(factory):
            factory_async = ensure_async(cast(Callable[..., Any], factory))
            executable_prompt = await factory_async()
            if getattr(action, '_executable_prompt', None) is None:
                setattr(action, '_executable_prompt', executable_prompt)  # noqa: B010
            return executable_prompt
        # Last resort: this shouldn't happen if prompts are loaded correctly
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
    _dir: str | Path | None = None,  # Accepted but not used
) -> ExecutablePrompt[Any, Any]:
    """Look up a prompt by name and optional variant.

    Can look up prompts that were:
    1. Defined programmatically using define_prompt()
    2. Loaded from .prompt files using load_prompt_folder()

    Args:
        registry: The registry to look up the prompt from.
        name: The name of the prompt.
        variant: Optional variant name.
        dir: Optional directory parameter (accepted for compatibility but not used).

    Returns:
        An ExecutablePrompt instance.

    Raises:
        GenkitError: If the prompt is not found.
    """
    return await lookup_prompt(registry, name, variant)
