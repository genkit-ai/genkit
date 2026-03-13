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

"""User-facing asyncio API for Genkit.

This module provides the primary entry point for using Genkit in an asynchronous
environment. The `Genkit` class coordinates plugins, registry, and execution
of AI actions like generation, embedding, and retrieval.

Key features provided by the `Genkit` class:
- **Generation**: Interface for unified model interaction via `generate` and `generate_stream`.
- **Flow Control**: Execution of granular steps with tracing via `run`.
- **Dynamic Extensibility**: On-the-fly creation of tools via `dynamic_tool`.
- **Observability**: Specialized methods for managing trace context and flushing telemetry.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar, cast, overload

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider

from genkit.aio._util import ensure_async
from genkit.aio.channel import Channel
from genkit.blocks.background_model import (
    check_operation as check_operation_impl,
    lookup_background_action,
)
from genkit.blocks.document import Document
from genkit.blocks.embedding import EmbedderRef
from genkit.blocks.evaluator import EvaluatorRef
from genkit.blocks.generate import (
    StreamingCallback as ModelStreamingCallback,
    generate_action,
)
from genkit.blocks.interfaces import Output, OutputConfigDict
from genkit.blocks.model import (
    GenerateResponseChunkWrapper,
    GenerateResponseWrapper,
    ModelMiddleware,
)
from genkit.blocks.prompt import PromptConfig, load_prompt_folder, to_generate_action_options
from genkit.blocks.retriever import IndexerRef, IndexerRequest, RetrieverRef
from genkit.core.action import Action, ActionRunContext
from genkit.core.action.types import ActionKind
from genkit.core.error import GenkitError
from genkit.core.plugin import Plugin
from genkit.core.tracing import run_in_new_span
from genkit.core.typing import (
    BaseDataPoint,
    Embedding,
    EmbedRequest,
    EvalRequest,
    EvalResponse,
    Operation,
    SpanMetadata,
)
from genkit.types import (
    DocumentData,
    GenerationCommonConfig,
    Message,
    OutputConfig,
    Part,
    RetrieverRequest,
    RetrieverResponse,
    ToolChoice,
)

from ._base_async import GenkitBase
from ._server import ServerSpec

T = TypeVar('T')
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


class Genkit(GenkitBase):
    """Genkit asyncio user-facing API."""

    def __init__(
        self,
        plugins: list[Plugin] | None = None,
        model: str | None = None,
        prompt_dir: str | Path | None = None,
        reflection_server_spec: ServerSpec | None = None,
    ) -> None:
        """Initialize a new Genkit instance.

        Args:
            plugins: List of plugins to initialize.
            model: Model name to use.
            prompt_dir: Directory to automatically load prompts from.
                If not provided, defaults to loading from './prompts' if it exists.
            reflection_server_spec: Server spec for the reflection
                server.
        """
        super().__init__(plugins=plugins, model=model, reflection_server_spec=reflection_server_spec)

        load_path = prompt_dir
        if load_path is None:
            default_prompts_path = Path('./prompts')
            if default_prompts_path.is_dir():
                load_path = default_prompts_path

        if load_path:
            load_prompt_folder(self.registry, dir_path=load_path)

    def _resolve_embedder_name(self, embedder: str | EmbedderRef | None) -> str:
        """Resolve embedder name from string or EmbedderRef.

        Args:
            embedder: The embedder specified as a string name or EmbedderRef.

        Returns:
            The resolved embedder name.

        Raises:
            ValueError: If embedder is not specified or is of invalid type.
        """
        if isinstance(embedder, EmbedderRef):
            return embedder.name
        elif isinstance(embedder, str):
            return embedder
        else:
            raise ValueError('Embedder must be specified as a string name or an EmbedderRef.')

    @overload
    async def generate(
        self,
        model: str | None = None,
        prompt: str | Part | list[Part] | None = None,
        system: str | Part | list[Part] | None = None,
        messages: list[Message] | None = None,
        tools: list[str] | None = None,
        return_tool_requests: bool | None = None,
        tool_choice: ToolChoice | None = None,
        tool_responses: list[Part] | None = None,
        config: GenerationCommonConfig | dict[str, object] | None = None,
        max_turns: int | None = None,
        on_chunk: ModelStreamingCallback | None = None,
        context: dict[str, object] | None = None,
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: bool | str | None = None,
        output_constrained: bool | None = None,
        *,
        output: Output[OutputT],
        use: list[ModelMiddleware] | None = None,
        docs: list[DocumentData] | None = None,
    ) -> GenerateResponseWrapper[OutputT]: ...

    @overload
    async def generate(
        self,
        model: str | None = None,
        prompt: str | Part | list[Part] | None = None,
        system: str | Part | list[Part] | None = None,
        messages: list[Message] | None = None,
        tools: list[str] | None = None,
        return_tool_requests: bool | None = None,
        tool_choice: ToolChoice | None = None,
        tool_responses: list[Part] | None = None,
        config: GenerationCommonConfig | dict[str, object] | None = None,
        max_turns: int | None = None,
        on_chunk: ModelStreamingCallback | None = None,
        context: dict[str, object] | None = None,
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: bool | str | None = None,
        output_constrained: bool | None = None,
        output: OutputConfig | OutputConfigDict | Output[Any] | None = None,
        use: list[ModelMiddleware] | None = None,
        docs: list[DocumentData] | None = None,
    ) -> GenerateResponseWrapper[Any]: ...

    async def generate(
        self,
        model: str | None = None,
        prompt: str | Part | list[Part] | None = None,
        system: str | Part | list[Part] | None = None,
        messages: list[Message] | None = None,
        tools: list[str] | None = None,
        return_tool_requests: bool | None = None,
        tool_choice: ToolChoice | None = None,
        tool_responses: list[Part] | None = None,
        config: GenerationCommonConfig | dict[str, object] | None = None,
        max_turns: int | None = None,
        on_chunk: ModelStreamingCallback | None = None,
        context: dict[str, object] | None = None,
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: bool | str | None = None,
        output_constrained: bool | None = None,
        output: OutputConfig | OutputConfigDict | Output[Any] | None = None,
        use: list[ModelMiddleware] | None = None,
        docs: list[DocumentData] | None = None,
    ) -> GenerateResponseWrapper[Any]:
        """Generates text or structured data using a language model.

        This function provides a flexible interface for interacting with various
        language models, supporting both simple text generation and more complex
        interactions involving tools and structured conversations.

        Args:
            model: Optional. The name of the model to use for generation. If not
                provided, a default model may be used.
            prompt: Optional. A single prompt string, a `Part` object, or a list
                of `Part` objects to provide as input to the model. This is used
                for simple text generation.
            system: Optional. A system message string, a `Part` object, or a
                list of `Part` objects to provide context or instructions to
                the model, especially for chat-based models.
            messages: Optional. A list of `Message` objects representing a
                conversation history.  This is used for chat-based models to
                maintain context.
            tools: Optional. A list of tool names (strings) that the model can
                use.
            return_tool_requests: Optional. If `True`, the model will return
                tool requests instead of executing them directly.
            tool_choice: Optional. A `ToolChoice` object specifying how the
                model should choose which tool to use.
            tool_responses: Optional. tool_responses should contain a list of
                tool response parts corresponding to interrupt tool request
                parts from the most recent model message. Each entry must have
                a matching `name` and `ref` (if supplied) for its tool request
                counterpart.
            config: Optional. A `GenerationCommonConfig` object or a dictionary
                containing configuration parameters for the generation process.
                This allows fine-tuning the model's behavior.
            max_turns: Optional. The maximum number of turns in a conversation.
            on_chunk: Optional. A callback function of type
                `ModelStreamingCallback` that is called for each chunk of
                generated text during streaming.
            context: Optional. A dictionary containing additional context
                information that can be used during generation.
            output_format: Optional. The format to use for the output (e.g.,
                'json').
            output_content_type: Optional. The content type of the output.
            output_instructions: Optional. Instructions for formatting the
                output.
            output_constrained: Optional. Whether to constrain the output to the
                schema.
            output: Optional. Use `Output(schema=YourSchema)` for typed responses.
                Can also be an `OutputConfig` object or dictionary.
            use: Optional. A list of `ModelMiddleware` functions to apply to the
                generation process. Middleware can be used to intercept and
                modify requests and responses.
            docs: Optional. A list of documents to be used for grounding.


        Returns:
            A `GenerateResponseWrapper` object containing the model's response,
            which may include generated text, tool requests, or other relevant
            information.

        Note:
            - The `tools`, `return_tool_requests`, and `tool_choice` arguments
              are used for models that support tool usage.
            - The `on_chunk` argument enables streaming responses, allowing you
              to process the generated content as it becomes available.
        """
        # Initialize output_schema - extracted from output parameter
        output_schema: type | dict[str, object] | None = None

        # Unpack output config if provided
        if output:
            if isinstance(output, Output):
                # Handle typed Output[T] - extract values from the typed wrapper
                if output_format is None:
                    output_format = output.format
                if output_content_type is None:
                    output_content_type = output.content_type
                if output_instructions is None:
                    output_instructions = output.instructions
                if output_schema is None:
                    output_schema = output.schema
                if output_constrained is None:
                    output_constrained = output.constrained
            elif isinstance(output, dict):
                # Handle dict input - extract values directly
                if output_format is None:
                    output_format = output.get('format')
                if output_content_type is None:
                    output_content_type = output.get('content_type')
                if output_instructions is None:
                    output_instructions = output.get('instructions')
                if output_schema is None:
                    output_schema = output.get('schema')
                if output_constrained is None:
                    output_constrained = output.get('constrained')
            else:
                # Handle OutputConfig object - use getattr for safety since
                # OutputConfig is auto-generated and may not have all fields.
                if output_format is None:
                    output_format = getattr(output, 'format', None)
                if output_content_type is None:
                    output_content_type = getattr(output, 'content_type', None)
                if output_instructions is None:
                    output_instructions = getattr(output, 'instructions', None)
                if output_schema is None:
                    output_schema = getattr(output, 'schema', None)
                if output_constrained is None:
                    output_constrained = getattr(output, 'constrained', None)

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
            on_chunk=on_chunk,
            middleware=use,
            context=context if context else ActionRunContext._current_context(),  # pyright: ignore[reportPrivateUsage]
        )

    @overload
    def generate_stream(
        self,
        model: str | None = None,
        prompt: str | Part | list[Part] | None = None,
        system: str | Part | list[Part] | None = None,
        messages: list[Message] | None = None,
        tools: list[str] | None = None,
        return_tool_requests: bool | None = None,
        tool_choice: ToolChoice | None = None,
        config: GenerationCommonConfig | dict[str, object] | None = None,
        max_turns: int | None = None,
        context: dict[str, object] | None = None,
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: bool | str | None = None,
        output_constrained: bool | None = None,
        *,
        output: Output[OutputT],
        use: list[ModelMiddleware] | None = None,
        docs: list[DocumentData] | None = None,
        timeout: float | None = None,
    ) -> tuple[
        AsyncIterator[GenerateResponseChunkWrapper],
        asyncio.Future[GenerateResponseWrapper[OutputT]],
    ]: ...

    @overload
    def generate_stream(
        self,
        model: str | None = None,
        prompt: str | Part | list[Part] | None = None,
        system: str | Part | list[Part] | None = None,
        messages: list[Message] | None = None,
        tools: list[str] | None = None,
        return_tool_requests: bool | None = None,
        tool_choice: ToolChoice | None = None,
        config: GenerationCommonConfig | dict[str, object] | None = None,
        max_turns: int | None = None,
        context: dict[str, object] | None = None,
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: bool | str | None = None,
        output_constrained: bool | None = None,
        output: OutputConfig | OutputConfigDict | Output[Any] | None = None,
        use: list[ModelMiddleware] | None = None,
        docs: list[DocumentData] | None = None,
        timeout: float | None = None,
    ) -> tuple[
        AsyncIterator[GenerateResponseChunkWrapper],
        asyncio.Future[GenerateResponseWrapper[Any]],
    ]: ...

    def generate_stream(
        self,
        model: str | None = None,
        prompt: str | Part | list[Part] | None = None,
        system: str | Part | list[Part] | None = None,
        messages: list[Message] | None = None,
        tools: list[str] | None = None,
        return_tool_requests: bool | None = None,
        tool_choice: ToolChoice | None = None,
        config: GenerationCommonConfig | dict[str, object] | None = None,
        max_turns: int | None = None,
        context: dict[str, object] | None = None,
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: bool | str | None = None,
        output_constrained: bool | None = None,
        output: OutputConfig | OutputConfigDict | Output[Any] | None = None,
        use: list[ModelMiddleware] | None = None,
        docs: list[DocumentData] | None = None,
        timeout: float | None = None,
    ) -> tuple[
        AsyncIterator[GenerateResponseChunkWrapper],
        asyncio.Future[GenerateResponseWrapper[Any]],
    ]:
        """Streams generated text or structured data using a language model.

        This function provides a flexible interface for interacting with various
        language models, supporting both simple text generation and more complex
        interactions involving tools and structured conversations.

        Args:
            model: Optional. The name of the model to use for generation. If not
                provided, a default model may be used.
            prompt: Optional. A single prompt string, a `Part` object, or a list
                of `Part` objects to provide as input to the model. This is used
                for simple text generation.
            system: Optional. A system message string, a `Part` object, or a
                list of `Part` objects to provide context or instructions to the
                model, especially for chat-based models.
            messages: Optional. A list of `Message` objects representing a
                conversation history.  This is used for chat-based models to
                maintain context.
            tools: Optional. A list of tool names (strings) that the model can
                use.
            return_tool_requests: Optional. If `True`, the model will return
                tool requests instead of executing them directly.
            tool_choice: Optional. A `ToolChoice` object specifying how the
                model should choose which tool to use.
            config: Optional. A `GenerationCommonConfig` object or a dictionary
                containing configuration parameters for the generation process.
                This allows fine-tuning the model's behavior.
            max_turns: Optional. The maximum number of turns in a conversation.
            context: Optional. A dictionary containing additional context
                information that can be used during generation.
            output_format: Optional. The format to use for the output (e.g.,
                'json').
            output_content_type: Optional. The content type of the output.
            output_instructions: Optional. Instructions for formatting the
                output.
            output_constrained: Optional. Whether to constrain the output to the
                schema.
            output: Optional. Use `Output(schema=YourSchema)` for typed responses.
                Can also be an `OutputConfig` object or dictionary.
            use: Optional. A list of `ModelMiddleware` functions to apply to the
                generation process. Middleware can be used to intercept and
                modify requests and responses.
            docs: Optional. A list of documents to be used for grounding.
            timeout: Optional. The timeout for the streaming action.

        Returns:
            A `GenerateResponseWrapper` object containing the model's response,
            which may include generated text, tool requests, or other relevant
            information.

        Note:
            - The `tools`, `return_tool_requests`, and `tool_choice` arguments
              are used for models that support tool usage.
            - The `on_chunk` argument enables streaming responses, allowing you
              to process the generated content as it becomes available.
        """
        stream: Channel[GenerateResponseChunkWrapper, GenerateResponseWrapper[Any]] = Channel(timeout=timeout)

        resp = self.generate(
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
            output_format=output_format,
            output_content_type=output_content_type,
            output_instructions=output_instructions,
            output_constrained=output_constrained,
            output=output,
            docs=docs,
            use=use,
            on_chunk=lambda c: stream.send(c),
        )
        stream.set_close_future(asyncio.create_task(resp))

        return stream, stream.closed

    async def retrieve(
        self,
        retriever: str | RetrieverRef | None = None,
        query: str | DocumentData | None = None,
        options: dict[str, object] | None = None,
    ) -> RetrieverResponse:
        """Retrieves documents based on query.

        Args:
            retriever: Optional retriever name or reference to use.
            query: Text query or a DocumentData containing query text.
            options: Optional retriever-specific options.

        Returns:
            The generated response with documents.
        """
        retriever_name: str
        retriever_config: dict[str, object] = {}

        if isinstance(retriever, RetrieverRef):
            retriever_name = retriever.name
            retriever_config = retriever.config or {}
            if retriever.version:
                retriever_config['version'] = retriever.version
        elif isinstance(retriever, str):
            retriever_name = retriever
        else:
            raise ValueError('Retriever must be specified as a string name or a RetrieverRef.')

        if isinstance(query, str):
            query = Document.from_text(query)

        request_options = {**(retriever_config or {}), **(options or {})}

        retrieve_action = await self.registry.resolve_retriever(retriever_name)
        if retrieve_action is None:
            raise ValueError(f'Retriever "{retriever_name}" not found')

        if query is None:
            raise ValueError('Query must be specified for retrieval.')

        return (
            await retrieve_action.arun(
                RetrieverRequest(
                    query=query,
                    options=request_options if request_options else None,
                )
            )
        ).response

    async def index(
        self,
        indexer: str | IndexerRef | None = None,
        documents: list[Document] | None = None,
        options: dict[str, object] | None = None,
    ) -> None:
        """Indexes documents.

        Args:
            indexer: Optional indexer name or reference to use.
            documents: Documents to index.
            options: Optional indexer-specific options.
        """
        indexer_name: str
        indexer_config: dict[str, object] = {}

        if isinstance(indexer, IndexerRef):
            indexer_name = indexer.name
            indexer_config = indexer.config or {}
            if indexer.version:
                indexer_config['version'] = indexer.version
        elif isinstance(indexer, str):
            indexer_name = indexer
        else:
            raise ValueError('Indexer must be specified as a string name or an IndexerRef.')

        req_options = {**(indexer_config or {}), **(options or {})}

        index_action = await self.registry.resolve_action(cast(ActionKind, ActionKind.INDEXER), indexer_name)
        if index_action is None:
            raise ValueError(f'Indexer "{indexer_name}" not found')

        if documents is None:
            raise ValueError('Documents must be specified for indexing.')

        _ = await index_action.arun(
            IndexerRequest(
                # Document subclasses DocumentData, so this is type-safe at runtime.
                # list is invariant so list[Document] isn't assignable to list[DocumentData]
                documents=cast(list[DocumentData], documents),
                options=req_options if req_options else None,
            )
        )

    async def embed(
        self,
        embedder: str | EmbedderRef | None = None,
        content: str | Document | DocumentData | None = None,
        metadata: dict[str, object] | None = None,
        options: dict[str, object] | None = None,
    ) -> list[Embedding]:
        """Embeds a single document or string.

        Generates vector embeddings for a single piece of content using the
        specified embedder. This is the primary method for embedding individual
        items.

        When using an EmbedderRef, the config and version from the ref are
        extracted and merged with any provided options. The merge order is:
        {version, ...config, ...options} (options take precedence).

        Args:
            embedder: Embedder name (e.g., 'googleai/text-embedding-004') or
                an EmbedderRef with configuration.
            content: A single string, Document, or DocumentData to embed.
            metadata: Optional metadata to apply to the document. Only used
                when content is a string.
            options: Optional embedder-specific options (e.g., task_type).

        Returns:
            A list containing the Embedding for the input content.

        Raises:
            ValueError: If embedder is not specified or not found.
            ValueError: If content is not specified.

        Example - Basic string embedding:
            >>> embeddings = await ai.embed(embedder='googleai/text-embedding-004', content='Hello, world!')
            >>> print(len(embeddings[0].embedding))  # Vector dimensions

        Example - With metadata:
            >>> embeddings = await ai.embed(
            ...     embedder='googleai/text-embedding-004',
            ...     content='Product description',
            ...     metadata={'category': 'electronics'},
            ... )

        Example - With embedder options:
            >>> embeddings = await ai.embed(
            ...     embedder='googleai/text-embedding-004',
            ...     content='Search query',
            ...     options={'task_type': 'RETRIEVAL_QUERY'},
            ... )

        Example - Using EmbedderRef:
            >>> ref = create_embedder_ref('googleai/text-embedding-004', config={'task_type': 'CLUSTERING'})
            >>> embeddings = await ai.embed(embedder=ref, content='Text')
        """
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

        # Document subclasses DocumentData, so this is type-safe at runtime.
        # list is invariant so list[Document] isn't assignable to list[DocumentData]
        response = (
            await embed_action.arun(
                EmbedRequest(
                    input=documents,  # pyright: ignore[reportArgumentType]
                    options=final_options,
                )
            )
        ).response
        return response.embeddings

    async def embed_many(
        self,
        embedder: str | EmbedderRef | None = None,
        content: list[str] | list[Document] | list[DocumentData] | None = None,
        metadata: dict[str, object] | None = None,
        options: dict[str, object] | None = None,
    ) -> list[Embedding]:
        """Embeds multiple documents or strings in a single batch call.

        Generates vector embeddings for multiple pieces of content in one API
        call. This is more efficient than calling embed() multiple times when
        you have a batch of items to embed.

        Important: Unlike embed(), this method does NOT extract config/version
        from EmbedderRef. It only uses the ref to resolve the embedder name
        and passes options directly. This matches the JS canonical behavior.

        Args:
            embedder: Embedder name (e.g., 'googleai/text-embedding-004') or
                an EmbedderRef.
            content: List of strings, Documents, or DocumentData to embed.
            metadata: Optional metadata to apply to all items. Only used when
                content items are strings.
            options: Optional embedder-specific options.

        Returns:
            List of Embedding objects, one per input item (same order).

        Raises:
            ValueError: If embedder is not specified or not found.
            ValueError: If content is not specified.

        Example - Basic batch embedding:
            >>> embeddings = await ai.embed_many(
            ...     embedder='googleai/text-embedding-004',
            ...     content=['Doc 1', 'Doc 2', 'Doc 3'],
            ... )
            >>> for i, emb in enumerate(embeddings):
            ...     print(f'Doc {i}: {len(emb.embedding)} dims')

        Example - With shared metadata:
            >>> embeddings = await ai.embed_many(
            ...     embedder='googleai/text-embedding-004',
            ...     content=['text1', 'text2'],
            ...     metadata={'batch_id': 'batch-001'},
            ... )

        Example - With options (EmbedderRef config is NOT extracted):
            >>> embeddings = await ai.embed_many(
            ...     embedder='googleai/text-embedding-004',
            ...     content=documents,
            ...     options={'task_type': 'RETRIEVAL_DOCUMENT'},
            ... )
        """
        if content is None:
            raise ValueError('Content must be specified for embedding.')

        # Convert strings to Documents if needed
        documents: list[Document | DocumentData] = [
            Document.from_text(item, metadata) if isinstance(item, str) else item for item in content
        ]

        # Resolve embedder name (JS embedMany does not extract config/version from ref)
        embedder_name = self._resolve_embedder_name(embedder)

        embed_action = await self.registry.resolve_embedder(embedder_name)
        if embed_action is None:
            raise ValueError(f'Embedder "{embedder_name}" not found')

        response = (await embed_action.arun(EmbedRequest(input=documents, options=options))).response
        return response.embeddings

    async def evaluate(
        self,
        evaluator: str | EvaluatorRef | None = None,
        dataset: list[BaseDataPoint] | None = None,
        options: dict[str, object] | None = None,
        eval_run_id: str | None = None,
    ) -> EvalResponse:
        """Evaluates a dataset using an evaluator.

        Args:
            evaluator: Name or reference of the evaluator to use.
            dataset: Dataset to evaluate.
            options: Evaluation options.
            eval_run_id: Optional ID for the evaluation run.

        Returns:
            The evaluation results.
        """
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
            await eval_action.arun(
                EvalRequest(
                    dataset=dataset,
                    options=final_options,
                    eval_run_id=eval_run_id,
                )
            )
        ).response

    @staticmethod
    def current_context() -> dict[str, Any] | None:
        """Retrieves the current execution context for the running action.

        This allows tools and other actions to access context data (like auth
        or metadata) passed through the execution chain via ContextVars.
        This provides parity with the JavaScript SDK's context handling.

        Returns:
            The current context dictionary, or None if not running in an action.
        """
        return ActionRunContext._current_context()  # pyright: ignore[reportPrivateUsage]

    def dynamic_tool(
        self,
        name: str,
        fn: Callable[..., object],
        description: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Action:
        """Creates an unregistered tool action.

        This is useful for creating tools that are passed directly to generate()
        without being registered in the global registry. Dynamic tools behave exactly
        like registered tools but offer more flexibility for runtime-defined logic.

        Args:
            name: The unique name of the tool.
            fn: The function that implements the tool logic.
            description: Optional human-readable description of what the tool does.
            metadata: Optional dictionary of metadata about the tool.

        Returns:
            An Action instance of kind TOOL, configured for dynamic execution.
        """
        tool_meta: dict[str, object] = metadata.copy() if metadata else {}
        tool_meta['type'] = 'tool'
        tool_meta['dynamic'] = True
        return Action(
            kind=ActionKind.TOOL,
            name=name,
            fn=fn,
            description=description,
            metadata=tool_meta,
        )

    async def flush_tracing(self) -> None:
        """Flushes all registered trace processors.

        This ensures all pending spans are exported before the application
        shuts down, preventing loss of telemetry data.
        """
        provider = trace_api.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            await ensure_async(provider.force_flush)()

    async def run(
        self,
        name: str,
        func_or_input: object,
        maybe_fn: Callable[..., T | Awaitable[T]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> T:
        """Runs a function as a discrete step within a trace.

        This method is used to create sub-spans (steps) within a flow or other action.
        Each run step is recorded separately in the trace, making it easier to
        debug and monitor the internal execution of complex flows.

        It supports two call signatures:
        1. `run(name, fn)`: Runs the provided function.
        2. `run(name, input, fn)`: Passes the input to the function and records it.

        Args:
            name: The descriptive name of the span/step.
            func_or_input: Either the function to execute, or input data to pass
                to `maybe_fn`.
            maybe_fn: An optional function to execute if `func_or_input` is
                provided as input data.
            metadata: Optional metadata to associate with the generated trace span.

        Returns:
            The result of the function execution.
        """
        fn: Callable[..., T | Awaitable[T]]
        input_data: Any = None
        has_input = False

        if maybe_fn:
            fn = maybe_fn
            input_data = func_or_input
            has_input = True
        elif callable(func_or_input):
            fn = cast(Callable[..., T | Awaitable[T]], func_or_input)
        else:
            raise ValueError('A function must be provided to run.')

        span_metadata = SpanMetadata(name=name, metadata=metadata)
        with run_in_new_span(span_metadata, labels={'genkit:type': 'flowStep'}) as span:
            try:
                if has_input:
                    span.set_input(input_data)
                    result = await ensure_async(fn)(input_data)
                else:
                    result = await ensure_async(fn)()

                span.set_output(result)
                return result
            except Exception:
                # We catch all exceptions here to ensure they are captured by
                # the trace span context manager before being re-raised.
                # The GenkitSpan wrapper (run_in_new_span) handles recording
                # the exception details.
                raise

    async def check_operation(self, operation: Operation) -> Operation:
        """Checks the status of a long-running background operation.

        This method matches JS checkOperation from js/ai/src/check-operation.ts.

        It looks up the background action by the operation's action key and
        calls its check method to get updated status.

        Args:
            operation: The Operation object to check. Must have an action
                field specifying which background model created it.

        Returns:
            An updated Operation object with the current status.

        Raises:
            ValueError: If the operation is missing original request information
                or if the background action cannot be resolved.

        Example:
            >>> # Start a background operation
            >>> response = await ai.generate(model=veo_model, prompt='A cat')
            >>> operation = response.operation
            >>> # Poll until done
            >>> while not operation.done:
            ...     await asyncio.sleep(5)
            ...     operation = await ai.check_operation(operation)
            >>> # Use the result
            >>> print(operation.output)
        """
        return await check_operation_impl(self.registry, operation)

    async def cancel_operation(self, operation: Operation) -> Operation:
        """Cancels a long-running background operation.

        This method attempts to cancel an in-progress operation. Not all
        background models support cancellation.

        If cancellation is not supported, returns the operation unchanged
        (matching JS behavior).

        Args:
            operation: The Operation object to cancel. Must have an action
                field specifying which background model created it.

        Returns:
            An updated Operation object reflecting the cancellation attempt.

        Raises:
            ValueError: If the operation is missing original request information
                or if the background action cannot be resolved.

        Example:
            >>> # Start a background operation
            >>> response = await ai.generate(model=veo_model, prompt='A cat')
            >>> operation = response.operation
            >>> # Cancel it
            >>> operation = await ai.cancel_operation(operation)
        """
        if not operation.action:
            raise ValueError('Provided operation is missing original request information')

        background_action = await lookup_background_action(self.registry, operation.action)
        if background_action is None:
            raise ValueError(f'Failed to resolve background action from original request: {operation.action}')

        return await background_action.cancel(operation)

    async def generate_operation(
        self,
        model: str | None = None,
        prompt: str | Part | list[Part] | None = None,
        system: str | Part | list[Part] | None = None,
        messages: list[Message] | None = None,
        tools: list[str] | None = None,
        return_tool_requests: bool | None = None,
        tool_choice: ToolChoice | None = None,
        config: GenerationCommonConfig | dict[str, object] | None = None,
        max_turns: int | None = None,
        context: dict[str, object] | None = None,
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: bool | str | None = None,
        output_constrained: bool | None = None,
        output: OutputConfig | OutputConfigDict | Output[Any] | None = None,
        use: list[ModelMiddleware] | None = None,
        docs: list[DocumentData] | None = None,
        on_chunk: ModelStreamingCallback | None = None,
    ) -> Operation:
        """Generate content using a long-running model and return an Operation.

        This method is for models that support long-running operations (like
        video generation with Veo). It returns an Operation that can be polled
        with check_operation() until completion.

        Note: This is a beta feature. Only models that support long-running
        operations (model.supports.long_running = True) can be used with this
        method.

        The Operation Flow
        ==================

        ┌─────────────────────────────────────────────────────────────────┐
        │                    generate_operation() Flow                     │
        ├─────────────────────────────────────────────────────────────────┤
        │                                                                  │
        │   ┌──────────────────┐                                          │
        │   │ Resolve Model    │                                          │
        │   │ Check supports   │                                          │
        │   │ long_running     │                                          │
        │   └────────┬─────────┘                                          │
        │            │                                                     │
        │      ┌─────┴─────┐                                              │
        │      │           │                                              │
        │      ▼           ▼                                              │
        │   ┌───────┐   ┌───────────────┐                                │
        │   │ Error │   │ Call generate │                                │
        │   │ (no   │   │ Get operation │                                │
        │   │ LRO)  │   └───────┬───────┘                                │
        │   └───────┘           │                                         │
        │                       ▼                                         │
        │              ┌──────────────┐                                   │
        │              │  Operation   │                                   │
        │              │  done=False  │ ──► poll with check_operation()  │
        │              │  id=...      │                                   │
        │              └──────────────┘                                   │
        │                                                                  │
        └─────────────────────────────────────────────────────────────────┘

        Args:
            model: The model to use for generation (must support long_running).
            prompt: The prompt text or parts.
            system: System message for the model.
            messages: Conversation history.
            tools: Tool names available to the model.
            return_tool_requests: Whether to return tool requests.
            tool_choice: How the model should choose tools.
            config: Generation configuration.
            max_turns: Maximum conversation turns.
            context: Additional context data.
            output_format: Output format (e.g., 'json').
            output_content_type: Output content type.
            output_instructions: Output formatting instructions.
            output_constrained: Whether to constrain output to schema.
            output: Typed output configuration.
            use: Middleware to apply.
            docs: Documents for grounding.
            on_chunk: Callback for streaming chunks.

        Returns:
            An Operation object for tracking the long-running generation.

        Raises:
            GenkitError: If the model doesn't support long-running operations.
            GenkitError: If the model didn't return an operation.

        Example:
            >>> # Generate video with Veo (long-running operation)
            >>> operation = await ai.generate_operation(
            ...     model='googleai/veo-2.0-generate-001',
            ...     prompt='A banana riding a bicycle.',
            ... )
            >>> # Poll until done
            >>> while not operation.done:
            ...     await asyncio.sleep(5)
            ...     operation = await ai.check_operation(operation)
            >>> # Access result
            >>> print(operation.output)
        """
        # Resolve the model and check for long_running support
        resolved_model = model or self.registry.default_model
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
        model_info = model_action.metadata.get('model') if model_action.metadata else None
        supports_long_running = False
        if model_info:
            # model_info can be ModelInfo or dict
            if hasattr(model_info, 'supports'):
                supports_attr = getattr(model_info, 'supports', None)
                if supports_attr:
                    supports_long_running = getattr(supports_attr, 'long_running', False)
            elif isinstance(model_info, dict):
                # Cast to dict[str, Any] for type checker
                model_info_dict: dict[str, Any] = model_info  # type: ignore[assignment]
                supports = model_info_dict.get('supports')
                if isinstance(supports, dict):
                    supports_long_running = bool(supports.get('longRunning', False))

        if not supports_long_running:
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
            output_format=output_format,
            output_content_type=output_content_type,
            output_instructions=output_instructions,
            output_constrained=output_constrained,
            output=output,
            use=use,
            docs=docs,
            on_chunk=on_chunk,
        )

        # Extract operation from response
        if not hasattr(response, 'operation') or not response.operation:
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message=f"Model '{model_action.name}' did not return an operation.",
            )

        return response.operation
