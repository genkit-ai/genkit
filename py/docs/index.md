# API Reference

!!! note
    Full Genkit documentation is available at [genkit.dev](https://genkit.dev/python/docs/get-started/)

## genkit

::: genkit.Genkit

::: genkit.Plugin

::: genkit.Action

::: genkit.Flow

::: genkit.ActionKind

::: genkit.ActionRunContext

::: genkit.ExecutablePrompt

::: genkit.PromptGenerateOptions

::: genkit.ResumeOptions

::: genkit.ToolRunContext

::: genkit.tool_response

::: genkit.StreamResponse

::: genkit.ModelStreamResponse

::: genkit.GenkitError

::: genkit.PublicError

::: genkit.ToolInterruptError

::: genkit.Message

::: genkit.Part

::: genkit.TextPart

::: genkit.MediaPart

::: genkit.Media

::: genkit.CustomPart

::: genkit.ReasoningPart

::: genkit.Role

::: genkit.Metadata

::: genkit.ToolRequest

::: genkit.ToolRequestPart

::: genkit.ToolResponse

::: genkit.ToolResponsePart

::: genkit.ToolDefinition

::: genkit.ToolChoice

::: genkit.Document

::: genkit.DocumentPart

::: genkit.EmbedderRef

::: genkit.EmbedderOptions

::: genkit.Embedding

::: genkit.EmbedRequest

::: genkit.EmbedResponse

::: genkit.ModelRequest

::: genkit.ModelResponse

::: genkit.ModelResponseChunk

::: genkit.ModelConfig

::: genkit.ModelInfo

::: genkit.ModelUsage

::: genkit.Constrained

::: genkit.Stage

::: genkit.Supports

::: genkit.FinishReason

## genkit.model

::: genkit.model.BackgroundAction

::: genkit.model.ModelRequest

::: genkit.model.ModelResponse

::: genkit.model.ModelResponseChunk

::: genkit.model.ModelUsage

::: genkit.model.Candidate

::: genkit.model.FinishReason

::: genkit.model.GenerateActionOptions

::: genkit.model.Error

::: genkit.model.Operation

::: genkit.model.ToolRequest

::: genkit.model.ToolDefinition

::: genkit.model.ToolResponse

::: genkit.model.ModelInfo

::: genkit.model.Supports

::: genkit.model.Constrained

::: genkit.model.Stage

::: genkit.model.model_action_metadata

::: genkit.model.model_ref

::: genkit.model.ModelRef

::: genkit.model.ModelConfig

::: genkit.model.Message

::: genkit.model.get_basic_usage_stats

## genkit.embedder

::: genkit.embedder.EmbedRequest

::: genkit.embedder.EmbedResponse

::: genkit.embedder.Embedding

::: genkit.embedder.embedder_action_metadata

::: genkit.embedder.embedder_ref

::: genkit.embedder.EmbedderRef

::: genkit.embedder.EmbedderSupports

::: genkit.embedder.EmbedderOptions

## genkit.plugin_api

::: genkit.plugin_api.Plugin

::: genkit.plugin_api.Action

::: genkit.plugin_api.ActionMetadata

::: genkit.plugin_api.ActionKind

::: genkit.plugin_api.ActionRunContext

::: genkit.plugin_api.StatusCodes

::: genkit.plugin_api.StatusName

::: genkit.plugin_api.GenkitError

::: genkit.plugin_api.GENKIT_CLIENT_HEADER

::: genkit.plugin_api.GENKIT_VERSION

::: genkit.plugin_api.loop_local_client

::: genkit.plugin_api.tracer

::: genkit.plugin_api.add_custom_exporter

::: genkit.plugin_api.AdjustingTraceExporter

::: genkit.plugin_api.RedactedSpan

::: genkit.plugin_api.to_display_path

::: genkit.plugin_api.to_json_schema

::: genkit.plugin_api.get_cached_client

::: genkit.plugin_api.get_callable_json

::: genkit.plugin_api.is_dev_environment

::: genkit.plugin_api.model_action_metadata

::: genkit.plugin_api.model_ref

::: genkit.plugin_api.ModelRef

::: genkit.plugin_api.embedder_action_metadata

::: genkit.plugin_api.embedder_ref

::: genkit.plugin_api.EmbedderRef

::: genkit.plugin_api.evaluator_action_metadata

::: genkit.plugin_api.evaluator_ref

::: genkit.plugin_api.EvaluatorRef

::: genkit.plugin_api.ContextProvider

::: genkit.plugin_api.RequestData

## genkit.evaluator

::: genkit.evaluator.EvalRequest

::: genkit.evaluator.EvalResponse

::: genkit.evaluator.EvalFnResponse

::: genkit.evaluator.Score

::: genkit.evaluator.Details

::: genkit.evaluator.BaseEvalDataPoint

::: genkit.evaluator.BaseDataPoint

::: genkit.evaluator.EvalStatusEnum

::: genkit.evaluator.evaluator_action_metadata

::: genkit.evaluator.evaluator_ref

::: genkit.evaluator.EvaluatorRef
