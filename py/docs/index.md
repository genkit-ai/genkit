# API Reference

!!! note
    Full Genkit documentation is available at [genkit.dev](https://genkit.dev/python/docs/get-started/)

## genkit

::: genkit.Genkit

::: genkit.Message

::: genkit.Role

::: genkit.Part

::: genkit.Media

::: genkit.Document

::: genkit.ToolChoice

::: genkit.ModelResponse

::: genkit.ModelResponseChunk

::: genkit.ModelStreamResponse

::: genkit.StreamResponse

::: genkit.FinishReason

::: genkit.Embedding

::: genkit.Operation

::: genkit.tool

::: genkit.Tool

::: genkit.ToolRunContext

::: genkit.Interrupt

::: genkit.response

::: genkit.MultipartToolResponse

::: genkit.ActionRunContext

::: genkit.Prompt

::: genkit.GenkitError

::: genkit.PublicError

::: genkit.RuntimeErrorReason

::: genkit.ContextProvider

::: genkit.RequestData

::: genkit.FormatDef

::: genkit.FormatterConfig

::: genkit.DynamicActionProvider

## genkit.model

::: genkit.model.BackgroundAction

::: genkit.model.GenerateActionOptions

::: genkit.model.ModelRequest

::: genkit.model.OutputConfig

::: genkit.model.ModelUsage

::: genkit.model.Candidate

::: genkit.model.OperationError

::: genkit.model.ToolRequest

::: genkit.model.ToolDefinition

::: genkit.model.ToolResponse

::: genkit.model.ModelInfo

::: genkit.model.Supports

::: genkit.model.Constrained

::: genkit.model.Stage

::: genkit.model.model_action_metadata

::: genkit.model.model_ref

::: genkit.model.get_basic_usage_stats

::: genkit.model.ModelRef

::: genkit.model.ModelConfig

## genkit.embedder

::: genkit.embedder.EmbedRequest

::: genkit.embedder.EmbedResponse

::: genkit.embedder.embedder_action_metadata

::: genkit.embedder.embedder_ref

::: genkit.embedder.EmbedderRef

::: genkit.embedder.EmbedderSupports

::: genkit.embedder.EmbedderInfo

## genkit.plugin_api

::: genkit.plugin_api.Plugin

::: genkit.plugin_api.Action

::: genkit.plugin_api.ActionMetadata

::: genkit.plugin_api.ActionKind

::: genkit.plugin_api.StatusName

::: genkit.plugin_api.GENKIT_CLIENT_HEADER

::: genkit.plugin_api.loop_local_client

::: genkit.plugin_api.to_json_schema

::: genkit.plugin_api.get_cached_client

::: genkit.plugin_api.is_dev_environment

## genkit.evaluator

::: genkit.evaluator.BaseDataPoint

::: genkit.evaluator.EvalRequest

::: genkit.evaluator.EvalFnResponse

::: genkit.evaluator.Score

::: genkit.evaluator.Details

::: genkit.evaluator.EvalStatusEnum

::: genkit.evaluator.evaluator_action_metadata

::: genkit.evaluator.evaluator_ref

::: genkit.evaluator.EvaluatorRef

## genkit.telemetry

::: genkit.telemetry.run_in_new_span

::: genkit.telemetry.SpanMetadata

## genkit.exp.agent

::: genkit.exp.agent.apply_json_patch

::: genkit.exp.agent.diff_json

::: genkit.exp.agent.JsonPatchOp

::: genkit.exp.agent.JsonPatchOperation

::: genkit.exp.agent.StateT

::: genkit.exp.agent.TERMINAL_STATUSES

::: genkit.exp.agent.SaveFn

::: genkit.exp.agent.apply_save

::: genkit.exp.agent.iterate_statuses

::: genkit.exp.agent.require_one_selector

::: genkit.exp.agent.session_id_of
