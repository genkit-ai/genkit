# Copyright 2026 Google LLC
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

"""Wire types for the Google AI Interactions API."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from genkit._core._typing import ToolChoice

API_REVISION = '2026-05-20'

ResponseModality = Literal['text', 'image', 'audio']
MediaResolution = Literal['low', 'medium', 'high', 'ultra_high']
InteractionStatus = Literal[
    'in_progress',
    'requires_action',
    'completed',
    'failed',
    'cancelled',
]
ThinkingSummaries = Literal['auto', 'none']
ThinkingLevel = Literal['minimal', 'low', 'medium', 'high']


class InteractionFunctionTool(TypedDict, total=False):
    type: Literal['function']
    name: str
    description: str
    parameters: dict[str, Any]


class InteractionGoogleSearchTool(TypedDict):
    type: Literal['google_search']


class InteractionCodeExecutionTool(TypedDict):
    type: Literal['code_execution']


class InteractionUrlContextTool(TypedDict):
    type: Literal['url_context']


class InteractionFileSearchTool(TypedDict, total=False):
    type: Literal['file_search']
    file_search_store_names: list[str]


class InteractionMcpServerTool(TypedDict, total=False):
    type: Literal['mcp_server']
    name: str
    url: str
    headers: dict[str, str]
    allowed_tools: list[str]


InteractionTool = (
    InteractionFunctionTool
    | InteractionGoogleSearchTool
    | InteractionCodeExecutionTool
    | InteractionUrlContextTool
    | InteractionFileSearchTool
    | InteractionMcpServerTool
)


class TextAnnotation(TypedDict, total=False):
    type: str
    start_index: int
    end_index: int
    url: str
    title: str
    source: str


class TextContent(TypedDict, total=False):
    type: Literal['text']
    text: str
    annotations: list[TextAnnotation]


class ImageContent(TypedDict, total=False):
    type: Literal['image']
    data: str
    uri: str
    mime_type: str
    resolution: MediaResolution


class AudioContent(TypedDict, total=False):
    type: Literal['audio']
    data: str
    uri: str
    mime_type: str


class DocumentContent(TypedDict, total=False):
    type: Literal['document']
    data: str
    uri: str
    mime_type: str


class VideoContent(TypedDict, total=False):
    type: Literal['video']
    data: str
    uri: str
    mime_type: str
    resolution: MediaResolution


class ThoughtContent(TypedDict, total=False):
    type: Literal['thought']
    signature: str
    summary: list[TextContent | ImageContent]


class FunctionCallContent(TypedDict, total=False):
    type: Literal['function_call']
    name: str
    arguments: dict[str, Any]
    id: str


class FunctionResultContent(TypedDict, total=False):
    type: Literal['function_result']
    name: str
    is_error: bool
    result: dict[str, Any] | str
    call_id: str


Content = (
    TextContent
    | ImageContent
    | AudioContent
    | DocumentContent
    | VideoContent
    | ThoughtContent
    | FunctionCallContent
    | FunctionResultContent
)


class ModelOutputStep(TypedDict):
    type: Literal['model_output']
    content: list[Content]


class UserInputStep(TypedDict):
    type: Literal['user_input']
    content: list[Content]


class GoogleSearchCallStep(TypedDict, total=False):
    type: Literal['google_search_call']
    id: str
    arguments: dict[str, list[str]]
    signature: str


class GoogleSearchResultStep(TypedDict, total=False):
    type: Literal['google_search_result']
    call_id: str
    result: dict[str, Any]
    signature: str


class CodeExecutionCallStep(TypedDict, total=False):
    type: Literal['code_execution_call']
    id: str
    arguments: dict[str, Any]
    signature: str


class CodeExecutionResultStep(TypedDict, total=False):
    type: Literal['code_execution_result']
    call_id: str
    result: dict[str, Any] | str
    signature: str


Step = (
    ModelOutputStep
    | UserInputStep
    | Content
    | GoogleSearchCallStep
    | GoogleSearchResultStep
    | CodeExecutionCallStep
    | CodeExecutionResultStep
)


class ModalityTokens(TypedDict, total=False):
    modality: ResponseModality
    tokens: int


class Usage(TypedDict, total=False):
    total_input_tokens: int
    input_tokens_by_modality: list[ModalityTokens]
    total_cached_tokens: int
    cached_tokens_by_modality: list[ModalityTokens]
    total_output_tokens: int
    output_tokens_by_modality: list[ModalityTokens]
    total_tool_use_tokens: int
    tool_use_by_modality: list[ModalityTokens]
    total_thought_tokens: int
    total_tokens: int


class SpeechConfig(TypedDict, total=False):
    voice: str
    language: str
    speaker: str


class ImageConfig(TypedDict, total=False):
    aspect_ratio: str
    image_size: str


class ModelGenerationConfig(TypedDict, total=False):
    temperature: float
    top_p: float
    seed: int
    stop_sequences: list[str]
    tool_choice: ToolChoice | dict[str, Any]
    thinking_level: ThinkingLevel
    thinking_summaries: ThinkingSummaries
    max_output_tokens: int
    speech_config: SpeechConfig
    image_config: ImageConfig


class DynamicAgentConfig(TypedDict):
    type: Literal['dynamic']


class DeepResearchAgentConfig(TypedDict, total=False):
    type: Literal['deep-research']
    thinking_summaries: ThinkingSummaries
    visualization: Literal['auto', 'off']
    collaborative_planning: bool


InteractionsAgentConfig = DynamicAgentConfig | DeepResearchAgentConfig


class CreateInteractionRequest(TypedDict, total=False):
    previous_interaction_id: str
    model: str
    agent: str
    environment: str | dict[str, Any]
    # API also accepts str | Step; we only ever send a list of steps.
    input: list[Step]
    system_instruction: str
    tools: list[InteractionTool]
    response_format: dict[str, Any] | list[dict[str, Any]]
    response_modalities: list[ResponseModality]
    stream: bool
    store: bool
    background: bool
    generation_config: ModelGenerationConfig
    agent_config: InteractionsAgentConfig


class GeminiInteraction(TypedDict, total=False):
    model: str
    agent: str
    environment_id: str
    id: str
    previous_interaction_id: str
    status: InteractionStatus
    created: str
    updated: str
    role: str
    steps: list[Step]
    usage: Usage
    error: dict[str, Any]


class ClientOptions(TypedDict, total=False):
    api_version: str
    base_url: str
    custom_headers: dict[str, str]
    timeout: float
    experimental_debug_traces: bool
