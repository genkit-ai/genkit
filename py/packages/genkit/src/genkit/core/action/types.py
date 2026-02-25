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

"""Action types module for defining and managing action types."""

from collections.abc import Callable
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from genkit.core._compat import StrEnum

# Type alias for action name.
# type ActionName = str
ActionName = str


# type ActionResolver = Callable[[ActionKind, str], None]
ActionResolver = Callable[['ActionKind', str], None]


class ActionKind(StrEnum):
    """Enumerates all the types of action that can be registered.

    This enum defines the various types of actions supported by the framework,
    including chat models, embedders, evaluators, and other utility functions.
    """

    BACKGROUND_MODEL = 'background-model'
    CANCEL_OPERATION = 'cancel-operation'
    CHECK_OPERATION = 'check-operation'
    CUSTOM = 'custom'
    DYNAMIC_ACTION_PROVIDER = 'dynamic-action-provider'
    EMBEDDER = 'embedder'
    EVALUATOR = 'evaluator'
    EXECUTABLE_PROMPT = 'executable-prompt'
    FLOW = 'flow'
    INDEXER = 'indexer'
    MODEL = 'model'
    PROMPT = 'prompt'
    RERANKER = 'reranker'
    RESOURCE = 'resource'
    RETRIEVER = 'retriever'
    TOOL = 'tool'
    UTIL = 'util'


ResponseT = TypeVar('ResponseT')


class ActionResponse(BaseModel, Generic[ResponseT]):
    """The response from an action.

    Attributes:
        response: The actual response data from the action execution.
        trace_id: A unique identifier for tracing the action execution.
        span_id: The span ID of the root action span.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra='forbid', populate_by_name=True, alias_generator=to_camel, arbitrary_types_allowed=True
    )

    response: ResponseT
    trace_id: str
    span_id: str = ''


class ActionMetadataKey(StrEnum):
    """Enumerates all the keys of the action metadata.

    Attributes:
        INPUT_KEY: Key for the input schema metadata.
        OUTPUT_KEY: Key for the output schema metadata.
        RETURN: Key for the return type metadata.
    """

    INPUT_KEY = 'inputSchema'
    OUTPUT_KEY = 'outputSchema'
    RETURN = 'return'
