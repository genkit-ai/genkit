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

"""Typed configuration schema for Anthropic models.

Extends the shared :class:`ModelConfig` (``version``, ``temperature``,
``maxOutputTokens``, ...) with Anthropic-specific options.

Unknown keys raise, at the top level and inside nested settings like
``thinking``, so a typo fails before the request is sent. Anthropic body
fields this class doesn't declare go in ``extra`` and are sent through
the SDK's ``extra_body``. Top-level Genkit fields
accept their usual camelCase aliases, while Anthropic-specific nested keys
match the Anthropic plugin shape field-by-field.
"""

from typing import Annotated, ClassVar, Literal, cast

from anthropic.types.beta.message_create_params import MessageCreateParamsBase as BetaMessageCreateParamsBase
from anthropic.types.message_create_params import MessageCreateParamsBase
from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema, model_validator
from pydantic.config import JsonDict

from genkit.model import ModelConfig
from genkit.plugin_api import GenkitConfig

BETA_ONLY_KEYS = frozenset(BetaMessageCreateParamsBase.__annotations__) - frozenset(
    MessageCreateParamsBase.__annotations__
)

_NESTED_CONFIG = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

_THINKING_SCHEMA = {
    'type': 'object',
    'properties': {
        'enabled': {'type': 'boolean'},
        'budgetTokens': {'type': 'integer', 'minimum': 1024},
        'budget_tokens': {'type': 'integer', 'minimum': 1024},
        'adaptive': {'type': 'boolean'},
        'display': {'type': 'string', 'enum': ['summarized', 'omitted']},
        'type': {'type': 'string', 'enum': ['enabled', 'disabled', 'adaptive']},
    },
    'additionalProperties': False,
    'description': (
        'The thinking configuration to use for the request. Thinking is a feature that '
        'allows the model to think about the request and provide a better response.'
    ),
}

_OUTPUT_CONFIG_SCHEMA = {
    'type': 'object',
    'properties': {
        'effort': {'type': 'string', 'enum': ['low', 'medium', 'high', 'xhigh', 'max']},
        'task_budget': {
            'type': 'object',
            'properties': {
                'type': {'type': 'string', 'const': 'tokens', 'default': 'tokens'},
                'total': {'type': 'integer', 'minimum': 20000},
            },
            'required': ['total'],
            'additionalProperties': False,
        },
    },
    'additionalProperties': False,
    'description': 'Configuration for output generation, such as setting the effort parameter and task budgets.',
}

_TOOL_CHOICE_SCHEMA = {
    'type': 'object',
    'properties': {
        'type': {
            'type': 'string',
            'enum': ['auto', 'any', 'tool', 'none'],
            'description': 'Tool choice mode.',
        },
        'name': {
            'type': 'string',
            'description': 'Tool name to require when type is tool.',
        },
        'disable_parallel_tool_use': {
            'type': 'boolean',
            'description': 'Allow at most one tool call in the reply. Not valid with type none.',
        },
    },
    'required': ['type'],
    'additionalProperties': False,
    'description': (
        'The tool choice to use for the request. This can be used to specify the tool to '
        'use for the request. If not specified, the model will choose the tool to use.'
    ),
}

_METADATA_SCHEMA = {
    'type': 'object',
    'properties': {'user_id': {'type': 'string'}},
    'additionalProperties': False,
    'description': 'The metadata to include in the request.',
}


def _anthropic_config_schema_extra(schema: JsonDict) -> None:
    """Tune the advertised Dev UI schema without changing runtime validation."""
    properties = schema.get('properties')
    if not isinstance(properties, dict):
        return
    props = cast(JsonDict, properties)

    props.update(
        cast(
            JsonDict,
            {
                'version': {
                    'type': 'string',
                    'title': 'Version',
                    'description': 'Per-request model version override.',
                },
                'temperature': {
                    'type': 'number',
                    'title': 'Temperature',
                    'description': 'Controls the randomness of the output.',
                },
                'maxOutputTokens': {
                    'type': 'number',
                    'title': 'Max output tokens',
                    'description': 'Maximum number of tokens to generate. Sent as 4096 when omitted.',
                },
                'topK': {
                    'type': 'number',
                    'title': 'Top K',
                    'description': 'Limits token sampling to the top K candidates.',
                },
                'topP': {
                    'type': 'number',
                    'title': 'Top P',
                    'description': 'Limits token sampling by cumulative probability.',
                },
                'stopSequences': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'title': 'Stop sequences',
                    'description': 'Sequences where generation should stop.',
                },
                'apiVersion': {
                    'type': 'string',
                    'enum': ['stable', 'beta'],
                    'title': 'API version',
                    'description': 'Selects the Anthropic API surface for this request.',
                },
                'betas': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'title': 'Betas',
                    'description': (
                        'Anthropic beta feature headers to enable for this request. '
                        'An empty list suppresses the defaults.'
                    ),
                },
            },
        )
    )


class ThinkingConfig(GenkitConfig):
    """Extended-thinking configuration.

    ``enabled``, ``adaptive`` and ``disabled`` are mutually exclusive, and
    ``budgetTokens`` is required when ``enabled`` is true.
    """

    enabled: bool | None = None
    # Adaptive mode allows a fractional budget it ignores; integers enforced only when enabled.
    budget_tokens: float | None = Field(default=None, ge=1024)
    adaptive: bool | None = None
    display: Literal['summarized', 'omitted'] | None = None
    # The API's own spelling (`{'type': 'enabled', ...}`). A mode this list lacks
    # can go out through `extra={'thinking': {...}}`, which replaces the setting.
    type: Literal['enabled', 'disabled', 'adaptive'] | None = None

    @model_validator(mode='after')
    def _check_thinking(self) -> 'ThinkingConfig':
        """Enforce cross-field thinking rules."""
        thinking_type = self.type
        enabled = self.enabled is True or thinking_type == 'enabled'
        adaptive = self.adaptive is True or thinking_type == 'adaptive'
        disabled = self.enabled is False or thinking_type == 'disabled'
        budget_implies_enabled = self.budget_tokens is not None and not adaptive and not disabled

        if enabled and adaptive:
            raise ValueError('Cannot use both enabled and adaptive thinking modes simultaneously')
        if disabled and (enabled or adaptive):
            raise ValueError('Cannot disable thinking and request an enabled or adaptive thinking mode simultaneously')
        if enabled and self.budget_tokens is None:
            raise ValueError('budgetTokens is required when thinking is enabled')
        if (
            (enabled or budget_implies_enabled)
            and self.budget_tokens is not None
            and not float(self.budget_tokens).is_integer()
        ):
            raise ValueError('budgetTokens must be an integer when thinking is enabled')
        return self


class TaskBudget(BaseModel):
    """Token budget for output generation."""

    model_config = _NESTED_CONFIG

    type: Literal['tokens'] = 'tokens'
    total: int = Field(ge=20000)


class OutputConfig(BaseModel):
    """Output-generation configuration (effort and task budget)."""

    model_config = _NESTED_CONFIG

    effort: Literal['low', 'medium', 'high', 'xhigh', 'max'] | None = None
    task_budget: TaskBudget | None = Field(default=None, alias='task_budget')


class AutoToolChoice(BaseModel):
    """Let the model decide whether to call a tool."""

    model_config = _NESTED_CONFIG
    type: Literal['auto']
    disable_parallel_tool_use: bool | None = None
    """Allow at most one tool call in the reply."""


class AnyToolChoice(BaseModel):
    """Require the model to call some tool."""

    model_config = _NESTED_CONFIG
    type: Literal['any']
    disable_parallel_tool_use: bool | None = None
    """Allow at most one tool call in the reply."""


class SpecificToolChoice(BaseModel):
    """Require the model to call the named tool."""

    model_config = _NESTED_CONFIG
    type: Literal['tool']
    name: str
    disable_parallel_tool_use: bool | None = None
    """Allow at most one tool call in the reply."""


class ToolChoiceNone(BaseModel):
    """Prevent the model from calling a tool."""

    model_config = _NESTED_CONFIG
    type: Literal['none']


ToolChoice = Annotated[
    AutoToolChoice | AnyToolChoice | SpecificToolChoice | ToolChoiceNone,
    Field(discriminator='type'),
]


class RequestMetadata(BaseModel):
    """Metadata to include in the request.

    Uses no alias generator, so ``user_id`` stays snake_case.
    """

    model_config = _NESTED_CONFIG

    user_id: str | None = None


class AnthropicConfig(ModelConfig):
    """Typed configuration for Anthropic (Claude) models.

    Extends the shared :class:`ModelConfig` with Anthropic-specific options.
    JSON keys stay snake_case for ``tool_choice`` and ``output_config`` and
    camelCase elsewhere (``apiVersion``, inherited ``maxOutputTokens``).

    Claude requires a length cap, so an omitted ``max_output_tokens`` is sent
    as 4096; a reply cut off there ends with finish reason ``length``.

    ``extra`` is for Anthropic request fields this class doesn't declare
    (``service_tier``, ``container``, ...). Its keys are merged into the top
    level of the request body and win over the declared settings; a nested
    value such as ``extra={'thinking': {...}}`` replaces the whole setting
    rather than merging into it. A beta-only field in ``extra`` sends the
    call to the beta API, the same as ``betas`` does.
    """

    model_config = ConfigDict(json_schema_extra=_anthropic_config_schema_extra)

    # Config fields that are never create() kwargs. api_version picks the API
    # surface. (api_key was removed from ModelConfig in #6597).
    SDK_UNSUPPORTED_KEYS: ClassVar[frozenset[str]] = frozenset({'api_version'})

    thinking: Annotated[ThinkingConfig | None, WithJsonSchema(_THINKING_SCHEMA)] = Field(
        default=None,
    )
    output_config: Annotated[OutputConfig | None, WithJsonSchema(_OUTPUT_CONFIG_SCHEMA)] = Field(
        default=None,
        alias='output_config',
    )
    tool_choice: Annotated[ToolChoice | None, WithJsonSchema(_TOOL_CHOICE_SCHEMA)] = Field(
        default=None,
        alias='tool_choice',
    )
    metadata: Annotated[RequestMetadata | None, WithJsonSchema(_METADATA_SCHEMA)] = Field(
        default=None,
    )
    api_version: Literal['stable', 'beta'] | None = Field(
        default=None,
        description='Selects the Anthropic API surface for this request.',
    )
    betas: list[str] | None = Field(
        default=None,
        description='Anthropic beta feature headers to enable for this request. An empty list suppresses the defaults.',
    )

    def beta_only_fields(self) -> set[str]:
        """Return the names of beta-only request fields set on this config, including in ``extra``."""
        present = {
            name
            for name, value in (self.extra or {}).items()
            if name in BETA_ONLY_KEYS and name != 'betas' and value is not None
        }
        if self.betas:
            present.add('betas')
        if self.output_config is not None and self.output_config.task_budget is not None:
            present.add('output_config.task_budget')
        return present

    @model_validator(mode='after')
    def _check_api_surface(self) -> 'AnthropicConfig':
        """Reject beta-only fields on the stable surface so an explicit apiVersion is never silently overridden."""
        if self.api_version != 'stable':
            return self
        beta_only = self.beta_only_fields()
        if beta_only:
            names = ', '.join(sorted(beta_only))
            raise ValueError(f"{names} require the beta API surface; remove them or set apiVersion to 'beta'")
        return self
