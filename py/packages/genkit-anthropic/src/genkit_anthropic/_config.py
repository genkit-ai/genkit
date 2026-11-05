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
``maxOutputTokens``, ...) with flat Anthropic options. Choices are exported
Literals; the plugin builds Anthropic's nested request fields from them.

Unknown keys raise, so a typo fails before the request is sent. Anthropic
body fields this class doesn't declare go in ``extra`` and are sent through
the SDK's ``extra_body``.
"""

from collections.abc import Mapping
from typing import ClassVar, Literal, TypeAlias, cast, get_args

from anthropic.types.beta.message_create_params import MessageCreateParamsBase as BetaMessageCreateParamsBase
from anthropic.types.message_create_params import MessageCreateParamsBase
from pydantic import ConfigDict, Field, model_validator
from pydantic.config import JsonDict

from genkit import GenkitError
from genkit.model import ModelConfig

BETA_ONLY_KEYS = frozenset(BetaMessageCreateParamsBase.__annotations__) - frozenset(
    MessageCreateParamsBase.__annotations__
)

# Choice sets, exported from the package root. Literals, so callers pass plain
# strings (thinking='adaptive') and type checkers reject a value outside the set.
ThinkingMode: TypeAlias = Literal['enabled', 'adaptive', 'disabled']
ThinkingDisplay: TypeAlias = Literal['summarized', 'omitted']
Effort: TypeAlias = Literal['low', 'medium', 'high', 'xhigh', 'max']

# Nested config keys that were flattened, and what replaces each.
_REMOVED_KEYS = {
    'output_config': 'output_config was flattened; use effort= and task_budget=',
    'outputConfig': 'outputConfig was flattened; use effort= and task_budget=',
    'metadata': 'metadata was flattened; use user_id=',
    'tool_choice': (
        "tool_choice is no longer a config field; pass ai.generate(tool_choice='auto' | 'required' | 'none'), "
        "set disable_parallel_tool_use=True for one tool call per reply, or send Anthropic's object as "
        "extra={'tool_choice': {...}}"
    ),
}
_REMOVED_KEYS['toolChoice'] = _REMOVED_KEYS['tool_choice'].replace('tool_choice is', 'toolChoice is', 1)


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
                'disableParallelToolUse': {
                    'type': 'boolean',
                    'title': 'Disable parallel tool use',
                    'description': 'Allow at most one tool call per reply.',
                },
                'thinking': {
                    'type': 'string',
                    'enum': list(get_args(ThinkingMode)),
                    'title': 'Thinking',
                    'description': 'Extended thinking mode. A thinking budget alone means enabled.',
                },
                'thinkingBudget': {
                    'type': 'integer',
                    'minimum': 1024,
                    'title': 'Thinking budget',
                    'description': (
                        'Thinking token budget. Required when thinking is enabled; ignored when adaptive or disabled.'
                    ),
                },
                'thinkingDisplay': {
                    'type': 'string',
                    'enum': list(get_args(ThinkingDisplay)),
                    'title': 'Thinking display',
                    'description': (
                        'Whether the reply includes summarized thinking or omits it. Ignored when thinking is disabled.'
                    ),
                },
                'effort': {
                    'type': 'string',
                    'enum': list(get_args(Effort)),
                    'title': 'Effort',
                    'description': 'How much effort the model spends on the reply.',
                },
                'taskBudget': {
                    'type': 'integer',
                    'minimum': 20000,
                    'title': 'Task budget',
                    'description': 'Token budget for the whole task. Sent on the beta API.',
                },
                'userId': {
                    'type': 'string',
                    'title': 'User ID',
                    'description': 'Opaque end-user ID, sent as metadata.user_id.',
                },
            },
        )
    )


class AnthropicConfig(ModelConfig):
    """Typed configuration for Anthropic (Claude) models.

    Extends the shared :class:`ModelConfig` with flat Anthropic options. The
    plugin builds Anthropic's nested request fields from them:

    - ``thinking``, ``thinking_budget``, ``thinking_display`` → ``thinking``
    - ``effort``, ``task_budget`` → ``output_config``
    - ``user_id`` → ``metadata``

    JSON keys are camelCase (``thinkingBudget``, inherited ``maxOutputTokens``).

    Tool choice is Genkit's ``ai.generate(tool_choice=...)``; the plugin sends
    it as Anthropic's ``tool_choice`` object.

    Claude requires a length cap, so an omitted ``max_output_tokens`` is sent
    as 4096; a reply cut off there ends with finish reason ``length``.

    ``extra`` is for Anthropic request fields this class doesn't declare
    (``service_tier``, ``container``, ...). Its keys are merged into the top
    level of the request body and win over the declared settings; a nested
    value such as ``extra={'thinking': {...}}`` replaces the whole built
    object rather than merging into it. A beta-only field in ``extra`` sends
    the call to the beta API, the same as ``betas`` does. One exception:
    ``extra={'tool_choice': ...}`` is dropped on a request with no tools, the
    same as ``generate(tool_choice=)``, because the API rejects tool_choice
    without tools.

    Layered config: Genkit merges config key by key, ModelRef default first,
    then the prompt's config, then the call's; a layer overrides only the
    keys it sets, and ``None`` clears an inherited value. Three rules follow:

    1. Rules that span fields are judged once, on the merged config, right
       before the request is built. A layer is never rejected for something
       another layer could still supply. Single-field checks (type, bounds,
       unknown keys) still run on each layer.
    2. When one field is a switch and the others only apply in some modes,
       the switch wins: a setting the mode doesn't use isn't sent, and
       nothing fails. ``thinking='disabled'`` over an inherited
       ``thinking_budget`` turns thinking off.
    3. When two fields pick the same thing and neither is a switch, a
       conflict in the merged config fails before the request is sent, and
       the message names the ``None`` that clears the inherited one.
    """

    model_config = ConfigDict(json_schema_extra=_anthropic_config_schema_extra)

    # Fields sent to messages.create() under their own name. Every other field
    # is built into a request field by the plugin (thinking, output_config,
    # metadata, tool_choice, model, max_tokens, betas) or picks the API surface
    # (api_version).
    CREATE_KWARGS: ClassVar[frozenset[str]] = frozenset({'temperature', 'top_k', 'top_p', 'stop_sequences'})

    thinking: ThinkingMode | None = Field(
        default=None,
        description='Extended thinking mode. A thinking_budget alone means enabled.',
    )
    thinking_budget: int | None = Field(
        default=None,
        ge=1024,
        description='Thinking token budget. Required when thinking is enabled; ignored when adaptive or disabled.',
    )
    thinking_display: ThinkingDisplay | None = Field(
        default=None,
        description='Whether the reply includes summarized thinking or omits it. Ignored when thinking is disabled.',
    )
    effort: Effort | None = Field(default=None, description='How much effort the model spends on the reply.')
    task_budget: int | None = Field(
        default=None,
        ge=20000,
        description='Token budget for the whole task. Sent on the beta API.',
    )
    user_id: str | None = Field(default=None, description='Opaque end-user ID, sent as metadata.user_id.')
    disable_parallel_tool_use: bool | None = Field(
        default=None,
        description=(
            'Allow at most one tool call per reply. Sent inside tool_choice; ignored when tool_choice is none.'
        ),
    )
    api_version: Literal['stable', 'beta'] | None = Field(
        default=None,
        description='Selects the Anthropic API surface for this request.',
    )
    betas: list[str] | None = Field(
        default=None,
        description='Anthropic beta feature headers to enable for this request. An empty list suppresses the defaults.',
    )

    @model_validator(mode='before')
    @classmethod
    def _name_replacements_for_removed_keys(cls, data: object) -> object:
        """Name the replacement for a removed key instead of the generic unknown-key error."""
        if not isinstance(data, Mapping):
            return data
        for key, value in data.items():
            if key == 'thinking' and isinstance(value, Mapping):
                message = (
                    "thinking was flattened; use thinking='enabled' | 'adaptive' | 'disabled', "
                    'thinking_budget= and thinking_display='
                )
            elif key in _REMOVED_KEYS:
                message = _REMOVED_KEYS[key]
            else:
                continue
            raise GenkitError(status='INVALID_ARGUMENT', message=f'anthropic: {message}')
        return data

    # How the flattened groups behave when config layers merge. See "Layered
    # config" in the class docstring for the three rules.
    #
    # - thinking / thinking_budget / thinking_display (rule 2): `thinking` is
    #   the switch. 'disabled' ignores the budget and the display; 'adaptive'
    #   ignores the budget. A call that sets thinking='disabled' over a default
    #   of thinking='enabled', thinking_budget=4096 turns thinking off instead
    #   of failing. The budget stays in the merged config, so a later
    #   thinking='enabled' picks it back up. _to_anthropic_thinking leaves the
    #   ignored fields out of the request. The one case this can't read: a
    #   call that sets only thinking_budget over an inherited 'disabled'
    #   merges to the same values as the case above, so it stays disabled;
    #   turning thinking on takes thinking='enabled'.
    # - effort / task_budget: independent; overriding one keeps the other.
    # - user_id: one field.
    # - disable_parallel_tool_use: ignored under generate(tool_choice='none'),
    #   since Anthropic's none variant has no such field (rule 2).
    # - api_version vs task_budget / betas (rule 3): neither side is a switch
    #   for the other. Dropping the beta field would quietly lose a spend cap
    #   or a feature; switching to beta would override an explicit 'stable'.
    #   So the merged config fails and the message names both ways out.

    def thinking_mode(self) -> ThinkingMode | None:
        """The thinking mode the request sends; a budget alone means enabled."""
        if self.thinking is not None:
            return self.thinking
        return 'enabled' if self.thinking_budget is not None else None

    def beta_only_fields(self) -> set[str]:
        """Return the names of beta-only request fields set on this config, including in ``extra``."""
        present = {
            name
            for name, value in (self.extra or {}).items()
            if name in BETA_ONLY_KEYS and name != 'betas' and value is not None
        }
        if self.betas:
            present.add('betas')
        if self.task_budget is not None:
            present.add('task_budget')
        return present

    def check_merged(self) -> None:
        """Raise ``ValueError`` if the merged config can't be sent (rule 1).

        These rules span fields, so they aren't model validators: a validator
        also runs on each layer by itself, and would reject
        ``thinking='enabled'`` in a call whose default supplies the budget.
        The plugin calls this once, on the config the request is built from.
        """
        # 'enabled' can't be sent without budget_tokens, and no layer set one.
        if self.thinking == 'enabled' and self.thinking_budget is None:
            raise ValueError("thinking_budget is required when thinking is 'enabled'")
        # No mode and no budget means no thinking object to carry the display,
        # so it would be dropped without a word.
        if self.thinking_display is not None and self.thinking_mode() is None:
            raise ValueError('thinking_display needs thinking= or thinking_budget=')
        # Rule 3: an explicit 'stable' and a beta-only field conflict.
        if self.api_version == 'stable':
            beta_only = self.beta_only_fields()
            if beta_only:
                names = ', '.join(sorted(beta_only))
                raise ValueError(
                    f"{names} require the beta API surface; set apiVersion to 'beta' or clear them with None"
                )
