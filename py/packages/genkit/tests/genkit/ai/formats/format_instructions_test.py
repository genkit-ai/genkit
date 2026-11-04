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

"""Which format instructions reach the prompt, and when a request goes out natively constrained."""

import pytest
from pydantic import BaseModel, ValidationError

from genkit import Genkit, Message, ModelResponse, Part
from genkit._ai._formats._json import JsonFormat
from genkit._ai._generate import apply_format
from genkit._core._model import GenerateActionOptions
from genkit._core._typing import ConstrainedSupport, GenerateActionOutputConfig, Role
from genkit.formats import FormatHandler, Formatter, FormatterConfig
from genkit.model import ModelContext, ModelInfo, ModelRequest, ModelSupports

CSV_INSTRUCTIONS = 'Reply as CSV with a header row: dish,price'
MENU_SCHEMA = {'type': 'object', 'properties': {'dish': {'type': 'string'}}}


class CsvFormat(Formatter):
    """A custom format whose handler has instructions."""

    def __init__(self) -> None:
        super().__init__(name='csv', config=FormatterConfig(format='csv'))

    def handle(self, schema: dict[str, object] | None) -> FormatHandler[str, str]:
        return FormatHandler(
            message_parser=lambda msg: msg.text,
            chunk_parser=lambda chunk: chunk.text,
            instructions=CSV_INSTRUCTIONS,
        )


class CitedJsonFormat(Formatter):
    """A custom JSON format on native JSON mode, with its own parsing and instructions."""

    def __init__(self) -> None:
        super().__init__(name='cited-json', config=FormatterConfig(format='json', constrained=True))

    def handle(self, schema: dict[str, object] | None) -> FormatHandler[str, str]:
        return FormatHandler(
            message_parser=lambda msg: msg.text,
            chunk_parser=lambda chunk: chunk.text,
            instructions='Reply as JSON with a "sources" list.',
        )


def _prompt_text(
    format_def: Formatter,
    fmt: str,
    instructions: bool | str | None,
    constrained: bool | None = None,
) -> str:
    options = GenerateActionOptions(
        messages=[Message(role=Role.USER, content=[Part.from_text('Lunch specials?')])],
        output=GenerateActionOutputConfig(
            format=fmt, json_schema=MENU_SCHEMA, instructions=instructions, constrained=constrained
        ),
    )
    out, _ = apply_format(options=options, format_def=format_def, model_supports_constrained=True)
    return ' '.join(part.text or '' for message in out.messages for part in message.content)


def test_handler_instructions_reach_the_prompt_by_default() -> None:
    """A format's handler instructions are added when the caller doesn't say otherwise."""
    assert CSV_INSTRUCTIONS in _prompt_text(CsvFormat(), 'csv', None)


def test_caller_false_leaves_handler_instructions_out() -> None:
    """output_instructions=False adds nothing."""
    assert CSV_INSTRUCTIONS not in _prompt_text(CsvFormat(), 'csv', False)


def test_caller_string_replaces_handler_instructions() -> None:
    """A caller's own string goes in instead of the handler's text."""
    text = _prompt_text(CsvFormat(), 'csv', 'One dish per line, no header.')

    assert 'One dish per line, no header.' in text
    assert CSV_INSTRUCTIONS not in text


def test_constrained_json_adds_schema_instructions_only_when_asked() -> None:
    """Natively constrained json follows the schema: no text by default, the handler's text with True."""
    default = _prompt_text(JsonFormat(), 'json', None)
    asked = _prompt_text(JsonFormat(), 'json', True)

    assert '"dish"' not in default
    assert '"dish"' in asked


def test_unconstrained_json_adds_schema_instructions() -> None:
    """With output_constrained=False the model needs the schema in the prompt, so it goes in."""
    assert '"dish"' in _prompt_text(JsonFormat(), 'json', None, constrained=False)


def test_caller_string_goes_in_under_native_mode() -> None:
    """An explicit string is always sent, even when native mode already enforces the schema."""
    assert 'Name the chef.' in _prompt_text(JsonFormat(), 'json', 'Name the chef.')


def test_custom_json_format_on_native_mode_behaves_like_json() -> None:
    """format='json', constrained=True on a custom format gets json's rule through public config."""
    assert 'sources' not in _prompt_text(CitedJsonFormat(), 'cited-json', None)
    assert 'sources' in _prompt_text(CitedJsonFormat(), 'cited-json', True)


def test_formatter_config_has_no_default_instructions() -> None:
    """The on/off knob is gone; a handler's instructions and the caller's override are the whole API."""
    with pytest.raises(ValidationError):
        FormatterConfig.model_validate({'default_instructions': False})


class Dish(BaseModel):
    """A menu item."""

    dish: str


def _define_menu_model(ai: Genkit, name: str, constrained: ConstrainedSupport | None) -> list[ModelRequest]:
    """Register a model that declares ``constrained`` and records what it receives."""
    seen: list[ModelRequest] = []

    async def menu_model(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        seen.append(request)
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('{"dish": "Ramen"}')]))

    ai.define_model(name=name, fn=menu_model, info=ModelInfo(supports=ModelSupports(constrained=constrained)))
    return seen


def _schema_in_prompt(request: ModelRequest) -> bool:
    return any('"dish"' in (part.text or '') for message in request.messages for part in message.content)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('constrained', 'with_tools', 'native'),
    [
        (ConstrainedSupport.ALL, False, True),
        (ConstrainedSupport.ALL, True, True),
        (ConstrainedSupport.NO_TOOLS, False, True),
        (ConstrainedSupport.NO_TOOLS, True, False),
        (ConstrainedSupport.NONE, False, False),
        (None, False, False),
    ],
)
async def test_json_uses_native_mode_only_when_the_model_supports_it(
    constrained: ConstrainedSupport | None, with_tools: bool, native: bool
) -> None:
    """json goes out constrained when the model can do it; otherwise the schema goes in the prompt."""
    # 1. A model that declares its native JSON support, plus a tool to optionally attach
    ai = Genkit()
    seen = _define_menu_model(ai, 'menuModel', constrained)

    @ai.tool()
    async def todays_specials() -> str:
        """List today's specials."""
        return 'Ramen'

    # 2. Ask for structured output
    response = await ai.generate(
        model='menuModel',
        prompt='Suggest a dish.',
        output_schema=Dish,
        tools=['todays_specials'] if with_tools else None,
    )

    # 3. Native mode and the prompt's schema text are opposites
    assert response.output == Dish(dish='Ramen')
    request = seen[-1]
    assert request.output is not None
    assert request.output.constrained is native
    assert _schema_in_prompt(request) is not native


@pytest.mark.asyncio
async def test_caller_constrained_false_wins_over_model_support() -> None:
    """output_constrained=False turns native mode off even on a model that has it."""
    ai = Genkit()
    seen = _define_menu_model(ai, 'menuModel', ConstrainedSupport.ALL)

    await ai.generate(model='menuModel', prompt='Suggest a dish.', output_schema=Dish, output_constrained=False)

    assert seen[-1].output is not None
    assert seen[-1].output.constrained is False
    assert _schema_in_prompt(seen[-1])
