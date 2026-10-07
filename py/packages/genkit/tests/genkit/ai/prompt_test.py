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


"""Tests for the action module."""

import inspect
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import ANY, MagicMock, patch

import pytest
from pydantic import BaseModel, Field, ValidationError

from genkit import (
    Document,
    FinishReason,
    Genkit,
    Message,
    ModelResponse,
    ModelResponseChunk,
    Part,
    Prompt,
    ToolRunContext,
    tool,
)
from genkit._ai._model import ModelRequest, text_from_message
from genkit._ai._prompt import (
    GenerateCall,
    ModelSettings,
    PromptGenerateOptions,
    PromptSettings,
    _parse_dotprompt_use,
    load_prompt_folder,
    lookup_prompt,
    prompt,
)
from genkit._core._action import Action, ActionKind
from genkit._core._dap import DapValue
from genkit._core._error import GenkitError, RuntimeErrorReason
from genkit._core._model import GenerateActionOptions, ModelConfig, resume_options_to_resume
from genkit._core._registry import define_dynamic_action_provider
from genkit._core._typing import Role
from genkit.middleware import BaseMiddleware, GenerateMiddlewareContext, MiddlewareRef, ModelHookParams
from genkit.plugin_api import MiddlewarePlugin, new_middleware
from genkit.testing import (
    EchoModel,
    ScriptedModel,
    define_echo_model,
    define_scripted_model,
)


class _PreMiddleware(BaseMiddleware):
    async def wrap_model(
        self,
        params: ModelHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        txt = ''.join(text_from_message(m) for m in params.request.messages)
        return await next_fn(
            ModelHookParams(
                request=ModelRequest(
                    messages=[Message(role=Role.USER, content=[Part.from_text(f'PRE {txt}')])],
                ),
            ),
            ctx,
        )


class _PostMiddleware(BaseMiddleware):
    async def wrap_model(
        self,
        params: ModelHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        resp: ModelResponse = await next_fn(params, ctx)
        assert resp.message is not None
        txt = text_from_message(resp.message)
        return ModelResponse(
            finish_reason=resp.finish_reason,
            message=Message(role=Role.USER, content=[Part.from_text(f'{txt} POST')]),
        )


class PrePostMiddlewarePlugin(MiddlewarePlugin):
    name = 'extension-middleware'
    middleware = [
        new_middleware(_PreMiddleware, name='pre_mw'),
        new_middleware(_PostMiddleware, name='post_mw'),
    ]


def setup_test() -> tuple[Genkit, EchoModel, ScriptedModel]:
    """Setup a test fixture for the prompt tests."""
    ai = Genkit(model='echoModel')

    pm, _ = define_scripted_model(ai)
    echo, _ = define_echo_model(ai)

    return (ai, echo, pm)


@pytest.mark.asyncio
async def test_simple_prompt() -> None:
    """Test simple prompt rendering."""
    ai, *_ = setup_test()

    want_txt = '[ECHO] user: "hi" {"temperature":11}'

    my_prompt = ai.define_prompt(prompt='hi', config={'temperature': 11})

    response = await my_prompt()

    assert response.text == want_txt

    # New API: stream returns ModelStreamResponse with .response property
    result = my_prompt.stream()

    assert (await result.response).text == want_txt


@pytest.mark.asyncio
async def test_simple_prompt_with_override_config() -> None:
    """Config passed on the call merges over the prompt's config instead of replacing it."""
    ai, *_ = setup_test()

    # Config is MERGED: prompt config (banana: true) + opts config (temperature: 12)
    want_txt = '[ECHO] user: "hi" {"banana":true,"temperature":12}'

    # banana is a pass-through test key, not a ModelConfigDict field
    prompt_config: dict[str, Any] = {'banana': True}
    my_prompt = ai.define_prompt(prompt='hi', config=prompt_config)

    # Pass config via kwargs — this MERGES with prompt config
    response = await my_prompt(config={'temperature': 12})

    assert response.text == want_txt

    # stream() also accepts the same kwargs
    result = my_prompt.stream(config={'temperature': 12})

    assert (await result.response).text == want_txt


@pytest.mark.asyncio
async def test_prompt_tool_choice_string_reaches_the_model() -> None:
    """A string tool_choice on the call overrides the prompt's and reaches the model."""
    ai, *_ = setup_test()

    my_prompt = ai.define_prompt(prompt='hi', tool_choice='required')
    response = await my_prompt(tool_choice='none')

    assert response.request is not None
    assert response.request.tool_choice == 'none'


@pytest.mark.asyncio
async def test_prompt_with_system() -> None:
    """Test that the prompt utilises both prompt and system prompt."""
    ai, *_ = setup_test()

    want_txt = '[ECHO] system: "talk like a pirate" user: "hi"'

    my_prompt = ai.define_prompt(prompt='hi', system='talk like a pirate')

    response = await my_prompt()

    assert response.text == want_txt

    # New API: stream returns ModelStreamResponse
    result = my_prompt.stream()

    assert (await result.response).text == want_txt


@pytest.mark.asyncio
async def test_prompt_with_kitchensink() -> None:
    """Test that the rendering works with all the options."""
    ai, *_ = setup_test()

    class PromptInput(BaseModel):
        name: str | None = Field(default=None, description='the name')

    class ToolInput(BaseModel):
        value: int | None = Field(default=None, description='value field')

    @ai.tool(name='testTool')
    async def test_tool(input: ToolInput) -> str:
        """The tool."""
        return 'abc'

    my_prompt = ai.define_prompt(
        system='pirate',
        prompt='hi',
        messages=[Message(role=Role.USER, content=[Part.from_text('history')])],
        tools=['testTool'],
        tool_choice='required',
        max_turns=5,
        input_schema=PromptInput.model_json_schema(),
        output_constrained=True,
        output_format='json',
        description='a prompt descr',
    )

    want_txt = (
        '[ECHO] system: "pirate" user: "history" user: "hi" tools=testTool '
        'tool_choice=required output={"format":"json","constrained":true,'
        '"contentType":"application/json"}'
    )

    response = await my_prompt()

    assert response.text == want_txt

    # New API: stream returns ModelStreamResponse
    result = my_prompt.stream()

    assert (await result.response).text == want_txt


test_cases_parse_partial_json = [
    (
        'renders system prompt',
        {
            'model': 'echoModel',
            'config': {'banana': 'ripe'},
            'input_schema': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                },
            },  # Note: Schema representation might need adjustment
            'system': 'hello {{name}} ({{@state.name}})',
            'metadata': {'state': {'name': 'bar'}},
        },
        {'name': 'foo'},
        ModelConfig.model_validate({'temperature': 11}),
        {},
        # Config is MERGED: prompt config (banana: ripe) + opts config (temperature: 11)
        """[ECHO] system: "hello foo (bar)" {"banana":"ripe","temperature":11.0}""",
    ),
    (
        'renders user prompt',
        {
            'model': 'echoModel',
            'config': {'banana': 'ripe'},
            'input_schema': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                },
            },  # Note: Schema representation might need adjustment
            'prompt': 'hello {{name}} ({{@state.name}})',
            'metadata': {'state': {'name': 'bar_system'}},
        },
        {'name': 'foo'},
        ModelConfig.model_validate({'temperature': 11}),
        {},
        # Config is MERGED: prompt config (banana: ripe) + opts config (temperature: 11)
        """[ECHO] user: "hello foo (bar_system)" {"banana":"ripe","temperature":11.0}""",
    ),
    (
        'renders user prompt with context',
        {
            'model': 'echoModel',
            'config': {'banana': 'ripe'},
            'input_schema': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                },
            },  # Note: Schema representation might need adjustment
            'prompt': 'hello {{name}} ({{@state.name}}, {{@auth.email}})',
            'metadata': {'state': {'name': 'bar'}},
        },
        {'name': 'foo'},
        ModelConfig.model_validate({'temperature': 11}),
        {'auth': {'email': 'a@b.c'}},
        # Config is MERGED: prompt config (banana: ripe) + opts config (temperature: 11)
        """[ECHO] user: "hello foo (bar, a@b.c)" {"banana":"ripe","temperature":11.0}""",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'test_case, prompt, input, input_option, context, want_rendered',
    test_cases_parse_partial_json,
    ids=[tc[0] for tc in test_cases_parse_partial_json],
)
async def test_prompt_rendering_dotprompt(
    test_case: str,
    prompt: dict[str, Any],
    input: dict[str, Any],
    input_option: ModelConfig,
    context: dict[str, Any],
    want_rendered: str,
) -> None:
    """Test prompt rendering."""
    ai, *_ = setup_test()

    my_prompt = ai.define_prompt(**prompt)

    # New API: use kwargs parameter to pass config and context
    response = await my_prompt(input, config=input_option, context=context)

    assert response.text == want_rendered


# Tests for prompt variants and partials
@pytest.mark.asyncio
async def test_load_prompt_variant() -> None:
    """Test loading and using a prompt variant."""
    ai, *_ = setup_test()

    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / 'prompts'
        prompt_dir.mkdir()

        # Create base prompt
        base_prompt = prompt_dir / 'greeting.prompt'
        base_prompt.write_text('---\nmodel: echoModel\n---\nHello {{name}}!')

        # Create variant prompt
        variant_prompt = prompt_dir / 'greeting.casual.prompt'
        variant_prompt.write_text("---\nmodel: echoModel\n---\nHey {{name}}, what's up?")

        load_prompt_folder(ai.registry, prompt_dir)

        # Test base prompt
        base_exec = await prompt(ai.registry, 'greeting')
        base_response = await base_exec({'name': 'Alice'})
        assert 'Hello' in base_response.text
        assert 'Alice' in base_response.text

        # Test variant prompt
        casual_exec = await prompt(ai.registry, 'greeting', variant='casual')
        casual_response = await casual_exec({'name': 'Bob'})
        assert 'Hey' in casual_response.text or "what's up" in casual_response.text.lower()
        assert 'Bob' in casual_response.text


@pytest.mark.asyncio
async def test_load_nested_prompt() -> None:
    """Test loading prompts from subdirectories."""
    ai, *_ = setup_test()

    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / 'prompts'
        prompt_dir.mkdir()

        # Create subdirectory
        sub_dir = prompt_dir / 'admin'
        sub_dir.mkdir()

        # Create prompt in subdirectory
        admin_prompt = sub_dir / 'dashboard.prompt'
        admin_prompt.write_text('---\nmodel: echoModel\n---\nWelcome Admin {{name}}')

        load_prompt_folder(ai.registry, prompt_dir)

        # Test loading nested prompt
        # Based on logic: name = "admin/dashboard"
        admin_exec = await prompt(ai.registry, 'admin/dashboard')
        response = await admin_exec({'name': 'SuperUser'})

        assert 'Welcome Admin' in response.text
        assert 'SuperUser' in response.text


@pytest.mark.asyncio
async def test_load_and_use_partial() -> None:
    """Test loading and using partials in prompts."""
    ai, *_ = setup_test()

    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / 'prompts'
        prompt_dir.mkdir()

        # Create partial
        partial_file = prompt_dir / '_greeting.prompt'
        partial_file.write_text('Hello from partial!')

        # Create prompt that uses the partial
        prompt_file = prompt_dir / 'story.prompt'
        prompt_file.write_text('---\nmodel: echoModel\n---\n{{>greeting}} Tell me about {{topic}}.')

        load_prompt_folder(ai.registry, prompt_dir)

        story_exec = await prompt(ai.registry, 'story')
        response = await story_exec({'topic': 'space'})

        # The partial should be included in the output
        assert 'Hello from partial' in response.text or 'space' in response.text


@pytest.mark.asyncio
async def test_define_partial_programmatically() -> None:
    """Test defining partials programmatically using ai.define_partial()."""
    ai, *_ = setup_test()

    # Define a partial programmatically
    ai.define_partial('myGreeting', 'Greetings, {{name}}!')

    # Create a prompt that uses the partial
    my_prompt = ai.define_prompt(
        messages='{{>myGreeting}} Welcome to Genkit.',
    )

    response = await my_prompt(input={'name': 'Developer'})

    # The partial should be included in the output
    assert 'Greetings' in response.text and 'Developer' in response.text


@pytest.mark.asyncio
async def test_prompt_with_tools_list() -> None:
    """Test prompt with tools parameter."""
    ai, *_ = setup_test()

    class ToolInput(BaseModel):
        value: int = Field(description='A value')

    @ai.tool(name='myTool')
    async def my_tool(input: ToolInput) -> int:
        return input.value * 2

    my_prompt = ai.define_prompt(
        prompt='Use the tool',
        tools=['myTool'],
    )

    rendered = await my_prompt.render()

    # Verify tools are in the rendered options
    assert rendered.tools is not None
    assert 'myTool' in rendered.tools


@pytest.mark.asyncio
async def test_prompt_action_binds_dap_selector() -> None:
    """PROMPT action expands ``mcp:tool/echo`` before resolve_tool."""
    ai, *_ = setup_test()

    async def echo_fn(x: str) -> str:
        return x

    echo = Action(name='echo', kind=ActionKind.TOOL, fn=echo_fn, metadata={'name': 'echo'})

    async def dap_fn() -> DapValue:
        return {'tool': [echo]}

    define_dynamic_action_provider(ai.registry, 'mcp', dap_fn)

    ai.define_prompt(name='withDap', prompt='ping', tools=['mcp:tool/echo'])
    prompt_action = await ai.registry.resolve_action(ActionKind.PROMPT, 'withDap')
    assert prompt_action is not None

    result = await prompt_action.run()
    request = result.response
    assert isinstance(request, ModelRequest)
    assert request.tools is not None
    assert [t.name for t in request.tools] == ['echo']
    assert 'echo' not in ai.registry._entries.get(ActionKind.TOOL, {})


@pytest.mark.asyncio
async def test_prompt_with_output_schema() -> None:
    """Test that output schema is preserved in rendering."""
    ai, *_ = setup_test()

    class OutputSchema(BaseModel):
        name: str = Field(description='A name')
        age: int = Field(description='An age')

    my_prompt = ai.define_prompt(
        prompt='Generate a person',
        output_schema=OutputSchema,
        output_format='json',
    )

    rendered = await my_prompt.render()

    # Verify output configuration
    assert rendered.output is not None
    assert rendered.output.format == 'json'
    assert rendered.output.json_schema is not None


@pytest.mark.asyncio
async def test_config_merge_priority() -> None:
    """Test that runtime config is MERGED with definition config.

    opts.config values override prompt config values, but prompt config values
    that aren't in opts.config are preserved.
    """
    ai, *_ = setup_test()

    # banana is a pass-through test key, not a ModelConfigDict field
    prompt_config: dict[str, Any] = {'temperature': 0.5, 'banana': 'yellow'}
    my_prompt = ai.define_prompt(
        prompt='test',
        config=prompt_config,
    )

    # New API: runtime config is MERGED with prompt config
    # - temperature: 0.9 (from opts, overrides 0.5)
    # - banana: 'yellow' (from prompt, preserved)
    rendered = await my_prompt.render(config={'temperature': 0.9})

    assert rendered.config is not None
    # Config is now a dict after merging
    assert rendered.config['temperature'] == 0.9
    assert rendered.config['banana'] == 'yellow'  # Preserved from prompt config


@pytest.mark.asyncio
async def test_prompt_call_model_keyword_switches_model() -> None:
    """`await p(model='scriptedModel')` runs the other model."""
    ai, _, pm = setup_test()

    pm.responses = [ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('pm response')]))]

    my_prompt = ai.define_prompt(
        model='echoModel',
        prompt='hello',
    )

    # Override model via kwargs
    response = await my_prompt(model='scriptedModel')

    # Should use scriptedModel, not echoModel
    assert response.text == 'pm response'


@pytest.mark.asyncio
async def test_generate_stream_response_api() -> None:
    """Test that ModelStreamResponse provides both stream and response."""
    ai, *_ = setup_test()

    my_prompt = ai.define_prompt(
        prompt='hello world',
    )

    # Get stream response
    result = my_prompt.stream()

    assert hasattr(result, 'stream')
    assert hasattr(result, 'response')

    # Stream may not have chunks (depends on model implementation),
    # but we can always await the response
    async for _ in result.stream:
        pass  # Consume stream if any chunks

    # Get final response - this should always work
    final_response = await result.response

    # Final response should be complete
    assert final_response.text is not None
    assert 'hello world' in final_response.text


@pytest.mark.asyncio
async def test_prompt_input_positional_opts_as_kwargs() -> None:
    """Prompt: input is positional, opts via kwargs after *."""
    ai, *_ = setup_test()

    my_prompt = ai.define_prompt(
        prompt='Recipe for {{cuisine}} {{dish}}',
        output_format='text',
    )

    assert isinstance(my_prompt, Prompt)

    rendered = await my_prompt.render(
        {'cuisine': 'Italian', 'dish': 'pasta'},
        config={'temperature': 0.1},
    )

    assert any('Italian' in str(m) for m in rendered.messages)
    assert any('pasta' in str(m) for m in rendered.messages)
    assert rendered.config is not None
    assert rendered.config['temperature'] == 0.1
    assert rendered.output is not None
    assert rendered.output.format == 'text'


class Recipe(BaseModel):
    title: str


class _Dish(BaseModel):
    name: str


class OtherOutput(BaseModel):
    score: int


class OtherRecipe(BaseModel):
    title: str


class ChefInput(BaseModel):
    food: str | None = None
    style: str | None = None


class RequiredFood(BaseModel):
    food: str


class RequiredDiet(BaseModel):
    diet: str
    food: str | None = None


class RamenOrder(BaseModel):
    dish: str
    size: str = 'regular'


_RECIPE_PROMPT = """---
input:
  schema:
    food: string
    style?: string
  default:
    food: banana bread
output:
  schema: Recipe
---
Make {{food}}{{#if style}} {{style}}{{/if}}.
"""

_RECIPE_JSON_PROMPT = """---
input:
  schema:
    food: string
  default:
    food: banana bread
output:
  schema: RecipeJson
---
Make {{food}}.
"""

_INLINE_OUTPUT_PROMPT = """---
input:
  schema:
    food: string
  default:
    food: banana bread
output:
  schema:
    title: string
---
Make {{food}}.
"""

_NO_DEFAULT_PROMPT = """---
input:
  schema:
    food: string
---
Make {{food}}.
"""

_NULLABLE_FOOD_PROMPT = """---
input:
  schema:
    food?: string
  default:
    food: banana bread
---
Make {{food}}.
"""

_RAMEN_ORDER_PROMPT = """---
input:
  schema:
    dish: string
    size?: string
  default:
    size: large
---
{{size}} {{dish}}
"""


def _prompt_file_ai(
    *files: tuple[str, str],
) -> tuple[Genkit, ScriptedModel, tempfile.TemporaryDirectory[str]]:
    """A Genkit that loads the given ``.prompt`` files and replies ``{"title": "pie"}``."""
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    for name, body in files:
        (root / name).write_text(body)
    ai = Genkit(prompt_dir=str(root), model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_text_reply('{"title": "pie"}')]
    return ai, pm, tmp


def _assert_invalid_prompt_input(error: GenkitError, *, prompt_name: str, field: str) -> None:
    assert error.status == 'INVALID_ARGUMENT'
    assert error.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'INVALID_INPUT' not in error.original_message
    assert f"Invalid input for action '{prompt_name}'" in error.original_message
    assert field in error.original_message


def _text_reply(text: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP, message=Message(role=Role.MODEL, content=[Part.from_text(text)])
    )


def _tool_call_reply(name: str, ref: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part.from_tool_request(name=name, input={}, ref=ref)]),
    )


def _setup_prompt_call() -> tuple[Genkit, ScriptedModel]:
    """A Genkit whose default model records each request and has `oven` and `grill` tools."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_text_reply('ok')]

    @ai.tool(name='oven')
    async def oven() -> str:
        return 'baked'

    @ai.tool(name='grill')
    async def grill() -> str:
        return 'grilled'

    return ai, pm


def _sent_tool_names(pm: ScriptedModel) -> list[str]:
    assert pm.last_request is not None
    return [t.name for t in pm.last_request.tools or []]


def _sent_doc_texts(pm: ScriptedModel) -> list[str]:
    assert pm.last_request is not None
    return [d.text for d in pm.last_request.docs or []]


@pytest.mark.asyncio
async def test_prompt_call_returns_the_defined_output_schema_type() -> None:
    """A prompt defined with `output_schema=Recipe` returns `res.output` as a `Recipe`, no per-call output needed."""
    ai, pm = _setup_prompt_call()
    pm.responses = [_text_reply('{"title": "pie"}')]
    recipe = ai.define_prompt(prompt='Make {{dish}}', output_schema=Recipe)

    res = await recipe({'dish': 'pie'})

    assert res.output == Recipe(title='pie')


# Keywords a call must reject: output and template are fixed by the definition,
# metadata and timeout aren't call options, `tool` is a typo of `tools`, and
# only `await prompt(...)` takes on_chunk.
_REJECTED_KEYWORDS = [
    (method, keyword)
    for method in ('__call__', 'stream', 'render')
    for keyword in ('output_schema', 'output_format', 'output', 'prompt', 'system', 'metadata', 'timeout', 'tool')
] + [('stream', 'on_chunk'), ('render', 'on_chunk')]


@pytest.mark.asyncio
@pytest.mark.parametrize(('method', 'keyword'), _REJECTED_KEYWORDS, ids=[f'{m}-{k}' for m, k in _REJECTED_KEYWORDS])
async def test_prompt_call_rejects_keyword(method: str, keyword: str) -> None:
    """A keyword the call doesn't take raises `TypeError` naming it."""
    ai, _ = _setup_prompt_call()
    recipe = ai.define_prompt(prompt='Make pie', output_schema=Recipe, tools=['oven'])

    with pytest.raises(TypeError, match=f"'{keyword}'"):
        if method == 'stream':
            recipe.stream(**{keyword: 'x'})
        else:
            await getattr(recipe, method)(**{keyword: 'x'})


@pytest.mark.asyncio
async def test_prompt_call_on_chunk_receives_each_chunk() -> None:
    """`await p(on_chunk=cb)` calls `cb` once per streamed chunk (control)."""
    ai, pm = _setup_prompt_call()
    pm.chunks = [[ModelResponseChunk(content=[Part.from_text('a')]), ModelResponseChunk(content=[Part.from_text('b')])]]
    p = ai.define_prompt(prompt='hi')
    seen: list[str] = []

    await p(on_chunk=lambda chunk: seen.append(chunk.text))

    assert seen == ['a', 'b']


@pytest.mark.asyncio
async def test_prompt_call_empty_tools_sends_no_tools() -> None:
    """`await p(tools=[])` sends the model no tools even though the prompt defines `['oven']`."""
    ai, pm = _setup_prompt_call()
    p = ai.define_prompt(prompt='hi', tools=['oven'])

    await p(tools=[])

    assert _sent_tool_names(pm) == []


@pytest.mark.asyncio
async def test_prompt_call_tools_list_replaces_prompt_tools() -> None:
    """`await p(tools=['grill'])` sends only `grill`."""
    ai, pm = _setup_prompt_call()
    p = ai.define_prompt(prompt='hi', tools=['oven'])

    await p(tools=['grill'])

    assert _sent_tool_names(pm) == ['grill']


@pytest.mark.asyncio
async def test_prompt_call_omitted_tools_keeps_prompt_tools() -> None:
    """`await p()` sends the prompt's `oven` tool (control)."""
    ai, pm = _setup_prompt_call()
    p = ai.define_prompt(prompt='hi', tools=['oven'])

    await p()

    assert _sent_tool_names(pm) == ['oven']


@pytest.mark.asyncio
async def test_prompt_call_empty_use_runs_no_middleware() -> None:
    """`await p(use=[])` runs none of the prompt's middleware."""
    ai, *_ = setup_test()
    p = ai.define_prompt(prompt='hi', use=[_PreMiddleware(), _PostMiddleware()])

    res = await p(use=[])

    assert res.text == '[ECHO] user: "hi"'


@pytest.mark.asyncio
async def test_prompt_call_omitted_use_runs_prompt_middleware() -> None:
    """`await p()` runs the prompt's middleware (control)."""
    ai, *_ = setup_test()
    p = ai.define_prompt(prompt='hi', use=[_PreMiddleware(), _PostMiddleware()])

    res = await p()

    assert res.text == '[ECHO] user: "PRE hi" POST'


@pytest.mark.asyncio
async def test_prompt_call_empty_docs_sends_no_docs() -> None:
    """`await p(docs=[])` sends no docs, even though the prompt defines one."""
    ai, pm = _setup_prompt_call()
    p = ai.define_prompt(prompt='hi', docs=[Document.from_text('prompt doc')])

    await p(docs=[])

    assert _sent_doc_texts(pm) == []


@pytest.mark.asyncio
async def test_prompt_call_docs_list_replaces_prompt_docs() -> None:
    """`await p(docs=[other])` sends only `other`, not the prompt's doc plus `other`."""
    ai, pm = _setup_prompt_call()
    p = ai.define_prompt(prompt='hi', docs=[Document.from_text('prompt doc')])

    await p(docs=[Document.from_text('other doc')])

    assert _sent_doc_texts(pm) == ['other doc']


@pytest.mark.asyncio
async def test_prompt_call_omitted_docs_keeps_prompt_docs() -> None:
    """`await p()` sends the prompt's doc (control)."""
    ai, pm = _setup_prompt_call()
    p = ai.define_prompt(prompt='hi', docs=[Document.from_text('prompt doc')])

    await p()

    assert _sent_doc_texts(pm) == ['prompt doc']


@pytest.mark.asyncio
async def test_prompt_call_config_merges_over_prompt_config() -> None:
    """`await p(config={'temperature': 0.9})` keeps the prompt's other config keys and replaces temperature."""
    ai, pm = _setup_prompt_call()
    p = ai.define_prompt(prompt='hi', config={'temperature': 0.5, 'top_k': 3})

    await p(config={'temperature': 0.9})

    assert pm.last_request is not None
    assert pm.last_request.config == {'temperature': 0.9, 'top_k': 3}


@pytest.mark.asyncio
async def test_prompt_call_return_tool_requests_false_overrides_prompt() -> None:
    """A prompt defined with `return_tool_requests=True`, called with `False`, runs the tool loop."""
    ai, pm = _setup_prompt_call()
    pm.responses = [_tool_call_reply('oven', 'r1'), _text_reply('done')]
    p = ai.define_prompt(prompt='hi', tools=['oven'], return_tool_requests=True)

    res = await p(return_tool_requests=False)

    assert res.text == 'done'
    assert pm.request_count == 2


@pytest.mark.asyncio
async def test_prompt_call_max_turns_overrides_prompt() -> None:
    """`await p(max_turns=1)` stops after one tool round even when the prompt says 5."""
    ai, pm = _setup_prompt_call()
    pm.responses = [_tool_call_reply('oven', str(i)) for i in range(6)]
    p = ai.define_prompt(prompt='hi', tools=['oven'], max_turns=5)

    res = await p(max_turns=1)

    assert res.finish_reason == FinishReason.ABORTED
    assert pm.request_count == 2


@pytest.mark.asyncio
async def test_prompt_stream_empty_tools_sends_no_tools() -> None:
    """`p.stream(tools=[])` follows the same "empty clears" rule as the call."""
    ai, pm = _setup_prompt_call()
    p = ai.define_prompt(prompt='hi', tools=['oven'])

    await p.stream(tools=[]).response

    assert _sent_tool_names(pm) == []


@pytest.mark.asyncio
async def test_prompt_render_empty_docs_renders_no_docs() -> None:
    """`await p.render(docs=[])` returns options with no docs."""
    ai, _ = _setup_prompt_call()
    p = ai.define_prompt(prompt='hi', docs=[Document.from_text('prompt doc')])

    rendered = await p.render(docs=[])

    assert not rendered.docs


@pytest.mark.asyncio
async def test_prompt_call_use_adds_middleware_to_prompt_without_any() -> None:
    """`await p(use=[...])` runs middleware on a prompt defined with none."""
    ai, *_ = setup_test()
    p = ai.define_prompt(prompt='hi')

    res = await p(use=[_PreMiddleware(), _PostMiddleware()])

    assert res.text == '[ECHO] user: "PRE hi" POST'


@pytest.mark.asyncio
async def test_prompt_call_use_list_replaces_prompt_middleware() -> None:
    """A non-empty `use=` replaces the prompt's middleware; it doesn't stack on top."""
    ai, *_ = setup_test()
    p = ai.define_prompt(prompt='hi', use=[_PreMiddleware()])

    res = await p(use=[_PostMiddleware()])

    assert res.text == '[ECHO] user: "hi" POST'


@pytest.mark.asyncio
async def test_looked_up_prompt_call_use_runs_middleware() -> None:
    """`ai.prompt(name)(use=[...])` applies per-call middleware too."""
    ai, *_ = setup_test()
    ai.define_prompt(name='greeting', prompt='hi')

    res = await ai.prompt('greeting')(use=[_PreMiddleware(), _PostMiddleware()])

    assert res.text == '[ECHO] user: "PRE hi" POST'


@pytest.mark.asyncio
async def test_prompt_call_return_tool_requests_true_overrides_prompt() -> None:
    """A prompt defined with `return_tool_requests=False`, called with `True`, stops at the tool request."""
    ai, pm = _setup_prompt_call()
    pm.responses = [_tool_call_reply('oven', 'r1'), _text_reply('done')]
    p = ai.define_prompt(prompt='hi', tools=['oven'], return_tool_requests=False)

    res = await p(return_tool_requests=True)

    assert pm.request_count == 1
    assert [r.tool_request.name for r in res.tool_requests if r.tool_request] == ['oven']


@pytest.mark.asyncio
async def test_prompt_render_omitted_tool_choice_keeps_prompt_value() -> None:
    """No `tool_choice=` on the call keeps the prompt's."""
    ai, _ = _setup_prompt_call()
    p = ai.define_prompt(prompt='hi', tools=['oven'], tool_choice='required')

    rendered = await p.render()

    assert rendered.tool_choice == 'required'


@pytest.mark.asyncio
async def test_prompt_render_docs_added_to_prompt_without_any() -> None:
    """`docs=` on a prompt defined with none reaches the request."""
    ai, _ = _setup_prompt_call()
    p = ai.define_prompt(prompt='hi')

    rendered = await p.render(docs=[Document.from_text('allergen chart')])

    assert [d.text for d in rendered.docs or []] == ['allergen chart']


@pytest.mark.asyncio
async def test_prompt_call_model_and_config_reach_the_new_model() -> None:
    """Switching model and passing config on the call sends that config to the new model."""
    ai, pm = _setup_prompt_call()
    p = ai.define_prompt(model='echoModel', prompt='hi')

    res = await p(model='scriptedModel', config={'temperature': 0.2})

    assert pm.request_count == 1
    assert res.request is not None
    assert res.request.config == {'temperature': 0.2}


@pytest.mark.asyncio
async def test_prompt_render_resume_options_pass_through() -> None:
    """`resume_*` on the call reach the rendered request; the prompt never defines them."""
    ai, _ = _setup_prompt_call()
    p = ai.define_prompt(prompt='hi', tools=['oven'])
    answer = Part.from_tool_response(name='oven', output={'temp': 400}, ref='r1')
    rerun = Part.from_tool_request(name='oven', input={}, ref='r2')

    rendered = await p.render(resume_respond=[answer], resume_restart=[rerun], resume_metadata={'approved_by': 'chef'})

    assert rendered.resume is not None
    assert rendered.resume.respond == [answer]
    assert rendered.resume.restart == [rerun]
    assert rendered.resume.metadata == {'approved_by': 'chef'}


def test_with_overrides_none_keeps_definition_value() -> None:
    """A None override falls back to the definition, for lists and scalars."""
    base = GenerateCall(tools=['oven'], max_turns=3, tool_choice='required', return_tool_requests=True)

    call = base.with_overrides({'tools': None, 'max_turns': None, 'tool_choice': None, 'return_tool_requests': None})

    assert call.tools == ['oven']
    assert call.max_turns == 3
    assert call.tool_choice == 'required'
    assert call.return_tool_requests is True


def test_with_overrides_empty_list_clears() -> None:
    """`tools=[]` is a value, so it replaces the definition's list."""
    call = GenerateCall(tools=['oven']).with_overrides({'tools': []})

    assert call.tools == []


def test_with_overrides_applies_every_prompt_setting() -> None:
    """Every prompt setting replaces its field. A misspelled key in the list would leave its field at the default."""
    answer = Part.from_tool_response(name='oven', output={'temp': 400}, ref='r1')
    opts: PromptGenerateOptions = {
        'tools': ['grill'],
        'tool_choice': 'none',
        'docs': [Document.from_text('allergen chart')],
        'use': [MiddlewareRef(name='allergy_check')],
        'max_turns': 2,
        'return_tool_requests': True,
        'resume_respond': [answer],
        'resume_restart': [answer],
        'resume_metadata': {'approved_by': 'chef'},
    }

    call = GenerateCall().with_overrides(opts)

    assert {k: getattr(call, k) for k in opts} == opts


def test_with_overrides_keeps_template_messages() -> None:
    """`opts['messages']` is chat history; it never replaces the template's messages."""
    base = GenerateCall(messages='{{role "user"}}What is the soup of the day?')
    history = [Message(role=Role.USER, content=[Part.from_text('earlier turn')])]
    opts: PromptGenerateOptions = {'messages': history}

    call = base.with_overrides(opts)

    assert call.messages == base.messages


def test_with_overrides_leaves_model_and_config_alone() -> None:
    """model/config resolve in Prompt._resolve_model, not here."""
    base = GenerateCall(model='echoModel', config={'temperature': 0.5})
    opts: PromptGenerateOptions = {'model': 'scriptedModel', 'config': {'temperature': 0.9}}

    call = base.with_overrides(opts)

    assert call.model == 'echoModel'
    assert call.config == {'temperature': 0.5}


def test_generate_call_rejects_unknown_field() -> None:
    """A misspelled field raises instead of being silently dropped."""
    with pytest.raises(ValidationError, match='tool'):
        GenerateCall(tool=['oven'])  # pyright: ignore[reportCallIssue]  # ty: ignore[unknown-argument]


@pytest.mark.asyncio
@pytest.mark.parametrize('key', sorted(PromptGenerateOptions.__optional_keys__))
async def test_prompt_render_explicit_none_matches_omitted(key: str) -> None:
    """Passing `key=None` renders exactly like leaving `key` out, for every override.

    Guards the public signature (e.g. a sentinel default sneaking in). The
    None-falls-back rule itself is test_with_overrides_none_keeps_definition_value.
    """
    ai, *_ = setup_test()
    p = ai.define_prompt(
        prompt='hi',
        config={'temperature': 0.5},
        tools=['oven'],
        tool_choice='required',
        docs=[Document.from_text('menu')],
        max_turns=3,
        return_tool_requests=True,
    )

    omitted = await p.render()
    explicit_none = await p.render(**{key: None})

    assert explicit_none.model_dump() == omitted.model_dump()


# Mirrors the "Fixed at definition" groups in GenerateCall.
_FIXED_AT_DEFINITION = {
    # PromptTemplate
    'system',
    'prompt',
    'messages',
    'input_schema',
    'metadata',
    # OutputSettings
    'output_schema',
    'output_format',
    'output_content_type',
    'output_instructions',
    'output_constrained',
}


def test_every_generate_call_field_is_fixed_or_changeable() -> None:
    """A new GenerateCall field has to be put in a group; it can't land in neither or both."""
    changeable = PromptSettings.__optional_keys__ | ModelSettings.__optional_keys__

    assert not (_FIXED_AT_DEFINITION & changeable)
    assert _FIXED_AT_DEFINITION | changeable == GenerateCall.model_fields.keys(), (
        'New GenerateCall field: add it to _FIXED_AT_DEFINITION, or to PromptSettings/ModelSettings.'
    )


def test_no_fixed_field_is_a_per_call_option() -> None:
    """Fixed fields can't be passed per call. `messages` is the one shared name: per call it's history."""
    assert _FIXED_AT_DEFINITION & PromptGenerateOptions.__optional_keys__ == {'messages'}


@pytest.mark.parametrize('method', [Prompt.__call__, Prompt.stream, Prompt.render])
def test_prompt_call_keywords_are_exactly_prompt_generate_options(method: Callable[..., object]) -> None:
    """The public keywords are PromptGenerateOptions (plus on_chunk on __call__), nothing more."""
    params = inspect.signature(method).parameters
    keywords = {name for name, p in params.items() if p.kind is inspect.Parameter.KEYWORD_ONLY}

    assert keywords - {'on_chunk'} == PromptGenerateOptions.__optional_keys__


class _StopAtPrepareError(Exception):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['__call__', 'stream', 'render'])
async def test_prompt_call_forwards_every_keyword_to_prepare(method: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every keyword reaches prepare_prompt with the value passed; a keyword left out of the opts dict fails here."""
    ai, _ = _setup_prompt_call()
    p = ai.define_prompt(prompt='Suggest a dish.')
    # A distinct value per keyword, so a dropped or swapped one shows up.
    sent = {key: object() for key in PromptGenerateOptions.__optional_keys__}
    seen: dict[str, object] = {}

    async def capture(*, prompt: object, input: object = None, opts: PromptGenerateOptions | None = None) -> None:
        seen.update(opts or {})
        raise _StopAtPrepareError

    monkeypatch.setattr('genkit._ai._prompt.prepare_prompt', capture)
    with pytest.raises(_StopAtPrepareError):
        if method == 'stream':
            await p.stream(None, **sent).response
        else:
            await getattr(p, method)(None, **sent)

    assert seen == sent


@pytest.mark.asyncio
async def test_prompt_call_with_every_option_keeps_template_and_output() -> None:
    """Setting every per-call option still renders the prompt's own words and output."""
    ai, _ = _setup_prompt_call()
    p = ai.define_prompt(
        system='You are a waiter at {{restaurant}}.',
        prompt='Suggest a dish for someone who likes {{taste}}.',
        output_schema=_Dish,
        output_instructions='Reply as JSON.',
        tools=['oven'],
        max_turns=3,
    )
    menu_input = {'restaurant': 'Luigi', 'taste': 'spicy'}
    history = [Message(role=Role.USER, content=[Part.from_text('earlier turn')])]
    answer = Part.from_tool_response(name='oven', output={'temp': 400}, ref='r1')
    rerun = Part.from_tool_request(name='oven', input={}, ref='r2')

    plain = await p.render(menu_input)
    everything = await p.render(
        menu_input,
        model='echoModel',
        config={'temperature': 0.1},
        messages=history,
        tools=['grill'],
        tool_choice='none',
        docs=[Document.from_text('allergen chart')],
        use=[_PostMiddleware()],
        max_turns=1,
        context={'auth': {'email': 'chef@luigi.it'}},
        return_tool_requests=True,
        resume_respond=[answer],
        resume_restart=[rerun],
        resume_metadata={'approved_by': 'chef'},
    )

    def texts(opts: GenerateActionOptions, role: Role) -> list[str]:
        return [text_from_message(m) for m in opts.messages if m.role == role and m.metadata is None]

    assert everything.output == plain.output
    assert texts(everything, Role.SYSTEM) == texts(plain, Role.SYSTEM) == ['You are a waiter at Luigi.']
    assert texts(everything, Role.USER)[-1] == texts(plain, Role.USER)[-1]


def _roles_and_texts(opts: GenerateActionOptions) -> list[tuple[str, str]]:
    return [(m.role, text_from_message(m)) for m in opts.messages]


_SOUP_HISTORY = [
    Message(role=Role.USER, content=[Part.from_text('Is the soup vegan?')]),
    Message(role=Role.MODEL, content=[Part.from_text('Yes, it is.')]),
]


@pytest.mark.asyncio
async def test_prompt_render_orders_system_template_messages_then_prompt() -> None:
    """A messages list on the definition sits between the system text and the user prompt."""
    ai, _ = _setup_prompt_call()
    p = ai.define_prompt(system='You are a waiter.', messages=_SOUP_HISTORY, prompt='I will have the soup.')

    rendered = await p.render()

    assert _roles_and_texts(rendered) == [
        (Role.SYSTEM, 'You are a waiter.'),
        (Role.USER, 'Is the soup vegan?'),
        (Role.MODEL, 'Yes, it is.'),
        (Role.USER, 'I will have the soup.'),
    ]


@pytest.mark.asyncio
async def test_prompt_render_puts_call_history_after_system() -> None:
    """`messages=` on the call is chat history: after the system text, before the user prompt."""
    ai, _ = _setup_prompt_call()
    p = ai.define_prompt(system='You are a waiter.', prompt='I will have the soup.')

    rendered = await p.render(messages=_SOUP_HISTORY)

    assert _roles_and_texts(rendered) == [
        (Role.SYSTEM, 'You are a waiter.'),
        (Role.USER, 'Is the soup vegan?'),
        (Role.MODEL, 'Yes, it is.'),
        (Role.USER, 'I will have the soup.'),
    ]


@pytest.mark.asyncio
async def test_prompt_render_puts_call_history_at_history_placeholder() -> None:
    """A messages template with `{{history}}` gets the call's history at that spot."""
    ai, _ = _setup_prompt_call()
    p = ai.define_prompt(
        messages='{{role "system"}}You are a waiter.{{history}}{{role "user"}}I will have the soup.',
    )

    rendered = await p.render(messages=_SOUP_HISTORY)

    assert _roles_and_texts(rendered) == [
        (Role.SYSTEM, 'You are a waiter.'),
        (Role.USER, 'Is the soup vegan?'),
        (Role.MODEL, 'Yes, it is.'),
        (Role.USER, 'I will have the soup.'),
    ]


@pytest.mark.asyncio
async def test_prompt_render_context_reaches_every_template() -> None:
    """`{{@auth}}` renders from the call's context in system, messages and prompt templates alike."""
    ai, _ = _setup_prompt_call()
    p = ai.define_prompt(
        system='Serve {{@auth.name}}.',
        messages='{{role "user"}}Table for {{@auth.name}}.',
        prompt='Order for {{@auth.name}}.',
    )

    rendered = await p.render(context={'auth': {'name': 'Ada'}})

    assert [text for _, text in _roles_and_texts(rendered)] == ['Serve Ada.', 'Table for Ada.', 'Order for Ada.']


def _allergy_check_prompt(ai: Genkit, pm: ScriptedModel, seen: list[tuple[str, dict[str, Any]]]) -> Prompt[Any, Any]:
    """A prompt whose model calls `check_allergies` once. Middleware and the tool record the context they ran with."""

    # The engine builds middleware from its class, so the recorder closes over `seen` instead of holding it.
    class RecordContext(BaseMiddleware):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            seen.append(('middleware', dict(ctx.custom_context)))
            return await next_fn(params, ctx)

    @ai.tool(name='check_allergies')
    async def check_allergies(_: dict, ctx: ToolRunContext) -> str:  # noqa: ARG001
        seen.append(('tool', dict(ctx.context)))
        return 'no nuts'

    pm.responses = [_tool_call_reply('check_allergies', 'r1'), _text_reply('done')]
    return ai.define_prompt(prompt='Plan the order.', tools=['check_allergies'], use=[RecordContext()])


def _context_on_each_hop(auth: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    # Model turn that asks for the tool, the tool run, then the model turn that answers.
    return [('middleware', auth), ('tool', auth), ('middleware', auth)]


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['__call__', 'stream'])
async def test_prompt_call_context_reaches_tools(method: str) -> None:
    """`context=` on the call goes to the run too, so middleware and tools see it, not just the template."""
    ai, pm = _setup_prompt_call()
    seen: list[tuple[str, dict[str, Any]]] = []
    p = _allergy_check_prompt(ai, pm, seen)
    auth = {'auth': {'uid': 'chef-1'}}

    if method == 'stream':
        await p.stream(context=auth).response
    else:
        await p(context=auth)

    assert seen == _context_on_each_hop(auth)


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['__call__', 'stream'])
async def test_prompt_call_without_context_uses_enclosing_flow_context(method: str) -> None:
    """Called inside a flow with no `context=`, middleware and tools see the flow's context."""
    ai, pm = _setup_prompt_call()
    seen: list[tuple[str, dict[str, Any]]] = []
    p = _allergy_check_prompt(ai, pm, seen)
    auth = {'auth': {'uid': 'chef-1'}}

    @ai.flow()
    async def plan_order(_: None) -> str:
        if method == 'stream':
            return (await p.stream().response).text
        return (await p()).text

    await plan_order.run(context=auth)

    assert seen == _context_on_each_hop(auth)


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['render', '__call__', 'stream'])
async def test_prompt_inside_flow_fills_auth_from_flow_context(method: str) -> None:
    """Inside a flow, omit `context=` and `{{@auth}}` still fills from the flow."""
    ai, echo, _ = setup_test()
    p = ai.define_prompt(prompt='hello {{@auth.name}}')

    async def greet(_: None) -> str:
        if method == 'render':
            rendered = await p.render()
            return text_from_message(rendered.messages[0])
        if method == 'stream':
            return (await p.stream().response).text
        return (await p()).text

    result = await Action(name='greet', kind=ActionKind.FLOW, fn=greet).run(context={'auth': {'name': 'Ada'}})

    assert 'hello Ada' in result.response
    if method != 'render':
        assert echo.last_request is not None
        assert 'hello Ada' in text_from_message(echo.last_request.messages[-1])


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', [ActionKind.PROMPT, ActionKind.EXECUTABLE_PROMPT])
async def test_registered_prompt_action_fills_auth_from_run_context(kind: ActionKind) -> None:
    """The Dev UI runs a prompt through its action with context=; the template fills from it."""
    ai, _ = _setup_prompt_call()
    ai.define_prompt(name='order', prompt='Order for {{@auth.uid}}.')
    action = await ai.registry.resolve_action(kind, 'order')
    assert action is not None

    rendered = (await action.run(None, context={'auth': {'uid': 'chef-1'}})).response

    assert [text_from_message(m) for m in rendered.messages] == ['Order for chef-1.']


@pytest.mark.asyncio
async def test_prompt_render_explicit_context_overrides_flow_context() -> None:
    """`context=` on render wins over the enclosing flow's auth."""
    ai, _ = _setup_prompt_call()
    p = ai.define_prompt(prompt='hello {{@auth.name}}')

    async def greet(_: None) -> str:
        rendered = await p.render(context={'auth': {'name': 'Bea'}})
        return text_from_message(rendered.messages[0])

    result = await Action(name='greet', kind=ActionKind.FLOW, fn=greet).run(context={'auth': {'name': 'Ada'}})

    assert result.response == 'hello Bea'


@pytest.mark.asyncio
async def test_prompt_render_outside_flow_without_context_leaves_auth_blank() -> None:
    """Outside a flow, omit `context=` and `{{@auth}}` stays empty."""
    ai, _ = _setup_prompt_call()
    p = ai.define_prompt(prompt='hello {{@auth.name}}')

    rendered = await p.render()

    assert text_from_message(rendered.messages[0]) == 'hello '


@pytest.mark.asyncio
async def test_prompt_render_keeps_call_state_when_metadata_has_no_state() -> None:
    """Definition metadata without `state` does not wipe `context={'state': ...}`."""
    ai, _ = _setup_prompt_call()
    p = ai.define_prompt(prompt='s={{@state.x}}', metadata={'owner': 'team'})

    rendered = await p.render(context={'state': {'x': 'call'}})

    assert text_from_message(rendered.messages[0]) == 's=call'


@pytest.mark.asyncio
async def test_prompt_render_uses_metadata_state_when_present() -> None:
    """Definition `metadata={'state': ...}` is what `{{@state}}` reads."""
    ai, _ = _setup_prompt_call()
    p = ai.define_prompt(prompt='s={{@state.x}}', metadata={'state': {'x': 'meta'}})

    rendered = await p.render(context={'state': {'x': 'call'}})

    assert text_from_message(rendered.messages[0]) == 's=meta'


@pytest.mark.asyncio
async def test_prompt_render_messages_template_keeps_call_state_when_metadata_has_no_state() -> None:
    """A messages= template with metadata that has no state still shows the call's context state."""
    ai, _ = _setup_prompt_call()
    p = ai.define_prompt(messages='s={{@state.x}}', metadata={'owner': 'team'})

    rendered = await p.render(context={'state': {'x': 'call'}})

    assert text_from_message(rendered.messages[0]) == 's=call'


@pytest.mark.asyncio
async def test_prompt_render_messages_template_uses_metadata_state_when_present() -> None:
    """A messages= template reads definition `metadata={'state': ...}` for `{{@state}}`."""
    ai, _ = _setup_prompt_call()
    p = ai.define_prompt(messages='s={{@state.x}}', metadata={'state': {'x': 'meta'}})

    rendered = await p.render(context={'state': {'x': 'call'}})

    assert text_from_message(rendered.messages[0]) == 's=meta'


@pytest.mark.asyncio
async def test_prompt_file_render_keeps_call_state() -> None:
    """A .prompt file with no state in frontmatter still shows the call's context state in {{@state}}."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        (Path(tmp_dir) / 'stateful.prompt').write_text('---\nmodel: scriptedModel\n---\ns={{@state.x}}\n')
        ai = Genkit(prompt_dir=tmp_dir, model='scriptedModel')
        define_scripted_model(ai)

        rendered = await ai.prompt('stateful').render(context={'state': {'x': 'call'}})

        assert text_from_message(rendered.messages[0]) == 's=call'


@pytest.mark.asyncio
async def test_prompt_render_system_template_keeps_call_state_when_metadata_has_no_state() -> None:
    """A system= template with metadata that has no state still shows the call's context state."""
    ai, _ = _setup_prompt_call()
    p = ai.define_prompt(system='s={{@state.x}}', metadata={'owner': 'team'})

    rendered = await p.render(context={'state': {'x': 'call'}})

    assert text_from_message(rendered.messages[0]) == 's=call'


@pytest.mark.asyncio
async def test_prompt_stream_yields_each_chunk() -> None:
    """Iterating `p.stream()` yields the model's chunks in order, then the final response resolves."""
    ai, pm = _setup_prompt_call()
    pm.chunks = [[ModelResponseChunk(content=[Part.from_text('a')]), ModelResponseChunk(content=[Part.from_text('b')])]]
    p = ai.define_prompt(prompt='hi')

    stream = p.stream()
    seen = [chunk.text async for chunk in stream.stream]

    assert seen == ['a', 'b']
    assert (await stream.response).text == 'ok'


@pytest.mark.asyncio
async def test_prompt_max_turns_applies_without_call_override() -> None:
    """The prompt's own `max_turns=1` stops the tool loop when the call doesn't pass one."""
    ai, pm = _setup_prompt_call()
    pm.responses = [_tool_call_reply('oven', str(i)) for i in range(6)]
    p = ai.define_prompt(prompt='hi', tools=['oven'], max_turns=1)

    res = await p()

    assert res.finish_reason == FinishReason.ABORTED
    assert pm.request_count == 2


@pytest.mark.asyncio
async def test_prompt_return_tool_requests_applies_without_call_override() -> None:
    """The prompt's own `return_tool_requests=True` stops at the tool request when the call doesn't pass one."""
    ai, pm = _setup_prompt_call()
    pm.responses = [_tool_call_reply('oven', 'r1'), _text_reply('done')]
    p = ai.define_prompt(prompt='hi', tools=['oven'], return_tool_requests=True)

    res = await p()

    assert pm.request_count == 1
    assert [r.tool_request.name for r in res.tool_requests if r.tool_request] == ['oven']


@pytest.mark.asyncio
@pytest.mark.parametrize('where', ['definition', 'call'])
async def test_prompt_inline_tool_object_runs(where: str) -> None:
    """A `Tool` object that was never registered on `ai` runs, whether the prompt or the call passes it."""
    ai, pm = _setup_prompt_call()
    ran: list[str] = []

    async def check_stock() -> str:
        ran.append('check_stock')
        return 'in stock'

    check_stock_tool = tool(check_stock, name='check_stock')
    pm.responses = [_tool_call_reply('check_stock', 'r1'), _text_reply('done')]
    if where == 'definition':
        res = await ai.define_prompt(prompt='hi', tools=[check_stock_tool])()
    else:
        res = await ai.define_prompt(prompt='hi')(tools=[check_stock_tool])

    assert ran == ['check_stock']
    assert res.text == 'done'
    # Registered on the call's child registry only.
    assert await ai.registry.resolve_action(ActionKind.TOOL, 'check_stock') is None


@pytest.mark.asyncio
async def test_prompt_output_settings_reach_the_request() -> None:
    """Every output setting on the definition reaches the request; a schema with no format defaults to json."""
    ai, _ = _setup_prompt_call()
    p = ai.define_prompt(
        prompt='hi',
        output_schema=Recipe,
        output_content_type='application/json',
        output_constrained=True,
    )

    rendered = await p.render()

    assert rendered.output is not None
    assert rendered.output.format == 'json'
    assert rendered.output.content_type == 'application/json'
    assert rendered.output.constrained is True
    assert rendered.output.json_schema is not None
    assert rendered.output.json_schema['properties'].keys() == {'title'}


@pytest.mark.asyncio
async def test_prompt_input_schema_reaches_registered_actions() -> None:
    """`input_schema` on the definition becomes the prompt actions' input schema (what the Dev UI form shows)."""
    ai, _ = _setup_prompt_call()

    class OrderInput(BaseModel):
        dish: str

    ai.define_prompt(name='order', prompt='Order {{dish}}', input_schema=OrderInput)

    for kind in (ActionKind.PROMPT, ActionKind.EXECUTABLE_PROMPT):
        action = await ai.registry.resolve_action(kind, 'order')
        assert action is not None
        schema = cast(dict[str, Any], action.input_schema)
        assert schema['properties'].keys() == {'dish'}


@pytest.mark.asyncio
async def test_prompt_compiles_each_template_once() -> None:
    """Repeat renders reuse the compiled system and prompt templates."""
    ai, _ = _setup_prompt_call()
    p = ai.define_prompt(system='You are a waiter.', prompt='Suggest a {{course}}.')

    with patch.object(ai.registry.dotprompt, 'compile', wraps=ai.registry.dotprompt.compile) as compile_spy:
        await p.render({'course': 'starter'})
        await p.render({'course': 'dessert'})

    assert compile_spy.call_count == 2


@pytest.mark.asyncio
async def test_looked_up_prompt_resolves_once() -> None:
    """`ai.prompt(name)` looks the definition up on first use, then reuses it."""
    ai, _ = _setup_prompt_call()
    ai.define_prompt(name='greeting', prompt='hi')
    greeting = ai.prompt('greeting')

    with patch('genkit._ai._prompt.lookup_prompt', wraps=lookup_prompt) as lookup_spy:
        await greeting.render()
        await greeting.render()

    assert lookup_spy.call_count == 1


@pytest.mark.asyncio
async def test_define_prompt_description_reaches_registered_actions() -> None:
    """`description=` on define_prompt is what the Dev UI shows for both prompt actions."""
    ai, _ = _setup_prompt_call()
    ai.define_prompt(name='suggestDish', description='Suggests a dish for a guest.', prompt='hi')

    for kind in (ActionKind.PROMPT, ActionKind.EXECUTABLE_PROMPT):
        action = await ai.registry.resolve_action(kind, 'suggestDish')
        assert action is not None
        assert action.description == 'Suggests a dish for a guest.'


@pytest.mark.asyncio
async def test_file_prompt_description_reaches_registered_actions() -> None:
    """A `.prompt` file's frontmatter `description` is on both actions before the prompt is first used."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        (Path(tmp_dir) / 'dessert.prompt').write_text(
            '---\ndescription: Suggests a dessert.\n---\nSuggest a dessert.\n'
        )
        ai = Genkit(prompt_dir=tmp_dir, model='echoModel')
        define_echo_model(ai)

        for kind in (ActionKind.PROMPT, ActionKind.EXECUTABLE_PROMPT):
            action = await ai.registry.resolve_action(kind, 'dessert')
            assert action is not None
            assert action.description == 'Suggests a dessert.'


@pytest.mark.asyncio
async def test_lazy_prompt_keeps_caller_output_type() -> None:
    """`ai.prompt(name, output_schema=T)` keeps T after loading the .prompt file's dict schema."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        (Path(tmp_dir) / 'dish.prompt').write_text(
            '---\noutput:\n  schema:\n    name: string\n---\nSuggest a dish.\n',
        )
        ai = Genkit(prompt_dir=tmp_dir, model='echoModel')
        define_echo_model(ai)

        rendered = await ai.prompt('dish', output_schema=_Dish).render()

        assert rendered.output is not None
        assert rendered.output.schema_type is _Dish


# Tests for file-based prompt loading and two-action structure
@pytest.mark.asyncio
async def test_file_based_prompt_registers_two_actions() -> None:
    """File-based prompts create both PROMPT and EXECUTABLE_PROMPT actions."""
    ai, *_ = setup_test()

    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / 'prompts'
        prompt_dir.mkdir()

        # Simple prompt file: name is "filePrompt"
        prompt_file = prompt_dir / 'filePrompt.prompt'
        prompt_file.write_text('hello {{name}}')

        # Load prompts from directory
        load_prompt_folder(ai.registry, prompt_dir)

        # Actions are registered with registry_definition_key (e.g., "filePrompt")
        # We need to look them up by kind and name (without the /prompt/ prefix)
        action_name = 'filePrompt'  # registry_definition_key format

        prompt_action = await ai.registry.resolve_action(ActionKind.PROMPT, action_name)
        executable_prompt_action = await ai.registry.resolve_action(ActionKind.EXECUTABLE_PROMPT, action_name)

        assert prompt_action is not None
        assert executable_prompt_action is not None


@pytest.mark.asyncio
async def test_prompt_and_executable_prompt_return_types() -> None:
    """PROMPT action returns ModelRequest, EXECUTABLE_PROMPT returns GenerateActionOptions."""
    ai, *_ = setup_test()

    # Test with file-based prompt (which creates both actions)
    # Programmatic prompts don't create actions - they're just Prompt instances
    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / 'prompts'
        prompt_dir.mkdir()

        prompt_file = prompt_dir / 'testPrompt.prompt'
        prompt_file.write_text('hello {{name}}')

        load_prompt_folder(ai.registry, prompt_dir)
        action_name = 'testPrompt'

        prompt_action = await ai.registry.resolve_action(ActionKind.PROMPT, action_name)
        executable_prompt_action = await ai.registry.resolve_action(ActionKind.EXECUTABLE_PROMPT, action_name)

        assert prompt_action is not None
        assert executable_prompt_action is not None

        prompt_result = await prompt_action.run(input={'name': 'World'})
        assert isinstance(prompt_result.response, ModelRequest)

        exec_result = await executable_prompt_action.run(input={'name': 'World'})
        assert isinstance(exec_result.response, GenerateActionOptions)


@pytest.mark.asyncio
async def test_lookup_prompt_returns_prompt() -> None:
    """lookup_prompt should return a Prompt that can be called."""
    ai, *_ = setup_test()

    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / 'prompts'
        prompt_dir.mkdir()

        prompt_file = prompt_dir / 'lookupTest.prompt'
        prompt_file.write_text('hi {{name}}')

        load_prompt_folder(ai.registry, prompt_dir)

        executable = await lookup_prompt(ai.registry, 'lookupTest')
        assert isinstance(executable, Prompt)

        response = await executable({'name': 'World'})
        assert 'World' in response.text


@pytest.mark.asyncio
async def test_prompt_function_uses_lookup_prompt() -> None:
    """Test using the prompt function from the Genkit class."""
    ai, *_ = setup_test()

    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / 'prompts'
        prompt_dir.mkdir()

        prompt_file = prompt_dir / 'promptFuncTest.prompt'
        prompt_file.write_text('hello {{name}}')

        load_prompt_folder(ai.registry, prompt_dir)

        # Use ai.prompt() to look up the file-based prompt
        executable = ai.prompt('promptFuncTest')

        # Verify it can be executed
        response = await executable({'name': 'Genkit'})
        assert 'Genkit' in response.text


@pytest.mark.asyncio
async def test_automatic_prompt_loading() -> None:
    """Test that Genkit automatically loads prompts from a directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a prompt file
        prompt_content = """---
name: testPrompt
---
Hello {{name}}!
"""
        prompt_file = Path(tmp_dir) / 'test.prompt'
        prompt_file.write_text(prompt_content)

        # Initialize Genkit with the temporary directory
        ai = Genkit(prompt_dir=tmp_dir)

        # Verify the prompt is registered
        # File-based prompts are registered with an empty namespace by default
        prompt_actions = await ai.registry.resolve_actions_by_kind(ActionKind.PROMPT)
        executable_prompt_actions = await ai.registry.resolve_actions_by_kind(ActionKind.EXECUTABLE_PROMPT)
        assert 'test' in prompt_actions
        assert 'test' in executable_prompt_actions


@pytest.mark.asyncio
async def test_automatic_prompt_loading_default_none() -> None:
    """Test that Genkit does not load prompts if prompt_dir is None."""
    ai = Genkit(prompt_dir=None)

    # Check that no prompts are registered (assuming a clean environment)
    prompt_actions = await ai.registry.resolve_actions_by_kind(ActionKind.PROMPT)
    executable_prompt_actions = await ai.registry.resolve_actions_by_kind(ActionKind.EXECUTABLE_PROMPT)
    assert len(prompt_actions) == 0
    assert len(executable_prompt_actions) == 0


@pytest.mark.asyncio
async def test_automatic_prompt_loading_defaults_mock() -> None:
    """Test that Genkit defaults to ./prompts when prompt_dir is not specified and dir exists."""
    with patch('genkit._ai._aio.load_prompt_folder') as mock_load, patch('genkit._ai._aio.Path') as mock_path:
        # Setup mock to simulate ./prompts existing
        mock_path_instance = MagicMock()
        mock_path_instance.is_dir.return_value = True
        mock_path.return_value = mock_path_instance

        Genkit()
        mock_load.assert_called_once_with(ANY, dir_path=mock_path_instance)


@pytest.mark.asyncio
async def test_automatic_prompt_loading_defaults_missing() -> None:
    """Test that Genkit skips loading when ./prompts is missing."""
    with patch('genkit._ai._aio.load_prompt_folder') as mock_load, patch('genkit._ai._aio.Path') as mock_path:
        # Setup mock to simulate ./prompts missing
        mock_path_instance = MagicMock()
        mock_path_instance.is_dir.return_value = False
        mock_path.return_value = mock_path_instance

        Genkit()
        mock_load.assert_not_called()


@pytest.mark.asyncio
async def test_variant_prompt_loading_does_not_recurse() -> None:
    """Regression: loading a .variant.prompt file must not cause infinite recursion.

    Before the fix, create_prompt_from_file() called resolve_action_by_key()
    on its own action key before setting _cached_prompt.  This triggered
    _trigger_lazy_loading() which re-invoked create_prompt_from_file(),
    recursing until RecursionError.
    See https://github.com/genkit-ai/genkit/issues/4491.
    """
    ai, *_ = setup_test()

    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / 'prompts'
        prompt_dir.mkdir()

        # Base prompt
        base = prompt_dir / 'recipe.prompt'
        base.write_text('---\nmodel: echoModel\n---\nMake a recipe for {{food}}.')

        # Variant prompt (this was the trigger for the visible failure)
        variant = prompt_dir / 'recipe.robot.prompt'
        variant.write_text('---\nmodel: echoModel\n---\nYou are a robot chef. Make a recipe for {{food}}.')

        load_prompt_folder(ai.registry, prompt_dir)

        # Should resolve without RecursionError
        base_exec = await prompt(ai.registry, 'recipe')
        base_response = await base_exec({'food': 'pizza'})
        assert 'pizza' in base_response.text

        robot_exec = await prompt(ai.registry, 'recipe', variant='robot')
        robot_response = await robot_exec({'food': 'pizza'})
        assert 'pizza' in robot_response.text


@pytest.mark.parametrize(
    ('raw', 'want'),
    [
        (None, None),
        ([], []),
        (['a', 'b'], [MiddlewareRef(name='a'), MiddlewareRef(name='b')]),
        (
            ['a', {'name': 'b', 'config': {'k': 1}}],
            [MiddlewareRef(name='a'), MiddlewareRef(name='b', config={'k': 1})],
        ),
        ([{'name': 'x'}], [MiddlewareRef(name='x')]),
    ],
)
def test_parse_dotprompt_use(raw: object, want: list[MiddlewareRef] | None) -> None:
    """Frontmatter ``use`` entries normalize to middleware refs."""
    assert _parse_dotprompt_use(raw) == want


@pytest.mark.parametrize(
    'raw',
    [
        'single',
        [''],
        [{'config': 'x'}],
        [42],
    ],
)
def test_parse_dotprompt_use_invalid(raw: object) -> None:
    """Malformed frontmatter ``use`` raises; reason is INVALID_INPUT."""
    with pytest.raises(GenkitError) as raised:
        _parse_dotprompt_use(raw)
    assert raised.value.status == 'INVALID_ARGUMENT'
    assert raised.value.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'INVALID_INPUT' not in raised.value.original_message


@pytest.mark.asyncio
async def test_load_prompt_with_use_middleware() -> None:
    """Dotprompt frontmatter ``use`` runs middleware on prompt execution."""
    ai = Genkit(model='echoModel', plugins=[PrePostMiddlewarePlugin()])
    define_echo_model(ai)

    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / 'prompts'
        prompt_dir.mkdir()
        (prompt_dir / 'with_mw.prompt').write_text('---\nmodel: echoModel\nuse:\n  - pre_mw\n  - post_mw\n---\nhi\n')
        load_prompt_folder(ai.registry, prompt_dir)

        with_mw = await prompt(ai.registry, 'with_mw')
        response = await with_mw()

    assert response.text == '[ECHO] user: "PRE hi" POST'


@pytest.mark.asyncio
async def test_load_prompt_with_use_middleware_not_registered() -> None:
    """Dotprompt ``use`` referencing unknown middleware fails at resolve time."""
    ai, *_ = setup_test()

    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / 'prompts'
        prompt_dir.mkdir()
        (prompt_dir / 'missing_mw.prompt').write_text('---\nmodel: echoModel\nuse:\n  - missing_mw\n---\nhi\n')
        load_prompt_folder(ai.registry, prompt_dir)

        missing = await prompt(ai.registry, 'missing_mw')
        with pytest.raises(GenkitError, match='missing_mw') as raised:
            await missing()
        assert raised.value.reason is RuntimeErrorReason.INVALID_INPUT
        assert 'INVALID_INPUT' not in raised.value.original_message


@pytest.mark.asyncio
async def test_load_prompt_with_use_not_a_list_raises_invalid_input() -> None:
    """Non-list dotprompt ``use`` fails when the prompt is first resolved."""
    ai, *_ = setup_test()

    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / 'prompts'
        prompt_dir.mkdir()
        (prompt_dir / 'bad_use.prompt').write_text('---\nmodel: echoModel\nuse: not-a-list\n---\nhi\n')
        load_prompt_folder(ai.registry, prompt_dir)

        with pytest.raises(GenkitError, match='must be a list') as raised:
            await prompt(ai.registry, 'bad_use')
        assert raised.value.status == 'INVALID_ARGUMENT'
        assert raised.value.reason is RuntimeErrorReason.INVALID_INPUT
        assert 'INVALID_INPUT' not in raised.value.original_message


@pytest.mark.asyncio
async def test_load_prompt_with_empty_use_entry_raises_invalid_input() -> None:
    """An empty middleware name in dotprompt ``use`` fails when the prompt is resolved."""
    ai, *_ = setup_test()

    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / 'prompts'
        prompt_dir.mkdir()
        (prompt_dir / 'empty_use.prompt').write_text('---\nmodel: echoModel\nuse:\n  - ""\n---\nhi\n')
        load_prompt_folder(ai.registry, prompt_dir)

        with pytest.raises(GenkitError, match='empty string') as raised:
            await prompt(ai.registry, 'empty_use')
        assert raised.value.status == 'INVALID_ARGUMENT'
        assert raised.value.reason is RuntimeErrorReason.INVALID_INPUT
        assert 'INVALID_INPUT' not in raised.value.original_message


@pytest.mark.asyncio
async def test_load_prompt_with_use_missing_name_raises_invalid_input() -> None:
    """A ``use`` map without ``name`` fails when the prompt is resolved."""
    ai, *_ = setup_test()

    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / 'prompts'
        prompt_dir.mkdir()
        (prompt_dir / 'no_name.prompt').write_text('---\nmodel: echoModel\nuse:\n  - config: x\n---\nhi\n')
        load_prompt_folder(ai.registry, prompt_dir)

        with pytest.raises(GenkitError, match='missing required `name`') as raised:
            await prompt(ai.registry, 'no_name')
        assert raised.value.status == 'INVALID_ARGUMENT'
        assert raised.value.reason is RuntimeErrorReason.INVALID_INPUT
        assert 'INVALID_INPUT' not in raised.value.original_message


@pytest.mark.asyncio
async def test_load_prompt_with_numeric_use_entry_raises_invalid_input() -> None:
    """A non-string, non-map ``use`` entry fails when the prompt is resolved."""
    ai, *_ = setup_test()

    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / 'prompts'
        prompt_dir.mkdir()
        (prompt_dir / 'num_use.prompt').write_text('---\nmodel: echoModel\nuse:\n  - 42\n---\nhi\n')
        load_prompt_folder(ai.registry, prompt_dir)

        with pytest.raises(GenkitError, match='must be a string or map') as raised:
            await prompt(ai.registry, 'num_use')
        assert raised.value.status == 'INVALID_ARGUMENT'
        assert raised.value.reason is RuntimeErrorReason.INVALID_INPUT
        assert 'INVALID_INPUT' not in raised.value.original_message


@pytest.mark.asyncio
async def test_load_prompt_with_use_middleware_metadata() -> None:
    """Resolved dotprompt actions expose ``use`` in metadata for the Dev UI."""
    ai, *_ = setup_test()

    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / 'prompts'
        prompt_dir.mkdir()
        (prompt_dir / 'with_meta.prompt').write_text(
            '---\nmodel: echoModel\nuse:\n  - mw1\n  - name: mw2\n    config:\n      foo: bar\n---\nhi\n'
        )
        load_prompt_folder(ai.registry, prompt_dir)

        with_meta = await prompt(ai.registry, 'with_meta')

        assert with_meta._def.use == [  # pyright: ignore[reportPrivateUsage]
            MiddlewareRef(name='mw1'),
            MiddlewareRef(name='mw2', config={'foo': 'bar'}),
        ]
        assert with_meta._def.metadata is not None
        prompt_md = with_meta._def.metadata['prompt']  # pyright: ignore[reportPrivateUsage]
        assert prompt_md['use'] == [
            {'name': 'mw1'},
            {'name': 'mw2', 'config': {'foo': 'bar'}},
        ]
        assert prompt_md['toolDefs'] == []


@pytest.mark.asyncio
async def test_load_prompt_metadata_tool_defs_empty_array() -> None:
    """Dev UI listActions rejects null toolDefs on prompt metadata."""
    ai, *_ = setup_test()

    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / 'prompts'
        prompt_dir.mkdir()
        (prompt_dir / 'no_tools.prompt').write_text('---\nmodel: echoModel\n---\nhi\n')
        load_prompt_folder(ai.registry, prompt_dir)

        no_tools = await prompt(ai.registry, 'no_tools')
        prompt_action = no_tools._prompt_action  # pyright: ignore[reportPrivateUsage]
        assert prompt_action is not None
        action_md = cast(dict[str, Any], prompt_action.metadata)
        for leaked in ('name', 'variant', 'model', 'tools', 'description', 'version', 'toolDefs'):
            assert leaked not in action_md
        assert action_md['type'] == 'prompt'
        prompt_md = cast(dict[str, Any], action_md['prompt'])
        assert prompt_md['toolDefs'] == []
        assert prompt_md['name'] == 'no_tools'


@pytest.mark.asyncio
async def test_define_prompt_primitive_with_output_instructions() -> None:
    """``define_prompt(registry, ...)`` primitive preserves output_instructions and injects on call."""
    ai, _, pm = setup_test()
    pm.responses = [
        ModelResponse(
            finish_reason='stop',
            message=Message(role='model', content=[Part.from_text('{"foo": 1}')]),
        )
    ]

    class TestSchema(BaseModel):
        foo: int | None = Field(None, description='foo field')

    def output_parts(resp: Any) -> list[Any]:
        msg = resp.request.messages[0]
        return [p for p in msg.content if (p.metadata or {}).get('purpose') == 'output']

    p_true = ai.define_prompt(
        name='p_true',
        model='scriptedModel',
        prompt='hi',
        output_format='json',
        output_schema=TestSchema,
        output_instructions=True,
    )
    rendered_true = await p_true.render()
    assert rendered_true.output is not None
    assert rendered_true.output.instructions is True

    resp_true = await p_true()
    injected_true = output_parts(resp_true)
    assert len(injected_true) == 1
    assert 'Output should be in JSON format and conform to the following schema' in (injected_true[0].text or '')

    p_custom = ai.define_prompt(
        name='p_custom',
        model='echoModel',
        prompt='hi',
        output_format='json',
        output_instructions='Only use single quotes in JSON keys if you dare',
    )
    rendered_custom = await p_custom.render()
    assert rendered_custom.output is not None
    assert rendered_custom.output.instructions == 'Only use single quotes in JSON keys if you dare'

    resp_custom = await p_custom()
    injected_custom = output_parts(resp_custom)
    assert len(injected_custom) == 1
    assert (injected_custom[0].text or '') == 'Only use single quotes in JSON keys if you dare'


@pytest.mark.asyncio
async def test_load_prompt_with_output_instructions() -> None:
    """File-based (.prompt) dotprompts preserve output.instructions and inject on call."""
    ai, _, pm = setup_test()
    pm.responses = [
        ModelResponse(
            finish_reason='stop',
            message=Message(role='model', content=[Part.from_text('{"foo": 1}')]),
        )
    ]

    def output_parts(resp: Any) -> list[Any]:
        msg = resp.request.messages[0]
        return [p for p in msg.content if (p.metadata or {}).get('purpose') == 'output']

    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / 'prompts'
        prompt_dir.mkdir()
        (prompt_dir / 'with_instructions.prompt').write_text(
            '---\nmodel: scriptedModel\noutput:\n  format: json\n  schema:\n'
            '    type: object\n    properties:\n      foo:\n        type: integer\n'
            '  instructions: true\n---\nhi\n'
        )
        load_prompt_folder(ai.registry, prompt_dir)

        loaded = await prompt(ai.registry, 'with_instructions')
        assert loaded._def.output_instructions is True  # pyright: ignore[reportPrivateUsage]

        rendered = await loaded.render()
        assert rendered.output is not None
        assert rendered.output.instructions is True

        resp = await loaded(model='scriptedModel')
        injected = output_parts(resp)
        assert len(injected) == 1
        assert 'Output should be in JSON format' in (injected[0].text or '')


def test_resume_options_to_resume_carries_metadata() -> None:
    """The flat ``resume_metadata`` kwarg is threaded onto ``Resume.metadata`` (not dropped)."""
    restart = Part.from_tool_request(name='t', ref='r1', input={})
    resume = resume_options_to_resume(resume_restart=restart, resume_metadata={'approved_by': 'test'})
    assert resume is not None
    assert resume.metadata == {'approved_by': 'test'}


def test_resume_options_to_resume_metadata_only_still_builds() -> None:
    """Even with only metadata (no respond/restart), a Resume is built so a stray
    ``resume_metadata`` forces a resume (and fails loudly downstream) rather than being
    silently dropped."""
    resume = resume_options_to_resume(resume_metadata={'x': 1})
    assert resume is not None
    assert resume.metadata == {'x': 1}


def test_resume_options_to_resume_none_when_all_empty() -> None:
    """No respond, restart, or metadata -> no Resume."""
    assert resume_options_to_resume() is None


@pytest.mark.asyncio
async def test_prompt_file_named_schema_returns_registered_class() -> None:
    """With `ai.define_schema('Recipe', Recipe)` and `schema: Recipe` in the file, `res.output` is a `Recipe`."""
    ai, _pm, tmp = _prompt_file_ai(('recipe.prompt', _RECIPE_PROMPT))
    with tmp:
        ai.define_schema('Recipe', Recipe)
        res = await ai.prompt('recipe')({})

        assert isinstance(res.output, Recipe)
        assert res.output == Recipe(title='pie')
        assert res.error is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('prompt_body', 'json_schema_name'),
    [
        pytest.param(_RECIPE_JSON_PROMPT, 'RecipeJson', id='named_json_schema'),
        pytest.param(_INLINE_OUTPUT_PROMPT, None, id='inline_picoschema'),
    ],
)
async def test_prompt_file_output_without_registered_class_returns_dict(
    prompt_body: str, json_schema_name: str | None
) -> None:
    """A file schema with no class behind it (`define_json_schema` name or inline picoschema) returns a dict."""
    ai, _pm, tmp = _prompt_file_ai(('recipe.prompt', prompt_body))
    with tmp:
        if json_schema_name:
            ai.define_json_schema(json_schema_name, Recipe.model_json_schema())
        res = await ai.prompt('recipe')({})

        assert res.output == {'title': 'pie'}
        assert res.error is None


@pytest.mark.asyncio
async def test_prompt_lookup_output_schema_class_wins_over_file_named_schema() -> None:
    """`ai.prompt('recipe', output_schema=OtherRecipe)` returns `OtherRecipe` even when the file names `Recipe`."""
    ai, _pm, tmp = _prompt_file_ai(('recipe.prompt', _RECIPE_PROMPT))
    with tmp:
        ai.define_schema('Recipe', Recipe)
        res = await ai.prompt('recipe', output_schema=OtherRecipe)({})

        assert type(res.output) is OtherRecipe
        assert res.output == OtherRecipe(title='pie')
        assert res.error is None


@pytest.mark.asyncio
async def test_prompt_file_omitted_input_uses_default() -> None:
    """`render()` with no input renders `Make banana bread.` from `input.default`."""
    ai, _pm, tmp = _prompt_file_ai(('recipe.prompt', _RECIPE_PROMPT))
    with tmp:
        ai.define_schema('Recipe', Recipe)
        rendered = await ai.prompt('recipe').render()

        assert rendered.messages is not None
        assert rendered.messages[0].text == 'Make banana bread.'


@pytest.mark.asyncio
async def test_prompt_file_partial_input_fills_missing_keys_from_default() -> None:
    """Passing `{'style': 'vegan'}` keeps the default `food` and uses the given `style`."""
    ai, _pm, tmp = _prompt_file_ai(('recipe.prompt', _RECIPE_PROMPT))
    with tmp:
        ai.define_schema('Recipe', Recipe)
        rendered = await ai.prompt('recipe').render({'style': 'vegan'})

        assert rendered.messages is not None
        assert rendered.messages[0].text == 'Make banana bread vegan.'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('passed', 'expected'),
    [
        pytest.param({'food': 'pie'}, 'Make pie.', id='value'),
        pytest.param({'food': None}, 'Make .', id='explicit_none'),
    ],
)
async def test_prompt_file_passed_key_overrides_default(passed: dict[str, Any], expected: str) -> None:
    """A key the caller passes wins over `input.default`, including an explicit `None`."""
    ai, _pm, tmp = _prompt_file_ai(('recipe.prompt', _NULLABLE_FOOD_PROMPT))
    with tmp:
        rendered = await ai.prompt('recipe').render(passed)

        assert rendered.messages is not None
        assert rendered.messages[0].text == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('instance', 'expected'),
    [
        pytest.param(ChefInput(food='pie'), 'Make pie.', id='set_field_wins'),
        pytest.param(ChefInput(style='vegan'), 'Make banana bread vegan.', id='unset_field_takes_default'),
    ],
)
async def test_prompt_model_instance_unset_fields_take_file_default(instance: ChefInput, expected: str) -> None:
    """Fields set on a model instance win; fields the caller left unset take the file `input.default`."""
    ai, _pm, tmp = _prompt_file_ai(('recipe.prompt', _RECIPE_PROMPT))
    with tmp:
        ai.define_schema('Recipe', Recipe)
        rendered = await ai.prompt('recipe', input_schema=ChefInput).render(instance)

        assert rendered.messages is not None
        assert rendered.messages[0].text == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('from_file', 'order', 'expected'),
    [
        pytest.param(False, RamenOrder(dish='ramen'), 'regular ramen', id='class_default_without_file_default'),
        pytest.param(True, RamenOrder(dish='ramen'), 'large ramen', id='file_default_beats_class_default'),
        pytest.param(True, RamenOrder(dish='ramen', size='small'), 'small ramen', id='caller_set_beats_both'),
    ],
)
async def test_prompt_model_instance_default_precedence(from_file: bool, order: RamenOrder, expected: str) -> None:
    """For a model instance: fields the caller set > file `input.default` > class default."""
    ai, _pm, tmp = _prompt_file_ai(('order.prompt', _RAMEN_ORDER_PROMPT))
    with tmp:
        if from_file:
            prompt = ai.prompt('order', input_schema=RamenOrder)
        else:
            prompt = ai.define_prompt(name='inline_order', prompt='{{size}} {{dish}}', input_schema=RamenOrder)
        rendered = await prompt.render(order)

        assert rendered.messages is not None
        assert rendered.messages[0].text == expected


@pytest.mark.asyncio
async def test_prompt_file_default_satisfies_required_field() -> None:
    """A field the class requires but the file defaults passes validation once the default is filled."""
    ai, pm, tmp = _prompt_file_ai(('recipe.prompt', _RECIPE_PROMPT))
    with tmp:
        ai.define_schema('Recipe', Recipe)
        res = await ai.prompt('recipe', input_schema=RequiredFood)({})

        assert isinstance(res.output, Recipe)
        assert res.output == Recipe(title='pie')
        assert pm.request_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('call_args', [pytest.param(({},), id='empty_dict'), pytest.param((), id='no_input')])
async def test_prompt_file_input_schema_rejects_missing_required_field(call_args: tuple[Any, ...]) -> None:
    """A required file field with no default raises `INVALID_ARGUMENT` before the model is called."""
    ai, pm, tmp = _prompt_file_ai(('recipe.prompt', _NO_DEFAULT_PROMPT))
    with tmp:
        with pytest.raises(GenkitError) as raised:
            await ai.prompt('recipe')(*call_args)
        _assert_invalid_prompt_input(raised.value, prompt_name='recipe', field='food')
        assert pm.request_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('entry_point', ['call', 'render', 'stream'])
async def test_prompt_file_input_schema_rejects_wrong_type(entry_point: str) -> None:
    """A file with `food: string` rejects `{'food': 1}` from call, `render()`, and `stream()` alike."""
    ai, pm, tmp = _prompt_file_ai(('recipe.prompt', _RECIPE_PROMPT))
    with tmp:
        ai.define_schema('Recipe', Recipe)
        prompt = ai.prompt('recipe')
        with pytest.raises(GenkitError) as raised:
            if entry_point == 'call':
                await prompt({'food': 1})
            elif entry_point == 'render':
                await prompt.render({'food': 1})
            else:
                await prompt.stream({'food': 1}).response
        _assert_invalid_prompt_input(raised.value, prompt_name='recipe', field='food')
        assert pm.request_count == 0


@pytest.mark.asyncio
async def test_prompt_file_input_schema_rejects_unknown_key() -> None:
    """A key the file schema does not declare raises `INVALID_ARGUMENT`."""
    ai, pm, tmp = _prompt_file_ai(('recipe.prompt', _RECIPE_PROMPT))
    with tmp:
        ai.define_schema('Recipe', Recipe)
        with pytest.raises(GenkitError) as raised:
            await ai.prompt('recipe')({'food': 'pie', 'surprise': 1})
        _assert_invalid_prompt_input(raised.value, prompt_name='recipe', field='surprise')
        assert pm.request_count == 0


@pytest.mark.asyncio
async def test_prompt_lookup_input_schema_rejects_missing_required_field() -> None:
    """A required lookup-class field that the file does not default raises `INVALID_ARGUMENT`, naming the field."""
    ai, pm, tmp = _prompt_file_ai(('recipe.prompt', _RECIPE_PROMPT))
    with tmp:
        ai.define_schema('Recipe', Recipe)
        with pytest.raises(GenkitError) as raised:
            await ai.prompt('recipe', input_schema=RequiredDiet)({})
        _assert_invalid_prompt_input(raised.value, prompt_name='recipe', field='diet')
        assert pm.request_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'input_schema',
    [
        pytest.param(ChefInput, id='class'),
        pytest.param(
            {
                'type': 'object',
                'properties': {'food': {'type': 'string'}},
                'required': ['food'],
                'additionalProperties': False,
            },
            id='json_schema',
        ),
    ],
)
async def test_define_prompt_input_schema_rejects_wrong_input(input_schema: type | dict[str, Any]) -> None:
    """`ai.define_prompt(input_schema=...)` rejects `{'food': 1}` for both a class and a JSON schema."""
    ai, pm = _setup_prompt_call()
    p = ai.define_prompt(name='chef', prompt='Make {{food}}.', input_schema=input_schema)

    with pytest.raises(GenkitError) as raised:
        await p({'food': 1})
    _assert_invalid_prompt_input(raised.value, prompt_name='chef', field='food')
    assert pm.request_count == 0


@pytest.mark.asyncio
async def test_prompt_without_input_schema_accepts_any_input() -> None:
    """A `define_prompt` with no `input_schema=` renders whatever dict the caller passed."""
    ai, pm = _setup_prompt_call()
    p = ai.define_prompt(prompt='Make {{food}}.')

    rendered = await p.render({'food': 1, 'surprise': True})

    assert rendered.messages is not None
    assert rendered.messages[0].text == 'Make 1.'
    assert pm.request_count == 0
