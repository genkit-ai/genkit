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

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import ANY, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from genkit.ai import Genkit
from genkit.blocks.prompt import load_prompt_folder, lookup_prompt, prompt
from genkit.core.action.types import ActionKind
from genkit.core.typing import (
    DocumentData,
    DocumentPart,
    GenerateActionOptions,
    GenerateRequest,
    GenerateResponse,
    GenerationCommonConfig,
    Message,
    Part,
    Role,
    TextPart,
    ToolChoice,
)
from genkit.testing import (
    EchoModel,
    ProgrammableModel,
    define_echo_model,
    define_programmable_model,
)


def setup_test() -> tuple[Genkit, EchoModel, ProgrammableModel]:
    """Setup a test fixture for the prompt tests."""
    ai = Genkit(model='echoModel')

    pm, _ = define_programmable_model(ai)
    echo, _ = define_echo_model(ai)

    return (ai, echo, pm)


@pytest.mark.asyncio
async def test_simple_prompt() -> None:
    """Test simple prompt rendering."""
    ai, *_ = setup_test()

    want_txt = '[ECHO] user: "hi" {"temperature":11.0}'

    my_prompt = ai.define_prompt(prompt='hi', config={'temperature': 11})

    response = await my_prompt()

    assert response.text == want_txt

    # New API: stream returns GenerateStreamResponse with .response property
    result = my_prompt.stream()

    assert (await result.response).text == want_txt


@pytest.mark.asyncio
async def test_simple_prompt_with_override_config() -> None:
    """Test the config provided at render time is MERGED (not replaced) with prompt config.

    This matches JS behavior where configs are merged: {...promptConfig, ...optsConfig}
    """
    ai, *_ = setup_test()

    # Config is MERGED: prompt config (banana: true) + opts config (temperature: 12)
    want_txt = '[ECHO] user: "hi" {"banana": true, "temperature": 12}'

    my_prompt = ai.define_prompt(prompt='hi', config={'banana': True})

    # New API: pass config via opts parameter - this MERGES with prompt config
    response = await my_prompt(opts={'config': {'temperature': 12}})

    assert response.text == want_txt

    # New API: stream also uses opts
    result = my_prompt.stream(opts={'config': {'temperature': 12}})

    assert (await result.response).text == want_txt


@pytest.mark.asyncio
async def test_prompt_with_system() -> None:
    """Test that the prompt utilises both prompt and system prompt."""
    ai, *_ = setup_test()

    want_txt = '[ECHO] system: "talk like a pirate" user: "hi"'

    my_prompt = ai.define_prompt(prompt='hi', system='talk like a pirate')

    response = await my_prompt()

    assert response.text == want_txt

    # New API: stream returns GenerateStreamResponse
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
    def test_tool(input: ToolInput) -> str:
        """The tool."""
        return 'abc'

    my_prompt = ai.define_prompt(
        system='pirate',
        prompt='hi',
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='history'))])],
        tools=['testTool'],
        tool_choice=ToolChoice.REQUIRED,
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

    # New API: stream returns GenerateStreamResponse
    result = my_prompt.stream()

    assert (await result.response).text == want_txt


@pytest.mark.asyncio
async def test_prompt_with_resolvers() -> None:
    """Test that the rendering works with resolvers."""
    ai, *_ = setup_test()

    async def system_resolver(input: dict[str, Any], context: object) -> str:
        return f'system {input["name"]}'

    def prompt_resolver(input: dict[str, Any], context: object) -> str:
        return f'prompt {input["name"]}'

    async def messages_resolver(input: dict[str, Any], context: object) -> list[Message]:
        return [Message(role=Role.USER, content=[Part(root=TextPart(text=f'msg {input["name"]}'))])]

    my_prompt = ai.define_prompt(
        system=system_resolver,
        prompt=prompt_resolver,
        messages=messages_resolver,
    )

    want_txt = '[ECHO] system: "system world" user: "msg world" user: "prompt world"'

    response = await my_prompt(input={'name': 'world'})

    assert response.text == want_txt


@pytest.mark.asyncio
async def test_prompt_with_docs_resolver() -> None:
    """Test that the rendering works with docs resolver."""
    ai, _, pm = setup_test()

    pm.responses = [GenerateResponse(message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='ok'))]))]

    async def docs_resolver(input: dict[str, Any], context: object) -> list[DocumentData]:
        return [DocumentData(content=[DocumentPart(root=TextPart(text=f'doc {input["name"]}'))])]

    my_prompt = ai.define_prompt(
        model='programmableModel',
        prompt='hi',
        docs=docs_resolver,
    )

    await my_prompt(input={'name': 'world'})

    # Check that PM received the docs
    assert pm.last_request is not None
    assert pm.last_request.docs is not None
    assert pm.last_request.docs[0].content[0].root.text == 'doc world'


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
        GenerationCommonConfig.model_validate({'temperature': 11}),
        {},
        # Config is MERGED: prompt config (banana: ripe) + opts config (temperature: 11)
        """[ECHO] system: "hello foo (bar)" {"banana": "ripe", "temperature": 11.0}""",
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
        GenerationCommonConfig.model_validate({'temperature': 11}),
        {},
        # Config is MERGED: prompt config (banana: ripe) + opts config (temperature: 11)
        """[ECHO] user: "hello foo (bar_system)" {"banana": "ripe", "temperature": 11.0}""",
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
        GenerationCommonConfig.model_validate({'temperature': 11}),
        {'auth': {'email': 'a@b.c'}},
        # Config is MERGED: prompt config (banana: ripe) + opts config (temperature: 11)
        """[ECHO] user: "hello foo (bar, a@b.c)" {"banana": "ripe", "temperature": 11.0}""",
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
    input_option: GenerationCommonConfig,
    context: dict[str, Any],
    want_rendered: str,
) -> None:
    """Test prompt rendering."""
    ai, *_ = setup_test()

    my_prompt = ai.define_prompt(**prompt)

    # New API: use opts parameter to pass config and context
    response = await my_prompt(input, opts={'config': input_option, 'context': context})

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
async def test_prompt_with_messages_list() -> None:
    """Test prompt with explicit messages list."""
    ai, *_ = setup_test()

    messages = [
        Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='You are helpful'))]),
        Message(role=Role.USER, content=[Part(root=TextPart(text='Hi there'))]),
    ]

    my_prompt = ai.define_prompt(
        messages=messages,
        prompt='How can I help?',
    )

    response = await my_prompt()

    # Should include system, user history, and final prompt
    assert 'helpful' in response.text.lower() or 'Hi there' in response.text


@pytest.mark.asyncio
async def test_messages_with_explicit_override() -> None:
    """Test that explicit messages in render options are included."""
    ai, *_ = setup_test()

    override_messages = [
        Message(role=Role.USER, content=[Part(root=TextPart(text='First message'))]),
        Message(role=Role.MODEL, content=[Part(root=TextPart(text='First response'))]),
    ]

    my_prompt = ai.define_prompt(
        messages=override_messages,
        prompt='Final question',
    )

    # New API: use opts parameter (or no opts for defaults)
    rendered = await my_prompt.render(input=None)

    # Check that we have the final prompt message
    assert any('Final question' in str(msg) for msg in rendered.messages)
    # And that the override messages appear as well
    assert any('First message' in str(msg) for msg in rendered.messages)
    assert any('First response' in str(msg) for msg in rendered.messages)


@pytest.mark.asyncio
async def test_prompt_with_tools_list() -> None:
    """Test prompt with tools parameter."""
    ai, *_ = setup_test()

    class ToolInput(BaseModel):
        value: int = Field(description='A value')

    @ai.tool(name='myTool')
    def my_tool(input: ToolInput) -> int:
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
async def test_system_and_prompt_together() -> None:
    """Test rendering system, messages, and prompt in correct order."""
    ai, *_ = setup_test()

    my_prompt = ai.define_prompt(
        system='System instruction',
        messages=[
            Message(role=Role.USER, content=[Part(root=TextPart(text='History user'))]),
            Message(role=Role.MODEL, content=[Part(root=TextPart(text='History model'))]),
        ],
        prompt='Final prompt',
    )

    response = await my_prompt()

    # All parts should be in the response
    text = response.text.lower()
    assert 'system' in text or 'instruction' in text
    assert 'history' in text or 'final' in text or 'prompt' in text


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

    This matches JS behavior: {...promptConfig, ...optsConfig}
    So opts.config values override prompt config values, but prompt config values
    that aren't in opts.config are preserved.
    """
    ai, *_ = setup_test()

    my_prompt = ai.define_prompt(
        prompt='test',
        config={'temperature': 0.5, 'banana': 'yellow'},
    )

    # New API: runtime config is MERGED with prompt config
    # - temperature: 0.9 (from opts, overrides 0.5)
    # - banana: 'yellow' (from prompt, preserved)
    rendered = await my_prompt.render(opts={'config': {'temperature': 0.9}})

    assert rendered.config is not None
    # Config is now a dict after merging
    assert rendered.config['temperature'] == 0.9
    assert rendered.config['banana'] == 'yellow'  # Preserved from prompt config


# Tests for new PromptGenerateOptions API
@pytest.mark.asyncio
async def test_opts_can_override_model() -> None:
    """Test that opts.model can override the prompt's default model."""
    ai, _, pm = setup_test()

    pm.responses = [
        GenerateResponse(message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='pm response'))]))
    ]

    my_prompt = ai.define_prompt(
        model='echoModel',
        prompt='hello',
    )

    # Override model via opts
    response = await my_prompt(opts={'model': 'programmableModel'})

    # Should use programmableModel, not echoModel
    assert response.text == 'pm response'


@pytest.mark.asyncio
async def test_opts_can_append_messages() -> None:
    """Test that opts.messages appends conversation history."""
    ai, *_ = setup_test()

    my_prompt = ai.define_prompt(
        system='You are helpful',
        prompt='Current question',
    )

    history_messages = [
        Message(role=Role.USER, content=[Part(root=TextPart(text='Previous question'))]),
        Message(role=Role.MODEL, content=[Part(root=TextPart(text='Previous answer'))]),
    ]

    # Append conversation history via opts
    rendered = await my_prompt.render(opts={'messages': history_messages})

    # Should have: system + history (2) + user prompt = 4 messages
    assert len(rendered.messages) == 4
    # Check that history is included
    assert any('Previous question' in str(msg) for msg in rendered.messages)
    assert any('Previous answer' in str(msg) for msg in rendered.messages)


@pytest.mark.asyncio
async def test_generate_stream_response_api() -> None:
    """Test that GenerateStreamResponse provides both stream and response."""
    ai, *_ = setup_test()

    my_prompt = ai.define_prompt(
        prompt='hello world',
    )

    # Get stream response
    result = my_prompt.stream()

    # Verify it has the expected properties (matching JS GenerateStreamResponse)
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
async def test_opts_can_override_output() -> None:
    """Test that opts.output can override output configuration."""
    ai, *_ = setup_test()

    class OutputSchema(BaseModel):
        name: str = Field(description='A name')

    my_prompt = ai.define_prompt(
        prompt='Generate a name',
        output_format='text',  # Default to text
    )

    # Override output via opts
    rendered = await my_prompt.render(
        opts={
            'output': {
                'format': 'json',
                'schema': OutputSchema,
            }
        }
    )

    # Should have json format, not text
    assert rendered.output is not None
    assert rendered.output.format == 'json'
    assert rendered.output.json_schema is not None


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
    """PROMPT action returns GenerateRequest, EXECUTABLE_PROMPT returns GenerateActionOptions."""
    ai, *_ = setup_test()

    # Test with file-based prompt (which creates both actions)
    # Programmatic prompts don't create actions - they're just ExecutablePrompt instances
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

        prompt_result = await prompt_action.arun(input={'name': 'World'})
        assert isinstance(prompt_result.response, GenerateRequest)

        exec_result = await executable_prompt_action.arun(input={'name': 'World'})
        assert isinstance(exec_result.response, GenerateActionOptions)


@pytest.mark.asyncio
async def test_lookup_prompt_returns_executable_prompt() -> None:
    """lookup_prompt should return an ExecutablePrompt that can be called."""
    ai, *_ = setup_test()

    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / 'prompts'
        prompt_dir.mkdir()

        prompt_file = prompt_dir / 'lookupTest.prompt'
        prompt_file.write_text('hi {{name}}')

        load_prompt_folder(ai.registry, prompt_dir)

        executable = await lookup_prompt(ai.registry, 'lookupTest')

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
    with patch('genkit.ai._aio.load_prompt_folder') as mock_load, patch('genkit.ai._aio.Path') as mock_path:
        # Setup mock to simulate ./prompts existing
        mock_path_instance = MagicMock()
        mock_path_instance.is_dir.return_value = True
        mock_path.return_value = mock_path_instance

        Genkit()
        mock_load.assert_called_once_with(ANY, dir_path=mock_path_instance)


@pytest.mark.asyncio
async def test_automatic_prompt_loading_defaults_missing() -> None:
    """Test that Genkit skips loading when ./prompts is missing."""
    with patch('genkit.ai._aio.load_prompt_folder') as mock_load, patch('genkit.ai._aio.Path') as mock_path:
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
    See https://github.com/firebase/genkit/issues/4491.
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
