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

"""Hello Vertex AI sample - Google Cloud's Vertex AI with Genkit.

This sample demonstrates how to use Vertex AI (Google Cloud's ML platform)
with Genkit for enterprise-grade AI applications.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Vertex AI           │ Google Cloud's AI platform. Like GoogleAI but      │
    │                     │ for enterprise with more security features.        │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ GCP Project         │ Your Google Cloud project. Like a folder that      │
    │                     │ holds all your cloud resources.                    │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Location            │ Which data center to use (us-central1, etc.).      │
    │                     │ Pick one near your users.                          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ ADC                 │ Application Default Credentials. Google's way      │
    │                     │ of auto-finding your login credentials.            │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ gcloud auth         │ The command to log in. Run once and Google         │
    │                     │ remembers who you are.                             │
    └─────────────────────┴────────────────────────────────────────────────────┘

Key Features
============
| Feature Description                                      | Example Function / Code Snippet        |
|----------------------------------------------------------|----------------------------------------|
| Plugin Initialization                                    | `ai = Genkit(plugins=[VertexAI(...)])` |
| Default Model Configuration                              | `ai = Genkit(model=...)`               |
| Defining Flows                                           | `@ai.flow()` decorator (multiple uses) |
| Defining Tools                                           | `@ai.tool()` decorator (multiple uses) |
| Pydantic for Tool Input Schema                           | `GablorkenInput`                       |
| Simple Generation (Prompt String)                        | `generate_greeting`                    |
| System Prompt                                            | `generate_with_system_prompt`          |
| Multi-turn Conversation                                  | `generate_multi_turn_chat`             |
| Generation with Messages (`Message`, `Role`, `TextPart`) | `simple_generate_with_tools_flow`      |
| Generation with Tools                                    | `simple_generate_with_tools_flow`      |
| Tool Response Handling                                   | `simple_generate_with_interrupts`      |
| Tool Interruption (`ctx.interrupt`)                      | `gablorken_tool2`                      |
| Embedding (`ai.embed`, `Document`)                       | `embed_docs`                           |
| Generation Configuration (`temperature`, etc.)           | `generate_with_config`                 |
| Streaming Generation (`ai.generate_stream`)              | `generate_streaming_story`             |
| Streaming Chunk Handling (`ctx.send_chunk`)              | `generate_streaming_story`, `generate_character`  |
| Structured Output (Schema)                               | `generate_character`                   |
| Streaming Structured Output                              | `streaming_structured_output`          |
| Pydantic for Structured Output Schema                    | `RpgCharacter`                         |
| Structured Output (Instruction-Based)                    | `generate_character_instructions`      |
| Code Generation                                          | `generate_code`                            |

Testing This Demo
=================
1. **Prerequisites**:
   ```bash
   # Set GCP project and location
   export GOOGLE_CLOUD_PROJECT=your_project_id
   export GOOGLE_CLOUD_LOCATION=us-central1

   # Authenticate with GCP
   gcloud auth application-default login
   ```

2. **Run the demo**:
   ```bash
   cd py/samples/provider-google-genai-vertexai-hello
   ./run.sh
   ```

3. **Open DevUI** at http://localhost:4000

4. **Test basic flows**:
   - [ ] `generate_greeting` - Simple generation
   - [ ] `generate_streaming_story` - Streaming response
   - [ ] `generate_with_config` - Custom config

5. **Test tools**:
   - [ ] `simple_generate_with_tools_flow` - Tool calling
   - [ ] `simple_generate_with_interrupts` - Tool interrupts

6. **Test structured output**:
   - [ ] `generate_character` - Constrained output
   - [ ] `generate_character_instructions` - Instruction-based

7. **Test embeddings**:
   - [ ] `embed_docs` - Document embedding

8. **Note**: Vertex AI requires a GCP project with billing enabled.
"""

import os

from pydantic import BaseModel, Field

from genkit.ai import Genkit, Output, ToolRunContext, tool_response
from genkit.core.action import ActionRunContext
from genkit.core.logging import get_logger
from genkit.plugins.google_genai import (
    EmbeddingTaskType,
    VertexAI,
)
from genkit.types import Embedding
from samples.shared import (
    CharacterInput,
    CodeInput,
    CurrencyExchangeInput,
    GreetingInput,
    MultiTurnInput,
    RpgCharacter,
    StreamingToolInput,
    StreamInput,
    SystemPromptInput,
    convert_currency as _convert_currency_tool,
    convert_currency_logic,
    generate_character_logic,
    generate_code_logic,
    generate_greeting_logic,
    generate_multi_turn_chat_logic,
    generate_streaming_story_logic,
    generate_streaming_with_tools_logic,
    generate_with_config_logic,
    generate_with_system_prompt_logic,
    get_weather,
    setup_sample,
)

setup_sample()

logger = get_logger(__name__)

# Check for GCLOUD_PROJECT or GOOGLE_CLOUD_PROJECT
# If GOOGLE_CLOUD_PROJECT is set but GCLOUD_PROJECT isn't, use it
if 'GCLOUD_PROJECT' not in os.environ:
    if 'GOOGLE_CLOUD_PROJECT' in os.environ:
        os.environ['GCLOUD_PROJECT'] = os.environ['GOOGLE_CLOUD_PROJECT']
    else:
        os.environ['GCLOUD_PROJECT'] = input('Please enter your GCLOUD_PROJECT_ID: ')


ai = Genkit(
    plugins=[VertexAI()],
    model='vertexai/gemini-2.5-flash',
)


ai.tool()(get_weather)
ai.tool()(_convert_currency_tool)


class GablorkenInput(BaseModel):
    """The Pydantic model for tools."""

    value: int = Field(description='value to calculate gablorken for')


class TemperatureInput(BaseModel):
    """Input for temperature config flow."""

    data: str = Field(default='Mittens', description='Name to greet')


class ToolsFlowInput(BaseModel):
    """Input for tools flow."""

    value: int = Field(default=42, description='Value for gablorken calculation')


@ai.tool(name='gablorkenTool')
def gablorken_tool(input_: GablorkenInput) -> int:
    """Calculate a gablorken.

    Args:
        input_: The input to calculate gablorken for.

    Returns:
        The calculated gablorken.
    """
    return input_.value * 3 - 5


@ai.tool(name='gablorkenTool2')
def gablorken_tool2(input_: GablorkenInput, ctx: ToolRunContext) -> None:
    """The user-defined tool function.

    Args:
        input_: the input to the tool
        ctx: the tool run context

    Returns:
        The calculated gablorken.
    """
    ctx.interrupt()


@ai.flow()
async def convert_currency(input: CurrencyExchangeInput) -> str:
    """Convert currency using tool calling.

    Args:
        input: Currency exchange parameters.

    Returns:
        Conversion result.
    """
    return await convert_currency_logic(ai, input)


@ai.flow()
async def embed_docs(docs: list[str] | None = None) -> list[Embedding]:
    """Generate an embedding for the words in a list.

    Args:
        docs: list of texts (string)

    Returns:
        The generated embedding.
    """
    if docs is None:
        docs = ['Hello world', 'Genkit is great', 'Embeddings are fun']
    options = {'task_type': EmbeddingTaskType.CLUSTERING}
    return await ai.embed_many(
        embedder='vertexai/gemini-embedding-001',
        content=docs,
        options=options,
    )


@ai.flow()
async def generate_character(input: CharacterInput) -> RpgCharacter:
    """Generate an RPG character with structured output.

    Args:
        input: Input with character name.

    Returns:
        The generated RPG character.
    """
    return await generate_character_logic(ai, input.name)


@ai.flow()
async def generate_character_instructions(
    input: CharacterInput,
    ctx: ActionRunContext | None = None,
) -> RpgCharacter:
    """Generate an RPG character using instruction-based structured output.

    Unlike ``generate_character`` which uses constrained decoding (the model
    is forced to output valid JSON matching the schema), this flow uses
    ``output_constrained=False`` to guide the model via prompt instructions
    instead. This is useful when::

        - The model doesn't support constrained decoding.
        - You want the model to have more flexibility in its output.
        - You're debugging schema adherence issues.

    See: https://genkit.dev/docs/models#structured-output

    Args:
        input: Input with character name.
        ctx: the context of the tool

    Returns:
        The generated RPG character.
    """
    result = await ai.generate(
        prompt=f'generate an RPG character named {input.name}',
        output=Output(schema=RpgCharacter),
        output_constrained=False,
        output_instructions=True,
    )
    return result.output


@ai.flow()
async def generate_greeting(input: GreetingInput) -> str:
    """Generate a simple greeting.

    Args:
        input: Input with name to greet.

    Returns:
        The generated response with a function.
    """
    return await generate_greeting_logic(ai, input.name)


@ai.flow()
async def generate_with_system_prompt(input: SystemPromptInput) -> str:
    """Demonstrate system prompts to control model persona and behavior.

    Args:
        input: Input with a question to ask.

    Returns:
        The model's response in the persona defined by the system prompt.
    """
    return await generate_with_system_prompt_logic(ai, input.question)


@ai.flow()
async def generate_multi_turn_chat(input: MultiTurnInput) -> str:
    """Demonstrate multi-turn conversations using the messages parameter.

    The messages parameter allows you to pass a conversation history to
    maintain context across multiple interactions with the model. Each
    message has a role ('user' or 'model') and content.

    See: https://genkit.dev/docs/models#multi-turn-conversations-with-messages

    Args:
        input: Input with a travel destination.

    Returns:
        The model's final response, demonstrating context retention.
    """
    return await generate_multi_turn_chat_logic(ai, input.destination)


@ai.flow()
async def generate_streaming_story(
    input: StreamInput,
    ctx: ActionRunContext | None = None,
) -> str:
    """Generate a streaming story response.

    Args:
        input: Input with name for streaming.
        ctx: the context of the tool

    Returns:
        The generated response with a function.
    """
    return await generate_streaming_story_logic(ai, input.name, ctx)


@ai.flow()
async def generate_with_config(input: TemperatureInput) -> str:
    """Generate a greeting with custom model configuration.

    Args:
        input: Input with name to greet.

    Returns:
        The generated response with a function.
    """
    return await generate_with_config_logic(ai, input.data)


@ai.flow()
async def simple_generate_with_interrupts(input: ToolsFlowInput) -> str:
    """Generate a greeting for the given name.

    Args:
        input: Input with value for gablorken calculation.

    Returns:
        The generated response with a function.
    """
    response1 = await ai.generate(
        prompt=f'what is a gablorken of {input.value}',
        tools=['gablorkenTool2'],
    )
    await logger.ainfo(f'len(response.tool_requests)={len(response1.tool_requests)}')
    if len(response1.tool_requests) == 0:
        return response1.text

    tr = tool_response(response1.tool_requests[0], 178)
    response = await ai.generate(
        messages=response1.messages,
        tool_responses=[tr],
        tools=['gablorkenTool'],
    )
    return response.text


@ai.flow()
async def simple_generate_with_tools_flow(input: ToolsFlowInput) -> str:
    """Generate a greeting for the given name.

    Args:
        input: Input with value for gablorken calculation.

    Returns:
        The generated response with a function.
    """
    response = await ai.generate(
        prompt=f'what is a gablorken of {input.value}',
        tools=['gablorkenTool'],
    )
    return response.text


@ai.flow()
async def streaming_structured_output(
    input: CharacterInput,
    ctx: ActionRunContext | None = None,
) -> RpgCharacter:
    """Demonstrate streaming with structured output schemas.

    Combines `generate_stream` with `Output(schema=...)` so the model
    streams JSON tokens that are progressively parsed into the Pydantic
    model. Each chunk exposes a partial `.output` you can forward to
    clients for incremental rendering.

    See: https://genkit.dev/docs/models#streaming

    Args:
        input: Input with character name.
        ctx: Action context for streaming partial outputs.

    Returns:
        The fully-parsed RPG character once streaming completes.
    """
    stream, result = ai.generate_stream(
        prompt=(
            f'Generate an RPG character named {input.name}. '
            'Include a creative backstory, 3-4 unique abilities, '
            'and skill ratings for strength, charisma, and endurance (0-100 each).'
        ),
        output=Output(schema=RpgCharacter),
    )
    async for chunk in stream:
        if ctx is not None:
            ctx.send_chunk(chunk.output)

    return (await result).output


@ai.flow()
async def generate_code(input: CodeInput) -> str:
    """Generate code using Vertex AI Gemini.

    Args:
        input: Input with coding task description.

    Returns:
        Generated code.
    """
    return await generate_code_logic(ai, input.task)


@ai.flow()
async def generate_streaming_with_tools(
    input: StreamingToolInput,
    ctx: ActionRunContext | None = None,
) -> str:
    """Demonstrate streaming generation with tool calling.

    The model streams its response while also calling tools mid-generation.
    Tool calls are resolved automatically and the model continues generating.

    Args:
        input: Input with location for weather lookup.
        ctx: Action context for streaming chunks to the client.

    Returns:
        The complete generated text.
    """
    return await generate_streaming_with_tools_logic(ai, input.location, ctx)


async def main() -> None:
    """Main function - runs when script starts."""
    await logger.ainfo('VertexAI Hello sample initialized. Use Dev UI to invoke flows.')


if __name__ == '__main__':
    ai.run_main(main())
