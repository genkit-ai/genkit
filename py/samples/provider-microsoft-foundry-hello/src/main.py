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

"""Microsoft Foundry hello sample - Microsoft Foundry models with Genkit.

This sample demonstrates how to use Microsoft Foundry models with Genkit.
Microsoft Foundry (formerly Azure AI Foundry) provides access to 11,000+ AI models.

Documentation:
- Microsoft Foundry Portal: https://ai.azure.com/
- Model Catalog: https://ai.azure.com/catalog/models
- SDK Overview: https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/sdk-overview
- Switching Endpoints: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/switching-endpoints

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Microsoft Foundry   │ Microsoft's AI supermarket. One place to access    │
    │                     │ GPT-4o, Claude, Llama, and 11,000+ more models.    │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Azure               │ Microsoft's cloud platform. Where the models       │
    │                     │ actually run and your data stays secure.           │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Endpoint            │ The web address where your AI models live.         │
    │                     │ Like your-resource.openai.azure.com.               │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ API Key             │ Your password to access the models. Keep it        │
    │                     │ secret! Set as AZURE_OPENAI_API_KEY.               │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Deployment          │ A specific instance of a model you've set up.      │
    │                     │ Like having your own copy of GPT-4o.               │
    └─────────────────────┴────────────────────────────────────────────────────┘

Key Features
============
| Feature                          | Example                                    |
|----------------------------------|--------------------------------------------|
| Plugin Initialization            | `MicrosoftFoundry(api_key=..., ...)`       |
| Default Model Configuration      | `ai = Genkit(model=gpt4o)`                 |
| Defining Flows                   | `@ai.flow()` decorator                     |
| Defining Tools                   | `@ai.tool()` decorator                     |
| Simple Generation                | `generate_greeting`                        |
| Streaming Generation             | `generate_streaming_story`                 |
| System Prompt                    | `generate_with_system_prompt`              |
| Multi-turn Conversation          | `generate_multi_turn_chat`                 |
| Generation with Tools            | `generate_weather`                         |
| Structured Output                | `generate_character`                       |
| Streaming Structured Output      | `streaming_structured_output`              |
| Multimodal (Image Input)         | `describe_image`                           |
| Reasoning (Chain-of-Thought)     | `solve_reasoning_problem`                  |
| Generation Configuration         | `generate_with_config`                     |
| Code Generation                  | `generate_code`                            |

Endpoint Types
==============
The plugin supports two endpoint types:

1. **Azure OpenAI endpoint** (traditional):
   Format: `https://<resource-name>.openai.azure.com/`
   Requires `api_version` parameter (e.g., '2024-10-21').

2. **Azure AI Foundry project endpoint** (new unified endpoint):
   Format: `https://<resource-name>.services.ai.azure.com/api/projects/<project-name>`
   Uses v1 API - no api_version needed.

The plugin auto-detects the endpoint type based on the URL format.

Authentication Methods
======================
1. **API Key** (simple):
   Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables.

2. **Azure AD / Managed Identity** (recommended for production):
   ```python
   from azure.identity import DefaultAzureCredential, get_bearer_token_provider

   credential = DefaultAzureCredential()
   token_provider = get_bearer_token_provider(credential, 'https://cognitiveservices.azure.com/.default')

   ai = Genkit(
       plugins=[
           MicrosoftFoundry(
               azure_ad_token_provider=token_provider,
               endpoint='https://your-resource.openai.azure.com/',
           )
       ]
   )
   ```

Finding Your Credentials
========================
1. Go to Microsoft Foundry Portal (https://ai.azure.com/)
2. Select your Project
3. Navigate to Models → Deployments
4. Click on your Deployment (e.g., gpt-4o)
5. Open the Details pane

You'll find:
- Target URI: Contains the endpoint URL and API version
- Key: Your API key
- Name: Your deployment name

Testing
=======
1. Set environment variables (extract from Target URI in the Details pane):

   # Example Target URI:
   # https://your-resource.cognitiveservices.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21

   export AZURE_OPENAI_ENDPOINT="https://your-resource.cognitiveservices.azure.com/"  # Base URL only
   export AZURE_OPENAI_API_KEY="your-api-key"  # From Key field
   export AZURE_OPENAI_API_VERSION="2024-10-21"  # Optional: from api-version in Target URI
   export AZURE_OPENAI_DEPLOYMENT="gpt-4o"  # From Name field

2. Run the sample:
   ./run.sh

3. Open the Genkit Dev UI and test the flows.

See Also:
    - Microsoft Foundry Documentation: https://learn.microsoft.com/en-us/azure/ai-foundry/
    - Model Catalog: https://ai.azure.com/catalog/models

Note:
    This is a community sample and is not officially endorsed by Microsoft.
    "Microsoft", "Azure", and "Microsoft Foundry" are trademarks of Microsoft Corporation.
"""

import asyncio
import os

from genkit.ai import Genkit, Output
from genkit.core.action import ActionRunContext
from genkit.core.logging import get_logger
from genkit.plugins.microsoft_foundry import MicrosoftFoundry, gpt4o, microsoft_foundry_model
from samples.shared import (
    CharacterInput,
    CodeInput,
    GreetingInput,
    ImageDescribeInput,
    MultiTurnInput,
    ReasoningInput,
    RpgCharacter,
    StreamingToolInput,
    StreamInput,
    SystemPromptInput,
    WeatherInput,
    describe_image_logic,
    generate_character_logic,
    generate_code_logic,
    generate_greeting_logic,
    generate_multi_turn_chat_logic,
    generate_streaming_story_logic,
    generate_streaming_with_tools_logic,
    generate_weather_logic,
    generate_with_config_logic,
    generate_with_system_prompt_logic,
    get_weather,
    setup_sample,
    solve_reasoning_problem_logic,
)

setup_sample()

# Configuration from environment variables
# Find these values in Microsoft Foundry Portal:
# ai.azure.com > [Project] > Models > Deployments > [Deployment] > Details
API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')
API_VERSION = os.environ.get('AZURE_OPENAI_API_VERSION')  # None = use plugin default
DEPLOYMENT = os.environ.get('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o')

logger = get_logger(__name__)

# Log configuration for debugging (mask API key for security)

if not API_KEY or not ENDPOINT:
    logger.warning(
        'AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set. Set these environment variables to use the sample.'
    )

ai = Genkit(
    plugins=[
        MicrosoftFoundry(
            api_key=API_KEY,
            endpoint=ENDPOINT,
            api_version=API_VERSION,  # None lets plugin use DEFAULT_API_VERSION
            deployment=DEPLOYMENT,
        )
    ],
    model=gpt4o,
)

ai.tool()(get_weather)


@ai.flow()
async def generate_greeting(input: GreetingInput) -> str:
    """Generate a simple greeting.

    Args:
        input: Input with name to greet.

    Returns:
        Greeting message.
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

    Args:
        input: Input with a travel destination.

    Returns:
        The model's final response, demonstrating context retention.
    """
    return await generate_multi_turn_chat_logic(ai, input.destination)


@ai.flow()
async def generate_weather(input: WeatherInput) -> str:
    """Get weather information using tool calling.

    Args:
        input: Input with location to get weather for.

    Returns:
        Weather information.
    """
    return await generate_weather_logic(ai, input)


@ai.flow()
async def generate_streaming_story(
    input: StreamInput,
    ctx: ActionRunContext = None,  # type: ignore[assignment]
) -> str:
    """Generate a streaming story response.

    Args:
        input: Input with name for streaming story.
        ctx: Action run context for streaming.

    Returns:
        Complete generated text.
    """
    return await generate_streaming_story_logic(ai, input.name, ctx)


@ai.flow()
async def describe_image(input: ImageDescribeInput) -> str:
    """Describe an image using Microsoft Foundry.

    Args:
        input: Input with image URL to describe.

    Returns:
        A textual description of the image.
    """
    return await describe_image_logic(ai, input.image_url)


@ai.flow()
async def generate_with_config(input: GreetingInput) -> str:
    """Generate a greeting with custom model configuration.

    Args:
        input: Input with name to greet.

    Returns:
        Greeting message.
    """
    return await generate_with_config_logic(ai, input.name)


@ai.flow()
async def generate_code(input: CodeInput) -> str:
    """Generate code using Microsoft Foundry models.

    Args:
        input: Input with coding task description.

    Returns:
        Generated code.
    """
    return await generate_code_logic(ai, input.task)


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
async def solve_reasoning_problem(input: ReasoningInput) -> str:
    """Solve reasoning problems using a reasoning model.

    Args:
        input: Input with reasoning question to solve.

    Returns:
        The reasoning and answer.
    """
    return await solve_reasoning_problem_logic(ai, input.prompt, model=microsoft_foundry_model('o4-mini'))


@ai.flow()
async def generate_streaming_with_tools(
    input: StreamingToolInput,
    ctx: ActionRunContext | None = None,
) -> str:
    """Demonstrate streaming generation with tool calling.

    Args:
        input: Input with location for weather lookup.
        ctx: Action context for streaming chunks to the client.

    Returns:
        The complete generated text.
    """
    return await generate_streaming_with_tools_logic(ai, input.location, ctx)


async def main() -> None:
    """Main entry point for the sample."""
    await logger.ainfo('Genkit server running. Press Ctrl+C to stop.')
    await logger.ainfo('Open the Genkit Dev UI to test the flows.')
    # Keep the process alive for Dev UI
    await asyncio.Event().wait()


if __name__ == '__main__':
    ai.run_main(main())
