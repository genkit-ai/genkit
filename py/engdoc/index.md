# Genkit

!!! note

    If you're a user of Genkit and landed here,
    this is engineering documentation that someone contributing
    to Genkit would use, not necessarily only use it.

    For more information about how to get started with using
    Genkit, please see: [User Guide](.)

## What is Genkit?

Genkit is a framework designed to help you build AI-powered applications and
features. It provides open source libraries and plus developer
tools for testing and debugging. The following language runtimes are supported:

| Language Runtime | Version | Support Tier |
|------------------|---------|--------------|
| Node.js          | 22.0+   | 1            |
| Go               | 1.22+   | 1            |
| Python           | 3.10+   | 1            |

It is designed to work with any generative AI model API or vector database.
While we offer integrations for Firebase and Google Cloud, you can use Genkit
independently of any Google services.

The framework provides an abstraction of components by wrapping them with
building blocks called actions, each of which is maintained in a registry. An
action can expose a component over HTTP as a cloud function or server endpoint
and is inspectable and discoverable via a reflection API. Flows are actions
defined by the user and plugins can be created by third parties to extend the
set of available actions.

## Key capabilities

| Feature                           | Description                                                                                                                                                                                  |
|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Unified API for AI generation** | Use one API to generate or stream content from various AI models. Works with multimodal input/output and custom model settings.                                                              |
| **Structured output**             | Generate or stream structured objects (like JSON) with built-in validation. Simplify integration with your app and convert unstructured data into a usable format.                           |
| **Tool calling**                  | Let AI models call your functions and APIs as tools to complete tasks. The model decides when and which tools to use.                                                                        |
| **Chat**                          | Genkit offers a chat-specific API that facilitates multi-turn conversations with AI models, which can be stateful and persistent.                                                            |
| **Agents**                        | Create intelligent agents that use tools (including other agents) to help automate complex tasks and workflows.                                                                              |
| **Data retrieval**                | Improve the accuracy and relevance of generated output by integrating your data. Simple APIs help you embed, index, and retrieve information from various sources.                           |
| **Prompt templating**             | Create effective prompts that include rich text templating, model settings, multimodal support, and tool integration - all within a compact, runnable [prompt file](/docs/genkit/dotprompt). |

See the following code samples for a concrete idea of how to use these
capabilities in code:

### Feature Parity

| Feature           | Python | JavaScript | Go |
|-------------------|--------|------------|----|
| Agents            | ❌     | ✅         | ✅ |
| Chat              | ✅     | ✅         | ✅ |
| Data retrieval    | ✅     | ✅         | ✅ |
| Generation        | ✅     | ✅         | ✅ |
| Prompt templating | ✅     | ✅         | ✅ |
| Structured output | ✅     | ✅         | ✅ |
| Tool calling      | ✅     | ✅         | ✅ |

### Plugin Parity

| Plugins                | Python | JavaScript | Go |
|------------------------|--------|------------|----|
| Amazon Bedrock         | ✅     | —          | —  |
| Anthropic              | ✅     | —          | —  |
| Checks                 | ✅     | ✅         | —  |
| Cloudflare Workers AI  | ✅     | —          | —  |
| Cohere                 | ✅     | —          | —  |
| Compat-OAI             | ✅     | —          | —  |
| DeepSeek               | ✅     | —          | —  |
| Dev Local Vectorstore  | ✅     | ✅         | —  |
| Dotprompt              | ✅     | ✅         | ✅ |
| Evaluators             | ✅     | ✅         | —  |
| FastAPI                | ✅     | —          | —  |
| Firebase               | ✅     | ✅         | ✅ |
| Flask                  | ✅     | —          | —  |
| Google Cloud           | ✅     | ✅         | ✅ |
| Google GenAI           | ✅     | ✅         | ✅ |
| Hugging Face           | ✅     | —          | —  |
| MCP                    | ✅     | —          | —  |
| Microsoft Foundry      | ✅     | —          | —  |
| Mistral                | ✅     | —          | —  |
| Observability          | ✅     | —          | —  |
| Ollama                 | ✅     | ✅         | —  |
| Vertex AI              | ✅     | ✅         | ✅ |
| xAI                    | ✅     | —          | —  |

## Examples

### Basic generation

=== "Python"

    ```python linenums="1"
    import asyncio

    from genkit.ai import Genkit
    from genkit.plugins.google_genai import GoogleAI

    ai = Genkit(
        plugins=[GoogleAI()],
        model='googleai/gemini-2.0-flash',
    )


    async def main() -> None:
        response = await ai.generate(prompt='Why is AI awesome?')
        print(response.text)

        stream, _ = ai.generate_stream(prompt='Tell me a story')
        async for chunk in stream:
            print(chunk.text, end='')


    if __name__ == '__main__':
        asyncio.run(main())
    ```

=== "JavaScript"

    ```javascript
    import { genkit } from 'genkit';
    import { googleAI, gemini15Flash } from '@genkit-ai/googleai';

    const ai = genkit({
      plugins: [googleAI()],
      model: gemini15Flash,  // Set default model
    });

    // Simple generation
    const { text } = await ai.generate('Why is AI awesome?');
    console.log(text);

    // Streamed generation
    const { stream } = await ai.generateStream('Tell me a story');
    for await (const chunk of stream) {
      console.log(chunk.text);
    }
    ```

=== "Go"

    ```go
    import "fmt"

    func main() {
      fmt.Println("Hello")
    }
    ```

### Structured output

=== "Python"

    ```python
    import asyncio
    from enum import Enum

    from pydantic import BaseModel

    from genkit.ai import Genkit, Output
    from genkit.plugins.google_genai import GoogleAI

    ai = Genkit(
        plugins=[GoogleAI()],
        model='googleai/gemini-2.0-flash',
    )


    class Role(str, Enum):
        KNIGHT = "knight"
        MAGE = "mage"
        ARCHER = "archer"


    class CharacterProfile(BaseModel):
        name: str
        role: Role
        backstory: str


    async def main() -> None:
        response = await ai.generate(
            prompt="Create a brief profile for a character in a fantasy video game.",
            output=Output(schema=CharacterProfile),
        )
        print(response.output)


    if __name__ == "__main__":
        asyncio.run(main())
    ```

=== "JavaScript"

    ```javascript
    import { genkit, z } from 'genkit';
    import { googleAI, gemini15Flash } from '@genkit-ai/googleai';

    const ai = genkit({
      plugins: [googleAI()],
      model: gemini15Flash,
    });

    const { output } = await ai.generate({
      prompt: 'Create a brief profile for a character in a fantasy video game.',
      // Specify output structure using Zod schema
      output: {
        format: 'json',
        schema: z.object({
          name: z.string(),
          role: z.enum(['knight', 'mage', 'archer']),
          backstory: z.string(),
        }),
      },

    });

    console.log(output);
    ```

### Function calling

=== "Python"

    ```python
    import asyncio

    from pydantic import BaseModel, Field

    from genkit.ai import Genkit
    from genkit.plugins.google_genai import GoogleAI

    ai = Genkit(
        plugins=[GoogleAI()],
        model='googleai/gemini-2.0-flash',
    )


    @ai.tool()
    async def get_weather(location: str = Field(description="The location to get the current weather for")) -> str:
        """Gets the current weather in a given location."""
        return f"The current weather in {location} is 63°F and sunny."


    async def main() -> None:
        response = await ai.generate(
            prompt="What is the weather like in New York?",
            tools=['get_weather'],
        )
        print(response.text)


    if __name__ == "__main__":
        asyncio.run(main())
    ```

=== "JavaScript"

    ```javascript
    import { genkit, z } from 'genkit';
    import { googleAI, gemini15Flash } from '@genkit-ai/googleai';

    const ai = genkit({
      plugins: [googleAI()],
      model: gemini15Flash,
    });

    // Define tool to get current weather for a given location
    const getWeather = ai.defineTool(
      {
        name: "getWeather",
        description: "Gets the current weather in a given location",
        inputSchema: z.object({
          location: z.string().describe('The location to get the current weather for')
        }),
        outputSchema: z.string(),
      },
      async (input) => {
        // Here, we would typically make an API call or database query. For this
        // example, we just return a fixed value.
        return `The current weather in ${input.location} is 63°F and sunny.`;
      }
    );

    const { text } = await ai.generate({
        tools: [getWeather], // Give the model a list of tools it can call
        prompt: 'What is the weather like in New York? ',
    });

    console.log(text);
    ```

### Chat

=== "Python"

    ```python
    import asyncio

    from genkit.ai import Genkit
    from genkit.plugins.google_genai import GoogleAI

    ai = Genkit(
        plugins=[GoogleAI()],
        model='googleai/gemini-2.0-flash',
    )


    async def main() -> None:
        response = await ai.generate(
            prompt='Hi, my name is Pavel',
            system='Talk like a pirate',
        )
        print(response.text)

        response = await ai.generate(
            prompt='What is my name?',
            system='Talk like a pirate',
            messages=response.messages,
        )
        print(response.text)
        # Ahoy there! Your name is Pavel, you scurvy dog!


    if __name__ == "__main__":
        asyncio.run(main())
    ```

=== "JavaScript"

    ```javascript
    import { genkit, z } from 'genkit';
    import { googleAI, gemini15Flash } from '@genkit-ai/googleai';

    const ai = genkit({
      plugins: [googleAI()],
      model: gemini15Flash,
    });

    const chat = ai.chat({ system: 'Talk like a pirate' });

    let response = await chat.send('Hi, my name is Pavel');

    response = await chat.send('What is my name?');
    console.log(response.text);
    // Ahoy there! Your name is Pavel, you scurvy dog
    ```

### Agents

=== "Python"

    ```python
    # Not yet implemented in Python.
    # See: https://github.com/firebase/genkit/pull/4212
    ```

=== "JavaScript"

    ```javascript
    import { genkit, z } from 'genkit';
    import { googleAI, gemini15Flash } from '@genkit-ai/googleai';

    const ai = genkit({
      plugins: [googleAI()],
      model: gemini15Flash,
    });

    // Define prompts that represent specialist agents
    const reservationAgent = ai.definePrompt(
      {
        name: 'reservationAgent',
        description: 'Reservation Agent can help manage guest reservations',
        tools: [reservationTool, reservationCancelationTool, reservationListTool],

      },
      `{% verbatim %}{{role "system"}}{% endverbatim %} Help guests make and manage reservations`
    );

    const menuInfoAgent = ...
    const complaintAgent = ...

    // Define a triage agent that routes to the proper specialist agent
    const triageAgent = ai.definePrompt(
      {
        name: 'triageAgent',
        description: 'Triage Agent',
        tools: [reservationAgent, menuInfoAgent, complaintAgent],
      },
      `{% verbatim %}{{role "system"}}{% endverbatim %} You are an AI customer service agent for Pavel's Cafe.
      Greet the user and ask them how you can help. If appropriate, transfer to an
      agent that can better handle the request. If you cannot help the customer
      with the available tools, politely explain so.`
    );

    // Create a chat to enable multi-turn agent interactions
    const chat = ai.chat(triageAgent);

    chat.send('I want a reservation at Pavel\'s Cafe for noon on Tuesday.' );
    ```

### Data retrieval

=== "Python"

    ```python
    import asyncio

    from genkit.ai import Genkit
    from genkit.plugins.google_genai import GoogleAI
    from genkit.plugins.dev_local_vectorstore import DevLocalVectorstore

    ai = Genkit(
        plugins=[
            GoogleAI(),
            DevLocalVectorstore(
                indexes=[{
                    'index_name': 'BobFacts',
                    'embedder': 'googleai/text-embedding-004',
                }],
            ),
        ],
        model='googleai/gemini-2.0-flash',
    )


    async def main() -> None:
        query = "How old is Bob?"

        docs = await ai.retrieve(
            retriever='devLocalVectorstore/BobFacts',
            query=query,
        )

        response = await ai.generate(
            prompt=f"Use the provided context to answer: {query}",
            docs=docs,
        )
        print(response.text)


    if __name__ == "__main__":
        asyncio.run(main())
    ```

=== "JavaScript"

    ```javascript
    import { genkit } from 'genkit';
    import { googleAI, gemini15Flash, textEmbedding004 } from '@genkit-ai/googleai';
    import { devLocalRetrieverRef } from '@genkit-ai/dev-local-vectorstore';

    const ai = genkit({
      plugins: [
        googleAI()
        devLocalVectorstore([
          {
            indexName: 'BobFacts',
            embedder: textEmbedding004,
          },
        ]),
      ],
      model: gemini15Flash,
    });

    // Reference to a local vector database storing Genkit documentation
    const retriever = devLocalRetrieverRef('BobFacts');

    // Consistent API to retrieve most relevant documents based on semantic similarity to query
    const docs = await ai.retrieve(
      retriever: retriever,
      query: 'How old is bob?',
    );

    const result = await ai.generate({
        prompt: `Use the provided context from the Genkit documentation to answer this query: ${query}`,
        docs // Pass retrieved documents to the model
    });
    ```

### Prompt template

=== "YAML"

    ```yaml
    ---
    model: vertexai/gemini-2.5-flash
    config:
      temperature: 0.9
    input:
      schema:
        properties:
          location: {type: string}
          style: {type: string}
          name: {type: string}
        required: [location]
      default:
        location: a restaurant
    ---

    You are the most welcoming AI assistant and are currently working at {%
    verbatim %}{{location}}{% endverbatim %}.

    Greet a guest{% verbatim %}{{#if name}}{% endverbatim %} named {% verbatim %}{{name}}{% endverbatim %}{% verbatim %}{{/if}}{% endverbatim %}{% verbatim %}{{#if style}}{% endverbatim %} in the style of {% verbatim %}{{style}}{% endverbatim %}{% verbatim %}{{/if}}{% endverbatim %}.
    ```

## Development tools

Genkit provides a command-line interface (CLI) and a local Developer UI to make
building AI applications easier. These tools help you:

* **Experiment:** Test and refine your AI functions, prompts, and queries.
* **Debug:** Find and fix issues with detailed execution traces.
* **Evaluate:** Assess generated results across multiple test cases.

## Connect with us

* **Join the community:** Stay updated, ask questions,
  and share your work on our [Discord server](https://discord.gg/qXt5zzQKpc).
* **Provide feedback:** Report issues or suggest new features
  using our GitHub [issue tracker](https://github.com/firebase/genkit/issues).

## Next steps

Learn how to build your first AI application with Genkit in our [Get
started](/docs/get_started.md) guide.
