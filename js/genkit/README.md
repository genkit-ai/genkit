# Genkit

Genkit is a framework for building AI-powered applications. It provides open source libraries for Node.js and Go, along with tools to help you debug and iterate quickly.

## Install Genkit dependencies

Install the following Genkit dependencies to use Genkit in your project:

- `genkit` provides Genkit core capabilities.
- `@genkit-ai/google-genai` provides access to the Google AI Gemini models. Check out other plugins: https://www.npmjs.com/search?q=keywords:genkit-plugin

```posix-terminal
npm install genkit @genkit-ai/google-genai
```

## Make your first request

Get started with Genkit in just a few lines of simple code.

```ts
// import the Genkit and Google AI plugin libraries
import { genkit } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({ plugins: [googleAI()] });

const { text } = await ai.generate({
    model: googleAI.model('gemini-2.5-flash'),
    prompt: 'Why is Genkit awesome?'
});
```

Genkit also provides middleware to add common functionality to your AI requests. For example, you can use the `retry` middleware to automatically retry failed requests:

```ts
import { retry } from 'genkit/model/middleware';

const { text } = await ai.generate({
    model: googleAI.model('gemini-2.5-flash'),
    prompt: 'Why is Genkit awesome?',
    use: [
      retry({
        maxRetries: 3,
        initialDelayMs: 1000,
        backoffFactor: 2,
      }),
    ],
});
```

Genkit also lets you build strongly typed, accessible from the client, fully observable AI flows:

```ts
import { googleAI } from '@genkit-ai/google-genai';
import { genkit, z } from 'genkit';

// Initialize Genkit with the Google AI plugin
const ai = genkit({
  plugins: [googleAI()],
  model: googleAI.model('gemini-2.5-flash', {
    temperature: 0.8
  }),
});

// Define input schema
const RecipeInputSchema = z.object({
  ingredient: z.string().describe('Main ingredient or cuisine type'),
  dietaryRestrictions: z.string().optional().describe('Any dietary restrictions'),
});

// Define output schema
const RecipeSchema = z.object({
  title: z.string(),
  description: z.string(),
  prepTime: z.string(),
  cookTime: z.string(),
  servings: z.number(),
  ingredients: z.array(z.string()),
  instructions: z.array(z.string()),
  tips: z.array(z.string()).optional(),
});

// Define a recipe generator flow
export const recipeGeneratorFlow = ai.defineFlow(
  {
    name: 'recipeGeneratorFlow',
    inputSchema: RecipeInputSchema,
    outputSchema: RecipeSchema,
  },
  async (input, { sendChunk }) => {
    // Create a prompt based on the input
    const prompt = `Create a recipe with the following requirements:
      Main ingredient: ${input.ingredient}
      Dietary restrictions: ${input.dietaryRestrictions || 'none'}`;

    // Generate structured recipe data using the same schema
    const { output } = await ai.generate({
      prompt,
      output: { schema: RecipeSchema },
      onChunk: sendChunk // stream output
    });

    if (!output) throw new Error('Failed to generate recipe');

    return output;
  }
);

// Run the flow locally
async function main() {
  const recipe = await recipeGeneratorFlow({
    ingredient: 'avocado',
    dietaryRestrictions: 'vegetarian'
  });

  console.log(recipe);
}

main().catch(console.error);
```

You can easily serve flows as an API:

```ts
import { startFlowServer } from '@genkit-ai/express'; // npm i @genkit-ai/express

startFlowServer({
  flows: [recipeGeneratorFlow],
});
```
And access the flow from the client:

```ts
import { runFlow } from 'genkit/beta/client';

const { stream } = streamFlow({
  url: 'http://localhost:3500/recipeGeneratorFlow',
  input: {
    ingredient: 'avocado',
    dietaryRestrictions: 'vegetarian'
  },
});

for await (const chunk of stream) {
  console.log(chunk);
}
```

For more details see: https://genkit.dev/docs/deploy-node

But you can also deploy to [Firebase](https://genkit.dev/docs/firebase/) or [Cloud Run](https://genkit.dev/docs/cloud-run/), etc.

## Next steps

Now that you’re set up to make model requests with Genkit, learn how to use more
Genkit capabilities to build your AI-powered apps and workflows. To get started
with additional Genkit capabilities, see the following guides:

- [Developer tools](https://genkit.dev/docs/devtools/): Learn how to set up and use
  Genkit’s CLI and developer UI to help you locally test and debug your app.
- [Generating content](https://genkit.dev/docs/models/): Learn how to use Genkit’s unified
  generation API to generate text and structured data from any supported
  model.
- [Creating flows](https://genkit.dev/docs/flows/): Learn how to use special Genkit
  functions, called flows, that provide end-to-end observability for workflows
  and rich debugging from Genkit tooling.
- [Managing prompts](https://genkit.dev/docs/dotprompt/): Learn how Genkit helps you manage
  your prompts and configuration together as code.

Learn more at [https://genkit.dev](https://genkit.dev)

License: Apache 2.0
