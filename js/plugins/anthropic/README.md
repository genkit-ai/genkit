# Genkit Anthropic AI Plugin

`@genkit-ai/anthropic` is Anthropic plugin for [Genkit](https://github.com/genkit-ai/genkit).
It supersedes the earlier community package `genkitx-anthropic` and is now maintained by Google.

## Supported models

The plugin supports the most recent Anthropic models like **Claude Haiku 4.5**,
**Claude Sonnet 4.5**, and **Claude Opus 4.5**. Additionally, the plugin
supports all of the
[non-retired older models](https://platform.claude.com/docs/en/about-claude/model-deprecations#model-status).

## Installation

Install the plugin in your project with your favorite package manager:

- `npm install @genkit-ai/anthropic`
- `yarn add @genkit-ai/anthropic`
- `pnpm add @genkit-ai/anthropic`

## Usage

### Initialize

```typescript
import { genkit } from "genkit";
import { anthropic } from "@genkit-ai/anthropic";

const ai = genkit({
  plugins: [anthropic({ apiKey: process.env.ANTHROPIC_API_KEY })],
  // specify a default model for generate here if you wish:
  model: anthropic.model("claude-sonnet-4-5"),
});
```

### Basic examples

The simplest way to generate text is by using the `generate` method:

```typescript
const response = await ai.generate({
  model: anthropic.model("claude-haiku-4-5"),
  prompt: "Tell me a joke.",
});

console.log(response.text);
```

### Multi-modal prompt

```typescript
// ...initialize Genkit instance (as shown above)...

const response = await ai.generate({
  prompt: [
    { text: "What animal is in the photo?" },
    { media: { url: imageUrl } },
  ],
});
console.log(response.text);
```

### Extended thinking

Claude 4.5 models can expose their internal reasoning. Enable it per-request
with the Anthropic thinking config and read the reasoning from the response:

```typescript
const response = await ai.generate({
  prompt: "Walk me through your reasoning for Fermat’s little theorem.",
  config: {
    thinking: {
      enabled: true,
      budgetTokens: 4096, // Must be >= 1024 and less than max_tokens
    },
  },
});

console.log(response.text); // Final assistant answer
console.log(response.reasoning); // Summarized thinking steps
```

When thinking is enabled, request bodies sent through the plugin include the
`thinking` payload (`{ type: 'enabled', budget_tokens: … }`) that Anthropic's
API expects, and streamed responses deliver `reasoning` parts as they arrive so
you can render the chain-of-thought incrementally.

### Document Citations

Claude can cite specific parts of documents you provide, making it easy to trace
where information in the response came from. Use the `anthropicDocument()`
helper to create citable documents. For more details, see the
[Anthropic Citations documentation](https://platform.claude.com/docs/en/build-with-claude/citations).

```typescript
import { anthropic, anthropicDocument } from "@genkit-ai/anthropic";

const response = await ai.generate({
  model: anthropic.model("claude-sonnet-4-5"),
  messages: [
    {
      role: "user",
      content: [
        anthropicDocument({
          source: {
            type: "text",
            data: "The grass is green. The sky is blue. Water is wet.",
          },
          title: "Basic Facts",
          citations: { enabled: true },
        }),
        { text: "What color is the grass? Cite your source." },
      ],
    },
  ],
});

// Access citations from response parts
const citations = response.message?.content?.flatMap(
  (part) => part.metadata?.citations || [],
) ?? [];

console.log("Citations:", citations);
```

**Important:** Citations must be enabled on all documents in a request, or on
none of them. You cannot mix documents with citations enabled and disabled in
the same request.

Supported document source types:

- `text` - Plain text documents (returns `char_location` citations)
- `base64` - Base64-encoded PDFs (returns `page_location` citations)
- `url` - PDFs accessible via URL (returns `page_location` citations)
- `content` - Custom content blocks with text/images (returns
  `content_block_location` citations)
- `file` - File references from Anthropic's Files API, beta API only (returns
  `page_location` citations)

Citations are returned in the response parts' metadata and include information
about the document index, cited text, and location (character indices, page
numbers, or block indices depending on the source type).

### Prompt Caching

You can cache prompts by adding `cache_control` metadata to the prompt. Use the
`cacheControl()` helper for type-safe cache configuration. You can cache system
messages, user messages, tools, and media.

```typescript
import { anthropic, cacheControl } from "@genkit-ai/anthropic";

const response = await ai.generate({
  model: anthropic.model("claude-sonnet-4-5"),
  system: {
    text: longSystemPrompt,
    metadata: { ...cacheControl() }, // default: ephemeral
  },
  messages: [
    {
      role: "user",
      content: [{ text: "What is the main idea of the text?" }],
    },
  ],
});

// Or with explicit TTL:
metadata: { ...cacheControl({ ttl: '1h' }) }

// Or using the type directly:
import { type AnthropicCacheControl } from "@genkit-ai/anthropic";
metadata: { cache_control: { type: 'ephemeral', ttl: '5m' } as AnthropicCacheControl }
```

Note: Caching is only used when the prompt exceeds a certain token length. This
token length is documented in the
[Anthropic API documentation](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

### Beta API Limitations

The beta API surface provides access to experimental features, but some
server-managed tool blocks are not yet supported by this plugin. The following
beta API features will cause an error if encountered:

- `web_fetch_tool_result`
- `code_execution_tool_result`
- `bash_code_execution_tool_result`
- `text_editor_code_execution_tool_result`
- `mcp_tool_result`
- `mcp_tool_use`
- `container_upload`

Note that `server_tool_use` and `web_search_tool_result` ARE supported and work
with both stable and beta APIs.

### Within a flow

```typescript
import { z } from "genkit";

// ...initialize Genkit instance (as shown above)...

export const jokeFlow = ai.defineFlow(
  {
    name: "jokeFlow",
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async (subject) => {
    const llmResponse = await ai.generate({
      prompt: `tell me a joke about ${subject}`,
    });
    return llmResponse.text;
  },
);
```

### Direct model usage (without Genkit instance)

The plugin supports Genkit Plugin API v2, which allows you to use models
directly without initializing the full Genkit framework:

```typescript
import { anthropic } from "@genkit-ai/anthropic";

// Create a model reference directly
const claude = anthropic.model("claude-sonnet-4-5");

// Use the model directly
const response = await claude({
  messages: [
    {
      role: "user",
      content: [{ text: "Tell me a joke." }],
    },
  ],
});

console.log(response);
```

You can also create model references using the plugin's `model()` method:

```typescript
import { anthropic } from "@genkit-ai/anthropic";

// Create model references
const claudeHaiku45 = anthropic.model("claude-haiku-4-5");
const claudeSonnet45 = anthropic.model("claude-sonnet-4-5");
const claudeOpus45 = anthropic.model("claude-opus-4-5");

// Use the model reference directly
const response = await claudeSonnet45({
  messages: [
    {
      role: "user",
      content: [{ text: "Hello!" }],
    },
  ],
});
```

This approach is useful for:

- Framework developers who need raw model access
- Testing models in isolation
- Using Genkit models in non-Genkit applications

## Acknowledgements

This plugin builds on the community work published as
[`genkitx-anthropic`](https://github.com/BloomLabsInc/genkit-plugins/blob/main/plugins/anthropic/README.md)
by Bloom Labs Inc. Their Apache 2.0–licensed implementation provided the
foundation for this maintained package.


## Credits

This plugin is maintained by Google with acknowledgement to the community
contributions from [Bloom Labs Inc](https://github.com/BloomLabsInc).
