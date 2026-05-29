# @genkit-ai/vercel-ai

A [Genkit](https://github.com/genkit-ai/genkit) plugin that provides a
[Vercel AI SDK](https://sdk.vercel.ai/) `ChatTransport` for connecting
`useChat` to Genkit Agents.

Use this plugin to build rich chat UIs in React (Next.js, Vite, etc.) that
stream responses from Genkit agents — including multi-turn conversations,
tool calls, and interrupt-based human-in-the-loop flows — with zero
custom plumbing.

## Installation

```bash
npm install @genkit-ai/vercel-ai
```

### Peer dependencies

| Package | Required | Notes |
| ------- | -------- | ----- |
| `genkit` | ✅ | `>=1.0.0` — the agent API used by this transport lives in the `genkit/beta` subpath. |
| `ai` | ✅ | `^6.0.0` — provides `ChatTransport`, `UIMessage`, etc. |
| `@ai-sdk/react` | Client only | `^3.0.0` — only needed where you call the `useChat` hook (browser). |

> **Note:** `@ai-sdk/react` is not a declared peer dependency of this package
> because the server-side entry point does not require it. Install it
> yourself in any project that uses `useChat`.

## Quick start

### 1. Define a Genkit agent (server)

```ts
// app/api/chat/weather/route.ts  (Next.js App Router example)
import { appRoute } from '@genkit-ai/next';
import { genkit } from 'genkit/beta';
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({ plugins: [googleAI()] });

const weatherAgent = ai.defineAgent({
  name: 'weatherAgent',
  model: 'googleai/gemini-flash-latest',
  tools: [/* your tools here */],
});

export const POST = appRoute(weatherAgent);
```

Or with Express:

```ts
import { expressHandler } from '@genkit-ai/express';
import express from 'express';

const app = express();
app.post('/api/chat/weather', expressHandler(weatherAgent));
app.listen(3000);
```

### 2. Connect from the client

```tsx
'use client';
import { useChat } from '@ai-sdk/react';
import { GenkitChatTransport } from '@genkit-ai/vercel-ai/client';
import { useMemo } from 'react';

export default function Chat() {
  const transport = useMemo(
    () => new GenkitChatTransport({ url: '/api/chat/weather' }),
    []
  );

  const { messages, input, handleInputChange, handleSubmit, status } =
    useChat({ transport });

  return (
    <div>
      {messages.map((m) => (
        <div key={m.id}>
          <strong>{m.role}:</strong>{' '}
          {m.parts
            .filter((p) => p.type === 'text')
            .map((p) => p.text)
            .join('')}
        </div>
      ))}
      <form onSubmit={handleSubmit}>
        <input value={input} onChange={handleInputChange} />
        <button type="submit" disabled={status !== 'ready'}>
          Send
        </button>
      </form>
    </div>
  );
}
```

That's it — text streaming, multi-turn history, tool calls, and interrupts
all work out of the box.

## Works with AI Elements

This transport is fully compatible with
[AI Elements](https://elements.ai-sdk.dev/) — the component library and
custom [shadcn/ui](https://ui.shadcn.com/) registry for building AI-native
applications. Since `GenkitChatTransport` implements the standard
`ChatTransport` interface, all AI Elements components (`Conversation`,
`Message`, `Tool`, `PromptInput`, etc.) work out of the box with Genkit
agents.

```tsx
'use client';
import { useChat } from '@ai-sdk/react';
import { GenkitChatTransport } from '@genkit-ai/vercel-ai/client';
import { useMemo } from 'react';

import {
  Conversation, ConversationContent, ConversationScrollButton,
} from '@/components/ai-elements/conversation';
import { Message, MessageContent, MessageResponse } from '@/components/ai-elements/message';
import {
  PromptInput, PromptInputTextarea, PromptInputFooter, PromptInputSubmit,
} from '@/components/ai-elements/prompt-input';
import { Tool, ToolHeader, ToolContent, ToolInput, ToolOutput } from '@/components/ai-elements/tool';

export default function Chat() {
  const transport = useMemo(
    () => new GenkitChatTransport({ url: '/api/chat/weather' }),
    []
  );
  const { messages, status, sendMessage } = useChat({ transport });

  return (
    <Conversation>
      <ConversationContent>
        {messages.map((message) => (
          <Message key={message.id} from={message.role}>
            <MessageContent>
              {message.parts.map((part, i) => {
                if (part.type === 'text' && part.text) {
                  return <MessageResponse key={i}>{part.text}</MessageResponse>;
                }
                if (part.type.startsWith('tool-')) {
                  const toolPart = part as any;
                  return (
                    <Tool key={i}>
                      <ToolHeader type={toolPart.type} state={toolPart.state} />
                      <ToolContent>
                        <ToolInput input={toolPart.input} />
                        <ToolOutput output={toolPart.output} />
                      </ToolContent>
                    </Tool>
                  );
                }
                return null;
              })}
            </MessageContent>
          </Message>
        ))}
      </ConversationContent>
      <ConversationScrollButton />
      <PromptInput onSubmit={(msg) => sendMessage({ text: msg.text })}>
        <PromptInputTextarea placeholder="Message..." />
        <PromptInputFooter>
          <div />
          <PromptInputSubmit status={status} />
        </PromptInputFooter>
      </PromptInput>
    </Conversation>
  );
}
```

See the full working example in
[`js/testapps/vercel-ai-elements`](../../testapps/vercel-ai-elements) which
demonstrates multi-agent selection, tool call rendering, and interrupt-based
human-in-the-loop approval flows — all powered by AI SDK Elements.

## Entry points

| Import path | Environment | Contents |
| ----------- | ----------- | -------- |
| `@genkit-ai/vercel-ai` | Server / shared | Mapping utilities (`mapUIMessageToGenkit`, `mapUIPartToGenkit`, etc.) and re-exported `UIMessage` type |
| `@genkit-ai/vercel-ai/client` | Browser / client | `GenkitChatTransport` class |

The `/client` entry point is browser-safe and has no Node.js dependencies.

## API

### `GenkitChatTransport`

```ts
import { GenkitChatTransport } from '@genkit-ai/vercel-ai/client';

const transport = new GenkitChatTransport({
  url: '/api/chat/weather',
  headers: { Authorization: 'Bearer token' },         // static headers
  // headers: () => ({ Authorization: `Bearer ${getToken()}` }),  // or dynamic
});
```

#### Configuration

| Option | Type | Description |
| ------ | ---- | ----------- |
| `url` | `string` | **Required.** URL of the Genkit agent endpoint. |
| `headers` | `Record<string, string> \| () => Record<string, string>` | Optional HTTP headers to include on every request. Accepts a static object or a function for dynamic values (e.g. auth tokens). |

#### How it works

1. **Sends messages** via Genkit's `streamFlow` client — a browser-safe HTTP
   client that POSTs to the agent endpoint and reads the NDJSON streaming
   response.
2. **Transforms** Genkit's `AgentStreamChunk` events into Vercel AI SDK
   `UIMessageChunk` events (`text-delta`, `tool-input-available`,
   `tool-output-available`, etc.).
3. **Tracks session state** client-side using a `chatId → snapshotId` map,
   so multi-turn conversations automatically resume from the correct
   server-side snapshot without any additional state management.
4. **Handles interrupts** — when an agent pauses for human input (via
   `defineInterrupt`), the transport detects the pending tool request and
   sets an internal interrupted flag. On the next `sendMessages` call with
   resolved tool results, it automatically sends a resume payload instead
   of a new message.

### Mapping utilities

The root entry point (`@genkit-ai/vercel-ai`) exports mapping functions
between Vercel AI SDK `UIMessage` types and Genkit `MessageData`, plus
helper functions used by the transport:

| Function | Direction | Description |
| -------- | --------- | ----------- |
| `mapUIMessageToGenkit(msg)` | UI → Genkit | Convert a Vercel `UIMessage` to a Genkit `MessageData` |
| `mapUIPartToGenkit(part)` | UI → Genkit | Convert a single `UIMessagePart` to Genkit `Part[]` |
| `extractResolvedToolResults(msgs)` | — | Extract resolved tool invocation results from the message array |
| `findLastUserMessage(msgs)` | — | Find the last user message in a `UIMessage[]` |

The `UIMessage` type is also re-exported from the `ai` package for convenience.

## Features

- **Text streaming** — streams token-by-token text deltas to the UI
- **Multi-turn conversations** — automatic snapshot-based session continuity
- **Tool calls** — surfaces tool inputs and outputs as they happen
- **Interrupts (human-in-the-loop)** — first-class support for `defineInterrupt` tools; pause and resume flows with user input
- **Custom headers** — static or dynamic headers for authentication
- **Browser-safe** — the `/client` entry point has no Node.js dependencies
- **Zero config** — no `ChatSessionStore` needed on the client side

## Requirements

- Genkit `>=1.0.0` — the agent API (`defineAgent`, `streamFlow`) used by this
  transport is exposed via the `genkit/beta` subpath of the main `1.x` package.
- Vercel AI SDK `>=6.0.0` (for `useChat` with `ChatTransport`)
- `@ai-sdk/react` `>=3.0.0` (client-side, for the `useChat` hook)
- Node.js `>=20` (server-side)

## License

Apache-2.0 — see [LICENSE](./LICENSE).
