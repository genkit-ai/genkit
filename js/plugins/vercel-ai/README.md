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
  // The `useChat` `id` is sent to the agent as its `sessionId`; it must be a UUID.
  const chatId = useMemo(() => crypto.randomUUID(), []);

  const { messages, input, handleInputChange, handleSubmit, status } =
    useChat({ id: chatId, transport });


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
  // The `useChat` `id` is sent to the agent as its `sessionId`; it must be a UUID.
  const chatId = useMemo(() => crypto.randomUUID(), []);
  const { messages, status, sendMessage } = useChat({ id: chatId, transport });


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

> **The `useChat` `id` must be a bare UUID.** The transport sends the chat
> `id` to the agent as its Genkit `sessionId`, and the agent server requires
> session ids to be bare UUIDs. Generate one with `crypto.randomUUID()` and
> pass it to `useChat({ id })`. If you omit `id`, the AI SDK generates a
> non-UUID id and the transport will return an error.

```tsx
import { useMemo } from 'react';

const chatId = useMemo(() => crypto.randomUUID(), []);
const { messages, sendMessage } = useChat({ id: chatId, transport });
```

#### Server-managed session state

Conversation state is **fully server-managed**. The transport sends the chat
`id` to the agent as a `sessionId`, and the agent persists per-session state in
its configured `SessionStore`. Each turn automatically resumes the session's
latest snapshot — there is **no client-side snapshot bookkeeping** to manage or
persist, and nothing is lost on a page reload (the server holds the state).

To continue an existing conversation, simply reuse the same UUID `id`.

#### Restoring a conversation after a reload

Because state is server-managed, resuming the *next* turn requires nothing more
than reusing the same UUID `id`. The only thing the client must rebuild is the
*rendered* message list (`useChat` owns those, and they are not persisted by
the transport).

Use `messagesFromSnapshot` to rebuild the visible messages from a Genkit
`SessionSnapshot` (loaded via a `/state` flow keyed by the same `sessionId`):

```tsx
import { useChat } from '@ai-sdk/react';
import { GenkitChatTransport } from '@genkit-ai/vercel-ai/client';
import { messagesFromSnapshot } from '@genkit-ai/vercel-ai';
import { runFlow } from 'genkit/beta/client';
import { useEffect, useMemo, useState } from 'react';

export default function RestoredChat({ sessionId }: { sessionId: string }) {
  const transport = useMemo(() => new GenkitChatTransport({ url: '/api/chat/weather' }), []);
  const [initialMessages, setInitialMessages] = useState([]);

  useEffect(() => {
    (async () => {
      // Load the latest snapshot for this session (returns a SessionSnapshot).
      const snapshot = await runFlow({ url: '/api/chat/weather/state', input: sessionId });
      // Rebuild the visible message list from the snapshot.
      setInitialMessages(messagesFromSnapshot(snapshot.state.messages));
    })();
  }, [sessionId]);

  // Reuse the same UUID `id` so the next turn resumes the server-side session.
  const { messages, sendMessage } = useChat({ id: sessionId, transport, messages: initialMessages });
  // ...render messages, send new turns — continuity is preserved.
}
```

`messagesFromSnapshot` merges Genkit's separate tool request / tool response
messages into single AI SDK tool parts (so each tool call renders as one
element), maps `reasoning` and `media` parts, preserves part `metadata`, and
emits any unresolved tool request (e.g. a pending interrupt) in the
`input-available` state so the UI can still resolve it.

> **Tip:** the `sessionId` you load the snapshot with must be the same UUID you
> pass as the `useChat` `id`, so the rendered messages and the server-side
> session stay in sync.

#### Regenerating a response

When the UI triggers a regeneration (the AI SDK sends
`trigger: 'regenerate-message'`), the transport re-runs the last user message
as a **fresh turn** against the current server-side session state. This
produces a new answer to the same prompt. Because state is server-managed by
`sessionId`, there is no client-side snapshot pointer to rewind to — the
regenerated turn is appended to the session like any other turn.

#### How it works

1. **Sends messages** via Genkit's `streamFlow` client — a browser-safe HTTP
   client that POSTs to the agent endpoint and reads the NDJSON streaming
   response.
2. **Transforms** Genkit's `AgentStreamChunk` events into Vercel AI SDK
   `UIMessageChunk` events (`text-delta`, `tool-input-available`,
   `tool-output-available`, etc.).
3. **Sends the session id** — the transport passes `init: { sessionId: chatId }`
   to the agent, which persists per-session state server-side and resumes the
   session's latest snapshot on each turn. No client-side snapshot tracking is
   needed.
4. **Handles interrupts** — when an agent pauses for human input (via
   `defineInterrupt`), the transport surfaces the pending tool call to the UI.
   On the next `sendMessages` call the resolved tool results are detected
   directly from the message history, and the transport automatically sends a
   `resume` payload instead of a new message.


### Interrupts (human-in-the-loop)

Genkit agents can **pause** mid-run to ask a human for input via
`defineInterrupt` (or a tool that calls `interrupt()`). When this happens,
the transport surfaces the paused tool call to `useChat` as a tool part in
the `input-available` state. There are two ways to resolve it:

1. **Respond** — supply the tool's output directly (the human's decision
   _is_ the result). This is what a `defineInterrupt` tool expects.
2. **Restart** — ask the agent to **re-run** the tool server-side, optionally
   attaching metadata. Useful for tools that `interrupt()` until confirmed and
   then compute a real result on resume (their `resumed` argument is set).

Both are driven by the AI SDK's native HITL primitives — `addToolResult` plus
`sendAutomaticallyWhen` — so no manual `setMessages`/`flushSync` is needed:

```tsx
import { useChat } from '@ai-sdk/react';
import {
  GenkitChatTransport,
  restartInterrupt,
} from '@genkit-ai/vercel-ai/client';
import { lastAssistantMessageIsCompleteWithToolCalls } from 'ai';

const { messages, addToolResult } = useChat({
  transport: new GenkitChatTransport({ url: '/api/chat/banking' }),
  // Auto-resume once every paused tool call has a result.
  sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithToolCalls,
});

// 1. Respond: provide the tool output (resume.respond).
addToolResult({
  tool: 'userApproval',
  toolCallId,
  output: { approved: true, feedback: 'Looks good' },
});

// 2. Restart: re-run the tool (resume.restart). The metadata becomes the
//    tool's `resumed` argument server-side.
addToolResult({
  tool: 'getExchangeRate',
  toolCallId,
  output: restartInterrupt({ confirmedAt: Date.now() }),
});
```

> **Note:** Genkit interrupts map to the AI SDK's `addToolResult` flow, **not**
> the tool-_approval_ flow (`needsApproval` / `addToolApprovalResponse`). A
> Genkit interrupt never executes server-side on its own — the value you pass
> to `addToolResult` _is_ the resolution. Use `restartInterrupt()` when you
> want the agent to re-run the tool instead of accepting a supplied output.

See the full working example in
[`js/testapps/vercel-ai-elements`](../../testapps/vercel-ai-elements), which
demonstrates both the approval (`userApproval`) and restart (`getExchangeRate`)
flows.

### Mapping utilities

The root entry point (`@genkit-ai/vercel-ai`) exports mapping functions
between Vercel AI SDK `UIMessage` types and Genkit `MessageData`, plus
helper functions used by the transport:

| Function | Direction | Description |
| -------- | --------- | ----------- |
| `mapUIMessageToGenkit(msg)` | UI → Genkit | Convert a Vercel `UIMessage` to a Genkit `MessageData` |
| `mapUIPartToGenkit(part)` | UI → Genkit | Convert a single `UIMessagePart` to Genkit `Part[]` |
| `messagesFromSnapshot(msgs)` | Genkit → UI | Convert a `SessionSnapshot`'s `state.messages` into `UIMessage[]` for rehydrating `useChat` (pairs tool requests/responses, preserves metadata) |
| `mapGenkitMessageToUI(msg)` | Genkit → UI | Convert a single Genkit `MessageData` into a `UIMessage` |
| `extractResolvedToolResults(msgs)` | — | Extract resolved tool invocation results from the message array |
| `findLastUserMessage(msgs)` | — | Find the last user message in a `UIMessage[]` |

Tool/part metadata is preserved in both directions: `mapUIPartToGenkit`
carries a UI part's `metadata` onto the Genkit part, and `messagesFromSnapshot`
carries a Genkit part's `metadata` back onto the UI part.


The `UIMessage` type is also re-exported from the `ai` package for convenience.

## Features

- **Text streaming** — streams token-by-token text deltas to the UI
- **Multi-turn conversations** — automatic server-managed session continuity keyed by the `useChat` `id` (`sessionId`)
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
