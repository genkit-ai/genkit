# RFC: Genkit Client SDK

**Status:** Draft
**Authors:** Elliot Hesp
**Created:** 2026-02-21
**Last Updated:** 2026-02-21

---

## Summary

This RFC proposes a modern, batteries-included client SDK for consuming Genkit flows from JavaScript/TypeScript client applications. The design introduces a client factory pattern, end-to-end type inference from server-defined flows, built-in retry/resilience, and framework-specific hooks for React and Vue (or others).

---

## Table of Contents

- [Motivation](#motivation)
- [Goals and Non-Goals](#goals-and-non-goals)
- [Current State](#current-state)
- [Proposed Design](#proposed-design)
  - [1. Client Factory](#1-client-factory)
  - [2. Type Inference](#2-type-inference)
  - [3. Typed Flow References](#3-typed-flow-references)
  - [4. Retry and Resilience](#4-retry-and-resilience)
  - [5. React Hooks](#5-react-hooks)
  - [6. Vue Composables](#6-vue-composables)
  - [7. Chat Primitives](#7-chat-primitives)
  - [8. Error Handling](#8-error-handling)
- [Package Structure](#package-structure)
- [Architecture](#architecture)
- [Migration Path](#migration-path)
- [Future Extensions](#future-extensions)
- [Open Questions](#open-questions)
- [Prior Art](#prior-art)

---

## Motivation

The current Genkit client (`js/genkit/src/client/client.ts`) is intentionally minimal -- two standalone functions (`runFlow` and `streamFlow`) with no shared state. Exported from `genkit/beta/client`:

```typescript
import { runFlow, streamFlow } from 'genkit/beta/client';

const result = await runFlow({
  url: 'https://my-app.example.com/api/myFlow',
  input: 'hello',
  headers: { Authorization: `Bearer ${token}` },
});
```

While this simplicity is a strength for low-level or one-off usage, it creates significant friction in modern client-side development:

### 1. No shared configuration

There is no client instance. Every call requires the full URL, headers, and auth tokens to be passed individually. There is no way to configure a base URL, default headers, or shared options once. In a real application with 10+ flows and authentication, this means duplicating configuration across every call site.

### 2. No type inference from flows

Consumers must manually declare `<O>` and `<S>` generics on each call. The types used in a flow are defined on the server -- the input schema, output schema, and stream schema are all known at `defineFlow` time -- but none of this information is available to the client. You end up re-declaring types that already exist:

```typescript
// Server already knows these types via defineFlow schemas
// But on the client, you have to manually re-state them:
const result = await runFlow<{ greeting: string }>({ url: '...', input: 'hello' });
```

The Next.js plugin (`js/plugins/next/src/client.ts`) has `Input<A>`, `Output<A>`, `Stream<A>` conditional type helpers that extract types from an `Action`, but these are scoped to that plugin and not reusable.

### 3. No retry or resilience

The client has zero retry logic. If a request fails due to a transient error (502, 503, 429, network blip), it simply throws. There is no backoff, no jitter, no retry count. Every consumer must build all of this themselves, which leads to inconsistent retry behavior across an application.

### 4. No framework integration

There are no React hooks, Vue composables, or framework adapters. Developers must manually wire up `useState`/`useEffect` for loading states, error handling, `AbortController` management, and streaming chunk accumulation. This is boilerplate that every consumer writes, and it's easy to get wrong (memory leaks from unclean abort, missing error boundaries, stale closures, etc.).

### 5. No client-side chat primitive

The server has `ai.chat()` with session/thread management (`js/ai/src/chat.ts`), but there is no client-side counterpart. Building a chat UI requires manually managing message arrays, appending streamed tokens, handling conversation history, and coordinating with the server's session state.

---

## Goals and Non-Goals

### Goals

- **Client factory** with shared configuration (base URL, headers, retry, custom fetch)
- **End-to-end type safety** by inferring input/output/stream types from server-defined flows
- **Built-in retry** with exponential backoff, jitter, and per-request overrides
- **React hooks** (`useFlow`, `useChat`) with loading/error/streaming state management
- **Vue composables** mirroring the React API
- **Browser-safe** with zero Node.js dependencies
- **Backward compatible** with the existing `genkit/beta/client` API
- **Framework-agnostic core** that React/Vue/Svelte/etc. layers build on top of

### Non-Goals

- Server-side changes to the HTTP/SSE protocol (the current protocol is sufficient)
- UI component library (future extension)
- WebSocket transport (future extension)
- Server-side code generation or schema discovery endpoints (future extension)
- Svelte/Solid/Angular adapters in the initial release (community-driven)

---

## Current State

### Client (`js/genkit/src/client/client.ts`)

Two exported functions, 227 lines total:

- `runFlow<O>({ url, input, headers?, abortSignal? }): Promise<O>` -- POSTs JSON, returns `{ result }` envelope
- `streamFlow<O, S>({ url, input, streamId?, headers?, abortSignal? })` -- POSTs with `Accept: text/event-stream`, parses SSE chunks (`data: {"message": chunk}\n\n`), returns `{ output: Promise<O>, stream: AsyncIterable<S>, streamId: Promise<string | null> }`

### Server Protocol

All server plugins (`@genkit-ai/web`, `@genkit-ai/express`, `@genkit-ai/next`) use the same wire format:

**Request:**
```
POST /<flowName>
Content-Type: application/json
Accept: text/event-stream  (for streaming)

{ "data": <input> }
```

**Non-streaming response:**
```json
{ "result": <output> }
```

**Streaming response (SSE):**
```
data: {"message": <chunk>}\n\n
data: {"message": <chunk>}\n\n
data: {"result": <output>}\n\n
```

**Error (streaming):**
```
error: {"error": {"status": "...", "message": "..."}}\n\n
```

The server also supports **durable streaming** via `StreamManager` and the `x-genkit-stream-id` header for reconnection.

### Next.js Plugin Type Helpers (`js/plugins/next/src/client.ts`)

The only place in the codebase with client-side type inference from `Action`:

```typescript
type Input<A extends Action> =
  A extends Action<infer I extends z.ZodTypeAny, any, any> ? z.infer<I> : never;
type Output<A extends Action> =
  A extends Action<any, infer O extends z.ZodTypeAny, any> ? z.infer<O> : never;
type Stream<A extends Action> =
  A extends Action<any, any, infer S extends z.ZodTypeAny> ? z.infer<S> : never;
```

These are not exported from `genkit` and require importing the full `Action` type from the server.

### Server-Side Chat (`js/ai/src/chat.ts`)

The `Chat` class provides:

- `send(options)` -- sends a message, returns `GenerateResponse`
- `sendStream(options)` -- sends with streaming, returns `{ response, stream }`
- `messages` -- conversation history (`MessageData[]`)
- Session/thread management via `Session`

No client-side equivalent exists.

---

## Proposed Design

### 1. Client Factory

Create a `createGenkitClient` factory function that returns a configured client instance. Inspired by [Better Auth's `createAuthClient`](https://www.better-auth.com/docs/installation#create-client-instance) and the [Cloudflare Agents SDK](https://developers.cloudflare.com/agents/).

```typescript
import { createGenkitClient } from 'genkit/client';

// Static headers
const client = createGenkitClient({
  baseURL: 'https://my-app.example.com/api',
  headers: { Authorization: `Bearer ${token}` },
  retry: {
    maxRetries: 3,
    backoff: 'exponential',
  },
});

// Or dynamic headers -- function is called fresh on every request,
// useful for tokens that expire or need to be fetched per-request.
const client = createGenkitClient({
  baseURL: 'https://my-app.example.com/api',
  headers: async () => {
    const session = await getSession();
    return { Authorization: `Bearer ${session.accessToken}` };
  },
});
```

#### `GenkitClientConfig`

```typescript
interface GenkitClientConfig {
  /** Base URL for all flow requests. Flow paths are appended to this. */
  baseURL: string;

  /**
   * Default headers included with every request.
   * Can be a static object or an async function for dynamic auth tokens.
   */
  headers?: Record<string, string> | (() => Record<string, string> | Promise<Record<string, string>>);

  /** Default retry configuration for all requests. */
  retry?: RetryConfig;

  /** Custom fetch implementation (for testing, middleware, or platform compatibility). */
  fetch?: typeof globalThis.fetch;

  /** Default AbortSignal applied to all requests (e.g., tied to component lifecycle). */
  abortSignal?: AbortSignal;
}
```

#### `GenkitClient`

```typescript
interface GenkitClient {
  /** Invoke a flow and return its output. */
  runFlow<O = any>(
    path: string,
    input?: any,
    options?: RequestOptions,
  ): Promise<O>;

  /** Invoke a flow and stream its response. */
  streamFlow<O = any, S = any>(
    path: string,
    input?: any,
    options?: RequestOptions,
  ): StreamResponse<O, S>;

  /** Create a typed, pre-bound flow reference. */
  flow<A extends Action>(path: string): FlowHandle<A>;
}

interface RequestOptions {
  headers?: Record<string, string>;
  retry?: RetryConfig;
  abortSignal?: AbortSignal;
  streamId?: string;
}

interface StreamResponse<O, S> {
  readonly output: Promise<O>;
  readonly stream: AsyncIterable<S>;
  readonly streamId: Promise<string | null>;
}
```

#### Usage

```typescript
const client = createGenkitClient({
  baseURL: 'https://my-app.example.com/api',
  headers: async () => ({ Authorization: `Bearer ${await getToken()}` }),
});

// Simple untyped call
const greeting = await client.runFlow('/greet', { name: 'world' });

// Streaming
const response = client.streamFlow('/summarize', { text: longArticle });
for await (const chunk of response.stream) {
  process.stdout.write(chunk);
}
const summary = await response.output;

// Per-request overrides
const result = await client.runFlow('/greet', { name: 'world' }, {
  retry: { maxRetries: 5 },
  headers: { 'x-custom': 'value' },
});
```

Dynamic headers are especially important for auth tokens that expire:

```typescript
const client = createGenkitClient({
  baseURL: '/api',
  headers: async () => {
    const session = await getSession();
    return { Authorization: `Bearer ${session.accessToken}` };
  },
});
```

---

### 2. Type Inference

#### The Problem

Today, the source of truth for a flow's types is the server-side `defineFlow` call. But on the client, you have to manually re-declare these types:

```typescript
// Server
const myFlow = ai.defineFlow({
  name: 'myFlow',
  inputSchema: z.string(),
  outputSchema: z.object({ greeting: z.string() }),
  streamSchema: z.string(),
}, async (input, { sendChunk }) => {
  sendChunk(`Processing ${input}...`);
  return { greeting: `Hello, ${input}!` };
});

// Client -- manual, error-prone, disconnected from server
const result = await client.runFlow<{ greeting: string }>('/myFlow', 'world');
```

#### The Solution

Export the flow's type from the server and use TypeScript's type inference on the client:

```typescript
// Server -- export the type (type-only, no runtime cost)
export type MyFlow = typeof myFlow;
```

```typescript
// Client -- import the type, everything is inferred
import type { MyFlow } from '../server/flows';

const result = await client.runFlow<MyFlow>('/myFlow', 'world');
//    ^? { greeting: string }
//                                                      ^? input: string
```

#### Type Utilities

Move and generalize the Next.js plugin's conditional type helpers into `genkit/client`:

```typescript
import type { Action, z } from 'genkit';

/** Extract the inferred input type from a Flow/Action. */
export type FlowInput<A extends Action> =
  A extends Action<infer I extends z.ZodTypeAny, any, any> ? z.infer<I> : never;

/** Extract the inferred output type from a Flow/Action. */
export type FlowOutput<A extends Action> =
  A extends Action<any, infer O extends z.ZodTypeAny, any> ? z.infer<O> : never;

/** Extract the inferred stream chunk type from a Flow/Action. */
export type FlowStream<A extends Action> =
  A extends Action<any, any, infer S extends z.ZodTypeAny> ? z.infer<S> : never;
```

These can be used standalone:

```typescript
import type { FlowInput, FlowOutput, FlowStream } from 'genkit/client';
import type { MyFlow } from '../server/flows';

type In = FlowInput<MyFlow>;     // string
type Out = FlowOutput<MyFlow>;   // { greeting: string }
type Chunk = FlowStream<MyFlow>; // string
```

#### Typed `runFlow` and `streamFlow`

The client methods accept an `Action` type parameter that constrains both the input and output:

```typescript
interface GenkitClient {
  runFlow<A extends Action>(
    path: string,
    input: FlowInput<A>,
    options?: RequestOptions,
  ): Promise<FlowOutput<A>>;

  runFlow<O = any>(
    path: string,
    input?: any,
    options?: RequestOptions,
  ): Promise<O>;

  streamFlow<A extends Action>(
    path: string,
    input: FlowInput<A>,
    options?: RequestOptions,
  ): StreamResponse<FlowOutput<A>, FlowStream<A>>;

  streamFlow<O = any, S = any>(
    path: string,
    input?: any,
    options?: RequestOptions,
  ): StreamResponse<O, S>;
}
```

The overloads mean untyped usage still works -- you're never forced to provide types.

---

### 3. Typed Flow References

For repeated use of the same flow, create a pre-bound typed reference:

```typescript
interface FlowHandle<A extends Action> {
  /** Run the flow (non-streaming). */
  run(input: FlowInput<A>, options?: RequestOptions): Promise<FlowOutput<A>>;

  /** Stream the flow. */
  stream(input: FlowInput<A>, options?: RequestOptions): StreamResponse<FlowOutput<A>, FlowStream<A>>;

  /** The path this handle is bound to. */
  readonly path: string;
}
```

Usage:

```typescript
import type { MyFlow } from '../server/flows';

const myFlow = client.flow<MyFlow>('/myFlow');

// Fully typed -- input, output, and stream chunks are all inferred
const result = await myFlow.run('hello');
//    ^? { greeting: string }

const response = myFlow.stream('hello');
for await (const chunk of response.stream) {
  console.log(chunk);  // ^? string
}
```

This pattern is particularly useful for hook integration:

```typescript
const myFlow = client.flow<MyFlow>('/myFlow');
const { run, data, isLoading } = useFlow(myFlow);
```

---

### 4. Retry and Resilience

The current client has zero retry logic. This section proposes built-in retry with sensible defaults.

#### Configuration

```typescript
interface RetryConfig {
  /** Maximum number of retry attempts. Default: 0 (no retries). */
  maxRetries: number;

  /** Initial delay before the first retry in ms. Default: 1000. */
  initialDelayMs?: number;

  /** Maximum delay between retries in ms. Default: 30000. */
  maxDelayMs?: number;

  /** Backoff strategy. Default: 'exponential'. */
  backoff?: 'exponential' | 'linear' | 'fixed';

  /** Add random jitter to avoid thundering herd. Default: true. */
  jitter?: boolean;

  /** HTTP status codes that are retryable. Default: [429, 502, 503, 504]. */
  retryableStatuses?: number[];

  /** Custom retry predicate for full control. */
  retryOn?: (error: Error, attempt: number) => boolean;
}
```

#### Behavior

- **`runFlow`**: retries the entire request on failure.
- **`streamFlow`**: if a `streamId` is available (durable streaming via `StreamManager`), reconnects to the existing stream. Otherwise, retries the entire request.
- **429 handling**: if the server returns a `Retry-After` header, the client respects it instead of using the calculated delay.
- **Abort**: retries are cancelled if the `AbortSignal` fires.
- **Per-request override**: any call can override or disable retry.

```typescript
// Disable retry for a specific call
await client.runFlow('/myFlow', input, { retry: { maxRetries: 0 } });

// Increase retry for a flaky endpoint
await client.runFlow('/myFlow', input, { retry: { maxRetries: 10 } });
```

#### Retry Delay Calculation

```
exponential: min(initialDelayMs * 2^attempt, maxDelayMs) + jitter
linear:      min(initialDelayMs * (attempt + 1), maxDelayMs) + jitter
fixed:       initialDelayMs + jitter
jitter:      random(0, delay * 0.1)
```

---

### 5. React Hooks

Exported from `genkit/client/react`. Inspired by [Vercel AI SDK's `useChat`](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot) and `useCompletion`.

#### `useFlow`

Manages the lifecycle of a single flow invocation with React state.

```typescript
import { useFlow } from 'genkit/client/react';
import type { MyFlow } from '../server/flows';

function SummarizeButton({ text }: { text: string }) {
  const {
    run,
    stream,
    data,
    chunks,
    isLoading,
    error,
    abort,
  } = useFlow<MyFlow>(client, '/summarize');

  return (
    <div>
      <button onClick={() => run(text)} disabled={isLoading}>
        {isLoading ? 'Summarizing...' : 'Summarize'}
      </button>

      {error && <p className="error">{error.message}</p>}

      {chunks.map((chunk, i) => (
        <span key={i}>{chunk}</span>
      ))}

      {data && <p className="result">{data.summary}</p>}
    </div>
  );
}
```

#### `useFlow` API

```typescript
function useFlow<A extends Action>(
  client: GenkitClient,
  path: string,
  options?: UseFlowOptions<A>,
): UseFlowReturn<A>;

// Also accepts a pre-bound FlowHandle
function useFlow<A extends Action>(
  handle: FlowHandle<A>,
  options?: UseFlowOptions<A>,
): UseFlowReturn<A>;

interface UseFlowOptions<A extends Action> {
  onSuccess?: (data: FlowOutput<A>) => void;
  onError?: (error: Error) => void;
  onChunk?: (chunk: FlowStream<A>) => void;
  retry?: RetryConfig;
}

interface UseFlowReturn<A extends Action> {
  /** Trigger a non-streaming flow run. */
  run: (input: FlowInput<A>, options?: RequestOptions) => void;

  /** Trigger a streaming flow run. */
  stream: (input: FlowInput<A>, options?: RequestOptions) => void;

  /** The resolved output (null until complete). */
  data: FlowOutput<A> | null;

  /** Accumulated stream chunks. */
  chunks: FlowStream<A>[];

  /** Whether a request is in flight. */
  isLoading: boolean;

  /** The most recent error, if any. */
  error: Error | null;

  /** Abort the current in-flight request. */
  abort: () => void;

  /** The durable stream ID, if available. */
  streamId: string | null;

  /** Reset state (data, chunks, error) back to initial values. */
  reset: () => void;
}
```

#### Streaming Example

```typescript
function LiveTranslation({ text, targetLang }: Props) {
  const { stream, chunks, data, isLoading } = useFlow<TranslateFlow>(
    client, '/translate',
    { onChunk: (chunk) => console.log('token:', chunk) }
  );

  return (
    <div>
      <button onClick={() => stream({ text, targetLang })}>
        Translate
      </button>
      <div className="live-output">
        {chunks.join('')}
      </div>
      {data && <div className="final">{data.translation}</div>}
    </div>
  );
}
```

---

### 6. Vue Composables

Exported from `genkit/client/vue`. Mirrors the React API using Vue 3's Composition API.

```typescript
import { useFlow, useChat } from 'genkit/client/vue';
import type { MyFlow } from '../server/flows';

const client = createGenkitClient({ baseURL: '/api' });
const { run, data, isLoading, error } = useFlow<MyFlow>(client, '/myFlow');
```

Vue composables return `Ref<T>` values instead of plain values, following Vue conventions:

```typescript
interface UseFlowReturn<A extends Action> {
  run: (input: FlowInput<A>, options?: RequestOptions) => void;
  stream: (input: FlowInput<A>, options?: RequestOptions) => void;
  data: Ref<FlowOutput<A> | null>;
  chunks: Ref<FlowStream<A>[]>;
  isLoading: Ref<boolean>;
  error: Ref<Error | null>;
  abort: () => void;
  streamId: Ref<string | null>;
  reset: () => void;
}
```

The composable auto-cleans up on component unmount via `onUnmounted`.

> **Note:** Because the core client (`genkit/client`) is framework-agnostic, the same pattern can be extended to any UI framework -- Solid, Angular, Svelte, etc. Each adapter is a thin wrapper that maps the core client's promises and async iterables into the framework's reactivity primitives (e.g., Solid signals, Angular observables, Svelte stores). React and Vue are proposed as the initial targets given their ecosystem share, but the architecture does not preclude others.

---

### 7. Chat Primitives

#### The Challenge: Genkit Flows Are Not Chat Endpoints

In Vercel AI SDK, `useChat` talks to a purpose-built chat endpoint with a well-defined wire protocol -- the server accepts messages and streams back tokens in a known format. The hook and the server are tightly coupled by protocol.

Genkit is architecturally different. There is no first-class "chat endpoint." On the server, `ai.chat()` wraps `generate()` calls with session-managed message history via the `Session` and `Chat` classes -- but over HTTP, everything is just a flow. A flow takes arbitrary input and returns arbitrary output. There is no special chat wire protocol.

This means `useChat` must bridge two worlds: the client-side chat UI pattern (message list, streaming tokens, send/regenerate) and whatever flow the developer has defined on the server. There are two viable approaches:

#### Approach A: Client-Managed History (Stateless Server)

The client owns the conversation history. Each call sends the **full message array** as the flow's input. The server flow is stateless -- it receives messages, calls `generate()`, and returns the response.

```typescript
// === Server ===
const chatFlow = ai.defineFlow({
  name: 'chat',
  inputSchema: z.object({
    messages: z.array(MessageSchema),
  }),
  outputSchema: GenerateResponseSchema,
  streamSchema: GenerateResponseChunkSchema,
}, async (input, { sendChunk }) => {
  const response = await ai.generate({
    model: gemini20Flash,
    messages: input.messages,
    onChunk: sendChunk,
  });
  return response.toJSON();
});

export type ChatFlow = typeof chatFlow;
```

```typescript
// === Client ===
import { useChat } from 'genkit/client/react';
import type { ChatFlow } from '../server/flows';

function ChatUI() {
  const { messages, sendMessage, isStreaming, stop } = useChat<ChatFlow>(client, '/chat');
  // ...
}
```

How `useChat` works internally with this approach:

1. `sendMessage('hello')` appends `{ role: 'user', content: [{ text: 'hello' }] }` to the local `messages` array
2. The hook calls `client.streamFlow('/chat', { messages })` with the **full history**
3. A placeholder assistant message is created; stream chunks update its content in real-time
4. When the stream completes, the final assistant message is resolved and appended
5. `regenerate()` pops the last assistant message and re-sends from the last user message
6. `stop()` aborts via `AbortController`

**Pros:** Simple, no server-side session state, works with any deployment (serverless, edge).
**Cons:** Full history sent every request (grows with conversation length), no server-side session persistence across page reloads.

#### Approach B: Server-Managed History (Stateful Server)

The server owns the conversation history via Genkit's `Session` system. The client sends only the **new user message** plus a **session ID**. The server appends to the session, calls `ai.chat().send()`, and returns the response.

```typescript
// === Server ===
const chatFlow = ai.defineFlow({
  name: 'chat',
  inputSchema: z.object({
    message: z.string(),
    sessionId: z.string().optional(),
  }),
  outputSchema: z.object({
    response: GenerateResponseSchema,
    sessionId: z.string(),
  }),
  streamSchema: GenerateResponseChunkSchema,
}, async (input, { sendChunk }) => {
  const session = input.sessionId
    ? await ai.loadSession(input.sessionId, { store: firestoreStore })
    : ai.createSession({ store: firestoreStore });

  const chat = session.chat({ model: gemini20Flash });
  const response = await chat.send({
    prompt: input.message,
    onChunk: sendChunk,
  });

  return {
    response: response.toJSON(),
    sessionId: session.id,
  };
});

export type ChatFlow = typeof chatFlow;
```

```typescript
// === Client ===
const { messages, sendMessage, isStreaming } = useChat<ChatFlow>(client, '/chat', {
  // useChat stores the sessionId from the first response
  // and passes it on subsequent requests automatically
});
```

**Pros:** Lightweight requests (only new message sent), server-side persistence across reloads, works with Genkit's existing Session/Store system.
**Cons:** Requires stateful server (session store), more complex server flow, session management adds operational overhead.

#### How `useChat` Supports Both

The hook doesn't need to know which approach the server uses. The key abstraction is a **message adapter** -- a pair of functions that map between the hook's internal message state and the flow's actual input/output shapes:

```typescript
interface ChatAdapter<A extends Action> {
  /** Build the flow input from the current messages. Called on each sendMessage. */
  toInput: (messages: ChatMessage[], context: { sessionId?: string }) => FlowInput<A>;

  /** Extract the assistant message and optional metadata from the flow output. */
  fromOutput: (output: FlowOutput<A>) => {
    message: ChatMessage;
    sessionId?: string;
  };

  /** Extract partial content from a stream chunk (for live token display). */
  fromChunk?: (chunk: FlowStream<A>) => string | Part[];
}
```

Default adapters are provided for common patterns, and developers can supply their own:

```typescript
// Uses the default adapter (assumes messages-in, response-out pattern)
const chat = useChat<ChatFlow>(client, '/chat');

// Custom adapter for a non-standard flow shape
const chat = useChat(client, '/myCustomChat', {
  adapter: {
    toInput: (messages) => ({
      conversation: messages.map(m => `${m.role}: ${m.content[0].text}`).join('\n'),
    }),
    fromOutput: (output) => ({
      message: { id: crypto.randomUUID(), role: 'model', content: [{ text: output.reply }], createdAt: new Date() },
    }),
    fromChunk: (chunk) => chunk.token,
  },
});
```

This means `useChat` works with any flow shape -- it's not tied to a specific chat protocol. The adapter bridges the gap between the generic flow model and the chat UI pattern.

#### `useChat` API

```typescript
function useChat<A extends Action = Action>(
  client: GenkitClient,
  path: string,
  options?: UseChatOptions<A>,
): UseChatReturn;

interface ChatMessage {
  id: string;
  role: 'user' | 'model' | 'system' | 'tool';
  content: Part[];
  metadata?: Record<string, any>;
  createdAt: Date;
}

interface UseChatOptions<A extends Action = Action> {
  /** Adapter to map between chat state and flow input/output. */
  adapter?: ChatAdapter<A>;

  /** Initial messages to populate the conversation. */
  initialMessages?: ChatMessage[];

  /** Session ID for server-managed history (Approach B). */
  sessionId?: string;

  /** Called when each stream chunk arrives. */
  onChunk?: (chunk: FlowStream<A>) => void;

  /** Called when a complete response is received. */
  onFinish?: (message: ChatMessage) => void;

  /** Called when an error occurs. */
  onError?: (error: Error) => void;

  /** Additional headers for chat requests. */
  headers?: Record<string, string>;
}

interface UseChatReturn {
  /** Full conversation history (user + assistant messages). */
  messages: ChatMessage[];

  /** Send a user message and trigger assistant response. */
  sendMessage: (content: string | Part[]) => void;

  /** Whether the assistant is currently streaming a response. */
  isStreaming: boolean;

  /** The most recent error, if any. */
  error: Error | null;

  /** Stop the current streaming response. */
  stop: () => void;

  /** Re-generate the last assistant response. */
  regenerate: () => void;

  /** Override the message history. */
  setMessages: (messages: ChatMessage[]) => void;

  /** The server-assigned session ID (if using server-managed history). */
  sessionId: string | null;

  /** Controlled input value (convenience for forms). */
  input: string;

  /** Set the input value. */
  setInput: (value: string) => void;
}
```

#### Example: Full Chat UI

```typescript
import { useChat } from 'genkit/client/react';
import type { ChatFlow } from '../server/flows';

function ChatUI() {
  const {
    messages,
    sendMessage,
    isStreaming,
    error,
    stop,
    regenerate,
    input,
    setInput,
  } = useChat<ChatFlow>(client, '/chat');

  return (
    <div className="chat">
      <div className="messages">
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.role}`}>
            {msg.content.map((part, i) =>
              part.text ? <span key={i}>{part.text}</span> : null
            )}
          </div>
        ))}
      </div>

      <form onSubmit={(e) => {
        e.preventDefault();
        sendMessage(input);
        setInput('');
      }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message..."
          disabled={isStreaming}
        />
        {isStreaming
          ? <button type="button" onClick={stop}>Stop</button>
          : <button type="submit">Send</button>
        }
      </form>

      {error && <p className="error">{error.message}</p>}
    </div>
  );
}
```

---

### 8. Error Handling

#### `GenkitClientError`

A structured error class that preserves server error details:

```typescript
class GenkitClientError extends Error {
  /** HTTP status code, if available. */
  readonly status?: number;

  /** Genkit status name (e.g., 'NOT_FOUND', 'PERMISSION_DENIED'). */
  readonly code?: string;

  /** Additional error details from the server. */
  readonly details?: unknown;

  /** Whether this error is retryable. */
  readonly retryable: boolean;
}
```

The client parses both JSON error responses and SSE error envelopes into `GenkitClientError`:

```typescript
try {
  await client.runFlow('/myFlow', input);
} catch (e) {
  if (e instanceof GenkitClientError) {
    console.log(e.code);    // 'PERMISSION_DENIED'
    console.log(e.status);  // 403
    console.log(e.details); // { reason: '...' }
  }
}
```

In hooks, the error is exposed reactively:

```typescript
const { error } = useFlow<MyFlow>(client, '/myFlow');
// error is GenkitClientError | null
```

---

## Package Structure

```
genkit/
  client                  -- Core: createGenkitClient, type utilities, retry logic
  client/react            -- React hooks: useFlow, useChat
  client/vue              -- Vue composables: useFlow, useChat
```

All sub-paths are browser-safe with zero Node.js dependencies. The core `genkit/client` module uses only `fetch`, `TextDecoder`, `AbortController`, and `ReadableStream` -- all available in browsers and modern runtimes.

### Dependency Graph

```
genkit/client/react  ──> genkit/client  (core)
genkit/client/vue    ──> genkit/client  (core)
genkit/client        ──> (no dependencies, browser-safe)
```

React hooks have a peer dependency on `react >= 18`. Vue composables have a peer dependency on `vue >= 3.3`.

### Package Exports (in `genkit/package.json`)

```json
{
  "exports": {
    "./client": {
      "types": "./lib/client/index.d.ts",
      "import": "./lib/client/index.mjs",
      "require": "./lib/client/index.js"
    },
    "./client/react": {
      "types": "./lib/client/react/index.d.ts",
      "import": "./lib/client/react/index.mjs",
      "require": "./lib/client/react/index.js"
    },
    "./client/vue": {
      "types": "./lib/client/vue/index.d.ts",
      "import": "./lib/client/vue/index.mjs",
      "require": "./lib/client/vue/index.js"
    }
  },
  "peerDependencies": {
    "react": ">=18",
    "vue": ">=3.3"
  },
  "peerDependenciesMeta": {
    "react": {
      "optional": true
    },
    "vue": {
      "optional": true
    }
  }
}
```

React and Vue are optional peer dependencies -- only required if you import from `genkit/client/react` or `genkit/client/vue` respectively. The core `genkit/client` entry point has no framework dependencies at all.

---

## Architecture

### Layer Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   Framework Hooks                        │
│  ┌──────────────────────┐  ┌──────────────────────────┐ │
│  │   genkit/client/react │  │   genkit/client/vue      │ │
│  │   - useFlow           │  │   - useFlow              │ │
│  │   - useChat           │  │   - useChat              │ │
│  └──────────┬───────────┘  └──────────┬───────────────┘ │
│             │                         │                  │
│             ▼                         ▼                  │
│  ┌──────────────────────────────────────────────────┐   │
│  │              genkit/client (Core)                 │   │
│  │  - createGenkitClient()                           │   │
│  │  - GenkitClient { runFlow, streamFlow, flow }     │   │
│  │  - RetryEngine                                    │   │
│  │  - GenkitClientError                              │   │
│  │  - FlowInput<A>, FlowOutput<A>, FlowStream<A>    │   │
│  └──────────────────────┬───────────────────────────┘   │
│                         │                                │
│                         ▼                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │              HTTP / SSE Protocol                   │   │
│  │  POST { data: input }                              │   │
│  │  → { result: output }                              │   │
│  │  → data: {"message": chunk}\n\n                    │   │
│  │  → data: {"result": output}\n\n                    │   │
│  └──────────────────────┬───────────────────────────┘   │
│                         │                                │
└─────────────────────────┼────────────────────────────────┘
                          │ network
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   Server Side                            │
│  @genkit-ai/web  |  @genkit-ai/express  |  @genkit-ai/next │
│                         │                                │
│                         ▼                                │
│               ai.defineFlow(config, fn)                  │
└─────────────────────────────────────────────────────────┘
```

### Data Flow: `useFlow` Streaming

```
1. Component calls stream(input)
     │
2. useFlow sets isLoading=true, chunks=[], data=null
     │
3. client.streamFlow(path, input)
     │
4. POST request with Accept: text/event-stream
     │
5. Server begins streaming SSE chunks
     │
6. ┌─ For each {"message": chunk}:
   │    - Append to chunks[]
   │    - Call onChunk callback
   │    - React re-renders with new chunks
   │
7. └─ On {"result": output}:
        - Set data = output
        - Set isLoading = false
        - Call onSuccess callback
        - React re-renders with final data

On error at any point:
  - Set error = GenkitClientError
  - Set isLoading = false
  - If retryable && attempts < maxRetries:
      wait(backoff) → goto step 3
```

## Future Extensions

These are explicitly **out of scope** for the initial release but inform the design to ensure extensibility.

### Tool Call Visibility and Client-Side Tool Execution

In Genkit today, tool calls are fully resolved **server-side**. When `generate()` encounters a tool call, the server executes the tool, feeds the result back into the model, and continues -- all within a single request. The client only sees the final output (or stream chunks of the model's text). Individual tool calls are invisible to the client.

There are two separate features worth considering here:

#### Tool Call Observability

Even without client-side execution, it's useful for the client to **see** that tools were called -- for rendering tool call cards, showing progress ("Searching for flights..."), or debugging. This could be achieved by extending the SSE stream chunk protocol to include tool call events:

```
data: {"toolCall": {"name": "searchFlights", "input": {"from": "SFO", "to": "JFK"}}}\n\n
data: {"toolResult": {"name": "searchFlights", "output": [...]}}\n\n
data: {"message": "I found 3 flights..."}\n\n
data: {"result": {...}}\n\n
```

The client hook would expose these as observable events:

```typescript
useChat(client, '/agentFlow', {
  onToolCall: ({ name, input }) => {
    showLoadingIndicator(`Running ${name}...`);
  },
  onToolResult: ({ name, output }) => {
    renderToolResultCard(name, output);
  },
});
```

This is read-only observation -- the tools still execute on the server.

#### Client-Side Tool Execution

A more ambitious extension where certain tools are **defined on the client** and executed in the browser. The server would pause generation, stream a tool call request to the client, wait for the client to respond with a result, then continue. This is how [Vercel AI SDK's `onToolCall`](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-tool-usage) works.

Use cases for client-side tools are things that only the client can do: showing a confirmation dialog, accessing browser APIs (geolocation, camera), reading from local storage, or rendering interactive UI.

This would require a bidirectional protocol (likely WebSocket, or a callback POST mechanism) and significant changes to how the server plugins handle streaming. It should be designed alongside Genkit's existing interrupt/resume mechanism, which already supports pausing a flow and resuming with external input -- but currently only at the flow level, not mid-generation.

### UI Components

Pre-built, styled components for common AI UI patterns:

```typescript
import { ChatWindow, MessageBubble, StreamingText, ToolCallCard } from '@genkit-ai/ui/react';
// or
import { ChatWindow, MessageBubble, StreamingText, ToolCallCard } from '@genkit-ai/ui/vue';
```

Possibly with ShadCN-style copy-paste components for maximum customizability.

### `useAgent` Hook

Inspired by [Cloudflare Agents SDK](https://developers.cloudflare.com/agents/):

```typescript
const agent = useAgent(client, '/myAgent', {
  onStateUpdate: (state) => { /* ... */ },
  onMessage: (message) => { /* ... */ },
});
```

Would require WebSocket transport for bidirectional real-time communication.

### Schema Discovery Endpoint

A server-side `/schema` endpoint that returns JSON Schema for all registered flows:

```json
{
  "flows": {
    "myFlow": {
      "inputSchema": { "type": "string" },
      "outputSchema": { "type": "object", "properties": { "greeting": { "type": "string" } } },
      "streamSchema": { "type": "string" }
    }
  }
}
```

Could enable codegen-free type safety or runtime validation on the client.

### SWR / TanStack Query Integration

Adapters for popular data-fetching libraries:

```typescript
import { genkitFetcher } from 'genkit/client/swr';

const { data, error } = useSWR('/myFlow', genkitFetcher(client));
```

---

## Open Questions

1. **WebSocket transport**: Should the client factory support WebSocket transport in addition to HTTP/SSE? This would be valuable for `useAgent` and real-time chat patterns but adds complexity. Should it be a separate transport layer that can be plugged in?

2. **Deprecation timeline**: Should `genkit/client` replace `genkit/beta/client` in the next minor release, or should there be a longer deprecation period?

3. **`onToolCall` protocol**: Should client-side tool call hooks reuse Genkit's existing interrupt/resume protocol, or define a new streaming sub-protocol? The interrupt protocol currently works at the flow level; tool call hooks would need chunk-level granularity.

4. **Lightweight flow contract type**: The type inference approach requires `import type { MyFlow } from '../server/flows'`, which imports the `Action` type from `genkit`. While `import type` has zero runtime cost, it still couples the client's type-checking to the server package. Should there be a lighter-weight "flow contract" type that avoids this coupling? For example:

   ```typescript
   // Server
   export type MyFlowContract = FlowContract<string, { greeting: string }, string>;

   // Client -- no dependency on genkit's Action type
   import type { MyFlowContract } from '../shared/contracts';
   client.runFlow<MyFlowContract>('/myFlow', 'hello');
   ```

5. **Chat message format**: Should `useChat` use Genkit's `MessageData` format (with `Part[]` content) or a simplified format more suited to client-side rendering? The `Part[]` structure supports multimodal content (text, media, tool calls) but is more complex than a plain `{ role, text }` shape.

6. **React Server Components**: Should the hooks work in RSC environments (Next.js App Router)? `useFlow` and `useChat` are inherently client-side (`"use client"`), but should there be server-side counterparts for RSC data fetching?

---

## Prior Art

| Library | Pattern | Relevant Feature |
|---------|---------|-----------------|
| [Vercel AI SDK](https://ai-sdk.dev/) | `useChat`, `useCompletion`, `useObject` | React hooks for streaming AI, `onToolCall`, transport abstraction |
| [Better Auth](https://www.better-auth.com/) | `createAuthClient()` | Client factory with base URL, framework-specific exports |
| [Cloudflare Agents SDK](https://developers.cloudflare.com/agents/) | `useAgent`, `Agent` class | Persistent agent with state sync, WebSocket transport |
| [tRPC](https://trpc.io/) | End-to-end type safety | Type inference from server procedures without codegen |
| [TanStack Query](https://tanstack.com/query) | `useQuery`, `useMutation` | Loading/error state, retry, caching, framework adapters |
| [Firebase JS SDK](https://firebase.google.com/docs/functions/callable) | `httpsCallable()` | Client factory for callable functions with streaming |
