# RFC: Agent Primitive

## Summary

Introduces `defineAgent`, a high-level abstraction built on top of Bidi Actions designed to simplify the creation of stateful, multi-turn agents. It unifies state management, allowing both client-side state handling and server-side persistence via pluggable stores.

`defineAgent` would replace the current Chat API as there is significant overlap and Agent primitive is more flexible.

## Motivation

Building agents often involves repetitive boilerplate:
1.  **State Management**: Loading conversation history, updating it with new messages, and persisting it.
2.  **Session Handling**: Managing session IDs and context.
3.  **Multi-turn Loops**: Processing a stream of user inputs and generating responses.
4.  **Interrupts**: Pausing execution for human feedback or tool approval.

The `Agent` primitive encapsulates these patterns, providing a standard interface for building chatbots and autonomous agents that can run efficiently in both serverless (stateless) and stateful environments.

## Design

### 1. `defineAgent`

The `defineAgent` function wraps a Bidi Flow, adding built-in support for initialization, state loading/saving, and standardized input/output schemas. Unlike high-level configuration-based agents, `defineAgent` gives you full control over the execution loop.

```typescript
export const myAgent = ai.defineAgent(
  {
    name: 'myAgent',
    store: myPostgresStore, // Optional: enables server-side state
  },
  async function* ({ inputStream, init, sendChunk }) {
    // Manually manage the conversation loop
  }
);
```

### 2. State Management Modes

The Agent abstraction supports two primary modes of operation, determined by the presence of a `store`.

#### A. Client-Managed State (Stateless Server)

In this mode, the server does not persist state. The client is responsible for maintaining the conversation history and passing it to the agent upon each invocation.

-   **Init**: Client sends `messages`, `artifacts`, etc.
-   **Execution**: Agent processes input, generates response.
-   **Output**: Agent returns the *updated* state (new history).
-   **Next Turn**: Client sends the updated history back in `init`.

**Pros**: Infinite scalability, no database required, REST-friendly.

#### B. Server-Managed State (Stateful Server)

In this mode, a `SessionStore` is configured. The server persists the state.

-   **Init**: Client sends `sessionId`.
-   **Execution**:
    1.  Framework loads state from `store` using `sessionId` (populating `init`).
    2.  Agent processes input, generates response.
    3.  Framework saves updated state to `store`.
-   **Output**: Agent returns the result.
-   **Next Turn**: Client sends `sessionId` again.

**Pros**: Thinner clients, secure context storage, background persistence.

### 3. Usage

#### Basic Example (Manual Loop)

This example demonstrates the core pattern: receiving input, calling `ai.generate`, and managing the messages array.

```typescript
import { genkit } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [googleAI()],
});

export const myAgent = ai.defineAgent(
  { name: 'myAgent' },
  async function* ({ sendChunk, inputStream, init }) {
    // 1. Initialize state from init payload (or empty)
    let messages = init?.messages ?? [];

    // 2. Process the input stream
    for await (const input of inputStream) {
      // 3. Generate response using a model
      const response = await ai.generate({
        messages: [...messages, input],
        model: googleAI.model('gemini-2.5-flash'),
        onChunk: (chunk) => sendChunk({ sessionId: init?.sessionId, chunk }),
      });
      
      messages = response.messages;

      // 4. Handle interrupts (e.g. tool calls)
      if (response.interrupts.length > 0) {
        return {
          sessionId: init?.sessionId,
          messages,
        };
      }
    }

    // 5. Return final state
    return {
      sessionId: init?.sessionId,
      messages,
      artifacts: [{ name: 'report', parts: [] }],
    };
  }
);
```

#### Example with Store (Server-Side Persistence)

Adding a `store` automatically handles state persistence. The implementation logic remains largely the same, but the state is preserved across network calls without the client sending it back.

```typescript
export const persistentAgent = ai.defineAgent(
  { 
    name: 'persistentAgent',
    store: postgresSessionStore({ connectionString: '...' })
  },
  async function* ({ sendChunk, inputStream, init }) {
    // init.messages is automatically populated from the store if sessionId exists
    let messages = init?.messages ?? [];

    for await (const input of inputStream) {
      const response = await ai.generate({
        messages: [...messages, input],
        model: googleAI.model('gemini-2.5-flash'),
        onChunk: (chunk) => sendChunk({ sessionId: init?.sessionId, chunk }),
      });
      messages = response.messages;

      // ... handling interrupts
    }

    // State is automatically saved to the store upon return
    return {
      sessionId: init?.sessionId,
      messages,
    };
  }
);
```

#### Example: Streaming State Updates

You can stream intermediate state updates to the client using `sendChunk`. This is useful for providing progress on long-running tasks or tool executions.

```typescript
export const toolAgent = ai.defineAgent(
  { name: 'toolAgent' },
  async function* ({ sendChunk, inputStream, init, session }) {
    for await (const input of inputStream) {
      // 1. Notify client that we are starting a tool
      sendChunk({ 
        statusUpdate: { status: 'executing_tool', tool: 'weather' } 
      });

      // 2. Execute tool (simulated)
      await new Promise(r => setTimeout(r, 1000));

      // 3. Notify client of completion
      sendChunk({ 
        statusUpdate: { status: 'tool_complete', tool: 'weather' } 
      });
      
      // ... continue generation
      await session.createSnapshot();
    }
  }
);
```

### 4. Schemas

The Agent primitive relies on strict Zod schemas to ensure type safety and compatibility.

#### Init Schema (`AgentInitSchema`)
```typescript

const AgentSnapshotSchema = z.object({
  // oneof {
  snapshotId: z.string().optional(),
  // {
  messages: z.array(MessageSchema).optional(),
  state: z.any().optional(),
  artifacts: z.array(AgentArtifactSchema).optional(),
  // }
});

const AgentInitSchema = z.object({
  snapshot: AgentSnapshotSchema.optional(),
});
```

#### Stream Schema (`AgentStreamSchema`)
```typescript
const AgentStreamSchema = z.object({
  chunk: GenerateResponseChunkSchema.optional(), // Token generation
  statusUpdate: z.any().optional(),
  artifact: AgentArtifactSchema.optional(), // New artifacts
  snapshotCreated: z.string().optional(),
});
```

#### Output Schema (`AgentResponseSchema`)
```typescript
const AgentResponseSchema = z.object({
  snapshot: AgentSnapshotSchema,
});
```

#### Artifact Schema (`AgentArtifactSchema`)
```typescript
const AgentArtifactSchema = z.object({
  name: z.string().optional(),
  parts: z.array(PartSchema), // Media, text, etc.
  metadata: z.record(z.any()).optional(),
});
```
