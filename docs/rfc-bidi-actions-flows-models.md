# RFC: Bidirectional Actions, Flows, and Models

## Summary

Introduces bidirectional streaming capabilities to Genkit Actions, Flows, and Models. This allows these primitives to accept a continuous input stream and an initialization payload, in addition to producing an output stream.

## Motivation

Core primitives need to support advanced interaction patterns beyond simple request/response:
- **Real-time interactions**: Voice or text chat where input and output happen simultaneously (e.g., Gemini Live, OpenAI Realtime).
- **Session management**: Long-running flows that maintain state across inputs.
- **Tool interruptions**: Human-in-the-loop scenarios where execution pauses for input.

## Design

### 1. Bidi Action Primitive

The `Action` interface is extended to support bidirectional communication. This is non-breaking; existing actions remain compatible.

- **`inputStream`**: An `AsyncIterable` of input chunks.
- **`init`**: An optional payload for setup context available before streaming begins.

#### Definition (`defineBidiAction`)

```typescript
import { defineBidiAction } from '@genkit-ai/core';
import { z } from 'zod';

export const bidiChat = defineBidiAction(
  {
    name: 'bidiChat',
    actionType: 'custom',
    inputSchema: z.string(),   // Schema for items in inputStream
    outputSchema: z.string(),  // Schema for the final return value
    streamSchema: z.string(),  // Schema for items in output stream
    initSchema: z.object({     // Schema for initialization data
      userId: z.string() 
    }), 
  },
  async function* ({ inputStream, init }) {
    console.log(`Starting chat for ${init.userId}`);
    
    for await (const chunk of inputStream) {
      yield `Echo: ${chunk}`;
    }
    return 'Conversation ended';
  }
);
```

### 2. Bidi Flows

`defineBidiFlow` exposes this capability in the Genkit public API, wrapping the action with observability and tracing.

#### Type Signature
`Flow<Input, Output, Stream, Init>`

#### Usage

```typescript
import { genkit, z } from 'genkit';

const ai = genkit({ ... });

export const flow = ai.defineBidiFlow(
  {
    name: 'chatFlow',
    inputSchema: z.string(),
    streamSchema: z.string(),
    initSchema: z.object({ topic: z.string() }),
  },
  async function* ({ inputStream, init }) {
    yield `Welcome to ${init.topic}`;
    
    for await (const msg of inputStream) {
      if (msg === 'bye') break;
      yield `You said: ${msg}`;
    }
  }
);
```

### 3. Bidi Models

`defineBidiModel` creates a specialized bidi action for LLMs, aimed at real-time model APIs.

#### The Role of `init`
For real-time sessions, the connection to the model API often requires configuration (temperature, system prompt, tools) to be established *before* the first user message is received. The `init` payload fulfills this requirement, separating session configuration from the conversation stream.

- **`init`**: `GenerateRequest` (contains config, tools, system prompt).
- **`inputStream`**: Stream of `GenerateRequest` (contains user messages/turns).
- **`stream`**: Stream of `GenerateResponseChunk`.

#### Definition

```typescript
export const myRealtimeModel = defineBidiModel(
  { name: 'myRealtimeModel' },
  async function* ({ inputStream, init }) {
    // 1. Establish session using configuration from init
    const session = await upstreamApi.connect({
      model: 'my-model',
      config: init?.config,
      systemPrompt: init?.messages?.find(m => m.role === 'system'),
      tools: init?.tools,
    });

    // 2. Handle conversation stream
    for await (const request of inputStream) {
      // Send new user input to the upstream session
      session.send(request.messages);

      // Yield responses from the upstream session
      for await (const response of session.receive()) {
         yield { content: [{ text: response.text }] };
      }
    }
    
    // 3. Return final result (usage stats, etc.)
    return {
      usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
      custom: session.getFinalMetadata(),
    };
  }
);
```

#### Usage (`generateBidi`)

`generateBidi` is the high-level API for interacting with bidi models. It is a subset of the standard `generate` API, currently supporting tool calling.

```typescript
const session = await ai.generateBidi({
  model: myRealtimeModel,
  config: { temperature: 0.7 }, // Passed via init
  system: 'You are a helpful assistant', // Passed via init
});

// The session is established. Now we can stream inputs.
session.send('Hello!');

// Listen for responses (simultaneously)
for await (const chunk of session.stream) {
  console.log(chunk.content);
}
```

### 4. Execution API (`streamBidi`)

Actions and Flows expose a `streamBidi` method that returns a `BidiStreamingResponse`.

```typescript
interface BidiStreamingResponse<O, S, I> {
  stream: AsyncGenerator<S>; // Output stream
  output: Promise<O>;        // Final result
  send(chunk: I): void;      // Push input
  close(): void;             // End input stream
}
```

#### Push Style (Manual Send)

```typescript
const session = flow.streamBidi(undefined, { 
  init: { topic: 'Support' } 
});

// Send inputs
session.send('Hello');
session.send('Help');
session.close();

// Consume output
for await (const chunk of session.stream) {
  console.log(chunk);
}
```

#### Pull Style (Generator)

```typescript
async function* inputSource() {
  yield 'Hello';
  yield 'World';
}

const session = flow.streamBidi(inputSource(), {
  init: { topic: 'Greeting' }
});

for await (const chunk of session.stream) {
  console.log(chunk);
}
```

## Integration with Reflection API

These features align with **Reflection API V2**, which uses WebSockets to support bidirectional streaming between the Runtime and the CLI/Manager.

- `runAction` now supports an `input` stream.
- `streamChunk` notifications are bidirectional (Manager <-> Runtime).
