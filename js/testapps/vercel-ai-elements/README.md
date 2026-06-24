# Genkit × Vercel AI Elements

A [Next.js](https://nextjs.org/) demo app showing how to build a rich,
streaming chat UI with [AI Elements](https://elements.ai-sdk.dev/) on top of
[Genkit](https://github.com/genkit-ai/genkit) agents — powered by the
[`@genkit-ai/vercel-ai`](../../plugins/vercel-ai) `GenkitChatTransport`.

It demonstrates everything the transport supports end-to-end:

- **Text streaming** — token-by-token responses rendered with `streamdown`.
- **Multi-turn conversations** — automatic snapshot-based session continuity.
- **Tool calls** — the **Weather** agent calls a `getWeather` tool and renders
  the tool input/output inline.
- **Interrupts (human-in-the-loop)** — the **Banking** agent uses a
  `userApproval` interrupt to pause and ask for confirmation before running a
  `transferMoney` tool.
- **Multi-agent selection** — switch between the Weather and Banking agents.

## Architecture

```
Browser (useChat + GenkitChatTransport)
        │  POST  (Genkit streamFlow protocol, NDJSON)
        ▼
Next.js Route Handler  (appRoute from @genkit-ai/next)
        │
        ▼
Genkit Agent  (defineAgent, tools, interrupts, InMemorySessionStore)
        │
        ▼
Gemini  (@genkit-ai/google-genai)
```

| Path | Description |
| ---- | ----------- |
| `lib/genkit.ts` | Genkit instance configured with the Google GenAI plugin. |
| `lib/agents.ts` | `weatherAgent` (tool calling) and `bankingAgent` (interrupt). |
| `app/api/chat/weather/route.ts` | Exposes `weatherAgent` via `appRoute`. |
| `app/api/chat/banking/route.ts` | Exposes `bankingAgent` via `appRoute`. |
| `app/page.tsx` | The chat UI: agent selector, message rendering, interrupt approval. |
| `components/ai-elements/*` | AI Elements components (from the shadcn registry). |

## Prerequisites

- Node.js `>=20`
- A Gemini API key — get one from
  [Google AI Studio](https://aistudio.google.com/apikey).

## Setup

From the repository root, build the workspace packages this app depends on
(`genkit`, `@genkit-ai/next`, `@genkit-ai/vercel-ai`, `@genkit-ai/google-genai`):

```bash
cd js
pnpm install
pnpm build
```

Then configure your API key:

```bash
cd testapps/vercel-ai-elements
cp .env.local.example .env.local
# edit .env.local and set GEMINI_API_KEY
```

## Run

```bash
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000).

Try it out:

- **Weather agent:** ask _"What's the weather in Tokyo?"_ — watch the
  `getWeather` tool call and its result render inline.
- **Banking agent:** ask _"Transfer $500 to account 12345"_ — the agent pauses
  with a `userApproval` interrupt; approve or reject it, and the conversation
  resumes from the server-side snapshot.

## How the transport is wired

The client creates a single `GenkitChatTransport` pointed at the selected
agent's route and hands it to `useChat`:

```tsx
import { useChat } from '@ai-sdk/react';
import { GenkitChatTransport } from '@genkit-ai/vercel-ai/client';

const transport = useMemo(
  () => new GenkitChatTransport({ url: `/api/chat/${agent}` }),
  [agent]
);
const { messages, status, sendMessage } = useChat({ transport });
```

The server side is just a Genkit agent exposed with `appRoute`:

```ts
import { appRoute } from '@genkit-ai/next';
import { weatherAgent } from '@/lib/agents';

export const POST = appRoute(weatherAgent);
```

No `ChatSessionStore` is needed on the client — the transport tracks the
`chatId → snapshotId` mapping for you so multi-turn history and interrupt
resume "just work".

## Learn more

- [`@genkit-ai/vercel-ai` plugin README](../../plugins/vercel-ai/README.md)
- [AI Elements](https://elements.ai-sdk.dev/)
- [Genkit documentation](https://genkit.dev/)
