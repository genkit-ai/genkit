# agent-dirs demo

Demo of the experimental [`@genkit-ai/agent-dirs`](../../plugins/agent-dirs)
plugin: agents defined entirely as directories under `agents/`, showing both
delegation styles.

- **support** - customer support agent: tools (`lookupOrder`, `checkStock`),
  a skill (`refund-policy`), OKF knowledge (`carriers`, `returns-address`).
- **shipping** - top-level specialist (`trackParcel`); support delegates to
  it via `delegates:` frontmatter - the shared-sibling style, and shipping
  is also served directly.
- **support/subagents/refunds** - nested subagent (`processRefund`,
  approval-gated); registered as `support.refunds` and auto-wired as a
  delegate just by being nested - the private-specialist style, reachable
  only through support, not served over HTTP.

## Prerequisites

```sh
pnpm i
pnpm -F @genkit-ai/agent-dirs build   # once; the CLI runs from lib/
gcloud auth application-default login # Vertex AI via ADC
export GCLOUD_PROJECT=<your-project>
```

## Run it

Three ways, same agents:

```sh
# 1. Zero host code - the folder is the only input
#    (--watch restarts on any agent-file edit):
pnpm server:zero --watch

# 2. Host-code path (src/index.ts) - same thing, written out; the template
#    for picking your own provider or embedding in an existing app:
pnpm server

# 3. Dev UI - chat with the agents interactively:
pnpm genkit:dev
```

`genkit:dev` starts the app under the Genkit CLI and prints a Dev UI URL
(default http://localhost:4000). Open it, pick an agent under **Agents**
(all three appear there, including `support.refunds`), and chat. Ask
support "where is order 1234?" to watch it call `lookupOrder`, or "refund
order 1234, it arrived broken" to watch it delegate to `support.refunds`
and pause on the `processRefund` approval. Every turn's trace (model calls,
tool calls, delegation) is inspectable under **Traces**.

## Talk to it

```sh
curl -N http://localhost:8080/api/support \
  -H 'Content-Type: application/json' \
  -d '{"data": {"message": {"role": "user", "content": [{"text": "where is order 1234?"}]}}}'
```

Or from code, with `remoteAgent`:

```ts
import { remoteAgent } from 'genkit/beta/client';
const support = remoteAgent({ url: 'http://localhost:8080/api/support' });
const chat = support.chat();
const res = await chat.send('where is order 1234?');
```

Each agent also exposes `POST /api/<name>/getSnapshot` and
`POST /api/<name>/abort`. Session snapshots persist under
`./.genkit/agent-snapshots/` (gitignored).
