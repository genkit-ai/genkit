# agent-dirs demo

Demo of the experimental [`@genkit-ai/agent-dirs`](../../plugins/agent-dirs)
plugin: two agents defined entirely as directories under `agents/`.

- **support** - customer support agent: tools (`lookupOrder`, `checkStock`),
  a skill (`refund-policy`), OKF knowledge (`carriers`, `returns-address`),
  and delegation to `shipping`.
- **shipping** - shipping specialist with one tool (`trackParcel`).

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
