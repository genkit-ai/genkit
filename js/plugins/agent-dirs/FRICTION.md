# Friction log: building agent-dirs on the beta agents API

Everything non-obvious we hit while building this plugin and its demo, as
input for API/docs work. Ordered by impact. "Fix" lines are suggestions, not
demands; several have candidate implementations in this PR.

## 1. No way to serve agents over HTTP without hand-rolling routes

`startFlowServer` (`@genkit-ai/express`) serves flows only: explicit list, no
registry discovery, and none of an agent's companion actions
(`getSnapshotDataAction`, `abortAgentAction`) - so resume inspection and
abort don't work over HTTP. Result: `testapps/agents/src/index.ts` hand-rolls
an `exposeAgent()` helper, and this package's demo independently reinvented
the same ~15 lines before we extracted `serveAgents()` (`src/server.ts`).

**Fix:** first-party `startAgentServer()` / `agents:` option in
`@genkit-ai/express`. `serveAgents()` here is a candidate implementation.

## 2. Agents registered by plugins are hard to reach

A plugin that calls `defineAgent` has no way to hand the `Agent` object to
user code. The only route back is `registry.lookupAction('/agent/<name>')` -
an undocumented key convention plus an unchecked cast (`Agent` has no runtime
brand; we duck-type on `.chat`). Same for enumeration (`listAgents` filters
`listActions()` keys by prefix).

**Fix:** `ai.agent(name)` instance method mirroring `ai.prompt(name)`, plus a
typed way to enumerate registered agents.

## 3. Slash-namespaced tool names are silently rewritten for the model

Tools named `<ns>/<name>` (the MCP convention, used here for per-agent
namespacing) are exposed to the model by short name only -
`toMap`/`resolveToolRequest` key on the segment after the last `/`
(`ai/src/generate/resolve-tool-requests.ts`). Undocumented; cost us a live
NOT_FOUND loop when a scripted model called the full name, and any prompt
text that mentions a tool by full name is silently wrong.

**Fix:** document the short-name contract where `defineTool` is documented.

## 4. `@genkit-ai/middleware` is invisible

We fully reimplemented progressive-disclosure skills (index + load tool)
before discovering `skills` middleware already ships first-party - along
with `agents` (delegation), `toolApproval` (interrupt-gated tools),
`artifacts`, `filesystem`, `retry`, `fallback`. None are referenced from the
docs surfaces a plugin author reads first. The package is arguably the most
Eve-shaped capability layer genkit has, and it's undiscoverable.

**Fix:** document the middleware catalog prominently; reference it from the
agents API docs.

## 5. `vertexAI()` cannot infer a project from gcloud ADC alone

With valid ADC and `gcloud config set project`, plugin init still fails:
"Unable to determine client options. Please set either apiKey or projectId
and location". Works only with `GCLOUD_PROJECT` env or explicit `projectId`.
The error suggests an API key first, which is the wrong path for the GCP-
native audience, and doesn't mention the env var it actually reads.

**Fix:** fall back to the gcloud config project like the CLI does, and name
`GCLOUD_PROJECT` in the error.

## 6. Deploy story: no environment-aware session store default

`FileSessionStore` defaults break on Cloud Run (ephemeral disk,
multi-instance); `FirestoreSessionStore` exists in the firebase plugin but
must be discovered and wired manually. One env-aware default (local disk in
dev, Firestore when `K_SERVICE` is set) plus `serveAgents()` would make
agent deployment literally `gcloud run deploy --source .` with zero keys
(ADC covers Vertex).

**Fix:** environment-aware store default, or at minimum a deploy guide for
the agents API.

## 7. Durability is turn-granular; the gaps compound on serverless

Snapshots persist per turn. A crash mid-turn (long tool chains) loses the
whole turn and re-executes side effects on retry - the
`TurnContext.snapshotId` idempotency hook exists but is convention only.
No intra-turn step journal means long turns can't span serverless request
timeouts; detached turns die with the instance (heartbeat -> `expired`).
Snapshots also carry no code version, so resuming after a deploy replays
against changed agent code unguarded.

**Fix (long-term):** opt-in intra-turn step journaling (middleware `tool`/
`model` hooks are the natural seam), durable resume triggers, version stamp
on snapshots.

## 8. `.prompt` parsing has no public API

Reading an `agent.prompt` file requires `ai.registry.dotprompt.parse()` -
reaching into the registry's internal Dotprompt instance. Fine in-tree,
awkward for external plugins wanting dotprompt-compatible files. Related:
unknown frontmatter keys surface only via the undocumented `parsed.raw`,
which this plugin relies on for `delegates`/`requireApproval`.

**Fix:** export a parse/loadPromptFolder-style helper; document `raw` (or a
blessed extension-keys mechanism).

## 9. Minor

- `mockModel` defaults to "does not support tools" warnings in agent tests;
  needs `info: { supports: { tools: true } }` boilerplate every time.
- Beta agents API (snapshot statuses, `use` on prompts/agents, detach,
  branching) is currently learnable only by reading source; this plugin was
  built entirely from `ai/src/agent.ts` comments - which are good, but
  they're the only docs.
