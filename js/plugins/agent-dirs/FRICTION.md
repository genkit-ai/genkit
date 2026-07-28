# Friction log: building agent-dirs on the beta agents API

Everything non-obvious we hit while building this plugin and its demo, as
input for API/docs work. Ordered by impact. "Fix" lines are suggestions, not
demands; several have candidate implementations in this PR. Claims were
adversarially fact-checked against source; evidence refs are to this repo.

## 1. No first-party way to serve agents over HTTP

The transport layer is fine - `expressHandler` fully supports agent actions,
including `init` forwarding (`js/plugins/express/src/index.ts:82,149-153`).
What's missing is the aggregation: `startFlowServer` is flows-only (explicit
list, no registry discovery, `express/src/index.ts:379-403`), fastify/next
are per-action handlers, and nothing wires an agent's companion actions
(`getSnapshotDataAction`, `abortAgentAction`).

That this is a gap rather than a design choice is visible in the client: the
first-party `remoteAgent` hard-codes a server URL contract - `${url}`,
`${url}/getSnapshot`, `${url}/abort` (`js/genkit/src/client/agent.ts:53-91`)
- that no first-party server implements. Both the upstream agents testapp
(`testapps/agents/src/index.ts:126-141`, `exposeAgent()`) and this package's
demo independently hand-rolled the same routes before we extracted
`serveAgents()` (`src/server.ts`). Likely rationale is release staging
(agents are beta, `@genkit-ai/express` is stable), not opinionation.

Related trap: durable stream reconnects need `X-Genkit-Stream-Id` in
`Access-Control-Expose-Headers`; the testapp hand-rolls this
(`testapps/agents/src/index.ts:108-122`) and `startFlowServer`'s default
`cors()` doesn't set it.

**Fix:** first-party `startAgentServer()` / `agents:` option in
`@genkit-ai/express`, with the stream-id CORS default. `serveAgents()` here
is a candidate implementation.

## 2. Agents registered by plugins are awkward to reach

A plugin that calls `defineAgent` has no way to hand the `Agent` object to
user code. The route back is `registry.lookupAction('/agent/<name>')` - an
undocumented key convention - plus a cast. A runtime brand does exist
(`__action.actionType === 'agent'`, set at `js/ai/src/agent.ts:1003`; the
registered action IS the full Agent composite via `Object.assign`,
`agent.ts:1390-1447`), but nothing documents that this is the supported way
to recover an `Agent`, and enumeration means filtering `listActions()` keys
by prefix.

**Fix:** `ai.agent(name)` instance method mirroring `ai.prompt(name)`
(`js/genkit/src/genkit.ts:348`), plus typed enumeration.

## 3. Slash-namespaced tool names are silently rewritten for the model

Tools named `<ns>/<name>` (the MCP convention, used here for per-agent
namespacing) are exposed to the model by short name only: `toToolDefinition`
strips the namespace (`js/ai/src/tool.ts:263-288`) and the tool loop keys on
the segment after the last `/`
(`js/ai/src/generate/resolve-tool-requests.ts:44-69`); calling the full name
hits `NOT_FOUND` (`:100-107`). Cost us a live NOT_FOUND loop, and any prompt
text naming a tool by full name is silently wrong. Not documented anywhere
in-repo; the MCP README shows namespaced names without saying the model sees
short ones.

Sharper edge: the namespace does not prevent collisions - two tools
`agentA/search` and `agentB/search` in one generate call throw
`Cannot provide two tools with the same name`
(`resolve-tool-requests.ts:56-69`). Per-agent namespacing (this plugin's
scheme) only helps while agents' toolsets stay disjoint per call; the
`agents` delegation middleware can put both in play.

**Fix:** document the short-name contract with `defineTool`; consider
collision-aware aliasing in the tool map.

## 4. `@genkit-ai/middleware` is easy to miss

We fully reimplemented progressive-disclosure skills before discovering the
first-party `skills` middleware - alongside `agents` (delegation),
`toolApproval`, `artifacts`, `filesystem`, `retry`, `fallback`. To be fair:
the genkit README does list the catalog (`js/genkit/README.md:187-199`) and
shows `toolApproval` in its agents section - we missed it. The residual
point: the middleware package is the most capability-shaped layer genkit has
(each entry maps to a folder-per-capability convention like this plugin's),
and neither the agents docs nor the plugin-authoring docs lead you to it.

**Fix:** cross-reference the middleware catalog from the agents docs.

## 5. Vertex plugin init failures are undiagnosable by design

Original claim was "vertexAI() can't infer the project from gcloud ADC" -
that's wrong at the code level: `getProjectId` falls back through
`GCLOUD_PROJECT`, `FIREBASE_CONFIG`, and `authClient.getProjectId()`
(`js/plugins/google-genai/src/vertexai/utils.ts:209-241`), and
google-auth-library does shell out to `gcloud config config-helper`. Our
failure was environmental.

The real defect our experience points at: `getDerivedOptions` swallows the
errors of all four fallback attempts and replaces them with one generic
message (`utils.ts:119-163`) - "Unable to determine client options. Please
set either apiKey or projectId and location" - which suggests an API key
first (wrong path for the GCP-native audience), names no env var, and
shadows the inner error at `utils.ts:244` that actually says
`GCLOUD_PROJECT`. Whatever actually failed (PATH, expired login, IAM) is
undiscoverable from the message.

**Fix:** surface the per-attempt errors (or the last one) and name the env
vars in the message.

## 6. Deploy story: no environment-aware session store default

A store-less agent defaults to client-managed state (throwaway per-invocation
`InMemorySessionStore`, `js/ai/src/agent.ts:1020`); `FileSessionStore` (the
jsdoc example) breaks on Cloud Run (ephemeral disk, multi-instance);
`FirestoreSessionStore` exists but only via `@genkit-ai/firebase/beta`
(`plugins/firebase/src/beta.ts:17-22`) and must be discovered and wired
manually. No environment detection anywhere (zero `K_SERVICE` hits in js/).
More building blocks exist than are discoverable - the firebase plugin also
ships `FirestoreStreamManager`/`RtdbStreamManager` for durable stream
reconnect on serverless - but nothing composes them into a default. One
env-aware store default plus an agent server helper would make deployment
literally `gcloud run deploy --source .` with zero keys (ADC covers Vertex).

**Fix:** environment-aware store default, or at minimum a deploy guide for
the agents API.

## 7. Durability is turn-granular; the gaps compound on serverless

Snapshots persist only at turn boundaries (`maybeSnapshot` calls,
`js/ai/src/agent.ts:469-474,511-516`). A crash mid-turn loses the whole turn
and re-executes side effects on retry - the `TurnContext.snapshotId`
idempotency hook (`agent.ts:151-170,444-460`) is convention only. No
intra-turn journal means long turns can't span serverless request timeouts.
Detached turns die with the instance: the heartbeat is an in-process
`setInterval` (`agent.ts:1143-1157`) and `expired` is computed on read, never
written back (`agent.ts:1334-1341`) - no durable resume trigger exists.
Snapshots carry no code-version field (`SessionSnapshotSchema`,
`js/ai/src/agent-types.ts:342-353`), so resume after a deploy replays
against changed agent code unguarded.

**Fix (long-term):** opt-in intra-turn step journaling (middleware
`tool`/`model` hooks are the natural seam), durable resume triggers, version
stamp on snapshots.

## 8. `.prompt` parsing: three routes, none blessed

An external plugin wanting dotprompt-compatible files can (a) use
`registry.dotprompt.parse()` - a public readonly field
(`js/core/src/registry.ts:164`) whose value over a fresh instance is the
registry-wired schema resolver; (b) use `loadPromptFolder`, publicly exported
from `@genkit-ai/ai` (`js/ai/src/index.ts:106`) but register-only - it
doesn't return parsed frontmatter; or (c) depend on the standalone
`dotprompt` npm package directly (where `ParsedPrompt.raw` is documented).
Nothing says which is the supported path, and the `genkit` package itself
re-exports none of it. This plugin uses (a) and reads custom frontmatter
keys (`delegates`, `requireApproval`) from `parsed.raw`.

**Fix:** document the blessed path (and an extension-keys convention for
frontmatter).

## 9. Sub-agent delegation is one-shot; interrupted sub-agents can't resume

The `agents` middleware (first-party) gives clean isolated delegation
(`delegate_to_<name>` tools, historyLength-gated context, artifact merging,
maxDelegations), but its own source states the limit: a sub-agent interrupt
is flattened into a plain tool response because "there is no stateful
sub-agent runtime to resume into" - so approvals inside a sub-agent can't
round-trip, and multi-turn orchestrator/sub-agent interaction isn't
possible. The building blocks already exist (sub-agents have stores,
sessions and snapshot ids); delegation just doesn't persist the sub-agent's
`sessionId` into parent state or offer a resume path.

**Fix:** stateful delegation in `@genkit-ai/middleware` - record sub-agent
sessionIds per invocation, propagate interrupts as interrupts, and let a
later turn resume the sub-agent session.

## 10. Dev UI / CLI failure modes are cryptic

Hit while demoing the agents Dev UI flow:

- **Version skew renders silently.** A globally-installed `genkit-cli`
  older than the runtime's agents API (`1.33.0-rc.1` UI against a `1.40.x`
  runtime) shows no agents at all - no "unknown action type", no "your CLI
  predates this runtime feature", just an empty UI. The update nudge the CLI
  prints is generic and doesn't connect "old CLI" to "missing surface".
- **Stale runtime files produce a raw TRPC error.** Runtimes register via
  JSON files under `.genkit/runtimes`; a process killed without SIGINT
  leaves its file behind, and the Dev UI then reports
  `TRPCClientError: No runtime found with ID <pid>-<port>` / "No app
  detected" instead of detecting the dead pid and cleaning up (or saying
  "this runtime is gone - restart your app"). Several overlapping
  `genkit start` invocations compound this: each takes the next UI port
  (4000, 4002, ...) and the browser tab keeps talking to a dead one.

**Fix:** UI-side runtime liveness check (pid probe) with a human message,
stale-file cleanup on scan, and a version handshake between UI and runtime
that names the mismatch.

## 11. Smaller items

- `ToolFnOptions` / `ToolRunOptions` (the tool fn's second argument: ambient
  context, `interrupt()`, `resumed`) are not exported from the `genkit`
  package, nor from `@genkit-ai/ai`'s root - only defined in
  `ai/src/tool.ts`. Helper libraries wrapping `defineTool` must re-declare
  the shape structurally (this plugin's `AgentDirToolContext`).

- `mockModel` needs `info: { supports: { tools: true } }` boilerplate to
  avoid "does not support tools" warnings in every agent test
  (`js/ai/src/testing/mock-model.ts:382` vs
  `js/ai/src/generate/action.ts:580-588`).
- `FileSessionStore` can serve a stale `sessionId` lookup after a crash
  between snapshot write and pointer write - acknowledged only in a private
  comment (`js/ai/src/session-stores.ts:541-551`).
- `resume.restart` validates by deep-equal input match against history
  (`js/ai/src/agent.ts:1672-1682`) - sharp for clients that re-serialize
  (dropped `undefined` fields, number formatting).
- Published agents docs exist (genkit.dev/docs/js/agents) but the beta
  surface (snapshot statuses, `use` on prompts/agents, detach, branching) is
  substantially deeper than them; `ai/src/agent.ts` source comments are the
  real reference today.
