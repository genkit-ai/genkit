# RFC: agent-dirs - a directory convention for Genkit agents

- Status: **retrospective RFC** - written after building the experimental
  prototype in this package, so every design claim below is backed by
  working code and a live demo rather than speculation.
- Artifacts: `@genkit-ai/agent-dirs` (this package), `testapps/agent-dirs-demo`,
  [FRICTION.md](./FRICTION.md) (API gaps hit along the way).

## Summary

An agent is a directory. `agents/<name>/` compiles to one
`ai.defineAgent(...)` call: `agent.prompt` supplies model/config/system
prompt, `tools/*.ts` supply tools, `skills/` and `knowledge/` folders and
`delegates:`/`requireApproval:` frontmatter keys each compile to an entry in
the standard `use: [...]` middleware chain. One call (`serveAgents`) exposes
every agent over HTTP in the exact contract the first-party `remoteAgent`
client already expects.

The convention adds **no new runtime**. It is a compiler from files to
existing Genkit primitives - the registry, dotprompt, the beta agents API,
and the `@genkit-ai/middleware` catalog - and every compiled agent can be
ejected to ~10 lines of hand-written code.

## Motivation

Frameworks like Vercel's eve demonstrate that "agent = directory of files,
folder per capability" is a DX that teams want: non-engineers edit prompts,
skills and knowledge as markdown; engineers add tools as single files;
deployment is one command. Genkit already has the *capability layer* for
this - the beta agents API (sessions, snapshots, interrupts, resume) and a
middleware catalog (`skills`, `agents` delegation, `toolApproval`,
`artifacts`, ...) that maps almost one-to-one onto eve's folder-per-
capability model - but no convention wires files to it, and several
last-mile pieces are missing entirely (agent HTTP serving, agent lookup by
name, a deployable store default).

The gap is visible in-tree: the agents testapp and this package's demo both
independently hand-rolled the same agent-serving routes, and the first-party
`remoteAgent` client hard-codes a server URL contract that no first-party
server implements.

## Design principles (validated by the prototype)

1. **Sugar, never a cage.** The compiler emits a plain `AgentConfig`; an
   optional `agent.ts` override receives it and may amend it (code wins,
   files fill in); the README shows the full eject. Litmus test used
   throughout: "could a user delete the convention and hand-write the
   equivalent in ~10 lines?"
2. **Compile to middleware, don't build features.** Each capability is a
   table entry contributing `{ middleware, toolNames, label }`. Skills use
   the first-party `skills` middleware (the prototype initially reimplemented
   it - deleting that code in favor of the catalog is the lesson), delegation
   uses `agents`, gating uses `toolApproval`. Adding a capability is one
   ~10-line entry; cross-cutting concerns (the approval allow-list) derive
   from the contributed tool names instead of hand-maintained lists.
3. **Native surfaces only.** Registration through the ordinary registry, so
   Dev UI, `remoteAgent`, evals and tracing work unchanged. Prompts are
   dotprompt files. Tool files use `defineTool`'s exact `(config, fn)`
   signature (name optional, defaulting to the filename) or a fully native
   `(ai) => ai.defineTool(...)` factory.
4. **Strict by default.** Authoring errors - broken frontmatter YAML
   (including dotprompt's silent fallback), unknown `requireApproval`/
   `delegates` names, broken tool files, agent-name collisions - throw at
   startup with pointed messages, and every registration logs a summary of
   exactly what compiled. The personas this convention serves are the ones
   silent degradation hurts most; approval gating in particular must be
   fail-closed.

## The convention

```
agents/
  support/
    agent.prompt        # dotprompt; template = system prompt (single-message)
    tools/*.ts          # defineDirTool({config}, fn) or (ai) => ai.defineTool(...)
    skills/<n>/SKILL.md # -> skills middleware (use_skill, progressive disclosure)
    knowledge/*.md      # OKF bundle -> okfKnowledge middleware (lookup_knowledge)
    agent.ts            # optional override: (config) => config
```

Frontmatter: standard dotprompt keys plus `delegates: [agent]` (delegation
tools via the `agents` middleware; sub-agents are just other agents,
directory-defined or not) and `requireApproval: [tool]` (interrupt-gated
tools via `toolApproval`, allow-list computed post-override as the
complement of all model-visible tool names).

Knowledge adopts Google Cloud's Open Knowledge Format: markdown + YAML
frontmatter (`type` required), `index.md` honored, `status: deprecated`
dropped from indexes and error hints, `stale_after` flagged. `okfKnowledge`
is implemented in the same shape as the first-party `skills` middleware
precisely so it can move to `@genkit-ai/middleware`.

Serving: `serveAgents(ai)` = registry-discovered agents at
`POST /api/<name>` + `/getSnapshot` + `/abort` (the `remoteAgent` contract),
CORS incl. `X-Genkit-Stream-Id`, optional `streamManager` for durable
reconnects, optional `app` mount.

## What the prototype proved end to end

Live against Vertex AI (ADC, no keys): a real model called a directory tool,
loaded a skill on demand via `use_skill`, answered from an OKF knowledge doc
via `lookup_knowledge`, delegated to a second directory agent
(`delegate_to_shipping` → sub-agent's own tool → answer surfaced by the
orchestrator), persisted a snapshot per turn, and resumed the session by
`sessionId` in a fresh chat over HTTP. Strict-mode negative paths verified:
broken YAML and a misspelled gated tool name both abort startup with exact,
named errors.

## Drawbacks and open risks

- **Template semantics divergence.** The `.prompt` template is used as a
  system prompt (single message); genkit's own loader maps templates to
  `messages`. The compiler rejects `{{role}}`/`{{history}}` templates loudly,
  but reusing the file extension while narrowing its semantics is a real
  cost. Alternative: a `messages`-channel compilation, or a distinct
  extension.
- **Beta-API churn.** Built entirely on `GenkitBeta`; `AgentConfig` isn't
  even exported (the prototype derives it via `Parameters<...>`), so
  framework evolution can break the plugin silently.
- **TS tool loading needs a TS-capable runtime** (tsx) in dev; production
  should precompile.
- Convention surface must be documented and maintained; three frontmatter
  dialects (dotprompt / SKILL.md / OKF) coexist by design but need a
  reference table (now in the README).

## Alternatives considered

- **Status quo (pure code-first):** rejected as the default DX - but
  preserved as the escape hatch at every level (factory tools, `agent.ts`
  override, full eject).
- **`ext`-namespaced frontmatter keys** (`agentDirs.delegates:`): more
  collision-proof per the dotprompt spec; bare keys chosen for authoring
  ergonomics, with unknown-key warnings as the guardrail. Revisit if
  dotprompt reserves new keys.
- **Nested sub-agent directories** (`agents/x/subagents/y`): rejected -
  flat directories + explicit `delegates:` keeps every agent addressable,
  reusable and individually servable.
- **Custom skills/knowledge formats:** rejected in favor of the first-party
  skills contract and OKF - conventions should adopt specs, not invent them.

## Proposed upstreaming (in order of leverage)

1. `startAgentServer()` / `agents:` option in `@genkit-ai/express`
   (`serveAgents` is a candidate implementation) - closes the
   `remoteAgent`-client-has-no-server gap for every agents user.
2. `ai.agent(name)` on `GenkitBeta`, mirroring `ai.prompt(name)`, plus typed
   agent enumeration; export `AgentConfig` from `genkit/beta`.
3. `okfKnowledge` into `@genkit-ai/middleware` next to `skills`.
4. Environment-aware session store default (file/dev, Firestore when on
   GCP) - with (1), makes agent deployment `gcloud run deploy --source .`.
5. Stateful sub-agent delegation in the `agents` middleware (persist
   sub-agent sessionIds, propagate interrupts, resume path).
6. The convention layer itself (`agentDirs`) as an experimental first-party
   plugin, once (1)-(3) land where they belong.

## Unresolved questions

- `state:` schema in frontmatter (picoschema → `stateSchema`) - wants a
  public picoschema conversion API.
- Watch mode / hot re-registration (skill and knowledge bodies are already
  live per-turn; prompts and tools need restart).
- Typegen for `directoryAgent(ai, 'name')` state types.
- Whether `input`/`output` dotprompt frontmatter should work on agents
  (currently warned and ignored).
- Channel adapters (Slack, cron) - out of scope here; `serveAgents`' `app`
  mount is the seam.

## References

- [FRICTION.md](./FRICTION.md) - verified API gaps with file:line evidence.
- [README.md](./README.md) - user-facing convention reference and eject
  example.
- Vercel eve (agent-as-directory prior art); Google Cloud OKF spec
  (knowledge format); `js/plugins/middleware` (capability catalog).
