# RFC: agent-dirs - a directory convention for Genkit agents

- Status: retrospective RFC. Written after building the prototype in this
  package, so the claims below describe what the code actually does, not a
  proposal on paper.
- Artifacts: `@genkit-ai/agent-dirs` (this package), `testapps/agent-dirs-demo`,
  [FRICTION.md](./FRICTION.md) (API gaps hit along the way, with file:line
  evidence).

## Summary

An agent is a directory. `agents/<name>/` compiles to one
`ai.defineAgent(...)` call: `agent.prompt` supplies model, config and system
prompt; `tools/*.ts` supply tools; `skills/` and `knowledge/` folders and the
`delegates:` / `requireApproval:` frontmatter keys each compile to an entry
on the standard `use: [...]` chain. `serveAgents(ai)` exposes every
registered agent over HTTP in the contract `remoteAgent` already expects.

The convention adds no runtime. It is a compiler from files to existing
primitives (registry, dotprompt, the beta agents API, `@genkit-ai/middleware`),
and every compiled agent can be written by hand in roughly ten lines, so
adopting the convention never locks anyone in.

## Motivation

Frameworks like Vercel's eve have shown there is real demand for
"agent = directory of files, one folder per capability": non-engineers edit
prompts, skills and knowledge as markdown, engineers add tools as single
files, deployment is one command. Genkit already has the underlying
capability layer, and the middleware catalog maps almost one-to-one onto the
folder-per-capability model. What's missing is the convention wiring files
to it, plus a few last pieces: agent HTTP serving, agent lookup by name, a
store default that survives deployment.

The serving gap is one of aggregation, not documentation: the documented
path (genkit.dev/docs/js/agents/http) is hand-writing three `expressHandler`
routes per agent (primary, `/getSnapshot`, `/abort`) to match the URL
contract `remoteAgent` expects (`js/genkit/src/client/agent.ts:53-91`).
`startFlowServer` is flows-only, so there is no helper for this;
`testapps/agents/src/index.ts:126-141` and this package's demo each ended up
hand-rolling the same wiring the docs show.

## Design decisions (and what the prototype showed)

1. **The convention is a thin compiler, with code as the escape hatch at
   every level.** The compiler emits a plain `AgentConfig`. An optional
   `agent.ts` override receives the compiled config and may amend it. Tool
   files can use the `defineDirTool` helper (identical to `defineTool`'s
   `(config, fn)` arguments, with `name` defaulting to the filename) or a
   fully native `(ai) => ai.defineTool(...)` factory. The README shows the
   complete hand-written equivalent.
2. **Capabilities compile to middleware rather than compiler features.**
   Each capability is a table entry contributing
   `{ middleware, toolNames, label }`. Skills use the first-party `skills`
   middleware. We initially reimplemented skills in the compiler before
   finding the catalog, then deleted our version; that convinced us the
   right shape is "convention wires folders to the middleware layer",
   nothing more. Cross-cutting concerns derive from the contributions: the
   `toolApproval` allow-list is computed from the union of contributed tool
   names, so adding a capability cannot silently gate its own tools.
3. **Native surfaces only.** Registration goes through the ordinary
   registry, so Dev UI, `remoteAgent`, evals and tracing work unchanged.
   Prompts are dotprompt files.
4. **Authoring errors fail loudly by default.** Broken frontmatter YAML
   (including dotprompt's silent fallback, where the unparsed file becomes
   the template), unknown `requireApproval` or `delegates` names, broken
   tool files and agent-name collisions all throw at startup with named
   errors, and each registration logs a summary of what compiled. The
   audience this convention targets is exactly the one silent degradation
   hurts most, and approval gating in particular has to be fail-closed: a
   misspelled tool name that silently disables a gate is worse than no gate.

## The convention

```
agents/
  support/
    agent.prompt        # dotprompt; template = system prompt (single message)
    tools/*.ts          # defineDirTool({config}, fn) or (ai) => ai.defineTool(...)
    skills/<n>/SKILL.md # -> skills middleware (use_skill, progressive disclosure)
    knowledge/*.md      # OKF bundle -> okfKnowledge middleware (lookup_knowledge)
    agent.ts            # optional override: (config) => config
```

Frontmatter is standard dotprompt plus `delegates: [agent]` (delegation
tools via the `agents` middleware; sub-agents are just other registered
agents, directory-defined or not) and `requireApproval: [tool]`
(interrupt-gated tools via `toolApproval`, allow-list computed after the
override runs).

Knowledge uses Google Cloud's Open Knowledge Format: markdown plus YAML
frontmatter with `type` required, `index.md` honored, `status: deprecated`
dropped from indexes and error hints, `stale_after` flagged. `okfKnowledge`
deliberately mirrors the `skills` middleware's structure so it could move to
`@genkit-ai/middleware` with minimal change.

Serving: `serveAgents(ai)` discovers agents from the registry and mounts
`POST /api/<name>` plus `/getSnapshot` and `/abort`, with
`X-Genkit-Stream-Id` exposed in CORS and an optional `streamManager` for
durable reconnects.

## What the prototype does end to end

Running against Vertex AI via ADC, in one session: the model called a
directory tool, loaded a skill on demand through `use_skill`, answered from
an OKF knowledge document through `lookup_knowledge`, delegated to a second
directory agent (`delegate_to_shipping`, which ran the sub-agent's own tool
and returned its output to the orchestrator), persisted a snapshot per turn,
and resumed the session by `sessionId` in a fresh chat over HTTP. The strict
negative paths are verified too: broken YAML and a misspelled gated tool
name both abort startup with specific errors.

## Costs and open risks

- **Template semantics.** We use the `.prompt` template as a system prompt
  (single message), where `loadPromptFolder` maps templates to `messages`.
  The compiler rejects `{{role}}` / `{{history}}` templates with a clear
  error, but reusing the file extension while narrowing its semantics is a
  real cost. Alternatives: compile through the `messages` channel, or use a
  distinct extension.
- **Beta-API surface.** Everything sits on `GenkitBeta`. `AgentConfig` is
  not exported from `genkit/beta`, so the plugin derives it via
  `Parameters<GenkitBeta['defineAgent']>[0]`, which collapses the generics
  and will break silently if the signature changes.
- **Tool loading needs a TS-capable runtime** (tsx) in dev; production
  should precompile.
- Three frontmatter dialects (dotprompt, SKILL.md, OKF) coexist by design
  but need the reference table now in the README.

## Alternatives considered

- Pure code-first (status quo): kept as the escape hatch at every level
  rather than the default DX.
- `ext`-namespaced frontmatter keys (`agentDirs.delegates:`): more
  collision-proof per the dotprompt spec. We chose bare keys for authoring
  ergonomics, with unknown-key warnings as the guardrail; worth revisiting
  if dotprompt reserves new keys.
- Nested sub-agent directories (`agents/x/subagents/y`): rejected. Flat
  directories plus explicit `delegates:` keeps every agent addressable,
  reusable and individually servable.
- Custom skills/knowledge formats: rejected in favor of the existing skills
  contract and OKF.

## Possible upstreaming

Whether any of this belongs first-party is your call; the pieces are
separable, and we're happy to pare back or drop any of them. Ordered by
what we think closes the biggest gaps:

1. An agent server in `@genkit-ai/express` (`startAgentServer()` or an
   `agents:` option on `startFlowServer`). The documented path today
   (genkit.dev/docs/js/agents/http) is three hand-written `expressHandler`
   routes per agent; a helper collapses that to one call and adds what the
   docs currently leave out (CORS with `X-Genkit-Stream-Id` exposed,
   optional stream-manager wiring). `serveAgents` here is a candidate
   implementation. Relevant to every agents user, convention or not.
2. `ai.agent(name)` on `GenkitBeta`, mirroring `ai.prompt(name)`, plus typed
   agent enumeration, and exporting `AgentConfig` from `genkit/beta`.
3. `okfKnowledge` into `@genkit-ai/middleware` next to `skills`.
4. An environment-aware session store default (file locally, Firestore on
   GCP). Together with (1) this makes agent deployment
   `gcloud run deploy --source .` with no keys.
5. Stateful sub-agent delegation in the `agents` middleware: persist
   sub-agent sessionIds per invocation, propagate interrupts, add a resume
   path. The stores/sessions/snapshot primitives already exist.
6. The convention layer itself, if (1)-(3) land where they belong.

## Unresolved questions

- `state:` schema in frontmatter (picoschema to `stateSchema`); wants a
  public picoschema conversion API.
- Watch mode. Skill and knowledge bodies are already re-read per turn;
  prompts and tools need a restart.
- Typegen for `directoryAgent(ai, 'name')` state types.
- Whether `input` / `output` dotprompt frontmatter should apply to agents
  (currently warned and ignored).
- Channel adapters (Slack, cron). Out of scope here; `serveAgents`' `app`
  option is the seam.

## References

- [FRICTION.md](./FRICTION.md), verified API gaps with file:line evidence.
- [README.md](./README.md), the convention reference and hand-written
  equivalent.
- Vercel eve (agent-as-directory prior art); Google Cloud OKF spec;
  `js/plugins/middleware`.
