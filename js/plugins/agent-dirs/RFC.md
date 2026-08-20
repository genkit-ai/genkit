# [Draft] RFC: agent-dirs - a directory convention for Genkit agents

- Status: retrospective RFC. Written after building the prototype, so the
  claims below describe what the code actually does, not a proposal on paper.
- Artifacts: `@genkit-ai/agent-dirs` (this package), `testapps/agent-dirs-demo`.

## Summary

An agent is a directory. To test how far Genkit already supports that model,
we built a thin compiler from files to existing primitives - registry,
dotprompt, the beta agents API, `@genkit-ai/middleware` - and ran it end to
end on Vertex AI: tools, skills, knowledge, delegation, approval gating,
snapshot resume over HTTP. It mostly just worked, which is the point: the
middleware catalog maps almost one-to-one onto folder-per-capability, and
every compiled agent can be written by hand in roughly ten lines, so the
convention adds no runtime and locks nobody in.

Where it didn't just work, the gaps were small and specific, and they're the
substance of this RFC: agent HTTP serving is
hand-rolled per agent today, agents can't be looked up by name, `AgentConfig`
isn't exported, and there's no session-store default that survives
deployment.

Concretely, `agents/<name>/` compiles to one `ai.defineAgent(...)` call:
`instructions.md` supplies model, config and system prompt; `tools/*.ts` supply
tools; `skills/` and `knowledge/` folders and the `delegates:` /
`requireApproval:` frontmatter keys each compile to an entry on the standard
`use: [...]` chain; `serveAgents(ai)` exposes every registered agent over
HTTP in the contract `remoteAgent` already expects.

## Motivation

The agent-as-directory model stopped being one vendor's idea in a single
week of June 2026. On June 12, Google Cloud published the Open Knowledge
Format, formalizing the LLM-wiki pattern: knowledge as a directory of
markdown files with YAML frontmatter, no runtime, no SDK. It shipped inside
Knowledge Catalog, with a second version following within six weeks. Five
days later, Vercel launched eve on the thesis that the entire agent is a
directory: one folder per capability, prompts and skills as markdown for
non-engineers, tools as single TypeScript files, deployment as one command.
Anthropic's SKILL.md had already become a de facto cross-agent standard by
then, with the same skill folders working unchanged across Claude Code,
OpenClaw, and Codex. Three organizations converged independently on "the
agent is what's on disk," each standardizing a different layer of it.

The demand signal is measurable and attaches to the authoring model
specifically. Eve reached 4.1k GitHub stars within six weeks of launch, and
its reception has centred on the directory convention rather than the
runtime, which Genkit's middleware catalog already covers almost one-to-one.
Meanwhile the most common published criticism of eve is that production
usage is tied to Vercel's platform. Genkit is positioned to be the open
counterpart: the capability layer exists, Cloud Run deployment exists, and,
via OKF, the knowledge layer of this convergence is already a Google
standard looking for an agent framework that executes it. This convention
deliberately adopts the existing contracts (SKILL.md, OKF, and one shared
markdown-plus-frontmatter shape for instructions) rather than inventing new
ones. The distance from Genkit today to the full
directory model turned out to be one thin compiler plus three or four small
first-party pieces: agent HTTP serving, agent lookup by name, and a
session-store default that survives deployment.

The largest of those is serving, and the gap is one of aggregation, not
documentation. The documented path (genkit.dev/docs/js/agents/http) is
hand-writing three `expressHandler` routes per agent (primary,
`/getSnapshot`, `/abort`) to match the URL contract `remoteAgent` expects
(`js/genkit/src/client/agent.ts:53-91`). `startFlowServer` is flows-only, so
no helper exists, and the cost shows inside the repo itself:
`testapps/agents/src/index.ts:126-141` hand-rolls exactly the wiring the
docs describe, and this package's demo had to do the same. The remaining
gaps are addressed in priority order under "Possible upstreaming."

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
   Instructions are markdown with YAML frontmatter, split by the
   convention's own parser (no dotprompt involvement) and compiled to
   `definePrompt` fields; the body reaches the model verbatim.
4. **Authoring errors fail loudly by default.** Broken frontmatter YAML,
   mistyped frontmatter values, unknown `requireApproval` or `delegates`
   names, broken tool files and agent-name collisions all throw at startup
   with named errors, and each registration logs a summary of what compiled. The
   audience this convention targets is exactly the one silent degradation
   hurts most, and approval gating in particular has to be fail-closed: a
   misspelled tool name that silently disables a gate is worse than no gate.

## The convention

```
agents/
  support/
    instructions.md     # YAML frontmatter + markdown; body = system prompt
    tools/*.ts          # defineDirTool({config}, fn) or (ai) => ai.defineTool(...)
    skills/<name>/SKILL.md # -> skills middleware (use_skill, progressive disclosure)
    knowledge/*.md      # OKF bundle -> okfKnowledge middleware (lookup_knowledge)
    agent.ts            # optional override: (config) => config
```

Frontmatter carries `description`, `model`, `config` and `tools`, plus
`delegates: [agent]` (delegation
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

- **Template semantics: resolved by removal.** The `instructions.md` body
  is the system prompt verbatim - the compiler passes it as a `Part`, which
  bypasses the dotprompt templating that `definePrompt` applies to string
  systems, so `{{...}}` is inert (and warns, as it usually signals an
  author expecting templating). Parameterized prompts stay a code feature
  (`agent.ts` override / plain `defineAgent`). (An earlier revision reused
  the `.prompt` extension and dotprompt's parser; we moved to
  `instructions.md` with our own frontmatter splitter so the file renders
  in review, matches the SKILL.md / OKF authoring shape, and cannot change
  meaning under it via live templates.)
- **Beta-API surface.** Everything sits on `GenkitBeta`. `AgentConfig` is
  not exported from `genkit/beta`, so the plugin derives it via
  `Parameters<GenkitBeta['defineAgent']>[0]`, which collapses the generics
  and will break silently if the signature changes.
- **Tool loading needs a TS-capable runtime** (tsx) in dev; production
  should precompile.
- Three frontmatter dialects (instructions.md, SKILL.md, OKF) coexist by
  design but need the reference table now in the README.

## Alternatives considered

- Pure code-first (status quo): kept as the escape hatch at every level
  rather than the default DX.
- `ext`-namespaced frontmatter keys (`agentDirs.delegates:`): more
  collision-proof per the dotprompt spec. Moot since the convention now
  owns the frontmatter format outright; bare keys with unknown-key
  warnings remain the guardrail.
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
- In-process watch mode. Skill and knowledge bodies are already re-read per
  turn, and `serve --watch` restarts the process on directory changes;
  re-registering in place needs registry un-register support.
- Typegen for `directoryAgent(ai, 'name')` state types.
- Whether `input` / `output` prompt-style frontmatter should apply to
  agents (currently unknown keys: warned and ignored).
- Channel adapters (Slack, cron). Out of scope here; `serveAgents`' `app`
  option is the seam.
