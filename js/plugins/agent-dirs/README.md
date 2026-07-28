# @genkit-ai/agent-dirs (experimental prototype)

"Agent = directory" convention layer over Genkit's beta agents API. One
sub-directory per agent compiles to a single `ai.defineAgent(...)` call, and
each capability folder or frontmatter key compiles to the corresponding
middleware - the convention is sugar, never a cage: anything the files express
can be written by hand in ~10 lines, and an `agent.ts` override lets code
amend the compiled config.

## Layout

```
agents/
  support/
    agent.prompt      # dotprompt: frontmatter (model, config, description,
                      # tool refs, delegates, requireApproval) + template =
                      # system prompt
    tools/
      lookupOrder.ts  # default-exports defineDirTool({config}, fn)
    skills/
      refund-policy/
        SKILL.md      # -> first-party `skills` middleware (`use_skill` tool
                      #    + system-prompt index, loaded progressively)
    knowledge/        # Open Knowledge Format bundle -> `okfKnowledge`
      carriers.md     #    middleware (`lookup_knowledge` tool + index;
                      #    honors index.md, status, stale_after)
    agent.ts          # optional: (compiledConfig) => config override
```

## How it compiles

Everything registers through the ordinary registry and the standard
`use: [...]` middleware chain, so Dev UI, `remoteAgent`, evals and tracing
work with no extra wiring:

| Convention | Compiles to |
| ---------- | ----------- |
| `agent.prompt` frontmatter + template | `definePrompt` fields (model, config, system) |
| `tools/*.ts` | `defineTool` under `<agent>/<file>` (model sees the short name) |
| `skills/<name>/SKILL.md` | `use: [skills({skillPaths})]` (`@genkit-ai/middleware`) |
| `knowledge/` (OKF bundle) | `use: [okfKnowledge({knowledgePaths})]` (this package; upstream candidate) |
| frontmatter `delegates: [agent]` | `use: [agents({agents})]` - `delegate_to_<name>` tools |
| frontmatter `requireApproval: [tool]` | `use: [toolApproval({approved: <complement>})]` - listed tools interrupt for approval |

## Usage

```ts
import { genkit } from 'genkit/beta';
import { agentDirs, directoryAgent } from '@genkit-ai/agent-dirs';

const ai = genkit({ plugins: [agentDirs({ dir: './agents' })] });

const support = await directoryAgent(ai, 'support');
const chat = support.chat();
const res = await chat.send('Where is order A123?');
```

(`directoryAgent` follows the `remoteAgent` idiom: a noun-phrase factory for
an Agent handle you didn't `defineAgent` yourself, named for where the agent
lives. A first-party version would more naturally be `ai.agent(name)`,
mirroring `ai.prompt(name)`.)

Serving over HTTP is one call:

```ts
import { serveAgents } from '@genkit-ai/agent-dirs';

await serveAgents(ai); // POST /api/<name> (+ /getSnapshot, /abort) per agent
```

Clients then chat with `remoteAgent({ url })` from `genkit/beta/client`; the
demo testapp is exactly this plus the Dev UI (`pnpm genkit:dev`). Nothing in
`serveAgents` is directory-specific - it wires the three routes any
`defineAgent` user needs (both in-tree agent testapps previously hand-rolled
them), so its natural upstream home is `@genkit-ai/express` as a sibling of
the flows-only `startFlowServer`. `listAgents(ai)` / `directoryAgent(ai,
name)` remain available for custom wiring.

Each agent gets a per-agent `FileSessionStore` under
`./.genkit/agent-snapshots/<name>` by default; pass `store: (name) => ...` for
Firestore or other stores.

## OKF knowledge

`knowledge/` is treated as an [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog)
bundle: markdown concepts with YAML frontmatter (`type` required). The
`okfKnowledge` middleware injects a `<knowledge>` index (root `index.md` when
present, otherwise synthesized from `title`/`description`/`type`) and serves
bodies via `lookup_knowledge`. Lifecycle is honored: `status: deprecated`
concepts are dropped from the index, `stale_after` in the past flags the entry
as stale. The middleware is exported standalone and is a candidate to move to
`@genkit-ai/middleware` next to `skills`.

## Prototype status / not yet done

- `state:` schema in frontmatter (picoschema) → `stateSchema`
- channel adapters (Slack/cron/web)
- watch mode / hot reload on directory changes
- typegen for a typed `agent(ai, 'name')`
- OKF attested computations, provenance/verification surfacing
- loading `tools/*.ts` requires a TS-capable runtime (tsx) - fine for dev;
  production builds should precompile to `.mjs`
