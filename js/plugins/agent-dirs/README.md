# @genkit-ai/agent-dirs (experimental prototype)

"Agent = directory" convention layer over Genkit's beta agents API. One
sub-directory per agent compiles to a single `ai.defineAgent(...)` call, and
each capability folder or frontmatter key compiles to the corresponding
middleware - the convention is sugar, never a cage (see
[Ejecting to code](#ejecting-to-code)).

## Quickstart

The minimum viable agent is one file. Create `agents/helper/agent.prompt`:

```yaml
---
description: A helpful assistant.
model: vertexai/gemini-2.5-flash
---
You are a helpful, concise assistant.
```

And a `src/index.ts`:

```ts
import { agentDirs, serveAgents } from '@genkit-ai/agent-dirs';
import { vertexAI } from '@genkit-ai/google-genai';
import { genkit } from 'genkit/beta';

const ai = genkit({
  plugins: [vertexAI(), agentDirs({ dir: './agents' })],
});
await serveAgents(ai); // POST /api/helper (+ /getSnapshot, /abort)
```

Test locally: `genkit start -- tsx src/index.ts` and chat with the agent in
the Dev UI, or talk to the HTTP endpoint with `remoteAgent`:

```ts
import { remoteAgent } from 'genkit/beta/client';
const helper = remoteAgent({ url: 'http://localhost:8080/api/helper' });
const chat = helper.chat();
const res = await chat.send('hi');
```

Add `.genkit/` to your `.gitignore`: by default every agent persists session
snapshots (full chat transcripts) to `./.genkit/agent-snapshots/<name>`.

## Full layout

```
agents/
  support/
    agent.prompt      # REQUIRED - dotprompt file; frontmatter + system prompt
    tools/
      lookupOrder.ts  # one file per tool; default-exports defineDirTool(...)
    skills/
      refund-policy/
        SKILL.md      # progressive-disclosure instructions (subdirectory
                      # per skill; the directory name is the skill name)
    knowledge/
      carriers.md     # Open Knowledge Format bundle (flat or nested .md)
    agent.ts          # optional code override: (config) => config
```

A tool file:

```ts
import { defineDirTool } from '@genkit-ai/agent-dirs';
import { z } from 'genkit';

export default defineDirTool(
  {
    description: 'Looks up an order by id.',
    inputSchema: z.object({ orderId: z.string() }),
    outputSchema: z.object({ status: z.string() }),
  },
  async ({ orderId }) => ({ status: `Order ${orderId} has shipped` })
);
```

## How it compiles

Everything registers through the ordinary registry and the standard
`use: [...]` middleware chain, so Dev UI, `remoteAgent`, evals and tracing
work with no extra wiring:

| Convention | Compiles to |
| ---------- | ----------- |
| `agent.prompt` frontmatter + template | `definePrompt` fields (model, config, system) |
| `tools/*.ts` | `defineTool` under `agent-dirs/<agent>/<file>` |
| `skills/<name>/SKILL.md` | `use: [skills({skillPaths})]` (`@genkit-ai/middleware`) |
| `knowledge/` (OKF bundle) | `use: [okfKnowledge({knowledgePaths})]` (this package) |
| frontmatter `delegates: [agent]` | `use: [agents({agents})]` - `delegate_to_<name>` tools |
| frontmatter `requireApproval: [tool]` | `use: [toolApproval(...)]` - listed tools interrupt for approval |

### Frontmatter reference (`agent.prompt`)

| Key | Type | Notes |
| --- | ---- | ----- |
| `description` | string | shown in Dev UI / delegation tool descriptions |
| `model` | string | e.g. `vertexai/gemini-2.5-flash` |
| `config` | object | model config (temperature, ...) |
| `tools` | string[] | names of tools registered elsewhere |
| `delegates` | string[] | other agent directory names; validated |
| `requireApproval` | string[] | model-visible tool names to gate; validated, fail-closed |

The template body is the agent's **system prompt**. Multi-message dotprompt
templates (`{{role}}`, `{{history}}`) are rejected at compile time. An empty
body means "no system prompt" (e.g. an `agent.ts` override supplies one).
Unknown frontmatter keys warn. `skills/<name>/SKILL.md` takes optional
`name`/`description` frontmatter (the *directory* name is authoritative);
`knowledge/*.md` takes OKF frontmatter (`type` required; `title`,
`description`, `status`, `stale_after` honored).

### Tool naming: what the model sees

Tools register as `agent-dirs/<agent>/<file>` but the model only ever sees
the **short name** - the segment after the last `/` (framework behavior).
Consequences: refer to tools by short name in prompt text and in
`requireApproval`; two agents may use the same tool filename, but if
delegation puts both toolsets into one model call, same short names collide
at generate time. Setting `config.name` in `defineDirTool` opts out of
namespacing.

### Strict by default

Authoring errors - unparseable frontmatter YAML, broken tool files, unknown
`delegates`/`requireApproval` names, agent-name collisions - **throw at
startup** with a pointed message. Pass `agentDirs({ strict: false })` to
warn-and-skip instead. Every registration logs a summary of exactly what
compiled:

```
[agent-dirs] registered agent 'support': tools [lookupOrder], skills [refund-policy], knowledge 2 docs
```

### Dev loop

Skill bodies, knowledge documents and the knowledge index are re-read per
turn, so edits are live in a running session. `agent.prompt` and tool code
are compiled at startup - restart (`tsx --watch` restarts on TS changes; a
prompt edit needs a manual restart). Real watch mode is future work.

## Serving and deployment

`serveAgents(ai)` exposes every registered agent: `POST /api/<name>` (turns,
streaming), `/getSnapshot`, `/abort` - the exact contract `remoteAgent`
expects. Options: `port`, `pathPrefix`, `cors`, `app` (mount on your own
express app), `streamManager` (durable stream reconnects). Nothing in it is
directory-specific; its natural upstream home is `@genkit-ai/express` as a
sibling of the flows-only `startFlowServer`.

For production, pass a persistent store - the file default does not survive
serverless instances:

```ts
import { FirestoreSessionStore } from '@genkit-ai/firebase/beta';
agentDirs({ store: (name) => new FirestoreSessionStore(`agents-${name}`) });
```

On Cloud Run, Vertex auth is ADC - no API keys needed.

## Ejecting to code

Every compiled agent is expressible by hand; the convention only arranges
existing primitives. The demo's support agent, ejected:

```ts
const support = ai.defineAgent({
  name: 'support',
  description: 'Customer support agent for the ACME store.',
  model: 'vertexai/gemini-2.5-flash',
  config: { temperature: 0.2 },
  system: 'You are ACME\'s customer support agent...',
  tools: [lookupOrder], // ai.defineTool(...)
  use: [
    skills({ skillPaths: ['./agents/support/skills'] }),
    okfKnowledge({ knowledgePaths: ['./agents/support/knowledge'] }),
  ],
  store: new FileSessionStore('./.genkit/agent-snapshots/support'),
});
```

An `agent.ts` file in the directory is the halfway house: it receives the
compiled config and returns an amended one.

## OKF knowledge

`knowledge/` is treated as an [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog)
bundle: markdown concepts with YAML frontmatter (`type` required). The
`okfKnowledge` middleware injects a `<knowledge>` index (root `index.md` when
present, otherwise synthesized from `title`/`description`/`type`) and serves
bodies via `lookup_knowledge`. Lifecycle is honored: `status: deprecated`
concepts are dropped from the index and from error-message hints (still
loadable by exact path, per OKF conformance); `stale_after` in the past flags
the entry as stale. Exported standalone; upstream candidate for
`@genkit-ai/middleware` next to `skills`.

## API summary

- `agentDirs(options)` - the plugin. `dir`, `store`, `snapshotDir`, `strict`.
- `directoryAgent(ai, name)` - resolve a registered agent (the `remoteAgent`
  naming idiom: a factory for an Agent handle named for where it lives).
- `listAgents(ai)` - every registered agent by name (not only directory
  ones).
- `serveAgents(ai, options)` - HTTP serving, see above.
- `defineDirTool(config, fn)` - typed authoring helper for tool files.
- `okfKnowledge(options)` - the OKF middleware, usable with any agent.

## Prototype status / not yet done

- `state:` schema in frontmatter (picoschema) → `stateSchema`
- `input`/`output` frontmatter (parsed but ignored, with a warning)
- watch mode / hot re-registration on directory changes
- typegen for a typed `directoryAgent(ai, 'name')`
- lazy plugin action listing (`listActionsFn`) - agents resolve eagerly
- channel adapters (Slack/cron); OKF attested computations
- loading `tools/*.ts` requires a TS-capable runtime (tsx) - fine for dev;
  production builds should precompile to `.mjs`
