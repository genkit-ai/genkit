# @genkit-ai/agent-dirs (experimental prototype)

"Agent = directory" convention layer over Genkit's beta agents API. One
sub-directory per agent compiles to a single `ai.defineAgent(...)` call, and
each capability folder or frontmatter key compiles to the corresponding
middleware. The convention is a thin compiler over existing primitives, and
anything it produces can be written by hand (see
[Ejecting to code](#ejecting-to-code)).

## Quickstart

The minimum viable agent is one file. Create `agents/helper/instructions.md`:

```markdown
You are a helpful, concise assistant.
```

Frontmatter is optional - without a `model` key the agent uses the plugin's
`defaultModel` option (default: `vertexai/gemini-3.5-flash`). To pick a
model per agent:

```yaml
---
description: A helpful assistant.
model: vertexai/gemini-2.5-flash
---
You are a helpful, concise assistant.
```

No host code is needed - the runner serves every agent directory:

```sh
agent-dirs serve ./agents   # POST /api/helper (+ /getSnapshot, /abort)
```

(The package is unpublished, so `npx agent-dirs` does not resolve yet;
inside this workspace, build the plugin once and use
`pnpm exec agent-dirs serve` from a package that depends on it.)

The runner binds `127.0.0.1` - the endpoints carry no auth. Pass
`--host 0.0.0.0` only behind a platform ingress (e.g. in a container).
It uses Vertex AI via ADC and the built-in defaults; per-project choices
(provider, session store) are platform configuration (firebase.json), not
project code. To embed in your own app instead:

```ts
import { agentDirs, serveAgents } from '@genkit-ai/agent-dirs';
import { vertexAI } from '@genkit-ai/google-genai';
import { genkit } from 'genkit/beta';

const ai = genkit({
  plugins: [vertexAI(), agentDirs({ dir: './agents' })],
});
await serveAgents(ai);
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
    instructions.md   # REQUIRED - YAML frontmatter + markdown system prompt
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

A tool file - `defineDirTool` takes exactly `ai.defineTool`'s `(config, fn)`
arguments, with `name` optional (defaults to the filename). The fn receives
the standard second argument: `ctx.context` (ambient ActionContext, e.g.
auth), `ctx.interrupt()` (human-in-the-loop pause), `ctx.resumed`:

```ts
import { defineDirTool } from '@genkit-ai/agent-dirs';
import { z } from 'genkit';

export default defineDirTool(
  {
    description: 'Looks up an order by id.',
    inputSchema: z.object({ orderId: z.string() }),
    outputSchema: z.object({ status: z.string() }),
  },
  async ({ orderId }, ctx) => {
    if (!ctx.context.auth) ctx.interrupt({ reason: 'auth required' });
    return { status: `Order ${orderId} has shipped` };
  }
);
```

Prefer the fully native API? A tool file may instead default-export a
factory - the escape hatch for the full `defineTool` surface (multipart,
custom metadata) or definition-time registry access. The factory owns the
tool's name (no automatic `agent-dirs/<agent>/` namespacing):

```ts
import type { GenkitBeta } from 'genkit/beta';
import { z } from 'genkit';

export default (ai: GenkitBeta) =>
  ai.defineTool(
    { name: 'checkStock', description: '...', inputSchema: z.object({}) },
    async () => 'in stock'
  );
```

## How it compiles

Everything registers through the ordinary registry and the standard
`use: [...]` middleware chain, so Dev UI, `remoteAgent`, evals and tracing
work with no extra wiring:

| Convention | Compiles to |
| ---------- | ----------- |
| `instructions.md` frontmatter + body | `definePrompt` fields (model, config, system) |
| `tools/*.ts` | `defineTool` under `agent-dirs/<agent>/<file>` |
| `skills/<name>/SKILL.md` | `use: [skills({skillPaths})]` (`@genkit-ai/middleware`) |
| `knowledge/` (OKF bundle) | `use: [okfKnowledge({knowledgePaths})]` (this package) |
| frontmatter `delegates: [agent]` | `use: [agents({agents})]` - `delegate_to_<name>` tools |
| frontmatter `requireApproval: [tool]` | `use: [toolApproval(...)]` - listed tools interrupt for approval |

### Frontmatter reference (`instructions.md`)

| Key | Type | Notes |
| --- | ---- | ----- |
| `description` | string | shown in Dev UI / delegation tool descriptions |
| `model` | string | e.g. `vertexai/gemini-2.5-flash`; when omitted: the plugin's `defaultModel` option, else the `genkit({ model })` instance default, else `vertexai/gemini-3.5-flash` |
| `config` | object | model config (temperature, ...) |
| `tools` | string[] | names of tools registered elsewhere |
| `delegates` | string[] | other agent directory names; validated |
| `requireApproval` | string[] | model-visible tool names to gate; validated, fail-closed |

The markdown body is the agent's **system prompt**. It still flows through
Genkit's prompt templating, so multi-message directives (`{{role}}`,
`{{history}}`) are rejected at compile time. An empty
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
turn, so edits are live in a running session. `instructions.md` and tool code
are compiled at startup - restart (`tsx --watch` restarts on TS changes; an
instructions edit needs a manual restart). Real watch mode is future work.

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

- `agent-dirs serve [dir] [--port <n>] [--host <h>]` - zero-code CLI runner
  (see Quickstart).
- `agentDirs(options)` - the plugin. `dir`, `store`, `snapshotDir`,
  `strict`, `defaultModel`.
- `directoryAgent(ai, name)` - resolve a registered agent (the `remoteAgent`
  naming idiom: a factory for an Agent handle named for where it lives).
- `listAgents(ai)` - every registered agent by name (not only directory
  ones).
- `serveAgents(ai, options)` - HTTP serving, see above.
- `defineDirTool(config, fn)` - typed authoring helper for tool files.
- `okfKnowledge(options)` - the OKF middleware, usable with any agent.

## Prototype status / not yet done

- `state:` schema in frontmatter (picoschema) → `stateSchema`
- `input`/`output` frontmatter (warned as unknown keys and ignored)
- watch mode / hot re-registration on directory changes
- typegen for a typed `directoryAgent(ai, 'name')`
- lazy plugin action listing (`listActionsFn`) - agents resolve eagerly
- channel adapters (Slack/cron); OKF attested computations
- loading `tools/*.ts` requires tsx (optional peer) - fine for dev;
  production builds should precompile to `.mjs`
- the CLI lives in-package, which makes `@genkit-ai/google-genai` a hard
  dependency of the library; a real release would split the runner into its
  own package (also giving `npx` a public entry point)
