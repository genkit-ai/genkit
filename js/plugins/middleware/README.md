# Genkit Middleware

This package provides a collection of useful middlewares for the Genkit JS SDK to enhance model execution, tool usage, and agentic workflows.

## Installation

```bash
npm install @genkit-ai/middleware
# or
pnpm add @genkit-ai/middleware
```

## Available Middlewares

### 1. Agents Middleware (`agents`)

Enables sub-agent delegation. For each configured agent the middleware injects a dedicated delegation tool (e.g. `delegate_to_researcher`) and appends a `<sub-agents>` block to the system prompt listing the available agents and their descriptions. When the model calls a delegation tool, the middleware resolves the target agent from the registry, runs it via its `run()` method, and returns the sub-agent's response as the tool result.

**Key behaviors:**
- Injects **one delegation tool per agent**, named `<toolPrefix>_<agentName>` (default prefix: `delegate_to`).
- Agent descriptions are auto-discovered from the registry (or can be overridden per-agent) and surfaced in the system prompt.
- Sub-agent interrupts and failures are returned as tool responses (not thrown), allowing the orchestrator to self-correct. (Interactive, stateful back-and-forth with an interrupted sub-agent is a future feature.)

- Sub-agent artifacts are merged into the parent session and/or returned inline, controlled by `artifactStrategy`.

**Options:**

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `agents` | `(string \| { name, description? })[]` | — (required) | Agents available for delegation. A string is the agent name; the object form lets you override the description. |
| `toolPrefix` | `string` | `'delegate_to'` | Prefix for generated delegation tool names. Set to `''` to use bare agent names. |
| `maxDelegations` | `number` | unlimited | Maximum sub-agent delegations allowed per generate call. Prevents runaway delegation loops. |
| `historyLength` | `number` | `0` | Number of recent conversation messages (user/model only) to forward to sub-agents as context. |
| `artifactStrategy` | `'inline' \| 'session'` | `'inline'` | `inline`: artifact content is included in the tool result **and** merged into the parent session. `session`: artifacts are merged into the parent session only (the tool result lists names only). Pair `session` with the `artifacts` middleware. |

```typescript
import { genkit } from 'genkit';
import { agents } from '@genkit-ai/middleware';

const ai = genkit({ ... });

// Define sub-agents
const researcher = ai.defineAgent({
  name: 'researcher',
  model: 'gemini-2.5-flash',
  description: 'Searches the web and summarizes findings.',
  system: 'You are a research assistant.',
  tools: [webSearchTool],
});

const coder = ai.defineAgent({
  name: 'coder',
  model: 'gemini-2.5-flash',
  system: 'You are an expert programmer.',
});

// Main orchestrator agent delegates to sub-agents.
// This injects `delegate_to_researcher` and `delegate_to_coder` tools.
const orchestrator = ai.defineAgent({
  name: 'orchestrator',
  model: 'gemini-2.5-flash',
  system: 'Delegate research to the researcher and coding to the coder.',
  use: [
    agents({ agents: ['researcher', 'coder'] })
  ]
});
```

You can customize the tool-name prefix and override descriptions:

```typescript
use: [
  agents({
    // Tools become `ask_researcher` and `ask_coder`.
    toolPrefix: 'ask',
    agents: [
      'researcher', // description auto-discovered from the registry
      { name: 'coder', description: 'Writes TypeScript code' }, // explicit override
    ],
    maxDelegations: 5,
    historyLength: 4,
  })
]
```

### 2. Artifacts Middleware (`artifacts`)

Gives the model tools to interact with session artifacts and injects an `<artifacts>` listing into the system prompt each turn. Useful standalone (e.g. a workspace-builder agent that produces files as artifacts) or combined with the `agents` middleware using `artifactStrategy: 'session'`, so the orchestrator can read artifacts produced by sub-agents.

**Tools provided:**
- `read_artifact` — reads a named artifact from the session and returns its text content.
- `write_artifact` — creates or updates a named artifact (deduplicated by name). Omitted when `readonly: true`.

**Options:**

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `readonly` | `boolean` | `false` | When true, only the `read_artifact` tool is provided. |

```typescript
import { genkit } from 'genkit';
import { agents, artifacts } from '@genkit-ai/middleware';

const ai = genkit({ ... });

// Standalone: an agent that creates and reads artifacts.
const builder = ai.defineAgent({
  name: 'builder',
  model: 'gemini-2.5-flash',
  system: 'You are a code generator. Use write_artifact to create files.',
  use: [artifacts()],
});

// Combined with the agents middleware (session strategy).
const orchestrator = ai.defineAgent({
  name: 'orchestrator',
  model: 'gemini-2.5-flash',
  system: 'You coordinate sub-agents and review their work.',
  use: [
    agents({ agents: ['researcher', 'coder'], artifactStrategy: 'session' }),
    artifacts({ readonly: true }), // can read sub-agent artifacts
  ],
});
```

### 3. FileSystem Middleware (`filesystem`)

Grants the model access to the local filesystem by injecting standard file manipulation tools (`list_files`, `read_file`, `write_file`, `search_and_replace`). All operations are safely restricted to a specified root directory. Note that write operations require setting `allowWriteAccess: true` in the middleware configuration.

```typescript
import { genkit } from 'genkit';
import { filesystem } from '@genkit-ai/middleware';

const ai = genkit({ ... });

const response = await ai.generate({
  model: 'gemini-2.5-flash',
  prompt: 'Create a hello world node app in the workspace',
  use: [
    filesystem({ rootDirectory: './workspace', allowWriteAccess: true })
  ]
});
```

### 4. Skills Middleware (`skills`)


Automatically scans a directory for `SKILL.md` files (and their YAML frontmatter) and injects them into the system prompt. It also provides a `use_skill` tool the model can use to retrieve more specific skills on demand.

```typescript
import { genkit } from 'genkit';
import { skills } from '@genkit-ai/middleware';

const ai = genkit({ ... });

const response = await ai.generate({
  prompt: 'How do I run tests in this repo?',
  use: [
    skills({ skillPaths: ['./skills'] })
  ]
});
```

### 5. Tool Approval Middleware (`toolApproval`)


Restricts execution of tools to an approved list. If the model attempts to call an unapproved tool, it throws a `ToolInterruptError` allowing you to prompt the user for manual confirmation before resuming.

```typescript
import { genkit, restartTool } from 'genkit';
import { toolApproval } from '@genkit-ai/middleware';

const ai = genkit({ ... });

// 1. Initial attempt
const response = await ai.generate({
  prompt: 'write a file',
  tools: [writeFileTool],
  use: [
    toolApproval({ approved: [] }) // Empty list means call triggers interrupt
  ]
});

if (response.finishReason === 'interrupted') {
  const interrupt = response.interrupts[0];
  
  // 2. Ask user for approval, then recreate the tool request with approval
  const approvedPart = restartTool(interrupt, { toolApproved: true });

  // 3. Resume execution
  const resumedResponse = await ai.generate({
    messages: response.messages,
    resume: { restart: [approvedPart] }, 
    use: [
      toolApproval({ approved: [] })
    ]
  });
}
```

### 6. Retry Middleware (`retry`)


Automatically retries failed model generations on transient error codes (like `RESOURCE_EXHAUSTED`, `UNAVAILABLE`) using exponential backoff with jitter.

```typescript
import { genkit } from 'genkit';
import { retry } from '@genkit-ai/middleware';

const ai = genkit({ ... });

const response = await ai.generate({
  model: googleAI.model('gemini-pro-latest'),
  prompt: 'Heavy reasoning task...',
  use: [
    retry({
      maxRetries: 3,
      initialDelayMs: 1000,
      backoffFactor: 2
    })
  ]
});
```

### 7. Fallback Middleware (`fallback`)


Automatically switches to a different model if the primary model fails on a specific set of error codes. Useful for falling back to a smaller/faster model when a large model exceeds quota limits.

```typescript
import { genkit } from 'genkit';
import { fallback } from '@genkit-ai/middleware';

const ai = genkit({ ... });

const response = await ai.generate({
  model: googleAI.model('gemini-pro-latest'),
  prompt: 'Try the pro model first...',
  use: [
    fallback({
      models: [googleAI.model('gemini-flash-latest')], // try flash if pro fails
      statuses: ['RESOURCE_EXHAUSTED']
    })
  ]
});
```
