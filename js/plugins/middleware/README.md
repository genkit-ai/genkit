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

Enables sub-agent delegation by injecting a `call_agent` tool that allows the model to delegate tasks to registered sub-agents. When the model calls the tool, the middleware intercepts the call, runs the specified sub-agent via `ai.generate()`, and returns its response as the tool result.

**Key behaviors:**
- Injects a system prompt describing available sub-agents
- Sub-agent interrupts propagate up as `ToolInterruptError` with metadata identifying the source agent
- Errors from sub-agents are returned as tool responses (not thrown), allowing the model to self-correct

```typescript
import { genkit } from 'genkit';
import { agents } from '@genkit-ai/middleware';

const ai = genkit({ ... });

// Define sub-agents
const researcher = ai.defineAgent({
  name: 'researcher',
  model: 'gemini-2.5-flash',
  system: 'You are a research assistant.',
  tools: [webSearchTool],
});

const coder = ai.defineAgent({
  name: 'coder',
  model: 'gemini-2.5-flash',
  system: 'You are an expert programmer.',
});

// Main orchestrator agent delegates to sub-agents
const orchestrator = ai.defineAgent({
  name: 'orchestrator',
  model: 'gemini-2.5-flash',
  system: 'Delegate research to the researcher and coding to the coder.',
  use: [
    agents({ agents: ['researcher', 'coder'] })
  ]
});
```

You can also customize the tool name:

```typescript
use: [
  agents({
    agents: ['researcher', 'coder'],
    toolName: 'delegate_to_agent' // default: 'call_agent'
  })
]
```

### 2. FileSystem Middleware (`filesystem`)

Enables sub-agent delegation by injecting a `call_agent` tool that allows the model to delegate tasks to registered sub-agents. When the model calls the tool, the middleware intercepts the call, runs the specified sub-agent via `ai.generate()`, and returns its response as the tool result.

**Key behaviors:**
- Injects a system prompt describing available sub-agents
- Sub-agent interrupts propagate up as `ToolInterruptError` with metadata identifying the source agent
- Errors from sub-agents are returned as tool responses (not thrown), allowing the model to self-correct

```typescript
import { genkit } from 'genkit';
import { agents } from '@genkit-ai/middleware';

const ai = genkit({ ... });

// Define sub-agents
const researcher = ai.defineAgent({
  name: 'researcher',
  model: 'gemini-2.5-flash',
  system: 'You are a research assistant.',
  tools: [webSearchTool],
});

const coder = ai.defineAgent({
  name: 'coder',
  model: 'gemini-2.5-flash',
  system: 'You are an expert programmer.',
});

// Main orchestrator agent delegates to sub-agents
const orchestrator = ai.defineAgent({
  name: 'orchestrator',
  model: 'gemini-2.5-flash',
  system: 'Delegate research to the researcher and coding to the coder.',
  use: [
    agents({ agents: ['researcher', 'coder'] })
  ]
});
```

You can also customize the tool name:

```typescript
use: [
  agents({
    agents: ['researcher', 'coder'],
    toolName: 'delegate_to_agent' // default: 'call_agent'
  })
]
```

### 2. FileSystem Middleware (`filesystem`)

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

### 3. Skills Middleware (`skills`)

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

### 4. Tool Approval Middleware (`toolApproval`)

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

### 5. Retry Middleware (`retry`)

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

### 6. Fallback Middleware (`fallback`)

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
