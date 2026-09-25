# RFC: Agent Backends for Genkit

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Authors** | Elliot Hesp |
| **Created** | 2026-06-02 |
| **Scope** | JS/TS in v1; protocol designed for all Genkit runtimes (Go fast-follow given first-party Modal/Daytona Go SDKs; Python/Dart protocol-only until native providers exist) |

> **Base branch:** this RFC is written against the agent stack on [`pj/agents-sample`](https://github.com/genkit-ai/genkit/tree/pj/agents-sample), not `main`. The referenced APIs (`defineAgent`, the `agents()` middleware, `FileSessionStore`, the coding-agent `run_shell` exhibit) live there.

## Summary

Genkit's agent stack (`defineAgent`, middleware, sessions, interrupts) is production-ready for **API-only agents**, but lacks a **pluggable backend layer** for filesystem access and command execution. Today, coding-style agents assume co-located host execution — which is **not secure** (commands and file I/O run in the app process with access to its credentials and environment) and **breaks in scaled deployments** (ephemeral disk, no cross-replica isolation, session store on local filesystem).

This RFC proposes an **agent-level backend** attached to sessions, consumed implicitly by existing middleware (`filesystem`, `skills`, etc.). Existing `rootDirectory` usage keeps working for backward compatibility, but **a local filesystem is a development convenience, not a production default**: nothing is configured unless you ask for it, and running an agent against the host filesystem is risky outside local dev or an already-sandboxed machine. Remote sandboxes (Modal, Daytona, LangSmith, etc.) become an opt-in factory on `defineAgent`, not a per-middleware configuration.

**Guiding principle:** middleware depends on a storage and execution *interface* and is agnostic to where it runs. `node:fs` is just the local implementation of that interface, not something middleware imports directly. Everything below follows from inverting that dependency.

---

## Motivating use case

We run an internal agent that, among other capabilities, reads and writes project files, installs dependencies, runs builds, and executes commands on the user's behalf, across many concurrent sessions. Capabilities like these need an isolated, durable per-session workspace. Files written in one turn must survive to the next, one session's workspace must never be visible to another, and command execution must not run inside the serving process.

This is precisely the workload Genkit cannot support today. The agent stack handles model orchestration well, but file and command access go straight to host `node:fs` and in-process `exec`, with a shared workspace per instance and no per-session isolation. Building an agent like this on the current primitives means running the execution layer on a separate stack and using Genkit for the LLM calls only, which removes most of the reason to adopt Genkit.

A pluggable, session-scoped backend closes that gap. The same agent code runs against a local filesystem in development and an isolated remote sandbox in production, with no change to flow logic.

---

## 1. Problem statement

Agents that read/write files or run shell commands need a **storage and execution environment**. In development, that environment is usually the developer's machine. In production, it must be:

1. **Isolated** — agent commands must not run in the same process/container as the serving app; path-prefix checks on a shared host are not sufficient
2. **Session-scoped** — each conversation gets its own namespace; multi-tenant safe
3. **Durable across turns** — files written in turn 1 must exist in turn 2 (within the session)
4. **Swappable** — same agent code, different backend (local dev vs remote sandbox)
5. **Shared across capabilities** — filesystem, skills, shell, and context offloading use one namespace

Genkit currently has none of this as a first-class abstraction. Middleware talks directly to host `fs` and ad-hoc `child_process.exec`. Session stores persist **conversation JSON**, not **agent working memory**.

Without a backend layer, "deploy a coding agent" means either:

- Accepting **insecure, co-located execution** on Cloud Run/K8s — the model (or a user prompt) can run arbitrary shell commands as your service account, read secrets from the environment, and touch the host filesystem within path-prefix limits, or
- Hand-rolling custom tools per sandbox provider per app

---

## 2. How this problem manifests in Genkit today

### 2.1 Coding agent pattern (host-local)

The agents testapp coding agent demonstrates the gap:

```typescript
// filesystem middleware → direct fs calls on host
filesystem({ rootDirectory: WORKSPACE_DIR, allowWriteAccess: true })

// custom tool → child_process.exec on host
const { stdout, stderr } = await execAsync(input.command, {
  cwd: WORKSPACE_DIR,
  env: { ...process.env, HOME: WORKSPACE_DIR },
});
```

"Sandboxed workspace" means **path prefix checking**, not isolation. This is a **security** problem as much as a scaling one — even on a single instance, the agent runs with the same privileges as your server. In production at scale:

| Issue | Impact |
|-------|--------|
| Ephemeral container disk | Files lost on restart; inconsistent across replicas |
| Shared filesystem per instance | Concurrent sessions collide |
| `exec` in app process | Full blast radius if model is jailbroken or tricked |
| Session store on local disk (`FileSessionStore`) | Interrupt/resume breaks across pods |

The `run_shell` tool already gates commands: a fast model classifies each one as `safe` or `risky`, and risky commands interrupt for human approval ([`coding-agent.ts:113`](https://github.com/genkit-ai/genkit/blob/8ac4b2393fa867d683d745592e1d62a6ac16fc7a/js/testapps/agents/src/coding-agent.ts#L113)). That gate is probabilistic and bypassable, and even when it fires the command still runs in-process with full host privileges, so it is not isolation. The same tool runs with `env: { ...process.env, HOME: WORKSPACE_DIR }` ([`coding-agent.ts:168`](https://github.com/genkit-ai/genkit/blob/8ac4b2393fa867d683d745592e1d62a6ac16fc7a/js/testapps/agents/src/coding-agent.ts#L168)), passing the entire parent environment, including every API key, into the model-driven shell. A sandboxed execution environment with a default-deny env is what actually closes this.

### 2.2 Middleware is not backend-aware

`@genkit-ai/middleware` `filesystem()` requires `rootDirectory` and uses Node `fs` directly. There is no hook to substitute storage. Skills middleware reads `SKILL.md` from disk paths. Sub-agent delegation (`agents` middleware) spawns fresh sub-agent sessions that inherit the same host-local assumptions.

The root issue is a dependency pointing the wrong way: middleware depends on concrete `node:fs`, when it should depend on a storage and execution *interface* and not care where that runs. A middleware's job is to expose file and command *tools* to the model and shape the request, not to perform IO. Storage and execution belong behind an interface that the middleware is handed, with `node:fs` as just one implementation.

### 2.3 Session store ≠ working memory

`SessionStore` persists snapshots of `{ messages, custom, artifacts }` — full JSON documents per turn. This is orthogonal to where agent **files** live. Conflating them would bloat snapshots with file contents.

### 2.4 What works fine without backends

Agents that only call APIs — banking, weather, research orchestration, sub-agent routing — do not need this RFC. The gap is specific to **file/shell agents** and **deployed coding assistants**.

---

## 3. Prior art: LangChain Deep Agents

Deep Agents solves this with a **backend protocol** passed once to the agent harness. See:

- [Backends (JavaScript)](https://docs.langchain.com/oss/javascript/deepagents/backends)
- [Sandboxes (JavaScript)](https://docs.langchain.com/oss/javascript/deepagents/sandboxes)
- [Backends (Python)](https://docs.langchain.com/oss/python/deepagents/backends)

### 3.1 Architecture

```
createDeepAgent({ backend })
       │
       ├── createFilesystemMiddleware({ backend })  → ls, read_file, write_file, edit_file, glob, grep
       ├── createSummarizationMiddleware({ backend }) → offload large context to files
       ├── createSkillsMiddleware({ backend })        → read SKILL.md from backend paths
       └── execute tool (conditional)                 → only if backend implements SandboxBackendProtocol
```

**One `backend` reference**, threaded internally. Users do not pass it to each middleware.

### 3.2 Backend types

| Backend | Scope | Use case |
|---------|-------|----------|
| `StateBackend` (default) | Per LangGraph thread | Ephemeral virtual FS in agent state |
| `FilesystemBackend` | Host disk | Local dev |
| `StoreBackend` | LangGraph store | Persistent `/memories/` across threads |
| `CompositeBackend` | Routed by path prefix | Hybrid ephemeral + persistent |
| Sandbox backends | Remote container/VM | Production isolation |

### 3.3 Sandbox protocol

From [Sandboxes docs](https://docs.langchain.com/oss/javascript/deepagents/sandboxes): providers implement **`execute()`** plus **`uploadFiles()` / `downloadFiles()`**. `BaseSandbox` builds `ls`, `read`, `write`, `grep`, `glob` on top via POSIX shell — no Node/Python required inside the container.

The `execute` tool is **conditionally exposed**: if the backend doesn't implement `SandboxBackendProtocol`, the model never sees it.

### 3.4 Lifecycle scoping

Deep Agents recommends **thread-scoped sandboxes** (default): one isolated environment per conversation, resolved via thread metadata, with TTL for idle cleanup. See [Sandboxes (JavaScript)](https://docs.langchain.com/oss/javascript/deepagents/sandboxes) and [frontend sandbox integration](https://docs.langchain.com/oss/python/deepagents/frontend/sandbox) for the `getOrCreate` pattern.

Genkit's analogue is **session-scoped**: map `sessionId` (and optionally `snapshotId` metadata in the store) to a sandbox instance for the lifetime of the conversation.

---

## 4. Goals

1. **Backward compatible** — existing `filesystem({ rootDirectory })` agents work unchanged
2. **Single backend per session** — attached at agent run start, not per middleware / per generate
3. **Implicit middleware consumption** — `filesystem()`, `skills()`, etc. resolve backend from session context
4. **No backend unless configured** — an explicit `rootDirectory` creates a local backend for that middleware only; nothing silently defaults to `process.cwd()` or the app container disk. Sandbox is opt-in
5. **Protocol-first** — `genkit/backends` with a stable cross-runtime interface; provider packages are optional plugins
6. **Session-scoped lifecycle** — create on session start, destroy on end/TTL; map to `sessionId` / `snapshotId`
7. **Function-based factories** — backends configured via typed factory functions (`daytonaBackend()`), not stringly-typed config objects

## 5. Non-goals (v1)

- Managed/hosted sandbox platform — out of scope; users bring providers
- Replacing `SessionStore` with backend storage
- Automatic migration of all middleware to backend protocol (skills v1 can remain path-based with fallback)

---

## 6. Proposed design

### 6.1 Layering

```
┌─────────────────────────────────────────────────┐
│  Client (browser, CLI, API)                     │
└────────────────────┬────────────────────────────┘
                     │ agent.run({ init: { snapshotId } })
┌────────────────────▼────────────────────────────┐
│  defineAgent                                     │
│  • SessionStore (conversation snapshots)         │
│  • BackendFactory (working memory + execution)   │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│  Session (AsyncLocalStorage)                     │
│  • messages, custom, artifacts                   │
│  • backend: AgentBackend (NEW)                   │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│  Middleware (filesystem, skills, agents, …)      │
│  resolveBackend() at tool-call time              │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│  AgentBackend implementation                     │
│  LocalFilesystem │ InMemory │ Modal │ Daytona │ … │
└─────────────────────────────────────────────────┘
```

**SessionStore** and **Backend** are intentionally separate:

| Concern | Owner |
|---------|-------|
| Message history, interrupts, resume | `SessionStore` |
| Files, packages, build artifacts, shell | `AgentBackend` |

### 6.2 Backend resolution (the key rule)

At **tool execution time**, middleware calls:

```typescript
function resolveBackend(registry: Registry, middlewareConfig?: FilesystemConfig): AgentBackend {
  const session = getCurrentSession(registry);

  // 1. A configured session backend wins. An explicit rootDirectory alongside it
  //    is a conflict (it would silently drop a sandboxed agent back onto the host),
  //    so error rather than override. See Open question 8.
  if (session?.backend) {
    if (middlewareConfig?.rootDirectory) {
      throw new BackendConflictError(
        'filesystem({ rootDirectory }) conflicts with a configured session backend; remove one'
      );
    }
    return session.backend;
  }

  // 2. No session backend: an explicit rootDirectory builds a local backend for
  //    this middleware only (today's behavior).
  if (middlewareConfig?.rootDirectory) {
    return new LocalFilesystemBackend(middlewareConfig);
  }

  throw new BackendNotConfiguredError(
    'filesystem requires defineAgent({ backend }) or an explicit rootDirectory'
  );
}
```

**Not** resolved at middleware `instantiate()` time (which runs per `generate()`). The backend instance lives on the **session**, created once when the agent run starts.

### 6.3 Session backend lifecycle

```typescript
// In definePromptAgent, when agent.run() starts:
async function attachSessionBackend(session: Session, config: AgentConfig) {
  if (!config.backend) return; // no factory → middleware uses rootDirectory fallback

  const ctx: BackendContext = {
    sessionId: session.sessionId,
    snapshotId: init.snapshotId,
    context: getContext(),
  };

  const backend =
    typeof config.backend === 'function'
      ? await config.backend(ctx)
      : config.backend; // pre-built instance (tests only)

  session.attachBackend(backend);
  // Teardown is intentionally not wired to a session event. There is no
  // `session.on('end')` (Session only emits `artifactAdded` / `artifactUpdated`),
  // and detached runs outlive invocation-return, so destroying on return would
  // kill a still-running flow. Cleanup is left to provider TTL / auto-archive or
  // an explicit caller-invoked `destroy()`. See Open question 9.
}
```

For **sandbox reuse across turns**, the factory uses `sessionId` / store metadata to call `getOrCreate` on the provider (same pattern as [Deep Agents frontend sandbox docs](https://docs.langchain.com/oss/python/deepagents/frontend/sandbox)).

### 6.4 Default = current filesystem

"No backend on defineAgent" + `filesystem({ rootDirectory })` = **exactly today's behavior**. No migration required.

Unified local backend (optional sugar):

```typescript
import { localFilesystemBackend } from 'genkit/backends';

defineAgent({
  backend: localFilesystemBackend({ rootDirectory: './workspace' }),
  use: [filesystem(), skills()],
});
```

---

## 7. Proposed API

### 7.1 Core protocol (`genkit/backends`)

```typescript
/** Structured results — errors as values, not throws, for LLM-friendly tool output */
interface ReadResult {
  error?: string;
  content?: string | Uint8Array;
  mimeType?: string;
}

interface WriteResult { error?: string; path?: string; }

interface ExecuteResult {
  stdout: string;
  stderr: string;            // kept separate, matching the existing run_shell tool
  exitCode: number | null;
  truncated: boolean;
  error?: string;            // infra failure (e.g. sandbox unreachable), distinct from a non-zero exitCode
}

interface ExecuteOptions {
  cwd?: string;
  timeoutMs?: number;
  env?: Record<string, string>;  // default-deny: only the keys passed here are exposed
  maxOutputBytes?: number;
  signal?: AbortSignal;          // cancellation
}

interface AgentBackend {
  readonly id: string;

  ls(path: string): Promise<LsResult>;
  read(filePath: string, offset?: number, limit?: number): Promise<ReadResult>;
  write(filePath: string, content: string): Promise<WriteResult>;
  edit(filePath: string, oldText: string, newText: string, replaceAll?: boolean): Promise<EditResult>;
  grep(pattern: string, path?: string, glob?: string | null): Promise<GrepResult>;
  glob(pattern: string, path?: string): Promise<GlobResult>;

  destroy?(): Promise<void>;
}

interface SandboxBackend extends AgentBackend {
  execute(command: string, opts?: ExecuteOptions): Promise<ExecuteResult>;
}

function isSandboxBackend(b: AgentBackend): b is SandboxBackend {
  return typeof (b as SandboxBackend).execute === 'function';
}
```

The protocol is shown here in its TypeScript binding, but it is meant to be language-neutral: `Uint8Array` is "bytes," `AbortSignal` is "a cancellation token," and `isSandboxBackend` is "does this backend implement the sandbox capability" (an interface assertion in Go/Dart). Each runtime binds these to its own idioms; the shared semantics (default-deny env, path confinement, errors-as-values, offset/limit reads) are the actual contract.

### 7.2 Backend factory on `defineAgent`

Backends are configured with **factory functions**, not config objects. A factory receives session context and returns (or resolves to) an `AgentBackend` instance. This keeps configuration typed, explicit, and tree-shakeable.

```typescript
interface BackendContext {
  sessionId: string;
  snapshotId?: string;
  context?: ActionContext;
}

/** Called once per agent session when the run starts. */
type BackendFactory = (
  ctx: BackendContext
) => AgentBackend | Promise<AgentBackend>;

interface AgentConfig<State> extends PromptConfig {
  /**
   * Optional. When omitted, middleware uses explicit rootDirectory (legacy).
   * Prefer a factory function from `genkit/backends` or a provider package.
   */
  backend?: BackendFactory;
}
```

Provider packages export named factories:

```typescript
// genkit/backends — built-ins
import { localFilesystemBackend, memoryBackend } from 'genkit/backends';

// genkit/backends/<provider> — optional provider plugins
import { daytonaBackend } from 'genkit/backends/daytona';
import { modalBackend } from 'genkit/backends/modal';

defineAgent({
  backend: daytonaBackend({ language: 'typescript' }),
  // ...
});

defineAgent({
  backend: modalBackend({ image: 'node:22-slim', ttlSeconds: 3600 }),
  // ...
});
```

Each factory closes over its options and implements session-scoped `getOrCreate` internally (using `ctx.sessionId` and store metadata for sandbox reuse across turns).

**Not supported:** stringly-typed config objects such as `{ type: 'daytona', ... }` on `defineAgent`. If a project scaffold generates agent definitions, it should emit code that imports and calls these factories — not pass JSON to the runtime.

### 7.3 Session extension

```typescript
class Session<S> {
  attachBackend(backend: AgentBackend): void;
  getBackend(): AgentBackend | undefined;
}

function getCurrentBackend(registry: Registry): AgentBackend | undefined {
  return getCurrentSession(registry)?.getBackend();
}
```

### 7.4 Middleware changes

**`filesystem()` stops doing IO and becomes a thin tool-wiring layer over the injected backend.** It declares the tool schemas (`list_files`, `read_file`, etc.), validates, and delegates every read/write to the `AgentBackend` resolved from the session. No `node:fs` in the middleware. `node:fs` lives only inside `LocalFilesystemBackend`.

This is a real rewrite, not a config flag. Each filesystem tool today imports `node:fs` and performs IO inside its own execute function (`read_file`, `write_file`, `list_files`, `search_and_replace`), so each is rewritten to call the backend instead. The middleware keeps its non-IO orchestration: `read_file`'s image-to-media-part conversion and its deferred message-queue injection stay in the middleware, and `backend.read()` only returns bytes plus a mime type. Tool names are independent of the protocol method names and can be chosen for clarity, so aligning the wire names with the protocol (e.g. naming the edit tool `edit`) is on the table.

```typescript
// Current — unchanged behavior. `rootDirectory` is now sugar that constructs a
// LocalFilesystemBackend for this middleware, not a special fs path in middleware.
filesystem({ rootDirectory: './workspace', allowWriteAccess: true })

// New — uses the session backend (local or sandbox), with no IO knowledge in the call site
filesystem({ allowWriteAccess: true })

// NOTE (open question, see §6.2 / Open questions): allowing an explicit rootDirectory to
// override a configured session sandbox is a footgun — it silently drops a sandboxed agent
// back onto the host. The proposed fail-safe is: session sandbox wins, and a conflicting
// rootDirectory errors rather than overriding. Left as a design decision.
filesystem({ rootDirectory: '/tmp/scratch' })
```

Tool names can align with Deep Agents (`read_file` / `write_file` / `edit_file` / `grep` / `glob`) or with the protocol method names.

**`execute` tool** — injected by a new `sandbox` middleware (or auto-injected when session backend is sandbox):

```typescript
// Only registered when isSandboxBackend(session.getBackend())
// Equivalent to coding-agent's custom run_shell, but namespace-aligned with filesystem
```

**`skills()` reads from the session backend:**

```typescript
skills({ paths: ['/skills/'] })  // reads skills from the session backend
```

Per Open question 6, skills live in the backend and the user provisions them there. `paths` is the only option.

### 7.5 Code examples

#### Example A: Today (unchanged)

```typescript
export const codingAgent = ai.defineAgent({
  name: 'codingAgent',
  store: new FileSessionStore('./.snapshots'),
  use: [
    toolApproval({ approved: ['list_files', 'read_file', ...] }),
    filesystem({ rootDirectory: './workspace', allowWriteAccess: true }),
    skills({ skillPaths: ['./skills'] }),
  ],
  tools: [runShell],  // ad-hoc host exec — still user's responsibility
});
```

#### Example B: Local backend, unified (new)

```typescript
import { localFilesystemBackend } from 'genkit/backends';

export const codingAgent = ai.defineAgent({
  name: 'codingAgent',
  backend: localFilesystemBackend({ rootDirectory: './workspace' }),
  store: firestoreSessionStore,
  use: [
    toolApproval({ ... }),
    filesystem({ allowWriteAccess: true }),  // no rootDirectory
    skills({ paths: ['/skills/'] }),          // reads from backend
  ],
});
```

#### Example C: Remote sandbox (production)

```typescript
import { daytonaBackend } from 'genkit/backends/daytona';

export const codingAgent = ai.defineAgent({
  name: 'codingAgent',
  backend: daytonaBackend({
    language: 'typescript',
    autoDeleteInterval: 3600,
  }),
  store: firestoreSessionStore,
  use: [
    toolApproval({ ... }),
    filesystem({ allowWriteAccess: true }),
    skills({ paths: ['/skills/'] }),
    sandbox(),
  ],
});
```

Modal, Deno, LangSmith, etc. follow the same pattern — one import, one factory call:

```typescript
import { modalBackend } from 'genkit/backends/modal';

backend: modalBackend({ image: 'node:22-slim', ttlSeconds: 3600 }),
```

#### Example D: Orchestrator unchanged

Sub-agents invoked via `agents` middleware get their **own** session (`init: {}`). Each sub-agent's backend factory runs independently — researcher can be API-only (no backend), coder can have a sandbox.

```typescript
export const orchestrator = ai.defineAgent({
  use: [
    agents({ agents: ['researcher', 'coder'], maxDelegations: 5 }),
  ],
});
```

---

## 8. Provider landscape and language support

Sandbox **providers** are generally **language-agnostic at the agent-tool level** (shell + files in a Linux environment). **Provider SDKs** for creating/managing sandboxes vary by language.

### 8.1 Provider SDK and Deep Agents adapter matrix

Deep Agents **backend adapter packages** (LangChain wrappers that implement `BackendProtocol`) exist for **JS and Python** only. **Provider SDKs** for creating/managing sandboxes are broader — several vendors ship official **Go** clients; none ship an official **Dart** sandbox SDK today.

| Provider | JS (Deep Agents) | Python (Deep Agents) | Go SDK | Dart SDK | Isolation | Notes |
|----------|-------------------|----------------------|--------|----------|-----------|-------|
| **Modal** | `@langchain/modal` | `langchain-modal` | [`github.com/modal-labs/modal-client/go`](https://modal.com/docs/guide/sdk-javascript-go) | — | Container (serverless) | Official JS + Go; Docker images; GPU support |
| **Daytona** | `@langchain/daytona` | `langchain-daytona` | [`github.com/daytonaio/daytona/libs/sdk-go`](https://www.daytona.io/docs/en/go-sdk/) | — | Container (VM) | Platform SDKs: [TS, Python, Ruby, Go, Java](https://www.daytona.io/docs/sandboxes.md) |
| **Deno Deploy** | `@langchain/deno` | — | — | — | microVM | Deep Agents adapter is npm-only |
| **LangSmith** | `LangSmithSandbox` | `langsmith.sandbox` | [`langsmith-go`](https://github.com/langchain-ai/langsmith-go) (REST/tracing only) | — | Platform-managed | [Sandbox API](https://docs.langchain.com/langsmith/sandbox-sdk) is Python + TS only |
| **node-vfs** | `@langchain/node-vfs` | — | — | — | In-process VFS | Dev/test; not production isolation |
| **quickjs** | `@langchain/quickjs` | — | — | — | WASM REPL | JS eval; not full shell |
| **LocalShell** | `LocalShellBackend` | Python equivalent | — | — | Host (unsafe) | Dev only |

**Go takeaway:** Modal and Daytona both have **first-party Go SDKs** with sandbox create/exec/filesystem APIs — viable targets for a native `go/backends/*` package.

**Dart takeaway:** No major sandbox provider publishes a Dart SDK. When a vendor client is unavailable, use a remote Genkit server that owns the backend, or provider REST via `package:http`.

### 8.2 Genkit runtime language matrix

| Genkit runtime | Backend package | Sandbox provider SDKs | Notes |
|----------------|-----------------|------------------------|-------|
| **JavaScript/TypeScript** | `genkit/backends/*` | Modal (npm), Daytona (npm), Deno, LangSmith | Thin-wrap deepagentsjs provider implementations |
| **Go** | `go/backends/*` | Modal + Daytona official Go SDKs | Native `daytonaBackend` / `modalBackend` via provider Go clients |
| **Python** | `genkit/backends/*` (Python) | Modal + Daytona via `langchain-*` adapters | Reuse `langchain-modal` / `langchain-daytona` where applicable |
| **Dart** | `genkit/backends/*` (Dart) | None official from providers | Same `defineAgent({ backend })` pattern; provider REST or remote Genkit server if a vendor lacks a Dart client |

**Important distinction:** A provider's **platform SDK language list** ≠ Deep Agents **backend adapter** availability. Example: Daytona documents [TS, Python, Ruby, Go, Java](https://www.daytona.io/docs/sandboxes.md) for sandbox management, but `@langchain/daytona` / `langchain-daytona` are the only Deep Agents adapters today. Deno, node-vfs, and quickjs remain **JS-only** at the Deep Agents adapter layer.

For **Go**, Modal and Daytona ship first-party sandbox clients — implement `AgentBackend` against [`modal-client/go`](https://github.com/modal-labs/modal-client/tree/main/go) or [`sdk-go`](https://www.daytona.io/docs/en/go-sdk/) directly, or use an HTTP gateway for providers without a Go client (e.g. LangSmith).

For **Dart**, no major sandbox provider publishes a Dart SDK today. When a vendor client is unavailable, apps can call a remote Genkit deployment that owns the backend (type-safe via `defineRemoteAction`) or use provider REST via `package:http`.

### 8.3 Recommended package layout

```
genkit/backends              # JS/TS — core protocol + localFilesystemBackend, memoryBackend
genkit/backends/modal        # modalBackend()
genkit/backends/daytona      # daytonaBackend()
genkit/backends/deno         # denoBackend()
genkit/backends/langsmith    # langsmithSandboxBackend() — optional

go/backends                  # Go — same protocol + provider factories
```

Published as `@genkit-ai/backends` with subpath exports (JS/TS), or re-exported from `genkit/backends` in the main package. Equivalent packages per runtime follow the same factory naming.

JS/TS can reuse deepagentsjs provider implementations (`deepagentsjs/libs/providers/*`) with thin Genkit adapter glue. Python can reuse `langchain-modal` / `langchain-daytona`. Go implements the same protocol against Modal/Daytona Go SDKs directly.

---

## 9. Migration and compatibility

| Existing code | Behavior after RFC |
|---------------|-------------------|
| `filesystem({ rootDirectory })` only | **Unchanged** — local backend via middleware config |
| `defineAgent` without `backend` | **Unchanged** |
| Custom `run_shell` tool | **Unchanged** — but docs recommend `sandbox()` middleware when using sandbox backend |
| `FileSessionStore` on disk | **Unchanged** — still discouraged for prod; orthogonal to backend |
| New sandbox agents | Add `backend` factory; drop `rootDirectory` from middleware; remove custom shell tool |

---

## 10. Open questions

1. **Execute tool ownership** — separate `sandbox()` middleware vs auto-inject when backend is sandbox?
2. **Backend metadata in SessionStore** — store `sandboxId` in snapshot `custom` vs separate index table?
3. **Sub-agent backends** — inherit parent backend, own sandbox per delegation, or config per sub-agent in `agents()` middleware?
4. **Composite routing** — v1 or defer `/memories/` vs `/workspace/` path prefixes?
5. **Artifact middleware** — relationship to backend files (deliverables vs working memory)?
6. **Cached backend reads, keying and auth.** Two parts. (a) *Hazard the redesign must avoid:* today `skillCache` / `scanPromise` in `skills.ts` are declared inside the middleware's `instantiate` body, which runs once per `generate()` call, so the cache is turn-scoped and there is no cross-session leak (the `filesystem.ts` middleware notes this explicitly: "Middleware is instantiated once per top generate call, so it's ok to keep state here"). The hazard is in the *new* design: if backend reads get cached at session or backend scope to avoid re-scanning across turns, that cache must be keyed on `backend.id + session.id + paths`, or one session's scan will leak into another. Do not hoist the cache without that key. (b) *Open question, with a recommended direction:* skills should be read straight from the backend, which is also where the user provisions them (clone a repo, copy files, or generate them into the workspace). On that model Genkit needs no skill cache at all, the backend or provider owns any caching, and the leak in (a) disappears because there is no Genkit-side memo. The open part is confirming that direction and defining how `skills()` (and similar middleware) discovers skills from the backend rather than from host paths. Related: should a sandbox be bound to the caller's auth, so a `snapshotId` resumed by a different principal cannot reuse it?
7. **Command-output streaming.** Does `execute` stream incremental output (and through what channel), or is buffered `stdout` / `stderr` enough for v1?
8. **Backend resolution precedence.** When both a session sandbox and an explicit middleware `rootDirectory` are present, which wins? *Recommended default:* the session sandbox wins and a conflicting `rootDirectory` raises an error (or a loud warning), rather than silently dropping a sandboxed agent back onto the host. Letting `rootDirectory` win is a silent isolation downgrade.
9. **Backend lifecycle ownership.** *Recommended stance:* Genkit creates the backend via the session-start factory and does not manage its lifecycle beyond an optional `destroy?()` hook; idle cleanup is the provider's job (TTL / auto-archive) or the caller's. Two facts to design around: there is no `session.on('end')` event (the end-of-run signal is the `finally` after `runWithSession`, where `maybeSnapshot('invocationEnd')` fires), and agents can detach (the run returns early while the flow continues in the background), so invocation-return is not the same as work-complete. Tearing a sandbox down on return would kill a still-running detached flow.

---

## 11. References

- [Deep Agents Backends (JS)](https://docs.langchain.com/oss/javascript/deepagents/backends)
- [Deep Agents Sandboxes (JS)](https://docs.langchain.com/oss/javascript/deepagents/sandboxes)
- [Deep Agents Backends (Python)](https://docs.langchain.com/oss/python/deepagents/backends)
- [DaytonaSandbox integration (Python)](https://docs.langchain.com/oss/python/integrations/sandboxes/daytona)
- [Daytona Sandboxes docs](https://www.daytona.io/docs/sandboxes.md)
- Genkit coding agent: `js/testapps/agents/src/coding-agent.ts`
- Genkit filesystem middleware: `js/plugins/middleware/src/filesystem.ts`
- Genkit session model: `js/ai/src/session.ts`, `js/ai/src/agent.ts`
- Deep Agents JS reference: `deepagentsjs/libs/deepagents/src/backends/`
