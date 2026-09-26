# Genkit [in a] Box (`@genkit-ai/box`)

Run Genkit actions (tools, flows, agents) somewhere other than your main
process, and call them as if they were local. "Somewhere" can be a subprocess,
a local OS sandbox, a container, or a remote runtime. You swap a `runner`, not
your code.

Two reasons to box:

- **Isolation**: you don't fully trust the code (e.g. a shell tool an LLM
  drives). Box the dangerous tool; keep the agent and model keys outside.
- **Relocation**: run an action elsewhere (another process, another language,
  another machine) behind the same call site.

> Status: experimental.

## Install

```bash
npm i @genkit-ai/box
```

## Proxies

A box hands out proxies: real Genkit actions that forward each call to the box.
They compose anywhere a local action does (`tools: [...]`, the registry, other
flows).

```ts
import { box } from '@genkit-ai/box';
import { genkit, z } from 'genkit';

const ai = genkit({ plugins: [googleAI()] });
const myBox = box(ai, { runner });

// Same language: hand over the real action, no schema restatement.
const boxedRunShell = myBox.fromTool(runShell);

// Different language (or no local copy): declare the shape.
const boxedSearch = myBox.tool({
  name: 'search',
  inputSchema: z.object({ query: z.string() }),
  outputSchema: z.object({ hits: z.array(z.string()) }),
});

export const codingAgent = ai.defineAgent({
  name: 'codingAgent',
  model: googleAI.model('gemini-flash-latest'),
  tools: [boxedRunShell, boxedSearch],
});
```

Flows work the same way (`myBox.flow(spec)`, `myBox.fromFlow(action)`), and
streamed chunks flow back through the proxy.

`.tool()`/`.flow()`/`.fromTool()`/`.fromFlow()` return unregistered proxies.
The `define*` variants (`defineTool(spec)`, `defineFromTool(action, { name })`,
...) also register the proxy so it shows up in the Dev UI. `defineFrom*`
proxies must be renamed, since the original already occupies its name.

## Agents

Box a whole agent (model calls, tools, conversation state) and register it,
so it is visible and chattable in the Dev UI like a local one:

```ts
// The agent lives only in the box (separate entry, or another language):
// declare what the Dev UI needs to know.
export const codingAgent = myBox.defineAgent<CodingState>({
  name: 'codingAgent',
  stateManagement: 'server', // must match the boxed agent (has a store)
  abortable: true,
  stateSchema: CodingStateSchema,
});

// Same language: hand over the real agent; its metadata is copied.
export const boxedCoder = myBox.defineFromAgent(codingAgent, {
  name: 'boxedCoder',
});

// Either way you get an Agent<State>:
const res = await codingAgent.chat({ sessionId }).send('fix the failing test');
```

`defineAgent` registers `/agent/<name>` with its `agent-snapshot` and
`agent-abort` companions. Each input is one boxed turn. When a caller streams
several inputs into one invocation, the proxy runs them turn by turn and
threads the snapshot (or client state) between them; the box then records one
trace per turn.

For an unregistered handle, `myBox.agent({ name, context })` returns just the
`AgentAPI`.

### One box per session

Route agent calls by session with `sessionIdOf`. It reads the session from the
agent's init (what the Dev UI and `chat({ sessionId })` send) and from
snapshot lookups:

```ts
import { box, execRunner, sessionIdOf } from '@genkit-ai/box';

const agentBox = box(ai, {
  runner: execRunner({ cmd: 'tsx src/boxed-agent.ts' }),
  route: (req, ctx) => String(ctx?.sessionId ?? sessionIdOf(req) ?? 'new'),
  retention: { idle: 10 * 60_000 },
});
```

Calls that only carry a `snapshotId` (a resume, a snapshot read, an abort)
still land on the session's box: a registered agent remembers which session
each snapshot it returned belongs to.

## Lifecycle: route + retention

`route` decides which box a call goes to; `retention` decides how long an idle
box lives. Presets ship as functions:

```ts
import { box, perRequest } from '@genkit-ai/box';

box(ai, { runner }); // singleton (default): one box, never reclaimed
box(ai, { runner, route: perRequest }); // fresh box per call, reclaimed after

// Session-scoped: you supply the key. Custom routes default to a 5 minute
// idle window.
box(ai, {
  runner,
  route: (req, ctx) => String(ctx?.sessionId ?? 'default'),
  retention: { idle: 10 * 60_000 },
});
```

A route sees the request about to be dispatched (`req.key`, `req.input`,
`req.init`) as well as the caller's context.

## Runners

A runner owns where boxes run and the routing key to box mapping:

```ts
interface BoxRunner {
  readonly name: string;
  attach?(box: { readonly id: string }): void;
  acquire(key: string, signal?: AbortSignal): Promise<BoxConnection>;
  release(key: string): Promise<void>;
  close(): Promise<void>;
}
```

Runners don't speak reflection themselves; they hand the core a
`BoxConnection` built from one of two shared clients:

- `ReflectionHost`: the reflection **v2** manager. Runtimes dial in over
  WebSocket (`GENKIT_REFLECTION_V2_SERVER`) and must present its per-host
  `secret` (`GENKIT_REFLECTION_SECRET_TOKEN`) in `register`.
- `ReflectionClientV1`: a client for the **v1** HTTP API, for runtimes that
  serve reflection themselves (e.g. in a container with a published port).

```ts
const client = new ReflectionClientV1('http://127.0.0.1:54321', { secret });
await client.waitForReady();
await client.runAction({ key: '/tool/runShell', input: { cmd: 'ls' } });
```

## Subprocesses: `execRunner()`

Runs each box as a local child process that dials back into a per-runner
reflection host.

```ts
import { box, execRunner } from '@genkit-ai/box';

// Self mode: re-run *this same program* as the box. One file defines the tool
// and boxes it; the body runs in the child.
const myBox = box(ai, { runner: execRunner({ self: true }) });
const boxedRunShell = myBox.fromTool(runShell);

// Separate entry: stronger code isolation, and it can be another language.
box(ai, { runner: execRunner({ cmd: 'tsx src/boxed.ts' }) });
```

The child inherits your environment (API keys included), so a plain subprocess
is for trusted code and relocation, not containment. Boxes nest: a boxed agent
can itself box a tool, each level in its own process. Call `box.close()` when
you are done; it stops every child the runner started.

### Local sandboxes: `isolate`

`isolate` jails the child with an OS sandbox, without changing anything else:

```ts
import { box, execRunner, localSandbox } from '@genkit-ai/box';

// seatbelt on macOS, bubblewrap on Linux; throws elsewhere (e.g. Windows)
box(ai, { runner: execRunner({ self: true, isolate: localSandbox() }) });
```

Or pick one explicitly: `sandboxExec()` (macOS), `bubblewrap()` (Linux).

### Isolation levels (read this before trusting it)

The local sandboxes are **dev-time guardrails against accidents and casual
misbehavior, not containment for hostile code.** Know exactly what each one
restricts:

| Option | FS read | FS write | Network egress | CPU/mem/syscalls |
| --- | --- | --- | --- | --- |
| none (no `isolate`) | full | full | full | none |
| `sandboxExec()` (macOS) | **full** | blocked except `/tmp` | **full** | none |
| `bubblewrap()` (Linux) | confined: ro `/usr /bin /lib /lib64 /etc` + tmpfs; home/repo/creds not present | `/tmp` tmpfs only | **full** | none |

- **Network egress is open on both.** The box must reach the reflection host
  on loopback, and the default profiles allow *all* network, not just loopback.
  A boxed tool can call out to the internet and exfiltrate.
- **macOS restricts writes only.** With `sandboxExec()` a boxed tool can still
  read anything you can (`~/.aws/credentials`, `~/.ssh`, source, env-bearing
  dotfiles). `sandbox-exec` is also deprecated by Apple (still works on current
  macOS, prints a warning).
- **Linux confines the filesystem view.** With `bubblewrap()` your home dir,
  repo, and credentials are simply not mounted.
- **No resource limits.** No CPU/memory/pid caps and no seccomp filtering. A
  boxed tool can peg the CPU or fork-bomb.
- **The child still inherits your environment**, API keys included. Pass a
  separate `cmd` entry and keep secrets out of its env if that matters.

For untrusted or hostile code, use `podmanRunner()` (below). The local
sandboxes are not containment.

Tightening the defaults:

```ts
// Linux: bind only your repo read-only, keep an ephemeral /tmp.
bubblewrap({ roBind: ['/path/to/repo'], tmpfs: ['/tmp'] });

// macOS: supply a full custom seatbelt profile.
sandboxExec({ profile: '(version 1)\n(deny default)\n(allow network* (local ip))' });
```

`bubblewrap({ unshareNet: true })` isolates the network namespace, but that
currently breaks the reflection dial-back (the host binds the host's loopback,
not the namespace's), so it is not usable for local boxes yet.

## Containers: `podmanRunner()`

For untrusted code, run the box in a container. The container runner is a
**runner**, not an `isolate:` provider, because it flips the reflection
direction: the box serves the **v1** HTTP reflection API and the runner
publishes that port to the host loopback (`-p`). Nothing dials out of the
container.

```ts
import { box, podmanRunner } from '@genkit-ai/box';

const myBox = box(ai, {
  runner: podmanRunner({
    image: 'node:22-slim',
    cmd: 'node dist/boxed.js', // runs INSIDE the container
    extraArgs: ['--memory=512m', '--pids-limit=256'],
  }),
});
```

This is the first option with real containment:

- **Egress is blocked by default.** Boxes run on an auto-created `--internal`
  podman network, so a boxed tool cannot reach the internet. Published ports
  still work, so reflection is unaffected. Opt out with `network: 'bridge'`.
- **Resource caps actually exist.** Pass `--memory`, `--cpus`, `--pids-limit`
  via `extraArgs`; the local sandboxes cannot do this at all.
- **Nothing is inherited.** A container gets no host env and no host
  filesystem beyond the project mount. Your `GEMINI_API_KEY` does not leak into
  the box unless you pass it explicitly via `env:`.
- **`docker` works too**, via `engine: 'docker'`.

### Why v1 here, and not v2

Everywhere else box uses reflection v2, where the runtime dials *out* to a
WebSocket server. v1 is deliberate here: **blocking egress and dialing back out
are mutually exclusive**, because they are the same route. On an `--internal`
network `host.containers.internal` still resolves but cannot be connected to,
while published ports keep working (the engine injects them from the host
side).

| | v2 (dial out) | v1 + `-p` (dial in) |
| --- | --- | --- |
| normal bridge | works | works |
| **egress blocked (`--internal`)** | **impossible** | **works** |

### Self-entry mode

`self: true` runs *this same program* in the container. The project is mounted
at its **own absolute path**, so `process.argv` and loader flags carry over
verbatim and relative symlinks still resolve.

```ts
podmanRunner({
  image: 'node:22-slim',
  self: true,
  modulesVolume: 'genkit-box-modules',
});
```

The catch is `node_modules`: host-installed deps are built for the host OS/arch
and **cannot load in a Linux container** (esbuild and other native addons fail
outright). `modulesVolume` shadows the mounted `node_modules` with a named
volume holding Linux-built deps. Prime it once:

```bash
podman run --rm -v "$PWD:$PWD" -w "$PWD" \
  -v genkit-box-modules:"$PWD/node_modules" node:22-slim npm ci
```

### Reflection contract

The runner starts each container with:

- `GENKIT_REFLECTION_PORT=3100`: bind exactly that port (it is what the runner
  publishes).
- `GENKIT_REFLECTION_HOST=0.0.0.0`: published ports arrive on the container's
  eth0, so the default loopback bind would be unreachable.
- `GENKIT_REFLECTION_SECRET_TOKEN`: a fresh random secret per container. The
  runner sends it on every call, and the host side of the port is bound to
  `127.0.0.1` as well.

> Cross-language boxes need a runtime that honors these variables.

## Tracing

Each proxied call records a span in the caller's trace, marked with
`genkit:metadata:box`. The box runs the action in its own trace; its trace id
is recorded on the proxy span as `box:traceId`.

## License

Apache 2.0
