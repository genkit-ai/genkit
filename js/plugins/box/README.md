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

A boxed agent is available as an `AgentAPI`:

```ts
const agent = myBox.agent<CodingState>({
  name: 'codingAgent',
  context: { sessionId },
});
const res = await agent.chat({ sessionId }).send('fix the failing test');
```

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

## Tracing

Each proxied call records a span in the caller's trace, marked with
`genkit:metadata:box`. The box runs the action in its own trace; its trace id
is recorded on the proxy span as `box:traceId`.

## License

Apache 2.0
