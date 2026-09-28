# Genkit in a Box testapp

One entry file defines a `runShell` tool and boxes it in a local OS sandbox
(`execRunner({ self: true, isolate: localSandbox() })`), while the agent and
the model API key stay in the main (unsandboxed) process. Only `runShell` calls
cross into the box.

## Run

```bash
export GEMINI_API_KEY=...
pnpm i
pnpm genkit:dev
```

Then, in the Dev UI:

- Call the `runInBox` flow with `{ "cmd": "echo $$" }`. The pid it prints is
  the box process, not the main one.
- Chat with the `codingAgent` agent; when it uses `runShell`, that call executes
  inside the box.
- Open a trace: the `runShell` span is marked as boxed, and `box:traceId` links
  it to the box's own trace.

## What the sandbox blocks

It differs by platform:

- **macOS** (`sandbox-exec`): filesystem writes are blocked except `/tmp`.
  Reads and network are still open. Try `echo hi > ~/nope.txt` and watch it
  fail, then note that `cat ~/.aws/credentials` still works.
- **Linux** (`bubblewrap`): the box sees only a read-only system plus an
  ephemeral `/tmp`; your home dir and repo are not even mounted, so they can't
  be read or written. Network is still open.

These are dev-time guardrails against accidents, not containment for hostile
code: network egress is open and there are no CPU/memory limits. See the
`@genkit-ai/box` README ("Isolation levels") for the full picture.

On a platform without a local sandbox (e.g. Windows), `localSandbox()` throws.
