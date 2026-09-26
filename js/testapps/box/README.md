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

On a platform without a local sandbox (e.g. Windows), `localSandbox()` throws;
use the container variant below.

## Container variant (`src/index-podman.ts`)

Same demo, but the boxed side runs in a podman container instead of a local OS
sandbox. Needs podman (`podman machine start` on macOS).

```bash
export GEMINI_API_KEY=...
pnpm genkit:dev:podman
```

Call `runInBox` with `{ "cmd": "uname -a" }` and note it reports **Linux**, not
your host, because the tool body really executed in the container.

What is different from the local-sandbox version:

- **Egress is blocked.** Boxes run on an auto-created `--internal` podman
  network. Try `{ "cmd": "getent hosts example.com || echo BLOCKED" }`.
- **Resource caps are real** (`--memory=512m`, `--pids-limit=256`).
- **No host env leaks in.** Try
  `{ "cmd": "printenv GEMINI_API_KEY || echo NO_KEY_IN_BOX" }`; the container
  inherits nothing, so your key is not there.
- **Two files, not one.** Self mode cannot work here: the host's node binary and
  darwin-built `node_modules` (esbuild especially) cannot run in a Linux
  container. So the boxed side is a separate compiled entry
  (`src/boxed-podman.ts` -> `lib/`), and the container runs the compiled JS
  rather than `tsx`. The tool is declared from a spec (`myBox.defineTool`)
  because there is no local action to hand over.

If the host process is killed before `close()`, strays are labelled:

```bash
podman rm -f $(podman ps -aq --filter label=genkit-box)
```
