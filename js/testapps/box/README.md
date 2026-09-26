# Genkit in a Box testapp

One entry file defines a `runShell` tool and boxes it (`execRunner({ self: true })`),
while the agent and the model API key stay in the main process. Only
`runShell` calls cross into the box.

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
