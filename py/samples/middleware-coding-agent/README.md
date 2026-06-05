# middleware-coding-agent

Interactive coding-agent REPL that wires up the
[`Filesystem`](../../plugins/middleware/src/genkit/plugins/middleware/_filesystem.py),
[`Skills`](../../plugins/middleware/src/genkit/plugins/middleware/_skills.py),
and [`ToolApproval`](../../plugins/middleware/src/genkit/plugins/middleware/_tool_approval.py)
middleware against a sandboxed workspace.

## What's here

```
middleware-coding-agent/
├── src/main.py                # interactive REPL
├── skills/
│   ├── python-expert/SKILL.md       # house style for editing Python
│   └── test-writer/SKILL.md         # house style for writing pytest tests
└── workspace/                  # sandbox the agent reads, writes, edits in
                                # (created on first run; contents gitignored)
```

The model gets:

- the contents of `workspace/` via `Filesystem(root_dir=…, allow_write_access=True)` —
  `list_files`, `read_file`, `write_file`, `edit_file`, all confined to that
  directory.
- a system prompt listing the two skills, plus a `use_skill` tool it calls
  to pull in the full `SKILL.md` content on demand.
- `ToolApproval(allowed_tools=['read_file', 'list_files', 'use_skill'])` —
  read-only tools run without prompting; anything that can mutate the
  workspace (`write_file`, `edit_file`) interrupts and waits for your
  `y/N` from the CLI before resuming.

## Run it

```bash
cd py/samples/middleware-coding-agent
GEMINI_API_KEY=... genkit start -- uv run src/main.py
```

Type a request at the REPL prompt in your terminal (e.g. `build a tiny
priority queue module with push/pop/peek and pytest tests`), hit enter,
and approve each write the agent proposes. Conversation history persists
across turns until you type `exit`.

If you want the agent to fix or extend an existing file instead of
starting from scratch, drop the file into `workspace/` first and reference
it by name in your prompt.

## Resetting between runs

The agent edits `workspace/` in place. To start over:

```bash
rm -rf py/samples/middleware-coding-agent/workspace/*
```
