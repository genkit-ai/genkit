# Branching samples

These examples show how snapshot forking works in-process. They use a simple
echo agent so you can run them without `GEMINI_API_KEY` — the interesting
part is how the client passes `sessionId` vs `snapshotId`.

## When to use what

- **First turn in a chat** — `init: { sessionId }`
- **Fork or regenerate** — `init: { snapshotId: parentSnap }` plus a new user message
- **Keep going after the user picked a branch** — `init: { snapshotId: activeLeafSnap }`

After siblings exist, resolving "latest" by `sessionId` alone fails — there
isn't one leaf anymore. The UI should store a `snapshotId` on each message and
track which branch is active.

## Samples

| File | Pattern |
|------|---------|
| `01_fork_sibling_snapshots.py` | Two paths from one checkpoint |
| `02_regenerate_alternate.py` | Try again without losing the first answer |
| `03_parallel_explore.py` | Several directions from the same starting point |
| `04_continue_active_branch.py` | Next turn follows the branch the user chose |

```bash
cd py/samples/agents
uv sync
uv run python branching_examples/01_fork_sibling_snapshots.py
```

Helpers: [`_helpers.py`](_helpers.py). Wire format: [`../../WIRE_PROTOCOL.md`](../../WIRE_PROTOCOL.md).
