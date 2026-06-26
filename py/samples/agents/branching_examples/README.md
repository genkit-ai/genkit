# Branching samples

Every turn in a store-backed agent is a snapshot you can fork from. These samples
build up from the core forking mechanics to two end-to-end demos, all in the same
hero style: `await session.send(...)` drives each turn and inline comments call out
what each step produces, so the script reads top-to-bottom like a story.

All of them use `googleai/gemini-flash-latest`, so they require `GEMINI_API_KEY`.

## The one idea

`session.snapshot_id` bookmarks any turn. `agent.load_chat(snapshot_id=...)` starts a
fresh session from that bookmark. Fork the same snapshot twice and you get sibling
timelines that never see each other — that's the whole primitive.

Once a session has more than one leaf, "the latest turn for this session" is
ambiguous, so resolve by `snapshot_id` (the branch you mean) rather than
`session_id` alone.

## Samples

| File | Pattern |
|------|---------|
| `01_fork_sibling_snapshots.py` | Two directions from one checkpoint; resolve the ambiguous-branch lookup |
| `02_regenerate_alternate.py` | "Try again" without losing the first answer |
| `03_parallel_explore.py` | Fan a checkpoint out into several branches concurrently (`asyncio.gather`) |
| `04_continue_active_branch.py` | Resume a normal linear chat on the branch the user picked |
| `05_time_travel_artifacts.py` | Git for agent state: rewind a build and watch the `Artifacts()` revert too |
| `06_parallel_deep_research.py` | Fork into parallel grounded research, then synthesize one brief |

The first four are the primitives; the last two combine them into things people
actually want to build.

## Run

```bash
cd py/samples/agents
uv sync
uv run branching_examples/01_fork_sibling_snapshots.py
```

Swap `InMemoryBranchingSessionStore` for `FileBranchingSessionStore('./tree')` in
any of them to persist the snapshot tree across process restarts.
