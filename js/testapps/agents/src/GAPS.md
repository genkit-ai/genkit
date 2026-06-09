# Git-Snapshot Agent — API gaps & findings

This sample was built primarily as an **exploratory exercise** to probe the
flexibility of Genkit's agent/session snapshot API: specifically, can a custom
agent snapshot an *external* resource (the filesystem, via git) alongside
session state, and have a conversation rollback/branch also roll back/branch
that external state?

**Short answer: yes**, and the working sample
([`git-snapshot-agent.ts`](./git-snapshot-agent.ts)) proves it — but only after
one small framework change. Below are the gaps and rough edges surfaced.

## The one gap that blocked it: snapshotId not known at turn start

The whole design hinges on naming a git branch/worktree after the snapshotId
*before* generating, then committing the filesystem under that same id so the
session snapshot (persisted at turn end) and the git commit share one key.

Previously the snapshotId was minted **inside the store at turn end** — it
didn't exist when the handler ran, so there was nothing to name the branch
after. The handler had no way to correlate "the snapshot this turn will
produce" with anything it set up up front.

### Fix applied (framework change in this PR)

- **`reserveSnapshotId(sessionId?, parentId?)`** (in `js/ai/src/session.ts`):
  mints a store-compatible `s_{convoId}_{suffix}` id ahead of time, so the id
  reserved at turn start round-trips through `FileSessionStore` /
  `InMemorySessionStore` (whose `parseSnapshotId` requires that exact shape).
- **`TurnContext`** (in `js/ai/src/agent.ts`): the `sess.run` handler now
  receives a second arg `{ snapshotId, parentSnapshotId, turnIndex }`. The
  `snapshotId` is reserved at turn start and the snapshot persisted at turn end
  reuses it (`maybeSnapshot` prefers the pre-reserved `newSnapshotId`).

```ts
await sess.run(async (input, { snapshotId, parentSnapshotId }) => {
  const worktree = await prepWorktree({ snapshotId, parentSnapshotId });
  const res = await ai.generate({ /* ... */, use: [filesystem({ rootDirectory: worktree })] });
  await commitWorktree({ worktreePath: worktree, snapshotId });
  return { finishReason: res.finishReason };
});
```

This is backward compatible: the second arg is optional, so existing handlers
(`definePromptAgent`, the other samples) keep working unchanged.

A latent bug was also fixed: the **detach** path reserved a raw
`crypto.randomUUID()`, which `FileSessionStore` would later reject when parsing.
It now routes through `reserveSnapshotId` too.

## Remaining gaps / rough edges (not addressed here)

1. **No "external snapshot" lifecycle hooks.** The correlation between a session
   snapshot and the git commit is maintained entirely in user land (matching
   ids by convention). There is no framework callback like
   `onSnapshotCommitted(snapshotId)` / `onSnapshotRolledBack(parentId)` /
   `onSnapshotDeleted(snapshotId)`. Consequences:
   - If the turn **fails after** `commitWorktree` but the framework decides not
     to persist the session snapshot (e.g. a selective `snapshotCallback`
     returns false, or a recovery snapshot is written under a *different* id),
     the git branch and the session snapshot can drift. The sample sidesteps
     this by always persisting (no `snapshotCallback`), but a real integration
     needs a transactional "commit external state iff the session snapshot was
     committed under this id" hook.
   - Pruned/garbage-collected session snapshots leave orphan git branches.
     There's no hook to GC the external resource alongside.

2. **`parentSnapshotId` is the runner's last persisted snapshot, not
   necessarily the resumed one.** On the *first* turn after
   `loadChat({ snapshotId })`, `parentSnapshotId` correctly equals the resumed
   snapshot. But the relationship is implicit — the handler can't independently
   ask "what snapshot did this session resume from?" There's no accessor like
   `sess.parentSnapshotId` / `sess.resumedFrom` outside the `run` callback.

3. **Reserved-but-unused ids.** If a turn reserves a snapshotId (and the handler
   names a branch after it) but then the turn produces no diff / is skipped by
   `snapshotCallback`, the reserved id is never persisted as a session snapshot
   — but the git branch already exists. Orphan again. A reservation that is
   explicitly "released" if unused would help.

4. **No multi-resource transaction.** This sample snapshots one external
   resource (filesystem). An agent touching several (filesystem + a DB + a
   cloud bucket) would need each to commit/rollback atomically with the session
   snapshot. The framework offers no coordination primitive; everything is
   best-effort in user land.

5. **Concurrency is the author's problem.** Worktrees make *parallel* branched
   turns possible, but the shared git repo's metadata ops must be serialized by
   hand (this sample uses a small promise-chain lock). Nothing in the agent API
   signals "these two turns are on divergent branches and may run concurrently."

6. **`getSnapshotData` exposes session state but not a way to enumerate a
   snapshot's lineage cheaply.** Reconstructing the branch graph (for, say, a
   "history tree" UI that also shows file diffs) means walking `parentId`
   pointers one snapshot at a time; there's no batch/lineage query.

## Verdict

The snapshot API is flexible enough to anchor external versioned state, and the
single `TurnContext.snapshotId`-at-turn-start affordance unlocks the whole
pattern. The biggest opportunity for a first-class feature is **external
snapshot lifecycle hooks** (commit / rollback / prune) so the correlation
between a session snapshot and its external resources can be made transactional
instead of by-convention.
