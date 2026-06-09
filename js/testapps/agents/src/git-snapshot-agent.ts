/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Git-Snapshot Agent — snapshotting the FILESYSTEM alongside session state
 *
 * This sample explores a powerful idea: Genkit's session store can snapshot
 * and roll back *conversation* state to any point, but a coding/file-editing
 * agent also mutates an external resource — the filesystem — that the session
 * store knows nothing about. If you roll back the conversation to an earlier
 * snapshot, the files on disk are still from the latest turn. The two are out
 * of sync.
 *
 * Here we keep them in sync by giving git the job of versioning the
 * filesystem, keyed by the *same* snapshotId the session store uses:
 *
 *   • Each agent turn gets a snapshotId reserved at turn START (via the new
 *     `TurnContext` passed to the `sess.run` handler).
 *   • Before generating, we `prepWorktree({ snapshotId, parentSnapshotId })`:
 *     create a git worktree + branch `genkit/<snapshotId>` based off the
 *     *parent* snapshot's branch (or an empty root for the first turn). The
 *     worktree starts with exactly the files the parent snapshot left behind.
 *   • The model edits files via the `filesystem` middleware scoped to that
 *     worktree.
 *   • At turn END we `commitWorktree(...)`: `git add -A && git commit`, so the
 *     filesystem state is captured under a branch named after the snapshotId.
 *     The session snapshot persisted at turn end reuses that very snapshotId.
 *
 * Because the git branch name === the session snapshotId, rolling back the
 * conversation to snapshot S "just works" for the filesystem too: the next
 * turn branched from S sees S's files (its `parentSnapshotId` is S, so its
 * worktree is created off `genkit/<S>`). Branching the conversation branches
 * the filesystem.
 *
 * Worktrees (vs. plain checkouts) mean each snapshot's filesystem lives in its
 * own directory, so parallel/branched turns never stomp on each other — the
 * door is open to running branches concurrently.
 *
 * NOTE: This is primarily an *exploratory* sample probing the flexibility of
 * the agent/session snapshot API. See GAPS.md (next to this file) for the API
 * gaps and rough edges it surfaced.
 */

import { filesystem } from '@genkit-ai/middleware';
import { exec } from 'child_process';
import * as fs from 'fs';
import { z } from 'genkit';
import { FileSessionStore } from 'genkit/beta';
import * as path from 'path';
import { promisify } from 'util';
import { ai } from './genkit.js';

const execAsync = promisify(exec);

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------
//
//   .git-snapshots/
//     repo/        ← the backing git repo (one empty root commit on `main`)
//     wt/<id>/     ← per-turn live worktree (created on prep, removed on commit)
//     inspect/<id> ← throwaway worktree used to read a snapshot's files
//
const BASE_DIR = path.resolve(__dirname, '..', '.git-snapshots');
const REPO_DIR = path.join(BASE_DIR, 'repo');
const WT_DIR = path.join(BASE_DIR, 'wt');
const INSPECT_DIR = path.join(BASE_DIR, 'inspect');

/** The git branch that tracks a given session snapshot's filesystem state. */
const branchFor = (snapshotId: string) => `genkit/${snapshotId}`;
/** A filesystem-safe directory name for a snapshotId (slashes/colons → `_`). */
const safeDir = (snapshotId: string) => snapshotId.replace(/[^\w.-]/g, '_');

// File-based session store — keyed snapshots persist across restarts, mirroring
// the durability of the git branches.
const store = new FileSessionStore<{}>('./.snapshots-git');

// ---------------------------------------------------------------------------
// Git plumbing
//
// All git metadata operations on the shared repo are serialized through a tiny
// promise-chain lock: worktree file I/O can happen in parallel, but `git
// worktree add/remove`, `commit`, etc. touch shared repo state and must not
// race.
// ---------------------------------------------------------------------------

let gitChain: Promise<unknown> = Promise.resolve();
function withGitLock<T>(fn: () => Promise<T>): Promise<T> {
  const next = gitChain.then(fn, fn);
  // Keep the chain alive even if a step rejects.
  gitChain = next.catch(() => {});
  return next;
}

const git = (args: string, cwd: string = REPO_DIR) =>
  execAsync(`git ${args}`, {
    cwd,
    maxBuffer: 8 * 1024 * 1024,
    env: { ...process.env, GIT_TERMINAL_PROMPT: '0' },
  });

let repoReady: Promise<void> | undefined;

/** Lazily initializes the backing repo with a single empty root commit. */
function ensureRepo(): Promise<void> {
  if (!repoReady) {
    repoReady = (async () => {
      if (fs.existsSync(path.join(REPO_DIR, '.git'))) return;
      await fs.promises.mkdir(REPO_DIR, { recursive: true });
      await git('init -b main');
      // Identity + an empty root commit so worktrees have a base to branch off.
      await git('config user.email genkit@example.com');
      await git('config user.name "Genkit Sample"');
      await git('commit --allow-empty -m "root"');
    })();
  }
  return repoReady;
}

/**
 * Creates a worktree for `snapshotId` based off the parent snapshot's branch
 * (or the empty root for a first turn) and returns the worktree path. The
 * worktree starts with exactly the files the parent snapshot left behind.
 */
async function prepWorktree(ctx: {
  snapshotId: string;
  parentSnapshotId?: string;
}): Promise<string> {
  await ensureRepo();
  const worktreePath = path.join(WT_DIR, safeDir(ctx.snapshotId));
  const branch = branchFor(ctx.snapshotId);
  // First turn branches off the empty root (`main`); subsequent turns branch
  // off their PARENT snapshot's branch — this is what makes a conversation
  // rollback/branch also roll back/branch the filesystem.
  const baseRef = ctx.parentSnapshotId
    ? branchFor(ctx.parentSnapshotId)
    : 'main';

  await withGitLock(async () => {
    await fs.promises.rm(worktreePath, { recursive: true, force: true });
    await git('worktree prune');
    // Recreate the branch fresh each time (a retried turn reuses the id).
    await git(`branch -f "${branch}" "${baseRef}"`);
    await git(`worktree add "${worktreePath}" "${branch}"`);
  });

  return worktreePath;
}

/**
 * Commits the worktree's current contents under the snapshot's branch, then
 * tears the worktree down (the branch is the durable reference). The git
 * commit is now keyed by the same snapshotId the session store persists.
 */
async function commitWorktree(opts: {
  worktreePath: string;
  snapshotId: string;
}): Promise<void> {
  await withGitLock(async () => {
    await git('add -A', opts.worktreePath);
    // `--allow-empty`: a turn that edited nothing still produces a commit, so
    // every snapshotId has a corresponding filesystem commit.
    await git(
      `commit --allow-empty -m "snapshot ${opts.snapshotId}"`,
      opts.worktreePath
    );
    await git(`worktree remove --force "${opts.worktreePath}"`);
  });
}

/**
 * Materializes a snapshot's filesystem into a throwaway worktree and returns
 * the relative paths of all files (excluding `.git`). Demonstrates that any
 * snapshotId can be rolled back to on disk, at any time.
 */
async function inspectSnapshotFiles(snapshotId: string): Promise<string[]> {
  await ensureRepo();
  const dir = path.join(INSPECT_DIR, safeDir(snapshotId));
  await withGitLock(async () => {
    await fs.promises.rm(dir, { recursive: true, force: true });
    await git('worktree prune');
    await git(`worktree add --detach "${dir}" "${branchFor(snapshotId)}"`);
  });
  const files = listFilesRecursive(dir, dir).sort();
  await withGitLock(() => git(`worktree remove --force "${dir}"`));
  return files;
}

/** Lists all files under `dir` (relative to `root`), skipping `.git`. */
function listFilesRecursive(dir: string, root: string): string[] {
  const out: string[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === '.git') continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...listFilesRecursive(full, root));
    } else {
      out.push(path.relative(root, full));
    }
  }
  return out;
}

/** Wipes all git-snapshot state — handy for a clean, repeatable demo run. */
async function resetGitState(): Promise<void> {
  repoReady = undefined;
  await fs.promises.rm(BASE_DIR, { recursive: true, force: true });
}

// ---------------------------------------------------------------------------
// Agent definition
// ---------------------------------------------------------------------------

export const gitSnapshotAgent = ai.defineCustomAgent(
  { name: 'gitSnapshotAgent', store },
  async (sess, { sendChunk, abortSignal }) => {
    let lastMessage: any;

    // The TurnContext (2nd arg) carries the snapshotId reserved at turn START,
    // plus the parent snapshot this turn continues from. That's the hook that
    // makes filesystem snapshotting possible: we can name the git
    // branch/worktree after the snapshot BEFORE generating, then commit under
    // the same id at the end.
    await sess.run(async (input, { snapshotId, parentSnapshotId }) => {
      // 1. Prepare a filesystem worktree for this snapshot, seeded from the
      //    parent snapshot's files.
      const worktreePath = await prepWorktree({ snapshotId, parentSnapshotId });

      // 2. Generate with the filesystem middleware scoped to THIS worktree.
      //    sess.run() has already appended input.messages, so getMessages()
      //    is the full conversation so far.
      const { stream, response } = ai.generateStream({
        messages: sess.getMessages(),
        abortSignal,
        system:
          'You are a file-editing assistant. You have list_files, read_file, ' +
          'write_file, and search_and_replace tools scoped to a workspace ' +
          'directory. Do exactly what the user asks, then briefly confirm what ' +
          'you did. Keep file contents short.',
        use: [filesystem({ rootDirectory: worktreePath, allowWriteAccess: true })],
      });

      for await (const chunk of stream) {
        sendChunk({ modelChunk: chunk });
      }
      const res = await response;
      lastMessage = res.message;
      if (lastMessage) {
        sess.addMessages([lastMessage]);
      }

      // 3. Commit the filesystem under the same snapshotId. The session
      //    snapshot persisted right after this handler returns reuses that id,
      //    binding the two together.
      await commitWorktree({ worktreePath, snapshotId });

      return { finishReason: res.finishReason as any };
    });

    return {
      message: lastMessage || {
        role: 'model' as const,
        content: [{ text: 'Done.' }],
      },
    };
  }
);

// ---------------------------------------------------------------------------
// Test flow — demonstrates snapshot/rollback of conversation + filesystem
//
//   turn 1 (S1): create a.txt              → filesystem: [a.txt]
//   turn 2 (S2): create b.txt              → filesystem: [a.txt, b.txt]
//   branch the conversation from S1, then
//   turn 3 (S3): create c.txt              → filesystem: [a.txt, c.txt]
//
// S3 has NO b.txt: branching the conversation back to S1 also rolled the
// filesystem back to S1's state, on top of which c.txt was added.
// ---------------------------------------------------------------------------

export const testGitSnapshotAgent = ai.defineFlow(
  {
    name: 'testGitSnapshotAgent',
    inputSchema: z.void(),
    outputSchema: z.object({
      s1: z.object({ snapshotId: z.string(), files: z.array(z.string()) }),
      s2: z.object({ snapshotId: z.string(), files: z.array(z.string()) }),
      s3: z.object({ snapshotId: z.string(), files: z.array(z.string()) }),
      branchedFromS1WithoutBTxt: z.boolean(),
    }),
  },
  async (_, { sendChunk }) => {
    // Start from a clean slate so the demo is repeatable.
    await resetGitState();

    const chat = gitSnapshotAgent.chat();

    // ── Turn 1 ────────────────────────────────────────────────────────
    sendChunk({ status: 'Turn 1: creating a.txt' } as any);
    let res = await chat.send(
      'Create a file named a.txt containing the text "from turn 1".'
    );
    const s1 = res.snapshotId!;
    const s1Files = await inspectSnapshotFiles(s1);

    // ── Turn 2 ────────────────────────────────────────────────────────
    sendChunk({ status: 'Turn 2: creating b.txt' } as any);
    res = await chat.send(
      'Now create a file named b.txt containing the text "from turn 2".'
    );
    const s2 = res.snapshotId!;
    const s2Files = await inspectSnapshotFiles(s2);

    // ── Branch the conversation back to S1, then add c.txt ─────────────
    // loadChat({ snapshotId: S1 }) resumes the EXACT snapshot S1. The next
    // turn's parentSnapshotId is S1, so its worktree is branched off
    // git branch genkit/<S1> — i.e. the filesystem rolls back to S1 too.
    sendChunk({ status: 'Branching from S1, then creating c.txt' } as any);
    const branched = await gitSnapshotAgent.loadChat({ snapshotId: s1 });
    res = await branched.send(
      'Create a file named c.txt containing the text "branched from turn 1".'
    );
    const s3 = res.snapshotId!;
    const s3Files = await inspectSnapshotFiles(s3);

    return {
      s1: { snapshotId: s1, files: s1Files },
      s2: { snapshotId: s2, files: s2Files },
      s3: { snapshotId: s3, files: s3Files },
      // The proof: the branched turn's filesystem has a.txt but NOT b.txt.
      branchedFromS1WithoutBTxt:
        s3Files.includes('a.txt') && !s3Files.includes('b.txt'),
    };
  }
);
