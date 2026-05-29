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

// ---------------------------------------------------------------------------
// Snapshot persistence
// ---------------------------------------------------------------------------
//
// The Genkit agent is stateful server-side: each completed turn returns a
// `snapshotId` that the client passes back to continue the conversation. The
// transport must therefore remember, per chat, the snapshot to resume from.
//
// By default this state lives in memory ({@link InMemorySnapshotStore}), which
// means a full page reload loses multi-turn continuity. Provide a persistent
// store (e.g. {@link LocalStorageSnapshotStore}) to survive reloads.

/**
 * Identifies a single tool request that the agent paused on (an interrupt)
 * and is awaiting client resolution for.
 */
export interface PendingInterrupt {
  /** The tool call id (`ref`) of the interrupted tool request. */
  toolCallId: string;
  /** The name of the interrupted tool. */
  toolName: string;
}

/**
 * Per-chat snapshot bookkeeping tracked by the transport.
 */
export interface ChatSnapshot {
  /** The snapshot to resume the conversation from on the next turn. */
  snapshotId?: string;
  /**
   * The snapshot from *before* the current one. Used to regenerate the last
   * assistant response: a `regenerate-message` re-runs from this snapshot so
   * the final turn is produced again from the prior conversation state.
   */
  previousSnapshotId?: string;
  /** Whether the chat is currently paused on an interrupt. */
  interrupted?: boolean;
  /**
   * The specific tool requests the agent is currently paused on. The transport
   * uses these refs to build a precise `resume` payload on the next turn —
   * sending back *only* the resolved interrupt results and ignoring unrelated
   * (e.g. auto-executed) tool outputs in the message history.
   */
  pendingInterrupts?: PendingInterrupt[];
}

/**
 * Pluggable storage for per-chat {@link ChatSnapshot} state.
 *
 * All methods are async-capable so implementations can be backed by
 * asynchronous storage (IndexedDB, remote APIs, etc.). The default
 * {@link InMemorySnapshotStore} resolves synchronously.
 */
export interface SnapshotStore {
  /** Returns the stored snapshot for a chat, or `undefined` if none. */
  get(
    chatId: string
  ): ChatSnapshot | undefined | Promise<ChatSnapshot | undefined>;
  /** Persists the snapshot for a chat. */
  set(chatId: string, snapshot: ChatSnapshot): void | Promise<void>;
  /** Removes all stored state for a chat. */
  delete(chatId: string): void | Promise<void>;
}

/**
 * Default in-memory {@link SnapshotStore}.
 *
 * State is held in a `Map` for the lifetime of the transport instance and is
 * lost on page reload. Use {@link LocalStorageSnapshotStore} (or a custom
 * implementation) for persistence across reloads.
 */
export class InMemorySnapshotStore implements SnapshotStore {
  private readonly snapshots = new Map<string, ChatSnapshot>();

  get(chatId: string): ChatSnapshot | undefined {
    return this.snapshots.get(chatId);
  }

  set(chatId: string, snapshot: ChatSnapshot): void {
    this.snapshots.set(chatId, snapshot);
  }

  delete(chatId: string): void {
    this.snapshots.delete(chatId);
  }
}

/**
 * A {@link SnapshotStore} backed by the browser `localStorage` API, so
 * multi-turn conversations survive a page reload.
 *
 * Each chat is stored under a namespaced key (`<prefix><chatId>`). If
 * `localStorage` is unavailable (e.g. during SSR or in a privacy mode that
 * throws on access), all operations degrade gracefully to no-ops.
 *
 * @example
 * ```ts
 * new GenkitChatTransport({
 *   url: '/api/chat/weather',
 *   store: new LocalStorageSnapshotStore(),
 * });
 * ```
 */
export class LocalStorageSnapshotStore implements SnapshotStore {
  private readonly prefix: string;

  constructor(options?: { prefix?: string }) {
    this.prefix = options?.prefix ?? 'genkit-chat:';
  }

  private storage(): Storage | undefined {
    try {
      if (typeof globalThis !== 'undefined' && globalThis.localStorage) {
        return globalThis.localStorage;
      }
    } catch {
      // Accessing localStorage can throw (e.g. blocked by privacy settings).
    }
    return undefined;
  }

  private key(chatId: string): string {
    return `${this.prefix}${chatId}`;
  }

  get(chatId: string): ChatSnapshot | undefined {
    const storage = this.storage();
    if (!storage) return undefined;
    try {
      const raw = storage.getItem(this.key(chatId));
      if (!raw) return undefined;
      return JSON.parse(raw) as ChatSnapshot;
    } catch {
      return undefined;
    }
  }

  set(chatId: string, snapshot: ChatSnapshot): void {
    const storage = this.storage();
    if (!storage) return;
    try {
      storage.setItem(this.key(chatId), JSON.stringify(snapshot));
    } catch {
      // Quota exceeded or storage disabled — ignore.
    }
  }

  delete(chatId: string): void {
    const storage = this.storage();
    if (!storage) return;
    try {
      storage.removeItem(this.key(chatId));
    } catch {
      // Ignore.
    }
  }
}
