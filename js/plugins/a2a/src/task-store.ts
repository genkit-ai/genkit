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
 * A pointer from an A2A `taskId` to the Genkit snapshot that currently backs
 * it, plus the task's `contextId` (the Genkit `sessionId`).
 */
export interface TaskRecord {
  /** The A2A `contextId` for this task (equals the Genkit `sessionId`). */
  contextId: string;
  /** The snapshot the task's current turn is persisted under. */
  snapshotId: string;
}

/**
 * A minimal key-value store the {@link GenkitA2ARequestHandler} uses to
 * remember which Genkit snapshot currently backs an A2A task.
 *
 * For a **server-managed** agent (one with a `SessionStore`) an A2A task id is
 * the snapshot id of the turn that *originated* the task, so the mapping is the
 * identity function for the common case and no entry is written. An entry is
 * recorded only when a task *advances* past its originating snapshot - i.e.
 * when an interrupted task is resumed and the resumed turn persists a new
 * snapshot. `resolve(taskId) = get(taskId)?.snapshotId ?? taskId`.
 *
 * Because only resumed (interrupt) tasks are ever stored, a purely
 * non-interrupting agent never writes to it. Durability across process
 * restarts therefore matters only for in-flight interrupt tasks; provide a
 * durable implementation (Redis / Firestore / SQL - any KV) via the handler's
 * `taskStore` option when that matters. The default is in-memory.
 */
export interface A2ATaskStore {
  /** Returns the advancement record for a task, or `undefined` if none. */
  get(taskId: string): Promise<TaskRecord | undefined>;
  /** Records (or updates) the advancement record for a task. */
  set(taskId: string, record: TaskRecord): Promise<void>;
  /** Removes a task's advancement record, if any. */
  delete(taskId: string): Promise<void>;
}

/**
 * The default in-memory {@link A2ATaskStore}. Backed by a `Map`, with an
 * optional soft cap that evicts the oldest entry once exceeded. Suitable for a
 * single-process server; supply a durable store for multi-process or
 * restart-surviving deployments.
 */
export class InMemoryA2ATaskStore implements A2ATaskStore {
  private readonly records = new Map<string, TaskRecord>();

  /**
   * @param maxEntries Soft cap on retained advancement records. When exceeded,
   *   the oldest entry is evicted. Defaults to 10000.
   */
  constructor(private readonly maxEntries = 10000) {}

  async get(taskId: string): Promise<TaskRecord | undefined> {
    return this.records.get(taskId);
  }

  async set(taskId: string, record: TaskRecord): Promise<void> {
    this.records.set(taskId, record);
    if (this.records.size > this.maxEntries) {
      const oldest = this.records.keys().next().value;
      if (oldest !== undefined) {
        this.records.delete(oldest);
      }
    }
  }

  async delete(taskId: string): Promise<void> {
    this.records.delete(taskId);
  }
}
