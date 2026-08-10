/**
 * Copyright 2025 Google LLC
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

import { App } from 'firebase-admin/app';
import { Database, getDatabase } from 'firebase-admin/database';
import {
  GenkitError,
  StreamNotFoundError,
  type ActionStreamInput,
  type ActionStreamSubscriber,
  type StreamManager,
} from 'genkit/beta';

export interface RtdbStreamManagerOptions {
  firebaseApp?: App;
  db?: Database;
  refPrefix?: string;
  timeout?: number;
}

class RtdbActionStream<S, O> implements ActionStreamInput<S, O> {
  private db: Database;
  private streamRef: string;
  private metadataRef: string;

  constructor(db: Database, streamRef: string) {
    this.db = db;
    this.streamRef = `${streamRef}/stream`;
    this.metadataRef = `${streamRef}/metadata`;
  }

  private async update(): Promise<void> {
    await this.db.ref(this.metadataRef).update({ updatedAt: Date.now() });
  }

  async write(chunk: S): Promise<void> {
    await this.db.ref(this.streamRef).push({ type: 'chunk', chunk });
    await this.update();
  }

  async done(output: O): Promise<void> {
    await this.db.ref(this.streamRef).push({ type: 'done', output });
    await this.update();
  }

  async error(err: any): Promise<void> {
    const serializableError = {
      message: err.message,
      stack: err.stack,
      ...err,
    };
    await this.db
      .ref(this.streamRef)
      .push({ type: 'error', err: serializableError });
    await this.update();
  }
}

export class RtdbStreamManager implements StreamManager {
  private db: Database;
  private refRoot: string;
  private timeout: number;

  constructor(opts: RtdbStreamManagerOptions) {
    this.refRoot = opts.refPrefix ?? 'genkit-streams';
    // refRoot should have a trailing slash for simplicity.
    if (this.refRoot && !this.refRoot.endsWith('/')) {
      this.refRoot += '/';
    }
    this.db =
      opts.db ??
      (opts.firebaseApp ? getDatabase(opts.firebaseApp) : getDatabase());
    this.timeout = opts.timeout ?? 60000;
  }

  async open<S, O>(streamId: string): Promise<ActionStreamInput<S, O>> {
    const streamRef = this.refRoot + streamId;
    await this.db.ref(streamRef).remove();
    await this.db.ref(`${streamRef}/metadata`).set({ createdAt: Date.now() });
    return new RtdbActionStream(this.db, streamRef);
  }

  async subscribe<S, O>(
    streamId: string,
    callbacks: ActionStreamSubscriber<S, O>
  ): Promise<{
    unsubscribe: () => void;
  }> {
    const streamRef = this.db.ref(`${this.refRoot}${streamId}`);
    const snapshot = await streamRef.get();
    if (!snapshot.exists()) {
      throw new StreamNotFoundError(`Stream ${streamId} not found.`);
    }
    const streamDataRef = streamRef.child('stream');
    let timeoutId: NodeJS.Timeout;
    const resetTimeout = () => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        callbacks.onError?.(
          new GenkitError({
            status: 'DEADLINE_EXCEEDED',
            message: 'Stream timed out.',
          })
        );
        unsubscribe();
      }, this.timeout);
    };
    const handleEvent = (snapshot) => {
      resetTimeout();
      if (!snapshot.exists()) {
        return;
      }
      const event = snapshot.val();
      if (event.type === 'chunk') {
        callbacks.onChunk?.(event.chunk);
      } else if (event.type === 'done') {
        clearTimeout(timeoutId);
        callbacks.onDone?.(event.output);
        unsubscribe();
      } else if (event.type === 'error') {
        clearTimeout(timeoutId);
        callbacks.onError?.(event.err);
        unsubscribe();
      }
    };
    const unsubscribe = () => {
      streamDataRef.off('child_added', handleEvent);
    };
    streamDataRef.on('child_added', handleEvent);
    resetTimeout();
    return { unsubscribe };
  }
}
