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

import { randomUUID } from 'crypto';
import { App } from 'firebase-admin/app';
import {
  DocumentReference,
  FieldValue,
  getFirestore,
  type Firestore,
} from 'firebase-admin/firestore';
import {
  GenkitError,
  StreamNotFoundError,
  type ActionStreamInput,
  type ActionStreamSubscriber,
  type StreamManager,
} from 'genkit/beta';

export interface FirestoreStreamManagerOptions {
  firebaseApp?: App;
  db?: Firestore;
  collection: string;
  timeout?: number;
}

class FirestoreActionStream<S, O> implements ActionStreamInput<S, O> {
  private streamDoc;

  constructor(streamDoc: DocumentReference) {
    this.streamDoc = streamDoc;
  }

  private async update(data: any): Promise<void> {
    await this.streamDoc.update({
      ...data,
      updatedAt: FieldValue.serverTimestamp(),
    });
  }

  async write(chunk: S): Promise<void> {
    await this.update({
      // We add a random ID to the chunk to prevent Firestore from deduplicating chunks
      // that have the same content.
      stream: FieldValue.arrayUnion({
        type: 'chunk',
        chunk,
        uuid: randomUUID(),
      }),
    });
  }

  async done(output: O): Promise<void> {
    await this.update({
      stream: FieldValue.arrayUnion({ type: 'done', output }),
    });
  }

  async error(err: any): Promise<void> {
    const serializableError = {
      message: err.message,
      stack: err.stack,
      ...err,
    };
    await this.update({
      stream: FieldValue.arrayUnion({ type: 'error', err: serializableError }),
    });
  }
}

export class FirestoreStreamManager implements StreamManager {
  private db: Firestore;
  private collection: string;
  private timeout: number;

  constructor(opts: FirestoreStreamManagerOptions) {
    this.collection = opts.collection;
    this.db =
      opts.db ??
      (opts.firebaseApp ? getFirestore(opts.firebaseApp) : getFirestore());
    this.timeout = opts.timeout ?? 60000;
  }

  async open<S, O>(streamId: string): Promise<ActionStreamInput<S, O>> {
    const streamDoc = this.db.collection(this.collection).doc(streamId);
    await streamDoc.set({
      createdAt: FieldValue.serverTimestamp(),
      stream: [],
    });
    return new FirestoreActionStream(streamDoc);
  }

  async subscribe<S, O>(
    streamId: string,
    callbacks: ActionStreamSubscriber<S, O>
  ): Promise<{
    unsubscribe: () => void;
  }> {
    const streamDoc = this.db.collection(this.collection).doc(streamId);
    const snapshot = await streamDoc.get();
    if (!snapshot.exists) {
      throw new StreamNotFoundError(`Stream ${streamId} not found.`);
    }
    let lastIndex = -1;
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
    const unsubscribe = streamDoc.onSnapshot((snapshot) => {
      resetTimeout();
      const data = snapshot.data();
      if (!data) {
        return;
      }
      const stream = data.stream || [];
      for (let i = lastIndex + 1; i < stream.length; i++) {
        const event = stream[i];
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
      }
      lastIndex = stream.length - 1;
    });
    resetTimeout();
    return { unsubscribe };
  }
}
