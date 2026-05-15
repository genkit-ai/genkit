/**
 * Copyright 2024 Google LLC
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

import { GenkitError } from './error';

/**
 * Error thrown when a stream cannot be found.
 */
export class StreamNotFoundError extends GenkitError {
  constructor(message: string) {
    super({ status: 'NOT_FOUND', message });
    this.name = 'StreamNotFoundError';
  }
}

/**
 * Interface for writing content to a stream.
 * @template S The type of the stream chunks.
 * @template O The type of the final output.
 */
export interface ActionStreamInput<S, O> {
  /**
   * Writes a chunk to the stream.
   * @param chunk The chunk data to write.
   */
  write(chunk: S): Promise<void>;
  /**
   * Closes the stream with a final output.
   * @param output The final output data.
   */
  done(output: O): Promise<void>;
  /**
   * Closes the stream with an error.
   * @param err The error that occurred.
   */
  error(err: any): Promise<void>;
}

/**
 * Subscriber callbacks for receiving stream events.
 * @template S The type of the stream chunks.
 * @template O The type of the final output.
 */
export type ActionStreamSubscriber<S, O> = {
  /** Called when a chunk is received. */
  onChunk: (chunk: S) => void;
  /** Called when the stream completes successfully. */
  onDone: (output: O) => void;
  /** Called when the stream encounters an error. */
  onError: (error: any) => void;
};

/**
 * Interface for managing streaming actions, allowing creation and subscription to streams.
 * Implementations can provide different storage backends (e.g., in-memory, database, cache).
 */
export interface StreamManager {
  /**
   * Opens a stream for writing.
   * @param streamId The unique identifier for the stream.
   * @returns An object to write to the stream.
   */
  open<S, O>(streamId: string): Promise<ActionStreamInput<S, O>>;
  /**
   * Subscribes to a stream to receive its events.
   * @param streamId The unique identifier for the stream.
   * @param options The subscriber callbacks.
   * @returns A promise resolving to an object containing an unsubscribe function.
   */
  subscribe<S, O>(
    streamId: string,
    options: ActionStreamSubscriber<S, O>
  ): Promise<{ unsubscribe: () => void }>;
}

type StreamState<S, O> =
  | {
      status: 'open';
      chunks: S[];
      subscribers: ActionStreamSubscriber<S, O>[];
      lastTouched: number;
    }
  | { status: 'done'; chunks: S[]; output: O; lastTouched: number }
  | { status: 'error'; chunks: S[]; error: any; lastTouched: number };

/**
 * An in-memory implementation of StreamManager.
 * Useful for testing or single-instance deployments where persistence is not required.
 */
export class InMemoryStreamManager implements StreamManager {
  private streams: Map<string, StreamState<any, any>> = new Map();

  /**
   * @param options Configuration options.
   * @param options.ttlSeconds Time-to-live for streams in seconds. Defaults to 5 minutes.
   */
  constructor(private options: { ttlSeconds?: number } = {}) {}

  private _cleanup() {
    const ttl = (this.options.ttlSeconds ?? 5 * 60) * 1000;
    const now = Date.now();
    for (const [streamId, stream] of this.streams.entries()) {
      if (stream.status !== 'open' && now - stream.lastTouched > ttl) {
        this.streams.delete(streamId);
      }
    }
  }

  async open<S, O>(streamId: string): Promise<ActionStreamInput<S, O>> {
    this._cleanup();
    if (this.streams.has(streamId)) {
      throw new Error(`Stream with id ${streamId} already exists.`);
    }
    this.streams.set(streamId, {
      status: 'open',
      chunks: [],
      subscribers: [],
      lastTouched: Date.now(),
    });

    return {
      write: async (chunk: S) => {
        const stream = this.streams.get(streamId);
        if (stream?.status === 'open') {
          stream.chunks.push(chunk);
          stream.subscribers.forEach((s) => s.onChunk(chunk));
          stream.lastTouched = Date.now();
        }
      },
      done: async (output: O) => {
        const stream = this.streams.get(streamId);
        if (stream?.status === 'open') {
          this.streams.set(streamId, {
            status: 'done',
            chunks: stream.chunks,
            output,
            lastTouched: Date.now(),
          });
          stream.subscribers.forEach((s) => s.onDone(output));
        }
      },
      error: async (err: any) => {
        const stream = this.streams.get(streamId);
        if (stream?.status === 'open') {
          stream.subscribers.forEach((s) => s.onError(err));
          this.streams.set(streamId, {
            status: 'error',
            chunks: stream.chunks,
            error: err,
            lastTouched: Date.now(),
          });
        }
      },
    };
  }

  async subscribe<S, O>(
    streamId: string,
    subscriber: ActionStreamSubscriber<S, O>
  ): Promise<{ unsubscribe: () => void }> {
    const stream = this.streams.get(streamId);
    if (!stream) {
      throw new StreamNotFoundError(`Stream with id ${streamId} not found.`);
    }

    if (stream.status === 'done') {
      for (const chunk of stream.chunks) {
        subscriber.onChunk(chunk);
      }
      subscriber.onDone(stream.output);
    } else if (stream.status === 'error') {
      for (const chunk of stream.chunks) {
        subscriber.onChunk(chunk);
      }
      subscriber.onError(stream.error);
    } else {
      stream.chunks.forEach((chunk) => subscriber.onChunk(chunk));
      stream.subscribers.push(subscriber);
    }

    return {
      unsubscribe: () => {
        const currentStream = this.streams.get(streamId);
        if (currentStream?.status === 'open') {
          const index = currentStream.subscribers.indexOf(subscriber);
          if (index > -1) {
            currentStream.subscribers.splice(index, 1);
          }
        }
      },
    };
  }
}
