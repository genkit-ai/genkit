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

import { getAsyncContext, type z } from '@genkit-ai/core';
import type { Registry } from '@genkit-ai/core/registry';
import { v4 as uuidv4 } from 'uuid';
import { type GenerateOptions, type MessageData } from './index.js';

export type BaseGenerateOptions<
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
> = Omit<GenerateOptions<O, CustomOptions>, 'prompt'>;

export interface SessionOptions<S = any> {
  /** Session store implementation for persisting the session state. */
  store?: SessionStore<S>;
  /** Initial state of the session.  */
  initialState?: S;
  /** Custom session Id. */
  sessionId?: string;
}

/**
 * Session encapsulates a stateful execution environment.
 *
 * A session holds shared state and per-thread message history, and provides a
 * context in which functions can be executed via {@link Session.run} so that
 * `ai.currentSession()` resolves to this session. State can be read via
 * {@link Session.state} and updated with {@link Session.updateState}, and
 * conversation history can be persisted per thread with
 * {@link Session.updateMessages}.
 */
export class Session<S = any> {
  readonly id: string;
  private sessionData?: SessionData<S>;
  private store: SessionStore<S>;

  constructor(
    readonly registry: Registry,
    options?: {
      id?: string;
      stateSchema?: S;
      sessionData?: SessionData<S>;
      store?: SessionStore<S>;
    }
  ) {
    this.id = options?.id ?? uuidv4();
    this.sessionData = options?.sessionData ?? {
      id: this.id,
    };
    if (!this.sessionData) {
      this.sessionData = { id: this.id };
    }
    if (!this.sessionData.threads) {
      this.sessionData!.threads = {};
    }
    this.store = options?.store ?? new InMemorySessionStore<S>();
  }

  get state(): S | undefined {
    return this.sessionData!.state;
  }

  /**
   * Update session state data by patching the existing state.
   * @param data Partial state update that will be merged with existing state
   */
  async updateState(data: Partial<S>): Promise<void> {
    let sessionData = this.sessionData;
    if (!sessionData) {
      sessionData = {} as SessionData<S>;
    }

    // Merge the new data with existing state
    sessionData.state = {
      ...sessionData.state,
      ...data,
    } as S;

    this.sessionData = sessionData;
    await this.store.save(this.id, sessionData);
  }

  /**
   * Update messages for a given thread.
   */
  async updateMessages(thread: string, messages: MessageData[]): Promise<void> {
    let sessionData = this.sessionData;
    if (!sessionData) {
      sessionData = {} as SessionData<S>;
    }
    if (!sessionData.threads) {
      sessionData.threads = {};
    }
    sessionData.threads[thread] = messages.map((m: any) =>
      m.toJSON ? m.toJSON() : m
    );
    this.sessionData = sessionData;

    await this.store.save(this.id, sessionData);
  }

  /**
   * Executes provided function within this session context allowing calling
   * `ai.currentSession().state`
   */
  run<O>(fn: () => O) {
    return runWithSession(this.registry, this, fn);
  }

  toJSON() {
    return this.sessionData;
  }
}

export interface SessionData<S = any> {
  id: string;
  state?: S;
  threads?: Record<string, MessageData[]>;
}

const sessionAlsKey = 'ai.session';

/**
 * Executes provided function within the provided session state.
 */
export function runWithSession<S = any, O = any>(
  registry: Registry,
  session: Session<S>,
  fn: () => O
): O {
  return getAsyncContext().run(sessionAlsKey, session, fn);
}

/** Returns the current session. */
export function getCurrentSession<S = any>(
  registry: Registry
): Session<S> | undefined {
  return getAsyncContext().getStore(sessionAlsKey);
}

/** Throw when session state errors occur, ex. missing state, etc. */
export class SessionError extends Error {
  constructor(msg: string) {
    super(msg);
  }
}

/** Session store persists session data such as state and chat messages. */
export interface SessionStore<S = any> {
  get(sessionId: string): Promise<SessionData<S> | undefined>;

  save(sessionId: string, data: Omit<SessionData<S>, 'id'>): Promise<void>;
}

export function inMemorySessionStore() {
  return new InMemorySessionStore();
}

class InMemorySessionStore<S = any> implements SessionStore<S> {
  private data: Record<string, SessionData<S>> = {};

  async get(sessionId: string): Promise<SessionData<S> | undefined> {
    return this.data[sessionId];
  }

  async save(sessionId: string, sessionData: SessionData<S>): Promise<void> {
    this.data[sessionId] = sessionData;
  }
}
