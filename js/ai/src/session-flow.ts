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

import {
  GenkitError,
  defineAction,
  defineBidiAction,
  getContext,
  z,
  type Action,
  type ActionFnArg,
  type BidiAction,
} from '@genkit-ai/core';
import { Channel } from '@genkit-ai/core/async';
import type { Registry } from '@genkit-ai/core/registry';
import { generateStream } from './generate.js';
import {
  MessageData,
  MessageSchema,
  ModelResponseChunkSchema,
  PartSchema,
} from './model-types.js';
import { type ToolRequestPart } from './parts.js';
import { PromptAction } from './prompt.js';
import {
  Artifact,
  ArtifactSchema,
  InMemorySessionStore,
  Session,
  SessionSnapshot,
  SessionState,
  SessionStateSchema,
  SessionStore,
  SnapshotCallback,
  type SessionStoreOptions,
} from './session.js';

/**
 * Schema for initializing a session flow.
 */
export const SessionFlowInitSchema = z.object({
  snapshotId: z.string().optional(),
  newSnapshotId: z.string().optional(),
  state: SessionStateSchema.optional(),
});

/**
 * Initialization options for a session flow turn.
 */
export interface SessionFlowInit<S = unknown, I = unknown> {
  snapshotId?: string;
  newSnapshotId?: string;
  state?: SessionState<S, I>;
}

/**
 * Schema for session flow input messages and commands.
 */
export const SessionFlowInputSchema = z.object({
  messages: z.array(MessageSchema).optional(),
  toolRestarts: z.array(PartSchema).optional(),
  detach: z.boolean().optional(),
});

/**
 * Input received by a session flow turn.
 */
export type SessionFlowInput = z.infer<typeof SessionFlowInputSchema>;

/**
 * Schema identifying a turn termination event.
 */
export const TurnEndSchema = z.object({
  snapshotId: z.string().optional(),
});

/**
 * Identifies a turn termination event.
 */
export type TurnEnd = z.infer<typeof TurnEndSchema>;

/**
 * Schema for stream chunks emitted during a session flow.
 */
export const SessionFlowStreamChunkSchema = z.object({
  modelChunk: ModelResponseChunkSchema.optional(),
  status: z.any().optional(),
  artifact: ArtifactSchema.optional(),
  turnEnd: TurnEndSchema.optional(),
});

/**
 * Streamed chunk emitted during session flow execution.
 */
export type SessionFlowStreamChunk<Stream = unknown> = z.infer<
  typeof SessionFlowStreamChunkSchema
>;

/**
 * Schema for final results of a session flow execution.
 */
export const SessionFlowResultSchema = z.object({
  message: MessageSchema.optional(),
  artifacts: z.array(ArtifactSchema).optional(),
});

/**
 * Result returned upon completing a session flow execution.
 */
export type SessionFlowResult = z.infer<typeof SessionFlowResultSchema>;

/**
 * Schema for output returned at turn completion.
 */
export const SessionFlowOutputSchema = z.object({
  snapshotId: z.string().optional(),
  state: SessionStateSchema.optional(),
  message: MessageSchema.optional(),
  artifacts: z.array(ArtifactSchema).optional(),
});

/**
 * Output returned at turn completion.
 */
export interface SessionFlowOutput<S = unknown> {
  artifacts?: Artifact[];
  message?: MessageData;
  snapshotId?: string;
  state?: SessionState<S>;
}

/**
 * Executor responsible for running turns over input streams and persisting state.
 */
export class SessionRunner<State = unknown, InputVariables = unknown> {
  readonly session: Session<State, InputVariables>;
  readonly inputCh: AsyncIterable<SessionFlowInput>;
  turnIndex: number = 0;
  public onEndTurn?: (snapshotId?: string) => void;
  public onDetach?: (snapshotId: string) => void;
  public isClientManagedState?: boolean;
  public newSnapshotId?: string;
  private snapshotCallback?: SnapshotCallback<State>;
  private lastSnapshot?: SessionSnapshot<State, InputVariables>;

  private lastSnapshotVersion: number = 0;
  private store?: SessionStore<State, InputVariables>;
  public isDetached: boolean = false;

  constructor(
    session: Session<State, InputVariables>,
    inputCh: AsyncIterable<SessionFlowInput>,
    options?: {
      snapshotCallback?: SnapshotCallback<State>;
      lastSnapshot?: SessionSnapshot<State, InputVariables>;
      store?: SessionStore<State, InputVariables>;
      onEndTurn?: (snapshotId?: string) => void;
      onDetach?: (snapshotId: string) => void;
      newSnapshotId?: string;
      isClientManagedState?: boolean;
    }
  ) {
    this.session = session;
    this.inputCh = inputCh;

    this.snapshotCallback = options?.snapshotCallback;
    this.lastSnapshot = options?.lastSnapshot;
    this.store = options?.store;
    this.onEndTurn = options?.onEndTurn;
    this.onDetach = options?.onDetach;
    this.newSnapshotId = options?.newSnapshotId;
    this.isClientManagedState = options?.isClientManagedState;
  }

  /**
   * Executes the flow handler against incoming input messages sequentially.
   */
  async run(fn: (input: SessionFlowInput) => Promise<void>): Promise<void> {
    for await (const input of this.inputCh) {
      if (input.messages) {
        this.session.addMessages(input.messages);
      }

      const turnSnapshotId = this.newSnapshotId || crypto.randomUUID();

      try {
        await fn(input);

        const snapshotId = await this.maybeSnapshot(
          'turnEnd',
          'done',
          undefined,
          turnSnapshotId
        );
        try {
          if (this.onEndTurn) {
            this.onEndTurn(snapshotId);
          }
        } catch (e) {
          // Stream was closed, absorb exception
        }
        this.turnIndex++;
      } catch (e: any) {
        const errStatus = e.status || 'INTERNAL';
        const errMessage = e.message || 'Internal failure';
        const errDetails = e.detail || e.details || e;
        const snapshotId = await this.maybeSnapshot(
          'turnEnd',
          'failed',
          {
            status: errStatus,
            message: errMessage,
            details: errDetails,
          },
          turnSnapshotId
        );
        try {
          if (this.onEndTurn) {
            this.onEndTurn(snapshotId);
          }
        } catch (_) {
          // Stream was closed, absorb exception
        }
        throw e;
      }
    }
  }

  /**
   * Evaluates whether to save a snapshot to the persistent store.
   */
  async maybeSnapshot(
    event: 'turnEnd' | 'invocationEnd',
    status?: 'pending' | 'done' | 'failed',
    error?: { status: string; message: string; details?: any },
    snapshotId?: string
  ): Promise<string | undefined> {
    if (
      !this.store ||
      (this.isDetached && snapshotId !== this.lastSnapshot?.snapshotId)
    )
      return this.lastSnapshot?.snapshotId;

    if (snapshotId) {
      const existing = await this.store.getSnapshot(snapshotId, {
        context: getContext(),
      });
      if (existing?.status === 'aborted') {
        return snapshotId;
      }
    }

    const currentVersion = this.session.getVersion();
    if (currentVersion === this.lastSnapshotVersion && !status) {
      return this.lastSnapshot?.snapshotId;
    }

    const currentState = this.session.getState();
    const prevState = this.lastSnapshot ? this.lastSnapshot.state : undefined;

    if (this.snapshotCallback && !this.isDetached) {
      if (
        !this.snapshotCallback({
          state: currentState as SessionState<State>,
          prevState: prevState as SessionState<State> | undefined,
          turnIndex: this.turnIndex,
          event: event,
        })
      ) {
        return undefined;
      }
    }

    const snapshot: SessionSnapshot<State, InputVariables> = {
      snapshotId: snapshotId || this.newSnapshotId || crypto.randomUUID(),
      createdAt: new Date().toISOString(),
      event: event,
      state: currentState as SessionState<State, InputVariables>,
      parentId: this.lastSnapshot?.snapshotId,
      status,
      error,
    };

    await this.store.saveSnapshot(snapshot, { context: getContext() });

    this.lastSnapshot = snapshot;
    this.lastSnapshotVersion = currentVersion;

    return snapshot.snapshotId;
  }
}

/**
 * Function handler definition for custom Session Flow actions.
 */
export type SessionFlowFn<Stream, State, InputVariables = unknown> = (
  sess: SessionRunner<State, InputVariables>,
  options: {
    sendChunk: (chunk: SessionFlowStreamChunk<Stream>) => void;
    abortSignal?: AbortSignal;
    context?: any;
  }
) => Promise<SessionFlowResult>;

export type GetSnapshotDataAction<S = unknown, I = unknown> = Action<
  z.ZodString,
  z.ZodType<SessionSnapshot<S, I>>
>;

/**
 * Represents a configured, registered Session Flow.
 */
export interface SessionFlow<State = unknown, InputVariables = unknown>
  extends BidiAction<
    typeof SessionFlowInputSchema,
    typeof SessionFlowOutputSchema,
    typeof SessionFlowStreamChunkSchema,
    typeof SessionFlowInitSchema
  > {
  getSnapshotData(
    snapshotId: string,
    options?: SessionStoreOptions
  ): Promise<SessionSnapshot<State, InputVariables> | undefined>;

  abort(snapshotId: string, options?: SessionStoreOptions): Promise<void>;

  readonly getSnapshotDataAction: GetSnapshotDataAction<State, InputVariables>;
  readonly abortSessionFlowAction: Action<z.ZodString, z.ZodVoid>;
}

/**
 * Registers a multi-turn Session Flow action capable of maintaining persistent state.
 */
export function defineSessionFlow<
  Stream = unknown,
  State = unknown,
  InputVariables = unknown,
>(
  registry: Registry,
  config: {
    name: string;
    description?: string;
    store?: SessionStore<State, InputVariables>;
    snapshotCallback?: SnapshotCallback<State>;
  },
  fn: SessionFlowFn<Stream, State, InputVariables>
): SessionFlow<State, InputVariables> {
  const primaryAction = defineBidiAction(
    registry,
    {
      name: config.name,
      description: config.description,
      actionType: 'session-flow',
      inputSchema: SessionFlowInputSchema,
      outputSchema: SessionFlowOutputSchema,
      streamSchema: SessionFlowStreamChunkSchema,
      initSchema: SessionFlowInitSchema,
    },
    async function* (
      arg: ActionFnArg<
        SessionFlowStreamChunk,
        SessionFlowInput,
        SessionFlowInit
      >
    ) {
      const init = arg.init;
      const store =
        config.store || new InMemorySessionStore<State, InputVariables>();

      let session: Session<State, InputVariables>;

      let snapshot: SessionSnapshot<State, InputVariables> | undefined;

      if (init?.snapshotId) {
        snapshot = await store.getSnapshot(init.snapshotId, {
          context: getContext(),
        });
        if (!snapshot) {
          throw new Error(`Snapshot ${init.snapshotId} not found`);
        }
        session = new Session<State, InputVariables>(
          snapshot.state as SessionState<State, InputVariables>
        );
      } else if (init?.state) {
        session = new Session<State, InputVariables>(
          init.state as SessionState<State, InputVariables>
        );
      } else {
        session = new Session<State, InputVariables>({
          custom: {} as State,
          artifacts: [],
          messages: [],
        });
      }

      let detachedSnapshotId: string | undefined;
      let resolveDetach:
        | ((value: void | PromiseLike<void>) => void)
        | undefined;
      let rejectDetach: ((reason: any) => void) | undefined;
      const detachPromise = new Promise<void>((resolve, reject) => {
        resolveDetach = resolve;
        rejectDetach = reject;
      });

      const abortController = new AbortController();
      let unsubscribe: any = undefined;

      let runner!: SessionRunner<State, InputVariables>;

      // We construct an asynchronous proxy channel over the inputStream.
      // This enables immediate interception of `detach: true` directives. Without this proxy,
      // a backlog of pre-queued inputs would have to be resolved sequentially by the runner first.
      const runnerInputChannel = new Channel<SessionFlowInput>();

      (async () => {
        try {
          for await (const input of arg.inputStream) {
            if (input.detach) {
              if (!config.store) {
                if (rejectDetach) {
                  rejectDetach(
                    new GenkitError({
                      status: 'FAILED_PRECONDITION',
                      message:
                        'Detach is only supported when a session store is provided.',
                    })
                  );
                }
              } else {
                const turnSnapshotId =
                  runner.newSnapshotId || crypto.randomUUID();
                runner.newSnapshotId = turnSnapshotId;
                await runner.maybeSnapshot(
                  'turnEnd',
                  'pending',
                  undefined,
                  turnSnapshotId
                );
                runner.isDetached = true;

                if (runner.onDetach) {
                  runner.onDetach(turnSnapshotId);
                }
              }
            }
            runnerInputChannel.send(input);
          }
          runnerInputChannel.close();
        } catch (e) {
          runnerInputChannel.error(e);
        }
      })();

      runner = new SessionRunner<State, InputVariables>(
        session,
        runnerInputChannel,
        {
          store,
          snapshotCallback: config.snapshotCallback,
          isClientManagedState: !config.store,
          lastSnapshot: snapshot,
          onDetach: (snapshotId) => {
            detachedSnapshotId = snapshotId;
            if (resolveDetach) {
              resolveDetach();
            }

            if (store.onSnapshotStateChange) {
              unsubscribe = store.onSnapshotStateChange(
                snapshotId,
                (snap) => {
                  if (snap.status === 'aborted') {
                    abortController.abort();
                    if (unsubscribe) unsubscribe();
                  }
                },
                { context: getContext() }
              );
            }
          },

          onEndTurn: (snapshotId) => {
            if (!runner.isDetached) {
              arg.sendChunk({ turnEnd: { snapshotId } });
            }
          },
        }
      );

      session.on('artifactAdded', (a: Artifact) => {
        if (!runner.isDetached) {
          arg.sendChunk({ artifact: a });
        }
      });

      const sendChunk = (chunk: SessionFlowStreamChunk<Stream>) => {
        if (!runner.isDetached) {
          arg.sendChunk(chunk as SessionFlowStreamChunk);
        }
      };

      const flowPromise = (async () => {
        try {
          const result = await fn(runner, {
            sendChunk,
            abortSignal: abortController.signal,
            context: getContext(),
          });
          const finalSnapshotId = await runner.maybeSnapshot('invocationEnd');
          return { result, finalSnapshotId };
        } finally {
          if (unsubscribe) unsubscribe();
        }
      })();

      // We race the background flow execution against the detach signal.
      // If detachment is requested, we yield output metadata early, but allow
      // the flow handler promise to continue its asynchronous completion.
      const outcome = await Promise.race([
        flowPromise,
        detachPromise.then(() => 'detached' as const),
      ]);

      if (outcome === 'detached') {
        return {
          artifacts: [],
          snapshotId: detachedSnapshotId!,
          state: config.store ? undefined : session.getState(),
        };
      }

      const { result, finalSnapshotId } = outcome;

      return {
        artifacts: result.artifacts || [],
        message: result.message,
        snapshotId: finalSnapshotId,
        state: config.store ? undefined : session.getState(),
      };
    }
  );

  const getSnapshotDataAction = defineAction(
    registry,
    {
      name: `${config.name}__getSnapshotData`,
      description: `Gets snapshot data for ${config.name} by snapshotId`,
      actionType: 'session-flow-snapshot',
      inputSchema: z.string(),
      outputSchema: z.any(), // SessionSnapshot Schema
    },
    async (snapshotId) => {
      const store =
        config.store || new InMemorySessionStore<State, InputVariables>();
      return await store.getSnapshot(snapshotId, { context: getContext() });
    }
  );

  const abortSessionFlowAction = defineAction(
    registry,
    {
      name: `${config.name}__abort`,
      description: `Aborts ${config.name} session flow by snapshotId`,
      actionType: 'session-flow',

      inputSchema: z.string(),
      outputSchema: z.void(),
    },
    async (snapshotId) => {
      const store =
        config.store || new InMemorySessionStore<State, InputVariables>();
      const snapshot = await store.getSnapshot(snapshotId, {
        context: getContext(),
      });
      if (snapshot) {
        snapshot.status = 'aborted';
        await store.saveSnapshot(snapshot, { context: getContext() });
      }
    }
  );

  const composite = Object.assign(primaryAction, {
    getSnapshotData: async (
      snapshotId: string,
      options?: SessionStoreOptions
    ) => {
      const store =
        config.store || new InMemorySessionStore<State, InputVariables>();
      return await store.getSnapshot(snapshotId, options);
    },
    abort: async (snapshotId: string, options?: SessionStoreOptions) => {
      const store =
        config.store || new InMemorySessionStore<State, InputVariables>();
      const snapshot = await store.getSnapshot(snapshotId, options);
      if (snapshot) {
        snapshot.status = 'aborted';
        await store.saveSnapshot(snapshot, options);
      }
    },
    getSnapshotDataAction:
      getSnapshotDataAction as unknown as GetSnapshotDataAction<
        State,
        InputVariables
      >,
    abortSessionFlowAction: abortSessionFlowAction as unknown as Action<
      z.ZodString,
      z.ZodVoid
    >,
  });

  return composite as unknown as SessionFlow<State, InputVariables>;
}

/**
 * Registers a Session Flow from an existing PromptAction.
 */
export function defineSessionFlowFromPrompt<
  PromptIn = unknown,
  State = unknown,
>(
  registry: Registry,
  config: {
    promptName: string;
    defaultInput: PromptIn;
    store?: SessionStore<State, PromptIn>;
    snapshotCallback?: SnapshotCallback<State>;
  }
) {
  const fn: SessionFlowFn<any, State, PromptIn> = async (
    sess,
    { sendChunk }
  ) => {
    await sess.run(async (input) => {
      const promptInput =
        sess.session.getState().inputVariables || config.defaultInput;

      const promptAction = (await registry.lookupAction(
        `/prompt/${config.promptName}`
      )) as PromptAction;
      if (!promptAction) {
        throw new Error(`Prompt ${config.promptName} not found`);
      }

      const genOpts = await promptAction.__executablePrompt.render(
        promptInput as unknown as z.ZodTypeAny
      );

      const promptMessageKey = '_genkit_prompt';
      if (genOpts.messages) {
        genOpts.messages = genOpts.messages.map((m) => ({
          ...m,
          metadata: { ...m.metadata, [promptMessageKey]: true },
        }));
      }

      genOpts.messages = [
        ...(genOpts.messages || []),
        ...(sess.session.getMessages() || []),
      ];

      if (input.toolRestarts && input.toolRestarts.length > 0) {
        genOpts.resume = {
          restart: input.toolRestarts as ToolRequestPart[],
        };
      }

      const result = generateStream(registry, genOpts);

      for await (const chunk of result.stream) {
        sendChunk({ modelChunk: chunk });
      }

      const res = await result.response;

      if (res.request?.messages) {
        const msgs = res.request.messages.filter(
          (m) => !m.metadata?.[promptMessageKey]
        );
        if (res.message) {
          msgs.push(res.message);
        }
        sess.session.setMessages(msgs);
      } else if (res.message) {
        sess.session.addMessages([res.message]);
      }

      if (res.finishReason === 'interrupted') {
        const parts =
          res.message?.content?.filter((p) => !!p.toolRequest) || [];
        if (parts.length > 0) {
          sendChunk({
            modelChunk: {
              role: 'tool',
              content: parts,
            },
          });
        }
      }
    });

    const msgs = sess.session.getMessages();
    return {
      artifacts: sess.session.getArtifacts(),
      message: msgs.length > 0 ? msgs[msgs.length - 1] : undefined,
    };
  };

  return defineSessionFlow<unknown, State, PromptIn>(
    registry,
    {
      name: config.promptName,
      store: config.store as SessionStore<State, PromptIn>,
      snapshotCallback: config.snapshotCallback,
    },
    fn
  );
}
