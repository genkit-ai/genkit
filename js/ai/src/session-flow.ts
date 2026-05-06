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
  run,
  z,
  type Action,
  type ActionContext,
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
import {
  definePrompt,
  type PromptAction,
  type PromptConfig,
} from './prompt.js';
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
  runWithSession,
  type SessionStoreOptions,
} from './session.js';

/**
 * Schema for initializing an agent turn.
 */
export const AgentInitSchema = z.object({
  snapshotId: z.string().optional(),
  newSnapshotId: z.string().optional(),
  state: SessionStateSchema.optional(),
});

/**
 * Initialization options for an agent turn.
 */
export interface AgentInit<S = unknown> {
  snapshotId?: string;
  newSnapshotId?: string;
  state?: SessionState<S>;
}

/**
 * Schema for agent input messages and commands.
 */
export const AgentInputSchema = z.object({
  messages: z.array(MessageSchema).optional(),
  toolRestarts: z.array(PartSchema).optional(),
  detach: z.boolean().optional(),
});

/**
 * Input received by an agent turn.
 */
export type AgentInput = z.infer<typeof AgentInputSchema>;

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
 * Schema for stream chunks emitted during agent execution.
 */
export const AgentStreamChunkSchema = z.object({
  modelChunk: ModelResponseChunkSchema.optional(),
  status: z.any().optional(),
  artifact: ArtifactSchema.optional(),
  turnEnd: TurnEndSchema.optional(),
});

/**
 * Streamed chunk emitted during agent execution.
 * The `Stream` parameter types the `status` field for custom status payloads.
 */
export type AgentStreamChunk<Stream = unknown> = Omit<
  z.infer<typeof AgentStreamChunkSchema>,
  'status'
> & { status?: Stream };

/**
 * Schema for final results of an agent execution.
 */
export const AgentResultSchema = z.object({
  message: MessageSchema.optional(),
  artifacts: z.array(ArtifactSchema).optional(),
});

/**
 * Result returned upon completing an agent execution.
 */
export type AgentResult = z.infer<typeof AgentResultSchema>;

/**
 * Schema for output returned at turn completion.
 */
export const AgentOutputSchema = z.object({
  snapshotId: z.string().optional(),
  state: SessionStateSchema.optional(),
  message: MessageSchema.optional(),
  artifacts: z.array(ArtifactSchema).optional(),
});

/**
 * Output returned at turn completion.
 */
export interface AgentOutput<S = unknown> {
  artifacts?: Artifact[];
  message?: MessageData;
  snapshotId?: string;
  state?: SessionState<S>;
}

/**
 * Executor responsible for running turns over input streams and persisting state.
 */
export class SessionRunner<State = unknown> {
  readonly session: Session<State>;
  readonly inputCh: AsyncIterable<AgentInput>;
  turnIndex: number = 0;
  public onEndTurn?: (snapshotId?: string) => void;
  public onDetach?: (snapshotId: string) => void;
  public newSnapshotId?: string;
  private snapshotCallback?: SnapshotCallback<State>;
  private lastSnapshot?: SessionSnapshot<State>;

  private lastSnapshotVersion: number = 0;
  private store?: SessionStore<State>;
  public isDetached: boolean = false;

  constructor(
    session: Session<State>,
    inputCh: AsyncIterable<AgentInput>,
    options?: {
      snapshotCallback?: SnapshotCallback<State>;
      lastSnapshot?: SessionSnapshot<State>;
      store?: SessionStore<State>;
      onEndTurn?: (snapshotId?: string) => void;
      onDetach?: (snapshotId: string) => void;
      newSnapshotId?: string;
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
  }

  /**
   * Executes the flow handler against incoming input messages sequentially.
   */
  async run(fn: (input: AgentInput) => Promise<void>): Promise<void> {
    for await (const input of this.inputCh) {
      if (input.messages) {
        this.session.addMessages(input.messages);
      }

      const turnSnapshotId = this.newSnapshotId || crypto.randomUUID();
      this.newSnapshotId = undefined;

      try {
        await run(`runTurn-${this.turnIndex + 1}`, input, async () => {
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
          return {
            lastSnapshot: this.lastSnapshot,
          };
        });
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

    const snapshot: SessionSnapshot<State> = {
      snapshotId: snapshotId || this.newSnapshotId || crypto.randomUUID(),
      createdAt: new Date().toISOString(),
      event: event,
      state: currentState as SessionState<State>,
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
 * Optional transform applied to session state before it is exposed to the
 * client (e.g. in `AgentOutput.state` or via `getSnapshotData`).  This lets
 * agents redact sensitive fields or reshape the state for the client.
 */
export type ClientStateTransform<S = unknown> = (
  state: SessionState<S>
) => SessionState;

/**
 * Function handler definition for custom agent actions.
 */
export type AgentFn<Stream, State> = (
  sess: SessionRunner<State>,
  options: {
    sendChunk: (chunk: AgentStreamChunk<Stream>) => void;
    abortSignal?: AbortSignal;
    context?: ActionContext;
  }
) => Promise<AgentResult>;

export type GetSnapshotDataAction<S = unknown> = Action<
  z.ZodString,
  z.ZodType<SessionSnapshot<S>>
>;

/**
 * Represents a configured, registered Agent.
 */
export interface Agent<State = unknown>
  extends BidiAction<
    typeof AgentInputSchema,
    typeof AgentOutputSchema,
    typeof AgentStreamChunkSchema,
    typeof AgentInitSchema
  > {
  getSnapshotData(
    snapshotId: string,
    options?: SessionStoreOptions
  ): Promise<SessionSnapshot<State> | undefined>;

  abort(
    snapshotId: string,
    options?: SessionStoreOptions
  ): Promise<SessionSnapshot['status'] | undefined>;

  readonly getSnapshotDataAction: GetSnapshotDataAction<State>;
  readonly abortAgentAction: Action<z.ZodString, z.ZodType<string | undefined>>;
}

/**
 * Registers a multi-turn custom agent action capable of maintaining persistent state.
 */
export function defineCustomAgent<Stream = unknown, State = unknown>(
  registry: Registry,
  config: {
    name: string;
    description?: string;
    store?: SessionStore<State>;
    snapshotCallback?: SnapshotCallback<State>;
    clientStateTransform?: ClientStateTransform<State>;
  },
  fn: AgentFn<Stream, State>
): Agent<State> {
  // Helper that applies the optional transform before exposing state to the
  // client.  When no transform is configured it returns the raw state.
  const toClientState = (
    state: SessionState<State>
  ): SessionState | undefined => {
    if (config.clientStateTransform) {
      return config.clientStateTransform(state);
    }
    return state as SessionState;
  };

  const primaryAction = defineBidiAction(
    registry,
    {
      name: config.name,
      description: config.description,
      actionType: 'agent',
      inputSchema: AgentInputSchema,
      outputSchema: AgentOutputSchema,
      streamSchema: AgentStreamChunkSchema,
      initSchema: AgentInitSchema,
      metadata: {
        stateManagement: config.store ? 'server' : 'client',
        abortable: !!config.store?.onSnapshotStateChange,
      },
    },
    async function* (
      arg: ActionFnArg<AgentStreamChunk, AgentInput, AgentInit>
    ) {
      const init = arg.init;
      const store = config.store || new InMemorySessionStore<State>();

      let session: Session<State>;

      let snapshot: SessionSnapshot<State> | undefined;

      if (init?.snapshotId) {
        snapshot = await store.getSnapshot(init.snapshotId, {
          context: getContext(),
        });
        if (!snapshot) {
          throw new Error(`Snapshot ${init.snapshotId} not found`);
        }
        session = new Session<State>(snapshot.state as SessionState<State>);
      } else if (init?.state && !config.store) {
        session = new Session<State>(init.state as SessionState<State>);
      } else {
        session = new Session<State>({
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

      let runner!: SessionRunner<State>;

      // We construct an asynchronous proxy channel over the inputStream.
      // This enables immediate interception of `detach: true` directives. Without this proxy,
      // a backlog of pre-queued inputs would have to be resolved sequentially by the runner first.
      const runnerInputChannel = new Channel<AgentInput>();

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
              // Only forward to runner if the input carries a payload beyond the
              // detach directive; a detach-only message has no turn to process.
              const hasPayload = !!(
                input.messages?.length || input.toolRestarts?.length
              );
              if (hasPayload) {
                runnerInputChannel.send(input);
              }
            } else {
              runnerInputChannel.send(input);
            }
          }
          runnerInputChannel.close();
        } catch (e) {
          runnerInputChannel.error(e);
        }
      })();

      runner = new SessionRunner<State>(session, runnerInputChannel, {
        store,
        snapshotCallback: config.snapshotCallback,
        lastSnapshot: snapshot,
        newSnapshotId: init?.newSnapshotId,
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
      });

      const sendArtifactChunk = (a: Artifact) => {
        if (!runner.isDetached) {
          arg.sendChunk({ artifact: a });
        }
      };
      session.on('artifactAdded', sendArtifactChunk);
      session.on('artifactUpdated', sendArtifactChunk);

      const sendChunk = (chunk: AgentStreamChunk<Stream>) => {
        if (!runner.isDetached) {
          arg.sendChunk(chunk as AgentStreamChunk);
        }
      };

      const flowPromise = (async () => {
        try {
          const result = await runWithSession(registry, session, () =>
            fn(runner, {
              sendChunk,
              abortSignal: abortController.signal,
              context: getContext(),
            })
          );
          const finalSnapshotId = await runner.maybeSnapshot('invocationEnd');
          return { result, finalSnapshotId };
        } finally {
          if (unsubscribe) unsubscribe();
          session.off('artifactAdded', sendArtifactChunk);
          session.off('artifactUpdated', sendArtifactChunk);
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
          state: config.store ? undefined : toClientState(session.getState()),
        };
      }

      const { result, finalSnapshotId } = outcome;

      return {
        artifacts: result.artifacts || [],
        message: result.message,
        snapshotId: finalSnapshotId,
        state: config.store ? undefined : toClientState(session.getState()),
      };
    }
  );

  // Helper that applies the clientStateTransform to a snapshot's state,
  // returning a new snapshot object with the transformed state.
  const toClientSnapshot = (
    snapshot: SessionSnapshot<State>
  ): SessionSnapshot => {
    if (!config.clientStateTransform) {
      return snapshot as SessionSnapshot;
    }
    return {
      ...snapshot,
      state: config.clientStateTransform(snapshot.state),
    };
  };

  const getSnapshotDataAction = defineAction(
    registry,
    {
      name: config.name,
      description: `Gets snapshot data for ${config.name} by snapshotId`,
      actionType: 'agent-snapshot',
      inputSchema: z.string(),
      outputSchema: z.any(), // SessionSnapshot Schema
    },
    async (snapshotId) => {
      if (!config.store) {
        throw new GenkitError({
          status: 'FAILED_PRECONDITION',
          message: `getSnapshotData requires a persistent store. Provide a 'store' when defining '${config.name}'.`,
        });
      }
      const snapshot = await config.store.getSnapshot(snapshotId, {
        context: getContext(),
      });
      return snapshot ? toClientSnapshot(snapshot) : undefined;
    }
  );

  const abortAgentAction = defineAction(
    registry,
    {
      name: config.name,
      description: `Aborts ${config.name} agent by snapshotId. Returns the previous status of the snapshot before it was set to 'aborted', or undefined if the snapshot was not found.`,
      actionType: 'agent-abort',
      inputSchema: z.string(),
      outputSchema: z.string().optional(),
    },
    async (snapshotId) => {
      if (!config.store) {
        throw new GenkitError({
          status: 'FAILED_PRECONDITION',
          message: `abort requires a persistent store. Provide a 'store' when defining '${config.name}'.`,
        });
      }
      const snapshot = await config.store.getSnapshot(snapshotId, {
        context: getContext(),
      });
      if (snapshot) {
        const previousStatus = snapshot.status;
        snapshot.status = 'aborted';
        await config.store.saveSnapshot(snapshot, { context: getContext() });
        return previousStatus;
      }
      return undefined;
    }
  );

  const composite = Object.assign(primaryAction, {
    getSnapshotData: async (
      snapshotId: string,
      options?: SessionStoreOptions
    ) => {
      if (!config.store) {
        throw new GenkitError({
          status: 'FAILED_PRECONDITION',
          message: `getSnapshotData requires a persistent store. Provide a 'store' when defining '${config.name}'.`,
        });
      }
      const snapshot = await config.store.getSnapshot(snapshotId, options);
      return snapshot ? toClientSnapshot(snapshot) : undefined;
    },
    abort: async (snapshotId: string, options?: SessionStoreOptions) => {
      if (!config.store) {
        throw new GenkitError({
          status: 'FAILED_PRECONDITION',
          message: `abort requires a persistent store. Provide a 'store' when defining '${config.name}'.`,
        });
      }
      const snapshot = await config.store.getSnapshot(snapshotId, options);
      if (snapshot) {
        const previousStatus = snapshot.status;
        snapshot.status = 'aborted';
        await config.store.saveSnapshot(snapshot, options);
        return previousStatus;
      }
      return undefined;
    },
    getSnapshotDataAction:
      getSnapshotDataAction as unknown as GetSnapshotDataAction<State>,
    abortAgentAction: abortAgentAction as unknown as Action<
      z.ZodString,
      z.ZodType<string | undefined>
    >,
  });

  return composite as unknown as Agent<State>;
}

/**
 * Registers an agent from an existing PromptAction.
 */
export function definePromptAgent<State = unknown>(
  registry: Registry,
  config: {
    promptName: string;
    store?: SessionStore<State>;
    snapshotCallback?: SnapshotCallback<State>;
    clientStateTransform?: ClientStateTransform<State>;
  }
) {
  let cachedPromptAction: PromptAction | undefined;

  const fn: AgentFn<unknown, State> = async (
    sess,
    { sendChunk, abortSignal }
  ) => {
    await sess.run(async (input) => {
      const promptInput = {};

      if (!cachedPromptAction) {
        cachedPromptAction = (await registry.lookupAction(
          `/prompt/${config.promptName}`
        )) as PromptAction;
        if (!cachedPromptAction) {
          throw new Error(
            `Prompt '${config.promptName}' not found. Ensure it is defined before the agent is invoked.`
          );
        }
      }

      const historyTag = '_genkit_history';
      const promptTag = 'agentPreamble';

      // Tag every history message so we can identify them after render.
      const history = (sess.session.getMessages() || []).map((m) => ({
        ...m,
        metadata: { ...m.metadata, [historyTag]: true },
      }));

      // Let the prompt control where history is placed (e.g. dotprompt
      // {{history}}).  When the prompt has no explicit `messages` config
      // the render helper simply appends history after system/user.
      const genOpts = await cachedPromptAction.__executablePrompt.render(
        promptInput as unknown as z.ZodTypeAny,
        { messages: history }
      );

      // After render: tag everything that is NOT history as a prompt
      // message so we can strip it after generation.  Also strip the
      // internal history tag — it is an implementation detail that
      // should not leak to the model.
      if (genOpts.messages) {
        genOpts.messages = genOpts.messages.map((m) => {
          if (m.metadata?.[historyTag]) {
            // Strip the history tag before sending to the model.
            const { [historyTag]: _, ...restMeta } = m.metadata!;
            return {
              ...m,
              metadata: Object.keys(restMeta).length ? restMeta : undefined,
            };
          }
          return { ...m, metadata: { ...m.metadata, [promptTag]: true } };
        });
      }

      if (input.toolRestarts && input.toolRestarts.length > 0) {
        genOpts.resume = {
          restart: input.toolRestarts as ToolRequestPart[],
        };
      }

      const result = generateStream(registry, { ...genOpts, abortSignal });

      for await (const chunk of result.stream) {
        sendChunk({ modelChunk: chunk });
      }

      const res = await result.response;

      // Keep everything that is NOT a prompt-template message:
      //   • history messages (clean — history tag was stripped before generate)
      //   • new messages from tool loops (untagged)
      //   • model response
      if (res.request?.messages) {
        const msgs = res.request.messages.filter(
          (m) => !m.metadata?.[promptTag]
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

  return defineCustomAgent<unknown, State>(
    registry,
    {
      name: config.promptName,
      store: config.store,
      snapshotCallback: config.snapshotCallback,
      clientStateTransform: config.clientStateTransform,
    },
    fn
  );
}

// ---------------------------------------------------------------------------
// defineAgent — shortcut that combines definePrompt + definePromptAgent
// ---------------------------------------------------------------------------

/**
 * Configuration for `defineAgent`, which combines prompt definition and agent
 * registration into a single call.
 */
export interface AgentConfig<State = unknown> extends PromptConfig {
  store?: SessionStore<State>;
  snapshotCallback?: SnapshotCallback<State>;
  clientStateTransform?: ClientStateTransform<State>;
}

/**
 * Defines and registers an agent by creating a prompt and wiring it into a
 * multi-turn agent in one step.
 *
 * This is a convenience shortcut for:
 * ```ts
 * definePrompt(registry, promptConfig);
 * definePromptAgent(registry, { promptName: promptConfig.name, ... });
 * ```
 */
export function defineAgent<State = unknown>(
  registry: Registry,
  config: AgentConfig<State>
): Agent<State> {
  // Extract prompt-specific fields from the combined config.
  const { store, snapshotCallback, clientStateTransform, ...promptConfig } =
    config;

  // Register the prompt.
  definePrompt(registry, promptConfig);

  // Wire it into a prompt agent.
  return definePromptAgent<State>(registry, {
    promptName: promptConfig.name,
    store,
    snapshotCallback,
    clientStateTransform,
  });
}
