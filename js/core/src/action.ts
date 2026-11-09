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

import type { JSONSchema7 } from 'json-schema';
import * as z from 'zod';
import { getAsyncContext } from './async-context.js';
import { Channel, lazy } from './async.js';
import { getContext, runWithContext, type ActionContext } from './context.js';
import type { ActionType, Registry } from './registry.js';
import {
  assertGenkitSchema,
  parseSchemaAsync,
  type GenkitSchema,
  type InferInput,
  type InferOutput,
  type ProvidedSchema,
} from './schema.js';
import {
  type ActionStreamInput,
  type ActionStreamSubscriber,
  type StreamManager,
} from './streaming.js';
import {
  SPAN_TYPE_ATTR,
  runInNewSpan,
  setCustomMetadataAttributes,
} from './tracing.js';

export {
  StatusCodes,
  StatusSchema,
  statusNameToCode,
  type Status,
} from './statusTypes.js';
export { InMemoryStreamManager, StreamNotFoundError } from './streaming.js';
export type {
  ActionStreamInput,
  ActionStreamSubscriber,
  JSONSchema7,
  StreamManager,
};

const makeNoopAbortSignal = () => new AbortController().signal;

/**
 * Action metadata.
 */
export interface ActionMetadata<
  I extends GenkitSchema = GenkitSchema,
  O extends GenkitSchema = GenkitSchema,
  S extends GenkitSchema = GenkitSchema,
> {
  /** The type of action (e.g. 'prompt', 'flow'). */
  actionType?: ActionType;
  /** Registry key (unique identifier). */
  key?: string;
  /** The name of the action. */
  name: string;
  /** Description of the action. */
  description?: string;
  /** Input Zod schema. */
  inputSchema?: I;
  /** Input JSON schema. */
  inputJsonSchema?: JSONSchema7;
  /** Output Zod schema. */
  outputSchema?: O;
  /** Output JSON schema. */
  outputJsonSchema?: JSONSchema7;
  /** Stream Zod schema. */
  streamSchema?: S;
  /** Schema of the initialization data. */
  initSchema?: GenkitSchema;
  /** JSON schema of the initialization data. */
  initJsonSchema?: JSONSchema7;
  /** Metadata for the action. */
  metadata?: Record<string, any>;
}

/** Zod schema for {@link ActionMetadata}. */
export const ActionMetadataSchema = z.object({
  key: z.string().optional(),
  actionType: z.string().optional(),
  name: z.string(),
  description: z.string().optional(),
  inputSchema: z.unknown().optional(),
  inputJsonSchema: z.record(z.string(), z.any()).optional(),
  outputSchema: z.unknown().optional(),
  outputJsonSchema: z.record(z.string(), z.any()).optional(),
  streamSchema: z.unknown().optional(),
  initSchema: z.unknown().optional(),
  initJsonSchema: z.record(z.string(), z.any()).optional(),
  metadata: z.record(z.string(), z.any()).optional(),
});

/**
 * Results of an action run. Includes telemetry.
 */
export interface ActionResult<O> {
  result: O;
  telemetry: {
    traceId: string;
    spanId: string;
  };
}

/**
 * Options (side channel) data to pass to the model.
 */
export interface ActionRunOptions<S, Init = any> {
  /**
   * Streaming callback (optional).
   */
  onChunk?: StreamingCallback<S>;

  /**
   * Additional runtime context data (ex. auth context data).
   */
  context?: ActionContext;

  /**
   * Additional span attributes to apply to OT spans.
   */
  telemetryLabels?: Record<string, string>;

  /**
   * Abort signal for the action request.
   */
  abortSignal?: AbortSignal;

  /**
   * Callback that fires immediately when the root span is created for the action,
   * providing early access to telemetry data (trace ID, span ID).
   * This is useful for scenarios where telemetry needs to be available before the action completes.
   * Note: This only fires once for the root action span, not for nested spans.
   */
  onTraceStart?: (traceInfo: { traceId: string; spanId: string }) => void;

  /**
   * Initialization data provided to the action.
   */
  init?: Init;
}

/**
 * Options (side channel) data to pass to the model for bi-directional actions.
 */
export interface BidiActionRunOptions<S, I = any, Init = any>
  extends ActionRunOptions<S, Init> {
  /**
   * Streaming input (optional).
   */
  inputStream?: AsyncIterable<I>;
}

/**
 * Options (side channel) data to pass to the model.
 */
export interface ActionFnArg<S, I = any, Init = any> {
  /**
   * Whether the caller of the action requested streaming.
   */
  streamingRequested: boolean;

  /**
   * Streaming callback (optional).
   */
  sendChunk: StreamingCallback<S>;

  /**
   * Additional runtime context data (ex. auth context data).
   */
  context?: ActionContext;

  /**
   * Trace context containing trace and span IDs.
   */
  trace: {
    traceId: string;
    spanId: string;
  };

  /**
   * Abort signal for the action request.
   */
  abortSignal: AbortSignal;

  registry?: Registry;

  /**
   * Streaming input.
   *
   * The action runtime always provides this. For single-input (unary) actions
   * it is a stream containing just the one input item.
   */
  inputStream: AsyncIterable<I>;

  /**
   * Initialization data provided to the action.
   */
  init: Init;
}

/**
 * Streaming response from an action.
 */
export interface StreamingResponse<
  O extends GenkitSchema = GenkitSchema,
  S extends GenkitSchema = GenkitSchema,
> {
  /** Iterator over the streaming chunks. */
  stream: AsyncGenerator<InferOutput<S>>;
  /** Final output of the action. */
  output: Promise<InferOutput<O>>;
}

/**
 * Streaming response from a bi-directional action.
 */
export interface BidiStreamingResponse<
  O extends GenkitSchema = GenkitSchema,
  S extends GenkitSchema = GenkitSchema,
  I extends GenkitSchema = GenkitSchema,
> extends StreamingResponse<O, S> {
  /**
   * Sends a chunk of data to the action (for bi-directional streaming).
   */
  send(chunk: InferInput<I>): void;
  /**
   * Closes the input stream to the action.
   */
  close(): void;
}

/**
 * Self-describing, validating, observable, locally and remotely callable function.
 */
export type Action<
  I extends GenkitSchema = GenkitSchema,
  O extends GenkitSchema = GenkitSchema,
  S extends GenkitSchema = GenkitSchema,
  RunOptions extends ActionRunOptions<
    InferOutput<S>,
    InferOutput<Init>
    // unfortunately we can't use Init generic param to type ActionRunOptions here (using any)
    // this is somewhat mitigated in BidiAction interface, so anyone interacting with
    // bidi actions will still get properly typed init option.
  > = ActionRunOptions<InferOutput<S>, any>,
  Init extends GenkitSchema = GenkitSchema,
> = ((
  input?: InferInput<I>,
  options?: RunOptions
) => Promise<InferOutput<O>>) & {
  /** @hidden */
  __action: ActionMetadata<I, O, S>;
  /** @hidden */
  __registry?: Registry;
  run(
    input?: InferInput<I>,
    options?: RunOptions
  ): Promise<ActionResult<InferOutput<O>>>;

  stream(input?: InferInput<I>, opts?: RunOptions): StreamingResponse<O, S>;
};

export interface BidiAction<
  IS extends GenkitSchema = GenkitSchema,
  O extends GenkitSchema = GenkitSchema,
  OS extends GenkitSchema = GenkitSchema,
  Init extends GenkitSchema = GenkitSchema,
  RunOptions extends BidiActionRunOptions<
    InferOutput<OS>,
    InferInput<IS>,
    InferInput<Init>
  > = BidiActionRunOptions<InferOutput<OS>, InferInput<IS>, InferInput<Init>>,
> extends Action<IS, O, OS, RunOptions, Init> {
  streamBidi(
    init?: InferInput<Init>,
    opts?: RunOptions
  ): BidiStreamingResponse<O, OS, IS>;
}

/**
 * Action factory params.
 */
export type ActionParams<
  I extends GenkitSchema,
  O extends GenkitSchema,
  S extends GenkitSchema = GenkitSchema,
  Init extends GenkitSchema = GenkitSchema,
> = {
  /**
   * Name of the action, or an object with pluginId and actionId.
   */
  name:
    | string
    | {
        pluginId: string;
        actionId: string;
      };
  /**
   * Description of the action.
   */
  description?: string;
  /**
   * Input Zod schema.
   */
  inputSchema?: I;
  /**
   * Input JSON schema.
   */
  inputJsonSchema?: JSONSchema7;
  /**
   * Output Zod schema.
   */
  outputSchema?: O;
  /**
   * Output JSON schema.
   */
  outputJsonSchema?: JSONSchema7;
  /**
   * Metadata for the action.
   */
  metadata?: Record<string, any>;
  /**
   * Middleware to apply to the action.
   */
  use?: Middleware<InferInput<I>, InferOutput<O>, InferOutput<S>>[];
  /**
   * Stream Zod schema.
   */
  streamSchema?: S;
  /**
   * The type of action.
   */
  actionType: ActionType;
  /**
   * Zod schema for the initialization data.
   */
  initSchema?: Init;
  /**
   * JSON schema for the initialization data.
   */
  initJsonSchema?: JSONSchema7;
};

/**
 * Configuration for an async action (lazy loaded).
 */
export type ActionAsyncParams<
  I extends GenkitSchema,
  O extends GenkitSchema,
  S extends GenkitSchema = GenkitSchema,
> = ActionParams<I, O, S> & {
  fn: (
    input: InferOutput<I>,
    options: ActionFnArg<InferOutput<S>, InferOutput<I>>
  ) => Promise<InferInput<O>>;
};

/**
 * Simple middleware that only modifies request/response.
 */
export type SimpleMiddleware<I = any, O = any> = (
  req: I,
  next: (req?: I) => Promise<O>
) => Promise<O>;

/**
 * Middleware that has access to options (including streaming callback).
 */
export type MiddlewareWithOptions<I = any, O = any, S = any> = (
  req: I,
  options: ActionRunOptions<S> | undefined,
  next: (req?: I, options?: ActionRunOptions<S>) => Promise<O>
) => Promise<O>;

/**
 * Middleware function for actions.
 */
export type Middleware<I = any, O = any, S = any> =
  | SimpleMiddleware<I, O>
  | MiddlewareWithOptions<I, O, S>;

/**
 * Creates an action with provided middleware.
 */
export function actionWithMiddleware<
  I extends GenkitSchema,
  O extends GenkitSchema,
  S extends GenkitSchema = GenkitSchema,
  Init extends GenkitSchema = GenkitSchema,
>(
  action: Action<I, O, S, any, Init>,
  middleware: Middleware<InferInput<I>, InferOutput<O>, InferOutput<S>>[]
): Action<I, O, S, any, Init> {
  const wrapped = (async (
    req: InferInput<I>,
    options?: ActionRunOptions<InferOutput<S>, InferInput<Init>>
  ) => {
    return (await wrapped.run(req, options)).result;
  }) as Action<I, O, S, any, Init>;
  wrapped.__action = action.__action;
  wrapped.run = async (
    req: InferInput<I>,
    options?: ActionRunOptions<InferOutput<S>, InferInput<Init>>
  ): Promise<ActionResult<InferOutput<O>>> => {
    let telemetry;
    const dispatch = async (
      index: number,
      req: InferInput<I>,
      opts?: ActionRunOptions<InferOutput<S>, InferInput<Init>>
    ) => {
      if (index === middleware.length) {
        // end of the chain, call the original model action
        const result = await action.run(req, opts);
        telemetry = result.telemetry;
        return result.result;
      }

      const currentMiddleware = middleware[index];
      if (currentMiddleware.length === 3) {
        return (
          currentMiddleware as MiddlewareWithOptions<
            InferInput<I>,
            InferOutput<O>,
            InferOutput<S>
          >
        )(req, opts, async (modifiedReq, modifiedOptions) =>
          dispatch(index + 1, modifiedReq || req, modifiedOptions || opts)
        );
      } else if (currentMiddleware.length === 2) {
        return (
          currentMiddleware as SimpleMiddleware<InferInput<I>, InferOutput<O>>
        )(req, async (modifiedReq) =>
          dispatch(index + 1, modifiedReq || req, opts)
        );
      } else {
        throw new Error('unspported middleware function shape');
      }
    };
    wrapped.stream = action.stream;
    if ((action as any as BidiAction).streamBidi) {
      (wrapped as BidiAction<I, O, S, Init>).streamBidi = (
        action as BidiAction<I, O, S, Init>
      ).streamBidi;
    }

    return { result: await dispatch(0, req, options), telemetry };
  };
  return wrapped;
}

/**
 * Creates an action with the provided config.
 */
export function action<
  I extends GenkitSchema,
  O extends GenkitSchema,
  S extends GenkitSchema = GenkitSchema,
  Init extends GenkitSchema = GenkitSchema,
>(
  config: ActionParams<I, O, S, Init>,
  fn: (
    input: InferOutput<I>,
    options: ActionFnArg<InferOutput<S>, InferOutput<I>, InferOutput<Init>>
  ) => Promise<InferInput<O>>
): Action<I, O, S, any, Init> {
  const actionName =
    typeof config.name === 'string'
      ? config.name
      : `${config.name.pluginId}/${config.name.actionId}`;
  // Fail early, naming the action, for schemas genkit can't use (typically a
  // plugin bundling zod v3).
  for (const key of [
    'inputSchema',
    'outputSchema',
    'streamSchema',
    'initSchema',
  ] as const) {
    if (config[key]) {
      assertGenkitSchema(
        config[key],
        `${key} of ${config.actionType} "${actionName}"`
      );
    }
  }
  const actionMetadata = {
    key: `/${config.actionType}/${actionName}`,
    name: actionName,
    description: config.description,
    inputSchema: config.inputSchema,
    inputJsonSchema: config.inputJsonSchema,
    outputSchema: config.outputSchema,
    outputJsonSchema: config.outputJsonSchema,
    streamSchema: config.streamSchema,
    initSchema: config.initSchema,
    initJsonSchema: config.initJsonSchema,
    metadata: config.metadata,
    actionType: config.actionType,
  } as ActionMetadata<I, O, S>;

  const actionFn = (async (
    input?: InferInput<I>,
    options?: ActionRunOptions<InferOutput<S>, InferInput<Init>>
  ) => {
    return (await actionFn.run(input, options)).result;
  }) as Action<I, O, S, any, Init>;
  actionFn.__action = { ...actionMetadata };

  actionFn.run = async (
    rawInput: InferInput<I>,
    rawOptions?: ActionRunOptions<InferOutput<S>, InferInput<Init>> & {
      inputStream?: AsyncIterable<InferInput<I>>;
    }
  ): Promise<ActionResult<InferOutput<O>>> => {
    // Parse with the schemas: defaults, transforms and refinements apply, and
    // fn receives the schema's output type. Without a schema this is a no-op.
    const inputSchema: ProvidedSchema = {
      schema: config.inputSchema,
      jsonSchema: config.inputJsonSchema,
    };
    const rawInputStream = rawOptions?.inputStream;
    // With an input stream the items are parsed instead; the unary input is
    // passed through as is (bidi callers send undefined).
    const input = rawInputStream
      ? (rawInput as unknown as InferOutput<I>)
      : await parseSchemaAsync<InferOutput<I>>(rawInput, inputSchema);
    const options: ActionRunOptions<InferOutput<S>, InferOutput<Init>> & {
      inputStream?: AsyncIterable<InferOutput<I>>;
    } = {
      ...rawOptions,
      inputStream: rawInputStream
        ? (async function* () {
            for await (const item of rawInputStream) {
              yield await parseSchemaAsync<InferOutput<I>>(item, inputSchema);
            }
          })()
        : undefined,
      init:
        config.initSchema || config.initJsonSchema
          ? await parseSchemaAsync<InferOutput<Init>>(rawOptions?.init, {
              schema: config.initSchema,
              jsonSchema: config.initJsonSchema,
            })
          : (rawOptions?.init as InferOutput<Init> | undefined),
    };

    let traceId;
    let spanId;
    const genkitKey = actionFn.__action.key;
    const output = await runInNewSpan(
      {
        metadata: {
          name: actionName,
          // Seed input (and init) up front so the realtime "pending" span
          // export carries them; both are already resolved by this point.
          // Both are recorded as the caller sent them (pre-parse): transforms
          // aren't idempotent, so replaying a parsed input (Dev UI rerun,
          // trace replay) could fail or apply them twice.
          input: rawInput,
          ...(rawOptions?.init !== undefined ? { init: rawOptions.init } : {}),
        },
        labels: {
          [SPAN_TYPE_ATTR]: 'action',
          'genkit:metadata:subtype': config.actionType,
          ...(genkitKey ? { 'genkit:key': genkitKey } : {}),
          ...options?.telemetryLabels,
        },
      },
      async (metadata, span) => {
        setCustomMetadataAttributes({
          subtype: config.actionType,
        });
        if (options?.context) {
          try {
            const tracedContext: Record<string, unknown> = {
              ...options.context,
            };
            if ('auth' in tracedContext) tracedContext.auth = '<redacted>';
            if ('secrets' in tracedContext)
              tracedContext.secrets = '<redacted>';
            setCustomMetadataAttributes({
              context: JSON.stringify(tracedContext),
            });
          } catch (e) {
            // Safely ignore or log telemetry serialization errors to prevent crashing the action
          }
        }

        traceId = span.spanContext().traceId;
        spanId = span.spanContext().spanId;
        if (options?.onTraceStart) {
          options.onTraceStart({ traceId, spanId });
        }
        metadata.name = actionName;

        try {
          const actFn = () =>
            fn(input, {
              ...options,
              // Context can either be explicitly set, or inherited from the parent action.
              context: {
                ...actionFn.__registry?.context,
                ...(options?.context ?? getContext()),
              },
              streamingRequested:
                !!options?.onChunk &&
                options.onChunk !== sentinelNoopStreamingCallback,
              sendChunk: options?.onChunk ?? sentinelNoopStreamingCallback,
              inputStream:
                options?.inputStream ?? asyncIterableFromArray([input]),
              trace: {
                traceId,
                spanId,
              },
              registry: actionFn.__registry,
              abortSignal: options?.abortSignal ?? makeNoopAbortSignal(),
            } as ActionFnArg<
              InferOutput<S>,
              InferOutput<I>,
              InferOutput<Init>
            >);
          // if context is explicitly passed in, we run action with the provided context,
          // otherwise we let upstream context carry through.
          const rawOutput = await runWithContext(options?.context, actFn);
          const output = await parseSchemaAsync<InferOutput<O>>(rawOutput, {
            schema: config.outputSchema,
            jsonSchema: config.outputJsonSchema,
          });
          metadata.output = JSON.stringify(output);
          return output;
        } catch (err) {
          if (typeof err === 'object') {
            (err as any).traceId = traceId;
          }
          throw err;
        }
      }
    );
    return {
      result: output,
      telemetry: {
        traceId,
        spanId,
      },
    };
  };

  actionFn.stream = (
    input?: InferInput<I>,
    opts?: ActionRunOptions<InferOutput<S>, InferInput<Init>> & {
      inputStream?: AsyncIterable<InferInput<I>>;
    }
  ): StreamingResponse<O, S> => {
    let chunkStreamController: ReadableStreamDefaultController<InferOutput<S>>;
    const chunkStream = new ReadableStream<InferOutput<S>>({
      start(controller) {
        chunkStreamController = controller;
      },
      pull() {},
      cancel() {},
    });

    const invocationPromise = actionFn
      .run(
        // run() parses the input (parsing twice would re-apply transforms).
        input as InferInput<I>,
        {
          onChunk: ((chunk: InferOutput<S>) => {
            chunkStreamController.enqueue(chunk);
          }) as S extends z.ZodVoid
            ? undefined
            : StreamingCallback<InferOutput<S>>,
          context: {
            ...actionFn.__registry?.context,
            ...(opts?.context ?? getContext()),
          },
          inputStream: opts?.inputStream,
          abortSignal: opts?.abortSignal,
          telemetryLabels: opts?.telemetryLabels,
          init: opts?.init,
        }
      )
      .then((s) => s.result)
      .finally(() => {
        try {
          chunkStreamController.close();
        } catch (e) {
          // Ignore if already closed or cancelled
        }
      });

    return {
      output: invocationPromise,
      stream: (async function* () {
        const reader = chunkStream.getReader();
        try {
          while (true) {
            const chunk = await reader.read();
            if (chunk.value) {
              yield chunk.value;
            }
            if (chunk.done) {
              break;
            }
          }
          return await invocationPromise;
        } finally {
          // Catch prevents unhandled promise rejection if the stream was already cleanly finalized
          reader.cancel().catch(() => {});
        }
      })(),
    };
  };

  if (config.use) {
    return actionWithMiddleware(actionFn, config.use);
  }
  return actionFn;
}

export function isAction(a: unknown): a is Action {
  return typeof a === 'function' && '__action' in a;
}

/**
 * Defines an action with the given config and registers it in the registry.
 */
export function defineAction<
  I extends GenkitSchema,
  O extends GenkitSchema,
  S extends GenkitSchema = GenkitSchema,
  Init extends GenkitSchema = GenkitSchema,
>(
  registry: Registry,
  config: ActionParams<I, O, S, Init>,
  fn: (
    input: InferOutput<I>,
    options: ActionFnArg<InferOutput<S>, InferOutput<I>, InferOutput<Init>>
  ) => Promise<InferInput<O>>
): Action<I, O, S, ActionRunOptions<InferOutput<S>, InferInput<Init>>, Init> {
  if (isInRuntimeContext()) {
    throw new Error(
      'Cannot define new actions at runtime.\n' +
        'See: https://github.com/genkit-ai/genkit/blob/main/docs/errors/no_new_actions_at_runtime.md'
    );
  }
  const act = action(
    config,
    async (i: InferOutput<I>, options): Promise<InferInput<O>> => {
      await registry.initializeAllPlugins();
      return await runInActionRuntimeContext(() => fn(i, options));
    }
  );
  act.__action.actionType = config.actionType;
  registry.registerAction(config.actionType, act);
  return act;
}

/**
 * Defines a bi-directional action with the given config and registers it in the registry.
 */
export function defineBidiAction<
  IS extends GenkitSchema,
  O extends GenkitSchema,
  OS extends GenkitSchema = GenkitSchema,
  Init extends GenkitSchema = GenkitSchema,
>(
  registry: Registry,
  config: ActionParams<IS, O, OS, Init>,
  fn: (
    input: ActionFnArg<InferOutput<OS>, InferOutput<IS>, InferOutput<Init>>
  ) => AsyncGenerator<InferOutput<OS>, InferInput<O>, void>
): BidiAction<IS, O, OS, Init> {
  const act = bidiAction(config, (input) => {
    return (async function* () {
      await registry.initializeAllPlugins();
      return yield* fn(input);
    })();
  });
  registry.registerAction(config.actionType, act);
  return act;
}

/**
 * Creates a bi-directional action with the given config.
 */
export function bidiAction<
  IS extends GenkitSchema,
  O extends GenkitSchema,
  OS extends GenkitSchema = GenkitSchema,
  Init extends GenkitSchema = GenkitSchema,
>(
  config: ActionParams<IS, O, OS, Init>,
  fn: (
    input: ActionFnArg<InferOutput<OS>, InferOutput<IS>, InferOutput<Init>>
  ) => AsyncGenerator<InferOutput<OS>, InferInput<O>, void>
): BidiAction<IS, O, OS, Init> {
  const meta = { ...config.metadata, bidi: true };
  const act = action({ ...config, metadata: meta }, async (input, options) => {
    const stream = options.inputStream;

    const outputGen = fn({
      ...options,
      inputStream: stream,
    } as ActionFnArg<InferOutput<OS>, InferOutput<IS>, InferOutput<Init>>);

    const iter = outputGen[Symbol.asyncIterator]();
    let result: InferInput<O>;
    try {
      while (true) {
        const { value, done } = await iter.next();
        if (done) {
          result = value;
          break;
        }
        options.sendChunk(value);
      }
    } finally {
      if (iter.return) {
        await iter.return(undefined as any);
      }
    }
    return result;
  }) as unknown as BidiAction<IS, O, OS, Init>;

  act.streamBidi = (init, opts) => {
    let channel: Channel<InferInput<IS>> | undefined;
    let stream = opts?.inputStream;
    if (!stream) {
      channel = new Channel<InferInput<IS>>();
      stream = channel;
    }

    const result = act.stream(undefined, {
      ...opts,
      init: init,
      inputStream: stream,
    });

    return {
      ...result,
      stream: (async function* () {
        try {
          for await (const chunk of result.stream) {
            yield chunk;
          }
        } finally {
          if (channel) {
            channel.close();
          }
        }
      })(),
      send: (chunk) => {
        if (!channel) {
          throw new Error('Cannot send to a provided stream.');
        }
        channel.send(chunk);
      },
      close: () => {
        if (!channel) {
          throw new Error('Cannot close a provided stream.');
        }
        channel.close();
      },
    };
  };

  return act;
}

/**
 * Defines an action with the given config promise and registers it in the registry.
 */
export function defineActionAsync<
  I extends GenkitSchema,
  O extends GenkitSchema,
  S extends GenkitSchema = GenkitSchema,
>(
  registry: Registry,
  actionType: ActionType,
  name:
    | string
    | {
        pluginId: string;
        actionId: string;
      },
  config: PromiseLike<ActionAsyncParams<I, O, S>>,
  onInit?: (action: Action<I, O, S>) => void
): PromiseLike<Action<I, O, S>> {
  const actionName =
    typeof name === 'string' ? name : `${name.pluginId}/${name.actionId}`;
  const actionPromise = lazy(() =>
    config.then((resolvedConfig) => {
      const act = action(
        resolvedConfig,
        async (i: InferOutput<I>, options): Promise<InferInput<O>> => {
          await registry.initializeAllPlugins();
          return await runInActionRuntimeContext(() =>
            resolvedConfig.fn(i, options)
          );
        }
      );
      act.__action.actionType = actionType;
      act.__action.key = `/${actionType}/${actionName}`;
      onInit?.(act);
      return act;
    })
  );
  registry.registerActionAsync(actionType, actionName, actionPromise);
  return actionPromise;
}

/** Callback function invoked with each streaming chunk during action execution. */
export type StreamingCallback<T> = (chunk: T) => void;

const streamingAlsKey = 'core.action.streamingCallback';
export const sentinelNoopStreamingCallback = () => null;

/**
 * Executes provided function with streaming callback in async local storage which can be retrieved
 * using {@link getStreamingCallback}.
 */
export function runWithStreamingCallback<S, O>(
  streamingCallback: StreamingCallback<S> | undefined,
  fn: () => O
): O {
  return getAsyncContext().run(
    streamingAlsKey,
    streamingCallback || sentinelNoopStreamingCallback,
    fn
  );
}

/**
 * Retrieves the {@link StreamingCallback} previously set by {@link runWithStreamingCallback}
 *
 * @hidden
 */
export function getStreamingCallback<S>(): StreamingCallback<S> | undefined {
  const cb = getAsyncContext().getStore<StreamingCallback<S>>(streamingAlsKey);
  if (cb === sentinelNoopStreamingCallback) {
    return undefined;
  }
  return cb;
}

const runtimeContextAslKey = 'core.action.runtimeContext';

/**
 * Checks whether the caller is currently in the runtime context of an action.
 */
export function isInRuntimeContext() {
  return getAsyncContext().getStore(runtimeContextAslKey) === 'runtime';
}

/**
 * Execute the provided function in the action runtime context.
 */
export function runInActionRuntimeContext<R>(fn: () => R) {
  return getAsyncContext().run(runtimeContextAslKey, 'runtime', fn);
}

/**
 * Execute the provided function outside the action runtime context.
 */
export function runOutsideActionRuntimeContext<R>(fn: () => R) {
  return getAsyncContext().run(runtimeContextAslKey, 'outside', fn);
}

async function* asyncIterableFromArray<T>(array: T[]): AsyncIterable<T> {
  for (const item of array) {
    yield item;
  }
}
