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
import { lazy } from './async.js';
import { getContext, runWithContext, type ActionContext } from './context.js';
import type { ActionType, Registry } from './registry.js';
import { parseSchema } from './schema.js';
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

export { StatusCodes, StatusSchema, type Status } from './statusTypes.js';
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
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
> {
  actionType?: ActionType;
  name: string;
  description?: string;
  inputSchema?: I;
  inputJsonSchema?: JSONSchema7;
  outputSchema?: O;
  outputJsonSchema?: JSONSchema7;
  streamSchema?: S;
  metadata?: Record<string, any>;
}

export const ActionMetadataSchema = z.object({
  actionType: z.string().optional(),
  name: z.string(),
  description: z.string().optional(),
  inputSchema: z.unknown().optional(),
  inputJsonSchema: z.object({}).optional(),
  outputSchema: z.unknown().optional(),
  outputJsonSchema: z.object({}).optional(),
  streamSchema: z.unknown().optional(),
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
export interface ActionRunOptions<S> {
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
}

/**
 * Options (side channel) data to pass to the model.
 */
export interface ActionFnArg<S> {
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
}

/**
 * Streaming response from an action.
 */
export interface StreamingResponse<
  O extends z.ZodTypeAny = z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
> {
  /** Iterator over the streaming chunks. */
  stream: AsyncGenerator<z.infer<S>>;
  /** Final output of the action. */
  output: Promise<z.infer<O>>;
}

/**
 * Self-describing, validating, observable, locally and remotely callable function.
 */
export type Action<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
  RunOptions extends ActionRunOptions<S> = ActionRunOptions<S>,
> = ((input?: z.infer<I>, options?: RunOptions) => Promise<z.infer<O>>) & {
  __action: ActionMetadata<I, O, S>;
  __registry?: Registry;
  run(
    input?: z.infer<I>,
    options?: ActionRunOptions<z.infer<S>>
  ): Promise<ActionResult<z.infer<O>>>;

  stream(
    input?: z.infer<I>,
    opts?: ActionRunOptions<z.infer<S>>
  ): StreamingResponse<O, S>;
};

/**
 * Action factory params.
 */
export type ActionParams<
  I extends z.ZodTypeAny,
  O extends z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
> = {
  name:
    | string
    | {
        pluginId: string;
        actionId: string;
      };
  description?: string;
  inputSchema?: I;
  inputJsonSchema?: JSONSchema7;
  outputSchema?: O;
  outputJsonSchema?: JSONSchema7;
  metadata?: Record<string, any>;
  use?: Middleware<z.infer<I>, z.infer<O>, z.infer<S>>[];
  streamSchema?: S;
  actionType: ActionType;
};

export type ActionAsyncParams<
  I extends z.ZodTypeAny,
  O extends z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
> = ActionParams<I, O, S> & {
  fn: (
    input: z.infer<I>,
    options: ActionFnArg<z.infer<S>>
  ) => Promise<z.infer<O>>;
};

export type SimpleMiddleware<I = any, O = any> = (
  req: I,
  next: (req?: I) => Promise<O>
) => Promise<O>;

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
  I extends z.ZodTypeAny,
  O extends z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
>(
  action: Action<I, O, S>,
  middleware: Middleware<z.infer<I>, z.infer<O>, z.infer<S>>[]
): Action<I, O, S> {
  const wrapped = (async (
    req: z.infer<I>,
    options?: ActionRunOptions<z.infer<S>>
  ) => {
    return (await wrapped.run(req, options)).result;
  }) as Action<I, O, S>;
  wrapped.__action = action.__action;
  wrapped.run = async (
    req: z.infer<I>,
    options?: ActionRunOptions<z.infer<S>>
  ): Promise<ActionResult<z.infer<O>>> => {
    let telemetry;
    const dispatch = async (
      index: number,
      req: z.infer<I>,
      opts?: ActionRunOptions<z.infer<S>>
    ) => {
      if (index === middleware.length) {
        // end of the chain, call the original model action
        const result = await action.run(req, opts);
        telemetry = result.telemetry;
        return result.result;
      }

      const currentMiddleware = middleware[index];
      if (currentMiddleware.length === 3) {
        return (currentMiddleware as MiddlewareWithOptions<I, O, z.infer<S>>)(
          req,
          opts,
          async (modifiedReq, modifiedOptions) =>
            dispatch(index + 1, modifiedReq || req, modifiedOptions || opts)
        );
      } else if (currentMiddleware.length === 2) {
        return (currentMiddleware as SimpleMiddleware<I, O>)(
          req,
          async (modifiedReq) => dispatch(index + 1, modifiedReq || req, opts)
        );
      } else {
        throw new Error('unspported middleware function shape');
      }
    };
    wrapped.stream = action.stream;

    return { result: await dispatch(0, req, options), telemetry };
  };
  return wrapped;
}

/**
 * Creates an action with the provided config.
 */
export function action<
  I extends z.ZodTypeAny,
  O extends z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
>(
  config: ActionParams<I, O, S>,
  fn: (
    input: z.infer<I>,
    options: ActionFnArg<z.infer<S>>
  ) => Promise<z.infer<O>>
): Action<I, O, z.infer<S>> {
  const actionName =
    typeof config.name === 'string'
      ? config.name
      : `${config.name.pluginId}/${config.name.actionId}`;
  const actionMetadata = {
    name: actionName,
    description: config.description,
    inputSchema: config.inputSchema,
    inputJsonSchema: config.inputJsonSchema,
    outputSchema: config.outputSchema,
    outputJsonSchema: config.outputJsonSchema,
    streamSchema: config.streamSchema,
    metadata: config.metadata,
    actionType: config.actionType,
  } as ActionMetadata<I, O, S>;

  const actionFn = (async (
    input?: I,
    options?: ActionRunOptions<z.infer<S>>
  ) => {
    return (await actionFn.run(input, options)).result;
  }) as Action<I, O, z.infer<S>>;
  actionFn.__action = { ...actionMetadata };

  actionFn.run = async (
    input: z.infer<I>,
    options?: ActionRunOptions<z.infer<S>>
  ): Promise<ActionResult<z.infer<O>>> => {
    input = parseSchema(input, {
      schema: config.inputSchema,
      jsonSchema: config.inputJsonSchema,
    });
    let traceId;
    let spanId;
    let output = await runInNewSpan(
      {
        metadata: {
          name: actionName,
        },
        labels: {
          [SPAN_TYPE_ATTR]: 'action',
          'genkit:metadata:subtype': config.actionType,
          ...options?.telemetryLabels,
        },
      },
      async (metadata, span) => {
        setCustomMetadataAttributes({
          subtype: config.actionType,
        });
        if (options?.context) {
          setCustomMetadataAttributes({
            context: JSON.stringify(options.context),
          });
        }

        traceId = span.spanContext().traceId;
        spanId = span.spanContext().spanId;
        if (options?.onTraceStart) {
          options.onTraceStart({ traceId, spanId });
        }
        metadata.name = actionName;
        metadata.input = input;

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
              trace: {
                traceId,
                spanId,
              },
              registry: actionFn.__registry,
              abortSignal: options?.abortSignal ?? makeNoopAbortSignal(),
            });
          // if context is explicitly passed in, we run action with the provided context,
          // otherwise we let upstream context carry through.
          const output = await runWithContext(options?.context, actFn);

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
    output = parseSchema(output, {
      schema: config.outputSchema,
      jsonSchema: config.outputJsonSchema,
    });
    return {
      result: output,
      telemetry: {
        traceId,
        spanId,
      },
    };
  };

  actionFn.stream = (
    input?: z.infer<I>,
    opts?: ActionRunOptions<z.infer<S>>
  ): StreamingResponse<O, S> => {
    let chunkStreamController: ReadableStreamController<z.infer<S>>;
    const chunkStream = new ReadableStream<z.infer<S>>({
      start(controller) {
        chunkStreamController = controller;
      },
      pull() {},
      cancel() {},
    });

    const invocationPromise = actionFn
      .run(config.inputSchema ? config.inputSchema.parse(input) : input, {
        onChunk: ((chunk: z.infer<S>) => {
          chunkStreamController.enqueue(chunk);
        }) as S extends z.ZodVoid ? undefined : StreamingCallback<z.infer<S>>,
        context: {
          ...actionFn.__registry?.context,
          ...(opts?.context ?? getContext()),
        },
        abortSignal: opts?.abortSignal,
        telemetryLabels: opts?.telemetryLabels,
      })
      .then((s) => s.result)
      .finally(() => {
        chunkStreamController.close();
      });

    return {
      output: invocationPromise,
      stream: (async function* () {
        const reader = chunkStream.getReader();
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
  I extends z.ZodTypeAny,
  O extends z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  config: ActionParams<I, O, S>,
  fn: (
    input: z.infer<I>,
    options: ActionFnArg<z.infer<S>>
  ) => Promise<z.infer<O>>
): Action<I, O, S> {
  if (isInRuntimeContext()) {
    throw new Error(
      'Cannot define new actions at runtime.\n' +
        'See: https://github.com/firebase/genkit/blob/main/docs/errors/no_new_actions_at_runtime.md'
    );
  }
  const act = action(config, async (i: I, options): Promise<z.infer<O>> => {
    await registry.initializeAllPlugins();
    return await runInActionRuntimeContext(() => fn(i, options));
  });
  act.__action.actionType = config.actionType;
  registry.registerAction(config.actionType, act);
  return act;
}

/**
 * Defines an action with the given config promise and registers it in the registry.
 */
export function defineActionAsync<
  I extends z.ZodTypeAny,
  O extends z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
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
        async (i: I, options): Promise<z.infer<O>> => {
          await registry.initializeAllPlugins();
          return await runInActionRuntimeContext(() =>
            resolvedConfig.fn(i, options)
          );
        }
      );
      act.__action.actionType = actionType;
      onInit?.(act);
      return act;
    })
  );
  registry.registerActionAsync(actionType, actionName, actionPromise);
  return actionPromise;
}

// Streaming callback function.
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
