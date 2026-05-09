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

import type { z } from 'zod';
import { ActionFnArg, action, type Action } from './action.js';
import { Registry, type HasRegistry } from './registry.js';
import { SPAN_TYPE_ATTR, runInNewSpan } from './tracing.js';

/**
 * Flow is an observable, streamable, (optionally) strongly typed function.
 */
export interface Flow<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
> extends Action<I, O, S> {}

/**
 * Configuration for a streaming flow.
 */
export interface FlowConfig<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
> {
  /** Name of the flow. */
  name: string;
  /** Schema of the input to the flow. */
  inputSchema?: I;
  /** Schema of the output from the flow. */
  outputSchema?: O;
  /** Schema of the streaming chunks from the flow. */
  streamSchema?: S;
  /** Metadata of the flow used by tooling. */
  metadata?: Record<string, any>;
}

/**
 * Flow execution context for flow to access the streaming callback and
 * side-channel context data. The context itself is a function, a short-cut
 * for streaming callback.
 */
export interface FlowSideChannel<S> extends ActionFnArg<S> {
  (chunk: S): void;
}

/**
 * Function to be executed in the flow.
 */
export type FlowFn<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
> = (
  /** Input to the flow. */
  input: z.infer<I>,
  /** Callback for streaming functions only. */
  streamingCallback: FlowSideChannel<z.infer<S>>
) => Promise<z.infer<O>> | z.infer<O>;

/**
 * Defines a  flow. This operates on the currently active registry.
 */
export function flow<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
>(config: FlowConfig<I, O, S> | string, fn: FlowFn<I, O, S>): Flow<I, O, S> {
  const resolvedConfig: FlowConfig<I, O, S> =
    typeof config === 'string' ? { name: config } : config;

  return flowAction(resolvedConfig, fn);
}

/**
 * Defines a non-streaming flow. This operates on the currently active registry.
 */
export function defineFlow<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  config: FlowConfig<I, O, S> | string,
  fn: FlowFn<I, O, S>
): Flow<I, O, S> {
  const f = flow(config, fn);

  registry.registerAction('flow', f);

  return f;
}

/**
 * Registers a flow as an action in the registry.
 */
function flowAction<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
>(config: FlowConfig<I, O, S>, fn: FlowFn<I, O, S>): Flow<I, O, S> {
  return action(
    {
      actionType: 'flow',
      name: config.name,
      inputSchema: config.inputSchema,
      outputSchema: config.outputSchema,
      streamSchema: config.streamSchema,
      metadata: config.metadata,
    },
    async (
      input,
      { sendChunk, context, trace, abortSignal, streamingRequested }
    ) => {
      const ctx = sendChunk;
      (ctx as FlowSideChannel<z.infer<S>>).sendChunk = sendChunk;
      (ctx as FlowSideChannel<z.infer<S>>).context = context;
      (ctx as FlowSideChannel<z.infer<S>>).trace = trace;
      (ctx as FlowSideChannel<z.infer<S>>).abortSignal = abortSignal;
      (ctx as FlowSideChannel<z.infer<S>>).streamingRequested =
        streamingRequested;
      return fn(input, ctx as FlowSideChannel<z.infer<S>>);
    }
  );
}

export function run<T>(
  name: string,
  func: () => Promise<T>,
  _?: Registry
): Promise<T>;

export function run<T>(
  name: string,
  input: any,
  func: (input?: any) => Promise<T>,
  registry?: Registry
): Promise<T>;

/**
 * A flow step that executes the provided function. Each run step is recorded separately in the trace.
 */
export function run<T>(
  name: string,
  funcOrInput: () => Promise<T>,
  fnOrRegistry?: Registry | HasRegistry | ((input?: any) => Promise<T>),
  _?: Registry | HasRegistry
): Promise<T> {
  let func;
  let input;
  let hasInput = false;
  if (typeof funcOrInput === 'function') {
    func = funcOrInput;
  } else {
    input = funcOrInput;
    hasInput = true;
  }
  if (typeof fnOrRegistry === 'function') {
    func = fnOrRegistry;
  }

  if (!func) {
    throw new Error('unable to resolve run function');
  }
  return runInNewSpan(
    {
      metadata: { name },
      labels: {
        [SPAN_TYPE_ATTR]: 'flowStep',
      },
    },
    async (meta) => {
      meta.input = input;
      const output = hasInput ? await func(input) : await func();
      meta.output = JSON.stringify(output);
      return output;
    }
  );
}
