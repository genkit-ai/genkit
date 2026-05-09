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
import { Action, ActionMetadata, Middleware, action } from './action.js';
import { ActionContext } from './context.js';
import { GenkitError } from './error.js';
import { ActionType, Registry } from './registry.js';
import { toJsonSchema } from './schema.js';

/**
 * Zod schema of an opration representing a background task.
 */
export const OperationSchema = z.object({
  action: z.string().optional(),
  id: z.string(),
  done: z.boolean().optional(),
  output: z.any().optional(),
  error: z.object({ message: z.string() }).passthrough().optional(),
  metadata: z.record(z.string(), z.any()).optional(),
});

/**
 * Background operation.
 */
export interface Operation<O = any> {
  action?: string;
  id: string;
  done?: boolean;
  output?: O;
  error?: { message: string; [key: string]: unknown };
  metadata?: Record<string, any>;
}

/**
 * Background action. Unlike regular action, background action can run for a long time in the background.
 * The returned operation can used to check the status of the background operation and retrieve the response.
 */
export interface BackgroundAction<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  RunOptions extends BackgroundActionRunOptions = BackgroundActionRunOptions,
> {
  __action: ActionMetadata<I, O>;

  readonly startAction: Action<I, typeof OperationSchema>;
  readonly checkAction: Action<typeof OperationSchema, typeof OperationSchema>;
  readonly cancelAction?: Action<
    typeof OperationSchema,
    typeof OperationSchema
  >;
  readonly supportsCancel: boolean;

  start(
    input?: z.infer<I>,
    options?: RunOptions
  ): Promise<Operation<z.infer<O>>>;

  check(operation: Operation<z.infer<O>>): Promise<Operation<z.infer<O>>>;

  cancel(operation: Operation<z.infer<O>>): Promise<Operation<z.infer<O>>>;
}

export async function lookupBackgroundAction<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  key: string
): Promise<BackgroundAction<I, O> | undefined> {
  const root: Action<I, typeof OperationSchema> = await registry.lookupAction<
    I,
    typeof OperationSchema,
    Action<I, typeof OperationSchema>
  >(key);
  if (!root) return undefined;
  const actionName = key.substring(key.indexOf('/', 1) + 1);
  return new BackgroundActionImpl(
    root,
    await registry.lookupAction<
      typeof OperationSchema,
      typeof OperationSchema,
      Action<typeof OperationSchema, typeof OperationSchema>
    >(`/check-operation/${actionName}/check`),
    await registry.lookupAction<
      typeof OperationSchema,
      typeof OperationSchema,
      Action<typeof OperationSchema, typeof OperationSchema>
    >(`/cancel-operation/${actionName}/cancel`)
  );
}

class BackgroundActionImpl<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  RunOptions extends BackgroundActionRunOptions = BackgroundActionRunOptions,
> implements BackgroundAction<I, O, RunOptions>
{
  __action: ActionMetadata<I, O>;

  readonly startAction: Action<I, typeof OperationSchema>;
  readonly checkAction: Action<typeof OperationSchema, typeof OperationSchema>;
  readonly cancelAction?: Action<
    typeof OperationSchema,
    typeof OperationSchema
  >;

  constructor(
    startAction: Action<I, typeof OperationSchema>,
    checkAction: Action<typeof OperationSchema, typeof OperationSchema>,
    cancelAction:
      | Action<typeof OperationSchema, typeof OperationSchema>
      | undefined
  ) {
    this.__action = {
      name: startAction.__action.name,
      description: startAction.__action.description,
      inputSchema: startAction.__action.inputSchema,
      inputJsonSchema: startAction.__action.inputJsonSchema,
      metadata: startAction.__action.metadata,
      actionType: startAction.__action.actionType,
    };
    this.startAction = startAction;
    this.checkAction = checkAction;
    this.cancelAction = cancelAction;
  }

  async start(
    input?: z.infer<I>,
    options?: RunOptions
  ): Promise<Operation<z.infer<O>>> {
    return await this.startAction(input, options);
  }

  async check(
    operation: Operation<z.infer<O>>
  ): Promise<Operation<z.infer<O>>> {
    return await this.checkAction(operation);
  }

  get supportsCancel(): boolean {
    return !!this.cancelAction;
  }

  async cancel(
    operation: Operation<z.infer<O>>
  ): Promise<Operation<z.infer<O>>> {
    if (!this.cancelAction) {
      return operation;
    }
    return await this.cancelAction(operation);
  }
}

/**
 * Options (side channel) data to pass to the model.
 */
export interface BackgroundActionRunOptions {
  /**
   * Additional runtime context data (ex. auth context data).
   */
  context?: ActionContext;

  /**
   * Additional span attributes to apply to OT spans.
   */
  telemetryLabels?: Record<string, string>;
}

/**
 * Options (side channel) data to pass to the model.
 */
export interface BackgroundActionFnArg<S> {
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
}

/**
 * Action factory params.
 */
export type BackgroundActionParams<
  I extends z.ZodTypeAny,
  O extends z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
> = {
  name: string;
  start: (
    input: z.infer<I>,
    options: BackgroundActionFnArg<z.infer<S>>
  ) => Promise<Operation<z.infer<O>>>;
  check: (input: Operation<z.infer<O>>) => Promise<Operation<z.infer<O>>>;
  cancel?: (input: Operation<z.infer<O>>) => Promise<Operation<z.infer<O>>>;
  actionType: ActionType;

  description?: string;
  inputSchema?: I;
  inputJsonSchema?: JSONSchema7;
  outputSchema?: O;
  outputJsonSchema?: JSONSchema7;
  metadata?: Record<string, any>;
  use?: Middleware<z.infer<I>, z.infer<O>>[];
  streamSchema?: S;
};

/**
 * Defines an action with the given config and registers it in the registry.
 */
export function defineBackgroundAction<
  I extends z.ZodTypeAny,
  O extends z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  config: BackgroundActionParams<I, O, S>
): BackgroundAction<I, O> {
  const act = backgroundAction(config);

  registerBackgroundAction(registry, act);

  return act;
}

export function registerBackgroundAction(
  registry: Registry,
  act: BackgroundAction<any, any>,
  opts?: { namespace?: string }
) {
  registry.registerAction(
    act.startAction.__action.actionType!,
    act.startAction,
    opts
  );
  registry.registerAction(
    act.checkAction.__action.actionType!,
    act.checkAction,
    opts
  );
  if (act.cancelAction) {
    registry.registerAction(
      act.cancelAction.__action.actionType!,
      act.cancelAction,
      opts
    );
  }
}

/**
 * Creates a background action with the given config.
 */
export function backgroundAction<
  I extends z.ZodTypeAny,
  O extends z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
>(config: BackgroundActionParams<I, O, S>): BackgroundAction<I, O> {
  const startAction = action(
    {
      actionType: config.actionType,
      name: config.name,
      description: config.description,
      inputSchema: config.inputSchema,
      inputJsonSchema: config.inputJsonSchema,
      outputSchema: OperationSchema,
      metadata: {
        ...config.metadata,
        outputSchema: toJsonSchema({
          schema: config.outputSchema,
          jsonSchema: config.outputJsonSchema,
        }),
      },
      use: config.use,
    },
    async (input, options) => {
      const operation = await config.start(input, options);
      operation.action = `/${config.actionType}/${config.name}`;
      return operation;
    }
  );
  const checkAction = action(
    {
      actionType: 'check-operation',
      name: `${config.name}/check`,
      description: config.description,
      inputSchema: OperationSchema,
      inputJsonSchema: config.inputJsonSchema,
      outputSchema: OperationSchema,
      metadata: {
        ...config.metadata,
        outputSchema: toJsonSchema({
          schema: config.outputSchema,
          jsonSchema: config.outputJsonSchema,
        }),
      },
    },
    async (input) => {
      const operation = await config.check(input);
      operation.action = `/${config.actionType}/${config.name}`;
      return operation;
    }
  );
  let cancelAction:
    | Action<typeof OperationSchema, typeof OperationSchema>
    | undefined = undefined;
  if (config.cancel) {
    cancelAction = action(
      {
        actionType: 'cancel-operation',
        name: `${config.name}/cancel`,
        description: config.description,
        inputSchema: OperationSchema,
        inputJsonSchema: config.inputJsonSchema,
        outputSchema: OperationSchema,
        metadata: {
          ...config.metadata,
          outputSchema: toJsonSchema({
            schema: config.outputSchema,
            jsonSchema: config.outputJsonSchema,
          }),
        },
      },
      async (input) => {
        if (!config.cancel) {
          throw new GenkitError({
            status: 'UNAVAILABLE',
            message: `${config.name} does not support cancellation.`,
          });
        }
        const operation = await config.cancel(input);
        operation.action = `/${config.actionType}/${config.name}`;
        return operation;
      }
    );
  }

  return new BackgroundActionImpl(startAction, checkAction, cancelAction);
}

export function isBackgroundAction(a: unknown): a is BackgroundAction {
  return a instanceof BackgroundActionImpl;
}
