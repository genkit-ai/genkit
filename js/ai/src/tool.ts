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

import {
  action,
  ActionFnArg,
  assertUnstable,
  isAction,
  stripUndefinedProps,
  z,
  type Action,
  type ActionContext,
  type ActionRunOptions,
  type JSONSchema7,
} from '@genkit-ai/core';
import type { Registry } from '@genkit-ai/core/registry';
import { parseSchema, toJsonSchema } from '@genkit-ai/core/schema';
import { setCustomMetadataAttributes } from '@genkit-ai/core/tracing';
import {
  type Part,
  type ToolDefinition,
  type ToolRequestPart,
  type ToolResponsePart,
} from './model.js';
import { MultipartToolResponseSchema } from './parts.js';
import { isExecutablePrompt, type ExecutablePrompt } from './prompt.js';

export interface Resumable<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
> {
  /**
   * respond constructs a tool response corresponding to the provided interrupt tool request
   * using the provided reply data, validating it against the output schema of the tool if
   * it exists.
   *
   * @beta
   */
  respond(
    /** The interrupt tool request to which you want to respond. */
    interrupt: ToolRequestPart,
    /**
     * The data with which you want to respond. Must conform to a tool's output schema or an
     * interrupt's input schema.
     **/
    outputData: z.infer<O>,
    options?: { metadata?: Record<string, any> }
  ): ToolResponsePart;

  /**
   * restart constructs a tool request corresponding to the provided interrupt tool request
   * that will then re-trigger the tool after e.g. a user confirms. The `resumedMetadata`
   * supplied to this method will be passed to the tool to allow for custom handling of
   * restart logic.
   *
   * @param interrupt The interrupt tool request you want to restart.
   * @param resumedMetadata The metadata you want to provide to the tool to aide in reprocessing. Defaults to `true` if none is supplied.
   * @param options Additional options for restarting the tool.
   *
   * @beta
   */
  restart(
    interrupt: ToolRequestPart,
    resumedMetadata?: any,
    options?: {
      /**
       * Replace the existing input arguments to the tool with different ones, for example
       * if the user revised an action before confirming. When input is replaced, the existing
       * tool request will be amended in the message history.
       **/
      replaceInput?: z.infer<I>;
    }
  ): ToolRequestPart;
}

/**
 * An action with a `tool` type.
 */
export type ToolAction<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
> = Action<I, O, z.ZodTypeAny, ToolRunOptions> &
  Resumable<I, O> & {
    __action: {
      metadata: {
        type: 'tool';
      };
    };
  };

/**
 * An action with a `tool.v2` type.
 */
export type MultipartToolAction<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
> = Action<
  I,
  typeof MultipartToolResponseSchema,
  z.ZodTypeAny,
  ToolRunOptions
> &
  Resumable<I, O> & {
    __action: {
      metadata: {
        type: 'tool.v2';
      };
    };
  };

/**
 * A dynamic action with a `tool` type. Dynamic tools are detached actions -- not associated with any registry.
 */
export type DynamicToolAction<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
> = Action<I, O, z.ZodTypeAny, ToolRunOptions> & {
  /** @deprecated no-op, for backwards compatibility only. */
  attach(registry: Registry): ToolAction<I, O>;
} & Resumable<I, O> & {
    __action: {
      metadata: {
        type: 'tool';
      };
    };
  };

export interface ToolRunOptions extends ActionRunOptions<z.ZodTypeAny> {
  /**
   * If resumed is supplied to a tool at runtime, that means that it was previously interrupted and this is a second
   * @beta
   **/
  resumed?: boolean | Record<string, any>;
  /** The metadata from the tool request that triggered this run. */
  metadata?: Record<string, any>;
}

/**
 * Configuration for a tool.
 */
export interface ToolConfig<I extends z.ZodTypeAny, O extends z.ZodTypeAny> {
  /** Unique name of the tool to use as a key in the registry. */
  name: string;
  /** Description of the tool. This is passed to the model to help understand what the tool is used for. */
  description: string;
  /** Input Zod schema. Mutually exclusive with `inputJsonSchema`. */
  inputSchema?: I;
  /** Input JSON schema. Mutually exclusive with `inputSchema`. */
  inputJsonSchema?: JSONSchema7;
  /** Output Zod schema. Mutually exclusive with `outputJsonSchema`. */
  outputSchema?: O;
  /** Output JSON schema. Mutually exclusive with `outputSchema`. */
  outputJsonSchema?: JSONSchema7;
  /** Metadata to be passed to the tool. */
  metadata?: Record<string, any>;
}

/**
 * A reference to a tool in the form of a name, definition, or the action itself.
 */
export type ToolArgument<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
> = string | ToolAction<I, O> | Action<I, O> | ExecutablePrompt<any, any, any>;

/**
 * Converts an action to a tool action by setting the appropriate metadata.
 */
export function asTool<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
  registry: Registry,
  action: Action<I, O>
): ToolAction<I, O> {
  if (action.__action?.metadata?.type === 'tool') {
    return action as ToolAction<I, O>;
  }

  const fn = ((input) => {
    setCustomMetadataAttributes({ subtype: 'tool' });
    return action(input);
  }) as ToolAction<I, O>;
  fn.__action = {
    ...action.__action,
    metadata: { ...action.__action.metadata, type: 'tool' },
  };
  return fn;
}

/**
 * Resolves a mix of various formats of tool references to a list of tool actions by looking them up in the registry.
 */
export async function resolveTools<
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  tools?: (ToolArgument | ToolDefinition)[]
): Promise<ToolAction[]> {
  if (!tools || tools.length === 0) {
    return [];
  }

  return await Promise.all(
    tools.map(async (ref): Promise<ToolAction> => {
      if (typeof ref === 'string') {
        return await lookupToolByName(registry, ref);
      } else if (isAction(ref)) {
        return asTool(registry, ref);
      } else if (isExecutablePrompt(ref)) {
        return await ref.asTool();
      } else if ((ref as ToolDefinition).name) {
        return await lookupToolByName(
          registry,
          (ref as ToolDefinition).metadata?.originalName ||
            (ref as ToolDefinition).name
        );
      }
      throw new Error('Tools must be strings, tool definitions, or actions.');
    })
  );
}

export async function lookupToolByName(
  registry: Registry,
  name: string
): Promise<ToolAction> {
  const tool =
    (await registry.lookupAction(name)) ||
    (await registry.lookupAction(`/tool/${name}`)) ||
    (await registry.lookupAction(`/tool.v2/${name}`)) ||
    (await registry.lookupAction(`/prompt/${name}`)) ||
    (await registry.lookupAction(`/dynamic-action-provider/${name}`));
  if (!tool) {
    throw new Error(`Tool ${name} not found`);
  }
  return tool as ToolAction;
}

/**
 * Converts a tool action to a definition of the tool to be passed to a model.
 */
export function toToolDefinition(
  tool: Action<z.ZodTypeAny, z.ZodTypeAny>
): ToolDefinition {
  const originalName = tool.__action.name;
  let name = originalName;
  if (originalName.includes('/')) {
    name = originalName.substring(originalName.lastIndexOf('/') + 1);
  }

  const out: ToolDefinition = {
    name,
    description: tool.__action.description || '',
    outputSchema: toJsonSchema({
      schema: tool.__action.outputSchema ?? z.void(),
      jsonSchema: tool.__action.outputJsonSchema,
    })!,
    inputSchema: toJsonSchema({
      schema: tool.__action.inputSchema ?? z.void(),
      jsonSchema: tool.__action.inputJsonSchema,
    })!,
  };

  if (originalName !== name) {
    out.metadata = { originalName };
  }

  return out;
}

export interface ToolFnOptions extends ActionFnArg<never> {
  /**
   * A function that can be called during tool execution that will result in the tool
   * getting interrupted (immediately) and tool request returned to the upstream caller.
   */
  interrupt: (metadata?: Record<string, any>) => never;

  context: ActionContext;
}

export type ToolFn<I extends z.ZodTypeAny, O extends z.ZodTypeAny> = (
  input: z.infer<I>,
  ctx: ToolFnOptions & ToolRunOptions
) => Promise<z.infer<O>>;

export type MultipartToolFn<I extends z.ZodTypeAny, O extends z.ZodTypeAny> = (
  input: z.infer<I>,
  ctx: ToolFnOptions & ToolRunOptions
) => Promise<{
  output?: z.infer<O>;
  content?: Part[];
}>;

export function defineTool<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
  registry: Registry,
  config: { multipart: true } & ToolConfig<I, O>,
  fn?: ToolFn<I, O>
): MultipartToolAction<I, O>;
export function defineTool<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
  registry: Registry,
  config: ToolConfig<I, O>,
  fn?: ToolFn<I, O>
): ToolAction<I, O>;

/**
 * Defines a tool.
 *
 * A tool is an action that can be passed to a model to be called automatically if it so chooses.
 */
export function defineTool<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
  registry: Registry,
  config: { multipart?: true } & ToolConfig<I, O>,
  fn?: ToolFn<I, O> | MultipartToolFn<I, O>
): ToolAction<I, O> | MultipartToolAction<I, O> {
  const a = tool(config, fn);
  delete a.__action.metadata.dynamic;
  registry.registerAction(config.multipart ? 'tool.v2' : 'tool', a);
  if (!config.multipart) {
    // For non-multipart tools, we register a v2 tool action as well
    registry.registerAction('tool.v2', basicToolV2(config, fn as ToolFn<I, O>));
  }
  return a as ToolAction<I, O>;
}

function implementTool<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
  a: ToolAction<I, O>,
  config: ToolConfig<I, O>,
  registry?: Registry
) {
  (a as ToolAction<I, O>).respond = (interrupt, responseData, options) => {
    if (registry) {
      assertUnstable(
        registry,
        'beta',
        "The 'tool.reply' method is part of the 'interrupts' beta feature."
      );
    }
    parseSchema(responseData, {
      jsonSchema: config.outputJsonSchema,
      schema: config.outputSchema,
    });
    return {
      toolResponse: stripUndefinedProps({
        name: interrupt.toolRequest.name,
        ref: interrupt.toolRequest.ref,
        output: responseData,
      }),
      metadata: {
        interruptResponse: options?.metadata || true,
      },
    };
  };

  (a as ToolAction<I, O>).restart = (interrupt, resumedMetadata, options) => {
    if (registry) {
      assertUnstable(
        registry,
        'beta',
        "The 'tool.restart' method is part of the 'interrupts' beta feature."
      );
    }
    let replaceInput = options?.replaceInput;
    if (replaceInput) {
      replaceInput = parseSchema(replaceInput, {
        schema: config.inputSchema,
        jsonSchema: config.inputJsonSchema,
      });
    }
    return {
      toolRequest: stripUndefinedProps({
        name: interrupt.toolRequest.name,
        ref: interrupt.toolRequest.ref,
        input: replaceInput || interrupt.toolRequest.input,
      }),
      metadata: stripUndefinedProps({
        ...interrupt.metadata,
        resumed: resumedMetadata || true,
        // annotate the original input if replacing it
        replacedInput: replaceInput ? interrupt.toolRequest.input : undefined,
      }),
    };
  };
}

/** InterruptConfig defines the options for configuring an interrupt. */
export type InterruptConfig<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  R extends z.ZodTypeAny = z.ZodTypeAny,
> = ToolConfig<I, R> & {
  /** requestMetadata adds additional `interrupt` metadata to the `toolRequest` generated by the interrupt */
  requestMetadata?:
    | Record<string, any>
    | ((
        input: z.infer<I>
      ) => Record<string, any> | Promise<Record<string, any>>);
};

export function isToolRequest(part: Part): part is ToolRequestPart {
  return !!part.toolRequest;
}

export function isToolResponse(part: Part): part is ToolResponsePart {
  return !!part.toolResponse;
}

export function isDynamicTool(
  t: unknown
): t is ToolAction | MultipartToolAction {
  return isAction(t) && t.__action?.metadata?.dynamic === true;
}
export function isMultipartTool(t: unknown): t is MultipartToolAction {
  return isAction(t) && t.__action?.metadata?.type === 'tool.v2';
}

export function interrupt<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
  config: InterruptConfig<I, O>
): ToolAction<I, O> {
  const { requestMetadata, ...toolConfig } = config;

  return tool<I, O>(toolConfig, async (input, { interrupt }) => {
    if (!config.requestMetadata) interrupt();
    else if (typeof config.requestMetadata === 'object')
      interrupt(config.requestMetadata);
    else interrupt(await Promise.resolve(config.requestMetadata(input)));
  });
}

export function defineInterrupt<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
  registry: Registry,
  config: InterruptConfig<I, O>
): ToolAction<I, O> {
  const i = interrupt(config);
  registry.registerAction('tool', i);
  return i;
}

/**
 * Thrown when tools execution is interrupted. It's meant to be caugh by the framework, not public API.
 */
export class ToolInterruptError extends Error {
  constructor(readonly metadata?: Record<string, any>) {
    super();
    this.name = 'ToolInterruptError';
  }
}

/**
 * Interrupts current tool execution causing tool request to be returned in the generation response.
 * Should only be called within a tool.
 */
function interruptTool(registry?: Registry) {
  return (metadata?: Record<string, any>): never => {
    if (registry) {
      assertUnstable(registry, 'beta', 'Tool interrupts are a beta feature.');
    }
    throw new ToolInterruptError(metadata);
  };
}

export function tool<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
  config: { multipart: true } & ToolConfig<I, O>,
  fn?: ToolFn<I, O>
): MultipartToolAction<I, O>;
export function tool<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
  config: ToolConfig<I, O>,
  fn?: ToolFn<I, O>
): ToolAction<I, O>;

/**
 * Defines a dynamic tool. Dynamic tools are just like regular tools but will not be registered in the
 * Genkit registry and can be defined dynamically at runtime.
 */
export function tool<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
  config: { multipart?: true } & ToolConfig<I, O>,
  fn?: ToolFn<I, O> | MultipartToolFn<I, O>
): ToolAction<I, O> | MultipartToolAction<I, O> {
  return config.multipart ? multipartTool(config, fn) : basicTool(config, fn);
}

function basicTool<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
  config: ToolConfig<I, O>,
  fn?: ToolFn<I, O>
): ToolAction<I, O> {
  const a = action(
    {
      ...config,
      actionType: 'tool',
      metadata: { ...(config.metadata || {}), type: 'tool', dynamic: true },
    },
    (i, runOptions) => {
      const interrupt = interruptTool(runOptions.registry);
      if (fn) {
        return fn(i, {
          ...runOptions,
          context: { ...runOptions.context },
          interrupt,
        });
      }
      return interrupt();
    }
  ) as ToolAction<I, O>;
  implementTool(a, config);
  return a;
}

function basicToolV2<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
  config: ToolConfig<I, O>,
  fn?: ToolFn<I, O>
): MultipartToolAction<I, O> {
  return multipartTool(config, async (input, ctx) => {
    if (!fn) {
      const interrupt = interruptTool(ctx.registry);
      return interrupt();
    }
    return {
      output: await fn(input, ctx),
    };
  });
}

function multipartTool<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
  config: ToolConfig<I, O>,
  fn?: MultipartToolFn<I, O>
): MultipartToolAction<I, O> {
  const a = action(
    {
      ...config,
      outputSchema: MultipartToolResponseSchema,
      actionType: 'tool.v2',
      metadata: {
        ...(config.metadata || {}),
        type: 'tool.v2',
        tool: { multipart: true },
        dynamic: true,
      },
    },
    (i, runOptions) => {
      const interrupt = interruptTool(runOptions.registry);
      if (fn) {
        return fn(i, {
          ...runOptions,
          context: { ...runOptions.context },
          interrupt,
        });
      }
      return interrupt() as any; // we cast to any because `interrupt` throws.
    }
  ) as MultipartToolAction<I, O>;
  implementTool(a as any, config);
  return a;
}

/**
 * Defines a dynamic tool. Dynamic tools are just like regular tools but will not be registered in the
 * Genkit registry and can be defined dynamically at runtime.
 *
 * @deprecated renamed to {@link tool}.
 */
export function dynamicTool<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
  config: ToolConfig<I, O>,
  fn?: ToolFn<I, O>
): DynamicToolAction<I, O> {
  const t = basicTool(config, fn) as DynamicToolAction<I, O>;
  t.attach = (_: Registry) => t;
  return t;
}
