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
  assertUnstable,
  GenkitError,
  isAction,
  Operation,
  runWithContext,
  sentinelNoopStreamingCallback,
  type Action,
  type ActionContext,
  type HttpErrorWireFormat,
  type StreamingCallback,
  type z,
} from '@genkit-ai/core';
import { Channel } from '@genkit-ai/core/async';
import { Registry } from '@genkit-ai/core/registry';
import { toJsonSchema } from '@genkit-ai/core/schema';
import type { DocumentData } from './document.js';
import {
  injectInstructions,
  resolveFormat,
  resolveInstructions,
} from './formats/index.js';
import {
  errorToThrow,
  generateHelper,
  partialResponseOf,
  shouldInjectFormatInstructions,
} from './generate/action.js';
import { GenerateResponseChunk } from './generate/chunk.js';
import {
  GenerateMiddleware,
  generateMiddleware,
  GenerateMiddlewareDef,
  resolveMiddleware,
} from './generate/middleware.js';
import { GenerateResponse } from './generate/response.js';
import { Message } from './message.js';
import {
  GenerateResponseChunkData,
  GenerateResponseData,
  ResolvedModel,
  resolveModel,
  type GenerateActionOptions,
  type GenerateRequest,
  type GenerationCommonConfigSchema,
  type MessageData,
  type MiddlewareRef,
  type ModelArgument,
  type ModelMiddlewareArgument,
  type Part,
  type ToolRequestPart,
  type ToolResponsePart,
} from './model.js';
import { isExecutablePrompt } from './prompt.js';
import {
  isDynamicResourceAction,
  resolveResources,
  ResourceAction,
  ResourceArgument,
} from './resource.js';
import {
  isDynamicTool,
  isMultipartTool,
  resolveTools,
  toToolDefinition,
  type ToolArgument,
} from './tool.js';

export { GenerateResponse, GenerateResponseChunk };

/** Specifies how tools should be called by the model. */
export type ToolChoice = 'auto' | 'required' | 'none';

/** Configuration for the desired output format and schema of a generate request. */
export interface OutputOptions<O extends z.ZodTypeAny = z.ZodTypeAny> {
  format?: string;
  contentType?: string;
  instructions?: boolean | string;
  schema?: O;
  jsonSchema?: any;
  constrained?: boolean;
}

/** ResumeOptions configure how to resume generation after an interrupt. */
export interface ResumeOptions {
  /**
   * respond should contain a single or list of `toolResponse` parts corresponding
   * to interrupt `toolRequest` parts from the most recent model message. Each
   * entry must have a matching `name` and `ref` (if supplied) for its `toolRequest`
   * counterpart.
   *
   * Tools have a `.respond` helper method to construct a reply ToolResponse and validate
   * the data against its schema. Call `myTool.respond(interruptToolRequest, yourReplyData)`.
   */
  respond?: ToolResponsePart | ToolResponsePart[];
  /**
   * restart will run a tool again with additionally supplied metadata passed through as
   * a `resumed` option in the second argument. This allows for scenarios like conditionally
   * requesting confirmation of an LLM's tool request.
   *
   * Tools have a `.restart` helper method to construct a restart ToolRequest. Call
   * `myTool.restart(interruptToolRequest, resumeMetadata)`.
   *
   */
  restart?: ToolRequestPart | ToolRequestPart[];
  /** Additional metadata to annotate the created tool message with in the "resume" key. */
  metadata?: Record<string, any>;
}

/**
 * Options for making a generation request using {@link Genkit.generate} or {@link Genkit.generateStream}.
 */
export interface GenerateOptions<
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
> {
  /** A model name (e.g. `vertexai/gemini-1.0-pro`) or reference. */
  model?: ModelArgument<CustomOptions>;
  /** The system prompt to be included in the generate request. Can be a string for a simple text prompt or one or more parts for multi-modal prompts (subject to model support). */
  system?: string | Part | Part[];
  /** The prompt for which to generate a response. Can be a string for a simple text prompt or one or more parts for multi-modal prompts. */
  prompt?: string | Part | Part[];
  /** Retrieved documents to be used as context for this generation. */
  docs?: DocumentData[];
  /** Conversation messages (history) for multi-turn prompting when supported by the underlying model. */
  messages?: (MessageData & { content: Part[] | string | (string | Part)[] })[];
  /** List of registered tool names or actions to treat as a tool for this generation if supported by the underlying model. */
  tools?: ToolArgument[];
  /** List of dynamic resources to be made available to this generate request. */
  resources?: ResourceArgument[];
  /** Specifies how tools should be called by the model.  */
  toolChoice?: ToolChoice;
  /** Configuration for the generation request. */
  config?: z.infer<CustomOptions>;
  /** Configuration for the desired output of the request. Defaults to the model's default output if unspecified. */
  output?: OutputOptions<O>;
  /**
   * resume provides convenient capabilities for continuing generation
   * after an interrupt is triggered. Example:
   *
   * ```ts
   * const myInterrupt = ai.defineInterrupt({...});
   *
   * const response = await ai.generate({
   *   tools: [myInterrupt],
   *   prompt: "Call myInterrupt",
   * });
   *
   * const interrupt = response.interrupts[0];
   *
   * const resumedResponse = await ai.generate({
   *   messages: response.messages,
   *   resume: myInterrupt.respond(interrupt, {note: "this is the reply data"}),
   * });
   * ```
   *
   * @beta
   */
  resume?: ResumeOptions;
  /** When true, return tool calls for manual processing instead of automatically resolving them. */
  returnToolRequests?: boolean;
  /** Maximum number of tool call iterations that can be performed in a single generate call (default 5). */
  maxTurns?: number;
  /** When provided, models supporting streaming will call the provided callback with chunks as generation progresses. */
  onChunk?: StreamingCallback<GenerateResponseChunk>;
  /**
   * When provided, models supporting streaming will call the provided callback with chunks as generation progresses.
   *
   * @deprecated use {@link onChunk} instead.
   */
  streamingCallback?: StreamingCallback<GenerateResponseChunk>;
  /** Middleware to be used with this model call. */
  use?: (ModelMiddlewareArgument | GenerateMiddleware | MiddlewareRef)[];
  /** Additional context (data, like e.g. auth) to be passed down to tools, prompts and other sub actions. */
  context?: ActionContext;
  /** Abort signal for the generate request. */
  abortSignal?: AbortSignal;
  /**
   * Whether a failed generation throws (the default) or resolves with a
   * response that reports it. With `false`, a failure once the request has
   * resolved returns a {@link GenerateResponse} carrying `error` and the
   * conversation the loop completed, ending at a turn seam (see
   * {@link generate}); a failure before that (an unknown model or tool,
   * invalid options) still throws, since there is nothing to return.
   */
  throwOnError?: boolean;
  /** Custom step name for this generate call to display in trace views. Defaults to "generate". */
  stepName?: string;
  /**
   * Additional metadata describing the GenerateOptions, used by tooling. If
   * this is an instance of a rendered dotprompt, will contain any prompt
   * metadata contained in the original frontmatter.
   **/
  metadata?: Record<string, any>;
}

export async function toGenerateRequest(
  registry: Registry,
  options: GenerateOptions
): Promise<GenerateRequest> {
  const messages: MessageData[] = [];
  if (options.system) {
    messages.push({
      role: 'system',
      content: Message.parseContent(options.system),
    });
  }
  if (options.messages) {
    messages.push(...options.messages.map((m) => Message.parseData(m)));
  }
  if (options.prompt) {
    messages.push({
      role: 'user',
      content: Message.parseContent(options.prompt),
    });
  }
  if (messages.length === 0) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: 'at least one message is required in generate request',
    });
  }
  if (
    options.resume &&
    !(
      messages.at(-1)?.role === 'model' &&
      messages.at(-1)?.content.find((p) => !!p.toolRequest)
    )
  ) {
    throw new GenkitError({
      status: 'FAILED_PRECONDITION',
      message: `Last message must be a 'model' role with at least one tool request to 'resume' generation.`,
      detail: messages.at(-1),
    });
  }
  let tools: Action<any, any>[] | undefined;
  if (options.tools) {
    tools = await resolveTools(registry, options.tools);
  }
  let resources: ResourceAction[] | undefined;
  if (options.resources) {
    resources = await resolveResources(registry, options.resources);
  }

  const resolvedSchema = toJsonSchema({
    schema: options.output?.schema,
    jsonSchema: options.output?.jsonSchema,
  });

  const resolvedFormat = await resolveFormat(registry, options.output);
  const instructions = resolveInstructions(
    resolvedFormat,
    resolvedSchema,
    options?.output?.instructions
  );

  const out = {
    messages: shouldInjectFormatInstructions(
      resolvedFormat?.config,
      options.output
    )
      ? injectInstructions(messages, instructions)
      : messages,
    config: options.config,
    docs: options.docs,
    tools: tools?.map(toToolDefinition) || [],
    resources: resources?.map((a) => a.__action) || [],
    output: {
      ...(resolvedFormat?.config || {}),
      ...options.output,
      schema: resolvedSchema,
    },
  } as GenerateRequest;
  if (!out?.output?.schema) delete out?.output?.schema;
  return out;
}

/**
 * An error that includes the partial {@link GenerateResponse} that triggered
 * it, on `detail.response`.
 *
 * A caller catches one when the model returned a blocked response or none, or
 * when the loop refused to run past `maxTurns`, as it always has. The loop
 * also builds one for every other failure once the request has resolved,
 * wrapping the cause (available as `cause`) with the conversation the loop
 * completed on `detail.response`; `generate` hands that response back when
 * asked to return failures, and otherwise throws the cause itself, so the
 * wrapper reaches the `generate` action's own callers (the reflection API)
 * rather than application code. See {@link generate} for the contract.
 *
 * Only the failure's classification travels on the wire: {@link toJSON}
 * carries the status, the message, and the response's finish reason, never
 * the response itself, so an HTTP error body does not repeat the request's
 * messages.
 */
export class GenerationResponseError extends GenkitError {
  detail: {
    response: GenerateResponse;
    [otherDetails: string]: any;
  };

  constructor(
    response: GenerateResponse<any>,
    message: string,
    status?: GenkitError['status'],
    detail?: Record<string, any>,
    options?: {
      /** The underlying error, exposed as the standard `Error.cause`. */
      cause?: unknown;
      /** Overrides the message sent to a client; see {@link GenkitError.publicMessage}. */
      publicMessage?: string;
    }
  ) {
    const cause = options?.cause;
    super({
      status: status || 'FAILED_PRECONDITION',
      message,
      cause,
      publicMessage: options?.publicMessage,
      ...(cause instanceof GenkitError && {
        source: cause.source,
        responseMetadata: cause.responseMetadata,
      }),
    });
    const inherited = cause instanceof GenkitError ? cause.detail : undefined;
    this.detail = {
      ...(inherited &&
      typeof inherited === 'object' &&
      !Array.isArray(inherited)
        ? inherited
        : {}),
      ...detail,
      response,
    };
  }

  /**
   * The wire form: the status, the message, and the response's finish reason
   * and finish message, with the same text a client is allowed to see (see
   * {@link GenkitError.publicMessage}). Neither the response nor a request
   * rides along.
   */
  toJSON(): HttpErrorWireFormat {
    const { response, request: _request, ...details } = this.detail;
    const finishMessage = this.publicMessage ?? response?.finishMessage;
    return {
      details: {
        ...details,
        finishReason: response?.finishReason,
        ...(finishMessage && { finishMessage }),
      },
      status: this.status,
      message: this.publicMessage ?? this.originalMessage,
    };
  }
}

/**
 * Built by the generate loop when it stopped because the caller stopped it
 * rather than because something broke. The partial response on
 * `detail.response` reports `finishReason` `aborted`. A caller catches one
 * for the `maxTurns` limit, which `generate` has always thrown as a
 * {@link GenerationResponseError} with status `ABORTED` and now throws as
 * this subclass; every other stop reaches a caller as the cancellation or
 * timeout error itself, or, with `throwOnError: false`, as a response that
 * reports `aborted`.
 *
 * The rule reads the request's `abortSignal` and the error's identity, never
 * a status: the loop stopped on the caller's behalf when the signal had fired
 * by the time a failure was classified, when the model call rejected with an
 * `AbortError` or `TimeoutError` (what the platform raises for a cancelled or
 * timed-out call, whether or not the loop's own signal fired), or when the
 * loop reached the `maxTurns` limit. A provider answering with a
 * `GenkitError` of status `ABORTED` or `DEADLINE_EXCEEDED` is a failure, not
 * a stop.
 *
 * `status` is `ABORTED` for the turn limit, `CANCELLED` or
 * `DEADLINE_EXCEEDED` for a bare cancellation or timeout, and, when the
 * signal fired while the model reported a `GenkitError`, that error's own
 * status. A tool that failed after the signal fired reports `CANCELLED`.
 */
export class GenerationAbortedError extends GenerationResponseError {}

async function toolsToActionRefs(
  registry: Registry,
  toolOpt?: ToolArgument[]
): Promise<string[] | undefined> {
  if (!toolOpt) return;

  const tools: string[] = [];

  for (const t of toolOpt) {
    if (typeof t === 'string') {
      const names = await resolveFullToolNames(registry, t);
      tools.push(...names);
    } else if (isAction(t) || isDynamicTool(t)) {
      tools.push(`/${t.__action.metadata?.type}/${t.__action.name}`);
    } else if (isExecutablePrompt(t)) {
      const promptToolAction = await t.asTool();
      tools.push(`/prompt/${promptToolAction.__action.name}`);
    } else {
      throw new Error(`Unable to determine type of tool: ${JSON.stringify(t)}`);
    }
  }
  return tools;
}

async function resourcesToActionRefs(
  registry: Registry,
  resOpt?: ResourceArgument[]
): Promise<string[] | undefined> {
  if (!resOpt) return;

  const resources: string[] = [];

  for (const r of resOpt) {
    if (typeof r === 'string') {
      const names = await resolveFullResourceNames(registry, r);
      resources.push(...names);
    } else if (isAction(r)) {
      resources.push(`/resource/${r.__action.name}`);
    } else {
      throw new Error(`Unable to resolve resource: ${JSON.stringify(r)}`);
    }
  }
  return resources;
}

function messagesFromOptions(options: GenerateOptions): MessageData[] {
  const messages: MessageData[] = [];
  if (options.system) {
    messages.push({
      role: 'system',
      content: Message.parseContent(options.system),
    });
  }
  if (options.messages) {
    messages.push(...options.messages);
  }
  if (options.prompt) {
    messages.push({
      role: 'user',
      content: Message.parseContent(options.prompt),
    });
  }
  if (messages.length === 0) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: 'at least one message is required in generate request',
    });
  }
  return messages;
}

/** A GenerationBlockedError is thrown when a generation is blocked. */
export class GenerationBlockedError extends GenerationResponseError {}

/**
 * Normalizes a mix of middleware representations into an array of standardized `MiddlewareRef`s.
 * Any raw functional middleware or unregistered middleware objects are dynamically registered
 * into the provided registry.
 *
 * @param registry The registry to use for looking up or dynamically registering middleware.
 * @param middlewareList An array of middleware functions, instances, or references.
 * @returns A promise resolving to an array of normalized `MiddlewareRef` objects.
 */
export async function normalizeMiddleware(
  registry: Registry,
  middlewareList?: (
    | ModelMiddlewareArgument
    | GenerateMiddleware
    | MiddlewareRef
  )[]
): Promise<MiddlewareRef[]> {
  if (!middlewareList || middlewareList.length === 0) {
    return [];
  }

  const refs: MiddlewareRef[] = [];

  for (let i = 0; i < middlewareList.length; i++) {
    const middleware = middlewareList[i];

    if (
      typeof middleware === 'function' &&
      (middleware as any).instantiate &&
      (middleware as any).plugin
    ) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message: `Middleware ${(middleware as any).name || 'function'} must be called with () when used in 'use' array.`,
      });
    }

    if (typeof middleware === 'function') {
      const name = `dynamic-middleware-${i}-${Math.random().toString(36).slice(2)}`;

      const wrappedDef = generateMiddleware(
        { name, metadata: { dynamic: true } },
        () => ({
          model: async (req, ctx, next) => {
            if (middleware.length === 3) {
              return (middleware as any)(
                req,
                ctx,
                async (modifiedReq: any, opts: any) =>
                  next(modifiedReq || req, opts || ctx)
              );
            } else {
              return (middleware as any)(req, async (modifiedReq: any) =>
                next(modifiedReq || req, ctx)
              );
            }
          },
        })
      );
      registry.registerValue('middleware', name, wrappedDef);
      refs.push({ name });
      continue;
    }

    if (
      typeof middleware === 'object' &&
      middleware !== null &&
      'instantiate' in middleware &&
      typeof middleware.instantiate === 'function'
    ) {
      const def = middleware as GenerateMiddleware;
      const registered = await registry.lookupValue<GenerateMiddleware>(
        'middleware',
        def.name
      );
      if (!registered) {
        registry.registerValue('middleware', def.name, def);
      }
      refs.push({ name: def.name });
      continue;
    }

    if (
      typeof middleware === 'object' &&
      middleware !== null &&
      'name' in middleware
    ) {
      const ref = middleware as MiddlewareRef & { __def?: GenerateMiddleware };
      const registered = await registry.lookupValue<GenerateMiddleware>(
        'middleware',
        ref.name
      );
      if (!registered && ref.__def) {
        registry.registerValue('middleware', ref.name, ref.__def);
      }
      refs.push({ name: ref.name, config: ref.config });
    }
  }

  return refs;
}

/**
 * Generate calls a generative model based on the provided prompt and configuration. If
 * `history` is provided, the generation will include a conversation history in its
 * request. If `tools` are provided, the generate method will automatically resolve
 * tool calls returned from the model unless `returnToolRequests` is set to `true`.
 *
 * See `GenerateOptions` for detailed information about available options.
 *
 * A failed generation throws by default: a tool's own error, the model's own
 * error, the validation error for structured output that does not match the
 * schema, a {@link GenerationBlockedError} for a blocked response, a
 * {@link GenerationResponseError} for a response without a message or for
 * the `maxTurns` limit, and a middleware hook's own error.
 *
 * With `throwOnError: false`, a failure once the request has resolved
 * resolves instead with a {@link GenerateResponse} that reports it. The
 * response's `error` is the failure classified (a status, a message, and the
 * error's own structured details) and its `finishMessage` the same text. It
 * reports `finishReason` `failed` when something broke (a model call, a
 * tool, a hook) and `aborted` when the caller stopped the loop: the
 * `abortSignal` fired, the model call was cancelled or timed out, or
 * `maxTurns` was reached. Such a response has no `message`, and its
 * `messages` end at a turn seam: the messages the failing turn started from,
 * which are the caller's own or a run of completed [model with tool requests,
 * tool with every response] rounds, and nothing from the failing turn, since
 * a conversation ending in a tool request nothing answered is one no provider
 * accepts. A failed tool discards the whole round it opened and is reported
 * as INTERNAL under `tool "<name>" failed`; text streamed before a failure
 * still reached `onChunk`. Three failures keep the model's own message and
 * finish reason beside `error`: a blocked response, a response without a
 * message, and a response the model completed but post-processing rejected
 * (structured output that does not match the schema). A resume whose
 * restarted tool interrupted again keeps `finishReason` `interrupted` under
 * its FAILED_PRECONDITION error and is answered with `resume` rather than
 * sent again.
 *
 * Errors raised before the request resolves (an unknown model, tool, or
 * resource, invalid options, or a `generate` middleware hook that fails
 * before running the first turn) throw in both modes: there is no response
 * to return.
 *
 * @param options The options for this generation request.
 * @returns The generated response based on the provided parameters.
 */
export async function generate<
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = typeof GenerationCommonConfigSchema,
>(
  registry: Registry,
  options:
    | GenerateOptions<O, CustomOptions>
    | PromiseLike<GenerateOptions<O, CustomOptions>>
): Promise<GenerateResponse<z.infer<O>>> {
  const resolvedOptions: GenerateOptions<O, CustomOptions> = {
    ...(await Promise.resolve(options)),
  };
  const resolvedFormat = await resolveFormat(registry, resolvedOptions.output);

  registry = Registry.withParent(registry);

  maybeRegisterDynamicTools(registry, resolvedOptions);
  maybeRegisterDynamicResources(registry, resolvedOptions);

  const middlewareRefs = await normalizeMiddleware(
    registry,
    resolvedOptions.use
  );
  resolvedOptions.use = middlewareRefs; // Cast back because `use` can be generic

  const params = await toGenerateActionOptions(registry, resolvedOptions);

  const tools = await toolsToActionRefs(registry, resolvedOptions.tools);
  const resources = await resourcesToActionRefs(
    registry,
    resolvedOptions.resources
  );
  const streamingCallback = stripNoop(
    resolvedOptions.onChunk ?? resolvedOptions.streamingCallback
  ) as StreamingCallback<GenerateResponseChunkData>;

  const resolvedMiddleware = await resolveMiddleware(registry, middlewareRefs);
  maybeRegisterDynamicMiddlewareTools(registry, resolvedMiddleware);

  let response: GenerateResponseData;
  try {
    response = await runWithContext(resolvedOptions.context, () =>
      generateHelper(registry, {
        rawRequest: params,
        middleware: resolvedMiddleware,
        abortSignal: resolvedOptions.abortSignal,
        streamingCallback,
      })
    );
  } catch (e) {
    // The loop wraps a failure with the conversation it completed. A caller
    // that asked for failures on the response gets that response; any other
    // gets the error the failure threw on its own.
    if (resolvedOptions.throwOnError === false) {
      const partial = partialResponseOf(e, registry);
      if (partial) return partial as GenerateResponse<z.infer<O>>;
    }
    throw errorToThrow(e);
  }
  const request = await toGenerateRequest(registry, {
    ...resolvedOptions,
    tools,
    resources,
  });
  return new GenerateResponse<O>(response, {
    request: response.request ?? request,
    parser: resolvedFormat?.handler(request.output?.schema).parseMessage,
  });
}

export async function generateOperation<
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = typeof GenerationCommonConfigSchema,
>(
  registry: Registry,
  options:
    | GenerateOptions<O, CustomOptions>
    | PromiseLike<GenerateOptions<O, CustomOptions>>
): Promise<Operation<GenerateResponseData>> {
  assertUnstable(registry, 'beta', 'generateOperation is a beta feature.');

  options = await options;
  const resolvedModel = await resolveModel(registry, options.model);
  if (
    !resolvedModel.modelAction.__action.metadata?.model.supports?.longRunning
  ) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: `Model '${resolvedModel.modelAction.__action.name}' does not support long running operations.`,
    });
  }

  const { operation } = await generate(registry, options);
  if (!operation) {
    throw new GenkitError({
      status: 'FAILED_PRECONDITION',
      message: `Model '${resolvedModel.modelAction.__action.name}' did not return an operation.`,
    });
  }
  return operation;
}

export function maybeRegisterDynamicMiddlewareTools(
  registry: Registry,
  middlewares?: GenerateMiddlewareDef[]
) {
  middlewares?.forEach((mw) => {
    mw.tools?.forEach((t) => {
      if (isDynamicTool(t)) {
        registry.registerAction('tool', t as Action);
      }
    });
  });
}

function maybeRegisterDynamicTools<
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = typeof GenerationCommonConfigSchema,
>(registry: Registry, options: GenerateOptions<O, CustomOptions>) {
  options?.tools?.forEach((t) => {
    if (isDynamicTool(t)) {
      if (isMultipartTool(t)) {
        registry.registerAction('tool.v2', t);
      } else {
        registry.registerAction('tool', t as Action);
      }
    }
  });
}

function maybeRegisterDynamicResources<
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = typeof GenerationCommonConfigSchema,
>(registry: Registry, options: GenerateOptions<O, CustomOptions>) {
  options?.resources?.forEach((r) => {
    if (isDynamicResourceAction(r)) {
      registry.registerAction('resource', r);
    }
  });
}

export async function toGenerateActionOptions<
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = typeof GenerationCommonConfigSchema,
>(
  registry: Registry,
  options: GenerateOptions<O, CustomOptions>
): Promise<GenerateActionOptions> {
  let resolvedModel: ResolvedModel<CustomOptions> | undefined;
  if (options.model) {
    resolvedModel = await resolveModel(registry, options.model);
  }
  const tools = await toolsToActionRefs(registry, options.tools);
  const resources = await resourcesToActionRefs(registry, options.resources);
  const messages: MessageData[] = messagesFromOptions(options);

  const resolvedSchema = toJsonSchema({
    schema: options.output?.schema,
    jsonSchema: options.output?.jsonSchema,
  });

  // If is schema is set but format is not explicitly set, default to `json` format.
  if (
    (options.output?.schema || options.output?.jsonSchema) &&
    !options.output?.format
  ) {
    options.output.format = 'json';
  }

  const params: GenerateActionOptions = {
    model: resolvedModel?.modelAction.__action.name,
    docs: options.docs,
    messages: messages,
    tools,
    resources,
    toolChoice: options.toolChoice,
    config: {
      version: resolvedModel?.version,
      ...stripUndefinedOptions(resolvedModel?.config),
      ...stripUndefinedOptions(options.config),
    },
    output: options.output && {
      ...options.output,
      format: options.output.format,
      jsonSchema: resolvedSchema,
    },
    // coerce reply and restart into arrays for the action schema
    resume: options.resume && {
      respond: [options.resume.respond || []].flat(),
      restart: [options.resume.restart || []].flat(),
      metadata: options.resume.metadata,
    },
    returnToolRequests: options.returnToolRequests,
    maxTurns: options.maxTurns,
    stepName: options.stepName,
    use: options.use as MiddlewareRef[] | undefined,
  };
  // if config is empty and it was not explicitly passed in, we delete it, don't want {}
  if (Object.keys(params.config).length === 0 && !options.config) {
    delete params.config;
  }
  return params;
}

/**
 * Check if the callback is a noop callback and return undefined -- downstream models
 * expect undefined if no streaming is requested.
 */
function stripNoop<T>(
  callback: StreamingCallback<T> | undefined
): StreamingCallback<T> | undefined {
  if (callback === sentinelNoopStreamingCallback) {
    return undefined;
  }
  return callback;
}

function stripUndefinedOptions(input?: any): any {
  if (!input) return input;
  const copy = { ...input };
  Object.keys(input).forEach((key) => {
    if (copy[key] === undefined) {
      delete copy[key];
    }
  });
  return copy;
}

async function resolveFullToolNames(
  registry: Registry,
  name: string
): Promise<string[]> {
  let names: string[];
  const parts = name.split(':');
  if (parts.length > 1) {
    // Dynamic Action Provider
    names = await registry.resolveActionNames(
      `/dynamic-action-provider/${name}`
    );
    if (names.length) {
      return names;
    }
  }
  if (await registry.lookupAction(`/tool/${name}`)) {
    return [`/tool/${name}`];
  }
  if (await registry.lookupAction(`/tool.v2/${name}`)) {
    return [`/tool.v2/${name}`];
  }
  if (await registry.lookupAction(`/prompt/${name}`)) {
    return [`/prompt/${name}`];
  }
  throw new Error(`Unable to resolve tool: ${name}`);
}

async function resolveFullResourceNames(
  registry: Registry,
  name: string
): Promise<string[]> {
  let names: string[];
  const parts = name.split(':');
  if (parts.length > 1) {
    // Dynamic Action Provider
    names = await registry.resolveActionNames(
      `/dynamic-action-provider/${name}`
    );
    if (names.length) {
      return names;
    }
  }
  if (await registry.lookupAction(`/resource/${name}`)) {
    return [`/resource/${name}`];
  }
  throw new Error(`Unable to resolve resource: ${name}`);
}

/** Options for {@link Genkit.generateStream}. Same as {@link GenerateOptions} but without `streamingCallback`. */
export type GenerateStreamOptions<
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = typeof GenerationCommonConfigSchema,
> = Omit<GenerateOptions<O, CustomOptions>, 'streamingCallback'>;

/** The return type of {@link Genkit.generateStream}, providing both a stream of chunks and the final response. */
export interface GenerateStreamResponse<O extends z.ZodTypeAny = z.ZodTypeAny> {
  get stream(): AsyncIterable<GenerateResponseChunk>;
  get response(): Promise<GenerateResponse<O>>;
}

export function generateStream<
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = typeof GenerationCommonConfigSchema,
>(
  registry: Registry,
  options:
    | GenerateOptions<O, CustomOptions>
    | PromiseLike<GenerateOptions<O, CustomOptions>>
): GenerateStreamResponse<O> {
  const channel = new Channel<GenerateResponseChunk>();

  const generated = Promise.resolve(options).then((resolvedOptions) =>
    generate<O, CustomOptions>(registry, {
      ...resolvedOptions,
      onChunk: (chunk) => channel.send(chunk),
    })
  );
  generated.then(
    () => channel.close(),
    (err) => channel.error(err)
  );

  return {
    response: generated,
    stream: channel,
  };
}
