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
  ActionRunOptions,
  GenkitError,
  StatusNameSchema,
  StreamingCallback,
  defineAction,
  getErrorMessage,
  stripUndefinedProps,
  type Action,
  type StatusName,
  type z,
} from '@genkit-ai/core';
import { logger } from '@genkit-ai/core/logging';
import { Registry } from '@genkit-ai/core/registry';
import { SPAN_TYPE_ATTR, runInNewSpan } from '@genkit-ai/core/tracing';
import {
  injectInstructions,
  resolveFormat,
  resolveInstructions,
} from '../formats/index.js';
import type { Formatter } from '../formats/types.js';
import {
  GenerateResponse,
  GenerationAbortedError,
  GenerationResponseError,
  maybeRegisterDynamicMiddlewareTools,
  normalizeMiddleware,
} from '../generate.js';
import { GenerateResponseChunk } from '../generate/chunk.js';
import type { MessageParser } from '../message.js';
import {
  GenerateActionOptionsSchema,
  GenerateResponseChunkSchema,
  GenerateResponseSchema,
  MessageData,
  resolveModel,
  type GenerateActionOptions,
  type GenerateActionOutputConfig,
  type GenerateRequest,
  type GenerateRequestSchema,
  type GenerateResponseChunkData,
  type GenerateResponseData,
  type ModelAction,
  type ModelInfo,
  type ModelRequest,
  type Part,
  type Role,
  type RuntimeError,
} from '../model.js';
import {
  findMatchingResource,
  resolveResources,
  type ResourceAction,
} from '../resource.js';
import { resolveTools, toToolDefinition, type ToolAction } from '../tool.js';
import { GenerateMiddlewareDef, resolveMiddleware } from './middleware.js';
import {
  ToolFailureError,
  assertValidToolNames,
  errorDetailsOf,
  resolveResumeOption,
  resolveToolRequests,
} from './resolve-tool-requests.js';

export type GenerateAction = Action<
  typeof GenerateActionOptionsSchema,
  typeof GenerateResponseSchema,
  typeof GenerateResponseChunkSchema
>;

/** Defines (registers) a utilty generate action. */
export function defineGenerateAction(registry: Registry): GenerateAction {
  return defineAction(
    registry,
    {
      actionType: 'util',
      name: 'generate',
      inputSchema: GenerateActionOptionsSchema,
      outputSchema: GenerateResponseSchema,
      streamSchema: GenerateResponseChunkSchema,
    },
    async (request, { streamingRequested, sendChunk, context }) => {
      let childRegistry = Registry.withParent(registry);
      const middlewareRefs = await normalizeMiddleware(
        childRegistry,
        request.use
      );
      request.use = middlewareRefs; // Cast back because `use` can be generic

      const resolvedMiddleware = await resolveMiddleware(
        childRegistry,
        request.use
      );
      maybeRegisterDynamicMiddlewareTools(childRegistry, resolvedMiddleware);

      const generateFn = (
        sendChunk?: StreamingCallback<GenerateResponseChunk>
      ) =>
        generateActionImpl(childRegistry, {
          rawRequest: request,
          currentTurn: 0,
          messageIndex: 0,
          middleware: resolvedMiddleware,
          streamingCallback: sendChunk,
          context,
        });
      return streamingRequested
        ? generateFn((c: GenerateResponseChunk) =>
            sendChunk(c.toJSON ? c.toJSON() : c)
          )
        : generateFn();
    }
  );
}

/**
 * Encapsulates all generate logic. This is similar to `generateAction` except not an action and can take middleware.
 */
export async function generateHelper(
  registry: Registry,
  options: {
    rawRequest: GenerateActionOptions;
    middleware?: GenerateMiddlewareDef[];
    currentTurn?: number;
    messageIndex?: number;
    abortSignal?: AbortSignal;
    streamingCallback?: StreamingCallback<GenerateResponseChunk>;
    context?: Record<string, any>;
  }
): Promise<GenerateResponseData> {
  const currentTurn = options.currentTurn ?? 0;
  const messageIndex = options.messageIndex ?? 0;
  // do tracing
  return await runInNewSpan(
    {
      metadata: {
        name: options.rawRequest.stepName || 'generate',
      },
      labels: {
        [SPAN_TYPE_ATTR]: 'util',
      },
    },
    async (metadata) => {
      metadata.name = options.rawRequest.stepName || 'generate';
      metadata.input = options.rawRequest;
      try {
        const output = await generateActionImpl(registry, {
          rawRequest: options.rawRequest,
          middleware: options.middleware,
          currentTurn,
          messageIndex,
          abortSignal: options.abortSignal,
          streamingCallback: options.streamingCallback,
          context: options.context,
        });
        metadata.output = JSON.stringify(output);
        return output;
      } catch (e) {
        // A failure can still carry a result: the loop throws the
        // conversation it completed alongside its error. Record it so the
        // span shows what the call produced and not only that it stopped.
        const partial = partialResponseOf(e, registry);
        if (partial) metadata.output = JSON.stringify(partial.toJSON());
        throw e;
      }
    }
  );
}

/** Take the raw request and resolve tools, model, and format into their registry action counterparts. */
async function resolveParameters(
  registry: Registry,
  request: GenerateActionOptions
) {
  const [model, tools, resources, format] = await Promise.all([
    resolveModel(registry, request.model, { warnDeprecated: true }).then(
      (r) => r.modelAction
    ),
    resolveTools(registry, request.tools),
    resolveResources(registry, request.resources),
    resolveFormat(registry, request.output),
  ]);
  return { model, tools, resources, format };
}

/** Given a raw request and a formatter, apply the formatter's logic and instructions to the request. */
function applyFormat(
  rawRequest: GenerateActionOptions,
  resolvedFormat?: Formatter
) {
  const outRequest = { ...rawRequest };
  // If is schema is set but format is not explicitly set, default to `json` format.
  if (rawRequest.output?.jsonSchema && !rawRequest.output?.format) {
    outRequest.output = { ...rawRequest.output, format: 'json' };
  }

  const instructions = resolveInstructions(
    resolvedFormat,
    outRequest.output?.jsonSchema,
    outRequest?.output?.instructions
  );

  if (resolvedFormat) {
    if (
      shouldInjectFormatInstructions(resolvedFormat.config, rawRequest?.output)
    ) {
      outRequest.messages = injectInstructions(
        outRequest.messages,
        instructions
      );
    }
    outRequest.output = {
      // use output config from the format
      ...resolvedFormat.config,
      // if anything is set explicitly, use that
      ...outRequest.output,
    };
  }

  return outRequest;
}

export function shouldInjectFormatInstructions(
  formatConfig?: Formatter['config'],
  rawRequestConfig?: z.infer<typeof GenerateActionOutputConfig>
) {
  return (
    formatConfig?.defaultInstructions !== false ||
    rawRequestConfig?.instructions
  );
}

/**
 * The failure record of one turn frame, the JS form of the Go loop's
 * `lastPartial`/`lastReq`. A frame is one `generateActionImpl` call, which
 * runs one turn through the `generate` middleware hooks; later turns nest
 * inside it, so a partial thrown deeper always passes back through this
 * frame's `runTurn` on its way up and is recorded here. That is what lets
 * {@link restorePartial} rewrap an error a hook threw in place of the loop's
 * own with the conversation the loop had actually completed.
 */
interface TurnState {
  /** The partial response of the innermost failure seen by this frame. */
  partial?: GenerateResponse;
  /**
   * The resolved request of this frame's turn, set once the turn reached the
   * model boundary. Its presence is what says an error escaping the hooks
   * without a partial still deserves one.
   */
  request?: GenerateRequest;
}

async function generateActionImpl(
  registry: Registry,
  args: {
    rawRequest: GenerateActionOptions;
    middleware: GenerateMiddlewareDef[] | undefined;
    currentTurn: number;
    messageIndex: number;
    abortSignal?: AbortSignal;
    streamingCallback?: StreamingCallback<GenerateResponseChunk>;
    context?: Record<string, any>;
  }
): Promise<GenerateResponseData> {
  const {
    rawRequest,
    middleware,
    currentTurn,
    messageIndex,
    abortSignal,
    streamingCallback,
    context,
  } = args;

  const format = await resolveFormat(registry, rawRequest.output);

  const sharedPreviousChunks: GenerateResponseChunkData[] = [];
  const parser = format?.handler(rawRequest.output?.jsonSchema).parseChunk;

  const turnState: TurnState = {};

  // runTurn is the end of the `generate` hook chain: it runs the turn and
  // records the partial of any failure that passes through it, the turn's own
  // or a deeper turn's. A fresh attempt (a hook calling `next` again after a
  // failure) invalidates the previous attempt's record, so a partial from a
  // failure that was recovered cannot pair with a later error.
  const runTurn = async (
    request: GenerateActionOptions,
    currentTurn: number,
    messageIndex: number,
    ctx: ActionRunOptions<any>
  ): Promise<GenerateResponseData> => {
    turnState.partial = undefined;
    turnState.request = undefined;
    try {
      return await generateActionTurn(registry, {
        rawRequest: request,
        middleware,
        currentTurn,
        messageIndex,
        abortSignal: ctx.abortSignal,
        streamingCallback: ctx.onChunk,
        context: ctx.context,
        sharedPreviousChunks,
        turnState,
      });
    } catch (e) {
      const partial = partialResponseOf(e, registry);
      if (partial) turnState.partial = partial;
      throw e;
    }
  };

  try {
    if (middleware && middleware.length > 0) {
      const dispatchGenerate = async (
        index: number,
        request: GenerateActionOptions,
        currentTurn: number,
        messageIndex: number,
        ctx: ActionRunOptions<any>
      ): Promise<any> => {
        if (index === middleware.length) {
          return runTurn(request, currentTurn, messageIndex, ctx);
        }
        const currentMiddleware = middleware[index];
        if (currentMiddleware.generate) {
          const wrappedOnChunk = ctx.onChunk
            ? (c: GenerateResponseChunk | GenerateResponseChunkData) => {
                if (c instanceof GenerateResponseChunk) {
                  ctx.onChunk!(c);
                } else {
                  const chunk = new GenerateResponseChunk(c, {
                    index: c.index !== undefined ? c.index : messageIndex,
                    role: c.role !== undefined ? c.role : 'model',
                    previousChunks: [...sharedPreviousChunks],
                    parser: parser,
                  });
                  sharedPreviousChunks.push(c); // Accumulate raw data!
                  ctx.onChunk!(chunk);
                }
              }
            : undefined;

          return currentMiddleware.generate(
            { request: request, currentTurn, messageIndex },
            { ...ctx, onChunk: wrappedOnChunk },
            async (modifiedEnvelope, opts) =>
              dispatchGenerate(
                index + 1,
                modifiedEnvelope?.request || request,
                modifiedEnvelope?.currentTurn !== undefined
                  ? modifiedEnvelope.currentTurn
                  : currentTurn,
                modifiedEnvelope?.messageIndex !== undefined
                  ? modifiedEnvelope.messageIndex
                  : messageIndex,
                opts || ctx
              )
          );
        } else {
          return dispatchGenerate(
            index + 1,
            request,
            currentTurn,
            messageIndex,
            ctx
          );
        }
      };
      return await dispatchGenerate(0, rawRequest, currentTurn, messageIndex, {
        abortSignal,
        onChunk: streamingCallback,
        context,
      });
    } else {
      return await runTurn(rawRequest, currentTurn, messageIndex, {
        abortSignal,
        onChunk: streamingCallback,
        context,
      });
    }
  } catch (e) {
    if (partialResponseOf(e, registry)) throw e;
    throw restorePartial(e, turnState, registry, abortSignal);
  }
}

async function generateActionTurn(
  registry: Registry,
  {
    rawRequest,
    middleware,
    currentTurn,
    messageIndex,
    abortSignal,
    streamingCallback,
    context,
    sharedPreviousChunks,
    turnState,
  }: {
    rawRequest: GenerateActionOptions;
    middleware: GenerateMiddlewareDef[] | undefined;
    currentTurn: number;
    messageIndex: number;
    abortSignal?: AbortSignal;
    streamingCallback?: StreamingCallback<GenerateResponseChunk>;
    context?: Record<string, any>;
    sharedPreviousChunks: GenerateResponseChunkData[];
    turnState: TurnState;
  }
): Promise<GenerateResponseData> {
  const { model, tools, resources, format } = await resolveParameters(
    registry,
    rawRequest
  );

  // Append tools supplied by middleware
  if (middleware) {
    tools.push(...middleware.flatMap((m) => m.tools || []));
  }
  rawRequest = applyFormat(rawRequest, format);
  rawRequest = await applyResources(registry, rawRequest, resources);

  // check to make sure we don't have overlapping tool names *before* generation
  await assertValidToolNames(tools);

  // The request has resolved. Every failure from here on throws a
  // GenerationResponseError carrying a partial response, which `generate`
  // hands back as the response when the caller asked for failures on the
  // response, and otherwise unwraps to the error the failure would have
  // thrown on its own (see `errorToThrow`). Errors above this line (an
  // unknown model, tool, or resource) carry none.
  const request = await actionToGenerateRequest(
    rawRequest,
    tools,
    format,
    model
  );
  turnState.request = request;
  const parser = format?.handler(request.output?.schema).parseMessage;
  // Builds the error for a failure whose partial ends at `messages`: the
  // conversation as it stood when the failing step began. `base` is the
  // failing turn's own model response, whose accounting the partial keeps.
  const failAt = (
    messages: MessageData[],
    cause: unknown,
    base?: GenerateResponseData
  ) =>
    failureError({ ...request, messages }, cause, {
      abortSignal,
      base,
      parser,
      registry,
    });

  let resumed: Awaited<ReturnType<typeof resolveResumeOption>>;
  try {
    resumed = await resolveResumeOption(
      registry,
      rawRequest,
      tools,
      middleware || [],
      { abortSignal }
    );
  } catch (e) {
    throw failAt(rawRequest.messages, e);
  }
  const {
    revisedRequest,
    interruptedResponse,
    toolMessage: resumedToolMessage,
  } = resumed;
  if (interruptedResponse) {
    // A restarted tool interrupted again. The response keeps `interrupted`
    // under its FAILED_PRECONDITION error: its message is the conversation's
    // revised last message (resolveResumeOption revises it in place), so the
    // request carries the messages before it and `messages` reproduces the
    // full conversation. That is the one non-seam shape a partial carries,
    // because it is answered with `resume` rather than sent again.
    const message =
      'One or more tools triggered an interrupt during a restarted execution.';
    const response = ownedBy(
      new GenerateResponse(
        {
          ...interruptedResponse,
          error: { status: 'FAILED_PRECONDITION', message },
        },
        {
          request: { ...request, messages: rawRequest.messages.slice(0, -1) },
          parser,
        }
      ),
      registry
    );
    throw withThrownForm(
      new GenerationResponseError(response, message, 'FAILED_PRECONDITION', {
        message: interruptedResponse.message,
      }),
      new GenkitError({
        status: 'FAILED_PRECONDITION',
        message,
        detail: { message: interruptedResponse.message },
      })
    );
  }
  if (revisedRequest && revisedRequest !== rawRequest) {
    if (resumedToolMessage && streamingCallback) {
      try {
        streamingCallback(
          new GenerateResponseChunk(
            {
              role: 'tool',
              content: resumedToolMessage.content,
            },
            {
              index: messageIndex,
              role: 'tool',
              previousChunks: [],
              parser: format?.handler(rawRequest.output?.jsonSchema).parseChunk,
            }
          )
        );
      } catch (e) {
        throw failAt(revisedRequest.messages, e);
      }
    }

    try {
      return await generateHelper(registry, {
        rawRequest: revisedRequest,
        middleware,
        currentTurn,
        messageIndex: messageIndex + (resumedToolMessage ? 1 : 0),
        abortSignal,
        streamingCallback,
        context,
      });
    } catch (e) {
      // A later turn's failure carries its own partial. An error a hook threw
      // before running that turn does not, and the conversation entering it
      // is the seam (the Go loop's lastReq).
      if (partialResponseOf(e, registry)) throw e;
      throw failAt(revisedRequest.messages, e);
    }
  }

  let chunkRole: Role = 'model';
  // convenience method to create a full chunk from role and data, append the chunk
  // to the sharedPreviousChunks array, and increment the message index as needed
  const makeChunk = (
    role: Role,
    chunk: GenerateResponseChunkData
  ): GenerateResponseChunk => {
    if (role !== chunkRole && sharedPreviousChunks.length) messageIndex++;
    chunkRole = role;

    const prevToSend = [...sharedPreviousChunks];
    sharedPreviousChunks.push(chunk);

    return new GenerateResponseChunk(chunk, {
      index: messageIndex,
      role,
      previousChunks: prevToSend,
      parser: format?.handler(request.output?.schema).parseChunk,
    });
  };

  var response: GenerateResponse;
  const sendChunk =
    streamingCallback &&
    ((chunk: GenerateResponseChunkData) =>
      streamingCallback(makeChunk('model', chunk)));
  const dispatchModel = async (
    index: number,
    req: z.infer<typeof GenerateRequestSchema>,
    actionOpts: ActionRunOptions<any>
  ): Promise<any> => {
    if (!middleware || index === middleware.length) {
      // end of the chain, call the original model action
      return await model(req, actionOpts);
    }

    const currentMiddleware = middleware[index];
    if (currentMiddleware.model) {
      return currentMiddleware.model(
        req,
        actionOpts,
        async (modifiedReq, opts) =>
          dispatchModel(index + 1, modifiedReq || req, opts || actionOpts)
      );
    } else {
      return dispatchModel(index + 1, req, actionOpts);
    }
  };

  // A caller that stopped the loop between turns is reported before the next
  // model call rather than by whatever that call happens to do with the
  // signal. The seam is this turn's own request, so the round the previous
  // turn completed is kept.
  if (abortSignal?.aborted) {
    throw failAt(request.messages, abortReason(abortSignal));
  }
  let modelResponse: any;
  try {
    modelResponse = await dispatchModel(0, request, {
      abortSignal,
      context,
      onChunk: sendChunk,
    });
  } catch (e) {
    // The model's own output is dropped, complete or not: only the
    // conversation entering this turn survives. Chunks already streamed
    // reached the callback, so nothing observable is lost.
    throw failAt(request.messages, e);
  }

  if (model.__action.actionType === 'background-model') {
    response = new GenerateResponse(
      { operation: modelResponse },
      {
        request,
        parser,
      }
    );
  } else {
    response = new GenerateResponse(modelResponse, {
      request,
      parser,
    });
  }
  ownedBy(response, registry);
  if (model.__action.actionType === 'background-model') {
    return response.toJSON();
  }

  // Throw an error if the response is not usable.
  response.assertValid();
  const generatedMessage = response.message!; // would have thrown if no message

  const toolRequests = generatedMessage.content.filter(
    (part) => !!part.toolRequest
  );

  if (rawRequest.returnToolRequests || toolRequests.length === 0) {
    if (toolRequests.length === 0) {
      try {
        response.assertValidSchema(request);
      } catch (e) {
        // The model finished; post-processing did not. The response rides
        // back with the model's own message and finish reason, not marked
        // failed: the raw output is often exactly what the caller needs to
        // see. It keeps `messages` ending in that message, so it is not a
        // conversation to send again as it stands.
        throw invalidOutputError(response, e);
      }
    }
    return response.toJSON();
  }

  const maxIterations = rawRequest.maxTurns ?? 5;
  if (currentTurn + 1 > maxIterations) {
    // The round the loop refused to run goes whole, the model message that
    // opened it included: a conversation ending in a tool request nothing
    // answered is one no provider accepts back. The turn's accounting stays.
    const message = `Exceeded maximum tool call iterations (${maxIterations})`;
    const cause = new GenkitError({ status: 'ABORTED', message });
    throw withThrownForm(
      new GenerationAbortedError(
        failurePartial(request, cause, {
          finishReason: 'aborted',
          base: response.toJSON(),
          parser,
          registry,
        }),
        message,
        'ABORTED',
        undefined,
        { cause }
      ),
      // What a caller catches: the error the limit has always thrown, with
      // the refused round's own response, as the subclass that names a stop
      // so a caller's own turn loop can tell it from a break.
      new GenerationAbortedError(response, message, 'ABORTED', { request })
    );
  }

  let resolvedTools: Awaited<ReturnType<typeof resolveToolRequests>>;
  try {
    resolvedTools = await resolveToolRequests(
      rawRequest,
      generatedMessage,
      tools,
      middleware || [],
      { abortSignal }
    );
  } catch (e) {
    // The whole round goes, the model message that opened it included: a
    // failed tool leaves its request unanswered, and the partial hands back
    // a conversation that can be re-sent. The error is reported as soon as
    // it arrives; a still-running sibling finishes detached and its result
    // goes with the discarded round.
    throw failAt(request.messages, e, response.toJSON());
  }
  const { revisedModelMessage, toolMessage } = resolvedTools;

  // if an interrupt message is returned, stop the tool loop and return a response
  if (revisedModelMessage) {
    return {
      ...response.toJSON(),
      finishReason: 'interrupted',
      finishMessage: 'One or more tool calls resulted in interrupts.',
      message: revisedModelMessage,
    };
  }

  // if the loop will continue, stream out the tool response message...
  if (toolMessage) {
    try {
      streamingCallback?.(
        makeChunk('tool', {
          content: toolMessage.content,
        })
      );
    } catch (e) {
      throw failAt(request.messages, e, response.toJSON());
    }
  }

  const messages = [...rawRequest.messages, generatedMessage.toJSON()];
  if (toolMessage) {
    messages.push(toolMessage);
  }

  let nextRequest = {
    ...rawRequest,

    messages,
  };

  // then recursively call for another loop
  try {
    return await generateHelper(registry, {
      rawRequest: nextRequest,
      middleware: middleware,
      currentTurn: currentTurn + 1,
      messageIndex: messageIndex + 1,
      streamingCallback,
      abortSignal,
      context,
    });
  } catch (e) {
    // A later turn's failure carries its own partial. An error a hook threw
    // before running that turn does not, and the conversation entering it,
    // this completed round included, is the seam (the Go loop's lastReq).
    if (partialResponseOf(e, registry)) throw e;
    throw failAt(messages, e);
  }
}

/**
 * The responses this loop built, keyed to the registry the invocation runs
 * under: `generate` gives every invocation its own child registry, which
 * every turn and hook of that invocation shares and a nested `generate` (one
 * a hook ran) does not.
 */
const loopPartials = new WeakMap<GenerateResponse, Registry>();

function ownedBy<R extends GenerateResponse>(
  response: R,
  registry: Registry
): R {
  loopPartials.set(response, registry);
  return response;
}

/**
 * The partial response an error carries, when it is this loop's own. The
 * check is by identity rather than shape: an error from a nested `generate`
 * a hook ran carries that loop's response, not this one's partial.
 */
export function partialResponseOf(
  e: unknown,
  registry: Registry
): GenerateResponse | undefined {
  const response = (e as any)?.detail?.response;
  return response instanceof GenerateResponse &&
    loopPartials.get(response) === registry
    ? response
    : undefined;
}

/**
 * The error a caller catches for each error the loop builds. The loop wraps
 * every failure once the request has resolved (see `failureError`), so the
 * partial response travels with it; a caller that did not ask for failures
 * on the response still gets the error the failure threw on its own, as it
 * always has: a tool's own error, the model's, the validation error, the
 * hook's. An error the loop throws in its own name (a blocked response, a
 * response without a message) is thrown as is.
 */
const thrownForms = new WeakMap<object, unknown>();

function withThrownForm<E extends object>(error: E, thrown: unknown): E {
  thrownForms.set(error, thrown);
  return error;
}

/**
 * What `generate` throws for `e`; see `thrownForms`. The wrapper passed the
 * loop's spans on its way out, and the error a caller catches carries the
 * same marks, so an enclosing span does not claim the failure again.
 */
export function errorToThrow(e: unknown): unknown {
  if (typeof e !== 'object' || e === null || !thrownForms.has(e)) return e;
  const thrown = thrownForms.get(e);
  if (typeof thrown === 'object' && thrown !== null) {
    for (const mark of ['ignoreFailedSpan', 'traceId'] as const) {
      if (
        (e as any)[mark] !== undefined &&
        (thrown as any)[mark] === undefined
      ) {
        (thrown as any)[mark] = (e as any)[mark];
      }
    }
  }
  return thrown;
}

/**
 * The error a failure throws on its own: the cause, except that a tool
 * failure's cause is the loop's classification of the tool's error, and the
 * tool's own error is what the caller was catching.
 */
function ownError(cause: unknown): unknown {
  return cause instanceof ToolFailureError ? cause.cause : cause;
}

/**
 * Reports whether the loop ended because the caller stopped it rather than
 * because something inside it broke: the request's abort signal fired, or the
 * model call rejected with the cancellation or timeout the platform raises for
 * one. Those report `finishReason` `aborted`; everything else reports
 * `failed`. It reads the signal and the error's identity, never its status: a
 * provider answering 409 or 504 lands on ABORTED or DEADLINE_EXCEEDED, and a
 * provider stopping the request is not the caller stopping the run.
 */
function callerStopped(
  abortSignal: AbortSignal | undefined,
  cause: unknown
): boolean {
  if (abortSignal?.aborted) return true;
  const name = (cause as { name?: unknown } | undefined)?.name;
  return name === 'AbortError' || name === 'TimeoutError';
}

/** The error an explicit abort check reports: the signal's reason when it is one. */
function abortReason(abortSignal: AbortSignal): unknown {
  const reason = abortSignal.reason;
  if (reason instanceof Error) return reason;
  return new GenkitError({
    status: 'CANCELLED',
    message: reason === undefined ? 'generation aborted' : String(reason),
  });
}

/**
 * Classifies a failure's cause as a status: a GenkitError's own, the status a
 * cancellation or timeout implies, otherwise INTERNAL.
 */
function statusOf(cause: unknown, abortSignal?: AbortSignal): StatusName {
  if (cause instanceof GenkitError) return cause.status;
  const c = cause as { name?: unknown; status?: unknown } | undefined;
  if (
    typeof c?.status === 'string' &&
    StatusNameSchema.safeParse(c.status).success
  ) {
    return c.status as StatusName;
  }
  if (c?.name === 'TimeoutError') return 'DEADLINE_EXCEEDED';
  if (c?.name === 'AbortError') return 'CANCELLED';
  if (abortSignal?.aborted) {
    return (abortSignal.reason as { name?: unknown } | undefined)?.name ===
      'TimeoutError'
      ? 'DEADLINE_EXCEEDED'
      : 'CANCELLED';
  }
  return 'INTERNAL';
}

/** The cause's own text, without the status prefix a GenkitError adds. */
function messageOf(cause: unknown): string {
  if (cause instanceof GenkitError) return cause.originalMessage;
  if (cause instanceof Error) return cause.message;
  const message = (cause as { message?: unknown } | undefined)?.message;
  return typeof message === 'string' ? message : getErrorMessage(cause);
}

/**
 * The structured error a partial response carries beside its finish message:
 * the cause classified, so a consumer reading the response as data branches on
 * a status rather than a string. Its details are the cause's own, without the
 * request or response payloads a nested loop error carries.
 */
function runtimeErrorOf(
  cause: unknown,
  abortSignal?: AbortSignal
): RuntimeError {
  const details = errorDetailsOf(cause);
  return {
    status: statusOf(cause, abortSignal),
    message: messageOf(cause),
    ...(details !== undefined && { details }),
  };
}

/**
 * The message safe to send to a client for a failure with this cause. A
 * GenkitError's message is already user-facing (or carries its own public
 * one); anything else is arbitrary text that stays in-process.
 */
function publicMessageOf(cause: unknown, aborted: boolean): string | undefined {
  if (cause instanceof GenkitError) return cause.publicMessage;
  return aborted ? 'generation aborted' : 'generation failed';
}

/**
 * Builds the partial response that accompanies the error when the generate
 * loop stops before it produced a final response.
 *
 * The partial ends at a turn seam: `request` carries the conversation as it
 * stood when the failing step began, which is either the caller's own
 * messages or a run of completed [model with tool requests, tool with every
 * response] rounds, and there is no message, so nothing half-finished rides
 * along. The failing turn's own output is dropped whatever it was: a
 * partially streamed model message, a model message whose tools did not all
 * answer, or the tool requests the turn limit refused to run, because a
 * conversation ending in an unanswered tool request is one no provider will
 * accept back. What the caller gets is therefore a conversation it can send
 * again.
 *
 * `base`, when given, is the failing turn's own model response, whose
 * accounting (usage and custom data) the partial keeps; a model call that
 * failed has none. Nothing is aggregated across turns, so a partial's usage
 * means what a final response's does.
 */
function failurePartial(
  request: GenerateRequest,
  cause: unknown,
  opts: {
    finishReason: 'failed' | 'aborted';
    registry: Registry;
    abortSignal?: AbortSignal;
    base?: GenerateResponseData;
    parser?: MessageParser<any>;
  }
): GenerateResponse {
  const error = runtimeErrorOf(cause, opts.abortSignal);
  return ownedBy(
    new GenerateResponse(
      {
        finishReason: opts.finishReason,
        finishMessage: error.message,
        error,
        usage: opts.base?.usage,
        custom: opts.base?.custom,
      },
      { request, parser: opts.parser }
    ),
    opts.registry
  );
}

/**
 * The error the loop throws for a failure once the request has resolved: the
 * cause wrapped with its partial response, classified `aborted` when the
 * caller stopped the loop (a {@link GenerationAbortedError}) and `failed`
 * otherwise.
 */
function failureError(
  request: GenerateRequest,
  cause: unknown,
  opts: {
    registry: Registry;
    abortSignal?: AbortSignal;
    base?: GenerateResponseData;
    parser?: MessageParser<any>;
  }
): GenerationResponseError {
  const aborted = callerStopped(opts.abortSignal, cause);
  const partial = failurePartial(request, cause, {
    ...opts,
    finishReason: aborted ? 'aborted' : 'failed',
  });
  const Ctor = aborted ? GenerationAbortedError : GenerationResponseError;
  return withThrownForm(
    new Ctor(
      partial,
      partial.error!.message,
      partial.error!.status as StatusName,
      undefined,
      { cause, publicMessage: publicMessageOf(cause, aborted) }
    ),
    ownError(cause)
  );
}

/**
 * The error for a completed response that post-processing rejected: the
 * response keeps the model's own message and finish reason and gains the
 * classified error, since the raw output is usually what the caller needs to
 * see. Not a loop stop.
 */
function invalidOutputError(
  response: GenerateResponse,
  cause: unknown
): GenerationResponseError {
  response.error = runtimeErrorOf(cause);
  return withThrownForm(
    new GenerationResponseError(
      response,
      response.error.message,
      response.error.status as StatusName,
      undefined,
      { cause, publicMessage: publicMessageOf(cause, false) }
    ),
    cause
  );
}

/**
 * Rewraps an error that left the turn without its partial response. A
 * `generate` hook that caught the loop's error and threw its own drops the
 * conversation the loop completed; the frame's record restores it, the way
 * the Go loop restores `lastPartial`, and the partial now reports the hook's
 * error, since that is the failure the caller is handed either way. The
 * partial keeps how the loop stopped (`failed` or `aborted`). An error
 * raised outside a turn that did reach the model boundary gets a partial
 * synthesized from that turn's request. An error from before the request
 * resolved is returned as is.
 */
function restorePartial(
  cause: unknown,
  turnState: TurnState,
  registry: Registry,
  abortSignal?: AbortSignal
): unknown {
  if (turnState.partial) {
    const partial = turnState.partial;
    const aborted = partial.finishReason === 'aborted';
    partial.error = runtimeErrorOf(cause, abortSignal);
    // A loop stop's finish message is its error's text; a response the model
    // completed keeps the model's own.
    if (partial.finishReason === 'failed' || aborted) {
      partial.finishMessage = partial.error.message;
    }
    const Ctor = aborted ? GenerationAbortedError : GenerationResponseError;
    return withThrownForm(
      new Ctor(
        partial,
        partial.error.message,
        partial.error.status as StatusName,
        undefined,
        { cause, publicMessage: publicMessageOf(cause, aborted) }
      ),
      ownError(cause)
    );
  }
  if (turnState.request) {
    return failureError(turnState.request, cause, { abortSignal, registry });
  }
  return cause;
}

async function actionToGenerateRequest(
  options: GenerateActionOptions,
  resolvedTools: ToolAction[] | undefined,
  resolvedFormat: Formatter | undefined,
  model: ModelAction
): Promise<GenerateRequest> {
  const modelInfo = model.__action.metadata?.model as ModelInfo;
  if (
    (options.tools?.length ?? 0) > 0 &&
    modelInfo?.supports &&
    !modelInfo?.supports?.tools
  ) {
    logger.warn(
      `The model '${model.__action.name}' does not support tools (you set: ${options.tools?.length} tools). ` +
        'The model may not behave the way you expect.'
    );
  }
  if (
    options.toolChoice &&
    modelInfo?.supports &&
    !modelInfo?.supports?.toolChoice
  ) {
    logger.warn(
      `The model '${model.__action.name}' does not support the 'toolChoice' option (you set: ${options.toolChoice}). ` +
        'The model may not behave the way you expect.'
    );
  }
  const out: ModelRequest = {
    messages: options.messages,
    config: options.config,
    docs: options.docs,
    tools: resolvedTools?.map(toToolDefinition) || [],
    output: stripUndefinedProps({
      constrained: options.output?.constrained,
      contentType: options.output?.contentType,
      format: options.output?.format,
      schema: options.output?.jsonSchema,
    }),
  };
  if (options.toolChoice) {
    out.toolChoice = options.toolChoice;
  }
  if (out.output && !out.output.schema) delete out.output.schema;
  return out;
}

export function inferRoleFromParts(parts: Part[]): Role {
  const uniqueRoles = new Set<Role>();
  for (const part of parts) {
    const role = getRoleFromPart(part);
    uniqueRoles.add(role);
    if (uniqueRoles.size > 1) {
      throw new Error('Contents contain mixed roles');
    }
  }
  return Array.from(uniqueRoles)[0];
}

function getRoleFromPart(part: Part): Role {
  if (part.toolRequest !== undefined) return 'model';
  if (part.toolResponse !== undefined) return 'tool';
  if (part.text !== undefined) return 'user';
  if (part.media !== undefined) return 'user';
  if (part.data !== undefined) return 'user';
  throw new Error('No recognized fields in content');
}

async function applyResources(
  registry: Registry,
  rawRequest: GenerateActionOptions,
  resources: ResourceAction[]
): Promise<GenerateActionOptions> {
  // quick check, if no resources bail.
  if (!rawRequest.messages.find((m) => !!m.content.find((c) => c.resource))) {
    return rawRequest;
  }

  const updatedMessages = [] as MessageData[];
  for (const m of rawRequest.messages) {
    if (!m.content.find((c) => c.resource)) {
      updatedMessages.push(m);
      continue;
    }
    const updatedContent = [] as Part[];
    for (const p of m.content) {
      if (!p.resource) {
        updatedContent.push(p);
        continue;
      }

      const resource = await findMatchingResource(
        registry,
        resources,
        p.resource
      );
      if (!resource) {
        throw new GenkitError({
          status: 'NOT_FOUND',
          message: `failed to find matching resource for ${p.resource.uri}`,
        });
      }
      const resourceParts = await resource(p.resource);
      updatedContent.push(...resourceParts.content);
    }

    updatedMessages.push({
      ...m,
      content: updatedContent,
    });
  }

  return {
    ...rawRequest,
    messages: updatedMessages,
  };
}
