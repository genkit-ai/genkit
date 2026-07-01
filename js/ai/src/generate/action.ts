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
  StreamingCallback,
  defineAction,
  stripUndefinedProps,
  type Action,
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
  GenerationResponseError,
  maybeRegisterDynamicMiddlewareTools,
  normalizeMiddleware,
} from '../generate.js';
import { GenerateResponseChunk } from '../generate/chunk.js';
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
} from '../model.js';
import {
  findMatchingResource,
  resolveResources,
  type ResourceAction,
} from '../resource.js';
import { resolveTools, toToolDefinition, type ToolAction } from '../tool.js';
import { GenerateMiddlewareDef, resolveMiddleware } from './middleware.js';
import {
  assertValidToolNames,
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

  if (middleware && middleware.length > 0) {
    const dispatchGenerate = async (
      index: number,
      request: GenerateActionOptions,
      currentTurn: number,
      messageIndex: number,
      ctx: ActionRunOptions<any>
    ): Promise<any> => {
      if (index === middleware.length) {
        return generateActionTurn(registry, {
          rawRequest: request,
          middleware,
          currentTurn,
          messageIndex,
          abortSignal: ctx.abortSignal,
          streamingCallback: ctx.onChunk,
          context: ctx.context,
          sharedPreviousChunks,
        });
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
    return dispatchGenerate(0, rawRequest, currentTurn, messageIndex, {
      abortSignal,
      onChunk: streamingCallback,
      context,
    });
  } else {
    return generateActionTurn(registry, {
      ...args,
      sharedPreviousChunks,
    });
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
  }: {
    rawRequest: GenerateActionOptions;
    middleware: GenerateMiddlewareDef[] | undefined;
    currentTurn: number;
    messageIndex: number;
    abortSignal?: AbortSignal;
    streamingCallback?: StreamingCallback<GenerateResponseChunk>;
    context?: Record<string, any>;
    sharedPreviousChunks: GenerateResponseChunkData[];
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

  const {
    revisedRequest,
    interruptedResponse,
    toolMessage: resumedToolMessage,
  } = await resolveResumeOption(registry, rawRequest, tools, middleware || []);
  // NOTE: in the future we should make it possible to interrupt a restart, but
  // at the moment it's too complicated because it's not clear how to return a
  // response that amends history but doesn't generate a new message, so we throw
  if (revisedRequest && revisedRequest !== rawRequest) {
    if (interruptedResponse) {
      throw new GenkitError({
        status: 'FAILED_PRECONDITION',
        message:
          'One or more tools triggered an interrupt during a restarted execution.',
        detail: { message: interruptedResponse.message },
      });
    }

    if (resumedToolMessage && streamingCallback) {
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
    }

    return await generateHelper(registry, {
      rawRequest: revisedRequest,
      middleware,
      currentTurn,
      messageIndex: messageIndex + (resumedToolMessage ? 1 : 0),
      abortSignal,
      streamingCallback,
      context,
    });
  }
  rawRequest = revisedRequest!;

  const request = await actionToGenerateRequest(
    rawRequest,
    tools,
    format,
    model
  );

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

  const modelResponse = await dispatchModel(0, request, {
    abortSignal,
    context,
    onChunk: sendChunk,
  });

  if (model.__action.actionType === 'background-model') {
    response = new GenerateResponse(
      { operation: modelResponse },
      {
        request,
        parser: format?.handler(request.output?.schema).parseMessage,
      }
    );
  } else {
    response = new GenerateResponse(modelResponse, {
      request,
      parser: format?.handler(request.output?.schema).parseMessage,
    });
  }
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
    if (toolRequests.length === 0) response.assertValidSchema(request);
    return response.toJSON();
  }

  const maxIterations = rawRequest.maxTurns ?? 5;
  if (currentTurn + 1 > maxIterations) {
    throw new GenerationResponseError(
      response,
      `Exceeded maximum tool call iterations (${maxIterations})`,
      'ABORTED',
      { request }
    );
  }

  const { revisedModelMessage, toolMessage } = await resolveToolRequests(
    rawRequest,
    generatedMessage,
    tools,
    middleware || []
  );

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
    streamingCallback?.(
      makeChunk('tool', {
        content: toolMessage.content,
      })
    );
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
  return await generateHelper(registry, {
    rawRequest: nextRequest,
    middleware: middleware,
    currentTurn: currentTurn + 1,
    messageIndex: messageIndex + 1,
    streamingCallback,
    abortSignal,
  });
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
