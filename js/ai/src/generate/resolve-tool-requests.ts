/**
 * Copyright 2025 Google LLC
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
  getErrorMessage,
  stripUndefinedProps,
  z,
} from '@genkit-ai/core';
import { logger } from '@genkit-ai/core/logging';
import type { Registry } from '@genkit-ai/core/registry';
import type {
  GenerateActionOptions,
  GenerateResponseData,
  MessageData,
  Part,
  ToolRequestPart,
  ToolResponsePart,
} from '../model.js';
import { MultipartToolResponseSchema, ToolResponse } from '../parts.js';

import {
  ToolInterruptError,
  isToolRequest,
  resolveTools,
  type ToolAction,
  type ToolRunOptions,
} from '../tool.js';
import { type GenerateMiddlewareDef } from './middleware.js';

export function toToolMap(tools: ToolAction[]): Record<string, ToolAction> {
  assertValidToolNames(tools);
  const out: Record<string, ToolAction> = {};
  for (const tool of tools) {
    const name = tool.__action.name;
    const shortName = name.substring(name.lastIndexOf('/') + 1);
    out[shortName] = tool;
  }
  return out;
}

/** Ensures that each tool has a unique name. */
export function assertValidToolNames(tools: ToolAction[]) {
  const nameMap: Record<string, string> = {};
  for (const tool of tools) {
    const name = tool.__action.name;
    const shortName = name.substring(name.lastIndexOf('/') + 1);
    if (nameMap[shortName]) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message: `Cannot provide two tools with the same name: '${name}' and '${nameMap[shortName]}'`,
      });
    }
    nameMap[shortName] = name;
  }
}

/** Options the loop passes to a tool: the request's metadata and the call's abort signal. */
export interface ToolResolveOptions {
  /** The generate call's abort signal, handed to every tool it runs. */
  abortSignal?: AbortSignal;
}

function toRunOptions(
  part: ToolRequestPart,
  options?: ToolResolveOptions
): ToolRunOptions {
  const out: ToolRunOptions = { metadata: part.metadata };
  if (part.metadata?.resumed) out.resumed = part.metadata.resumed;
  if (options?.abortSignal) out.abortSignal = options.abortSignal;
  return out;
}

/**
 * Classifies a tool's error for the loop. A tool that failed on its own terms
 * is an INTERNAL failure of the generation, since a tool's failure is not a
 * failure of the caller's request, and the tool's own error is the `cause`. A
 * tool that stopped because the call's abort signal fired is not a tool
 * failure at all: that error carries CANCELLED, so the partial response
 * reports `aborted` rather than blaming the tool for a stop the caller asked
 * for. The check keys on the call's signal, so a tool that throws an abort
 * error on its own terms is still a tool failure.
 *
 * The tool's own text stays in-process unless it is already a GenkitError: an
 * HTTP handler sends a GenkitError's message to the client, and a tool's
 * arbitrary exception is not written for one.
 */
function toolFailureError(
  name: string,
  cause: unknown,
  abortSignal?: AbortSignal
): GenkitError {
  const stopped = !!abortSignal?.aborted;
  const verb = stopped ? 'stopped' : 'failed';
  const isGenkit = cause instanceof GenkitError;
  const text = isGenkit ? cause.originalMessage : getErrorMessage(cause);
  return new ToolFailureError({
    status: stopped ? 'CANCELLED' : 'INTERNAL',
    message: `tool "${name}" ${verb}: ${text}`,
    detail: errorDetailsOf(cause),
    cause,
    publicMessage: isGenkit
      ? cause.publicMessage && `tool "${name}" ${verb}: ${cause.publicMessage}`
      : `tool "${name}" ${verb}`,
  });
}

/**
 * A tool's failure as the loop classifies it (see `toolFailureError`). The
 * tool's own error is the `cause`, which is what `generate` throws for it by
 * default: the classification is for the partial response and the wire, not
 * for a caller that was catching the tool's error before the loop classified
 * it.
 */
export class ToolFailureError extends GenkitError {}

/**
 * The structured details a cause contributes to the error that wraps it: a
 * GenkitError's own `detail`, without the request or response payloads a
 * generation error carries, which would otherwise repeat a conversation (or
 * nest one per agent depth) inside the wrapper's details.
 */
export function errorDetailsOf(cause: unknown): unknown {
  if (!(cause instanceof GenkitError)) return undefined;
  const detail = cause.detail;
  if (!detail || typeof detail !== 'object' || Array.isArray(detail)) {
    return detail;
  }
  const { request: _request, response: _response, ...rest } = detail;
  return Object.keys(rest).length > 0 ? rest : undefined;
}

/**
 * Records a resolved tool call's response on its request part so a later
 * resume replays it instead of running the tool again (see
 * `resolveResumedToolRequest`). The response's multipart content and its
 * metadata ride under their own keys; `pendingOutput` itself stays
 * output-only for cross-SDK parity.
 */
export function toPendingOutput(
  part: ToolRequestPart,
  response: ToolResponsePart
): ToolRequestPart {
  const metadata: Record<string, any> = {
    ...part.metadata,
    // A void output is stashed as null: it must survive a session store's
    // JSON, which drops a key holding undefined.
    pendingOutput: response.toolResponse.output ?? null,
  };
  if (response.toolResponse.content?.length) {
    metadata.pendingContent = response.toolResponse.content;
  }
  if (response.metadata && Object.keys(response.metadata).length > 0) {
    metadata.pendingMetadata = response.metadata;
  }
  return { ...part, metadata };
}

export async function resolveToolRequest(
  rawRequest: GenerateActionOptions,
  part: ToolRequestPart,
  toolMap: Record<string, ToolAction>,
  middleware: GenerateMiddlewareDef[] = [],
  runOptions?: ToolRunOptions
): Promise<{
  response?: ToolResponsePart;
  interrupt?: ToolRequestPart;
}> {
  const tool = toolMap[part.toolRequest.name];
  if (!tool) {
    throw new GenkitError({
      status: 'NOT_FOUND',
      message: `Tool ${part.toolRequest.name} not found`,
      detail: { request: rawRequest },
    });
  }

  const dispatch = async (
    index: number,
    req: ToolRequestPart,
    ctx: ActionRunOptions<any>
  ): Promise<ToolResponsePart | undefined> => {
    if (index === middleware.length) {
      return executeTool(req, ctx);
    }
    const currentMiddleware = middleware[index];
    if (currentMiddleware.tool) {
      return currentMiddleware.tool(req, ctx, async (modifiedReq, opts) =>
        dispatch(index + 1, modifiedReq || req, opts || ctx)
      );
    } else {
      return dispatch(index + 1, req, ctx);
    }
  };

  const executeTool = async (
    req: ToolRequestPart,
    ctx: ActionRunOptions<any>
  ): Promise<ToolResponsePart> => {
    // execute the tool and catch interrupts
    const output = await tool(req.toolRequest.input, ctx as ToolRunOptions);
    if (tool.__action.actionType === 'tool.v2') {
      const multipartResponse = output as z.infer<
        typeof MultipartToolResponseSchema
      >;
      return stripUndefinedProps({
        toolResponse: {
          name: req.toolRequest.name,
          ref: req.toolRequest.ref,
          output: multipartResponse.output,
          content: multipartResponse.content,
        } as ToolResponse,
        metadata: multipartResponse.metadata,
      });
    } else {
      return stripUndefinedProps({
        toolResponse: {
          name: req.toolRequest.name,
          ref: req.toolRequest.ref,
          output,
        },
      });
    }
  };

  const initialCtx = runOptions ?? toRunOptions(part);
  try {
    const dispatchResult = await dispatch(0, part, initialCtx);
    return dispatchResult ? { response: dispatchResult } : {};
  } catch (e) {
    if (
      e instanceof ToolInterruptError ||
      // There's an inexplicable case when the above type check fails, only in tests.
      (e as Error).name === 'ToolInterruptError'
    ) {
      const ie = e as ToolInterruptError;
      logger.debug(
        `tool '${tool.__action.name}' triggered an interrupt${ie.metadata ? `: ${JSON.stringify(ie.metadata)}` : ''}`
      );
      return {
        interrupt: {
          toolRequest: part.toolRequest,
          metadata: { ...part.metadata, interrupt: ie.metadata || true },
        },
      };
    }
    throw toolFailureError(part.toolRequest.name, e, initialCtx.abortSignal);
  }
}

/**
 * resolveToolRequests is responsible for executing the tools requested by the model for a single turn. it
 * returns either a toolMessage to append or a revisedModelMessage when an interrupt occurs.
 */
export async function resolveToolRequests(
  rawRequest: GenerateActionOptions,
  generatedMessage: MessageData,
  tools: ToolAction[],
  middleware: GenerateMiddlewareDef[] = [],
  options?: ToolResolveOptions
): Promise<{
  revisedModelMessage?: MessageData;
  toolMessage?: MessageData;
}> {
  const toolMap = toToolMap(tools);

  // Tools run concurrently and finish in any order. Responses are keyed by
  // their request's position in the model message and emitted in that order
  // below, so the tool message is deterministic across runs.
  const responseByIndex = new Map<number, ToolResponsePart>();
  let hasInterrupts = false;

  const revisedModelMessage = {
    ...generatedMessage,
    content: [...generatedMessage.content],
  };

  await Promise.all(
    revisedModelMessage.content.map(async (part, i) => {
      if (!part.toolRequest) return; // skip non-tool-request parts

      const { response, interrupt } = await resolveToolRequest(
        rawRequest,
        part as ToolRequestPart,
        toolMap,
        middleware,
        toRunOptions(part as ToolRequestPart, options)
      );

      if (response) {
        responseByIndex.set(i, response);
        revisedModelMessage.content[i] = toPendingOutput(part, response);
      }

      if (interrupt) {
        revisedModelMessage.content[i] = interrupt;
        hasInterrupts = true;
      }
    })
  );

  if (hasInterrupts) {
    return { revisedModelMessage };
  }

  if (responseByIndex.size === 0) {
    return {};
  }

  return {
    toolMessage: { role: 'tool', content: inRequestOrder(responseByIndex) },
  };
}

/** Returns the collected tool responses ordered by their request's position. */
function inRequestOrder(
  responseByIndex: Map<number, ToolResponsePart>
): ToolResponsePart[] {
  return [...responseByIndex.keys()]
    .sort((a, b) => a - b)
    .map((i) => responseByIndex.get(i)!);
}

function findCorrespondingToolRequest(
  parts: Part[],
  part: ToolRequestPart | ToolResponsePart
): ToolRequestPart | undefined {
  const name = part.toolRequest?.name || part.toolResponse?.name;
  const ref = part.toolRequest?.ref || part.toolResponse?.ref;

  return parts.find(
    (p) => p.toolRequest?.name === name && p.toolRequest?.ref === ref
  ) as ToolRequestPart | undefined;
}

function findCorrespondingToolResponse(
  parts: Part[],
  part: ToolRequestPart | ToolResponsePart
): ToolResponsePart | undefined {
  const name = part.toolRequest?.name || part.toolResponse?.name;
  const ref = part.toolRequest?.ref || part.toolResponse?.ref;

  return parts.find(
    (p) => p.toolResponse?.name === name && p.toolResponse?.ref === ref
  ) as ToolResponsePart | undefined;
}

async function resolveResumedToolRequest(
  rawRequest: GenerateActionOptions,
  part: ToolRequestPart,
  toolMap: Record<string, ToolAction>,
  middleware: GenerateMiddlewareDef[] = [],
  options?: ToolResolveOptions
): Promise<{
  toolRequest?: ToolRequestPart;
  toolResponse?: ToolResponsePart;
  interrupt?: ToolRequestPart;
}> {
  // Key presence, not truthiness: a tool that legitimately returned a falsy
  // output still completed, and its outcome is what the replay restores.
  if (part.metadata && 'pendingOutput' in part.metadata) {
    const { pendingOutput, pendingContent, pendingMetadata, ...metadata } =
      part.metadata;
    // Restore the multipart content and the response metadata the original
    // call carried, stashed next to pendingOutput by `toPendingOutput`. Both
    // may have been through a JSON round-trip, so they are taken as-is.
    const toolResponse: ToolResponsePart = {
      toolResponse: {
        name: part.toolRequest.name,
        ref: part.toolRequest.ref,
        output: pendingOutput,
        ...(Array.isArray(pendingContent) &&
          pendingContent.length > 0 && { content: pendingContent }),
      },
      metadata: {
        ...metadata,
        source: 'pending',
        ...(pendingMetadata && typeof pendingMetadata === 'object'
          ? pendingMetadata
          : {}),
      },
    };

    // strip the pending keys from metadata when returning
    return stripUndefinedProps({
      toolResponse,
      toolRequest: { ...part, metadata },
    });
  }

  // if there's a corresponding reply, append it to toolResponses
  const providedResponse = findCorrespondingToolResponse(
    rawRequest.resume?.respond || [],
    part
  );
  if (providedResponse) {
    const toolResponse = providedResponse;

    // remove the 'interrupt' but leave a 'resolvedInterrupt'
    const { interrupt, ...metadata } = part.metadata || {};
    return stripUndefinedProps({
      toolResponse,
      toolRequest: {
        ...part,
        metadata: { ...metadata, resolvedInterrupt: interrupt },
      },
    });
  }

  // if there's a corresponding restart, execute then add to toolResponses
  const restartRequest = findCorrespondingToolRequest(
    rawRequest.resume?.restart || [],
    part
  );
  if (restartRequest) {
    const { response, interrupt } = await resolveToolRequest(
      rawRequest,
      restartRequest,
      toolMap,
      middleware,
      toRunOptions(restartRequest, options)
    );

    // if there's a new interrupt, return it
    if (interrupt) return { interrupt };

    if (response) {
      const toolResponse = response;

      // remove the 'interrupt' but leave a 'resolvedInterrupt'
      const { interrupt, ...metadata } = part.metadata || {};
      return stripUndefinedProps({
        toolResponse,
        toolRequest: {
          ...part,
          metadata: { ...metadata, resolvedInterrupt: interrupt },
        },
      });
    }
  }

  throw new GenkitError({
    status: 'INVALID_ARGUMENT',
    message: `Unresolved tool request '${part.toolRequest.name}${part.toolRequest.ref ? `#${part.toolRequest.ref}` : ''}' was not handled by the 'resume' argument. You must supply replies or restarts for all interrupted tool requests.'`,
  });
}

/** Amends message history to handle `resume` arguments. Returns the amended history. */
export async function resolveResumeOption(
  registry: Registry,
  rawRequest: GenerateActionOptions,
  tools: ToolAction[],
  middleware: GenerateMiddlewareDef[] = [],
  options?: ToolResolveOptions
): Promise<{
  revisedRequest?: GenerateActionOptions;
  interruptedResponse?: GenerateResponseData;
  toolMessage?: MessageData;
}> {
  if (!rawRequest.resume) return { revisedRequest: rawRequest }; // no-op if no resume option
  const toolMap = toToolMap(tools);

  const messages = rawRequest.messages;
  const lastMessage = messages.at(-1);

  if (
    !lastMessage ||
    lastMessage.role !== 'model' ||
    !lastMessage.content.find((p) => p.toolRequest)
  ) {
    throw new GenkitError({
      status: 'FAILED_PRECONDITION',
      message: `Cannot 'resume' generation unless the previous message is a model message with at least one tool request.`,
    });
  }

  // Directives resolve concurrently; responses are keyed by their request's
  // position so the resumed tool message is emitted in request order, matching
  // a first-run tool message.
  const responseByIndex = new Map<number, ToolResponsePart>();
  let interrupted = false;

  const newContent = await Promise.all(
    lastMessage.content.map(async (part, i) => {
      if (!isToolRequest(part)) return part;
      const resolved = await resolveResumedToolRequest(
        rawRequest,
        part,
        toolMap,
        middleware,
        options
      );
      if (resolved.interrupt) {
        interrupted = true;
        return resolved.interrupt;
      }

      responseByIndex.set(i, resolved.toolResponse!);
      return resolved.toolRequest!;
    })
  );

  if (interrupted) {
    // Siblings resolved in this resume (restarted runs, supplied responses,
    // replayed pending outputs) are preserved as pendingOutput on their
    // request parts, the way a first-run interrupt preserves completed
    // siblings, so the next resume replays their outcomes instead of
    // demanding new directives.
    for (const [i, response] of responseByIndex) {
      newContent[i] = toPendingOutput(
        newContent[i] as ToolRequestPart,
        response
      );
    }
    lastMessage.content = newContent;
    return {
      interruptedResponse: {
        finishReason: 'interrupted',
        finishMessage:
          'One or more tools triggered interrupts while resuming generation. The model was not called.',
        message: lastMessage,
      },
    };
  }
  lastMessage.content = newContent;

  const numToolRequests = lastMessage.content.filter(
    (p) => !!p.toolRequest
  ).length;
  if (responseByIndex.size !== numToolRequests) {
    throw new GenkitError({
      status: 'FAILED_PRECONDITION',
      message: `Expected ${numToolRequests} tool responses but resolved to ${responseByIndex.size}.`,
      detail: {
        toolResponses: inRequestOrder(responseByIndex),
        message: lastMessage,
      },
    });
  }

  const toolMessage: MessageData = {
    role: 'tool',
    content: inRequestOrder(responseByIndex),
    metadata: {
      resumed: rawRequest.resume.metadata || true,
    },
  };

  return stripUndefinedProps({
    revisedRequest: {
      ...rawRequest,
      resume: undefined,
      messages: [...messages, toolMessage],
    },
    toolMessage,
  });
}

export async function resolveRestartedTools(
  registry: Registry,
  rawRequest: GenerateActionOptions,
  middleware: GenerateMiddlewareDef[] = [],
  options?: ToolResolveOptions
): Promise<ToolRequestPart[]> {
  const tools = await resolveTools(registry, rawRequest.tools);
  // rawRequest.tools only holds user-provided tools (treated as immutable). We must
  // harvest active middleware tools here to ensure we can resolve tools dynamically
  // injected by plugins.
  tools.push(...middleware.flatMap((m) => m.tools || []));
  const toolMap = toToolMap(tools);
  const lastMessage = rawRequest.messages.at(-1);
  if (!lastMessage || lastMessage.role !== 'model') return [];

  const restarts = lastMessage.content.filter(
    (p) => p.toolRequest && p.metadata?.resumed
  ) as ToolRequestPart[];

  return await Promise.all(
    restarts.map(async (p) => {
      const { response, interrupt } = await resolveToolRequest(
        rawRequest,
        p,
        toolMap,
        middleware,
        toRunOptions(p, options)
      );

      // this means that it interrupted *again* after the restart
      if (interrupt) return interrupt;
      return toPendingOutput(p, response!);
    })
  );
}
