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

import type { GenerateRequest, GenerateResponseData, Part } from 'genkit';
import { logger } from 'genkit/logging';
import type {
  Response,
  ResponseOutputItem,
  ResponseOutputText,
  ResponseStatus,
} from 'openai/resources/responses/responses';
import type { Citation } from './types';

/**
 * Map a Responses API top-level {@link ResponseStatus} (plus optional
 * `incomplete_details.reason`) to a Genkit `finishReason`.
 *
 * Refusal-driven blocks are detected in {@link fromResponsesResponse} by
 * scanning the output items, since `status === 'completed'` is reported
 * even when the model refused.
 */
function statusToFinishReason(
  status: ResponseStatus,
  incompleteReason?: string
): GenerateResponseData['finishReason'] {
  switch (status) {
    case 'completed':
      return 'stop';
    case 'incomplete':
      if (incompleteReason === 'max_output_tokens') {
        return 'length';
      }
      if (incompleteReason === 'content_filter') {
        return 'blocked';
      }
      return 'other';
    case 'failed':
      return 'other';
    case 'cancelled':
      return 'interrupted';
    case 'in_progress':
    case 'queued':
      // Reaching final-response handling with a non-terminal status is
      // an invariant violation — log loudly so operators can diagnose.
      logger.warn(
        `[openai-responses] non-terminal status "${status}" on final ` +
          `response; treating as 'unknown' finishReason`
      );
      return 'unknown';
    default: {
      // Exhaustiveness check — TypeScript will fail here if OpenAI adds
      // a new ResponseStatus literal that we haven't handled.
      const _exhaustive: never = status;
      logger.warn(
        `[openai-responses] unrecognized response status: ${String(_exhaustive)}`
      );
      return 'unknown';
    }
  }
}

/**
 * Convert annotation array on a Responses API `output_text` part into the
 * structured {@link Citation}[] we attach to the corresponding Genkit text
 * Part via `metadata.citations`.
 *
 * `url_citation` (web search) and `file_citation` (file_search) become
 * separate Citation variants. `file_path` annotations are dropped —
 * they point at sandbox-generated files on the OpenAI side and are not
 * actionable for callers.
 */
function annotationsToCitations(
  annotations: ResponseOutputText['annotations']
): Citation[] {
  const out: Citation[] = [];
  for (const a of annotations) {
    if (a.type === 'url_citation') {
      out.push({
        type: 'url_citation',
        url: a.url,
        title: a.title,
        startIndex: a.start_index,
        endIndex: a.end_index,
      });
    } else if (a.type === 'file_citation') {
      // `a.index` from the SDK is the position of the file in the
      // file_search_call's results[] array — NOT a character offset
      // into the text. Surface it as a discriminated `file_citation`
      // with an explicit `fileIndex` field.
      out.push({
        type: 'file_citation',
        fileId: a.file_id,
        fileIndex: a.index,
      });
    }
  }
  return out;
}

/**
 * Convert a single {@link ResponseOutputItem} to one or more Genkit Parts.
 *
 * Returns an array because a single `message` item can produce multiple
 * parts (one per content block), and a single `function_call` item produces
 * exactly one toolRequest Part.
 */
function outputItemToParts(item: ResponseOutputItem): {
  parts: Part[];
  blocked?: { reason: string };
} {
  switch (item.type) {
    case 'message': {
      const parts: Part[] = [];
      let blocked: { reason: string } | undefined;
      for (const c of item.content) {
        if (c.type === 'output_text') {
          const citations = annotationsToCitations(c.annotations);
          const part: Part = { text: c.text };
          if (citations.length > 0) {
            part.metadata = { citations };
          }
          parts.push(part);
        } else if (c.type === 'refusal') {
          blocked = { reason: c.refusal };
        }
      }
      return { parts, blocked };
    }
    case 'reasoning': {
      const summary = item.summary
        .map((s) => s.text)
        .filter(Boolean)
        .join('\n');
      if (!summary && !item.encrypted_content) {
        return { parts: [] };
      }
      const part: Part = { reasoning: summary };
      if (item.encrypted_content) {
        part.metadata = { encrypted: true };
      }
      return { parts: [part] };
    }
    case 'function_call': {
      let args: unknown;
      let malformed = false;
      try {
        args = JSON.parse(item.arguments || '{}');
      } catch (e) {
        malformed = true;
        args = item.arguments;
        logger.warn(
          `[openai-responses] function_call "${item.name}" (call_id=${item.call_id}) ` +
            `returned non-JSON arguments; surfacing raw string with ` +
            `metadata.malformedArguments=true. Error: ${(e as Error).message}`
        );
      }
      const part: Part = {
        toolRequest: {
          name: item.name,
          ref: item.call_id,
          input: args,
        },
      };
      if (malformed) {
        part.metadata = { malformedArguments: true };
      }
      return { parts: [part] };
    }
    case 'web_search_call': {
      // Surface as a custom Part so downstream code can show progress
      // / debug. The actual citations on text Parts carry the
      // user-visible value.
      return {
        parts: [
          {
            custom: {
              kind: 'web_search_call',
              id: item.id,
              status: item.status,
            },
          },
        ],
      };
    }
    case 'file_search_call': {
      return {
        parts: [
          {
            custom: {
              kind: 'file_search_call',
              id: item.id,
              status: item.status,
              queries: item.queries,
              results: item.results ?? null,
            },
          },
        ],
      };
    }
    case 'code_interpreter_call': {
      return {
        parts: [
          {
            custom: {
              kind: 'code_interpreter_call',
              id: item.id,
              status: item.status,
              code: item.code,
              // `outputs` is only populated when the caller
              // requested `include: ['code_interpreter_call.outputs']`.
              outputs: (item as { outputs?: unknown }).outputs ?? null,
            },
          },
        ],
      };
    }
    default:
      // Unknown / future item types (computer_use, mcp, image_gen,
      // local_shell) are surfaced as opaque custom Parts for forward
      // compatibility — better to expose them than silently drop.
      return {
        parts: [
          {
            custom: {
              kind: 'opaque_output_item',
              type: (item as { type: string }).type,
              data: item as unknown,
            },
          },
        ],
      };
  }
}

/**
 * Convert an OpenAI Responses API {@link Response} into a Genkit
 * {@link GenerateResponseData}.
 *
 * Surfaces server-side metadata via `custom.responseId` (so callers can
 * pass it back as `previousResponseId` to chain turns) and
 * `usage.custom.reasoningTokens` / `cachedInputTokens`.
 */
export function fromResponsesResponse(
  response: Response,
  _request: GenerateRequest
): GenerateResponseData {
  const allParts: Part[] = [];
  let blockedReason: string | undefined;

  for (const item of response.output) {
    const { parts, blocked } = outputItemToParts(item);
    allParts.push(...parts);
    if (blocked) {
      blockedReason = blocked.reason;
    }
  }

  let finishReason = statusToFinishReason(
    response.status as ResponseStatus,
    response.incomplete_details?.reason
  );
  let finishMessage: string | undefined;
  if (blockedReason) {
    finishReason = 'blocked';
    finishMessage = blockedReason;
  } else if (response.error?.message) {
    finishMessage = response.error.message;
  }
  if (response.status === 'failed' && response.error) {
    // Operational error from OpenAI — log so operators can correlate
    // with quota/rate/account issues rather than silently degrading.
    logger.error(
      `[openai-responses] response failed (id=${response.id}, ` +
        `code=${response.error.code ?? 'unknown'}): ${response.error.message ?? '<no message>'}`
    );
  }

  return {
    finishReason,
    ...(finishMessage ? { finishMessage } : {}),
    message: {
      role: 'model',
      content: allParts,
    },
    usage: response.usage
      ? {
          inputTokens: response.usage.input_tokens,
          outputTokens: response.usage.output_tokens,
          totalTokens: response.usage.total_tokens,
          ...(response.usage.input_tokens_details?.cached_tokens != null
            ? {
                cachedContentTokens:
                  response.usage.input_tokens_details.cached_tokens,
              }
            : {}),
          custom: {
            ...(response.usage.output_tokens_details?.reasoning_tokens != null
              ? {
                  reasoningTokens:
                    response.usage.output_tokens_details.reasoning_tokens,
                }
              : {}),
          },
        }
      : undefined,
    custom: {
      responseId: response.id,
      ...(response.metadata ? { metadata: response.metadata } : {}),
      ...(response.error?.code ? { errorCode: response.error.code } : {}),
    },
    raw: response,
  };
}
