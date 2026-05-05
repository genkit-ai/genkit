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

import type {
  GenerateResponseChunkData,
  Part,
  StreamingCallback,
} from 'genkit';
import { GenkitError } from 'genkit';
import { logger } from 'genkit/logging';
import type { ResponseStreamEvent } from 'openai/resources/responses/responses';

/**
 * Per-output-item state we accumulate while consuming the SSE event stream.
 *
 * Discriminated union by `type` so each branch carries only the fields it
 * actually uses — no optional state correlated with another field.
 */
type ItemState =
  | MessageItemState
  | FunctionCallItemState
  | BuiltInCallItemState
  | ReasoningItemState
  | UnknownItemState;

interface MessageItemState {
  type: 'message';
  /** Annotations buffered until item.done so we can flush a metadata-only chunk. */
  annotations: unknown[];
}

interface FunctionCallItemState {
  type: 'function_call';
  /** Concatenated argument deltas. Replaced verbatim on `.done` events. */
  argsBuf: string;
  callId: string;
  name: string;
}

interface BuiltInCallItemState {
  type: 'web_search_call' | 'file_search_call' | 'code_interpreter_call';
}

interface ReasoningItemState {
  type: 'reasoning';
}

interface UnknownItemState {
  type: 'unknown';
}

/** Citation as it can appear inside `output_text_annotation.added`. */
interface UrlCitationAnnotation {
  type: 'url_citation';
  url: string;
  title?: string;
  start_index?: number;
  end_index?: number;
}

interface FileCitationAnnotation {
  type: 'file_citation';
  file_id: string;
  index?: number;
}

function isUrlCitation(a: unknown): a is UrlCitationAnnotation {
  return (
    typeof a === 'object' &&
    a !== null &&
    (a as { type?: unknown }).type === 'url_citation' &&
    typeof (a as { url?: unknown }).url === 'string'
  );
}

function isFileCitation(a: unknown): a is FileCitationAnnotation {
  return (
    typeof a === 'object' &&
    a !== null &&
    (a as { type?: unknown }).type === 'file_citation' &&
    typeof (a as { file_id?: unknown }).file_id === 'string'
  );
}

/**
 * Built-in tool call event type families. Each family has 3-4 lifecycle
 * sub-events (in_progress, searching/interpreting, completed). We list
 * every (kind, status) pair explicitly rather than substring-matching the
 * raw event type string — this avoids drift if OpenAI ever introduces
 * `web_search_call_v2` or similar.
 */
const BUILT_IN_CALL_EVENTS: ReadonlyMap<
  string,
  { kind: BuiltInCallItemState['type']; status: string }
> = new Map([
  [
    'response.web_search_call.in_progress',
    { kind: 'web_search_call', status: 'in_progress' },
  ],
  [
    'response.web_search_call.searching',
    { kind: 'web_search_call', status: 'searching' },
  ],
  [
    'response.web_search_call.completed',
    { kind: 'web_search_call', status: 'completed' },
  ],
  [
    'response.file_search_call.in_progress',
    { kind: 'file_search_call', status: 'in_progress' },
  ],
  [
    'response.file_search_call.searching',
    { kind: 'file_search_call', status: 'searching' },
  ],
  [
    'response.file_search_call.completed',
    { kind: 'file_search_call', status: 'completed' },
  ],
  [
    'response.code_interpreter_call.in_progress',
    { kind: 'code_interpreter_call', status: 'in_progress' },
  ],
  [
    'response.code_interpreter_call.interpreting',
    { kind: 'code_interpreter_call', status: 'interpreting' },
  ],
  [
    'response.code_interpreter_call.completed',
    { kind: 'code_interpreter_call', status: 'completed' },
  ],
]);

/**
 * Drive the OpenAI Responses API SSE event stream and translate events to
 * Genkit chunks.
 *
 * The aggregator does NOT itself produce a `GenerateResponseData` — that's
 * the job of `fromResponsesResponse(stream.finalResponse())` in the
 * runner. Here we emit one chunk per *meaningful* event so callers see
 * tokens as they arrive, and we keep enough state to attach annotations
 * and to surface aggregated tool calls on `output_item.done`.
 *
 * Throws a {@link GenkitError} if the stream emits a terminal `error`
 * event — the SDK uses this for stream-level failures (network, mid-stream
 * abort, server error) that won't otherwise surface through
 * `stream.finalResponse()`.
 *
 * @param stream Async iterable returned by `client.responses.stream(...)`.
 * @param sendChunk Genkit's streaming callback.
 */
export async function streamResponsesEvents(
  stream: AsyncIterable<ResponseStreamEvent>,
  sendChunk: StreamingCallback<GenerateResponseChunkData>
): Promise<void> {
  const items = new Map<number, ItemState>();
  let streamError: { message: string; code?: string } | undefined;

  for await (const event of stream) {
    switch (event.type) {
      case 'response.created':
      case 'response.in_progress':
      case 'response.queued':
        // Lifecycle markers — no chunk needed.
        break;

      case 'response.output_item.added': {
        const item = event.item;
        if (item.type === 'message') {
          items.set(event.output_index, { type: 'message', annotations: [] });
        } else if (item.type === 'function_call') {
          items.set(event.output_index, {
            type: 'function_call',
            argsBuf: '',
            callId: item.call_id,
            name: item.name,
          });
        } else if (item.type === 'reasoning') {
          items.set(event.output_index, { type: 'reasoning' });
        } else if (
          item.type === 'web_search_call' ||
          item.type === 'file_search_call' ||
          item.type === 'code_interpreter_call'
        ) {
          items.set(event.output_index, { type: item.type });
        } else {
          items.set(event.output_index, { type: 'unknown' });
        }
        break;
      }

      case 'response.output_text.delta': {
        sendChunk({
          index: event.output_index,
          role: 'model',
          content: [{ text: event.delta }],
        });
        break;
      }

      case 'response.output_text_annotation.added': {
        const state = items.get(event.output_index);
        if (state?.type === 'message') {
          state.annotations.push(event.annotation);
        } else {
          // Annotation arrived for a non-message item or before
          // output_item.added — drop with a debug log so a future
          // SDK quirk is at least diagnosable.
          logger.debug(
            `[openai-responses] dropping annotation for output_index=` +
              `${event.output_index} (state=${state?.type ?? 'missing'})`
          );
        }
        break;
      }

      case 'response.output_text.done': {
        // Final text per item — annotations attached on item.done.
        break;
      }

      case 'response.refusal.delta': {
        // Surface refusal text via `custom.refusalDelta` so callers can
        // accumulate or display it. We deliberately omit `text` here:
        // an empty-string text Part would be concatenated by Genkit's
        // chunk merger into the final message and pollute the model's
        // output. The refusal also surfaces structurally on the final
        // Response (see fromResponsesResponse → blockedReason).
        sendChunk({
          index: event.output_index,
          role: 'model',
          content: [{ custom: { refusalDelta: event.delta } }],
        });
        break;
      }

      case 'response.reasoning_summary_text.delta': {
        sendChunk({
          index: event.output_index,
          role: 'model',
          content: [{ reasoning: event.delta }],
        });
        break;
      }

      case 'response.function_call_arguments.delta': {
        const state = items.get(event.output_index);
        if (state && state.type === 'function_call') {
          state.argsBuf += event.delta;
        } else {
          // output_item.added missed (network reorder / SDK glitch).
          // Lazy-init so we don't lose the deltas — `name` and
          // `call_id` will be filled from event.item on .done.
          logger.warn(
            `[openai-responses] function_call_arguments.delta for ` +
              `output_index=${event.output_index} without prior ` +
              `output_item.added — initializing state lazily`
          );
          items.set(event.output_index, {
            type: 'function_call',
            argsBuf: event.delta,
            callId: '',
            name: '',
          });
        }
        break;
      }

      case 'response.function_call_arguments.done': {
        const state = items.get(event.output_index);
        if (state && state.type === 'function_call' && event.arguments) {
          // Prefer the canonical full string from .done over our
          // accumulated buffer — but only if .done actually carries
          // arguments (don't clobber a populated buffer with empty).
          state.argsBuf = event.arguments;
        }
        break;
      }

      case 'response.web_search_call.in_progress':
      case 'response.web_search_call.searching':
      case 'response.web_search_call.completed':
      case 'response.file_search_call.in_progress':
      case 'response.file_search_call.searching':
      case 'response.file_search_call.completed':
      case 'response.code_interpreter_call.in_progress':
      case 'response.code_interpreter_call.interpreting':
      case 'response.code_interpreter_call.completed': {
        const meta = BUILT_IN_CALL_EVENTS.get(event.type);
        if (!meta) {
          break;
        }
        const itemId = (event as { item_id?: string }).item_id;
        sendChunk({
          index: event.output_index,
          role: 'model',
          content: [
            {
              custom: {
                kind: meta.kind,
                status: meta.status,
                ...(itemId ? { itemId } : {}),
              },
            },
          ],
        });
        break;
      }

      case 'response.output_item.done': {
        const state = items.get(event.output_index);
        if (event.item.type === 'function_call') {
          // Resilient even if state was never created (output_item.added
          // missed). Use event.item as the source of truth for name /
          // call_id; arguments come from buffer or fall back to event.item.
          const fcState =
            state && state.type === 'function_call' ? state : undefined;
          const rawArgs =
            fcState?.argsBuf || event.item.arguments || '{}';
          let parsed: unknown;
          let malformed = false;
          try {
            parsed = JSON.parse(rawArgs);
          } catch (e) {
            malformed = true;
            parsed = rawArgs;
            logger.warn(
              `[openai-responses] function_call "${event.item.name}" ` +
                `(call_id=${event.item.call_id}) returned non-JSON ` +
                `arguments; surfacing raw string with ` +
                `metadata.malformedArguments=true. Error: ${(e as Error).message}`
            );
          }
          const part: Part = {
            toolRequest: {
              name: event.item.name,
              ref: event.item.call_id,
              input: parsed,
            },
          };
          if (malformed) {
            part.metadata = { malformedArguments: true };
          }
          sendChunk({
            index: event.output_index,
            role: 'model',
            content: [part],
          });
        } else if (
          event.item.type === 'message' &&
          state?.type === 'message' &&
          state.annotations.length > 0
        ) {
          // Flush a final chunk that carries citations on metadata.
          // The text content has already been streamed via deltas;
          // this chunk carries metadata only (empty text avoids
          // re-emitting the full text).
          const citations: Array<
            | {
                type: 'url_citation';
                url: string;
                title?: string;
                startIndex?: number;
                endIndex?: number;
              }
            | { type: 'file_citation'; fileId: string; fileIndex?: number }
          > = [];
          for (const a of state.annotations) {
            if (isUrlCitation(a)) {
              citations.push({
                type: 'url_citation',
                url: a.url,
                title: a.title,
                startIndex: a.start_index,
                endIndex: a.end_index,
              });
            } else if (isFileCitation(a)) {
              citations.push({
                type: 'file_citation',
                fileId: a.file_id,
                fileIndex: a.index,
              });
            }
          }
          if (citations.length > 0) {
            const part: Part = { text: '', metadata: { citations } };
            sendChunk({
              index: event.output_index,
              role: 'model',
              content: [part],
            });
          }
        }
        items.delete(event.output_index);
        break;
      }

      case 'error': {
        // Stream-level error from the SDK (network, mid-stream abort,
        // server error). Capture and re-throw at end of iteration so
        // the caller's try/catch sees a structured failure rather than
        // a silently-truncated response.
        const e = event as {
          message?: string;
          code?: string;
          error?: { message?: string; code?: string };
        };
        streamError = {
          message:
            e.message ?? e.error?.message ?? 'unknown OpenAI stream error',
          code: e.code ?? e.error?.code,
        };
        logger.error(
          `[openai-responses] stream error event received: ` +
            `${streamError.message}` +
            (streamError.code ? ` (code=${streamError.code})` : '')
        );
        break;
      }

      case 'response.completed':
      case 'response.incomplete':
      case 'response.failed':
        // Terminal events. The runner reads `stream.finalResponse()`
        // for the canonical aggregated Response; we don't emit a chunk
        // here so callers are not double-counted.
        break;

      default:
        // Forward-compat: unknown events are ignored. The final
        // aggregated response (via finalResponse()) still carries
        // the full structure, so callers don't lose data.
        break;
    }
  }

  if (streamError) {
    throw new GenkitError({
      status: 'INTERNAL',
      message:
        `OpenAI Responses stream error: ${streamError.message}` +
        (streamError.code ? ` (code=${streamError.code})` : ''),
    });
  }
}
