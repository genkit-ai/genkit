/**
 * Copyright 2026 Google LLC
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

import { GenkitError } from 'genkit';
import { RawJson } from './state.js';

/**
 * Sends the native body with the images as raw base64, which Ollama takes in
 * place of data URLs. Ollama requires text in the state, so a request with no
 * text, such as one of images alone, is refused before it is sent.
 */
export function ollamaRoute(
  _model: string,
  body: Record<string, unknown>
): { path?: string; body: unknown } {
  const urls = Array.isArray(body.images) ? (body.images as string[]) : [];
  if (!hasText(body.state)) {
    let message = 'systemone: Ollama requires text in the state';
    if (urls.length > 0) message += ', so a request with images needs text too';
    throw new GenkitError({ status: 'INVALID_ARGUMENT', message });
  }
  if (urls.length > 0) {
    body.images = urls.map((url) => {
      const i = url.indexOf(';base64,');
      if (i < 0) {
        // The plugin builds every image as a base64 data URL, so this is a
        // bug, never an image to send empty.
        throw new GenkitError({
          status: 'INTERNAL',
          message: 'systemone: an image is not a base64 data URL',
        });
      }
      return url.slice(i + ';base64,'.length);
    });
  }
  return { body };
}

/**
 * Reports whether a state carries any text: a nonempty string, a turn or
 * document with text, or any other value, such as the object of a data part.
 */
function hasText(state: unknown): boolean {
  if (state === undefined || state === null) return false;
  if (typeof state === 'string') return state !== '';
  if (state instanceof RawJson) return true;
  if (Array.isArray(state)) {
    // Turns are read for their text; any other list is a data part's value,
    // or a message's several values, which carry text.
    if (state.length > 0 && state.every(isTurn)) {
      return state.some((turn) => hasText(turn.content));
    }
    return true;
  }
  if (typeof state === 'object') {
    const value = state as Record<string, unknown>;
    if (Array.isArray(value.messages) && value.messages.every(isTurn)) {
      const docs = Array.isArray(value.context) ? value.context : [];
      return hasText(value.messages) || docs.some(hasText);
    }
    return Object.keys(value).length > 0;
  }
  return true;
}

function isTurn(value: unknown): value is { role: string; content: unknown } {
  return (
    !!value &&
    typeof value === 'object' &&
    'role' in value &&
    'content' in value
  );
}
