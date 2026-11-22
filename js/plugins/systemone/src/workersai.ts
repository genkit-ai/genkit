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
import { errorMessage } from './client.js';

/**
 * Builds a request the way Workers AI runs the model. Its own models, under
 * `@cf/`, run at `/ai/run/<ID>` with the native body, which names the model
 * by its last segment, as `clef`. A partner's model runs at `/ai/run` with
 * the native body under `input` and the model beside it. A model ID goes into
 * the path one escaped segment at a time, and one with an empty or dot segment
 * is refused, so no ID can change the query or reach another path of the
 * account's API with its token.
 */
export function workersAIRoute(
  model: string,
  body: Record<string, unknown>
): { path?: string; body: unknown } {
  if (!model.startsWith('@cf/')) {
    delete body.model;
    return { body: { model, input: body } };
  }
  const segments = model.split('/');
  for (const segment of segments) {
    if (segment === '' || segment === '.' || segment === '..') {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message: `systemone: "${model}" is not a Workers AI model ID`,
      });
    }
  }
  body.model = segments[segments.length - 1];
  return {
    path: '/' + segments.map(pathEscape).join('/'),
    body,
  };
}

/**
 * Handles both response shapes the Workers AI REST API is documented with: the
 * bare model output, and the `{result, success, errors}` envelope the rest of
 * the API uses. A failed envelope's messages are read the way an error
 * response's are.
 */
export function unwrapWorkersAI(body: unknown): unknown {
  if (
    !body ||
    typeof body !== 'object' ||
    typeof (body as any).success !== 'boolean'
  ) {
    return body;
  }
  const envelope = body as { success: boolean; result?: unknown };
  if (!envelope.success) {
    throw new Error(errorMessage(JSON.stringify(body)));
  }
  return envelope.result;
}

/**
 * Escapes one path segment as Go's url.PathEscape does, which the Workers AI
 * paths are documented with: `@`, `:`, `$`, `&`, `+`, and `=` stay as they
 * are, and everything else outside the unreserved set is escaped.
 */
function pathEscape(segment: string): string {
  return encodeURIComponent(segment)
    .replace(/%(24|26|2B|3A|3D|40)/gi, (escape) => decodeURIComponent(escape))
    .replace(
      /[!'()*]/g,
      (c) => `%${c.charCodeAt(0).toString(16).toUpperCase()}`
    );
}
