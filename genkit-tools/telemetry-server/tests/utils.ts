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

import type { SpanData } from '@genkit-ai/tools-common';

export function span(
  traceId: string,
  id: string,
  inputLength: number | undefined,
  outputLength: number | undefined,
  startTime?: number,
  endTime?: number
): SpanData {
  const attributes: Record<string, any> = {
    'genkit:type': 'flow',
  };
  if (inputLength) {
    attributes['genkit:input'] = generateString(inputLength);
  }
  if (outputLength) {
    attributes['genkit:output'] = generateString(outputLength);
  }
  return {
    traceId: traceId,
    spanId: id,
    displayName: `Span ${id}`,
    startTime: startTime || 1,
    endTime: endTime || 2,
    instrumentationLibrary: { name: 'genkit' },
    spanKind: 'INTERNAL',
    attributes,
    status: { code: 0 },
  } as SpanData;
}

function generateString(length: number) {
  let str = '';
  while (str.length < length) {
    str += 'blah ';
  }
  return str.substring(0, length);
}

export async function sleep(ms: number) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}
