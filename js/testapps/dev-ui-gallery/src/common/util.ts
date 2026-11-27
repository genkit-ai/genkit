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

const loremIpsum = [
  'lorem',
  'ipsum',
  'dolor',
  'sit',
  'amet',
  'consectetur',
  'adipiscing',
  'elit',
];

export function generateString(length: number) {
  let str = '';
  while (str.length < length) {
    str += loremIpsum[Math.floor(Math.random() * loremIpsum.length)] + ' ';
  }
  return str.substring(0, length);
}

/**
 * Returns a sorted, deduplicated list of registered tool names from a Genkit instance.
 */
export function getRegisteredToolNames(ai?: { registry?: unknown }): string[] {
  const actionsById: Record<string, unknown> =
    (ai?.registry as any)?.actionsById ?? {};
  return Array.from(
    new Set(
      Object.keys(actionsById)
        .filter(
          (key) => key.startsWith('/tool/') || key.startsWith('/tool.v2/')
        )
        .map((key) => key.replace(/^\/tool(?:\.v2)?\//, ''))
    )
  ).sort();
}

/**
 * Returns an array reference whose `toJSON()` hook resolves the registered
 * tool names lazily when serialized by the reflection server for the Dev UI.
 */
export function lazyToolNames(getAi: () => { registry?: unknown }): string[] {
  return Object.assign([] as string[], {
    toJSON: () => getRegisteredToolNames(getAi()),
  });
}
