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

// Type-level assertions for wire types. Their schemas keep unknown keys at
// runtime (`wireObject`), but the inferred types must stay closed: no index
// signature, so typos are still caught. Never executed: type-checked by
// `tsconfig.types-test.json` (part of `pnpm check`).

import type { DocumentData } from '../../src/document.js';
import type {
  GenerateResponseData,
  MessageData,
} from '../../src/model-types.js';
import type { Part } from '../../src/parts.js';

// @ts-expect-error unknown keys are not part of `Part`.
export const part: Part = { text: 'hi', extra: 1 };

export const message: MessageData = {
  role: 'user',
  content: [{ text: 'hi' }],
  // @ts-expect-error unknown keys are not part of `MessageData`.
  extra: 1,
};

export const doc: DocumentData = {
  content: [{ text: 'hi' }],
  // @ts-expect-error unknown keys are not part of `DocumentData`.
  extra: 1,
};

export function noIndexSignature(p: Part, r: GenerateResponseData) {
  // @ts-expect-error reading an undeclared key is an error, not `unknown`.
  void p.extra;
  // @ts-expect-error reading an undeclared key is an error, not `unknown`.
  void r.extra;
  // @ts-expect-error same for nested wire types.
  void r.usage?.extra;
}
