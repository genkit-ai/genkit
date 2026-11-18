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

/**
 * Env var carrying the id of the box a runtime serves. Lets nested self-mode
 * boxes tell "I am this box's runtime" (don't re-spawn) from "I am a manager
 * for a different box" (spawn normally). Runners set it on every runtime they
 * start, overwriting any inherited value at each nesting level.
 */
export const BOX_SELF_ID_ENV = 'GENKIT_BOX_SELF_ID';
