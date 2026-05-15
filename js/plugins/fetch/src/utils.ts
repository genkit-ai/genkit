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
/**
 * Extracts error message from the given error object, or if input is not an error then just turn the error into a string.
 */
export function getErrorMessage(e: any): string {
  if (e instanceof Error) {
    return e.message;
  }
  return `${e}`;
}
/**
 * Extracts stack trace from the given error object, or if input is not an error then returns undefined.
 */
export function getErrorStack(e: any): string | undefined {
  if (e instanceof Error) {
    return e.stack;
  }
  return undefined;
}
