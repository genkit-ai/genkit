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

/**
 * Deletes any properties with `undefined` values in the provided object.
 * Modifies the provided object.
 */
export function deleteUndefinedProps(obj: any) {
  for (const prop in obj) {
    if (obj[prop] === undefined) {
      delete obj[prop];
    } else {
      if (typeof obj[prop] === 'object') {
        deleteUndefinedProps(obj[prop]);
      }
    }
  }
}

/**
 * Strips (non distructively) any properties with `undefined` values in the provided object and returns
 */
export function stripUndefinedProps<T>(input: T): T {
  if (
    input === undefined ||
    input === null ||
    Array.isArray(input) ||
    typeof input !== 'object'
  ) {
    return input;
  }
  const out = {} as T;
  for (const key in input) {
    if (input[key] !== undefined) {
      out[key] = stripUndefinedProps(input[key]);
    }
  }
  return out;
}

/**
 * Returns the current environment that the app code is running in.
 *
 * @hidden
 */
export function getCurrentEnv(): string {
  return process.env.GENKIT_ENV || 'prod';
}

/**
 * Whether the current environment is `dev`.
 */
export function isDevEnv(): boolean {
  return getCurrentEnv() === 'dev';
}

/**
 * Adds flow-specific prefix for OpenTelemetry span attributes.
 */
export function featureMetadataPrefix(name: string) {
  return `feature:${name}`;
}

/**
 * Deep-equality check for plain JSON-serializable values.
 * Handles objects, arrays, and primitives. Does not handle functions, dates,
 * or other non-JSON types.
 */
export function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a == null || b == null) return a === b;
  if (typeof a !== typeof b) return false;
  if (typeof a !== 'object') return false;

  if (Array.isArray(a)) {
    if (!Array.isArray(b) || a.length !== b.length) return false;
    return a.every((v, i) => deepEqual(v, b[i]));
  }

  const aObj = a as Record<string, unknown>;
  const bObj = b as Record<string, unknown>;
  const aKeys = Object.keys(aObj).sort();
  const bKeys = Object.keys(bObj).sort();
  if (aKeys.length !== bKeys.length) return false;
  return aKeys.every(
    (key, i) => key === bKeys[i] && deepEqual(aObj[key], bObj[key])
  );
}
