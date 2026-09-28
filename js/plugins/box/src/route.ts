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

import { randomUUID } from 'node:crypto';
import type { Retention, RouteFn } from './types.js';

/**
 * Route functions may advertise a sensible default retention via this symbol.
 * The box core uses it when the caller doesn't set `retention` explicitly.
 */
export const RETENTION = Symbol('box.retention');

/** A route function that also carries a default retention. */
export interface PresetRouteFn extends RouteFn {
  [RETENTION]?: Retention;
}

/** Fixed routing key for the single long-lived box. */
export const SINGLETON_KEY = '__singleton__';

/** Default idle window (ms) for session routing when nothing else applies. */
export const DEFAULT_SESSION_IDLE_MS = 5 * 60_000;

/**
 * Singleton (default): every call goes to one long-lived box, never reclaimed.
 */
export const singleton: PresetRouteFn = Object.assign(
  (): string => SINGLETON_KEY,
  { [RETENTION]: {} satisfies Retention } // no idle -> never reclaimed
);

/**
 * Per-request: a fresh box per call, reclaimed as soon as the call returns.
 */
export const perRequest: PresetRouteFn = Object.assign(
  (): string => randomUUID(),
  { [RETENTION]: { idle: 0 } satisfies Retention }
);

/** Resolves the effective retention for a box from its options + route. */
export function resolveRetention(
  route: RouteFn,
  explicit?: Retention
): Retention {
  if (explicit) return explicit;
  const preset = (route as PresetRouteFn)[RETENTION];
  if (preset) return preset;
  // Custom route with no explicit retention: treat as session-scoped.
  return { idle: DEFAULT_SESSION_IDLE_MS };
}
