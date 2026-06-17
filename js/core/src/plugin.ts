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

import type { z } from 'zod';
import type { Action, ActionMetadata } from './action.js';
import type { BackgroundAction } from './background-action.js';
import type { ActionType } from './registry.js';

export interface Provider<T> {
  id: string;
  value: T;
}

export interface PluginProvider {
  name: string;
  initializer: () =>
    | InitializedPlugin
    | void
    | Promise<InitializedPlugin | void>;
  resolver?: (action: ActionType, target: string) => Promise<void>;
  listActions?: () => Promise<ActionMetadata[]>;
}

export interface InitializedPlugin {
  models?: Action<z.ZodTypeAny, z.ZodTypeAny>[];
  retrievers?: Action<z.ZodTypeAny, z.ZodTypeAny>[];
  embedders?: Action<z.ZodTypeAny, z.ZodTypeAny>[];
  indexers?: Action<z.ZodTypeAny, z.ZodTypeAny>[];
  evaluators?: Action<z.ZodTypeAny, z.ZodTypeAny>[];
  /** @deprecated */
  flowStateStore?: Provider<any> | Provider<any>[];
  /** @deprecated */
  traceStore?: Provider<any> | Provider<any>[];
  /** @deprecated */
  telemetry?: any;
}

export type Plugin<T extends any[]> = (...args: T) => PluginProvider;

export type ResolvableAction = Action | BackgroundAction;
