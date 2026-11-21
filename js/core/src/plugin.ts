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

import type { Action, ActionMetadata } from './action.js';
import type { BackgroundAction } from './background-action.js';
import type { ActionType } from './registry.js';

/**
 * Registry-level plugin hooks. Internal plumbing: plugin authors use
 * `genkitPlugin({...})` from `genkit/plugin`, and `Genkit` adapts it into this
 * shape when registering the plugin.
 */
export interface PluginProvider {
  name: string;
  initializer: () => void | Promise<void>;
  resolver?: (action: ActionType, target: string) => Promise<void>;
  listActions?: () => Promise<ActionMetadata[]>;
}

export type ResolvableAction = Action | BackgroundAction;
