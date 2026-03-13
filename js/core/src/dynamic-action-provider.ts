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

import * as z from 'zod';
import {
  Action,
  ActionMetadata,
  ActionMetadataSchema,
  defineAction,
} from './action.js';
import { GenkitError } from './error.js';
import { ActionMetadataRecord, ActionType, Registry } from './registry.js';

type DapValue = {
  [K in ActionType]?: Action<z.ZodTypeAny, z.ZodTypeAny, z.ZodTypeAny>[];
};

class SimpleCache {
  private value: DapValue | undefined;
  private expiresAt: number | undefined;
  private ttlMillis: number;
  private dap: DynamicActionProviderAction | undefined;
  private dapFn: DapFn;
  private fetchPromise: Promise<DapValue> | null = null;

  constructor(config: DapConfig, dapFn: DapFn) {
    this.dapFn = dapFn;
    this.ttlMillis = !config.cacheConfig?.ttlMillis
      ? 3 * 1000
      : config.cacheConfig?.ttlMillis;
  }

  setDap(dap: DynamicActionProviderAction) {
    this.dap = dap;
  }

  setValue(value: DapValue) {
    this.value = value;
    this.expiresAt = Date.now() + this.ttlMillis;
  }

  /**
   * Gets or fetches the DAP data.
   * @param skipTrace Don't run the action. i.e. don't create a trace log.
   * @returns The DAP data
   */
  async getOrFetch(params?: { skipTrace?: boolean }): Promise<DapValue> {
    const isStale =
      !this.value ||
      !this.expiresAt ||
      this.ttlMillis < 0 ||
      Date.now() > this.expiresAt;
    if (!isStale) {
      return this.value!;
    }

    if (!this.fetchPromise) {
      this.fetchPromise = (async () => {
        try {
          if (this.dap && !params?.skipTrace) {
            await this.dap.run(); // calls setValue
          } else {
            this.setValue(await this.dapFn());
          }
          if (!this.value) {
            throw new Error('value is undefined');
          }
          return this.value;
        } catch (error) {
          console.error('Error fetching Dynamic Action Provider value:', error);
          this.invalidate();
          throw error; // Rethrow to reject the fetchPromise
        } finally {
          // Allow new fetches after this one completes or fails.
          this.fetchPromise = null;
        }
      })();
    }
    return await this.fetchPromise;
  }

  invalidate() {
    this.value = undefined;
  }
}

export interface DynamicRegistry {
  __cache: SimpleCache;
  invalidateCache(): void;
  getAction(
    actionType: string,
    actionName: string
  ): Promise<Action<z.ZodTypeAny, z.ZodTypeAny, z.ZodTypeAny> | undefined>;
  listActionMetadata(
    actionType: string,
    actionName: string
  ): Promise<ActionMetadata[]>;
  getActionMetadataRecord(dapPrefix: string): Promise<ActionMetadataRecord>;
}

export type DynamicActionProviderAction = Action<
  z.ZodVoid,
  z.ZodArray<typeof ActionMetadataSchema>,
  z.ZodTypeAny
> &
  DynamicRegistry & {
    __action: {
      metadata: {
        type: 'dynamic-action-provider';
      };
    };
  };

export function isDynamicActionProvider(
  obj: Action<z.ZodTypeAny, z.ZodTypeAny>
): obj is DynamicActionProviderAction {
  return obj.__action?.metadata?.type == 'dynamic-action-provider';
}

export interface DapConfig {
  name: string;
  description?: string;
  cacheConfig?: {
    // Negative = no caching
    // Zero or undefined = default (3000 milliseconds)
    // Positive number = how many milliseconds the cache is valid for
    ttlMillis: number | undefined;
  };
  metadata?: Record<string, any>;
}

export type DapFn = () => Promise<DapValue>;
export type DapMetadata = {
  [K in ActionType]?: ActionMetadata[];
};

function transformDapValue(value: DapValue): ActionMetadata[] {
  return Object.values(value).flatMap(
    (actions) => actions?.map((a) => a.__action) || []
  );
}

export function defineDynamicActionProvider(
  registry: Registry,
  config: DapConfig | string,
  fn: DapFn
): DynamicActionProviderAction {
  let cfg: DapConfig;
  if (typeof config == 'string') {
    cfg = { name: config };
  } else {
    cfg = { ...config };
  }
  const cache = new SimpleCache(cfg, fn);
  const a = defineAction(
    registry,
    {
      ...cfg,
      inputSchema: z.void(),
      outputSchema: z.array(ActionMetadataSchema),
      actionType: 'dynamic-action-provider',
      metadata: { ...(cfg.metadata || {}), type: 'dynamic-action-provider' },
    },
    async (_options) => {
      const dapValue = await fn();
      cache.setValue(dapValue);
      return transformDapValue(dapValue);
    }
  );
  implementDap(a as DynamicActionProviderAction, cache);
  return a as DynamicActionProviderAction;
}

function implementDap(dap: DynamicActionProviderAction, cache: SimpleCache) {
  cache.setDap(dap);
  dap.__cache = cache;
  dap.invalidateCache = () => {
    dap.__cache.invalidate();
  };

  dap.getAction = async (actionType: string, actionName: string) => {
    const result = await dap.__cache.getOrFetch();
    if (result[actionType]) {
      return result[actionType].find((t) => t.__action.name == actionName);
    }
    return undefined;
  };

  dap.listActionMetadata = async (actionType: string, actionName: string) => {
    const result = await dap.__cache.getOrFetch();
    if (!result[actionType]) {
      return [];
    }

    // Match everything in the actionType
    const metadata = result[actionType].map((a) => a.__action);
    if (actionName == '*') {
      return metadata;
    }

    // Prefix matching
    if (actionName.endsWith('*')) {
      const prefix = actionName.slice(0, -1);
      return metadata.filter((m) => m.name.startsWith(prefix));
    }

    // Single match or empty array
    return metadata.filter((m) => m.name == actionName);
  };

  // This is called by listResolvableActions which is used by the
  // reflection API.
  dap.getActionMetadataRecord = async (dapPrefix: string) => {
    const dapActions = {} as ActionMetadataRecord;
    // We want to skip traces so we don't get a new action trace
    // every time the DevUI requests the list of actions.
    // This is ok, because the DevUI will show the actions, so
    // not having them in the trace is fine.
    const result = await dap.__cache.getOrFetch({ skipTrace: true });
    for (const [actionType, actions] of Object.entries(result)) {
      const metadataList = actions.map((a) => a.__action);
      for (const metadata of metadataList) {
        if (!metadata.name) {
          throw new GenkitError({
            status: 'INVALID_ARGUMENT',
            message: `Invalid metadata when listing dynamic actions from ${dapPrefix} - name required`,
          });
        }
        const key = `${dapPrefix}:${actionType}/${metadata.name}`;
        dapActions[key] = metadata;
      }
    }
    return dapActions;
  };
}
