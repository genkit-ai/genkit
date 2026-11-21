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

import {
  getApp,
  initializeServerApp,
  type FirebaseApp,
  type FirebaseOptions,
  type FirebaseServerApp,
} from 'firebase/app';
import {
  executeMutation,
  executeQuery,
  getDataConnect,
  mutationRef,
  queryRef,
} from 'firebase/data-connect';
import { readFileSync } from 'fs';
import { GenkitError, tool as createTool, type JSONSchema7 } from 'genkit';
import { logger } from 'genkit/logging';
import { genkitPlugin, type GenkitPlugin } from 'genkit/plugin';

export interface DataConnectTool {
  name: string;
  type: 'query' | 'mutation';
  description?: string;
  parameters: JSONSchema7;
}

export interface ToolsConfig {
  connector: string;
  location: string;
  service: string;
  tools: DataConnectTool[];
}

export interface DataConnectToolsOptions {
  /** Provide a name for the plugin. All tools will have this prefix, e.g. `myname/myTool` */
  name: string;
  /** Pass in tool definitions as generated from Data Connect's `llmTools` generator. */
  config?: ToolsConfig;
  /** Path to the file output by the `llmTools` generator. */
  configFile: string;
  /**
   * Your Firebase client SDK config or a FirebaseApp or a FirebaseServerApp. If not provided
   * the plugin will attempt to use `context.firebaseApp` for an in-context app or will fall
   * back the default app.
   */
  firebaseApp?: FirebaseOptions | FirebaseApp;
  /**
   * How to handle errors coming from Data Connect tools:
   *
   * - `return`: (default) return the error to the model to allow it to decide how to proceed
   * - `throw`: throw an error, halting execution
   *
   * You may also supply a custom error handler. Whatever is returned from the handler will
   * be returned to the LLM. If you throw an exception from the handler, execution will be
   * halted with an error.
   */
  onError?:
    | 'return'
    | 'throw'
    | ((tool: DataConnectTool, error: Error) => any | Promise<any>);
}

export function serverAppFromContext(
  context: Record<string, any>,
  config?: FirebaseOptions | FirebaseApp
): FirebaseServerApp | FirebaseApp {
  if ('firebaseApp' in context)
    return context.firebaseApp as FirebaseApp | FirebaseServerApp;
  try {
    if (!config) config = getApp();
  } catch (e) {
    throw new GenkitError({
      status: 'FAILED_PRECONDITION',
      message: `Must either supply a 'firebaseApp' option or have already initialized a default FirebaseApp when calling Data Connect tools.`,
    });
  }

  return initializeServerApp(config, {
    authIdToken: context.auth?.rawToken,
    appCheckToken: context.app?.rawToken,
  });
}

/**
 * dataConnectTools connects Genkit to your Data Connect operations by creating tools from
 * your connector's queries and mutations. This plugin is driven by the generated JSON file
 * from the `llmTools` option. You can generate this file using the `llmTools` option in
 * `connector.yaml`, for example:
 *
 * ```yaml
 * connectorId: tools
 * generate:
 *   llmTools:
 *     outputFile: ../../tools.json
 * ```
 *
 * Once you have the tools file, you can use this function to register the tools as a Genkit
 * plugin:
 *
 * ```ts
 * const app = initializeFirebase({...});
 *
 * const ai = genkit({
 *   plugins: [..., dataConnectTool({
 *     name: 'myTools',
 *     configFile: 'tools.json',
 *     sdk: app,
 *   })]
 * })
 * ```
 *
 * **IMPORTANT:** This plugin relies on the *client SDKs* for Firebase and does not have
 * administrative access to Data Connect. To authenticate this plugin requires a `firebaseApp`
 * instance on `context` with appropriate privileges or the use of the `firebaseContext()`
 * context provider.
 *
 * @param options Configuration options for the plugin.
 * @returns A Genkit plugin.
 */
export function dataConnectTools(
  options: DataConnectToolsOptions
): GenkitPlugin {
  if (!options.config && !options.configFile)
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message:
        'Must supply `config` or `configFile` when initializing a Data Connect tools plugin.',
    });

  if (!options.config) {
    try {
      options.config = JSON.parse(
        readFileSync(options.configFile, 'utf8')
      ) as ToolsConfig;
    } catch (e) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message: `Could not parse Data Connect tools config from ${options.configFile}: ${(e as any).message}`,
      });
    }
  }

  const config = options.config;
  return genkitPlugin({
    name: options.name,
    init: () =>
      config.tools.map((tool) =>
        createTool(
          {
            name: tool.name,
            description: tool.description || '',
            inputJsonSchema: tool.parameters,
          },
          async (input, { context }) => {
            const serverApp = serverAppFromContext(
              context,
              options.firebaseApp
            );
            const dc = getDataConnect(serverApp, {
              connector: config.connector,
              location: config.location,
              service: config.service,
            });

            try {
              if (tool.type === 'query') {
                const { data } = await executeQuery(
                  queryRef(dc, tool.name, input)
                );
                return data;
              }

              const { data } = await executeMutation(
                mutationRef(dc, tool.name, input)
              );
              logger.debug(
                `[dataConnectTools] ${tool.name}(${JSON.stringify(input)}) -> ${JSON.stringify(data)}`
              );
              return data;
            } catch (e) {
              logger.info('[dataConnectTools] error on tool call:', e);
              if (options.onError === 'throw') throw e;
              if (typeof options.onError === 'function')
                return Promise.resolve(options.onError(tool, e as Error));
              if (options.onError === 'return' || !options.onError)
                return { error: (e as any).message };
            }
          }
        )
      ),
  });
}
