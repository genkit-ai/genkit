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

import {
  action,
  generateMiddleware,
  genkitPluginV2,
  z,
  type GenkitPluginV2,
} from 'genkit';

const developmentModeMiddlware = generateMiddleware(
  {
    name: 'developmentMode',
    description: 'Enables thinking for Gemini models in dev mode.',
  },
  () => ({
    generate: async (envelope, ctx, next) => {
      if (
        process.env.GENKIT_ENV === 'dev' &&
        envelope.request.model?.includes('gemini')
      ) {
        envelope.request.config = {
          ...envelope.request.config,
          thinkingConfig: {
            includeThoughts: true,
            thinkingBudget: 1024,
          },
        };
      }
      return next(envelope, ctx);
    },
  })
);

const traceDecorator = action(
  {
    actionType: 'custom',
    name: 'trace-decorator',
    inputSchema: z.any(), // JSON encoded span data
    outputSchema: z.object({
      badges: z.array(
        z.object({
          label: z.string(),
          icon: z.string(),
          theme: z.string(),
          tooltip: z.string().optional(),
        })
      ),
    }),
  },
  async () => {
    return {
      badges: [
        {
          label: 'hello-world',
          icon: 'hand_gesture',
          theme: 'success',
          tooltip: 'This is a sample badge from dev-ui-extensions plugin',
        },
      ],
    };
  }
);

/**
 * Plugin that provides extensions for the Genkit Dev UI.
 */
export const devUiExtensions: GenkitPluginV2 = genkitPluginV2({
  name: 'dev-ui-extensions',
  async init() {
    return [traceDecorator];
  },
  middleware() {
    return [developmentModeMiddlware];
  },
  async devUiHooks() {
    return [
      {
        slot: 'trace-decorator',
        actionId: '/custom/dev-ui-extensions/trace-decorator',
      },
    ];
  },
});

export default devUiExtensions;
