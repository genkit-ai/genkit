/**
 * @license
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
 * Model Armor middleware for Genkit.
 *
 * @module model-armor
 */

import { ModelArmorClient, protos } from '@google-cloud/modelarmor';
import {
  GenkitError,
  generateMiddleware,
  z,
  type GenerateMiddleware,
} from 'genkit';
import type {
  GenerateRequest,
  GenerateResponseData,
  MessageData,
  Part,
} from 'genkit/model';
import { runInNewSpan } from 'genkit/tracing';

export const ModelArmorOptionsSchema = z.object({
  /**
   * Resource name of the Model Armor template, e.g.
   * `projects/.../locations/.../templates/...`.
   */
  templateName: z
    .string()
    .describe('Resource name of the Model Armor template.'),
  /** What to sanitize. Defaults to 'all'. */
  protectionTarget: z
    .enum(['all', 'userPrompt', 'modelResponse'])
    .optional()
    .describe("What to sanitize. Defaults to 'all'."),
  /**
   * Whether to block on SDP match even if the content was successfully
   * de-identified. Defaults to false (lenient).
   */
  strictSdpEnforcement: z
    .boolean()
    .optional()
    .describe(
      'Block on SDP match even if the content was de-identified. Defaults to false.'
    ),
  /**
   * Filters to enforce, e.g. 'rai', 'pi_and_jailbreak', 'malicious_uris',
   * 'csam', 'sdp'. If not specified, all filters are enforced.
   */
  filters: z
    .array(z.string())
    .optional()
    .describe(
      "Filters to enforce: 'rai', 'pi_and_jailbreak', 'malicious_uris', 'csam', 'sdp'. Defaults to all."
    ),
  /**
   * Whether to apply de-identification (SDP) results to the content, replacing
   * the text and keeping other parts. A custom function set in the plugin
   * options is used instead unless this is `false`. Defaults to false.
   */
  applyDeidentificationResults: z
    .boolean()
    .optional()
    .describe('Apply de-identification results to the content.'),
});

/** Per-call options of the {@link modelArmor} middleware. */
export type ModelArmorOptions = z.infer<typeof ModelArmorOptionsSchema>;

/** Custom way to apply de-identification results to the messages. */
export type ApplyDeidentificationResultsFn = (data: {
  messages: MessageData[];
  sdpResult: protos.google.cloud.modelarmor.v1.ISdpFilterResult;
}) => MessageData[] | undefined;

/**
 * Instance-level options of the {@link modelArmor} middleware, set once with
 * `modelArmor.plugin(...)`. Kept out of the per-call config: they are not
 * serializable and would otherwise end up in traces.
 */
export interface ModelArmorPluginOptions {
  /** The Model Armor client to use. Defaults to one built from `clientOptions`. */
  client?: ModelArmorClient;
  /** Options for the default Model Armor client (e.g. `apiEndpoint`). */
  clientOptions?: ConstructorParameters<typeof ModelArmorClient>[0];
  /**
   * Applies de-identification results to the messages and returns the new
   * messages (or `undefined` to leave them unchanged).
   */
  applyDeidentificationResults?: ApplyDeidentificationResultsFn;
}

/** Options the sanitization helpers work with. */
interface ResolvedModelArmorOptions {
  templateName: string;
  strictSdpEnforcement?: boolean;
  filters?: string[];
  applyDeidentificationResults: boolean | ApplyDeidentificationResultsFn;
}

function extractText(parts: Part[]): string {
  return parts.map((p) => p.text || '').join('');
}

/**
 * If SDP (Sensitive Data Protection) filter returns sanitized data,
 * we swap out the data with sanitized data.
 */
function applySdp(
  messages: MessageData[],
  targetIndex: number,
  result: protos.google.cloud.modelarmor.v1.ISanitizationResult,
  options: ResolvedModelArmorOptions
): { sdpApplied: boolean; messages: MessageData[] } {
  const sdpFilterResult = result.filterResults?.['sdp']?.sdpFilterResult;

  if (!sdpFilterResult) {
    return { sdpApplied: false, messages };
  }

  // If user provided applyDeidentificationResults, we use it to apply
  // the deidentification results.
  if (typeof options.applyDeidentificationResults === 'function') {
    const newMessages = options.applyDeidentificationResults({
      messages,
      sdpResult: sdpFilterResult,
    });
    if (!newMessages) {
      return { sdpApplied: false, messages };
    }
    const sdpApplied = !!sdpFilterResult.deidentifyResult?.data?.text;
    return { sdpApplied, messages: newMessages };
  }

  // if applyDeidentificationResults is set to true, we use the default/basic
  // approach to apply the results.
  if (options.applyDeidentificationResults === true) {
    const deidentifyResult = sdpFilterResult.deidentifyResult;
    if (deidentifyResult && deidentifyResult.data?.text) {
      const targetMessage = messages[targetIndex];
      const nonTextParts = targetMessage.content.filter((p) => !p.text);
      const newContent = [
        ...nonTextParts,
        { text: deidentifyResult.data.text },
      ];
      const newMessages = [...messages];
      newMessages[targetIndex] = { ...targetMessage, content: newContent };
      return {
        sdpApplied: true,
        messages: newMessages,
      };
    }
  }

  return { sdpApplied: false, messages };
}

function shouldBlock(
  result: protos.google.cloud.modelarmor.v1.ISanitizationResult,
  options: ResolvedModelArmorOptions,
  sdpApplied: boolean
): boolean {
  if (result.filterMatchState !== 'MATCH_FOUND') {
    return false;
  }
  // Check if we should block.
  // If strict SDP enforcement is enabled and SDP was applied, we must block.
  if (options.strictSdpEnforcement && sdpApplied) {
    return true;
  }
  // Otherwise, check if any active filter matched.
  if (result.filterResults) {
    for (const [key, filterResult] of Object.entries(result.filterResults)) {
      if (options.filters && !options.filters.includes(key)) continue;
      if (key === 'sdp' && sdpApplied) continue;

      // Look for matchState in the nested object
      // e.g. filterResult.raiFilterResult.matchState
      const nestedResult = Object.values(filterResult)[0];
      if (nestedResult?.matchState === 'MATCH_FOUND') {
        return true;
      }
    }
  }
  return false;
}

async function sanitizeUserPrompt(
  req: GenerateRequest,
  client: ModelArmorClient,
  options: ResolvedModelArmorOptions
) {
  let targetMessageIndex = -1;
  // Find the last user message to sanitize
  for (let i = req.messages.length - 1; i >= 0; i--) {
    if (req.messages[i].role === 'user') {
      targetMessageIndex = i;
      break;
    }
  }

  if (targetMessageIndex !== -1) {
    const userMessage = req.messages[targetMessageIndex];
    const promptText = extractText(userMessage.content);

    if (promptText) {
      const request = {
        name: options.templateName,
        userPromptData: {
          text: promptText,
        },
      };
      // Input is set at creation so the realtime pending export carries it.
      await runInNewSpan(
        { metadata: { name: 'sanitizeUserPrompt', input: request } },
        async (meta) => {
          const [response] = await client.sanitizeUserPrompt(request);
          meta.output = response;

          if (response.sanitizationResult) {
            const result = response.sanitizationResult;
            const { sdpApplied, messages: modifiedMessages } = applySdp(
              req.messages,
              targetMessageIndex,
              result,
              options
            );

            if (
              sdpApplied ||
              typeof options.applyDeidentificationResults === 'function'
            ) {
              req.messages = modifiedMessages;
            }

            if (shouldBlock(result, options, sdpApplied)) {
              throw new GenkitError({
                status: 'PERMISSION_DENIED',
                message: 'Model Armor blocked user prompt.',
                detail: result,
              });
            }
          }
        }
      );
    }
  }
}

async function sanitizeModelResponse(
  response: GenerateResponseData,
  client: ModelArmorClient,
  options: ResolvedModelArmorOptions
) {
  const usingMessageProp = !!response.message;
  const candidates = response.message
    ? [{ index: 0, message: response.message, finishReason: 'stop' }]
    : response.candidates || [];

  for (const candidate of candidates) {
    const modelText = extractText(candidate.message.content);

    if (modelText) {
      const request = {
        name: options.templateName,
        modelResponseData: {
          text: modelText,
        },
      };
      // Input is set at creation so the realtime pending export carries it.
      await runInNewSpan(
        { metadata: { name: 'sanitizeModelResponse', input: request } },
        async (meta) => {
          const [apiResponse] = await client.sanitizeModelResponse(request);
          meta.output = apiResponse;

          if (apiResponse.sanitizationResult) {
            const result = apiResponse.sanitizationResult;
            const { sdpApplied, messages: modifiedMessages } = applySdp(
              [candidate.message],
              0,
              result,
              options
            );

            if (
              sdpApplied ||
              typeof options.applyDeidentificationResults === 'function'
            ) {
              candidate.message = modifiedMessages[0];
            }

            if (shouldBlock(result, options, sdpApplied)) {
              throw new GenkitError({
                status: 'PERMISSION_DENIED',
                message: 'Model Armor blocked model response.',
                detail: result,
              });
            }
          }
        }
      );
    }
  }

  if (usingMessageProp && candidates.length > 0) {
    response.message = candidates[0].message;
  }
}

// One client per plugin instance, and a shared default for when the
// middleware is used without `.plugin(...)`, so a client is not built on every
// generate call.
const clientsByPluginOptions = new WeakMap<
  ModelArmorPluginOptions,
  ModelArmorClient
>();
let defaultClient: ModelArmorClient | undefined;

function clientFor(
  pluginOptions: ModelArmorPluginOptions | undefined
): ModelArmorClient {
  if (!pluginOptions) {
    defaultClient ??= new ModelArmorClient();
    return defaultClient;
  }
  if (pluginOptions.client) return pluginOptions.client;
  let client = clientsByPluginOptions.get(pluginOptions);
  if (!client) {
    client = new ModelArmorClient(pluginOptions.clientOptions);
    clientsByPluginOptions.set(pluginOptions, client);
  }
  return client;
}

/**
 * Generate middleware that uses Google Cloud Model Armor to sanitize user
 * prompts and model responses.
 *
 * ```ts
 * const ai = genkit({
 *   plugins: [
 *     googleAI(),
 *     // Optional: only needed to configure the client.
 *     modelArmor.plugin({
 *       clientOptions: { apiEndpoint: 'modelarmor.us-central1.rep.googleapis.com' },
 *     }),
 *   ],
 * });
 *
 * const { text } = await ai.generate({
 *   prompt,
 *   use: [modelArmor({ templateName: 'projects/p/locations/l/templates/t' })],
 * });
 * ```
 */
export const modelArmor: GenerateMiddleware<
  typeof ModelArmorOptionsSchema,
  ModelArmorPluginOptions | undefined
> = generateMiddleware(
  {
    name: 'modelArmor',
    description:
      'Sanitizes user prompts and model responses with Google Cloud Model Armor.',
    configSchema: ModelArmorOptionsSchema,
  },
  ({ config, pluginConfig }) => {
    if (!config?.templateName) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message:
          "modelArmor requires a 'templateName', e.g. 'use: [modelArmor({ templateName })]'.",
      });
    }
    const client = clientFor(pluginConfig);
    const options: ResolvedModelArmorOptions = {
      templateName: config.templateName,
      strictSdpEnforcement: config.strictSdpEnforcement,
      filters: config.filters,
      applyDeidentificationResults:
        config.applyDeidentificationResults === false
          ? false
          : (pluginConfig?.applyDeidentificationResults ??
            config.applyDeidentificationResults ??
            false),
    };
    const protectionTarget = config.protectionTarget ?? 'all';
    const protectUserPrompt =
      protectionTarget === 'all' || protectionTarget === 'userPrompt';
    const protectModelResponse =
      protectionTarget === 'all' || protectionTarget === 'modelResponse';

    return {
      model: async (req, ctx, next) => {
        if (protectUserPrompt) {
          await sanitizeUserPrompt(req, client, options);
        }
        const response = await next(req, ctx);
        if (protectModelResponse) {
          await sanitizeModelResponse(response, client, options);
        }
        return response;
      },
    };
  }
);
