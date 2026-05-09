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

import { ModelArmorClient, protos } from '@google-cloud/modelarmor';
import { GenkitError } from 'genkit';
import {
  GenerateRequest,
  GenerateResponseData,
  MessageData,
  ModelMiddleware,
  Part,
} from 'genkit/model';
import { runInNewSpan } from 'genkit/tracing';

export interface ModelArmorOptions {
  templateName: string;
  client?: ModelArmorClient;
  /**
   * Options for the Model Armor client (e.g. apiEndpoint).
   */
  clientOptions?: ConstructorParameters<typeof ModelArmorClient>[0];
  /**
   * What to sanitize. Defaults to 'all'.
   */
  protectionTarget?: 'all' | 'userPrompt' | 'modelResponse';
  /**
   * Whether to block on SDP match even if the content was successfully de-identified.
   * Defaults to false (lenient).
   */
  strictSdpEnforcement?: boolean;
  /**
   * List of filters to enforce. If not specified, all filters are enforced.
   * Possible values: 'rai', 'pi_and_jailbreak', 'malicious_uris', 'csam', 'sdp'.
   */
  filters?: (
    | 'rai'
    | 'pi_and_jailbreak'
    | 'malicious_uris'
    | 'csam'
    | 'sdp'
    | (string & {})
  )[];
  /**
   * Whether to apply the de-identification results to the content.
   * - If true, the default logic (replace text, preserve structure) is used.
   * - If false, no changes are applied.
   * - If a function, it is called with the messages and SDP result, and should return the new messages.
   *
   * Defaults to false.
   */
  applyDeidentificationResults?:
    | boolean
    | ((data: {
        messages: MessageData[];
        sdpResult: protos.google.cloud.modelarmor.v1.ISdpFilterResult;
      }) => MessageData[] | undefined);
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
  options: ModelArmorOptions
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
  options: ModelArmorOptions,
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
  options: ModelArmorOptions
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
      await runInNewSpan(
        { metadata: { name: 'sanitizeUserPrompt' } },
        async (meta) => {
          meta.input = {
            name: options.templateName,
            userPromptData: {
              text: promptText,
            },
          };
          const [response] = await client.sanitizeUserPrompt({
            name: options.templateName,
            userPromptData: {
              text: promptText,
            },
          });
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
  options: ModelArmorOptions
) {
  const usingMessageProp = !!response.message;
  const candidates = response.message
    ? [{ index: 0, message: response.message, finishReason: 'stop' }]
    : response.candidates || [];

  for (const candidate of candidates) {
    const modelText = extractText(candidate.message.content);

    if (modelText) {
      await runInNewSpan(
        { metadata: { name: 'sanitizeModelResponse' } },
        async (meta) => {
          meta.input = {
            name: options.templateName,
            modelResponseData: {
              text: modelText,
            },
          };
          const [apiResponse] = await client.sanitizeModelResponse({
            name: options.templateName,
            modelResponseData: {
              text: modelText,
            },
          });
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

/**
 * Model Middleware that uses Google Cloud Model Armor to sanitize user prompts and model responses.
 */
export function modelArmor(options: ModelArmorOptions): ModelMiddleware {
  const client = options.client || new ModelArmorClient(options.clientOptions);
  const protectionTarget = options.protectionTarget ?? 'all';
  const protectUserPrompt =
    protectionTarget === 'all' || protectionTarget === 'userPrompt';
  const protectModelResponse =
    protectionTarget === 'all' || protectionTarget === 'modelResponse';

  return async (req, next) => {
    // 1. Sanitize User Prompt
    if (protectUserPrompt) {
      await sanitizeUserPrompt(req, client, options);
    }

    // 2. Call Model
    const response = await next(req);

    // 3. Sanitize Model Response
    if (protectModelResponse) {
      await sanitizeModelResponse(response, client, options);
    }

    return response;
  };
}
