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

import type { v2 } from '@google-cloud/dlp';
import { generateMiddleware } from 'genkit';
import { credentialsFromEnvironment } from '../auth.js';

// Option 1: Configure redaction options inline.
export interface SdpInlineConfig {
  /**
   * 1. Select which infoTypes to inspect and replace.
   * Default: ['CREDIT_CARD_NUMBER', 'EMAIL_ADDRESS', 'PHONE_NUMBER']
   * All available infoTypes: https://cloud.google.com/sensitive-data-protection/docs/infotypes-reference#descriptions
   */
  infoTypes?: string[];

  /**
   * 2. Select the transformation method.
   * - 'INFOTYPE' (default): Replaces with the type name (e.g., [EMAIL_ADDRESS])
   * - 'MASK': Replaces characters with a symbol (e.g., *****)
   * - 'CUSTOM_STRING': Replaces with a fixed string (e.g., [REDACTED])
   */
  transformation?: 'INFOTYPE' | 'MASK' | 'CUSTOM_STRING';

  // Masking configuration when transformation is 'MASK'. Defaults to '*' }
  maskConfig?: { maskingCharacter: string };

  // Custom string configuration when transformation is 'CUSTOM_STRING'. Defaults to '[REDACTED]' */
  customConfig?: string;
}

// Option 2: Create a custom config template in the Google Cloud Console.
// Create in Google Cloud Console > Security > Sensitive Data Protection > Configuration.
// Instructions: https://docs.cloud.google.com/sensitive-data-protection/docs/create-inspection-template
export interface SdpTemplateConfig {
  inspectTemplateName: string;
  deidentifyTemplateName: string;
}

export interface SdpOptions {
  templates?: SdpTemplateConfig;
  inline?: SdpInlineConfig;

  projectId?: string; // (Optional) Explicitly set the Google Cloud Project ID
}

export async function createDlpClient(
  options: SdpOptions
): Promise<v2.DlpServiceClient> {
  let dlpModule;
  try {
    dlpModule = await import('@google-cloud/dlp');
  } catch (e) {
    throw new Error(
      'Please install the @google-cloud/dlp package to use the SDP middleware.'
    );
  }

  const envAuth = await credentialsFromEnvironment();
  const projectId = options.projectId || envAuth.projectId;

  return new dlpModule.v2.DlpServiceClient({
    credentials: envAuth.credentials as any,
    projectId: projectId,
  });
}

// 1. Inspect config
export function buildInspectConfig(options: SdpOptions) {
  const defaultInfoTypes = [
    'CREDIT_CARD_NUMBER',
    'EMAIL_ADDRESS',
    'PHONE_NUMBER',
  ];
  const inlineConfig = options.inline || {};
  const infoTypes =
    inlineConfig.infoTypes && inlineConfig.infoTypes.length > 0
      ? inlineConfig.infoTypes
      : defaultInfoTypes;

  return {
    infoTypes: infoTypes.map((name) => ({ name })),
  };
}

// 2. De-identify config
export function buildDeidentifyConfig(options: SdpOptions): any {
  const inlineConfig = options.inline || {};
  let primitiveTransformation: any = {
    replaceWithInfoTypeConfig: {}, // Default behavior
  };

  if (inlineConfig.transformation === 'CUSTOM_STRING') {
    primitiveTransformation = {
      replaceConfig: {
        newValue: { stringValue: inlineConfig.customConfig || '[REDACTED]' },
      },
    };
  } else if (inlineConfig.transformation === 'MASK') {
    primitiveTransformation = {
      characterMaskConfig: inlineConfig.maskConfig || { maskingCharacter: '*' },
    };
  }

  return {
    infoTypeTransformations: {
      transformations: [{ primitiveTransformation }],
    },
  };
}

export async function sanitizeInput(
  dlp: v2.DlpServiceClient,
  text: string,
  options: SdpOptions,
  projectId: string
) {
  const request: any = {
    parent: `projects/${projectId}/locations/global`,
    item: { value: text },
  };

  if (options.templates) {
    request.inspectTemplateName = options.templates.inspectTemplateName;
    request.deidentifyTemplateName = options.templates.deidentifyTemplateName;
  } else {
    request.inspectConfig = buildInspectConfig(options);
    request.deidentifyConfig = buildDeidentifyConfig(options);
  }

  const [response] = await dlp.deidentifyContent(request);
  return response.item?.value || text;
}

export const sensitiveDataProtection = generateMiddleware<SdpOptions>(
  { name: 'sensitiveDataProtection' },
  ({ config, pluginConfig }) => {
    const options = { ...pluginConfig, ...config } as SdpOptions;
    let clientPromise: Promise<{
      client: v2.DlpServiceClient;
      projectId: string;
    }> | null = null;
    return {
      generate: async (envelope, ctx, next) => {
        const opts = options || {};
        if (!clientPromise) {
          clientPromise = Promise.all([
            createDlpClient(opts),
            credentialsFromEnvironment(),
          ]).then(([client, envAuth]) => {
            return {
              client,
              projectId: opts.projectId || envAuth.projectId || '',
            };
          });
        }

        const { client, projectId } = await clientPromise;

        // Intercept input
        const redactionPromises: Promise<void>[] = [];

        if (envelope.request?.messages) {
          for (const message of envelope.request.messages) {
            if (message.content) {
              for (const part of message.content) {
                if (part.text && !part.metadata?.isCleaned) {
                  const promise = sanitizeInput(
                    client,
                    part.text,
                    opts,
                    projectId
                  ).then((sanitizedText) => {
                    part.text = sanitizedText;
                    part.metadata = { ...part.metadata, isCleaned: true }; // Tag the part as cleaned
                  });
                  redactionPromises.push(promise);
                }
              }
            }
          }
        }

        // Wait for all network calls to finish in parallel
        await Promise.all(redactionPromises);

        const res = await next(envelope, ctx);
        // Intercept output
        return res;
      },
    };
  }
);
