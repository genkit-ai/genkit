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

import type { protos, v2 } from '@google-cloud/dlp';
import { generateMiddleware } from 'genkit';
import { credentialsFromEnvironment } from '../auth.js';

// Option 1: Configure redaction options inline.
export interface SdpInfoType {
  name: string;
  version?: string;
}

interface SdpInlineConfigBase {
  /**
   * Which infoTypes to inspect and replace.
   * Default: ['CREDIT_CARD_NUMBER', 'EMAIL_ADDRESS', 'PHONE_NUMBER']
   * All available infoTypes: https://cloud.google.com/sensitive-data-protection/docs/infotypes-reference#descriptions
   */
  infoTypes?: (
    | 'CREDIT_CARD_NUMBER'
    | 'EMAIL_ADDRESS'
    | 'PHONE_NUMBER'
    | 'STREET_ADDRESS'
    | 'US_SOCIAL_SECURITY_NUMBER'
    | 'IP_ADDRESS'
    | 'PASSPORT'
    | 'FINANCIAL_ACCOUNT_NUMBER'
    | (string & {})
    | SdpInfoType
  )[];
}

/**
 * The transformation method.
 */
export type SdpInlineConfig = SdpInlineConfigBase &
  (
    | {
        /** Replaces with the type name (e.g., [EMAIL_ADDRESS]) */
        transformation?: 'INFOTYPE';
      }
    | {
        /** Replaces characters with a symbol (e.g., *****) */
        transformation: 'MASK';
        /**
         * Masking configuration when transformation is 'MASK'. Defaults to '*'
         * Note: The masking string must be a single character.
         */
        maskConfig?: { maskingCharacter: string };
      }
    | {
        /** Replaces with a fixed string (e.g., [REDACTED]) */
        transformation: 'CUSTOM_STRING';
        /** Custom string configuration when transformation is 'CUSTOM_STRING'. Defaults to '[REDACTED]' */
        customConfig?: string;
      }
  );

// Option 2: Create a custom config template in the Google Cloud Console.
// Create in Google Cloud Console > Security > Sensitive Data Protection > Configuration.
// Instructions: https://docs.cloud.google.com/sensitive-data-protection/docs/create-inspection-template
export type SdpTemplateConfig =
  | {
      inspectTemplateName: string;
      deidentifyTemplateName?: string;
    }
  | {
      inspectTemplateName?: string;
      deidentifyTemplateName: string;
    };

export interface SdpOptionsBase {
  projectId?: string; // (Optional) Explicitly set the Google Cloud Project ID
  credentials?: any; // (Optional) Explicitly set the Google Cloud credentials
}

async function createDlpClient(
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

  let envAuth;
  if (!options.projectId || !options.credentials) {
    envAuth = await credentialsFromEnvironment();
  }

  const projectId = options.projectId || envAuth?.projectId;
  const credentials = options.credentials || envAuth?.credentials;

  return new dlpModule.v2.DlpServiceClient({
    credentials: credentials as any as NonNullable<
      ConstructorParameters<typeof v2.DlpServiceClient>[0]
    >['credentials'],
    projectId: projectId,
  });
}

export type SdpOptions = SdpOptionsBase &
  (
    | { templates: SdpTemplateConfig; inline?: SdpInlineConfig }
    | { inline: SdpInlineConfig; templates?: SdpTemplateConfig }
  );

// 1. Inspect config
function buildInspectConfig(
  options: SdpOptions
): protos.google.privacy.dlp.v2.IInspectConfig {
  const defaultInfoTypes = [
    'CREDIT_CARD_NUMBER',
    'EMAIL_ADDRESS',
    'PHONE_NUMBER',
  ];

  if (!('inline' in options) || !options.inline) {
    return { infoTypes: defaultInfoTypes.map((name) => ({ name })) };
  }

  const infoTypes =
    options.inline.infoTypes && options.inline.infoTypes.length > 0
      ? options.inline.infoTypes
      : defaultInfoTypes;

  return {
    infoTypes: infoTypes.map((infoType) => {
      if (typeof infoType === 'string') return { name: infoType };
      return { name: infoType.name, version: infoType.version };
    }),
  };
}

// 2. De-identify config
function buildDeidentifyConfig(
  options: SdpOptions
): protos.google.privacy.dlp.v2.IDeidentifyConfig {
  const inlineConfig =
    'inline' in options && options.inline ? options.inline : {};

  let primitiveTransformation: protos.google.privacy.dlp.v2.IPrimitiveTransformation =
    {
      replaceWithInfoTypeConfig: {}, // Default behavior
    };

  if (inlineConfig.transformation === 'CUSTOM_STRING') {
    primitiveTransformation = {
      replaceConfig: {
        newValue: { stringValue: inlineConfig.customConfig ?? '[REDACTED]' },
      },
    };
  } else if (inlineConfig.transformation === 'MASK') {
    primitiveTransformation = {
      characterMaskConfig: inlineConfig.maskConfig || {
        maskingCharacter: '*',
      },
    };
  }

  return {
    infoTypeTransformations: {
      transformations: [{ primitiveTransformation }],
    },
  };
}

async function sanitizeInput(
  dlp: v2.DlpServiceClient,
  text: string,
  options: SdpOptions,
  projectId: string
): Promise<string> {
  const request: protos.google.privacy.dlp.v2.IDeidentifyContentRequest = {
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
          clientPromise = (async () => {
            const envAuth = await credentialsFromEnvironment();
            const projectId = opts.projectId || envAuth.projectId;
            const credentials = opts.credentials || envAuth.credentials;

            if (!projectId) {
              throw new Error(
                'Project ID is required for Sensitive Data Protection. Please set the projectId option or configure it in your environment.'
              );
            }

            const client = await createDlpClient({
              ...opts,
              projectId,
              credentials,
            });

            return { client, projectId };
          })().catch((err) => {
            clientPromise = null;
            throw err;
          });
        }

        const { client, projectId } = await clientPromise;

        // Intercept input
        if (envelope.request?.messages) {
          const redactionPromises = envelope.request.messages
            // ignore empty messages
            .filter((message) => !!message.content)
            // extract all message content into a single array
            // @ts-ignore - flatMap creates a flat array of content parts
            .flatMap((message) => message.content!)
            // ignore multimedia content and content that has been cleaned already
            .filter((part) => part.text && !part.metadata?.isCleaned)
            // de-identify message content
            .map(async (part): Promise<void> => {
              part.text = await sanitizeInput(
                client,
                part.text!,
                opts,
                projectId
              );
              part.metadata = { ...part.metadata, isCleaned: true };
            });

          // Wait for all network calls to finish in parallel
          await Promise.all(redactionPromises);
        }

        const res = await next(envelope, ctx);

        // Intercept output
        if (res.message?.content) {
          const outputRedactionPromises = res.message.content
            // ignore multimedia content and content that has been cleaned already
            .filter((part) => part.text && !part.metadata?.isCleaned)
            // de-identify message content
            .map(async (part): Promise<void> => {
              part.text = await sanitizeInput(
                client,
                part.text!,
                opts,
                projectId
              );
              part.metadata = { ...part.metadata, isCleaned: true };
            });

          await Promise.all(outputRedactionPromises);
        }

        return res;
      },
    };
  }
);
