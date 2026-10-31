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
import {
  generateMiddleware,
  z,
  type GenerateMiddleware,
  type Part,
} from 'genkit';
import { credentialsFromEnvironment } from '../auth.js';

// Option 1: Configure redaction options inline.
export const SdpInfoTypeSchema = z.object({
  name: z.string().describe('The name of the infoType.'),
  version: z.string().optional().describe('The version of the infoType.'),
});

export const SdpInlineConfigSchema = z.object({
  /**
   * Which infoTypes to inspect and replace.
   * Default: ['CREDIT_CARD_NUMBER', 'EMAIL_ADDRESS', 'PHONE_NUMBER']
   * All available infoTypes: https://cloud.google.com/sensitive-data-protection/docs/infotypes-reference#descriptions
   */
  infoTypes: z
    .array(z.union([z.string(), SdpInfoTypeSchema]))
    .optional()
    .describe(
      'Which infoTypes to inspect and replace. Default: [CREDIT_CARD_NUMBER, EMAIL_ADDRESS, PHONE_NUMBER]'
    ),
  /**
   * The transformation method.
   */
  transformation: z
    .enum(['INFOTYPE', 'MASK', 'CUSTOM_STRING'])
    .optional()
    .describe('The transformation method: INFOTYPE, MASK, or CUSTOM_STRING'),
  /**
   * Masking configuration when transformation is 'MASK'. Defaults to '*'
   * Note: The masking string must be a single character.
   */
  maskConfig: z
    .object({
      maskingCharacter: z.string(),
    })
    .optional()
    .describe(
      "Masking configuration when transformation is 'MASK'. Defaults to '*'"
    ),
  /**
   * Custom string configuration when transformation is 'CUSTOM_STRING'. Defaults to '[REDACTED]'
   */
  customConfig: z
    .string()
    .optional()
    .describe(
      "Custom string configuration when transformation is 'CUSTOM_STRING'. Defaults to '[REDACTED]'"
    ),
});

// Option 2: Create a custom config template in the Google Cloud Console.
// Create in Google Cloud Console > Security > Sensitive Data Protection > Configuration.
// Instructions: https://docs.cloud.google.com/sensitive-data-protection/docs/create-inspection-template
export const SdpTemplateConfigSchema = z.union([
  z.object({
    inspectTemplateName: z
      .string()
      .describe('Resource name of the inspect template'),
    deidentifyTemplateName: z
      .string()
      .optional()
      .describe('Resource name of the deidentify template'),
  }),
  z.object({
    inspectTemplateName: z
      .string()
      .optional()
      .describe('Resource name of the inspect template'),
    deidentifyTemplateName: z
      .string()
      .describe('Resource name of the deidentify template'),
  }),
]);

export const SdpOptionsBaseSchema = z.object({
  projectId: z
    .string()
    .optional()
    .describe('(Optional) Explicitly set the Google Cloud Project ID'),
  credentials: z
    .any()
    .optional()
    .describe('(Optional) Explicitly set the Google Cloud credentials'),
});

export const SdpOptionsSchema = z.intersection(
  SdpOptionsBaseSchema,
  z.union([
    z.object({
      templates: SdpTemplateConfigSchema,
      inline: SdpInlineConfigSchema.optional(),
    }),
    z.object({
      inline: SdpInlineConfigSchema,
      templates: SdpTemplateConfigSchema.optional(),
    }),
    z.object({
      inline: SdpInlineConfigSchema.optional(),
      templates: SdpTemplateConfigSchema.optional(),
    }),
  ])
);

export type SdpInfoType = z.infer<typeof SdpInfoTypeSchema>;
export type SdpInlineConfig = z.infer<typeof SdpInlineConfigSchema>;
export type SdpTemplateConfig = z.infer<typeof SdpTemplateConfigSchema>;
export type SdpOptionsBase = z.infer<typeof SdpOptionsBaseSchema>;
export type SdpOptions = z.infer<typeof SdpOptionsSchema>;

async function createDlpClient(
  options: SdpOptionsBase
): Promise<v2.DlpServiceClient> {
  let dlpModule;
  try {
    dlpModule = await import('@google-cloud/dlp');
  } catch (e) {
    throw new Error(
      'Please install the @google-cloud/dlp package to use the SDP middleware.'
    );
  }

  return new dlpModule.v2.DlpServiceClient({
    credentials: options.credentials as any,
    projectId: options.projectId,
  });
}

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
  return response.item?.value ?? text;
}

/**
 * **Warning:** In streaming mode, chunks are streamed from the model before redaction runs and will not be redacted.
 */
export const sensitiveDataProtection: GenerateMiddleware<
  typeof SdpOptionsSchema
> = generateMiddleware(
  {
    name: 'sensitiveDataProtection',
    description:
      'Intercepts prompt inputs and model outputs to redact sensitive data using Google Cloud Sensitive Data Protection (DLP). Note: In streaming mode, chunks emitted in real time are not redacted.',
    configSchema: SdpOptionsSchema,
  },
  ({ config, pluginConfig }) => {
    const options = {
      ...(pluginConfig as object | undefined),
      ...config,
    } as SdpOptions;
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
            .flatMap((message): Part[] => message.content!)
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
