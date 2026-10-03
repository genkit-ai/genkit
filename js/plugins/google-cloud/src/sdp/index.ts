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
        /** Masking configuration when transformation is 'MASK'. Defaults to '*' }
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

export const sensitiveDataProtection = generateMiddleware<SdpOptions>(
  { name: 'sensitiveDataProtection' },
  ({ config, pluginConfig }) => {
    return {
      generate: async (envelope, ctx, next) => {
        // Intercept input
        const res = await next(envelope, ctx);
        // Intercept output
        return res;
      },
    };
  }
);
