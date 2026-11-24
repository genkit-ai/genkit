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

import { z } from 'genkit';
import { SPAN_TYPE_ATTR, appendSpan } from 'genkit/tracing';

/** Explicit user sentiment of response. */
export enum FirebaseUserFeedbackEnum {
  /** The user reacted positively to the response. */
  POSITIVE = 'positive',
  /** The user reacted negatively to the response. */
  NEGATIVE = 'negative',
}

/** Implicit user acceptance of response. */
export enum FirebaseUserAcceptanceEnum {
  /** The user took the desired action. */
  ACCEPTED = 'accepted',
  /** The user did not take the desired action. */
  REJECTED = 'rejected',
}

/** Explicit user feedback on response. */
export const FirebaseUserFeedbackSchema = z.object({
  /** User sentiment of response. */
  value: z.nativeEnum(FirebaseUserFeedbackEnum),
  /** Optional free text feedback to supplement score. */
  text: z.optional(z.string()),
});

/** Implicit user acceptance of response. */
export const FirebaseUserAcceptanceSchema = z.object({
  /** Whether the user took the desired action based on the response. */
  value: z.nativeEnum(FirebaseUserAcceptanceEnum),
});

/** Schema for providing user engagement metadata. One or both of feedback and acceptance should be provided. */
export const FirebaseUserEngagementSchema = z.object({
  /** Flow or feature name. */
  name: z.string(),
  /**
   * The trace ID of the execution for which we've received user engagement data.
   */
  traceId: z.string(),
  /** The root span ID of the execution for which we've received user engagement data. */
  spanId: z.string(),
  /** Explicit user feedback on response. */
  feedback: z.optional(FirebaseUserFeedbackSchema),
  /** Implicit user acceptance of response. */
  acceptance: z.optional(FirebaseUserAcceptanceSchema),
});
export type FirebaseUserEngagement = z.infer<
  typeof FirebaseUserEngagementSchema
>;

/** Associates user engagement metadata with the specified flow execution. */
export async function collectUserEngagement(
  userEngagement: FirebaseUserEngagement
) {
  // Collect user feedback, if provided
  if (userEngagement.feedback?.value) {
    const metadata: Record<string, any> = {
      feedbackValue: userEngagement.feedback.value,
      subtype: 'userFeedback',
    };
    if (userEngagement.feedback.text) {
      metadata['textFeedback'] = userEngagement.feedback.text;
    }

    await appendSpan(
      userEngagement.traceId,
      userEngagement.spanId,
      {
        name: 'user-feedback',
        path: `/{${userEngagement.name}}`,
        metadata: metadata,
      },
      {
        [SPAN_TYPE_ATTR]: 'userEngagement',
      }
    );
  }

  // Collect user acceptance, if provided
  if (userEngagement.acceptance?.value) {
    await appendSpan(
      userEngagement.traceId,
      userEngagement.spanId,
      {
        name: 'user-acceptance',
        path: `/{${userEngagement.name}}`,
        metadata: {
          acceptanceValue: userEngagement.acceptance.value,
          subtype: 'userAcceptance',
        },
      },
      {
        [SPAN_TYPE_ATTR]: 'userEngagement',
      }
    );
  }
}
