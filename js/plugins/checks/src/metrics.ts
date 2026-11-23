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

/**
 * Currently supported Checks AI Safety policies.
 */
export enum ChecksEvaluationMetricType {
  // The model facilitates, promotes or enables access to harmful goods,
  // services, and activities.
  DANGEROUS_CONTENT = 'DANGEROUS_CONTENT',
  // The model reveals an individual’s personal information and data.
  PII_SOLICITING_RECITING = 'PII_SOLICITING_RECITING',
  // The model generates content that is malicious, intimidating, bullying, or
  // abusive towards another individual.
  HARASSMENT = 'HARASSMENT',
  // The model generates content that is sexually explicit in nature.
  SEXUALLY_EXPLICIT = 'SEXUALLY_EXPLICIT',
  // The model promotes violence, hatred, discrimination on the basis of race,
  // religion, etc.
  HATE_SPEECH = 'HATE_SPEECH',
  // The model facilitates harm by providing health advice or guidance.
  MEDICAL_INFO = 'MEDICAL_INFO',
  // The model generates content that contains gratuitous, realistic
  // descriptions of violence or gore.
  VIOLENCE_AND_GORE = 'VIOLENCE_AND_GORE',
  // The model generates content that contains vulgar, profane, or offensive
  // language.
  OBSCENITY_AND_PROFANITY = 'OBSCENITY_AND_PROFANITY',
}

/**
 * Checks evaluation metric config. Use `threshold` to override the default violation threshold.
 * The value of `metricSpec` will be included in the request to the API. See the API documentation
 */
export type ChecksEvaluationMetricConfig = {
  type: ChecksEvaluationMetricType;
  threshold?: number;
};

export type ChecksEvaluationMetric =
  ChecksEvaluationMetricType | ChecksEvaluationMetricConfig;

export function isConfig(
  config: ChecksEvaluationMetric
): config is ChecksEvaluationMetricConfig {
  return (config as ChecksEvaluationMetricConfig).type !== undefined;
}
