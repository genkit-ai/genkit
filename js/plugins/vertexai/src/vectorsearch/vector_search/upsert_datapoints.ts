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

import type { GoogleAuth } from 'google-auth-library';
import type { IIndexDatapoint } from './types.js';

interface UpsertDatapointsParams {
  datapoints: IIndexDatapoint[];
  authClient: GoogleAuth;
  projectId: string;
  location: string;
  indexId: string;
}

/**
 * Upserts datapoints into a specified index.
 *
 * This function sends a request to the Google AI Platform to upsert datapoints
 * into a specified index using the provided parameters.
 *
 * @param {UpsertDatapointsParams} params - The parameters required to upsert datapoints.
 * @param {IIndexDatapoint[]} params.datapoints - The datapoints to be upserted.
 * @param {GoogleAuth} params.authClient - The GoogleAuth client for authorization.
 * @param {string} params.projectId - The ID of the Google Cloud project.
 * @param {string} params.location - The location of the AI Platform index.
 * @param {string} params.indexId - The ID of the index.
 * @returns {Promise<void>} - A promise that resolves when the upsert is complete.
 * @throws {Error} - Throws an error if the upsert fails.
 */
export async function upsertDatapoints(
  params: UpsertDatapointsParams
): Promise<void> {
  const { datapoints, authClient, projectId, location, indexId } = params;
  const accessToken = await authClient.getAccessToken();
  const url = `https://${location}-aiplatform.googleapis.com/v1/projects/${projectId}/locations/${location}/indexes/${indexId}:upsertDatapoints`;

  const requestBody = {
    datapoints: datapoints.map((dp) => {
      const newDp: Record<string, unknown> = {
        datapoint_id: dp.datapointId,
        feature_vector: dp.featureVector,
      };
      if (dp.restricts) {
        newDp.restricts =
          dp.restricts?.map((r) => ({
            namespace: r.namespace,
            allow_list: r.allowList,
            deny_list: r.denyList,
          })) || [];
      }
      if (dp.numericRestricts) {
        newDp.numeric_restricts =
          dp.numericRestricts?.map((nr) => {
            const newNR: Record<string, unknown> = {
              namespace: nr.namespace,
            };
            if (nr.valueInt) {
              newNR.value_int = nr.valueInt;
            }
            if (nr.valueFloat) {
              newNR.value_float = nr.valueFloat;
            }
            if (nr.valueDouble) {
              newNR.value_double = nr.valueDouble;
            }
            return newNR;
          }) || [];
      }
      if (dp.crowdingTag) {
        newDp.crowding_tag = dp.crowdingTag;
      }
      return newDp;
    }),
  };

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${accessToken}`,
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const errMsg = (await response.json()).error?.message || '';
    throw new Error(
      `Error upserting datapoints into index ${indexId}: ${response.statusText}. ${errMsg}`
    );
  }
}
