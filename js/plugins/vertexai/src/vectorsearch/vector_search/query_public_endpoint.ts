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

import type {
  FindNeighborsResponse,
  INumericRestriction,
  IRestriction,
} from './types.js';

interface QueryPublicEndpointParams {
  featureVector: number[];
  neighborCount: number;
  accessToken: string;
  projectId: string;
  location: string;
  indexEndpointId: string;
  publicDomainName: string;
  projectNumber: string;
  deployedIndexId: string;
  restricts?: IRestriction[];
  numericRestricts?: INumericRestriction[];
}
/**
 * Queries a public index endpoint to find neighbors for a given feature vector.
 *
 * This function sends a request to a specified public endpoint to find neighbors
 * for a given feature vector using the provided parameters.
 *
 * @param {QueryPublicEndpointParams} params - The parameters required to query the public endpoint.
 * @param {number[]} params.featureVector - The feature vector for which to find neighbors.
 * @param {number} params.neighborCount - The number of neighbors to retrieve.
 * @param {string} params.accessToken - The access token for authorization.
 * @param {string} params.projectId - The ID of the Google Cloud project.
 * @param {string} params.location - The location of the index endpoint.
 * @param {string} params.indexEndpointId - The ID of the index endpoint.
 * @param {string} params.publicDomainName - The domain name of the public endpoint.
 * @param {string} params.projectNumber - The project number.
 * @param {string} params.deployedIndexId - The ID of the deployed index.
 * @returns {Promise<FindNeighborsResponse>} - The response from the public endpoint.
 */
export async function queryPublicEndpoint(
  params: QueryPublicEndpointParams
): Promise<FindNeighborsResponse> {
  const {
    featureVector,
    neighborCount,
    accessToken,
    indexEndpointId,
    publicDomainName,
    projectNumber,
    deployedIndexId,
    location,
    restricts,
    numericRestricts,
  } = params;
  const url = new URL(
    `https://${publicDomainName}/v1/projects/${projectNumber}/locations/${location}/indexEndpoints/${indexEndpointId}:findNeighbors`
  );

  const requestBody = {
    deployed_index_id: deployedIndexId,
    queries: [
      {
        datapoint: {
          datapoint_id: '0',
          feature_vector: featureVector,
          restricts:
            restricts?.map((r) => ({
              namespace: r.namespace,
              allow_list: r.allowList,
              deny_list: r.denyList,
            })) || [],
          numeric_restricts:
            numericRestricts?.map((nr) => {
              const newNR: Record<string, unknown> = {
                namespace: nr.namespace,
              };
              // Exactly one of these should be set in a valid request.
              // If there are more or less, vector search will complain
              // and we can just pass the error on, rather than randomly
              // selecting exactly one of them here (as that would be difficult
              // to debug for the user)
              if (nr.valueInt !== undefined) {
                newNR.value_int = nr.valueInt;
              }
              if (nr.valueFloat !== undefined) {
                newNR.value_float = nr.valueFloat;
              }
              if (nr.valueDouble !== undefined) {
                newNR.value_double = nr.valueDouble;
              }
              newNR.op = nr.op;
              return newNR;
            }) || [],
        },
        neighbor_count: neighborCount,
      },
    ],
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
    throw new Error(`Error querying index: ${response.statusText}. ${errMsg}`);
  }
  return (await response.json()) as FindNeighborsResponse;
}
