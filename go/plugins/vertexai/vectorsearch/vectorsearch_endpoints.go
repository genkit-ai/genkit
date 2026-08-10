// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package vectorsearch

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

// A client is used to perform vector search operations.
type client struct {
	TokenSource oauth2.TokenSource
	AccessToken string
}

// newClient builds a client.
func newClient(ctx context.Context) (*client, error) {
	// Load credentials from the environment or default locations.
	creds, err := google.FindDefaultCredentials(ctx, "https://www.googleapis.com/auth/cloud-platform")
	if err != nil {
		return nil, fmt.Errorf("failed to find default credentials: %v", err)
	}

	// Create a token source from the credentials.
	tokenSource := creds.TokenSource
	token, err := tokenSource.Token()
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve access token: %v", err)
	}

	return &client{
		TokenSource: tokenSource,
		AccessToken: token.AccessToken,
	}, nil
}

// UpsertDatapoints upserts datapoints into a specified index.
func (v *VertexAIVectorSearch) UpsertDatapoints(params UpsertDatapointsParams) error {
	// Construct the URL for the API endpoint.
	url := fmt.Sprintf("https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/indexes/%s:upsertDatapoints",
		params.Location, params.ProjectID, params.Location, params.IndexID)

	// Prepare the request body.
	requestBody := map[string]interface{}{
		"datapoints": params.Datapoints,
	}
	bodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return fmt.Errorf("failed to marshal request body: %v", err)
	}

	// Create the HTTP request.
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(bodyBytes))
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %v", err)
	}

	// Add headers.
	req.Header.Set("Content-Type", "application/json")
	token, err := v.client.TokenSource.Token()
	if err != nil {
		return fmt.Errorf("failed to get token from auth client: %v", err)
	}
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token.AccessToken))

	// Send the request.
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send HTTP request: %v", err)
	}
	defer resp.Body.Close()

	// Handle the response.
	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("error upserting datapoints into index %s: %s. %s",
			params.IndexID, resp.Status, string(respBody))
	}

	return nil
}

// QueryPublicEndpoint queries a public index endpoint to find neighbors for a given feature vector.
func (v *VertexAIVectorSearch) FindNeighbors(params FindNeighborsParams) (*FindNeighborsResponse, error) {
	// Construct the URL for the API endpoint.

	url := fmt.Sprintf("https://%s/v1/projects/%s/locations/%s/indexEndpoints/%s:findNeighbors",
		params.PublicDomainName, params.ProjectNumber, params.Location, params.IndexEndpointID)

	// Prepare the request body.
	requestBody := map[string]interface{}{
		"deployedIndexId": params.DeployedIndexID,
		"queries": []map[string]interface{}{
			{
				"datapoint": map[string]interface{}{
					"featureVector":    params.FeatureVector,
					"restricts":        params.Restricts,
					"numericRestricts": params.NumericRestricts,
				},
				"neighborCount": params.NeighborCount,
			},
		},
	}
	bodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %v", err)
	}

	// Create the HTTP request.
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %v", err)
	}

	// Add headers.
	req.Header.Set("Content-Type", "application/json")
	token, err := v.client.TokenSource.Token()
	if err != nil {
		return nil, fmt.Errorf("failed to get token from auth client: %v", err)
	}
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token.AccessToken))

	// Send the request.
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send HTTP request: %v", err)
	}
	defer resp.Body.Close()

	// Handle the response.
	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("error querying index: %s. %s", resp.Status, string(respBody))
	}

	// Parse the response body.
	var response FindNeighborsResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response body: %v", err)
	}

	return &response, nil
}
