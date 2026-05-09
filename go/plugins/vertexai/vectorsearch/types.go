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
	"context"

	"github.com/firebase/genkit/go/ai"
	"golang.org/x/oauth2/google"
)

// IndexParams represents the parameters required for indexing documents into a vector search index.
type IndexParams struct {
	Docs            []*ai.Document // The documents to be indexed.
	Embedder        ai.Embedder    // The AI embedder used to convert documents into vector embeddings.
	EmbedderOptions any            // Optional settings specific to the chosen embedder.
	ProjectID       string         // The Google Cloud Project ID where the index is located.
	Location        string         // The geographical region of the index (e.g., "us-central1").
	IndexID         string         // The unique ID of the vector search index.
}

// DocumentRetriever defines the interface (function type) for retrieving original documents
// given a set of search results (neighbors).
type DocumentRetriever func(ctx context.Context, neighbors []Neighbor, options any) ([]*ai.Document, error)

// DocumentIndexer defines the interface (function type) for indexing documents
// into a vector search index.
type DocumentIndexer func(ctx context.Context, docs []*ai.Document) ([]string, error)

// RetrieveParams represents the parameters required for retrieving documents from a vector search index
// based on a query document.
type RetrieveParams struct {
	Content           *ai.Document        // The query document to find similar documents for.
	Embedder          ai.Embedder         // The embedder used to vectorize the query content.
	EmbedderOptions   any                 // Optional settings for the query embedder.
	AuthClient        *google.Credentials // Authentication credentials for Google Cloud services.
	ProjectNumber     string              // The numeric ID of the Google Cloud project.
	Location          string              // The geographical region of the index endpoint.
	IndexEndpointID   string              // The ID of the specific index endpoint to query.
	PublicDomainName  string              // The public domain name of the deployed index within the endpoint.
	DeployedIndexID   string              // The ID of the deployed index within the endpoint.
	NeighborCount     int                 // The number of nearest neighbors to retrieve.
	Restricts         []Restrict          // Categorical restrictions to apply to the search.
	NumericRestricts  []NumericRestrict   // Numerical restrictions to apply to the search.
	DocumentRetriever DocumentRetriever   `json:"-"` // A function to retrieve original documents; excluded from JSON serialization.
}

// IndexDatapoint represents a single data point to be indexed in the vector search.
type IndexDatapoint struct {
	DatapointID      string            `json:"datapoint_id"`                // A unique identifier for this data point.
	FeatureVector    []float32         `json:"feature_vector"`              // The numerical vector embedding of the data point.
	Restricts        []Restrict        `json:"restricts,omitempty"`         // Optional categorical restrictions for the data point.
	NumericRestricts []NumericRestrict `json:"numeric_restricts,omitempty"` // Optional numerical restrictions for the data point.
	CrowdingTag      string            `json:"crowding_tag,omitempty"`      // An optional tag for crowding control in search results.
}

// UpsertDatapointsParams represents the parameters required to add or update (upsert)
// multiple data points in a vector search index.
type UpsertDatapointsParams struct {
	Datapoints []IndexDatapoint // The slice of data points to upsert.
	ProjectID  string           // The Google Cloud Project ID.
	Location   string           // The geographical region of the index.
	IndexID    string           // The unique ID of the vector search index.
}

// Config represents the configuration settings for the Vertex AI vector search plugin.
type Config struct {
	IndexID string // The unique ID of the vector search index to use.
}

// Restrict represents a categorical filter (allow or deny list) for a specific namespace.
type Restrict struct {
	Namespace string   `json:"namespace"`           // The category or attribute name for the restriction.
	AllowList []string `json:"allowList,omitempty"` // List of allowed values; if empty, all are allowed unless denied.
	DenyList  []string `json:"denyList,omitempty"`  // List of denied values; if empty, none are denied.
}

// NumericRestrict represents a numerical filter for a specific namespace, allowing for range or exact matches.
type NumericRestrict struct {
	Namespace   string  `json:"namespace"`             // The numerical attribute name for the restriction.
	ValueFloat  float32 `json:"valueFloat,omitempty"`  // Floating-point value for comparison.
	ValueInt    int64   `json:"valueInt,omitempty"`    // Integer value for comparison.
	ValueDouble float64 `json:"valueDouble,omitempty"` // Double-precision floating-point value for comparison.
	Op          string  `json:"op,omitempty"`          // The comparison operator (e.g., "=", ">", "<=").
}

// FindNeighborsResponse represents the structured response from the Vertex AI FindNeighbors API.
type FindNeighborsResponse struct {
	NearestNeighbors []struct {
		Neighbors []Neighbor `json:"neighbors"` // A list of the nearest neighbors found for each query.
	} `json:"nearestNeighbors"`
}

// Neighbor represents a single nearest neighbor result, including the datapoint and its distance.
type Neighbor struct {
	Datapoint Datapoint `json:"datapoint"` // The actual data point found.
	Distance  float64   `json:"distance"`  // The calculated distance (similarity score) to the query.
}

// Datapoint represents the structure of a single data point returned in the FindNeighbors response.
type Datapoint struct {
	DatapointId      string            `json:"datapointId"`                // The unique ID of the data point.
	FeatureVector    []float32         `json:"featureVector"`              // The numerical vector embedding of the data point.
	Restricts        []Restrict        `json:"restricts,omitempty"`        // Categorical restrictions associated with this data point.
	NumericRestricts []NumericRestrict `json:"numericRestricts,omitempty"` // Numerical restrictions associated with this data point.
	CrowdingTag      CrowdingAttribute `json:"crowdingTag,omitempty"`      // Optional crowding attribute for diversity control.
}

// CrowdingAttribute represents an optional attribute used for controlling result diversity (crowding).
type CrowdingAttribute struct {
	CrowdingAttribute string `json:"crowdingAttribute,omitempty"` // The value of the crowding tag.
}

// FindNeighborsParams represents the parameters required to query a public endpoint
// for finding nearest neighbors in a vector search index.
type FindNeighborsParams struct {
	FeatureVector    []float32           // The vector embedding of the query content.
	NeighborCount    int                 // The desired number of nearest neighbors to retrieve.
	AuthClient       *google.Credentials // Authentication credentials for the API call.
	ProjectNumber    string              // The numeric ID of the Google Cloud project.
	Location         string              // The geographical region of the index endpoint.
	IndexEndpointID  string              // The ID of the specific index endpoint to query.
	PublicDomainName string              // The public domain name of the Vertex AI service.
	DeployedIndexID  string              // The ID of the deployed index within the endpoint.
	Restricts        []Restrict          // Categorical restrictions to apply to the search query.
	NumericRestricts []NumericRestrict   // Numerical restrictions to apply to the search query.
}
