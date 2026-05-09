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
//
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"log"

	"cloud.google.com/go/firestore"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/vertexai/vectorsearch"
)

// menuItem is the data model for an item on the menu.
type menuItem struct {
	Title       string  `json:"title" jsonschema_description:"The name of the menu item"`
	Description string  `json:"description" jsonschema_description:"Details including ingredients and preparation"`
	Price       float64 `json:"price" jsonschema_description:"Price in dollars"`
}

// menuQuestionInput is a question about the menu.
type menuQuestionInput struct {
	Question string `json:"question"`
}

// answerOutput is an answer to a question.
type answerOutput struct {
	Answer string `json:"answer"`
}

// dataMenuQuestionInput is a question about the menu,
// where the menu is provided in the JSON data.
type dataMenuQuestionInput struct {
	MenuData []*menuItem `json:"menuData"`
	Question string      `json:"question"`
}

// textMenuQuestionInput is for a question about the menu,
// where the menu is provided as unstructured text.
type textMenuQuestionInput struct {
	MenuText string `json:"menuText"`
	Question string `json:"question"`
}

type VectorsearchConfig struct {
	ProjectID         string      `json:"projectId"`
	Location          string      `json:"location"`
	IndexID           string      `json:"indexId"`
	IndexEndpointID   string      `json:"indexEndpointId"`
	DeployedIndexID   string      `json:"deployedIndexId"`
	ProjectNumber     string      `json:"projectNumber"`
	PublicDomainName  string      `json:"publicDomainName"`
	Embedder          ai.Embedder `json:"embedder"`
	NeighborsCount    int         `json:"neighborsCount,omitempty"`
	DocumentIndexer   vectorsearch.DocumentIndexer
	DocumentRetriever vectorsearch.DocumentRetriever
}

func main() {
	ctx := context.Background()
	vectorsearchPlugin := &vectorsearch.VertexAIVectorSearch{
		ProjectID: "${GOOGLE_CLOUD_PROJECT_ID}",       // Replace with your Google Cloud project ID
		Location:  "${GOOGLE_CLOUD_PROJECT_LOCATION}", // Replace with your desired location
	}
	g := genkit.Init(ctx,
		genkit.WithPlugins(&googlegenai.VertexAI{
			ProjectID: vectorsearchPlugin.ProjectID,
			Location:  vectorsearchPlugin.Location,
		}, vectorsearchPlugin))

	model := googlegenai.VertexAIModel(g, "gemini-2.5-flash")

	databaseId := "${FIRESTORE_DATABASE_ID}"         // Replace with your Firestore database ID
	collectionName := "${FIRESTORE_COLLECTION_NAME}" // Replace with your Firestore collection name
	firestoreClient, err := firestore.NewClientWithDatabase(ctx, vectorsearchPlugin.ProjectID, databaseId)
	documentIndexer := vectorsearch.GetFirestoreDocumentIndexer(firestoreClient, collectionName)
	documentRetriever := vectorsearch.GetFirestoreDocumentRetriever(firestoreClient, collectionName)

	// Define Vectorsearch parameters.
	vectorsearchParams := &VectorsearchConfig{
		ProjectID:         vectorsearchPlugin.ProjectID,
		Location:          vectorsearchPlugin.Location,
		IndexID:           "${VECTOR_SEARCH_INDEX_ID}",                           // Replace with your index ID
		IndexEndpointID:   "${VECTOR_SEARCH_INDEX_ENDPOINT_ID}",                  // Replace with your index endpoint ID
		DeployedIndexID:   "${VECTOR_SEARCH_DEPLOYED_INDEX_ID}",                  // Replace with your deployed index ID
		ProjectNumber:     "${GOOGLE_CLOUD_PROJECT_NUMBER}",                      // Replace with your Google Cloud project number
		PublicDomainName:  "${VECTOR_SEARCH_PUBLIC_DOMAIN_NAME}",                 // Replace with your public domain name
		Embedder:          googlegenai.VertexAIEmbedder(g, "text-embedding-004"), // Replace with your desired embedder
		NeighborsCount:    10,                                                    // Number of neighbors to retrieve
		DocumentIndexer:   documentIndexer,
		DocumentRetriever: documentRetriever,
	}

	// Define the retriever for vector search.
	retriever, err := vectorsearch.DefineRetriever(ctx, g, vectorsearch.Config{
		IndexID: vectorsearchParams.IndexID, // Replace with your index ID
	}, nil)
	if err != nil {
		log.Fatal(err)
	}
	if err := menu(ctx, g, retriever, model, vectorsearchParams); err != nil {
		log.Fatal(err)
	}

	<-ctx.Done()
}
