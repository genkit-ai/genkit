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
	"encoding/json"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/vertexai/vectorsearch"
	"google.golang.org/genai"
)

func menu(ctx context.Context, g *genkit.Genkit, retriever ai.Retriever, model ai.Model, vectorsearchParams *VectorsearchConfig) error {
	ragDataMenuPrompt := genkit.DefinePrompt(g, "ragDataMenu",
		ai.WithPrompt(`
You are acting as Walt, a helpful AI assistant here at the restaurant.
You can answer questions about the food on the menu or any other questions
customers have about food in general.

Here are some items that are on today's menu that are relevant to
helping you answer the customer's question:
{{#each menuData~}}
- {{this.title}} \${{this.price}}
{{this.description}}
{{~/each}}

Answer this customer's question:
{{question}}?
`),
		ai.WithModel(model),
		ai.WithInputType(dataMenuQuestionInput{}),
		ai.WithOutputFormat(ai.OutputFormatText),
		ai.WithConfig(&genai.GenerateContentConfig{
			Temperature: genai.Ptr[float32](0.3),
		}),
	)

	type flowOutput struct {
		Rows int `json:"rows"`
	}

	genkit.DefineFlow(g, "indexMenuItems",
		func(ctx context.Context, input []*menuItem) (*flowOutput, error) {
			var docs []*ai.Document
			for _, m := range input {
				s := fmt.Sprintf("%s %g \n %s", m.Title, m.Price, m.Description)
				metadata := map[string]any{
					"menuItem": m,
				}
				docs = append(docs, ai.DocumentFromText(s, metadata))
			}

			// Index the menu items.
			if err := vectorsearch.Index(ctx, g, vectorsearch.IndexParams{
				IndexID:         vectorsearchParams.IndexID,
				Embedder:        vectorsearchParams.Embedder,
				EmbedderOptions: nil,
				Docs:            docs,
				ProjectID:       vectorsearchParams.ProjectID,
				Location:        vectorsearchParams.Location,
			}, vectorsearchParams.DocumentIndexer); err != nil {
				return nil, err
			}

			ret := &flowOutput{
				Rows: len(input),
			}
			return ret, nil
		},
	)

	genkit.DefineFlow(g, "ragMenuQuestion",
		func(ctx context.Context, input *menuQuestionInput) (*answerOutput, error) {
			resp, err := retriever.Retrieve(ctx, &ai.RetrieverRequest{
				Query: ai.DocumentFromText(input.Question, nil),
				Options: &vectorsearch.RetrieveParams{
					Embedder:          vectorsearchParams.Embedder,
					NeighborCount:     vectorsearchParams.NeighborsCount,
					IndexEndpointID:   vectorsearchParams.IndexEndpointID,
					DeployedIndexID:   vectorsearchParams.DeployedIndexID,
					PublicDomainName:  vectorsearchParams.PublicDomainName,
					ProjectNumber:     vectorsearchParams.ProjectNumber,
					DocumentRetriever: vectorsearchParams.DocumentRetriever,
				}})
			if err != nil {
				return nil, err
			}

			var menuItems []*menuItem
			for _, doc := range resp.Documents {
				// 1. Get the interface{} value associated with "menuItem"
				menuItemInterface, ok := doc.Metadata["menuItem"]
				if !ok {
					fmt.Printf("Error: 'menuItem' key not found in document metadata: %+v\n", doc.Metadata)
					continue // Skip to the next document
				}

				// 2. Marshal the interface{} (which is a map[string]interface{}) into JSON bytes
				jsonData, err := json.Marshal(menuItemInterface)
				if err != nil {
					fmt.Printf("Error marshaling menuItem data to JSON: %v\n", err)
					continue // Skip to the next document
				}

				// 3. Unmarshal the JSON bytes into the menuItem struct
				var mi menuItem
				err = json.Unmarshal(jsonData, &mi)
				if err != nil {
					fmt.Printf("Error unmarshaling JSON into menuItem: %v\n", err)
					continue // Skip to the next document
				}

				menuItems = append(menuItems, &mi)

			}
			questionInput := &dataMenuQuestionInput{
				MenuData: menuItems,
				Question: input.Question,
			}

			presp, err := ragDataMenuPrompt.Execute(ctx, ai.WithInput(questionInput))
			if err != nil {
				return nil, err
			}

			ret := &answerOutput{
				Answer: presp.Message.Content[0].Text,
			}
			return ret, nil
		},
	)

	return nil
}
