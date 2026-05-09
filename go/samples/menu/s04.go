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
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/localvec"
	"google.golang.org/genai"
)

func setup04(ctx context.Context, g *genkit.Genkit, docStore *localvec.DocStore, retriever ai.Retriever, model ai.Model) error {
	ragDataMenuPrompt := genkit.DefinePrompt(g, "s04_ragDataMenu",
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

	genkit.DefineFlow(g, "s04_indexMenuItems",
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
			if err := localvec.Index(ctx, docs, docStore); err != nil {
				return nil, err
			}

			ret := &flowOutput{
				Rows: len(input),
			}
			return ret, nil
		},
	)

	genkit.DefineFlow(g, "s04_ragMenuQuestion",
		func(ctx context.Context, input *menuQuestionInput) (*answerOutput, error) {
			resp, err := genkit.Retrieve(ctx, g,
				ai.WithRetriever(retriever),
				ai.WithTextDocs(input.Question),
				ai.WithConfig(&localvec.RetrieverOptions{
					K: 3,
				}))
			if err != nil {
				return nil, err
			}

			var menuItems []*menuItem
			for _, doc := range resp.Documents {
				menuItems = append(menuItems, doc.Metadata["menuItem"].(*menuItem))
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
