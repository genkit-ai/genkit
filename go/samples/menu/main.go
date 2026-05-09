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

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/localvec"
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

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.VertexAI{}))

	model := googlegenai.VertexAIModel(g, "gemini-2.5-flash")
	embedder := googlegenai.VertexAIEmbedder(g, "text-embedding-004")

	if err := setup01(g, model); err != nil {
		log.Fatal(err)
	}
	if err := setup02(g, model); err != nil {
		log.Fatal(err)
	}
	if err := setup03(g, model); err != nil {
		log.Fatal(err)
	}

	if err := localvec.Init(); err != nil {
		log.Fatal(err)
	}

	retOpts := &ai.RetrieverOptions{
		ConfigSchema: core.InferSchemaMap(localvec.RetrieverOptions{}),
		Label:        "go-menu_items",
		Supports: &ai.RetrieverSupports{
			Media: false,
		},
	}
	docStore, retriever, err := localvec.DefineRetriever(g, "go-menu_items", localvec.Config{
		Embedder: embedder,
	}, retOpts)
	if err != nil {
		log.Fatal(err)
	}
	if err := setup04(ctx, g, docStore, retriever, model); err != nil {
		log.Fatal(err)
	}

	if err := setup05(g, model); err != nil {
		log.Fatal(err)
	}

	<-ctx.Done()
}
