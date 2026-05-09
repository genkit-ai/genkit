// Copyright 2024 Google LLC
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

package main

import (
	"context"
	"log"
	"strings"

	// Import Genkit and the Google AI plugin

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

func main() {
	ctx := context.Background()
	ctx = core.WithActionContext(ctx, core.ActionContext{
		"greeting": "hello",
		"name":     "John Doe",
	})

	g := genkit.Init(ctx,
		genkit.WithDefaultModel("googleai/gemini-2.5-flash"),
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
	)

	genkit.DefinePartial(g, "header", "Welcome {{@name}}!")
	genkit.DefineHelper(g, "uppercase", func(s string) string {
		return strings.ToUpper(s)
	})

	p := genkit.DefinePrompt(g, "test", ai.WithPrompt(`{{> header}} {{uppercase @greeting}}`))

	result, err := p.Execute(ctx)
	if err != nil {
		log.Fatal(err)
	}
	text := result.Text()
	log.Printf("Response: %s", text)

}
