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

package main

import (
	"context"
	"log"
	"math"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/server"
)

type StoryCharacter struct {
	Name       string `json:"name"`
	Age        int    `json:"age"`
	Hometown   string `json:"hometown"`
	Profession string `json:"profession"`
	Gablorken  int    `json:"gablorken"`
}

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx,
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
		genkit.WithDefaultModel("googleai/gemini-2.5-flash"),
	)

	var callback func(context.Context, *ai.ModelResponseChunk) error

	gablorkenTool := genkit.DefineTool(g, "gablorken", "use when need to calculate a gablorken",
		func(ctx *ai.ToolContext, input struct {
			Value float64
			Over  float64
		},
		) (float64, error) {
			return math.Pow(input.Value, input.Over), nil
		},
	)

	defaultPrompt := genkit.DefinePrompt(g, "defaultInstructions",
		ai.WithPrompt("Generate a children's book story character about someone named {{name}} and generate the gablorken of 2 over 3."),
		ai.WithTools(gablorkenTool),
		ai.WithOutputType([]StoryCharacter{}),
		ai.WithOutputFormat(ai.OutputFormatJSONL),
	)

	customPrompt := genkit.DefinePrompt(g, "customInstructions",
		ai.WithPrompt("Generate a children's book story character about someone named {{name}}."),
		ai.WithOutputInstructions("The output should be JSON and match the schema of the following object: "+
			"{name: string, age: number, homeTown: string, profession: string}"),
	)

	genkit.DefineStreamingFlow(g, "defaultInstructionsFlow", func(ctx context.Context, _ any, cb func(context.Context, string) error) ([]*StoryCharacter, error) {
		if cb != nil {
			callback = func(ctx context.Context, c *ai.ModelResponseChunk) error {
				return cb(ctx, c.Text())
			}
		}
		resp, err := defaultPrompt.Execute(ctx,
			ai.WithInput(StoryCharacter{Name: "Willy the Pig"}),
			ai.WithStreaming(callback),
		)
		if err != nil {
			return nil, err
		}

		var defaultCharacter []*StoryCharacter
		if err := resp.Output(&defaultCharacter); err != nil {
			return nil, err
		}

		resp, err = customPrompt.Execute(ctx, ai.WithInput(StoryCharacter{Name: "Markie the Doberman"}))
		if err != nil {
			return nil, err
		}

		var customCharacter []*StoryCharacter
		if err := resp.Output(&customCharacter); err != nil {
			return nil, err
		}

		return append(defaultCharacter, customCharacter...), nil
	})

	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
