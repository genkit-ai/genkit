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
	"errors"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/vertexai/modelgarden"
)

func main() {
	ctx := context.Background()

	// Llama MaaS is served from us-central1. Override Location if your project
	// has Llama enabled in a different region.
	g := genkit.Init(ctx, genkit.WithPlugins(
		&modelgarden.Llama{Location: "us-central1"},
	))

	// Define a flow that uses Llama 3.3 70B to describe a topic.
	genkit.DefineFlow(g, "llamaFlow", func(ctx context.Context, input string) (string, error) {
		m := modelgarden.LlamaModel(g, "meta/llama-3.3-70b-instruct-maas")
		if m == nil {
			return "", errors.New("llamaFlow: failed to find model")
		}

		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithPrompt("In one short sentence, describe %s", input))
		if err != nil {
			return "", err
		}
		return resp.Text(), nil
	})

	<-ctx.Done()
}
