// Copyright 2026 Google LLC
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

// dev-ui-qa is a consolidated QA testapp for driving the Dev UI against the
// Go SDK. It registers every action type in one process so the full surface
// is visible at once. Cases are grouped by the sections of the
// "Dev UI x Go SDK Audit" tab of the plugin gap-analysis doc.
//
// To run:
//
//	genkit start -- go run .
package main

import (
	"context"
	"log"
	"net/http"
	"os"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/anthropic"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/server"
)

func main() {
	ctx := context.Background()

	// Plugins init only when creds are present: both plugins panic at Init on
	// missing auth (ANT-52, GGA-48), and a partial surface is more useful for
	// QA than no surface. The panic itself is a lifecycle case; see
	// lifecycle.go.
	var plugins []api.Plugin
	if os.Getenv("GEMINI_API_KEY") != "" || os.Getenv("GOOGLE_API_KEY") != "" {
		plugins = append(plugins, &googlegenai.GoogleAI{})
	} else {
		log.Println("dev-ui-qa: GEMINI_API_KEY not set, skipping googleai")
	}
	if os.Getenv("GOOGLE_CLOUD_PROJECT") != "" && (os.Getenv("GOOGLE_CLOUD_LOCATION") != "" || os.Getenv("GOOGLE_CLOUD_REGION") != "") {
		plugins = append(plugins, &googlegenai.VertexAI{})
	} else {
		log.Println("dev-ui-qa: GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION/REGION not set, skipping vertexai")
	}
	if os.Getenv("ANTHROPIC_API_KEY") != "" || os.Getenv("ANTHROPIC_AUTH_TOKEN") != "" {
		plugins = append(plugins, &anthropic.Anthropic{})
	} else {
		log.Println("dev-ui-qa: ANTHROPIC_API_KEY not set, skipping anthropic")
	}

	g := genkit.Init(ctx, genkit.WithPlugins(plugins...))

	genkit.DefineFlow(g, "smoke", func(ctx context.Context, input string) (string, error) {
		return "ok: " + input, nil
	})

	registerModelCases(g)
	registerFlowAndToolCases(g)
	registerEmbedderCases(g)
	registerLifecycleCases(g)

	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	if err := server.Start(ctx, "127.0.0.1:8080", mux); err != nil {
		log.Fatal(err)
	}
}
