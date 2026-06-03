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
//
// SPDX-License-Identifier: Apache-2.0

package bedrock

import (
	"context"
	"flag"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

var (
	testRegion             = flag.String("test-bedrock-region", "", "AWS region for Bedrock live tests (e.g. us-east-1)")
	testModelClaude        = flag.String("test-bedrock-model-claude", "", "Claude model ID (e.g. us.anthropic.claude-haiku-4-5-20251001-v1:0)")
	testModelNova          = flag.String("test-bedrock-model-nova", "", "Nova model ID (e.g. amazon.nova-pro-v1:0)")
	testTitanEmbedder      = flag.String("test-bedrock-titan-embedder", "", "Titan text embedder model ID (e.g. amazon.titan-embed-text-v2:0)")
	testTitanImageEmbedder = flag.String("test-bedrock-titan-image-embedder", "", "Titan image embedder model ID (e.g. amazon.titan-embed-image-v1)")
	testCohereEmbedder     = flag.String("test-bedrock-cohere-embedder", "", "Cohere embedder model ID (e.g. cohere.embed-english-v3)")
	testNovaMMEmbedder     = flag.String("test-bedrock-nova-mm-embedder", "", "Nova multimodal embedder model ID (e.g. amazon.nova-2-multimodal-embeddings-v1:0)")
	testRerankModel        = flag.String("test-bedrock-rerank-model", "", "Cohere rerank model ID (e.g. cohere.rerank-v3-5:0)")
)

const testPNGDataURL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAXklEQVR4nO3PMQ0AMAzAsPInvYLYYVWKESTzjhsd8KsBrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BbQHKU9LC7/CP1AAAAABJRU5ErkJggg=="

// requireLive asserts the live-test prerequisites and skips otherwise.
// Bedrock model access is region-scoped and requires manual opt-in per model
// in the AWS console; the test only validates that the plugin's request shape
// is acceptable to the API, not that any particular model is granted.
func requireLive(t *testing.T) {
	t.Helper()
	if *testRegion == "" {
		t.Skip("bedrock live tests skipped; pass -test-bedrock-region=<region>")
	}
}

// TestBedrockLive_ClaudeSync exercises the synchronous Converse path against
// the configured Claude model.
func TestBedrockLive_ClaudeSync(t *testing.T) {
	requireLive(t)
	if *testModelClaude == "" {
		t.Skip("pass -test-bedrock-model-claude=<model-id> to run")
	}
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&Bedrock{Region: *testRegion}))
	m, err := DefineModel(g, *testModelClaude, nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("Reply with exactly the word OK and nothing else."),
		ai.WithConfig(&Config{MaxTokens: 32}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.ToUpper(resp.Text()), "OK") {
		t.Errorf("response = %q; expected to contain OK", resp.Text())
	}
}

// TestBedrockLive_ClaudeStream exercises ConverseStream — confirms at least
// one chunk is delivered to the callback and the final response is non-empty.
func TestBedrockLive_ClaudeStream(t *testing.T) {
	requireLive(t)
	if *testModelClaude == "" {
		t.Skip("pass -test-bedrock-model-claude=<model-id> to run")
	}
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&Bedrock{Region: *testRegion}))
	m, err := DefineModel(g, *testModelClaude, nil)
	if err != nil {
		t.Fatal(err)
	}
	var chunks int
	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("Count from one to three, one number per line."),
		ai.WithConfig(&Config{MaxTokens: 64}),
		ai.WithStreaming(func(ctx context.Context, c *ai.ModelResponseChunk) error {
			chunks++
			return nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if chunks == 0 {
		t.Error("expected at least one streaming chunk")
	}
	if resp.Text() == "" {
		t.Error("final response text is empty")
	}
}

// TestBedrockLive_ClaudeTool checks a one-turn tool round-trip: Claude calls
// the declared tool, the runtime invokes the tool, then Claude's follow-up
// produces a final answer.
func TestBedrockLive_ClaudeTool(t *testing.T) {
	requireLive(t)
	if *testModelClaude == "" {
		t.Skip("pass -test-bedrock-model-claude=<model-id> to run")
	}
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&Bedrock{Region: *testRegion}))
	m, err := DefineModel(g, *testModelClaude, nil)
	if err != nil {
		t.Fatal(err)
	}

	type weatherIn struct {
		Location string `json:"location"`
	}
	type weatherOut struct {
		TempF float32 `json:"temp_f"`
	}
	tool := genkit.DefineTool(g, "get_weather", "Look up the current temperature in a city.",
		func(ctx *ai.ToolContext, in weatherIn) (weatherOut, error) {
			return weatherOut{TempF: 72}, nil
		})

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithSystem("Use the get_weather tool to answer questions about temperature."),
		ai.WithPrompt("What's the temperature in San Francisco? Reply with one short sentence."),
		ai.WithTools(tool),
		ai.WithConfig(&Config{MaxTokens: 256}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(resp.Text(), "72") {
		t.Errorf("expected response to contain 72, got %q", resp.Text())
	}
}

// TestBedrockLive_NovaSync exercises Nova on the same sync path. Models from
// different families exercise different Converse code branches on the AWS
// side, so a Claude pass alone isn't enough to call the plugin "working".
func TestBedrockLive_NovaSync(t *testing.T) {
	requireLive(t)
	if *testModelNova == "" {
		t.Skip("pass -test-bedrock-model-nova=<model-id> to run")
	}
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&Bedrock{Region: *testRegion}))
	m, err := DefineModel(g, *testModelNova, nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("Reply with exactly the word OK and nothing else."),
		ai.WithConfig(&Config{MaxTokens: 32}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() == "" {
		t.Error("response empty")
	}
}

// TestBedrockLive_TitanEmbedder verifies the InvokeModel-based embedder path.
func TestBedrockLive_TitanEmbedder(t *testing.T) {
	requireLive(t)
	if *testTitanEmbedder == "" {
		t.Skip("pass -test-bedrock-titan-embedder=<model-id> to run")
	}
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&Bedrock{Region: *testRegion}))
	emb, err := DefineEmbedder(g, *testTitanEmbedder, nil)
	if err != nil {
		t.Fatal(err)
	}
	out, err := genkit.Embed(ctx, g, ai.WithEmbedder(emb), ai.WithTextDocs("hello world"))
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Embeddings) != 1 {
		t.Fatalf("got %d embeddings, want 1", len(out.Embeddings))
	}
	if len(out.Embeddings[0].Embedding) == 0 {
		t.Error("embedding has zero dimensions")
	}
}

func TestBedrockLive_TitanImageEmbedder(t *testing.T) {
	requireLive(t)
	if *testTitanImageEmbedder == "" {
		t.Skip("pass -test-bedrock-titan-image-embedder=<model-id> to run")
	}
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&Bedrock{Region: *testRegion}))
	emb, err := DefineEmbedder(g, *testTitanImageEmbedder, nil)
	if err != nil {
		t.Fatal(err)
	}
	out, err := genkit.Embed(ctx, g, ai.WithEmbedder(emb), ai.WithDocs(imageDoc()))
	if err != nil {
		t.Fatal(err)
	}
	assertOneEmbedding(t, out)
}

func TestBedrockLive_CohereEmbedderTextAndImage(t *testing.T) {
	requireLive(t)
	if *testCohereEmbedder == "" {
		t.Skip("pass -test-bedrock-cohere-embedder=<model-id> to run")
	}
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&Bedrock{Region: *testRegion}))
	emb, err := DefineEmbedder(g, *testCohereEmbedder, nil)
	if err != nil {
		t.Fatal(err)
	}

	textOut, err := genkit.Embed(ctx, g, ai.WithEmbedder(emb), ai.WithTextDocs("hello world", "goodbye world"))
	if err != nil {
		t.Fatal(err)
	}
	if len(textOut.Embeddings) != 2 {
		t.Fatalf("text embeddings = %d, want 2", len(textOut.Embeddings))
	}
	for i, e := range textOut.Embeddings {
		if len(e.Embedding) == 0 {
			t.Fatalf("text embedding %d has zero dimensions", i)
		}
	}

	imageOut, err := genkit.Embed(ctx, g, ai.WithEmbedder(emb), ai.WithDocs(imageDoc()))
	if err != nil {
		t.Fatal(err)
	}
	assertOneEmbedding(t, imageOut)
}

func TestBedrockLive_NovaMultimodalEmbedder(t *testing.T) {
	requireLive(t)
	if *testNovaMMEmbedder == "" {
		t.Skip("pass -test-bedrock-nova-mm-embedder=<model-id> to run")
	}
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&Bedrock{Region: *testRegion}))
	emb, err := DefineEmbedder(g, *testNovaMMEmbedder, nil)
	if err != nil {
		t.Fatal(err)
	}
	out, err := genkit.Embed(ctx, g, ai.WithEmbedder(emb),
		ai.WithDocs(ai.DocumentFromText("hello world", nil), imageDoc()))
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Embeddings) != 2 {
		t.Fatalf("embeddings = %d, want 2", len(out.Embeddings))
	}
	for i, e := range out.Embeddings {
		if len(e.Embedding) == 0 {
			t.Fatalf("embedding %d has zero dimensions", i)
		}
	}
}

func TestBedrockLive_Rerank(t *testing.T) {
	requireLive(t)
	if *testRerankModel == "" {
		t.Skip("pass -test-bedrock-rerank-model=<model-id> to run")
	}
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&Bedrock{Region: *testRegion}))
	out, err := Rerank(ctx, g, *testRerankModel, &ai.RerankerRequest{
		Query: ai.DocumentFromText("What is the capital of France?", nil),
		Documents: []*ai.Document{
			ai.DocumentFromText("The capital of France is Paris.", nil),
			ai.DocumentFromText("The tallest mountain is Everest.", nil),
			ai.DocumentFromText("Bananas are yellow.", nil),
		},
		Options: &RerankOptions{TopN: 2},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Documents) != 2 {
		t.Fatalf("ranked docs = %d, want 2", len(out.Documents))
	}
	if got := docText(&ai.Document{Content: out.Documents[0].Content}); !strings.Contains(got, "Paris") {
		t.Fatalf("top reranked doc = %q, want Paris doc", got)
	}
	if out.Documents[0].Metadata == nil || out.Documents[1].Metadata == nil {
		t.Fatal("expected rerank scores in metadata")
	}
	if out.Documents[0].Metadata.Score < out.Documents[1].Metadata.Score {
		t.Fatalf("scores not descending: %f < %f", out.Documents[0].Metadata.Score, out.Documents[1].Metadata.Score)
	}
}

func imageDoc() *ai.Document {
	return &ai.Document{Content: []*ai.Part{ai.NewMediaPart("image/png", testPNGDataURL)}}
}

func assertOneEmbedding(t *testing.T, out *ai.EmbedResponse) {
	t.Helper()
	if len(out.Embeddings) != 1 {
		t.Fatalf("got %d embeddings, want 1", len(out.Embeddings))
	}
	if len(out.Embeddings[0].Embedding) == 0 {
		t.Fatal("embedding has zero dimensions")
	}
}
