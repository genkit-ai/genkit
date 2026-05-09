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

package ai

import (
	"context"
	"os"
	"testing"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/registry"
	"github.com/goccy/go-yaml"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

type specSuite struct {
	Tests []testCase `yaml:"tests"`
}

type testCase struct {
	Name           string                  `yaml:"name"`
	Input          *GenerateActionOptions  `yaml:"input"`
	StreamChunks   [][]*ModelResponseChunk `yaml:"streamChunks,omitempty"`
	ModelResponses []*ModelResponse        `yaml:"modelResponses"`
	ExpectResponse *ModelResponse          `yaml:"expectResponse,omitempty"`
	Stream         bool                    `yaml:"stream,omitempty"`
	ExpectChunks   []*ModelResponseChunk   `yaml:"expectChunks,omitempty"`
}

type programmableModel struct {
	r           api.Registry
	handleResp  func(ctx context.Context, req *ModelRequest, cb func(context.Context, *ModelResponseChunk) error) (*ModelResponse, error)
	lastRequest *ModelRequest
}

func (pm *programmableModel) Name() string {
	return "programmableModel"
}

func (pm *programmableModel) Generate(ctx context.Context, r api.Registry, req *ModelRequest, toolCfg *ToolConfig, cb func(context.Context, *ModelResponseChunk) error) (*ModelResponse, error) {
	// Make a copy of the request to modify for testing purposes
	if req != nil && req.Tools != nil {
		for _, tool := range req.Tools {
			if tool.Name == "testTool" {
				// Set the schema fields directly
				tool.InputSchema = map[string]any{"$schema": "http://json-schema.org/draft-07/schema#"}
				tool.OutputSchema = map[string]any{"$schema": "http://json-schema.org/draft-07/schema#"}
			}
		}
	}
	pm.lastRequest = req
	return pm.handleResp(ctx, req, cb)
}

func defineProgrammableModel(r api.Registry) *programmableModel {
	pm := &programmableModel{r: r}
	supports := &ModelSupports{
		Tools:     true,
		Multiturn: true,
	}
	DefineModel(r, "programmableModel", &ModelOptions{Supports: supports}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		return pm.Generate(ctx, r, req, &ToolConfig{MaxTurns: 5}, cb)
	})
	return pm
}

func TestGenerateAction(t *testing.T) {
	data, err := os.ReadFile("../../tests/specs/generate.yaml")
	if err != nil {
		t.Fatalf("failed to read spec file: %v", err)
	}

	var suite specSuite
	if err := yaml.Unmarshal(data, &suite); err != nil {
		t.Fatalf("failed to parse spec file: %v", err)
	}

	for _, tc := range suite.Tests {
		t.Run(tc.Name, func(t *testing.T) {
			ctx := context.Background()

			r := registry.New()
			ConfigureFormats(r)

			pm := defineProgrammableModel(r)

			DefineTool(r, "testTool", "description",
				func(ctx *ToolContext, input any) (any, error) {
					return "tool called", nil
				})

			if len(tc.ModelResponses) > 0 || len(tc.StreamChunks) > 0 {
				reqCounter := 0
				pm.handleResp = func(ctx context.Context, req *ModelRequest, cb func(context.Context, *ModelResponseChunk) error) (*ModelResponse, error) {
					if len(tc.StreamChunks) > 0 && cb != nil {
						for _, chunk := range tc.StreamChunks[reqCounter] {
							if err := cb(ctx, chunk); err != nil {
								return nil, err
							}
						}
					}
					resp := tc.ModelResponses[reqCounter]
					resp.Request = req
					resp.Custom = map[string]any{}
					resp.Request.Output = &ModelOutputConfig{}
					resp.Usage = &GenerationUsage{}
					reqCounter++
					return resp, nil
				}
			}

			genAction := DefineGenerateAction(ctx, r)

			if tc.Stream {
				chunks := []*ModelResponseChunk{}
				streamCb := func(ctx context.Context, chunk *ModelResponseChunk) error {
					chunks = append(chunks, chunk)
					return nil
				}

				resp, err := genAction.Run(ctx, tc.Input, streamCb)
				if err != nil {
					t.Fatalf("action failed: %v", err)
				}

				if diff := cmp.Diff(tc.ExpectChunks, chunks, cmp.Options{
					cmpopts.IgnoreFields(ModelResponseChunk{}, "formatHandler"),
				}); diff != "" {
					t.Errorf("chunks mismatch (-want +got):\n%s", diff)
				}

				if diff := cmp.Diff(tc.ExpectResponse, resp, cmp.Options{
					cmpopts.EquateEmpty(),
					cmpopts.IgnoreFields(ModelResponse{}, "LatencyMs", "formatHandler"),
					cmpopts.IgnoreFields(GenerationUsage{}, "InputCharacters", "OutputCharacters"),
					cmpopts.IgnoreFields(ToolDefinition{}, "Metadata"),
				}); diff != "" {
					t.Errorf("response mismatch (-want +got):\n%s", diff)
				}
			} else {
				resp, err := genAction.Run(ctx, tc.Input, nil)
				if err != nil {
					t.Fatalf("action failed: %v", err)
				}

				if diff := cmp.Diff(tc.ExpectResponse, resp, cmp.Options{
					cmpopts.EquateEmpty(),
					cmpopts.IgnoreFields(ModelResponse{}, "LatencyMs", "formatHandler"),
					cmpopts.IgnoreFields(GenerationUsage{}, "InputCharacters", "OutputCharacters"),
					cmpopts.IgnoreFields(ToolDefinition{}, "Metadata"),
				}); diff != "" {
					t.Errorf("response mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}
