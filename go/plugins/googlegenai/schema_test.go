// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
)

func TestToGeminiSchema(t *testing.T) {
	tests := []struct {
		name         string
		genkitSchema map[string]any
		want         *genai.Schema
		wantErr      bool
	}{
		{
			name: "string type",
			genkitSchema: map[string]any{
				"type": "string",
			},
			want: &genai.Schema{
				Type: genai.TypeString,
			},
		},
		{
			name: "array type [string, null]",
			genkitSchema: map[string]any{
				"type": []any{"string", "null"},
			},
			want: &genai.Schema{
				Type:     genai.TypeString,
				Nullable: genai.Ptr(true),
			},
		},
		{
			name: "array type [null, integer]",
			genkitSchema: map[string]any{
				"type": []any{"null", "integer"},
			},
			want: &genai.Schema{
				Type:     genai.TypeInteger,
				Nullable: genai.Ptr(true),
			},
		},
		{
			name: "array type [null]",
			genkitSchema: map[string]any{
				"type": []any{"null"},
			},
			want: &genai.Schema{
				Nullable: genai.Ptr(true),
			},
		},
		{
			name: "array type [array, null] with items",
			genkitSchema: map[string]any{
				"type": []any{"array", "null"},
				"items": map[string]any{
					"type": "number",
				},
			},
			want: &genai.Schema{
				Type:     genai.TypeArray,
				Nullable: genai.Ptr(true),
				Items: &genai.Schema{
					Type: genai.TypeNumber,
				},
			},
		},
		{
			name: "object with array-typed property (MCP reverse_list style)",
			genkitSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"numbers": map[string]any{
						"type": []any{"array", "null"},
						"items": map[string]any{
							"type": "number",
						},
						"description": "List of numbers to reverse",
					},
				},
				"required": []any{"numbers"},
			},
			want: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"numbers": {
						Type:     genai.TypeArray,
						Nullable: genai.Ptr(true),
						Items: &genai.Schema{
							Type: genai.TypeNumber,
						},
						Description: "List of numbers to reverse",
					},
				},
				Required: []string{"numbers"},
			},
		},
		{
			name: "anyOf with []any (as decoded by json.Unmarshal)",
			genkitSchema: map[string]any{
				"anyOf": []any{
					map[string]any{"type": "string"},
					map[string]any{"type": "null"},
				},
				"title":       "Domain",
				"description": "A domain",
			},
			want: &genai.Schema{
				Type:        genai.TypeString,
				Title:       "Domain",
				Description: "A domain",
			},
		},
		{
			name: "anyOf with []map[string]any",
			genkitSchema: map[string]any{
				"anyOf": []map[string]any{
					{"type": "string"},
					{"type": "null"},
				},
				"title":       "Domain",
				"description": "A domain",
			},
			want: &genai.Schema{
				Type:        genai.TypeString,
				Title:       "Domain",
				Description: "A domain",
			},
		},
		{
			name: "anyOf with subschema using array type",
			genkitSchema: map[string]any{
				"anyOf": []any{
					map[string]any{"type": []any{"string", "null"}},
				},
				"title": "MaybeString",
			},
			want: &genai.Schema{
				Type:     genai.TypeString,
				Nullable: genai.Ptr(true),
				Title:    "MaybeString",
			},
		},
		{
			name: "type as []string",
			genkitSchema: map[string]any{
				"type": []string{"string", "null"},
			},
			want: &genai.Schema{
				Type:     genai.TypeString,
				Nullable: genai.Ptr(true),
			},
		},
		{
			name: "unsupported type string",
			genkitSchema: map[string]any{
				"type": "bogus",
			},
			wantErr: true,
		},
		{
			name: "type is wrong shape (int)",
			genkitSchema: map[string]any{
				"type": 42,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := toGeminiSchema(tt.genkitSchema, tt.genkitSchema)
			if (err != nil) != tt.wantErr {
				t.Errorf("toGeminiSchema() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("toGeminiSchema() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
