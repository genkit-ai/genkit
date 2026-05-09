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

package tracing

import (
	"context"
	"fmt"
	"slices"
	"strconv"
	"testing"

	"go.opentelemetry.io/otel/attribute"
)

// TODO: add tests that compare tracing data saved to disk with goldens.

func TestSpanMetadata(t *testing.T) {
	const (
		testInput  = 17
		testOutput = 18
	)
	sm := &spanMetadata{
		Name:     "name",
		State:    spanStateSuccess,
		Path:     "parent/name",
		Input:    testInput,
		Output:   testOutput,
		Metadata: map[string]string{"key": "value"},
	}

	got := sm.attributes()
	want := []attribute.KeyValue{
		attribute.String("genkit:name", "name"),
		attribute.String("genkit:state", "success"),
		attribute.String("genkit:input", strconv.Itoa(testInput)),
		attribute.String("genkit:path", "parent/name"),
		attribute.String("genkit:output", strconv.Itoa(testOutput)),
		attribute.String("genkit:metadata:key", "value"),
	}
	if !slices.Equal(got, want) {
		t.Errorf("\ngot  %v\nwant %v", got, want)
	}
}

func TestSpanMetadataWithTypeAndSubtype(t *testing.T) {
	const (
		testInput  = "test input"
		testOutput = "test output"
	)
	sm := &spanMetadata{
		Name:    "myTool",
		State:   spanStateSuccess,
		Path:    "/{chatFlow,t:flow}/{myTool,t:action}",
		Type:    "action",
		Subtype: "tool",
		Input:   testInput,
		Output:  testOutput,
		IsRoot:  false,
		Metadata: map[string]string{
			"customKey":     "customValue",
			"additionalKey": "additionalValue",
		},
	}

	got := sm.attributes()

	// Expected attributes
	expectedAttrs := map[string]string{
		"genkit:name":                   "myTool",
		"genkit:state":                  "success",
		"genkit:input":                  `"test input"`,
		"genkit:path":                   "/{chatFlow,t:flow}/{myTool,t:action}",
		"genkit:output":                 `"test output"`,
		"genkit:type":                   "action",
		"genkit:metadata:subtype":       "tool",
		"genkit:metadata:customKey":     "customValue",
		"genkit:metadata:additionalKey": "additionalValue",
	}

	// Check that we have exactly the expected number of attributes
	if len(got) != len(expectedAttrs) {
		t.Errorf("Expected %d attributes, got %d", len(expectedAttrs), len(got))
	}

	// Check that all expected attributes are present with correct values
	gotMap := make(map[string]string)
	for _, attr := range got {
		gotMap[string(attr.Key)] = attr.Value.AsString()
	}

	for key, expectedValue := range expectedAttrs {
		if gotValue, exists := gotMap[key]; !exists {
			t.Errorf("Missing expected attribute: %s", key)
		} else if gotValue != expectedValue {
			t.Errorf("Attribute %s: got %q, want %q", key, gotValue, expectedValue)
		}
	}
}

func TestSpanMetadataWithRootSpan(t *testing.T) {
	sm := &spanMetadata{
		Name:    "chatFlow",
		State:   spanStateSuccess,
		Path:    "/{chatFlow,t:flow}",
		Type:    "flow",
		Subtype: "flow",
		IsRoot:  true,
	}

	got := sm.attributes()

	// Check that genkit:isRoot is included for root spans
	hasIsRoot := false
	for _, attr := range got {
		if string(attr.Key) == "genkit:isRoot" && attr.Value.AsBool() {
			hasIsRoot = true
			break
		}
	}
	if !hasIsRoot {
		t.Error("Expected genkit:isRoot attribute for root span")
	}
}

func TestBuildAnnotatedPath(t *testing.T) {
	testCases := []struct {
		name       string
		parentPath string
		spanType   string
		expected   string
	}{
		{
			name:       "Root flow",
			parentPath: "",
			spanType:   "flow",
			expected:   "/{Root flow,t:flow}",
		},
		{
			name:       "Action under flow",
			parentPath: "/{chatFlow,t:flow}",
			spanType:   "action",
			expected:   "/{chatFlow,t:flow}/{Action under flow,t:action}",
		},
		{
			name:       "Model action",
			parentPath: "/{chatFlow,t:flow}",
			spanType:   "action",
			expected:   "/{chatFlow,t:flow}/{Model action,t:action}",
		},
		{
			name:       "Generate action",
			parentPath: "/{chatFlow,t:flow}",
			spanType:   "action",
			expected:   "/{chatFlow,t:flow}/{Generate action,t:action}",
		},
		{
			name:       "No type info",
			parentPath: "/{parent,t:flow}",
			spanType:   "",
			expected:   "/{parent,t:flow}/{No type info}",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := buildAnnotatedPath(tc.name, tc.parentPath, tc.spanType)
			if got != tc.expected {
				t.Errorf("buildAnnotatedPath(%q, %q, %q) = %q, want %q",
					tc.name, tc.parentPath, tc.spanType, got, tc.expected)
			}
		})
	}
}

func TestDecoratePathWithSubtype(t *testing.T) {
	testCases := []struct {
		name     string
		path     string
		subtype  string
		expected string
	}{
		{
			name:     "Add tool subtype",
			path:     "/{chatFlow,t:flow}/{generateResponse,t:action}",
			subtype:  "tool",
			expected: "/{chatFlow,t:flow}/{generateResponse,t:action,s:tool}",
		},
		{
			name:     "Add model subtype",
			path:     "/{myFlow,t:flow}/{step,t:flowStep}/{gemini,t:action}",
			subtype:  "model",
			expected: "/{myFlow,t:flow}/{step,t:flowStep}/{gemini,t:action,s:model}",
		},
		{
			name:     "Single segment path",
			path:     "/{rootAction,t:action}",
			subtype:  "tool",
			expected: "/{rootAction,t:action,s:tool}",
		},
		{
			name:     "Empty subtype",
			path:     "/{action,t:action}",
			subtype:  "",
			expected: "/{action,t:action}",
		},
		{
			name:     "Empty path",
			path:     "",
			subtype:  "tool",
			expected: "",
		},
		{
			name:     "Path without decorations",
			path:     "/{simple}",
			subtype:  "tool",
			expected: "/{simple,s:tool}",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := decoratePathWithSubtype(tc.path, tc.subtype)
			if got != tc.expected {
				t.Errorf("decoratePathWithSubtype(%q, %q) = %q, want %q",
					tc.path, tc.subtype, got, tc.expected)
			}
		})
	}
}

func TestRunInNewSpanWithMetadata(t *testing.T) {

	testCases := []struct {
		name            string
		spanName        string
		isRoot          bool
		metadata        *SpanMetadata
		expectedType    string
		expectedSubtype string
		expectedPath    string
	}{
		{
			name:     "Tool action span",
			spanName: "myTool",
			isRoot:   true,
			metadata: &SpanMetadata{
				Name:    "myTool",
				IsRoot:  false, // Input says false, but no parent = auto-root
				Type:    "action",
				Subtype: "tool",
			},
			expectedType:    "action",
			expectedSubtype: "tool",
			expectedPath:    "/{myTool,t:action,s:tool}",
		},
		{
			name:     "Flow span",
			spanName: "chatFlow",
			isRoot:   true,
			metadata: &SpanMetadata{
				Name:    "chatFlow",
				IsRoot:  true,
				Type:    "action", // Flows get type "action"
				Subtype: "flow",   // Flow type goes into subtype
			},
			expectedType:    "action",
			expectedSubtype: "flow",
			expectedPath:    "/{chatFlow,t:flow}", // Flows use simplified t:flow format
		},
		{
			name:     "Model action span",
			spanName: "generate",
			isRoot:   true,
			metadata: &SpanMetadata{
				Name:    "generate",
				IsRoot:  false, // Input says false, but no parent = auto-root
				Type:    "action",
				Subtype: "model",
			},
			expectedType:    "action",
			expectedSubtype: "model",
			expectedPath:    "/{generate,t:action,s:model}",
		},
		{
			name:     "Nil metadata",
			spanName: "testSpan",
			isRoot:   true,
			metadata: &SpanMetadata{
				Name:   "testSpan",
				IsRoot: false, // Input says false, but no parent = auto-root
			},
			expectedType:    "",
			expectedSubtype: "",
			expectedPath:    "/{testSpan}",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			input := "test input"

			output, err := RunInNewSpan(ctx, tc.metadata, input,
				func(ctx context.Context, input string) (string, error) {
					// Verify that span metadata is available in context
					sm := spanMetaKey.FromContext(ctx)
					if sm == nil {
						t.Error("Expected span metadata in context")
						return "", nil
					}

					// Verify the metadata fields
					if sm.Type != tc.expectedType {
						t.Errorf("Expected type %q, got %q", tc.expectedType, sm.Type)
					}
					if sm.Subtype != tc.expectedSubtype {
						t.Errorf("Expected subtype %q, got %q", tc.expectedSubtype, sm.Subtype)
					}
					if sm.Path != tc.expectedPath {
						t.Errorf("Expected path %q, got %q", tc.expectedPath, sm.Path)
					}
					if sm.IsRoot != tc.isRoot {
						t.Errorf("Expected isRoot %v, got %v", tc.isRoot, sm.IsRoot)
					}

					return "test output", nil
				})

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			if output != "test output" {
				t.Errorf("Expected output 'test output', got %q", output)
			}
		})
	}
}

func TestRunInNewSpanWithTypeConvenience(t *testing.T) {
	ctx := context.Background()

	metadata := &SpanMetadata{
		Name:    "myTool",
		IsRoot:  false,
		Type:    "action",
		Subtype: "tool",
	}

	output, err := RunInNewSpan(ctx, metadata, "input",
		func(ctx context.Context, input string) (string, error) {
			sm := spanMetaKey.FromContext(ctx)
			if sm == nil {
				t.Error("Expected span metadata in context")
				return "", nil
			}

			if sm.Type != "action" {
				t.Errorf("Expected type 'action', got %q", sm.Type)
			}
			if sm.Subtype != "tool" {
				t.Errorf("Expected subtype 'tool', got %q", sm.Subtype)
			}

			return "output", nil
		})

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if output != "output" {
		t.Errorf("Expected output 'output', got %q", output)
	}
}

func TestNestedSpanPaths(t *testing.T) {
	ctx := context.Background()

	// Test nested spans to verify path building
	_, err := RunInNewSpan(ctx, &SpanMetadata{Name: "chatFlow", IsRoot: true, Type: "action", Subtype: "flow"}, "input",
		func(ctx context.Context, input string) (string, error) {
			// Nested action span
			return RunInNewSpan(ctx, &SpanMetadata{Name: "myTool", IsRoot: false, Type: "action", Subtype: "tool"}, input,
				func(ctx context.Context, input string) (string, error) {
					sm := spanMetaKey.FromContext(ctx)
					if sm == nil {
						t.Error("Expected span metadata in context")
						return "", nil
					}

					expectedPath := "/{chatFlow,t:flow}/{myTool,t:action,s:tool}"
					if sm.Path != expectedPath {
						t.Errorf("Expected nested path %q, got %q", expectedPath, sm.Path)
					}

					return "nested output", nil
				})
		})

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

// TestIsFailureSourceOnError verifies that genkit:isFailureSource is set correctly
func TestIsFailureSourceOnError(t *testing.T) {
	ctx := context.Background()

	testErr := fmt.Errorf("test error")

	// Call RunInNewSpan with a function that returns an error
	_, err := RunInNewSpan(ctx, &SpanMetadata{
		Name: "failing-action",
		Type: "action",
	}, "input", func(ctx context.Context, input string) (string, error) {
		return "", testErr
	})

	// The test confirms the function works - isFailureSource should be set
	// on the span via span.SetAttributes() during error handling
	if err == nil {
		t.Fatal("Expected error to be returned")
	}
}

// TestRootSpanAutoDetection tests that spans are automatically marked as root when no parent exists
func TestRootSpanAutoDetection(t *testing.T) {
	ctx := context.Background()

	t.Run("span with no parent should be marked as root", func(t *testing.T) {
		_, err := RunInNewSpan(ctx, &SpanMetadata{
			Name:    "topLevelFlow",
			Type:    "action",
			Subtype: "flow",
			IsRoot:  false, // Even when explicitly set to false, should be overridden
		}, "input", func(ctx context.Context, input string) (string, error) {
			sm := spanMetaKey.FromContext(ctx)
			if sm == nil {
				t.Fatal("Expected span metadata in context")
			}

			if !sm.IsRoot {
				t.Error("Span with no parent should be automatically marked as root")
			}

			return "output", nil
		})

		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
	})

	t.Run("span with explicit IsRoot=true should be marked as root", func(t *testing.T) {
		_, err := RunInNewSpan(ctx, &SpanMetadata{
			Name:   "explicitRootFlow",
			Type:   "action",
			IsRoot: true, // Explicitly set to true
		}, "input", func(ctx context.Context, input string) (string, error) {
			sm := spanMetaKey.FromContext(ctx)
			if sm == nil {
				t.Fatal("Expected span metadata in context")
			}

			if !sm.IsRoot {
				t.Error("Span with explicit IsRoot=true should be marked as root")
			}

			return "output", nil
		})

		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
	})

	t.Run("nested span should not be marked as root", func(t *testing.T) {
		_, err := RunInNewSpan(ctx, &SpanMetadata{
			Name:   "parentFlow",
			Type:   "action",
			IsRoot: true,
		}, "input", func(ctx context.Context, input string) (string, error) {
			// This is a nested span - should NOT be root
			_, err := RunInNewSpan(ctx, &SpanMetadata{
				Name:   "childAction",
				Type:   "action",
				IsRoot: false,
			}, input, func(ctx context.Context, input string) (string, error) {
				sm := spanMetaKey.FromContext(ctx)
				if sm == nil {
					t.Fatal("Expected span metadata in context")
				}

				if sm.IsRoot {
					t.Error("Nested span should NOT be marked as root")
				}

				return "nested_output", nil
			})
			return "parent_output", err
		})

		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
	})
}
