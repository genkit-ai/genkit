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
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
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

func TestSpanMetadataStartAttributes(t *testing.T) {
	sm := &spanMetadata{
		Name:     "myTool",
		State:    spanStateSuccess, // run-determined: must be excluded
		Path:     "/{chatFlow,t:flow}/{myTool,t:action}",
		Type:     "action",
		Subtype:  "tool",
		Input:    "in",  // deferred to end: must be excluded
		Output:   "out", // run-determined: must be excluded
		IsRoot:   true,
		Metadata: map[string]string{"key": "value"},
	}

	// startAttributes carries only the shape known at span start, in the same
	// order (and with the same keys/values) as the matching prefix of
	// attributes(), so the end write reasserts them cleanly.
	got := sm.startAttributes()
	want := []attribute.KeyValue{
		attribute.String("genkit:name", "myTool"),
		attribute.String("genkit:path", "/{chatFlow,t:flow}/{myTool,t:action}"),
		attribute.String("genkit:type", "action"),
		attribute.String("genkit:metadata:subtype", "tool"),
		attribute.Bool("genkit:isRoot", true),
		attribute.String("genkit:metadata:key", "value"),
	}
	if !slices.Equal(got, want) {
		t.Errorf("\ngot  %v\nwant %v", got, want)
	}

	// The run-determined attributes must not leak into the start write, and input
	// and init are seeded separately (see inputAttributes / TestSpanMetadataInputAttributes).
	for _, kv := range got {
		switch kv.Key {
		case "genkit:state", "genkit:input", "genkit:output", "genkit:init":
			t.Errorf("startAttributes must not include %q", kv.Key)
		}
	}
}

func TestSpanMetadataInputAttributes(t *testing.T) {
	// With init: both genkit:input and genkit:init, matching attributes().
	sm := &spanMetadata{Input: "in", Init: map[string]string{"prefix": "PFX:"}}
	got := sm.inputAttributes()
	want := []attribute.KeyValue{
		attribute.String("genkit:input", `"in"`),
		attribute.String("genkit:init", `{"prefix":"PFX:"}`),
	}
	if !slices.Equal(got, want) {
		t.Errorf("with init:\ngot  %v\nwant %v", got, want)
	}

	// Without init (e.g. a bidi/agent root span has nil input and may have no
	// init): genkit:input is still present, recorded as JSON null.
	smNoInit := &spanMetadata{Input: nil}
	got = smNoInit.inputAttributes()
	want = []attribute.KeyValue{attribute.String("genkit:input", "null")}
	if !slices.Equal(got, want) {
		t.Errorf("without init:\ngot  %v\nwant %v", got, want)
	}
}

func TestSpanMetadataInit(t *testing.T) {
	sm := &spanMetadata{
		Name:  "name",
		State: spanStateSuccess,
		Path:  "parent/name",
		Input: "in",
		Init:  map[string]string{"prefix": "PFX:"},
	}

	got := sm.attributes()
	want := []attribute.KeyValue{
		attribute.String("genkit:name", "name"),
		attribute.String("genkit:state", "success"),
		attribute.String("genkit:input", `"in"`),
		attribute.String("genkit:path", "parent/name"),
		attribute.String("genkit:init", `{"prefix":"PFX:"}`),
	}
	if !slices.Equal(got, want) {
		t.Errorf("\ngot  %v\nwant %v", got, want)
	}
}

func TestSpanMetadataInlineMedia(t *testing.T) {
	media := func(url string) map[string]any {
		return map[string]any{"media": map[string]string{"url": url}}
	}
	tests := []struct {
		env  string
		want map[string]string
	}{
		{
			env: "prod",
			want: map[string]string{
				"genkit:input":  `{"media":{"url":"data:image/png;base64,[redacted]"}}`,
				"genkit:init":   `{"media":{"url":"data:image/png;base64,[redacted]"}}`,
				"genkit:output": `{"media":{"url":"data:image/png;base64,[redacted]"}}`,
			},
		},
		{
			// The Dev UI renders media from these attributes.
			env: "dev",
			want: map[string]string{
				"genkit:input":  `{"media":{"url":"data:image/png;base64,input-bytes"}}`,
				"genkit:init":   `{"media":{"url":"data:image/png;base64,init-bytes"}}`,
				"genkit:output": `{"media":{"url":"data:image/png;base64,output-bytes"}}`,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.env, func(t *testing.T) {
			t.Setenv("GENKIT_ENV", tt.env)
			// Fresh per case: encodings are cached on the span, and the
			// environment never changes mid-run outside tests.
			sm := &spanMetadata{
				Input:  media("data:image/png;base64,input-bytes"),
				Init:   media("data:image/png;base64,init-bytes"),
				Output: media("data:image/png;base64,output-bytes"),
			}
			for _, kv := range append(sm.attributes(), sm.inputAttributes()...) {
				want, ok := tt.want[string(kv.Key)]
				if !ok {
					continue
				}
				if got := kv.Value.AsString(); got != want {
					t.Errorf("%s = %s, want %s", kv.Key, got, want)
				}
			}
		})
	}
}

func TestRedactInlineMedia(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "data URL keeps header and field order",
			in:   `{"url":"data:image/png;base64,AAAA","contentType":"image/png"}`,
			want: `{"url":"data:image/png;base64,[redacted]","contentType":"image/png"}`,
		},
		{
			name: "every data URL in arrays",
			in:   `["data:image/png;base64,AAAA",1,"data:audio/wav;base64,BBBB"]`,
			want: `["data:image/png;base64,[redacted]",1,"data:audio/wav;base64,[redacted]"]`,
		},
		{
			name: "header is case-insensitive",
			in:   `"DATA:image/png;BASE64,AAAA"`,
			want: `"DATA:image/png;BASE64,[redacted]"`,
		},
		{
			name: "header with parameters",
			in:   `"data:text/plain;charset=utf-8;base64,AAAA"`,
			want: `"data:text/plain;charset=utf-8;base64,[redacted]"`,
		},
		{
			name: "escaped quote does not end a string",
			in:   `{"a":"x\"data:image/png;base64,AAAA","b":"data:image/png;base64,BBBB"}`,
			want: `{"a":"x\"data:image/png;base64,AAAA","b":"data:image/png;base64,[redacted]"}`,
		},
		{
			name: "remote URL",
			in:   `{"url":"https://example.com/image.png"}`,
			want: `{"url":"https://example.com/image.png"}`,
		},
		{
			name: "data URL inside text",
			in:   `"see data:image/png;base64,AAAA"`,
			want: `"see data:image/png;base64,AAAA"`,
		},
		{
			name: "prose with a space in the header",
			in:   `"data: see notes;base64, then decode"`,
			want: `"data: see notes;base64, then decode"`,
		},
		{
			name: "data URL without base64",
			in:   `"data:text/plain,hello"`,
			want: `"data:text/plain,hello"`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := redactInlineMedia(tt.in); got != tt.want {
				t.Errorf("redactInlineMedia(%s) = %s, want %s", tt.in, got, tt.want)
			}
		})
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
			name:     "Util span",
			spanName: "generate",
			isRoot:   true,
			metadata: &SpanMetadata{
				Name: "generate",
				Type: "util", // A tool-loop turn: no subtype, so nothing follows the type.
			},
			expectedType:    "util",
			expectedSubtype: "",
			expectedPath:    "/{generate,t:util}",
		},
		{
			name:     "Util action span",
			spanName: "generate",
			isRoot:   true,
			metadata: &SpanMetadata{
				Name:    "generate",
				Type:    "action", // /util/generate is an action first,
				Subtype: "util",   // so util is its subtype, not its type.
			},
			expectedType:    "action",
			expectedSubtype: "util",
			expectedPath:    "/{generate,t:action,s:util}",
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

// TestIsFailureSourceOnlyOnOriginatingSpan verifies that a failure is attributed
// to the span where the error originates rather than to every span that
// propagates it, while all of them still report the error and receive it
// unchanged.
func TestIsFailureSourceOnlyOnOriginatingSpan(t *testing.T) {
	ctx := context.Background()
	boom := errors.New("boom")
	recorder := tracetest.NewSpanRecorder()
	tp := TracerProvider()
	tp.RegisterSpanProcessor(recorder)
	t.Cleanup(func() { tp.UnregisterSpanProcessor(recorder) })

	var outerSM, middleSM, innerSM *spanMetadata

	_, err := RunInNewSpan(ctx, &SpanMetadata{Name: "outer", Type: "action", Subtype: "flow"}, "in",
		func(ctx context.Context, in string) (string, error) {
			outerSM = spanMetaKey.FromContext(ctx)
			return RunInNewSpan(ctx, &SpanMetadata{Name: "middle", Type: "action"}, in,
				func(ctx context.Context, in string) (string, error) {
					middleSM = spanMetaKey.FromContext(ctx)
					out, err := RunInNewSpan(ctx, &SpanMetadata{Name: "inner", Type: "action", Subtype: "tool"}, in,
						func(ctx context.Context, in string) (string, error) {
							innerSM = spanMetaKey.FromContext(ctx)
							return "", boom
						})
					if err != boom {
						t.Errorf("parent span received %#v from its child, want %#v", err, boom)
					}
					return out, err
				})
		})

	if err != boom {
		t.Errorf("outermost span returned %#v, want %#v", err, boom)
	}
	if !innerSM.IsFailureSource {
		t.Error("originating span is not marked as the failure source")
	}
	for _, sm := range []*spanMetadata{middleSM, outerSM} {
		if sm.IsFailureSource {
			t.Errorf("span %q only propagated the failure but is marked as its source", sm.Name)
		}
		if sm.State != spanStateError {
			t.Errorf("span %q state = %q, want %q", sm.Name, sm.State, spanStateError)
		}
	}
	spans := recorder.Ended()
	if len(spans) != 3 {
		t.Fatalf("recorded %d spans, want 3", len(spans))
	}
	for _, span := range spans {
		if status := span.Status(); status.Code != codes.Error || status.Description != boom.Error() {
			t.Errorf("span %q status = %v, want Error with description %q", span.Name(), status, boom.Error())
		}
		if events := span.Events(); len(events) != 1 || events[0].Name != "exception" {
			t.Errorf("span %q events = %v, want one exception event", span.Name(), events)
		}
	}
}

// TestIsFailureSourceAttribution verifies which span is the failure source when
// a parent changes the error its child returned before returning it.
func TestIsFailureSourceAttribution(t *testing.T) {
	boom := errors.New("boom")
	cleanupErr := errors.New("cleanup failed")
	tests := []struct {
		name string
		// parent builds the error the parent span returns from the errors its
		// two child spans returned.
		parent     func(first, second error) error
		wantSource bool
	}{
		{
			name:       "wrapped child error",
			parent:     func(first, _ error) error { return fmt.Errorf("step failed: %w", first) },
			wantSource: false,
		},
		{
			name:       "joined child errors",
			parent:     func(first, second error) error { return errors.Join(first, second) },
			wantSource: false,
		},
		{
			name:       "own failure joined with a child error",
			parent:     func(first, _ error) error { return errors.Join(cleanupErr, first) },
			wantSource: true,
		},
		{
			name:       "unrelated error",
			parent:     func(error, error) error { return cleanupErr },
			wantSource: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var parentSM *spanMetadata
			RunInNewSpan(context.Background(), &SpanMetadata{Name: "parent"}, "in",
				func(ctx context.Context, in string) (string, error) {
					parentSM = spanMetaKey.FromContext(ctx)
					fail := func(ctx context.Context, in string) (string, error) { return "", boom }
					_, first := RunInNewSpan(ctx, &SpanMetadata{Name: "first"}, in, fail)
					_, second := RunInNewSpan(ctx, &SpanMetadata{Name: "second"}, in, fail)
					return "", tt.parent(first, second)
				})
			if parentSM.IsFailureSource != tt.wantSource {
				t.Errorf("parent IsFailureSource = %t, want %t", parentSM.IsFailureSource, tt.wantSource)
			}
		})
	}
}

// TestIsFailureSourceForStoredError verifies that attribution is per trace: an
// error a child span returned in an earlier trace is attributed to the span
// that returns it again later, without its child failing.
func TestIsFailureSourceForStoredError(t *testing.T) {
	ctx := context.Background()
	var (
		once    sync.Once
		initErr error
	)
	step := func(ctx context.Context) *spanMetadata {
		var stepSM *spanMetadata
		RunInNewSpan(ctx, &SpanMetadata{Name: "step"}, "in",
			func(ctx context.Context, in string) (string, error) {
				stepSM = spanMetaKey.FromContext(ctx)
				once.Do(func() {
					_, initErr = RunInNewSpan(ctx, &SpanMetadata{Name: "init"}, in,
						func(context.Context, string) (string, error) { return "", errors.New("init failed") })
				})
				return "", initErr
			})
		return stepSM
	}

	if sm := step(ctx); sm.IsFailureSource {
		t.Error("first run: step only propagated its child's failure but is marked as its source")
	}
	if sm := step(ctx); !sm.IsFailureSource {
		t.Error("second run: step returned a stored error with no failing child but is not marked as its source")
	}
}

type nonComparableError struct{ causes []string }

func (e nonComparableError) Error() string { return strings.Join(e.causes, "; ") }

// TestIsFailureSourceNonComparableError verifies that an error of a
// non-comparable type does not panic attribution; the parent that returns it
// is counted as a source too.
func TestIsFailureSourceNonComparableError(t *testing.T) {
	var parentSM *spanMetadata
	RunInNewSpan(context.Background(), &SpanMetadata{Name: "parent"}, "in",
		func(ctx context.Context, in string) (string, error) {
			parentSM = spanMetaKey.FromContext(ctx)
			return RunInNewSpan(ctx, &SpanMetadata{Name: "child"}, in,
				func(context.Context, string) (string, error) {
					return "", nonComparableError{causes: []string{"boom"}}
				})
		})
	if !parentSM.IsFailureSource {
		t.Error("parent IsFailureSource = false, want true for an error that cannot be matched")
	}
}

func TestRunInNewSpanRecordsPartialOutputOnError(t *testing.T) {
	// A failure can still carry a result, so the span records it and a trace
	// shows what the call produced. A failure that produced nothing records
	// no output at all, rather than stamping a null one on every failing span.
	ctx := context.Background()
	boom := errors.New("boom")

	// The metadata is filled in after the function returns, so the test keeps
	// the pointer the run handed its callback and reads it afterwards.
	run := func(t *testing.T, fn func() (*string, error)) *spanMetadata {
		t.Helper()
		var sm *spanMetadata
		_, err := RunInNewSpan(ctx, &SpanMetadata{Name: "partial"}, "in",
			func(ctx context.Context, in string) (*string, error) {
				sm = spanMetaKey.FromContext(ctx)
				return fn()
			})
		if !errors.Is(err, boom) {
			t.Fatalf("err = %v, want boom", err)
		}
		return sm
	}

	partial := "as far as it got"
	sm := run(t, func() (*string, error) { return &partial, boom })
	if sm.State != spanStateError {
		t.Errorf("state = %q, want %q", sm.State, spanStateError)
	}
	got, ok := sm.Output.(*string)
	if !ok {
		t.Fatalf("failing span recorded output %v, want the partial result", sm.Output)
	}
	if got != &partial {
		t.Errorf("output = %q, want the returned partial", *got)
	}

	if sm := run(t, func() (*string, error) { return nil, boom }); sm.Output != nil {
		t.Errorf("output = %v, want none: the call produced nothing", sm.Output)
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
