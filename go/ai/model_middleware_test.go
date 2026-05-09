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

package ai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestValidateSupport(t *testing.T) {
	tests := []struct {
		name    string
		opts    *ModelOptions
		input   *ModelRequest
		wantErr bool
	}{
		{
			name: "valid request with no special features",
			opts: &ModelOptions{
				Supports: &ModelSupports{
					Media:      false,
					Tools:      false,
					Multiturn:  false,
					ToolChoice: false,
					SystemRole: false,
				},
			},
			input: &ModelRequest{
				Messages: []*Message{
					NewUserTextMessage("hello"),
				},
			},
			wantErr: false,
		},
		{
			name: "media not supported but requested",
			opts: &ModelOptions{
				Supports: &ModelSupports{
					Media: false,
				},
			},
			input: &ModelRequest{
				Messages: []*Message{
					{Content: []*Part{NewMediaPart("image/png", "data:image/png;base64,...")}},
				},
			},
			wantErr: true,
		},
		{
			name: "tools not supported but requested",
			opts: &ModelOptions{
				Supports: &ModelSupports{
					Tools: false,
				},
			},
			input: &ModelRequest{
				Tools: []*ToolDefinition{
					{
						Name:        "test-tool",
						Description: "A test tool",
					},
				},
			},
			wantErr: true,
		},
		{
			name: "multiturn not supported but requested",
			opts: &ModelOptions{
				Supports: &ModelSupports{
					Multiturn: false,
				},
			},
			input: &ModelRequest{
				Messages: []*Message{
					NewUserTextMessage("message 1"),
					NewUserTextMessage("message 2"),
				},
			},
			wantErr: true,
		},
		{
			name: "tool choice not supported but requested",
			opts: &ModelOptions{
				Supports: &ModelSupports{
					ToolChoice: false,
				},
			},
			input: &ModelRequest{
				ToolChoice: ToolChoiceRequired,
			},
			wantErr: true,
		},
		{
			name: "system role not supported but requested",
			opts: &ModelOptions{
				Supports: &ModelSupports{
					SystemRole: false,
				},
			},
			input: &ModelRequest{
				Messages: []*Message{
					{Role: RoleSystem, Content: []*Part{NewTextPart("system instruction")}},
				},
			},
			wantErr: true,
		},
		{
			name: "all features supported and used",
			opts: &ModelOptions{
				Supports: &ModelSupports{
					Media:      true,
					Tools:      true,
					Multiturn:  true,
					ToolChoice: true,
					SystemRole: true,
				},
			},
			input: &ModelRequest{
				Messages: []*Message{
					NewSystemTextMessage("be helpful"),
					NewUserTextMessage("hello! look at this image"),
					{Content: []*Part{NewMediaPart("image/png", "data:image/png;base64,...")}},
				},
				Tools: []*ToolDefinition{
					{
						Name:        "test-tool",
						Description: "A test tool",
					},
				},
				ToolChoice: ToolChoiceNone,
			},
			wantErr: false,
		},
		{
			name: "nil supports defaults to no features",
			opts: nil,
			input: &ModelRequest{
				Messages: []*Message{
					{Content: []*Part{NewMediaPart("image/png", "data:image/png;base64,...")}},
				},
			},
			wantErr: true,
		},
		{
			name: "mixed content types in message",
			opts: &ModelOptions{
				Supports: &ModelSupports{
					Media: false,
				},
			},
			input: &ModelRequest{
				Messages: []*Message{
					{Content: []*Part{
						NewTextPart("text content"),
						NewMediaPart("image/png", "data:image/png;base64,..."),
					}},
				},
			},
			wantErr: true,
		},
		{
			name: "supported version",
			opts: &ModelOptions{
				Supports: &ModelSupports{},
				Versions: []string{"v1", "v2"},
			},
			input: &ModelRequest{
				Messages: []*Message{
					NewUserTextMessage("hello"),
				},
				Config: map[string]any{"version": "v1"},
			},
			wantErr: false,
		},
		{
			name: "unsupported version",
			opts: &ModelOptions{
				Supports: &ModelSupports{},
				Versions: []string{"v1", "v2"},
			},
			input: &ModelRequest{
				Messages: []*Message{
					NewUserTextMessage("hello"),
				},
				Config: map[string]any{"version": "v3"},
			},
			wantErr: true,
		},
		{
			name: "non-string version",
			opts: &ModelOptions{
				Supports: &ModelSupports{},
				Versions: []string{"v1", "v2"},
			},
			input: &ModelRequest{
				Messages: []*Message{
					NewUserTextMessage("hello"),
				},
				Config: map[string]any{"version": 1},
			},
			wantErr: true,
		},
		{
			name: "struct with supported version",
			opts: &ModelOptions{
				Supports: &ModelSupports{},
				Versions: []string{"v1", "v2"},
			},
			input: &ModelRequest{
				Messages: []*Message{
					NewUserTextMessage("hello"),
				},
				Config: &GenerationCommonConfig{Version: "v1"},
			},
			wantErr: false,
		},
		{
			name: "struct with unsupported version",
			opts: &ModelOptions{
				Supports: &ModelSupports{},
				Versions: []string{"v1", "v2"},
			},
			input: &ModelRequest{
				Messages: []*Message{
					NewUserTextMessage("hello"),
				},
				Config: &GenerationCommonConfig{Version: "v3"},
			},
			wantErr: true,
		},
	}

	mockModelFunc := func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		return &ModelResponse{}, nil
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handler := validateSupport("test-model", tt.opts)(mockModelFunc)
			_, err := handler(context.Background(), tt.input, nil)

			if (err != nil) != tt.wantErr {
				t.Errorf("validateSupport() error = %v, wantErr %v", err, tt.wantErr)
				if err != nil {
					t.Logf("Error message: %v", err)
				}
			}
		})
	}
}

func TestSimulateSystemPrompt(t *testing.T) {
	testCases := []struct {
		name        string
		opts        *ModelOptions
		options     map[string]string
		input       *ModelRequest
		expected    *ModelRequest
		supportsSys bool
	}{
		{
			name: "system role not supported, system message present",
			opts: &ModelOptions{Supports: &ModelSupports{SystemRole: false}},
			input: &ModelRequest{
				Messages: []*Message{
					{Role: "system", Content: []*Part{NewTextPart("Be helpful.")}},
					{Role: "user", Content: []*Part{NewTextPart("Hello.")}},
				},
			},
			expected: &ModelRequest{
				Messages: []*Message{
					{Role: "user", Content: []*Part{NewTextPart("SYSTEM INSTRUCTIONS:\n"), NewTextPart("Be helpful.")}},
					{Role: "model", Content: []*Part{NewTextPart("Understood.")}},
					{Role: "user", Content: []*Part{NewTextPart("Hello.")}},
				},
			},
			supportsSys: true,
		},
		{
			name: "system role supported, no system message",
			opts: &ModelOptions{Supports: &ModelSupports{SystemRole: true}},
			input: &ModelRequest{
				Messages: []*Message{
					{Role: "user", Content: []*Part{NewTextPart("Hello.")}},
				},
			},
			expected: &ModelRequest{
				Messages: []*Message{
					{Role: "user", Content: []*Part{NewTextPart("Hello.")}},
				},
			},
			supportsSys: true,
		},
		{
			name: "system role supported, with system message",
			opts: &ModelOptions{Supports: &ModelSupports{SystemRole: true}},
			input: &ModelRequest{
				Messages: []*Message{
					{Role: "system", Content: []*Part{NewTextPart("Be helpful.")}},
					{Role: "user", Content: []*Part{NewTextPart("Hello.")}},
				},
			},
			expected: &ModelRequest{
				Messages: []*Message{
					{Role: "system", Content: []*Part{NewTextPart("Be helpful.")}},
					{Role: "user", Content: []*Part{NewTextPart("Hello.")}},
				},
			},
			supportsSys: false,
		},
		{
			name: "custom preface and acknowledgement",
			opts: &ModelOptions{Supports: &ModelSupports{SystemRole: false}},
			options: map[string]string{
				"preface":         "CUSTOM PREFACE:\n",
				"acknowledgement": "OKAY!",
			},
			input: &ModelRequest{
				Messages: []*Message{
					{Role: "system", Content: []*Part{NewTextPart("Be helpful.")}},
					{Role: "user", Content: []*Part{NewTextPart("Hello.")}},
				},
			},
			expected: &ModelRequest{
				Messages: []*Message{
					{Role: "user", Content: []*Part{NewTextPart("CUSTOM PREFACE:\n"), NewTextPart("Be helpful.")}},
					{Role: "model", Content: []*Part{NewTextPart("OKAY!")}},
					{Role: "user", Content: []*Part{NewTextPart("Hello.")}},
				},
			},
			supportsSys: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			next := func(ctx context.Context, input *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				if !reflect.DeepEqual(input, tc.expected) {
					t.Errorf("Input messages were not modified as expected. got: %+v, want: %+v", input, tc.expected)
				}
				return &ModelResponse{}, nil
			}
			middleware := simulateSystemPrompt(tc.opts, tc.options)
			_, _ = middleware(next)(context.Background(), tc.input, nil)
		})
	}
}

func TestDownloadRequestMedia(t *testing.T) {
	testCases := []struct {
		name           string
		input          *ModelRequest
		options        *DownloadMediaOptions
		setupServer    func() *httptest.Server
		expectedResult *ModelRequest
	}{
		{
			name: "successful download",
			input: &ModelRequest{
				Messages: []*Message{
					NewUserMessage(NewMediaPart("image/png", "http://127.0.0.1:60289")),
				},
			},
			setupServer: func() *httptest.Server {
				testData := []byte("data:image/png;base64,dGVzdCBpbWFnZSBkYXRh")
				contentType := "image/png"
				ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", contentType)
					w.Write(testData)
				}))
				return ts
			},
			expectedResult: &ModelRequest{
				Messages: []*Message{
					NewUserMessage(NewMediaPart("image/png", "data:image/png;base64,dGVzdCBpbWFnZSBkYXRh")),
				},
			},
		},
		{
			name: "base64 media not to download",
			input: &ModelRequest{
				Messages: []*Message{
					NewUserMessage(NewMediaPart("image/png", "data:image/png;base64,dGVzdCBpbWFnZSBkYXRh")),
				},
			},
			expectedResult: &ModelRequest{
				Messages: []*Message{
					NewUserMessage(NewMediaPart("image/png", "data:image/png;base64,dGVzdCBpbWFnZSBkYXRh")),
				},
			},
		},
		{
			name: "filter applied not satisfied",
			input: &ModelRequest{
				Messages: []*Message{
					{
						Content: []*Part{
							{
								ContentType: "image/png",
								Text:        "http://127.0.0.1:60289",
							},
						},
					},
				},
			},
			options: &DownloadMediaOptions{
				Filter: func(part *Part) bool {
					return true
				},
			},
			setupServer: func() *httptest.Server {
				testData := []byte("data:image/png;base64,dGVzdCBpbWFnZSBkYXRh")
				contentType := "image/png"
				ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", contentType)
					w.Write(testData)
				}))
				return ts
			},
			expectedResult: &ModelRequest{
				Messages: []*Message{
					{
						Content: []*Part{
							{
								ContentType: "image/png",
								Text:        "data:image/png;base64,dGVzdCBpbWFnZSBkYXRh",
							},
						},
					},
				},
			},
		},
		{
			name: "filter applied satisfied",
			input: &ModelRequest{
				Messages: []*Message{
					{
						Content: []*Part{
							{
								ContentType: "image/png",
								Text:        "http://127.0.0.1:60289",
							},
						},
					},
				},
			},
			options: &DownloadMediaOptions{
				Filter: func(part *Part) bool {
					return false
				},
			},
			expectedResult: &ModelRequest{
				Messages: []*Message{
					{
						Content: []*Part{
							{
								ContentType: "image/png",
								Text:        "http://127.0.0.1:60289",
							},
						},
					},
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var ts *httptest.Server
			if tc.setupServer != nil {
				ts = tc.setupServer()
				// Get the response body from the test server
				resp, err := http.Get(ts.URL)
				if err != nil {
					t.Fatalf("Error getting test server response: %v", err)
				}
				defer resp.Body.Close()
				body, err := io.ReadAll(resp.Body)
				if err != nil {
					t.Fatalf("Error reading test server response body: %v", err)
				}

				if resp.StatusCode == http.StatusOK {
					// Set the text to the response body
					tc.input.Messages[0].Content[0].Text = string(body)
				}
				defer ts.Close()

			}
			next := func(ctx context.Context, input *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				return &ModelResponse{}, nil
			}
			middleware := DownloadRequestMedia(tc.options)
			_, err := middleware(next)(context.Background(), tc.input, nil)

			if err != nil {
				t.Errorf("Expected no error, but got: %v", err)
			} else if !reflect.DeepEqual(tc.input, tc.expectedResult) {
				t.Errorf("Expected result: %v, but got: %v", tc.expectedResult, tc.input)
			}
		})
	}
}

func TestAugmentWithContext(t *testing.T) {
	testCases := []struct {
		name         string
		opts         *ModelOptions
		options      *AugmentWithContextOptions
		input        *ModelRequest
		expected     *ModelRequest
		wantMetadata map[string]any
	}{
		{
			name: "model supports context, should bypass middleware",
			opts: &ModelOptions{Supports: &ModelSupports{Context: true}},
			input: &ModelRequest{
				Messages: []*Message{
					NewSystemTextMessage("You are a narrator."),
					NewUserTextMessage("Poem on star."),
				},
			},
			expected: &ModelRequest{
				Messages: []*Message{
					NewSystemTextMessage("You are a narrator."),
					NewUserTextMessage("Poem on star."),
				},
			},
		},
		{
			name: "no docs, should bypass middleware",
			opts: &ModelOptions{Supports: &ModelSupports{Context: false}},
			input: &ModelRequest{
				Messages: []*Message{
					NewSystemTextMessage("You are a narrator."),
					NewUserTextMessage("Poem on star."),
				},
			},
			expected: &ModelRequest{
				Messages: []*Message{
					NewSystemTextMessage("You are a narrator."),
					NewUserTextMessage("Poem on star."),
				},
			},
		},
		{
			name: "no user messages, should bypass middleware",
			opts: &ModelOptions{Supports: &ModelSupports{Context: false}},
			input: &ModelRequest{
				Messages: []*Message{
					NewSystemTextMessage("You are a narrator."),
				},
			},
			expected: &ModelRequest{
				Messages: []*Message{
					NewSystemTextMessage("You are a narrator."),
				},
			},
		},
		{
			name: "context already present , should bypass middleware",
			opts: &ModelOptions{Supports: &ModelSupports{Context: false}},
			input: &ModelRequest{
				Messages: []*Message{
					NewSystemTextMessage("You are a narrator."),
					NewUserTextMessage("Poem on star."),
					NewUserMessageWithMetadata(map[string]any{"purpose": "context"}, NewTextPart("This is the context")),
				},
			},
			expected: &ModelRequest{
				Messages: []*Message{
					NewSystemTextMessage("You are a narrator."),
					NewUserTextMessage("Poem on star."),
					NewUserMessageWithMetadata(map[string]any{"purpose": "context"}, NewTextPart("This is the context")),
				},
			},
		},
		{
			name: "context not present multiple docs present , should get augment",
			opts: &ModelOptions{Supports: &ModelSupports{Context: false}},
			input: &ModelRequest{
				Messages: []*Message{
					NewSystemTextMessage("You are a narrator."),
					NewUserTextMessage("My kid is 4 years old."),
					NewUserTextMessage("Poem on star."),
				},
				Docs: []*Document{
					DocumentFromText("this is test doc 1", map[string]any{"purpose": "context", "ref": "doc1"}),
					DocumentFromText("this is test doc 2", map[string]any{"purpose": "context", "ref": "doc2"}),
				},
			},
			expected: &ModelRequest{
				Messages: []*Message{
					NewSystemTextMessage("You are a narrator."),
					NewUserTextMessage("My kid is 4 years old."),
					NewUserMessage(NewTextPart("Poem on star."),
						&Part{Text: "\n\nUse the following information " +
							"to complete your task:\n\n- [doc1]: this is test doc 1\n- [doc2]: this is test doc 2\n\n",
							Metadata: map[string]any{"purpose": "context"}}),
				},
				Docs: []*Document{
					DocumentFromText("this is test doc 1", map[string]any{"purpose": "context", "ref": "doc1"}),
					DocumentFromText("this is test doc 2", map[string]any{"purpose": "context", "ref": "doc2"}),
				},
			},
		},
		{
			name: "custom preface for middleware",
			opts: &ModelOptions{Supports: &ModelSupports{Context: false}},
			options: &AugmentWithContextOptions{
				Preface: func() *string {
					s := "\n\nCustom Preface : Use the following information to complete your task:\n\n"
					return &s
				}(),
			},
			input: &ModelRequest{
				Messages: []*Message{
					NewSystemTextMessage("You are a narrator."),
					NewUserTextMessage("Poem on star."),
				},
				Docs: []*Document{
					DocumentFromText("this is test doc 1", map[string]any{"purpose": "context", "ref": "doc1"}),
				},
			},
			expected: &ModelRequest{
				Messages: []*Message{
					NewSystemTextMessage("You are a narrator."),
					NewUserMessage(
						NewTextPart("Poem on star."),
						&Part{Text: "\n\nCustom Preface : Use the following information " +
							"to complete your task:\n\n- [doc1]: this is test doc 1\n\n",
							Metadata: map[string]any{"purpose": "context"}}),
				},
				Docs: []*Document{
					DocumentFromText("this is test doc 1", map[string]any{"purpose": "context", "ref": "doc1"}),
				},
			},
		},
		{
			name: "custom item template for middleware",
			opts: &ModelOptions{Supports: &ModelSupports{Context: false}},
			options: &AugmentWithContextOptions{
				ItemTemplate: func(d Document, index int, options *AugmentWithContextOptions) string {
					out := "- The new context is doc3"
					return out
				},
			},
			input: &ModelRequest{
				Messages: []*Message{
					NewSystemTextMessage("You are a narrator."),
					NewUserTextMessage("Poem on star."),
				},
				Docs: []*Document{
					DocumentFromText("this is test doc 1", map[string]any{"purpose": "context", "ref": "doc1"}),
				},
			},
			expected: &ModelRequest{
				Messages: []*Message{
					NewSystemTextMessage("You are a narrator."),
					{Role: "user", Content: []*Part{
						NewTextPart("Poem on star."),
						{Text: "\n\nUse the following information to complete your task:\n\n- The new context is doc3\n",
							Metadata: map[string]any{"purpose": "context"}}},
					},
				},
				Docs: []*Document{
					DocumentFromText("this is test doc 1", map[string]any{"purpose": "context", "ref": "doc1"}),
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			next := func(ctx context.Context, input *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				if !reflect.DeepEqual(input, tc.expected) {
					t.Errorf("Input messages were not modified as expected. got: %+v, want: %+v", input, tc.expected)
				}
				return &ModelResponse{}, nil
			}
			middleware := augmentWithContext(tc.opts, tc.options)
			_, _ = middleware(next)(context.Background(), tc.input, nil)
		})
	}
}
