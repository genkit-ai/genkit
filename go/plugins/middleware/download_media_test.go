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

package middleware

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/firebase/genkit/go/ai"
)

func TestDownloadRequestMediaRegisters(t *testing.T) {
	// NewMiddleware infers a config schema from the prototype; the Filter func
	// field (json:"-") must be skipped rather than panic schema inference.
	d := ai.NewMiddleware("download media", &DownloadRequestMedia{MaxBytes: 10})
	if want := provider + "/download-request-media"; d.Name != want {
		t.Fatalf("Name = %q, want %q", d.Name, want)
	}

	hooks, err := (&DownloadRequestMedia{}).New(context.Background())
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if hooks.WrapModel == nil {
		t.Fatal("expected a WrapModel hook")
	}
}

func TestDownloadRequestMedia(t *testing.T) {
	testCases := []struct {
		name           string
		input          *ai.ModelRequest
		mw             *DownloadRequestMedia
		setupServer    func() *httptest.Server
		expectedResult *ai.ModelRequest
	}{
		{
			name: "successful download",
			input: &ai.ModelRequest{
				Messages: []*ai.Message{
					ai.NewUserMessage(ai.NewMediaPart("image/png", "http://127.0.0.1:60289")),
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
			expectedResult: &ai.ModelRequest{
				Messages: []*ai.Message{
					ai.NewUserMessage(ai.NewMediaPart("image/png", "data:image/png;base64,dGVzdCBpbWFnZSBkYXRh")),
				},
			},
		},
		{
			name: "base64 media not to download",
			input: &ai.ModelRequest{
				Messages: []*ai.Message{
					ai.NewUserMessage(ai.NewMediaPart("image/png", "data:image/png;base64,dGVzdCBpbWFnZSBkYXRh")),
				},
			},
			expectedResult: &ai.ModelRequest{
				Messages: []*ai.Message{
					ai.NewUserMessage(ai.NewMediaPart("image/png", "data:image/png;base64,dGVzdCBpbWFnZSBkYXRh")),
				},
			},
		},
		{
			name: "filter applied not satisfied",
			input: &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Content: []*ai.Part{
							{
								ContentType: "image/png",
								Text:        "http://127.0.0.1:60289",
							},
						},
					},
				},
			},
			mw: &DownloadRequestMedia{
				Filter: func(part *ai.Part) bool {
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
			expectedResult: &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Content: []*ai.Part{
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
			input: &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Content: []*ai.Part{
							{
								ContentType: "image/png",
								Text:        "http://127.0.0.1:60289",
							},
						},
					},
				},
			},
			mw: &DownloadRequestMedia{
				Filter: func(part *ai.Part) bool {
					return false
				},
			},
			expectedResult: &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Content: []*ai.Part{
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
			next := func(ctx context.Context, input *ai.ModelRequest, _ any, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
				return &ai.ModelResponse{}, nil
			}
			mw := tc.mw
			if mw == nil {
				mw = &DownloadRequestMedia{}
			}
			_, err := mw.WrapModelFunc(next)(context.Background(), tc.input, nil, nil)

			if err != nil {
				t.Errorf("Expected no error, but got: %v", err)
			} else if !reflect.DeepEqual(tc.input, tc.expectedResult) {
				t.Errorf("Expected result: %v, but got: %v", tc.expectedResult, tc.input)
			}
		})
	}
}
