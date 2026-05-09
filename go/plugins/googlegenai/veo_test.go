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

package googlegenai

import (
	"context"
	"fmt"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"google.golang.org/genai"
)

func TestExtractTextFromRequest(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		request  *ai.ModelRequest
		expected string
	}{
		{
			name: "single text message",
			request: &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role: ai.RoleUser,
						Content: []*ai.Part{
							{Text: "Generate a video of a dancing robot"},
						},
					},
				},
			},
			expected: "Generate a video of a dancing robot",
		},
		{
			name: "multiple messages with text",
			request: &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role: ai.RoleUser,
						Content: []*ai.Part{
							{Text: "First message"},
						},
					},
					{
						Role: ai.RoleUser,
						Content: []*ai.Part{
							{Text: "Second message"},
						},
					},
				},
			},
			expected: "First message", // Should return the first text found
		},
		{
			name: "empty messages",
			request: &ai.ModelRequest{
				Messages: []*ai.Message{},
			},
			expected: "",
		},
		{
			name: "messages with empty text",
			request: &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role: ai.RoleUser,
						Content: []*ai.Part{
							{Text: ""}, // Empty text part
						},
					},
					{
						Role: ai.RoleUser,
						Content: []*ai.Part{
							{Text: "Valid text"},
						},
					},
				},
			},
			expected: "Valid text",
		},
		{
			name:     "nil request",
			request:  &ai.ModelRequest{Messages: nil},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractTextFromRequest(tt.request)
			if result != tt.expected {
				t.Errorf("extractTextFromRequest() = %q, want %q", result, tt.expected)
			}
		})
	}
}

func TestExtractVeoImageFromRequest(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		request  *ai.ModelRequest
		expected *genai.Image
	}{
		{
			name: "text-only request",
			request: &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role: ai.RoleUser,
						Content: []*ai.Part{
							{Text: "Generate a video"},
						},
					},
				},
			},
			expected: nil,
		},
		{
			name: "request with media (not yet implemented)",
			request: &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role: ai.RoleUser,
						Content: []*ai.Part{
							ai.NewMediaPart("image/jpeg", "data:image/jpeg;base64,/9j/4AAQ..."),
						},
					},
				},
			},
			expected: nil, // Currently returns nil as implementation is TODO
		},
		{
			name: "empty messages",
			request: &ai.ModelRequest{
				Messages: []*ai.Message{},
			},
			expected: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractVeoImageFromRequest(tt.request)
			if result != tt.expected {
				t.Errorf("extractVeoImageFromRequest() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestToVeoParameters(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		request     *ai.ModelRequest
		expected    genai.GenerateVideosConfig
		expectError bool
	}{
		{
			name: "request with no config",
			request: &ai.ModelRequest{
				Config: nil,
			},
			expected:    genai.GenerateVideosConfig{},
			expectError: false,
		},
		{
			name: "request with valid GenerateVideosConfig",
			request: &ai.ModelRequest{
				Config: &genai.GenerateVideosConfig{
					AspectRatio:      "16:9",
					DurationSeconds:  genai.Ptr(int32(5)),
					PersonGeneration: "allow_adult",
				},
			},
			expected: genai.GenerateVideosConfig{
				AspectRatio:      "16:9",
				DurationSeconds:  genai.Ptr(int32(5)),
				PersonGeneration: "allow_adult",
			},
			expectError: false,
		},
		{
			name: "request with valid map config",
			request: &ai.ModelRequest{
				Config: map[string]any{
					"aspectRatio":      "16:9",
					"durationSeconds":  5,
					"personGeneration": "allow_adult",
				},
			},
			expected: genai.GenerateVideosConfig{
				AspectRatio:      "16:9",
				DurationSeconds:  genai.Ptr(int32(5)),
				PersonGeneration: "allow_adult",
			},
			expectError: false,
		},
		{
			name: "request with different config type",
			request: &ai.ModelRequest{
				Config: &genai.GenerateContentConfig{
					MaxOutputTokens: int32(100),
				},
			},
			expected:    genai.GenerateVideosConfig{},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := toVeoParameters(tt.request)

			if tt.expectError {
				if err == nil {
					t.Errorf("toVeoParameters() expected error but got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("toVeoParameters() unexpected error: %v", err)
			}

			// Compare AspectRatio
			if result.AspectRatio != tt.expected.AspectRatio {
				t.Errorf("toVeoParameters() AspectRatio = %q, want %q", result.AspectRatio, tt.expected.AspectRatio)
			}

			// Compare DurationSeconds pointers
			if (result.DurationSeconds == nil) != (tt.expected.DurationSeconds == nil) {
				t.Errorf("toVeoParameters() DurationSeconds nil mismatch: got %v, want %v", result.DurationSeconds, tt.expected.DurationSeconds)
			} else if result.DurationSeconds != nil && tt.expected.DurationSeconds != nil {
				if *result.DurationSeconds != *tt.expected.DurationSeconds {
					t.Errorf("toVeoParameters() DurationSeconds = %v, want %v", *result.DurationSeconds, *tt.expected.DurationSeconds)
				}
			}

			// Compare PersonGeneration
			if result.PersonGeneration != tt.expected.PersonGeneration {
				t.Errorf("toVeoParameters() PersonGeneration = %q, want %q", result.PersonGeneration, tt.expected.PersonGeneration)
			}
		})
	}
}

func TestFromVeoOperation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name                 string
		veoOp                *genai.GenerateVideosOperation
		expectedID           string
		expectedDone         bool
		expectedError        error
		expectedHasOutput    bool
		expectedStatusMsg    string
		expectedFinishReason ai.FinishReason
		expectedMediaParts   int
	}{
		{
			name: "pending operation",
			veoOp: &genai.GenerateVideosOperation{
				Name: "operations/test-operation-123",
				Done: false,
			},
			expectedID:           "operations/test-operation-123",
			expectedDone:         false,
			expectedError:        nil,
			expectedHasOutput:    true,
			expectedStatusMsg:    "Video generation in progress...",
			expectedFinishReason: "",
			expectedMediaParts:   0,
		},
		{
			name: "completed operation with video",
			veoOp: &genai.GenerateVideosOperation{
				Name: "operations/test-operation-456",
				Done: true,
				Response: &genai.GenerateVideosResponse{
					GeneratedVideos: []*genai.GeneratedVideo{
						{
							Video: &genai.Video{
								URI: "https://storage.googleapis.com/test-bucket/video123.mp4",
							},
						},
					},
				},
			},
			expectedID:           "operations/test-operation-456",
			expectedDone:         true,
			expectedError:        nil,
			expectedHasOutput:    true,
			expectedStatusMsg:    "",
			expectedFinishReason: ai.FinishReasonStop,
			expectedMediaParts:   1,
		},
		{
			name: "operation with error",
			veoOp: &genai.GenerateVideosOperation{
				Name: "operations/test-operation-error",
				Done: true,
				Error: map[string]any{
					"message": "Video generation failed due to content policy",
					"code":    400,
				},
			},
			expectedID:           "operations/test-operation-error",
			expectedDone:         true,
			expectedError:        fmt.Errorf("%s", "Video generation failed due to content policy"),
			expectedHasOutput:    false,
			expectedStatusMsg:    "",
			expectedFinishReason: "",
			expectedMediaParts:   0,
		},
		{
			name: "operation with malformed error",
			veoOp: &genai.GenerateVideosOperation{
				Name: "operations/test-operation-bad-error",
				Done: true,
				Error: map[string]any{
					"code":    500,
					"details": "Internal error",
				},
			},
			expectedID:           "operations/test-operation-bad-error",
			expectedDone:         true,
			expectedError:        fmt.Errorf("operation error: map[code:500 details:Internal error]"),
			expectedHasOutput:    false,
			expectedStatusMsg:    "",
			expectedFinishReason: "",
			expectedMediaParts:   0,
		},
		{
			name: "completed operation with multiple videos",
			veoOp: &genai.GenerateVideosOperation{
				Name: "operations/test-operation-multi",
				Done: true,
				Response: &genai.GenerateVideosResponse{
					GeneratedVideos: []*genai.GeneratedVideo{
						{
							Video: &genai.Video{
								URI: "https://storage.googleapis.com/test-bucket/video1.mp4",
							},
						},
						{
							Video: &genai.Video{
								URI: "https://storage.googleapis.com/test-bucket/video2.mp4",
							},
						},
					},
				},
			},
			expectedID:           "operations/test-operation-multi",
			expectedDone:         true,
			expectedError:        nil,
			expectedHasOutput:    true,
			expectedStatusMsg:    "",
			expectedFinishReason: ai.FinishReasonStop,
			expectedMediaParts:   2,
		},
		{
			name: "completed operation without videos",
			veoOp: &genai.GenerateVideosOperation{
				Name: "operations/test-operation-no-videos",
				Done: true,
				Response: &genai.GenerateVideosResponse{
					GeneratedVideos: []*genai.GeneratedVideo{},
				},
			},
			expectedID:           "operations/test-operation-no-videos",
			expectedDone:         true,
			expectedError:        nil,
			expectedHasOutput:    true,
			expectedStatusMsg:    "Video generation completed but no videos were generated",
			expectedFinishReason: ai.FinishReasonStop,
			expectedMediaParts:   0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := fromVeoOperation(tt.veoOp)

			// Check basic operation fields
			if result.ID != tt.expectedID {
				t.Errorf("fromVeoOperation() ID = %q, want %q", result.ID, tt.expectedID)
			}

			if result.Done != tt.expectedDone {
				t.Errorf("fromVeoOperation() Done = %t, want %t", result.Done, tt.expectedDone)
			}

			// Check metadata is initialized
			if result.Metadata == nil {
				t.Error("fromVeoOperation() Metadata is nil, expected initialized map")
			}

			// Compare errors
			if (result.Error == nil) != (tt.expectedError == nil) {
				t.Errorf("fromVeoOperation() Error nil mismatch: got %v, want %v", result.Error, tt.expectedError)
			} else if result.Error != nil && tt.expectedError != nil {
				if result.Error.Error() != tt.expectedError.Error() {
					t.Errorf("fromVeoOperation() Error = %q, want %q", result.Error.Error(), tt.expectedError.Error())
				}
			}

			// Check output presence
			hasOutput := result.Output != nil
			if hasOutput != tt.expectedHasOutput {
				t.Errorf("fromVeoOperation() has output = %t, want %t", hasOutput, tt.expectedHasOutput)
			}

			// Only validate output structure if we expect output
			if !tt.expectedHasOutput {
				return
			}

			// Validate output structure
			if result.Output == nil {
				t.Fatal("fromVeoOperation() Output is nil, expected ModelResponse")
			}

			if result.Output.Message == nil {
				t.Fatal("fromVeoOperation() ModelResponse.Message is nil")
			}

			if result.Output.Message.Role != ai.RoleModel {
				t.Errorf("fromVeoOperation() Message.Role = %v, want %v", result.Output.Message.Role, ai.RoleModel)
			}

			if len(result.Output.Message.Content) == 0 {
				t.Fatal("fromVeoOperation() Message.Content is empty")
			}

			// Check status message for text-based responses
			if tt.expectedStatusMsg != "" {
				firstPart := result.Output.Message.Content[0]
				if firstPart.Text != tt.expectedStatusMsg {
					t.Errorf("fromVeoOperation() status message = %q, want %q", firstPart.Text, tt.expectedStatusMsg)
				}
			}

			// Check media parts count
			mediaCount := 0
			for _, part := range result.Output.Message.Content {
				if part.IsMedia() {
					mediaCount++
				}
			}
			if mediaCount != tt.expectedMediaParts {
				t.Errorf("fromVeoOperation() media parts count = %d, want %d", mediaCount, tt.expectedMediaParts)
			}

			// Check finish reason
			if result.Output.FinishReason != tt.expectedFinishReason {
				t.Errorf("fromVeoOperation() FinishReason = %v, want %v", result.Output.FinishReason, tt.expectedFinishReason)
			}
		})
	}
}

// TestCheckVeoOperation tests the checkVeoOperation function with a mock scenario
// Note: This is a basic structure test. In a real scenario, you'd need to mock the genai.Client
func TestCheckVeoOperationStructure(t *testing.T) {
	t.Parallel()

	// Test the function signature and basic structure
	ctx := context.Background()

	// This test verifies the function exists and has the correct signature
	// In a real test environment, you would mock the genai.Client and its methods

	ops := &core.Operation[*ai.ModelResponse]{
		ID:   "operations/test-123",
		Done: false,
	}

	// We can't easily test this without mocking the genai.Client
	// But we can verify the function exists and the parameters are correctly typed
	var client *genai.Client = nil // This would be a mock in real tests

	if client != nil {
		_, err := checkVeoOperation(ctx, client, ops)
		if err != nil {
			t.Logf("checkVeoOperation returned expected error with nil client: %v", err)
		}
	} else {
		t.Log("checkVeoOperation function structure test passed (mock client would be needed for full test)")
	}
}
