// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai_test

import (
	"context"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/plugins/googlegenai"
)

// TestVertexAIInit_InvalidAPIVersion verifies that a bad APIVersion is
// rejected before any credential detection or network access is attempted,
// so the failure is immediate and clear rather than a confusing 404 at
// request time.
func TestVertexAIInit_InvalidAPIVersion(t *testing.T) {
	v := &googlegenai.VertexAI{
		ProjectID:  "test-project",
		Location:   "us-central1",
		APIVersion: "v1beta", // invalid: valid values are "v1" and "v1beta1"
	}

	var recovered any
	func() {
		defer func() {
			recovered = recover()
		}()
		v.Init(context.Background())
	}()

	if recovered == nil {
		t.Fatal("expected Init to panic on invalid APIVersion, but it did not")
	}
	msg, ok := recovered.(string)
	if !ok {
		t.Fatalf("expected panic value to be a string, got %T: %v", recovered, recovered)
	}
	if !strings.Contains(msg, "v1beta") {
		t.Errorf("panic message %q does not mention the invalid value", msg)
	}
}
