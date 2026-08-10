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

package core

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestWithActionContext(t *testing.T) {
	t.Run("adds action context to context", func(t *testing.T) {
		ctx := context.Background()
		actionCtx := ActionContext{
			"userId":    "user-123",
			"sessionId": "session-456",
		}

		newCtx := WithActionContext(ctx, actionCtx)

		retrieved := FromContext(newCtx)
		if diff := cmp.Diff(actionCtx, retrieved); diff != "" {
			t.Errorf("ActionContext mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("replaces existing action context", func(t *testing.T) {
		ctx := context.Background()
		first := ActionContext{"key": "first"}
		second := ActionContext{"key": "second"}

		ctx = WithActionContext(ctx, first)
		ctx = WithActionContext(ctx, second)

		retrieved := FromContext(ctx)
		if retrieved["key"] != "second" {
			t.Errorf("key = %v, want %q", retrieved["key"], "second")
		}
	})

	t.Run("allows nil action context", func(t *testing.T) {
		ctx := context.Background()
		newCtx := WithActionContext(ctx, nil)

		retrieved := FromContext(newCtx)
		if retrieved != nil {
			t.Errorf("expected nil, got %v", retrieved)
		}
	})
}

func TestFromContext(t *testing.T) {
	t.Run("returns nil when no action context", func(t *testing.T) {
		ctx := context.Background()
		retrieved := FromContext(ctx)

		if retrieved != nil {
			t.Errorf("expected nil, got %v", retrieved)
		}
	})

	t.Run("returns action context when present", func(t *testing.T) {
		ctx := context.Background()
		actionCtx := ActionContext{
			"requestId": "req-789",
		}
		ctx = WithActionContext(ctx, actionCtx)

		retrieved := FromContext(ctx)
		if retrieved["requestId"] != "req-789" {
			t.Errorf("requestId = %v, want %q", retrieved["requestId"], "req-789")
		}
	})

	t.Run("returns correct context from nested contexts", func(t *testing.T) {
		ctx := context.Background()
		actionCtx := ActionContext{"level": "root"}
		ctx = WithActionContext(ctx, actionCtx)

		// Create child context with deadline (doesn't affect action context)
		childCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		retrieved := FromContext(childCtx)
		if retrieved["level"] != "root" {
			t.Errorf("level = %v, want %q", retrieved["level"], "root")
		}
	})
}

func TestActionContextModification(t *testing.T) {
	t.Run("modifications to retrieved context affect original", func(t *testing.T) {
		ctx := context.Background()
		actionCtx := ActionContext{"mutable": "original"}
		ctx = WithActionContext(ctx, actionCtx)

		retrieved := FromContext(ctx)
		retrieved["mutable"] = "modified"

		// Check that modification affected the stored context
		// (maps are reference types, so this behavior is expected)
		secondRetrieval := FromContext(ctx)
		if secondRetrieval["mutable"] != "modified" {
			t.Errorf("mutable = %v, want %q", secondRetrieval["mutable"], "modified")
		}
	})
}
