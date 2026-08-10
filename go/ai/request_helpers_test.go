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
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestNewModelRequest(t *testing.T) {
	t.Run("creates request with config and messages", func(t *testing.T) {
		config := &GenerationCommonConfig{Temperature: 0.7}
		msg1 := NewUserTextMessage("hello")
		msg2 := NewModelTextMessage("hi there")

		req := NewModelRequest(config, msg1, msg2)

		if req.Config != config {
			t.Error("Config not set correctly")
		}
		if len(req.Messages) != 2 {
			t.Errorf("len(Messages) = %d, want 2", len(req.Messages))
		}
		if req.Messages[0] != msg1 {
			t.Error("First message not set correctly")
		}
		if req.Messages[1] != msg2 {
			t.Error("Second message not set correctly")
		}
	})

	t.Run("creates request with nil config", func(t *testing.T) {
		msg := NewUserTextMessage("hello")
		req := NewModelRequest(nil, msg)

		if req.Config != nil {
			t.Errorf("Config = %v, want nil", req.Config)
		}
		if len(req.Messages) != 1 {
			t.Errorf("len(Messages) = %d, want 1", len(req.Messages))
		}
	})

	t.Run("creates request with no messages", func(t *testing.T) {
		config := map[string]any{"temp": 0.5}
		req := NewModelRequest(config)

		if req.Config == nil {
			t.Error("Config should not be nil")
		}
		if len(req.Messages) != 0 {
			t.Errorf("len(Messages) = %d, want 0", len(req.Messages))
		}
	})
}

func TestNewUserMessage(t *testing.T) {
	t.Run("creates user message with parts", func(t *testing.T) {
		parts := []*Part{NewTextPart("text"), NewMediaPart("image/png", "data:...")}
		msg := NewUserMessage(parts...)

		if msg.Role != RoleUser {
			t.Errorf("Role = %q, want %q", msg.Role, RoleUser)
		}
		if len(msg.Content) != 2 {
			t.Errorf("len(Content) = %d, want 2", len(msg.Content))
		}
		if msg.Metadata != nil {
			t.Errorf("Metadata = %v, want nil", msg.Metadata)
		}
	})

	t.Run("creates user message with no parts", func(t *testing.T) {
		msg := NewUserMessage()

		if msg.Role != RoleUser {
			t.Errorf("Role = %q, want %q", msg.Role, RoleUser)
		}
		if len(msg.Content) != 0 {
			t.Errorf("len(Content) = %d, want 0", len(msg.Content))
		}
	})
}

func TestNewUserMessageWithMetadata(t *testing.T) {
	t.Run("creates user message with metadata", func(t *testing.T) {
		metadata := map[string]any{"purpose": "context"}
		parts := []*Part{NewTextPart("text")}
		msg := NewUserMessageWithMetadata(metadata, parts...)

		if msg.Role != RoleUser {
			t.Errorf("Role = %q, want %q", msg.Role, RoleUser)
		}
		if diff := cmp.Diff(metadata, msg.Metadata); diff != "" {
			t.Errorf("Metadata mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("creates user message with nil metadata", func(t *testing.T) {
		msg := NewUserMessageWithMetadata(nil, NewTextPart("text"))

		if msg.Role != RoleUser {
			t.Errorf("Role = %q, want %q", msg.Role, RoleUser)
		}
		if msg.Metadata != nil {
			t.Errorf("Metadata = %v, want nil", msg.Metadata)
		}
	})
}

func TestNewUserTextMessage(t *testing.T) {
	t.Run("creates text message with user role", func(t *testing.T) {
		msg := NewUserTextMessage("hello world")

		if msg.Role != RoleUser {
			t.Errorf("Role = %q, want %q", msg.Role, RoleUser)
		}
		if len(msg.Content) != 1 {
			t.Fatalf("len(Content) = %d, want 1", len(msg.Content))
		}
		if msg.Content[0].Text != "hello world" {
			t.Errorf("Text = %q, want %q", msg.Content[0].Text, "hello world")
		}
	})

	t.Run("creates text message with empty string", func(t *testing.T) {
		msg := NewUserTextMessage("")

		if msg.Role != RoleUser {
			t.Errorf("Role = %q, want %q", msg.Role, RoleUser)
		}
		if len(msg.Content) != 1 {
			t.Fatalf("len(Content) = %d, want 1", len(msg.Content))
		}
		if msg.Content[0].Text != "" {
			t.Errorf("Text = %q, want empty string", msg.Content[0].Text)
		}
	})
}

func TestNewModelMessage(t *testing.T) {
	t.Run("creates model message with parts", func(t *testing.T) {
		parts := []*Part{NewTextPart("response")}
		msg := NewModelMessage(parts...)

		if msg.Role != RoleModel {
			t.Errorf("Role = %q, want %q", msg.Role, RoleModel)
		}
		if len(msg.Content) != 1 {
			t.Errorf("len(Content) = %d, want 1", len(msg.Content))
		}
	})
}

func TestNewModelTextMessage(t *testing.T) {
	t.Run("creates text message with model role", func(t *testing.T) {
		msg := NewModelTextMessage("model response")

		if msg.Role != RoleModel {
			t.Errorf("Role = %q, want %q", msg.Role, RoleModel)
		}
		if len(msg.Content) != 1 {
			t.Fatalf("len(Content) = %d, want 1", len(msg.Content))
		}
		if msg.Content[0].Text != "model response" {
			t.Errorf("Text = %q, want %q", msg.Content[0].Text, "model response")
		}
	})
}

func TestNewSystemMessage(t *testing.T) {
	t.Run("creates system message with parts", func(t *testing.T) {
		parts := []*Part{NewTextPart("system instruction")}
		msg := NewSystemMessage(parts...)

		if msg.Role != RoleSystem {
			t.Errorf("Role = %q, want %q", msg.Role, RoleSystem)
		}
		if len(msg.Content) != 1 {
			t.Errorf("len(Content) = %d, want 1", len(msg.Content))
		}
	})
}

func TestNewSystemTextMessage(t *testing.T) {
	t.Run("creates text message with system role", func(t *testing.T) {
		msg := NewSystemTextMessage("be helpful")

		if msg.Role != RoleSystem {
			t.Errorf("Role = %q, want %q", msg.Role, RoleSystem)
		}
		if len(msg.Content) != 1 {
			t.Fatalf("len(Content) = %d, want 1", len(msg.Content))
		}
		if msg.Content[0].Text != "be helpful" {
			t.Errorf("Text = %q, want %q", msg.Content[0].Text, "be helpful")
		}
	})
}

func TestNewMessage(t *testing.T) {
	t.Run("creates message with all fields", func(t *testing.T) {
		metadata := map[string]any{"key": "value"}
		parts := []*Part{NewTextPart("content")}
		msg := NewMessage(RoleTool, metadata, parts...)

		if msg.Role != RoleTool {
			t.Errorf("Role = %q, want %q", msg.Role, RoleTool)
		}
		if diff := cmp.Diff(metadata, msg.Metadata); diff != "" {
			t.Errorf("Metadata mismatch (-want +got):\n%s", diff)
		}
		if len(msg.Content) != 1 {
			t.Errorf("len(Content) = %d, want 1", len(msg.Content))
		}
	})
}

func TestNewTextMessage(t *testing.T) {
	t.Run("creates text message with specified role", func(t *testing.T) {
		msg := NewTextMessage(RoleTool, "tool output")

		if msg.Role != RoleTool {
			t.Errorf("Role = %q, want %q", msg.Role, RoleTool)
		}
		if len(msg.Content) != 1 {
			t.Fatalf("len(Content) = %d, want 1", len(msg.Content))
		}
		if msg.Content[0].Text != "tool output" {
			t.Errorf("Text = %q, want %q", msg.Content[0].Text, "tool output")
		}
	})
}

func TestWithCacheTTL(t *testing.T) {
	t.Run("adds cache TTL to message without existing metadata", func(t *testing.T) {
		original := NewUserTextMessage("hello")
		result := original.WithCacheTTL(3600)

		// Original should be unchanged
		if original.Metadata != nil {
			t.Error("original message metadata should be nil")
		}

		// Result should have cache metadata
		if result.Metadata == nil {
			t.Fatal("result metadata should not be nil")
		}
		cache, ok := result.Metadata["cache"].(map[string]any)
		if !ok {
			t.Fatal("cache metadata not found or wrong type")
		}
		if cache["ttlSeconds"] != 3600 {
			t.Errorf("ttlSeconds = %v, want 3600", cache["ttlSeconds"])
		}

		// Content and role should be preserved
		if result.Role != original.Role {
			t.Errorf("Role changed: got %q, want %q", result.Role, original.Role)
		}
		if len(result.Content) != len(original.Content) {
			t.Errorf("Content length changed")
		}
	})

	t.Run("adds cache TTL to message with existing metadata", func(t *testing.T) {
		original := NewUserMessageWithMetadata(
			map[string]any{"existing": "value"},
			NewTextPart("hello"),
		)
		result := original.WithCacheTTL(1800)

		// Result should have both existing and cache metadata
		if result.Metadata["existing"] != "value" {
			t.Error("existing metadata not preserved")
		}
		cache, ok := result.Metadata["cache"].(map[string]any)
		if !ok {
			t.Fatal("cache metadata not found")
		}
		if cache["ttlSeconds"] != 1800 {
			t.Errorf("ttlSeconds = %v, want 1800", cache["ttlSeconds"])
		}
	})

	t.Run("chained with WithCacheName", func(t *testing.T) {
		msg := NewUserTextMessage("hello").
			WithCacheTTL(3600).
			WithCacheName("my-cache")

		cache, ok := msg.Metadata["cache"].(map[string]any)
		if !ok {
			t.Fatal("cache metadata not found")
		}
		// Note: second call overwrites the cache object
		if cache["name"] != "my-cache" {
			t.Errorf("cache name = %v, want %q", cache["name"], "my-cache")
		}
	})
}

func TestWithCacheName(t *testing.T) {
	t.Run("adds cache name to message without existing metadata", func(t *testing.T) {
		original := NewUserTextMessage("hello")
		result := original.WithCacheName("my-cache")

		// Original should be unchanged
		if original.Metadata != nil {
			t.Error("original message metadata should be nil")
		}

		// Result should have cache metadata
		if result.Metadata == nil {
			t.Fatal("result metadata should not be nil")
		}
		cache, ok := result.Metadata["cache"].(map[string]any)
		if !ok {
			t.Fatal("cache metadata not found or wrong type")
		}
		if cache["name"] != "my-cache" {
			t.Errorf("name = %v, want %q", cache["name"], "my-cache")
		}
	})

	t.Run("adds cache name to message with existing metadata", func(t *testing.T) {
		original := NewUserMessageWithMetadata(
			map[string]any{"existing": "value"},
			NewTextPart("hello"),
		)
		result := original.WithCacheName("another-cache")

		// Result should have both existing and cache metadata
		if result.Metadata["existing"] != "value" {
			t.Error("existing metadata not preserved")
		}
		cache, ok := result.Metadata["cache"].(map[string]any)
		if !ok {
			t.Fatal("cache metadata not found")
		}
		if cache["name"] != "another-cache" {
			t.Errorf("name = %v, want %q", cache["name"], "another-cache")
		}
	})

	t.Run("with empty name", func(t *testing.T) {
		msg := NewUserTextMessage("hello").WithCacheName("")

		cache, ok := msg.Metadata["cache"].(map[string]any)
		if !ok {
			t.Fatal("cache metadata not found")
		}
		if cache["name"] != "" {
			t.Errorf("name = %v, want empty string", cache["name"])
		}
	})
}
