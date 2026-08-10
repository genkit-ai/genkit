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

import "maps"

// NewModelRequest create a new ModelRequest with provided config and
// messages.
func NewModelRequest(config any, messages ...*Message) *ModelRequest {
	return &ModelRequest{
		Config:   config,
		Messages: messages,
	}
}

// NewUserMessageWithMetadata creates a new Message with role "user" with provided metadata and  parts.
// Use NewUserTextMessage if you have a text-only message.
func NewUserMessageWithMetadata(metadata map[string]any, parts ...*Part) *Message {
	return NewMessage(RoleUser, metadata, parts...)
}

// NewUserMessage creates a new Message with role "user" and provided parts.
// Use NewUserTextMessage if you have a text-only message.
func NewUserMessage(parts ...*Part) *Message {
	return NewMessage(RoleUser, nil, parts...)
}

// NewUserTextMessage creates a new Message with role "user" and content with
// a single text part with the content of provided text.
func NewUserTextMessage(text string) *Message {
	return NewTextMessage(RoleUser, text)
}

// NewModelMessage creates a new Message with role "model" and provided parts.
// Use NewModelTextMessage if you have a text-only message.
func NewModelMessage(parts ...*Part) *Message {
	return NewMessage(RoleModel, nil, parts...)
}

// NewModelTextMessage creates a new Message with role "model" and content with
// a single text part with the content of provided text.
func NewModelTextMessage(text string) *Message {
	return NewTextMessage(RoleModel, text)
}

// NewSystemMessage creates a new Message with role "system" and provided parts.
// Use NewSystemTextMessage if you have a text-only message.
func NewSystemMessage(parts ...*Part) *Message {
	return NewMessage(RoleSystem, nil, parts...)
}

// NewSystemTextMessage creates a new Message with role "system" and content with
// a single text part with the content of provided text.
func NewSystemTextMessage(text string) *Message {
	return NewTextMessage(RoleSystem, text)
}

// NewMessage creates a new Message with the provided role, metadata and parts.
// Use NewTextMessage if you have a text-only message.
func NewMessage(role Role, metadata map[string]any, parts ...*Part) *Message {
	return &Message{
		Role:     role,
		Content:  parts,
		Metadata: metadata,
	}
}

// NewTextMessage creates a new Message with the provided role and content with
// a single part containint provided text.
func NewTextMessage(role Role, text string) *Message {
	return &Message{
		Role:    role,
		Content: []*Part{NewTextPart(text)},
	}
}

// WithCacheTTL adds cache TTL configuration for the desired message
func (m *Message) WithCacheTTL(ttlSeconds int) *Message {
	metadata := make(map[string]any)

	if m.Metadata != nil {
		metadata = m.Metadata
	}

	cache := map[string]any{
		"cache": map[string]any{
			"ttlSeconds": ttlSeconds,
		},
	}
	maps.Copy(metadata, cache)

	return &Message{
		Content:  m.Content,
		Role:     m.Role,
		Metadata: metadata,
	}
}

// WithCacheName adds cache name to use in the generate request
func (m *Message) WithCacheName(n string) *Message {
	metadata := make(map[string]any)

	if m.Metadata != nil {
		metadata = m.Metadata
	}

	cache := map[string]any{
		"cache": map[string]any{
			"name": n,
		},
	}

	maps.Copy(metadata, cache)

	return &Message{
		Content:  m.Content,
		Role:     m.Role,
		Metadata: metadata,
	}
}
