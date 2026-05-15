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

package api

import (
	"fmt"
	"strings"
)

// KeyFromName creates a new action key from an action type and a name string (which may include a provider).
func KeyFromName(typ ActionType, name string) string {
	provider, id := ParseName(name)
	return NewKey(typ, provider, id)
}

// NewKey creates a new action key for the given type, provider, and name.
func NewKey(typ ActionType, provider, id string) string {
	if provider != "" {
		return fmt.Sprintf("/%s/%s/%s", typ, provider, id)
	}
	return fmt.Sprintf("/%s/%s", typ, id)
}

// ParseKey parses an action key into a type, provider, and name.
func ParseKey(key string) (ActionType, string, string) {
	parts := strings.Split(key, "/")
	if len(parts) < 4 || parts[0] != "" {
		// Return empty values if the key doesn't have the expected format
		return "", "", ""
	}
	id := strings.Join(parts[3:], "/")
	return ActionType(parts[1]), parts[2], id
}

// NewName creates a new action name for the given provider and id.
func NewName(provider, id string) string {
	if provider != "" {
		return fmt.Sprintf("%s/%s", provider, id)
	}
	return id
}

// ParseName parses an action name into a provider and id.
func ParseName(name string) (string, string) {
	parts := strings.Split(name, "/")
	if len(parts) < 2 {
		return "", name
	}
	id := strings.Join(parts[1:], "/")
	return parts[0], id
}
