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
	"github.com/google/dotprompt/go/dotprompt"
)

// Registry holds all registered actions and associated types,
// and provides methods to register, query, and look up actions.
//
// For internal use only. API may change without notice.
type Registry interface {
	// NewChild creates a new child registry that inherits from this registry.
	// Child registries are cheap to create and will fall back to the parent
	// for lookups if a value is not found in the child.
	NewChild() Registry

	// IsChild returns true if the registry is a child of another registry.
	IsChild() bool

	// RegisterPlugin records the plugin in the registry.
	// It panics if a plugin with the same name is already registered.
	RegisterPlugin(name string, p Plugin)

	// RegisterAction records the action in the registry.
	// It panics if an action with the same type, provider and name is already
	// registered.
	RegisterAction(key string, action Action)

	// RegisterValue records an arbitrary value in the registry.
	// It panics if a value with the same name is already registered.
	RegisterValue(name string, value any)

	// RegisterSchema records a JSON schema in the registry.
	// It panics if a value with the same name is already registered.
	RegisterSchema(name string, schema map[string]any)

	// LookupPlugin returns the plugin for the given name.
	// It first checks the current registry, then falls back to the parent if not found.
	// Returns nil if the plugin is not found in the registry hierarchy.
	LookupPlugin(name string) Plugin

	// LookupAction returns the action for the given key.
	// It first checks the current registry, then falls back to the parent if not found.
	LookupAction(key string) Action

	// LookupValue returns the value for the given name.
	// It first checks the current registry, then falls back to the parent if not found.
	// Returns nil if the value is not found in the registry hierarchy.
	LookupValue(name string) any

	// LookupSchema returns a JSON schema for the given name.
	// It first checks the current registry, then falls back to the parent if not found.
	// Returns nil if the value is not found in the registry hierachy.
	LookupSchema(name string) map[string]any

	// ResolveAction looks up an action by key. If the action is not found, it attempts dynamic resolution.
	// Returns the action if found, or nil if not found.
	ResolveAction(key string) Action

	// ListActions returns a list of all registered actions.
	// This includes actions from both the current registry and its parent hierarchy.
	// Child registry actions take precedence over parent actions with the same key.
	ListActions() []Action

	// ListPlugins returns a list of all registered plugins.
	ListPlugins() []Plugin

	// ListValues returns a list of values of all registered values.
	// This includes values from both the current registry and its parent hierarchy.
	// Child registry values take precedence over parent values with the same key.
	ListValues() map[string]any

	// RegisterPartial adds the partial to the list of partials to the dotprompt instance
	RegisterPartial(name string, source string)

	// RegisterHelper adds a helper function to the dotprompt instance
	RegisterHelper(name string, fn any)

	// Dotprompt returns a clone of the Dotprompt instance.
	Dotprompt() *dotprompt.Dotprompt
}
