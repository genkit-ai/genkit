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

package registry

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"maps"
	"sync"

	"github.com/firebase/genkit/go/core/api"
	"github.com/google/dotprompt/go/dotprompt"
	"github.com/invopop/jsonschema"
)

// This file implements registries of actions and other values.

// Registry holds all registered actions and associated types,
// and provides methods to register, query, and look up actions.
type Registry struct {
	mu        sync.RWMutex
	resolveMu sync.RWMutex
	parent    api.Registry
	actions   map[string]api.Action
	plugins   map[string]api.Plugin
	schemas   map[string]map[string]any
	values    map[string]any // Values can truly be anything.
	dotprompt *dotprompt.Dotprompt
}

// New creates a new root registry.
func New() *Registry {
	r := &Registry{
		actions: map[string]api.Action{},
		plugins: map[string]api.Plugin{},
		values:  map[string]any{},
		schemas: make(map[string]map[string]any),
	}
	r.dotprompt = dotprompt.NewDotprompt(&dotprompt.DotpromptOptions{
		Helpers:  make(map[string]any),
		Partials: make(map[string]string),
		SchemaResolver: func(name string) (*jsonschema.Schema, error) {
			s := r.LookupSchema(name)
			if s == nil {
				// Schema not found - return a reference schema with just the name.
				// The caller (LoadPrompt) will detect this and store the name
				// for lazy resolution at execution time.
				return &jsonschema.Schema{
					Ref: name,
				}, nil
			}
			var schema jsonschema.Schema
			schemaBytes, err := json.Marshal(s)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal schema map for %q: %w", name, err)
			}
			if err := json.Unmarshal(schemaBytes, &schema); err != nil {
				slog.Error("failed to unmarshal schema from registry", "name", name, "error", err)
				return nil, fmt.Errorf("failed to unmarshal schema(%q) from registry", name)
			}
			return &schema, nil
		},
	})
	return r
}

// NewChild creates a new child registry that inherits from this registry.
// Child registries are cheap to create and will fall back to the parent
// for lookups if a value is not found in the child.
func (r *Registry) NewChild() api.Registry {
	child := &Registry{
		parent:    r,
		actions:   map[string]api.Action{},
		plugins:   map[string]api.Plugin{},
		values:    map[string]any{},
		schemas:   make(map[string]map[string]any),
		dotprompt: r.dotprompt,
	}
	return child
}

// IsChild returns true if the registry is a child of another registry.
func (r *Registry) IsChild() bool {
	return r.parent != nil
}

// RegisterPlugin records the plugin in the registry.
// It panics if a plugin with the same name is already registered.
func (r *Registry) RegisterPlugin(name string, p api.Plugin) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.plugins[name]; ok {
		panic(fmt.Sprintf("plugin %q is already registered", name))
	}
	r.plugins[name] = p
	slog.Debug("RegisterPlugin", "name", name)
}

// RegisterAction records the action in the registry.
// It panics if an action with the same type, provider and name is already
// registered.
func (r *Registry) RegisterAction(key string, action api.Action) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.actions[key]; ok {
		panic(fmt.Sprintf("action %q is already registered", key))
	}
	r.actions[key] = action
	slog.Debug("RegisterAction", "key", key)
}

// RegisterSchema records a JSON schema (as a map[string]any) in the registry.
// It panics if a schema with the same name is already registered.
func (r *Registry) RegisterSchema(name string, schema map[string]any) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.schemas[name]; ok {
		panic(fmt.Sprintf("schema %q is already registered", name))
	}
	r.schemas[name] = schema
	slog.Debug("RegisterSchema", "name", name)
}

// LookupSchema returns a JSON schema (as a map[string]any) for the given name.
// It first checks the current registry, then falls back to the parent if not found.
// Returns nil if the schema is not found in the registry hierarchy.
func (r *Registry) LookupSchema(name string) map[string]any {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if schema, ok := r.schemas[name]; ok {
		return schema
	}

	if r.parent != nil {
		return r.parent.LookupSchema(name)
	}

	return nil
}

// LookupPlugin returns the plugin for the given name.
// It first checks the current registry, then falls back to the parent if not found.
// Returns nil if the plugin is not found in the registry hierarchy.
func (r *Registry) LookupPlugin(name string) api.Plugin {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if plugin, ok := r.plugins[name]; ok {
		return plugin
	}

	if r.parent != nil {
		return r.parent.LookupPlugin(name)
	}

	return nil
}

// RegisterValue records an arbitrary value in the registry.
// It panics if a value with the same name is already registered.
func (r *Registry) RegisterValue(name string, value any) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.values[name]; ok {
		panic(fmt.Sprintf("value %q is already registered", name))
	}
	r.values[name] = value
	slog.Debug("RegisterValue", "name", name)
}

// LookupValue returns the value for the given name.
// It first checks the current registry, then falls back to the parent if not found.
// Returns nil if the value is not found in the registry hierarchy.
func (r *Registry) LookupValue(name string) any {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if value, ok := r.values[name]; ok {
		return value
	}

	if r.parent != nil {
		return r.parent.LookupValue(name)
	}

	return nil
}

// LookupAction returns the action for the given key.
// It first checks the current registry, then falls back to the parent if not found.
func (r *Registry) LookupAction(key string) api.Action {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if action, ok := r.actions[key]; ok {
		return action
	}

	if r.parent != nil {
		return r.parent.LookupAction(key)
	}

	return nil
}

// ResolveAction looks up an action by key. If the action is not found, it attempts dynamic resolution.
// Returns the action if found, or nil if not found.
// This method is safe to call concurrently and uses a single mutex to serialize all resolution operations.
func (r *Registry) ResolveAction(key string) api.Action {
	action := r.LookupAction(key)
	if action != nil {
		return action
	}

	r.resolveMu.Lock()
	defer r.resolveMu.Unlock()

	action = r.LookupAction(key)
	if action != nil {
		return action
	}

	typ, provider, name := api.ParseKey(key)
	if typ == "" || name == "" {
		slog.Debug("ResolveAction: failed to parse action key", "key", key)
		return nil
	}

	plugins := r.ListPlugins()
	for _, plugin := range plugins {
		if dp, ok := plugin.(api.DynamicPlugin); ok && dp.Name() == provider {
			resolvedAction := dp.ResolveAction(typ, name)
			if resolvedAction != nil {
				resolvedAction.Register(r)
			}
			break
		}
	}

	return r.LookupAction(key)
}

// ListActions returns a list of all registered actions.
// This includes actions from both the current registry and its parent hierarchy.
// Child registry actions take precedence over parent actions with the same key.
func (r *Registry) ListActions() []api.Action {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var actions []api.Action

	// recursively check all the registry parents
	if r.parent != nil {
		parentValues := r.parent.ListActions()
		for _, pv := range parentValues {
			found := false
			for _, cv := range r.actions {
				if pv.Name() == cv.Name() {
					found = true
					break
				}
			}
			if !found {
				actions = append(actions, pv)
			}
		}
	}
	for _, v := range r.actions {
		actions = append(actions, v)
	}
	return actions
}

// ListPlugins returns a list of all registered plugins.
// This includes plugins from both the current registry and its parent hierarchy.
// Child registry plugins take precedence over parent plugins with the same key.
func (r *Registry) ListPlugins() []api.Plugin {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var plugins []api.Plugin

	// recursively check all the registry parents
	if r.parent != nil {
		parentValues := r.parent.ListPlugins()
		for _, pv := range parentValues {
			found := false
			for _, cv := range r.plugins {
				if pv.Name() == cv.Name() {
					found = true
					break
				}
			}
			if !found {
				plugins = append(plugins, pv)
			}
		}
	}

	for _, p := range r.plugins {
		plugins = append(plugins, p)
	}
	return plugins
}

// ListValues returns a list of values of all registered values.
// This includes values from both the current registry and its parent hierarchy.
// Child registry values take precedence over parent values with the same key.
func (r *Registry) ListValues() map[string]any {
	r.mu.RLock()
	defer r.mu.RUnlock()

	allValues := make(map[string]any)

	if r.parent != nil {
		parentValues := r.parent.ListValues()
		maps.Copy(allValues, parentValues)
	}

	maps.Copy(allValues, r.values)

	return allValues
}

// RegisterPartial adds the partial to the list of partials to the dotprompt instance
func (r *Registry) RegisterPartial(name string, source string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.dotprompt == nil {
		r.dotprompt = dotprompt.NewDotprompt(nil)
	}
	if r.dotprompt.Partials == nil {
		r.dotprompt.Partials = make(map[string]string)
	}
	if r.dotprompt.Partials[name] != "" {
		panic(fmt.Sprintf("partial %q is already defined", name))
	}
	r.dotprompt.Partials[name] = source
}

// RegisterHelper adds a helper function to the dotprompt instance
func (r *Registry) RegisterHelper(name string, fn any) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.dotprompt == nil {
		r.dotprompt = dotprompt.NewDotprompt(nil)
	}
	if r.dotprompt.Helpers == nil {
		r.dotprompt.Helpers = make(map[string]any)
	}
	if r.dotprompt.Helpers[name] != nil {
		panic(fmt.Sprintf("helper %q is already defined", name))
	}
	r.dotprompt.Helpers[name] = fn
}

// Dotprompt returns a clone of the Dotprompt instance.
func (r *Registry) Dotprompt() *dotprompt.Dotprompt {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.dotprompt.Clone()
}
