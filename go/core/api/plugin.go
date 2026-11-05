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
	"context"
)

// Plugin is the interface implemented by types that extend Genkit's functionality.
// Plugins are typically used to integrate external services like model providers,
// vector databases, or monitoring tools.
// They are registered and initialized via [WithPlugins] during [Init].
type Plugin interface {
	// Name returns the unique identifier for the plugin.
	// This name is used for registration and lookup.
	Name() string
	// Init initializes the plugin. It is called once during [Init].
	Init(ctx context.Context) []Action
}

// DynamicPlugin is a [Plugin] that can dynamically resolve actions.
type DynamicPlugin interface {
	Plugin
	// ListActions returns a list of action descriptors that the plugin is capable of resolving to [Action]s.
	ListActions(ctx context.Context) []ActionDesc
	// ResolveAction resolves an action type and ID to a [Action]. id is the
	// bare action ID: the registry has already parsed the key and matched the
	// provider to this plugin, so the prefix is gone by the time it arrives.
	//
	// The registry registers the returned action and then looks up the
	// requested key, so the action's Register may cover sibling keys beyond
	// the requested one. In particular, an action bundle (e.g. a background
	// action, whose Register covers its start, check, and cancel actions) is
	// the correct return value for a request for any of its component types;
	// do not hand-roll a standalone action per component type.
	ResolveAction(atype ActionType, id string) Action
}
