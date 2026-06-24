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

package base

import "context"

// promptStateKey holds a getter for the type-erased state that prompt rendering
// injects into templates as {{@state}}. The getter indirection lets a
// higher-level package publish its session state for prompt rendering without a
// circular import: the session lives in a package that imports go/ai, so go/ai
// cannot import it back to read the state directly.
var promptStateKey = NewContextKey[func() any]()

// WithPromptState returns ctx carrying a getter for the state exposed to prompt
// templates via {{@state}}. getState is evaluated lazily at render time, so it
// observes the latest state rather than a snapshot taken when the context was
// built. A nil getState detaches any state previously attached.
func WithPromptState(ctx context.Context, getState func() any) context.Context {
	return promptStateKey.NewContext(ctx, getState)
}

// PromptStateFromContext returns the state attached by [WithPromptState], or nil
// if none is attached. The getter is invoked on each call.
func PromptStateFromContext(ctx context.Context) any {
	getState := promptStateKey.FromContext(ctx)
	if getState == nil {
		return nil
	}
	return getState()
}
