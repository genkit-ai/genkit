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

package core

import (
	"github.com/firebase/genkit/go/core/api"
)

// Test-local define helpers: New* + Register in one call, mirroring the
// removed registry-taking Define functions so tests stay concise.

func defineAction[In, Out any](r api.Registry, name string, atype api.ActionType, metadata map[string]any, inputSchema map[string]any, fn Func[In, Out]) *Action[In, Out, struct{}] {
	a := NewActionOf(atype, name, &ActionOptions{Metadata: metadata, InputSchema: inputSchema}, fn)
	a.Register(r)
	return a
}

func defineStreamingAction[In, Out, Stream any](r api.Registry, name string, atype api.ActionType, metadata map[string]any, inputSchema map[string]any, fn StreamingFunc[In, Out, Stream]) *Action[In, Out, Stream] {
	a := NewStreamingActionOf(atype, name, &ActionOptions{Metadata: metadata, InputSchema: inputSchema}, fn)
	a.Register(r)
	return a
}

func defineBackgroundAction[In, Out any](r api.Registry, name string, atype api.ActionType, metadata map[string]any, startFn StartOpFunc[In, Out], checkFn CheckOpFunc[Out], cancelFn CancelOpFunc[Out]) *BackgroundAction[In, Out] {
	a := NewBackgroundActionOf(atype, name, &BackgroundActionOptions[In, Out]{Metadata: metadata, Check: checkFn, Cancel: cancelFn}, startFn)
	a.Register(r)
	return a
}

func defineFlow[In, Out any](r api.Registry, name string, fn Func[In, Out]) *Flow[In, Out, struct{}] {
	f := NewFlow(name, fn)
	f.Register(r)
	return f
}

func defineStreamingFlow[In, Out, Stream any](r api.Registry, name string, fn StreamingFunc[In, Out, Stream]) *Flow[In, Out, Stream] {
	f := NewStreamingFlow(name, fn)
	f.Register(r)
	return f
}
