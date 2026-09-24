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

// usageSinkKey holds the callback that receives the usage of each model call
// made under a context. The agent runtime installs one per invocation to total
// usage by turn, invocation, and session; go/ai reports to it. The value is
// type-erased for the same reason as [WithPromptState]: the runtime lives in a
// package that imports go/ai, so go/ai cannot name its types.
var usageSinkKey = NewContextKey[func(usage any)]()

// WithUsageSink returns ctx carrying sink, which receives the usage of each
// model call made under ctx: a non-nil *ai.GenerationUsage, called from
// whatever goroutine made the call. It replaces any sink ctx already carries,
// so a nested agent counts only its own calls.
func WithUsageSink(ctx context.Context, sink func(usage any)) context.Context {
	return usageSinkKey.NewContext(ctx, sink)
}

// UsageSinkFromContext returns the sink attached by [WithUsageSink], or nil.
func UsageSinkFromContext(ctx context.Context) func(usage any) {
	return usageSinkKey.FromContext(ctx)
}
