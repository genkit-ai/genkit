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

package genkit

import (
	"context"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/genkitbridge"
)

// Expose first-party hooks into a *Genkit to subpackages (genkit/exp) without
// adding public accessors. genkitbridge lives under go/internal, so only code
// inside the Genkit module can read it. See [genkitbridge.RegistryOf] and
// [genkitbridge.SeedContext].
func init() {
	genkitbridge.RegistryOf = func(host any) api.Registry {
		return host.(*Genkit).reg
	}
	genkitbridge.SeedContext = func(ctx context.Context, host any) context.Context {
		return genkitCtxKey.NewContext(ctx, host.(*Genkit))
	}
}
