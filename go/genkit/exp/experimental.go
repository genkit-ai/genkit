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

package exp

import (
	"fmt"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/internal/genkitbridge"
)

// requireExperimental panics unless g was initialized with
// [genkit.WithExperimental]. Every constructor in this package calls it before
// registering anything, so the experimental surface is unreachable until the
// caller has explicitly opted in. fn is the name of the calling constructor and
// is woven into the panic message to make it actionable.
func requireExperimental(g *genkit.Genkit, fn string) {
	reg := genkitbridge.RegistryOf(g)
	if enabled, _ := reg.LookupValue(api.ExperimentalKey).(bool); enabled {
		return
	}
	panic(fmt.Sprintf(
		"genkit/exp.%s: experimental features are not enabled. "+
			"Pass genkit.WithExperimental() to genkit.Init() before using the genkit/exp surface.\n\n"+
			"These features are in preview and/or under active development, so their APIs may have "+
			"breaking or backward-incompatible changes between minor releases.",
		fn))
}
