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

// Package genkitbridge is an internal bridge that lets first-party Genkit
// subpackages (notably genkit/exp) reach the [api.Registry] backing a
// *genkit.Genkit without the genkit package exposing a public accessor.
//
// The genkit package installs [RegistryOf] from its init; genkit/exp calls it
// to hand the registry to the registry-level constructors in ai/exp. Because
// this package lives under go/internal, code outside the Genkit module cannot
// import it, so registry access stays first-party only.
package genkitbridge

import (
	"context"

	"github.com/firebase/genkit/go/core/api"
)

// RegistryOf returns the [api.Registry] backing host, which must be a
// *genkit.Genkit. It is installed by the genkit package's init.
//
// The host parameter is typed as any rather than *genkit.Genkit to keep this
// package free of an import cycle with genkit (genkit imports this package to
// install the extractor). First-party callers always pass a *genkit.Genkit.
var RegistryOf func(host any) api.Registry

// SeedContextForRegistry returns ctx with the *genkit.Genkit backing reg
// attached, so it can be retrieved with genkit.FromContext. It is installed by
// the genkit package's init. ai/exp's agent constructors call it to seed the
// Genkit instance into every agent turn, and ai's generate action calls it to
// seed each run, so prompts, tools, and middleware can resolve and run other
// actions without direct registry access.
//
// The Genkit instance is reconstructed from reg (a *genkit.Genkit is a thin
// wrapper over its registry), so ai and ai/exp need not hold a *genkit.Genkit
// themselves and stay genkit-agnostic. It is nil until the genkit package is
// linked into the build; callers treat a nil hook as "no seeding", leaving
// actions defined on a bare registry untouched.
var SeedContextForRegistry func(ctx context.Context, reg api.Registry) context.Context
