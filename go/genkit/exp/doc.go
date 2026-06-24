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

/*
Package exp holds experimental genkit helpers that are still taking shape.
It currently provides:

  - An HTTP route layout for serving agents and flows: the [Route] value and
    the [AgentRoutes] / [AllAgentRoutes] / [FlowRoutes] / [AllFlowRoutes]
    builders. Range over the routes and wire each onto an [http.ServeMux] (or
    any router) with [Route.Pattern] and [Route.Handler]. The handlers
    themselves come from the stable genkit package ([genkit.Handler]); this
    package only lays out which paths map to which actions, so the routing
    layer can evolve without touching genkit's stable surface.

  - A channel-based streaming flow constructor, [DefineStreamingFlow]: an
    alternative to the callback-based [genkit.DefineStreamingFlow] for logic
    that is more naturally expressed by writing chunks to a channel.

APIs in this package are under active development and may change in any minor
version release.
*/
package exp
