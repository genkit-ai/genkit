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
	"net/http"

	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

// Base paths for the built-in serving layouts.
const (
	agentBasePath = "/agents"
	flowBasePath  = "/flows"
)

// Route is one HTTP route in a primitive's default serving layout: the
// method and path to mount and the action to serve.
//
// [AgentRoutes], [AllAgentRoutes], [FlowRoutes], and [AllFlowRoutes]
// produce Routes; range over them and wire each onto an [http.ServeMux]
// with [Route.Pattern] and [Route.Handler]. The fields are exported so
// other routers (Gin, Chi, Echo) can mount the same layout: read Method
// and Path and serve Action with [genkit.Handler].
// Every route is a POST that speaks the {"data": ...} / {"result": ...}
// envelope of the reflection API (the agent turn route also streams via
// ?stream=true), so a single client transport reaches all of them.
type Route struct {
	// Method is the HTTP method; always "POST" for the built-in layouts.
	Method string
	// Path is the URL path to mount, e.g. "/agents/chat/getSnapshot".
	Path string
	// Action is the action served at this route via [genkit.Handler].
	Action api.Action
}

// Pattern returns the "METHOD /path" pattern for [http.ServeMux.HandleFunc].
func (r Route) Pattern() string {
	return r.Method + " " + r.Path
}

// Handler builds the HTTP handler for this route with [genkit.Handler],
// applying opts.
func (r Route) Handler(opts ...genkit.HandlerOption) http.HandlerFunc {
	return genkit.Handler(r.Action, opts...)
}

// AllAgentRoutes returns the default serving layout for every agent
// registered with g, the iterate-over-all counterpart to [AgentRoutes].
// Mount the result onto an [http.ServeMux] (range over it and call
// [Route.Handler]) or hand it to a router of your choice. See
// [AgentRoutes] for the per-agent layout and the route set each agent
// contributes.
func AllAgentRoutes(g *genkit.Genkit) []Route {
	var routes []Route
	for _, act := range ListAgents(g) {
		name := act.Name()
		// The snapshot-lifecycle companions register independently under
		// their own action types, keyed by the agent's name (see the agent
		// package's snapshot companions). Recover them by key so the layout
		// depends only on the registry, not on the concrete agent type.
		// LookupAction returns nil for a capability the agent lacks (a
		// client-managed agent has no companions), and buildAgentRoutes omits
		// the route for a nil companion.
		snapshot := genkit.LookupAction(g, api.KeyFromName(api.ActionTypeAgentSnapshot, name))
		abort := genkit.LookupAction(g, api.KeyFromName(api.ActionTypeAgentAbort, name))
		routes = append(routes, buildAgentRoutes(name, act, snapshot, abort)...)
	}
	return routes
}

// AgentRoutes returns the default serving layout for a single agent, so you
// can mount specific agents rather than every registered one. Mount the
// result onto an [http.ServeMux], or onto a router of your choice.
//
// The route set mirrors what the agent can do:
//
//   - POST /agents/{name}                the agent, one turn per request
//   - POST /agents/{name}/getSnapshot    getSnapshot (store-backed agents)
//   - POST /agents/{name}/abort          abort (abortable stores)
//
// Each takes the {"data": ...} request envelope and returns {"result":
// ...}; the snapshot ID rides in the body ({"data": {"snapshotId": ...}}),
// the same as the reflection API. Companion routes are omitted for
// capabilities the agent lacks; a client-managed agent contributes only
// its turn route.
func AgentRoutes[State any](a *aix.Agent[State]) []Route {
	return buildAgentRoutes(a.Name(), a, a.GetSnapshotAction(), a.AbortAction())
}

// buildAgentRoutes builds an agent's route set from its run action and the
// companion actions it has (either may be nil). Shared by AllAgentRoutes
// (companions looked up by key) and AgentRoutes (companions from the typed
// ref's accessors).
func buildAgentRoutes(name string, run, snapshot, abort api.Action) []Route {
	routes := []Route{{Method: http.MethodPost, Path: agentBasePath + "/" + name, Action: run}}
	if snapshot != nil {
		routes = append(routes, Route{
			Method: http.MethodPost,
			Path:   agentBasePath + "/" + name + "/getSnapshot",
			Action: snapshot,
		})
	}
	if abort != nil {
		routes = append(routes, Route{
			Method: http.MethodPost,
			Path:   agentBasePath + "/" + name + "/abort",
			Action: abort,
		})
	}
	return routes
}

// AllFlowRoutes returns the default serving layout for every flow
// registered with g, the iterate-over-all counterpart to [FlowRoutes].
// Mount the result onto an [http.ServeMux], or onto a router of your choice.
func AllFlowRoutes(g *genkit.Genkit) []Route {
	var routes []Route
	for _, f := range genkit.ListFlows(g) {
		routes = append(routes, buildFlowRoute(f))
	}
	return routes
}

// FlowRoutes returns the default serving layout for a single flow: one
// route, POST /flows/{name}, taking its input from the request body. It
// returns a slice for symmetry with [AgentRoutes], so route lists compose
// with append.
func FlowRoutes(f api.Action) []Route {
	return []Route{buildFlowRoute(f)}
}

func buildFlowRoute(f api.Action) Route {
	return Route{Method: http.MethodPost, Path: flowBasePath + "/" + f.Name(), Action: f}
}
