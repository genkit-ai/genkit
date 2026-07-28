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

package firebase

import (
	"context"
	"encoding/json"
	"strings"

	"firebase.google.com/go/v4/auth"
	genkit "github.com/firebase/genkit/go"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/status"
)

// AuthContext is the context of an authenticated request.
type AuthContext = *auth.Token

// AuthPolicy is a function that validates an incoming request.
type AuthPolicy = func(context.Context, AuthContext, json.RawMessage) error

// AuthClient is a client for the Firebase Auth service.
type AuthClient interface {
	VerifyIDToken(context.Context, string) (*auth.Token, error)
}

// ContextProvider creates a Firebase context provider for Genkit actions.
func ContextProvider(ctx context.Context, g *genkit.Genkit, policy AuthPolicy) (core.ContextProvider, error) {
	f, err := resolvePlugin(g)
	if err != nil {
		return nil, err
	}
	client, err := f.Auth(ctx)
	if err != nil {
		return nil, err
	}

	return func(ctx context.Context, input core.RequestData) (core.ActionContext, error) {
		authHeader, ok := input.Headers["authorization"]
		if !ok {
			return nil, status.PublicErrorf(status.ErrUnauthenticated, "authorization header is required but not provided")
		}

		const bearerPrefix = "bearer "

		if !strings.HasPrefix(strings.ToLower(authHeader), bearerPrefix) {
			return nil, status.PublicErrorf(status.ErrUnauthenticated, "invalid authorization header format")
		}

		token := authHeader[len(bearerPrefix):]
		authCtx, err := client.VerifyIDToken(ctx, token)
		if err != nil {
			return nil, status.PublicErrorf(status.ErrUnauthenticated, "invalid ID token: %v", err)
		}

		if policy != nil {
			if err := policy(ctx, authCtx, input.Input); err != nil {
				return nil, err
			}
		}

		return core.ActionContext{
			"auth": authCtx,
		}, nil
	}, nil
}
