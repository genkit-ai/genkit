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
	"errors"
	"fmt"
	"os"
	"sync"

	"cloud.google.com/go/firestore"
	firebasev4 "firebase.google.com/go/v4"
	"firebase.google.com/go/v4/auth"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

const provider = "firebase"
const projectIdEnv = "FIREBASE_PROJECT_ID"

const pluginInstruction = "Pass the Firebase plugin to genkit.Init():\n" +
	"  g := genkit.Init(ctx, genkit.WithPlugins(&firebase.Firebase{ProjectId: \"your-project\"}))"

var errPluginNotInitialized = errors.New("firebase: plugin not initialized. " + pluginInstruction)
var errPluginNotFound = errors.New("firebase: plugin not found. " + pluginInstruction)
var errCredentials = "Ensure you have proper credentials. For local development, run: gcloud auth application-default login"

// Firebase is the Genkit plugin for Firebase services.
// It provides integration with Firebase Firestore for retrievers, indexers, and durable streaming.
//
// Usage:
//
//	g := genkit.Init(ctx, genkit.WithPlugins(&firebase.Firebase{ProjectId: "my-project"}))
//
// Or with an existing Firebase app:
//
//	g := genkit.Init(ctx, genkit.WithPlugins(&firebase.Firebase{App: myFirebaseApp}))
type Firebase struct {
	// ProjectId is the Firebase/GCP project ID. If set, a Firebase app is created automatically.
	// Can also be set via the FIREBASE_PROJECT_ID environment variable.
	ProjectId string
	// App is an existing Firebase app instance. Provide either ProjectId or App, not both.
	App *firebasev4.App

	mu              sync.Mutex
	initted         bool
	firestoreClient *firestore.Client
	authClient      *auth.Client
}

// Name returns the name of the plugin.
func (f *Firebase) Name() string {
	return provider
}

// Init initializes the Firebase plugin. Called automatically by genkit.Init().
func (f *Firebase) Init(ctx context.Context) []api.Action {
	f.mu.Lock()
	defer f.mu.Unlock()

	if f.initted {
		panic("firebase.Init: plugin already initialized")
	}

	projectId := resolveProjectId(f.ProjectId)

	if f.App == nil && projectId == "" {
		panic("firebase.Init: Firebase plugin requires either ProjectId or App to be set.\n" +
			"  Option 1: Set ProjectId directly: &firebase.Firebase{ProjectId: \"your-project-id\"}\n" +
			"  Option 2: Set FIREBASE_PROJECT_ID environment variable\n" +
			"  Option 3: Provide an existing Firebase App: &firebase.Firebase{App: yourApp}")
	}

	if f.App != nil && f.ProjectId != "" {
		panic("firebase.Init: provide either ProjectId or App, not both")
	}

	if f.App == nil {
		firebaseApp, err := firebasev4.NewApp(ctx, &firebasev4.Config{ProjectID: projectId})
		if err != nil {
			panic(fmt.Errorf("firebase.Init: failed to initialize Firebase App: %v", err))
		}
		f.App = firebaseApp
	}

	f.initted = true
	return []api.Action{}
}

// Firestore returns a cached Firestore client for the Firebase project.
// The client is created lazily on first call and reused for subsequent calls.
// This client is shared across all Firebase plugin features (retrievers, stream managers, etc.).
func (f *Firebase) Firestore(ctx context.Context) (*firestore.Client, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if !f.initted {
		return nil, errPluginNotInitialized
	}

	if f.firestoreClient != nil {
		return f.firestoreClient, nil
	}

	client, err := f.App.Firestore(ctx)
	if err != nil {
		return nil, fmt.Errorf("firebase: failed to create Firestore client: %w. %s", err, errCredentials)
	}

	f.firestoreClient = client
	return client, nil
}

// Auth returns a cached Firebase Auth client for the Firebase project.
// The client is created lazily on first call and reused for subsequent calls.
func (f *Firebase) Auth(ctx context.Context) (*auth.Client, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if !f.initted {
		return nil, errPluginNotInitialized
	}

	if f.authClient != nil {
		return f.authClient, nil
	}

	client, err := f.App.Auth(ctx)
	if err != nil {
		return nil, fmt.Errorf("firebase: failed to create Auth client: %w. %s", err, errCredentials)
	}

	f.authClient = client
	return client, nil
}

// DefineRetriever defines a Firestore vector retriever with the given configuration.
// The Firebase plugin must be registered with genkit.Init() before calling this function.
func DefineRetriever(ctx context.Context, g *genkit.Genkit, opts RetrieverOptions) (ai.Retriever, error) {
	f, err := resolvePlugin(g)
	if err != nil {
		return nil, err
	}

	firestoreClient, err := f.Firestore(ctx)
	if err != nil {
		return nil, err
	}

	retriever, err := defineFirestoreRetriever(g, opts, firestoreClient)
	if err != nil {
		return nil, fmt.Errorf("firebase.DefineRetriever: failed to initialize retriever %q: %w", opts.Name, err)
	}
	return retriever, nil
}

// resolveProjectId resolves the Firebase project ID from various sources.
func resolveProjectId(projectId string) string {
	if projectId != "" {
		return projectId
	}
	return os.Getenv(projectIdEnv)
}

// resolvePlugin resolves the Firebase plugin from the Genkit registry.
func resolvePlugin(g *genkit.Genkit) (*Firebase, error) {
	plugin := genkit.LookupPlugin(g, provider)
	if plugin == nil {
		return nil, errPluginNotFound
	}
	f, ok := plugin.(*Firebase)
	if !ok {
		return nil, fmt.Errorf("firebase: unexpected plugin type %T for provider %q", plugin, provider)
	}
	return f, nil
}
