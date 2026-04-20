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

package genkit

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal"
)

type streamingCallback[Stream any] = func(context.Context, Stream) error

// runtimeFileData is the data written to the file describing this runtime.
type runtimeFileData struct {
	ID                       string `json:"id"`
	PID                      int    `json:"pid"`
	ReflectionServerURL      string `json:"reflectionServerUrl"`
	Timestamp                string `json:"timestamp"`
	GenkitVersion            string `json:"genkitVersion"`
	ReflectionApiSpecVersion int    `json:"reflectionApiSpecVersion"`
}

// reflectionServer encapsulates everything needed to serve the Reflection API.
type reflectionServer struct {
	*http.Server
	RuntimeFilePath string            // Path to the runtime file that was written at startup.
	activeActions   *activeActionsMap // Tracks active actions for cancellation support.
}

// activeAction represents an in-flight action that can be cancelled.
type activeAction struct {
	cancel    context.CancelFunc
	startTime time.Time
	traceID   string
}

// activeActionsMap safely manages active actions.
type activeActionsMap struct {
	mu      sync.RWMutex
	actions map[string]*activeAction
}

func newActiveActionsMap() *activeActionsMap {
	return &activeActionsMap{
		actions: make(map[string]*activeAction),
	}
}

func (m *activeActionsMap) Set(traceID string, action *activeAction) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.actions[traceID] = action
}

func (m *activeActionsMap) Get(traceID string) (*activeAction, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	action, ok := m.actions[traceID]
	return action, ok
}

func (m *activeActionsMap) Delete(traceID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.actions, traceID)
}

func (s *reflectionServer) runtimeID() string {
	_, port, err := net.SplitHostPort(s.Addr)
	if err != nil {
		// This should not happen with a valid address.
		return strconv.Itoa(os.Getpid())
	}
	return fmt.Sprintf("%d-%s", os.Getpid(), port)
}

// findAvailablePort finds the next available port starting from the given port number.
func findAvailablePort(startPort int) (string, error) {
	for port := startPort; port < startPort+100; port++ {
		addr := fmt.Sprintf("127.0.0.1:%d", port)
		listener, err := net.Listen("tcp", addr)
		if err == nil {
			listener.Close()
			return addr, nil
		}
	}
	return "", fmt.Errorf("no available port found in range %d-%d", startPort, startPort+99)
}

// startReflectionServer starts the Reflection API server listening at the
// value of the environment variable GENKIT_REFLECTION_PORT for the port,
// or finds the next available port starting at 3100 if it is empty.
func startReflectionServer(ctx context.Context, g *Genkit, errCh chan<- error, serverStartCh chan<- struct{}) *reflectionServer {
	if g == nil {
		errCh <- fmt.Errorf("nil Genkit provided")
		return nil
	}

	var addr string
	if envPort := os.Getenv("GENKIT_REFLECTION_PORT"); envPort != "" {
		addr = "127.0.0.1:" + envPort
	} else {
		var err error
		addr, err = findAvailablePort(3100)
		if err != nil {
			errCh <- fmt.Errorf("failed to find available port: %w", err)
			return nil
		}
	}

	s := &reflectionServer{
		Server: &http.Server{
			Addr: addr,
		},
		activeActions: newActiveActionsMap(),
	}
	s.Handler = serveMux(g, s)

	slog.Debug("starting reflection server", "addr", s.Addr)

	if err := s.writeRuntimeFile(s.Addr); err != nil {
		errCh <- fmt.Errorf("failed to write runtime file: %w", err)
		return nil
	}

	serverCtx, cancel := context.WithCancel(context.Background())

	go func() {
		// First check that the port is available before signaling a server start success.
		listener, err := net.Listen("tcp", s.Addr)
		if err != nil {
			errCh <- fmt.Errorf("failed to create listener: %w", err)
			return
		}

		close(serverStartCh)

		if err := s.Serve(listener); err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
		// If the server shuts down unexpectedly, this will trigger the cleanup.
		cancel()
	}()

	go func() {
		// Blocks here until the context is done or the server crashes.
		select {
		case <-ctx.Done():
		case <-serverCtx.Done():
			return
		}

		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		if err := s.Shutdown(shutdownCtx); err != nil {
			slog.Error("reflection server shutdown error", "error", err)
		}

		if err := s.cleanupRuntimeFile(); err != nil {
			slog.Error("failed to cleanup runtime file", "error", err)
		}
	}()

	return s
}

// writeRuntimeFile writes a file describing the runtime to the project root.
func (s *reflectionServer) writeRuntimeFile(url string) error {
	projectRoot, err := findProjectRoot()
	if err != nil {
		return fmt.Errorf("failed to find project root: %w", err)
	}

	runtimesDir := filepath.Join(projectRoot, ".genkit", "runtimes")
	if err := os.MkdirAll(runtimesDir, 0755); err != nil {
		return fmt.Errorf("failed to create runtimes directory: %w", err)
	}

	runtimeID := os.Getenv("GENKIT_RUNTIME_ID")
	if runtimeID == "" {
		runtimeID = s.runtimeID()
	}

	timestamp := time.Now().UTC().Format(time.RFC3339)
	// remove colons to avoid problems with different OS file name restrictions
	timestamp = strings.ReplaceAll(timestamp, ":", "_")

	s.RuntimeFilePath = filepath.Join(runtimesDir, fmt.Sprintf("%d-%s.json", os.Getpid(), timestamp))

	data := runtimeFileData{
		ID:                       runtimeID,
		PID:                      os.Getpid(),
		ReflectionServerURL:      fmt.Sprintf("http://%s", url),
		Timestamp:                timestamp,
		GenkitVersion:            "go/" + internal.Version,
		ReflectionApiSpecVersion: internal.GENKIT_REFLECTION_API_SPEC_VERSION,
	}

	fileContent, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal runtime data: %w", err)
	}

	if err := os.WriteFile(s.RuntimeFilePath, fileContent, 0644); err != nil {
		return fmt.Errorf("failed to write runtime file: %w", err)
	}

	slog.Debug("runtime file written", "path", s.RuntimeFilePath)
	return nil
}

// cleanupRuntimeFile removes the runtime file associated with the dev server.
func (s *reflectionServer) cleanupRuntimeFile() error {
	if s.RuntimeFilePath == "" {
		return nil
	}

	content, err := os.ReadFile(s.RuntimeFilePath)
	if err != nil {
		return fmt.Errorf("failed to read runtime file: %w", err)
	}

	var data runtimeFileData
	if err := json.Unmarshal(content, &data); err != nil {
		return fmt.Errorf("failed to unmarshal runtime data: %w", err)
	}

	if data.PID == os.Getpid() {
		if err := os.Remove(s.RuntimeFilePath); err != nil {
			return fmt.Errorf("failed to remove runtime file: %w", err)
		}
		slog.Debug("runtime file cleaned up", "path", s.RuntimeFilePath)
	}

	return nil
}

// findProjectRoot finds the project root by looking for a go.mod file.
func findProjectRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}

	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			slog.Warn("could not find project root (go.mod not found)")
			return os.Getwd()
		}
		dir = parent
	}
}

// serveMux returns a new ServeMux configured for the required Reflection API endpoints.
func serveMux(g *Genkit, s *reflectionServer) *http.ServeMux {
	mux := http.NewServeMux()
	// Skip wrapHandler here to avoid logging constant polling requests.
	mux.HandleFunc("GET /api/__health", func(w http.ResponseWriter, r *http.Request) {
		if id := r.URL.Query().Get("id"); id != "" && id != s.runtimeID() {
			http.Error(w, "Invalid runtime ID", http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("GET /api/actions", wrapReflectionHandler(handleListActions(g)))
	mux.HandleFunc("POST /api/runAction", wrapReflectionHandler(handleRunAction(g, s.activeActions)))
	mux.HandleFunc("POST /api/notify", wrapReflectionHandler(handleNotify()))
	mux.HandleFunc("POST /api/cancelAction", wrapReflectionHandler(handleCancelAction(s.activeActions)))
	mux.HandleFunc("GET /api/values", wrapReflectionHandler(handleListValues(g)))
	return mux
}

// wrapReflectionHandler wraps an HTTP handler function with common logging and error handling.
func wrapReflectionHandler(h func(w http.ResponseWriter, r *http.Request) error) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		logger.FromContext(ctx).Debug("request start", "method", r.Method, "path", r.URL.Path)

		var err error
		defer func() {
			if err != nil {
				logger.FromContext(ctx).Error("request end", "err", err)
			} else {
				logger.FromContext(ctx).Debug("request end")
			}
		}()

		w.Header().Set("x-genkit-version", "go/"+internal.Version)

		if err = h(w, r); err != nil {
			errorResponse := core.ToReflectionError(err)
			w.WriteHeader(errorResponse.Code)
			writeJSON(ctx, w, errorResponse)
		}
	}
}

// handleRunAction looks up an action by name in the registry, runs it with the
// provided JSON input, and writes back the JSON-marshaled request.
func handleRunAction(g *Genkit, activeActions *activeActionsMap) func(w http.ResponseWriter, r *http.Request) error {
	return func(w http.ResponseWriter, r *http.Request) error {
		ctx := r.Context()

		var body struct {
			Key             string          `json:"key"`
			Input           json.RawMessage `json:"input"`
			Context         json.RawMessage `json:"context"`
			TelemetryLabels json.RawMessage `json:"telemetryLabels"`
		}
		defer r.Body.Close()
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			return core.NewError(core.INVALID_ARGUMENT, err.Error())
		}

		stream, err := parseBoolQueryParam(r, "stream")
		if err != nil {
			return err
		}

		logger.FromContext(ctx).Debug("running action", "key", body.Key, "stream", stream)

		// Create cancellable context for this action
		actionCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		// Track whether headers have been sent
		headersSent := false
		var callbackTraceID string // Trace ID captured from telemetry callback for early header sending
		var mu sync.Mutex

		// Set up telemetry callback to capture and send trace ID early
		// This is used for BOTH streaming and non-streaming to match JS behavior
		telemetryCb := func(tid string, sid string) {
			mu.Lock()
			defer mu.Unlock()

			if !headersSent {
				callbackTraceID = tid

				// Track active action for cancellation
				activeActions.Set(callbackTraceID, &activeAction{
					cancel:    cancel,
					startTime: time.Now(),
					traceID:   callbackTraceID,
				})

				// Send headers immediately with trace ID
				w.Header().Set("X-Genkit-Trace-Id", callbackTraceID)
				w.Header().Set("X-Genkit-Span-Id", sid)
				w.Header().Set("X-Genkit-Version", "go/"+internal.Version)

				if stream {
					w.Header().Set("Content-Type", "text/plain")
					w.Header().Set("Transfer-Encoding", "chunked")
				} else {
					w.Header().Set("Content-Type", "application/json")
				}

				w.WriteHeader(http.StatusOK)
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
				headersSent = true
			}
		}

		// Set up streaming callback if needed
		var cb streamingCallback[json.RawMessage]
		if stream {
			cb = func(ctx context.Context, msg json.RawMessage) error {
				_, err := fmt.Fprintf(w, "%s\n", msg)
				if err != nil {
					return err
				}
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
				return nil
			}
		}

		var contextMap core.ActionContext = nil
		if body.Context != nil {
			json.Unmarshal(body.Context, &contextMap)
		}

		// Attach telemetry callback to context so action can invoke it when span is created
		actionCtx = tracing.WithTelemetryCallback(actionCtx, telemetryCb)
		resp, err := runAction(actionCtx, g, body.Key, body.Input, body.TelemetryLabels, cb, contextMap)

		// Clean up active action using the trace ID from response
		if resp != nil && resp.Telemetry.TraceID != "" {
			activeActions.Delete(resp.Telemetry.TraceID)
		}

		if err != nil {
			// Check if context was cancelled
			if errors.Is(err, context.Canceled) {
				// Use gRPC CANCELLED code (1) in JSON body to match TypeScript behavior
				var traceIDPtr *string
				if resp != nil && resp.Telemetry.TraceID != "" {
					traceIDPtr = &resp.Telemetry.TraceID
				}
				errResp := errorResponse{
					Error: core.ReflectionError{
						Code:    core.CodeCancelled, // gRPC CANCELLED = 1
						Message: "Action was cancelled",
						Details: &core.ReflectionErrorDetails{
							TraceID: traceIDPtr,
						},
					},
				}

				if stream {
					// For streaming, write error as final chunk
					json.NewEncoder(w).Encode(errResp)
				} else {
					// For non-streaming, return error response
					if !headersSent {
						w.WriteHeader(http.StatusOK) // Match TS: response.status(200).json(...)
					}
					json.NewEncoder(w).Encode(errResp)
				}
				return nil
			}

			// Handle other errors
			if stream {
				refErr := core.ToReflectionError(err)
				if resp != nil && resp.Telemetry.TraceID != "" {
					refErr.Details.TraceID = &resp.Telemetry.TraceID
				}

				reflectErr, err := json.Marshal(refErr)
				if err != nil {
					logger.FromContext(ctx).Error("writing output", "err", err)
					return nil
				}
				_, err = fmt.Fprintf(w, "{\"error\": %s }", reflectErr)
				if err != nil {
					return err
				}
				return nil
			}

			// Non-streaming error
			errorResponse := core.ToReflectionError(err)
			if resp != nil && resp.Telemetry.TraceID != "" {
				errorResponse.Details.TraceID = &resp.Telemetry.TraceID
			}

			reflectErr, err := json.Marshal(errorResponse)
			if err != nil {
				logger.FromContext(ctx).Error("writing output", "err", err)
				return nil
			}

			_, err = fmt.Fprintf(w, "{\"error\": %s }", reflectErr)
			if err != nil {
				return err
			}
			return nil
		}

		// Success case
		if stream {
			// For streaming, write the final chunk with result and telemetry
			// This matches JS: response.write(JSON.stringify({result, telemetry}))
			finalResponse := runActionResponse{
				Result:    resp.Result,
				Telemetry: telemetry{TraceID: resp.Telemetry.TraceID},
			}
			data, err := json.Marshal(finalResponse)
			if err != nil {
				logger.FromContext(ctx).Error("writing output", "err", err)
				return nil
			}

			w.Write(data)
		} else {
			// For non-streaming, headers were already sent via telemetry callback
			// Response already includes telemetry.traceId in body
			return writeJSON(ctx, w, resp)
		}

		return nil
	}
}

// handleCancelAction cancels an in-flight action by trace ID.
func handleCancelAction(activeActions *activeActionsMap) func(w http.ResponseWriter, r *http.Request) error {
	return func(w http.ResponseWriter, r *http.Request) error {
		var body struct {
			TraceID string `json:"traceId"`
		}

		defer r.Body.Close()
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			return core.NewError(core.INVALID_ARGUMENT, err.Error())
		}

		if body.TraceID == "" {
			return core.NewError(core.INVALID_ARGUMENT, "traceId is required")
		}

		action, exists := activeActions.Get(body.TraceID)
		if !exists {
			w.WriteHeader(http.StatusNotFound)
			return writeJSON(r.Context(), w, map[string]string{
				"error": "Action not found or already completed",
			})
		}

		// Cancel the action's context
		action.cancel()
		activeActions.Delete(body.TraceID)

		return writeJSON(r.Context(), w, map[string]string{
			"message": "Action cancelled",
		})
	}
}

// handleNotify configures the telemetry server URL from the request.
func handleNotify() func(w http.ResponseWriter, r *http.Request) error {
	return func(w http.ResponseWriter, r *http.Request) error {
		var body struct {
			TelemetryServerURL       string `json:"telemetryServerUrl"`
			ReflectionApiSpecVersion int    `json:"reflectionApiSpecVersion"`
		}

		defer r.Body.Close()
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			return core.NewError(core.INVALID_ARGUMENT, err.Error())
		}

		if os.Getenv("GENKIT_TELEMETRY_SERVER") == "" && body.TelemetryServerURL != "" {
			tracing.WriteTelemetryImmediate(tracing.NewHTTPTelemetryClient(body.TelemetryServerURL))
			slog.Debug("connected to telemetry server", "url", body.TelemetryServerURL)
		}

		if body.ReflectionApiSpecVersion != internal.GENKIT_REFLECTION_API_SPEC_VERSION {
			slog.Error("Genkit CLI version is not compatible with runtime library. Please use `genkit-cli` version compatible with runtime library version.")
		}

		w.WriteHeader(http.StatusOK)
		_, err := w.Write([]byte("OK"))
		return err
	}
}

// handleListActions lists all the registered actions.
// The list is sorted by action name and contains unique action names.
func handleListActions(g *Genkit) func(w http.ResponseWriter, r *http.Request) error {
	return func(w http.ResponseWriter, r *http.Request) error {
		ads := listResolvableActions(r.Context(), g)
		descMap := map[string]api.ActionDesc{}
		for _, d := range ads {
			descMap[d.Key] = d
		}
		return writeJSON(r.Context(), w, descMap)
	}
}

// handleListValues returns registered values filtered by type query parameter.
// Matches JS: GET /api/values?type=middleware
func handleListValues(g *Genkit) func(w http.ResponseWriter, r *http.Request) error {
	return func(w http.ResponseWriter, r *http.Request) error {
		valueType := r.URL.Query().Get("type")
		if valueType == "" {
			return core.NewError(core.INVALID_ARGUMENT, `query parameter "type" is required`)
		}
		prefix := "/" + valueType + "/"
		result := map[string]any{}
		for key, val := range g.reg.ListValues() {
			if strings.HasPrefix(key, prefix) {
				name := strings.TrimPrefix(key, prefix)
				result[name] = val
			}
		}
		return writeJSON(r.Context(), w, result)
	}
}

// listActions lists all the registered actions.
func listActions(g *Genkit) []api.ActionDesc {
	ads := []api.ActionDesc{}

	actions := g.reg.ListActions()
	for _, a := range actions {
		ads = append(ads, a.Desc())
	}

	sort.Slice(ads, func(i, j int) bool {
		return ads[i].Name < ads[j].Name
	})

	return ads
}

// listResolvableActions lists all the registered and resolvable actions.
func listResolvableActions(ctx context.Context, g *Genkit) []api.ActionDesc {
	ads := listActions(g)
	keys := make(map[string]struct{})

	plugins := g.reg.ListPlugins()
	for _, p := range plugins {
		dp, ok := p.(api.DynamicPlugin)
		if !ok {
			// Not all plugins are DynamicPlugins; skip if not.
			continue
		}

		for _, desc := range dp.ListActions(ctx) {
			if _, exists := keys[desc.Name]; !exists {
				ads = append(ads, desc)
				keys[desc.Name] = struct{}{}
			}
		}
	}

	sort.Slice(ads, func(i, j int) bool {
		return ads[i].Name < ads[j].Name
	})

	return ads
}

// TODO: Pull these from common types in genkit-tools.

type runActionResponse struct {
	Result    json.RawMessage `json:"result"`
	Telemetry telemetry       `json:"telemetry"`
}

type telemetry struct {
	TraceID string `json:"traceId"`
}

type errorResponse struct {
	Error core.ReflectionError `json:"error"`
}

func runAction(ctx context.Context, g *Genkit, key string, input json.RawMessage, telemetryLabels json.RawMessage, cb streamingCallback[json.RawMessage], runtimeContext map[string]any) (*runActionResponse, error) {
	action := g.reg.ResolveAction(key)
	if action == nil {
		return nil, core.NewError(core.NOT_FOUND, "action %q not found", key)
	}
	if runtimeContext != nil {
		ctx = core.WithActionContext(ctx, runtimeContext)
	}

	// Parse telemetry attributes if provided
	var telemetryAttributes map[string]string
	if telemetryLabels != nil {
		err := json.Unmarshal(telemetryLabels, &telemetryAttributes)
		if err != nil {
			return nil, core.NewError(core.INTERNAL, "Error unmarshalling telemetryLabels: %v", err)
		}
	}

	// Run the action and capture trace ID. We need to ensure there's a valid trace context.
	var traceID string
	output, err := func() (json.RawMessage, error) {
		r, err := action.RunJSONWithTelemetry(ctx, input, cb)
		if r != nil {
			traceID = r.TraceId
		}
		if err != nil {
			return nil, err
		}
		return r.Result, err
	}()
	if err != nil {
		return &runActionResponse{
			Telemetry: telemetry{TraceID: traceID},
		}, err
	}

	return &runActionResponse{
		Result:    output,
		Telemetry: telemetry{TraceID: traceID},
	}, nil
}

// writeJSON writes a JSON-marshaled value to the response writer.
func writeJSON(ctx context.Context, w http.ResponseWriter, value any) error {
	w.Header().Set("Content-Type", "application/json")

	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	_, err = w.Write(data)
	if err != nil {
		logger.FromContext(ctx).Error("writing output", "err", err)
	}
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
	return nil
}
