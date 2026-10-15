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
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"go.opentelemetry.io/otel/propagation"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/wire"
)

// --- RemoteAgentOption ---

// RemoteAgentOption configures a remote agent built by [NewRemoteAgent]. A
// later option of the same kind replaces an earlier one.
type RemoteAgentOption interface {
	applyRemoteAgent(*remoteAgentOptions)
}

type remoteAgentOptions struct {
	httpClient  *http.Client
	headers     func(context.Context) (http.Header, error)
	routes      *AgentRouteURLs
	description string
	metadata    *AgentMetadata
}

func (o *remoteAgentOptions) applyRemoteAgent(opts *remoteAgentOptions) {
	if o.httpClient != nil {
		opts.httpClient = o.httpClient
	}
	if o.headers != nil {
		opts.headers = o.headers
	}
	if o.routes != nil {
		opts.routes = o.routes
	}
	if o.metadata != nil {
		opts.metadata = o.metadata
	}
}

// WithHTTPClient sets the HTTP client that reaches a remote agent. Use it for
// authentication (an ID token or OAuth2 client), mTLS, and transport timeouts.
// The default is [http.DefaultClient].
//
// The remote agent refuses redirects unless c sets its own CheckRedirect, so
// a request body and its credentials are not replayed to another host.
func WithHTTPClient(c *http.Client) RemoteAgentOption {
	return &remoteAgentOptions{httpClient: c}
}

// WithHeaders sets a function that returns headers to add to every request to
// a remote agent, such as a bearer token read from ctx. An error from fn fails
// the request before it is sent.
func WithHeaders(fn func(ctx context.Context) (http.Header, error)) RemoteAgentOption {
	return &remoteAgentOptions{headers: fn}
}

// AgentRouteURLs names the snapshot companion routes of a remote agent whose
// server does not serve them next to its turn route, such as a Next.js app
// with a route file per action. Each is an absolute http or https URL; an
// empty field keeps the default, the turn route's URL with "/getSnapshot",
// "/waitForSnapshot", or "/abort" appended.
type AgentRouteURLs struct {
	GetSnapshot     string
	WaitForSnapshot string
	Abort           string
}

// WithRouteURLs sets the URLs of a remote agent's snapshot companion routes.
// A server that serves no wait route is polled through GetSnapshot instead.
func WithRouteURLs(urls AgentRouteURLs) RemoteAgentOption {
	return &remoteAgentOptions{routes: &urls}
}

// WithAgentMetadata declares a remote agent's capability metadata: who manages
// its state and whether its background work can be aborted. Callers such as
// the agents middleware gate on it, so declare what the agent's server
// actually supports. A field left unset reads as "no"; without this option
// the metadata is unknown, which callers treat as "ask the agent and see".
func WithAgentMetadata(meta *AgentMetadata) RemoteAgentOption {
	return &remoteAgentOptions{metadata: cloneAgentMetadata(meta)}
}

// --- NewRemoteAgent ---

// NewRemoteAgent returns a handle to an agent that a Genkit app serves over
// HTTP, in Go (the genkit/exp package's AgentRoutes layout), JS, or Dart. url
// is the agent's turn route (e.g. "https://billing.internal/agents/billing");
// the snapshot companions are at url+"/getSnapshot", url+"/waitForSnapshot",
// and url+"/abort", unless [WithRouteURLs] puts them elsewhere. It performs no
// I/O.
//
// The handle's operations work as they do for an agent in this process,
// detached runs and their tasks included, as far as the server allows. Errors
// from the server keep their status name; a route the server does not publish
// is UNIMPLEMENTED, and a wait the server cannot answer, such as one to a
// server without a wait route, falls back to polling GetSnapshot.
//
// To call the agent by name (from the agents middleware, the Dev UI, or
// [LookupAgent]), register the handle with [AgentHandle.Register], or use the
// genkit/exp package's DefineRemoteAgent.
//
// It panics if name is empty or url, or a URL [WithRouteURLs] sets, is not
// an absolute http or https URL without user info, query, or fragment.
func NewRemoteAgent(name, url string, opts ...RemoteAgentOption) *AgentHandle {
	if name == "" {
		panic("aix.NewRemoteAgent: name is required")
	}
	var o remoteAgentOptions
	for _, opt := range opts {
		opt.applyRemoteAgent(&o)
	}
	urls, err := routeURLs(url, o.routes)
	if err != nil {
		panic(fmt.Sprintf("aix.NewRemoteAgent: agent %q: %v", name, err))
	}
	client := http.DefaultClient
	if o.httpClient != nil {
		client = o.httpClient
	}
	// A copy, so refusing redirects does not change the caller's client.
	c := *client
	if c.CheckRedirect == nil {
		c.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	}
	return NewAgentHandle(name,
		&AgentHandleOptions{Description: o.description, Metadata: o.metadata},
		&httpTransport{name: name, urls: urls, client: &c, headers: o.headers})
}

// routeURLs returns the URL of each route a remote agent's transport calls,
// keyed by its suffix under the turn route ("" for the turn route itself).
func routeURLs(turn string, set *AgentRouteURLs) (map[string]string, error) {
	base, err := parseAgentURL(turn)
	if err != nil {
		return nil, err
	}
	if set == nil {
		set = &AgentRouteURLs{}
	}
	urls := map[string]string{"": base.String()}
	for suffix, raw := range map[string]string{
		"/getSnapshot":     set.GetSnapshot,
		"/waitForSnapshot": set.WaitForSnapshot,
		"/abort":           set.Abort,
	} {
		if raw == "" {
			u := *base
			u.Path += suffix
			urls[suffix] = u.String()
			continue
		}
		u, err := parseAgentURL(raw)
		if err != nil {
			return nil, err
		}
		urls[suffix] = u.String()
	}
	return urls, nil
}

// parseAgentURL parses the URL of a remote agent's route. A query or fragment
// would not survive the companion paths appended to it, and user info would
// put a credential in every log line that prints the URL.
func parseAgentURL(raw string) (*url.URL, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return nil, fmt.Errorf("invalid url %q: %w", raw, err)
	}
	switch {
	case u.Scheme != "http" && u.Scheme != "https":
		return nil, fmt.Errorf("url %q: scheme must be http or https", raw)
	case u.Host == "":
		return nil, fmt.Errorf("url %q: host is required", raw)
	case u.User != nil:
		return nil, fmt.Errorf("url %q: user info is not allowed; authenticate with WithHTTPClient or WithHeaders", raw)
	case u.RawQuery != "" || u.ForceQuery || u.Fragment != "":
		return nil, fmt.Errorf("url %q: query and fragment are not allowed", raw)
	}
	u.Path = strings.TrimSuffix(u.Path, "/")
	u.RawPath = ""
	return u, nil
}

// --- HTTP transport ---

// maxResponseBytes bounds a non-streaming response and one streamed event, so
// a misbehaving server cannot exhaust memory.
const maxResponseBytes = 64 << 20

// Wait fallback cadence, for a server that does not publish waitForSnapshot.
const (
	remotePollInitial = 500 * time.Millisecond
	remotePollMax     = 5 * time.Second
)

// httpTransport reaches an agent that a Genkit app serves over HTTP, through
// the routes of the genkit/exp package's AgentRoutes layout, speaking the
// wire format in internal/wire.
type httpTransport struct {
	name    string
	urls    map[string]string // by route suffix; see routeURLs
	client  *http.Client
	headers func(context.Context) (http.Header, error)
}

var _ SnapshotTransport = (*httpTransport)(nil)

// endpoint returns the URL of the route suffix names.
func (t *httpTransport) endpoint(suffix string) string {
	return t.urls[suffix]
}

// Run posts the turn to the agent's turn route, streaming, so chunks reach cb
// as the agent sends them and a failure arrives with its status name.
func (t *httpTransport) Run(ctx context.Context, input *AgentInput, init *AgentInit[json.RawMessage], cb func(context.Context, json.RawMessage) error) (*AgentOutput[json.RawMessage], error) {
	var initJSON json.RawMessage
	if init != nil {
		var err error
		if initJSON, err = json.Marshal(init); err != nil {
			return nil, fmt.Errorf("agent %q: marshal init: %w", t.name, err)
		}
	}
	resp, err := t.post(ctx, "", input, initJSON, true)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64<<10), maxResponseBytes)
	for scanner.Scan() {
		line := scanner.Bytes()
		if frame, ok := bytes.CutPrefix(line, []byte(errorFramePrefix)); ok {
			return nil, t.errorFrame(frame)
		}
		data, ok := bytes.CutPrefix(line, []byte(wire.SSEDataPrefix))
		if !ok {
			continue
		}
		// The events of the wire format, decoded in one pass; RawMessage
		// copies, so Message outlives the scanner's buffer.
		var event struct {
			Message json.RawMessage               `json:"message"`
			Result  *AgentOutput[json.RawMessage] `json:"result"`
			Error   *status.Error                 `json:"error"`
		}
		if err := json.Unmarshal(data, &event); err != nil {
			return nil, status.Errorf(status.ErrInternal, "agent %q: malformed stream event: %w", t.name, err)
		}
		switch {
		case event.Error != nil:
			return nil, event.Error
		case event.Result != nil:
			if err := t.checkOutput(event.Result); err != nil {
				return nil, err
			}
			return event.Result, nil
		case event.Message != nil:
			if cb != nil {
				if err := cb(ctx, event.Message); err != nil {
					return nil, err
				}
			}
		}
	}
	if err := scanner.Err(); err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, status.Errorf(status.ErrUnavailable, "agent %q: read stream: %w", t.name, err)
	}
	return nil, status.Errorf(status.ErrUnavailable, "agent %q: the stream ended without a result", t.name)
}

// errorFramePrefix starts the frame that JS servers end a failed stream with,
// in place of a data event carrying the error.
const errorFramePrefix = "error: "

// errorFrame decodes an error frame: {"error": {status, message}} as the
// Express plugin sends it, or a bare {status, message} as the Next.js plugin
// does.
func (t *httpTransport) errorFrame(frame []byte) error {
	var wrapped struct {
		Error *status.Error `json:"error"`
	}
	var bare status.Error
	switch {
	case json.Unmarshal(frame, &wrapped) == nil && wrapped.Error != nil && wrapped.Error.Status != "":
		return wrapped.Error
	case json.Unmarshal(frame, &bare) == nil && bare.Status != "":
		return &bare
	}
	return status.Errorf(status.ErrInternal, "agent %q: malformed error frame: %.200s", t.name, frame)
}

// checkOutput rejects an output the middleware and handles could not act on.
func (t *httpTransport) checkOutput(out *AgentOutput[json.RawMessage]) error {
	if out.FinishReason == AgentFinishReasonDetached && out.SnapshotID == "" {
		return status.Errorf(status.ErrInternal, "agent %q: the server detached the invocation without naming its snapshot", t.name)
	}
	return nil
}

func (t *httpTransport) GetSnapshot(ctx context.Context, req *GetSnapshotRequest) (*SessionSnapshot[json.RawMessage], error) {
	return t.snapshot(ctx, "/getSnapshot", req)
}

// WaitForSnapshot makes one wait request; the server answers once the snapshot
// settles or its wait limit passes, and [AgentHandle.WaitForSnapshot] asks
// again until it settles. For a server that publishes no wait route it polls
// GetSnapshot until the snapshot settles. When the HTTP client's own timeout
// ends the request first, it answers with the snapshot as it stands, so a
// client timeout under the server's wait limit costs a re-request, not the
// wait.
//
// A NOT_FOUND wait is confirmed by the poll's first read: a router that
// answers a missing route in plain text (Dart's does, with "Not found") reads
// the same as a server that has no such snapshot.
func (t *httpTransport) WaitForSnapshot(ctx context.Context, req *GetSnapshotRequest) (*SessionSnapshot[json.RawMessage], error) {
	snap, err := t.snapshot(ctx, "/waitForSnapshot", req)
	var netErr net.Error
	switch {
	case status.Of(err) == status.Unimplemented, status.Of(err) == status.NotFound:
		return t.pollSnapshot(ctx, req)
	case ctx.Err() == nil && errors.As(err, &netErr) && netErr.Timeout():
		return t.readSnapshotNow(ctx, req)
	}
	return snap, err
}

// readSnapshotNow reads the snapshot's metadata and, once it is settled, the
// full snapshot, so a pending snapshot costs no state payload. A JS server
// drops the metadataOnly field it does not know and sends the full snapshot
// each time, which costs payload but changes no result.
func (t *httpTransport) readSnapshotNow(ctx context.Context, req *GetSnapshotRequest) (*SessionSnapshot[json.RawMessage], error) {
	read := *req
	read.MetadataOnly = true
	snap, err := t.GetSnapshot(ctx, &read)
	if err != nil || !snap.Status.Terminal() {
		return snap, err
	}
	return t.GetSnapshot(ctx, req)
}

// pollSnapshot reads the snapshot until it settles, backing off between reads.
// Like the runtime's own wait, it rides out a few consecutive transient read
// failures (see [IsRetryableReadError]).
func (t *httpTransport) pollSnapshot(ctx context.Context, req *GetSnapshotRequest) (*SessionSnapshot[json.RawMessage], error) {
	interval := remotePollInitial
	failures := 0
	for {
		snap, err := t.readSnapshotNow(ctx, req)
		switch {
		case err == nil && snap.Status.Terminal():
			return snap, nil
		case err == nil:
			failures = 0
		case !IsRetryableReadError(err) || failures >= snapshotWaitReadRetries:
			return nil, err
		default:
			failures++
		}
		timer := time.NewTimer(interval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
		interval = min(2*interval, remotePollMax)
	}
}

func (t *httpTransport) Abort(ctx context.Context, req *AgentAbortRequest) (*AgentAbortResponse, error) {
	resp, err := call[AgentAbortResponse](ctx, t, "/abort", req)
	if err != nil {
		return nil, err
	}
	// A JS server answers with the status from before the abort, and with
	// none when there is no such snapshot.
	switch resp.Status {
	case "":
		return nil, status.Errorf(status.ErrNotFound, "agent %q: snapshot %q not found", t.name, req.SnapshotID)
	case SnapshotStatusPending:
		resp.Status = SnapshotStatusAborting
	}
	return resp, nil
}

func (t *httpTransport) snapshot(ctx context.Context, suffix string, req *GetSnapshotRequest) (*SessionSnapshot[json.RawMessage], error) {
	return call[SessionSnapshot[json.RawMessage]](ctx, t, suffix, req)
}

// call posts data to the route suffix names and returns its result.
func call[T any](ctx context.Context, t *httpTransport, suffix string, data any) (*T, error) {
	resp, err := t.post(ctx, suffix, data, nil, false)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	// wire.ResultResponse, decoded in one pass.
	var body struct {
		Result *T `json:"result"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxResponseBytes)).Decode(&body); err != nil {
		return nil, status.Errorf(status.ErrInternal, "agent %q: malformed response from %s: %w", t.name, suffix, err)
	}
	if body.Result == nil {
		return nil, status.Errorf(status.ErrInternal, "agent %q: the response from %s carries no result", t.name, suffix)
	}
	return body.Result, nil
}

// post sends one request to the route suffix names and returns the response
// once its status is 200. The caller closes the body. Any other response is
// returned as the error it carries.
func (t *httpTransport) post(ctx context.Context, suffix string, data any, init json.RawMessage, stream bool) (*http.Response, error) {
	dataJSON, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("agent %q: marshal request: %w", t.name, err)
	}
	body, err := json.Marshal(wire.Request{Data: dataJSON, Init: init})
	if err != nil {
		return nil, fmt.Errorf("agent %q: marshal request: %w", t.name, err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.endpoint(suffix), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("agent %q: build request: %w", t.name, err)
	}
	if t.headers != nil {
		extra, err := t.headers(ctx)
		if err != nil {
			return nil, fmt.Errorf("agent %q: request headers: %w", t.name, err)
		}
		for k, vs := range extra {
			for _, v := range vs {
				req.Header.Add(k, v)
			}
		}
	}
	req.Header.Set("Content-Type", "application/json")
	if stream {
		req.Header.Set("Accept", "text/event-stream")
	} else {
		req.Header.Set("Accept", "application/json")
	}
	// Carry the caller's trace, so a server that reads it can join the trace.
	propagation.TraceContext{}.Inject(ctx, propagation.HeaderCarrier(req.Header))

	resp, err := t.client.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, status.Errorf(status.ErrUnavailable, "agent %q: %w", t.name, err)
	}
	if resp.StatusCode == http.StatusOK {
		return resp, nil
	}
	defer resp.Body.Close()
	return nil, t.responseError(resp, suffix)
}

// responseError turns a failed response into an error with a status name. A
// Genkit server sends a [wire.Error] body; anything else is classified by its
// HTTP status, except a 404 from a router, which means the server publishes no
// such route.
func (t *httpTransport) responseError(resp *http.Response, suffix string) error {
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	mt, _, _ := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	if mt == "application/json" {
		var e status.Error
		if json.Unmarshal(raw, &e) == nil && e.Status != "" {
			return &e
		}
	}
	// The Next.js plugin refuses a stream before it starts with an error
	// frame, ended by a blank line.
	if frame, ok := bytes.CutPrefix(raw, []byte(errorFramePrefix)); ok {
		frame, _, _ = bytes.Cut(frame, []byte("\n"))
		return t.errorFrame(frame)
	}
	route := "turn"
	if suffix != "" {
		route = strings.TrimPrefix(suffix, "/")
	}
	switch {
	case resp.StatusCode == http.StatusNotFound && !olderGenkitError(mt, raw):
		return status.Errorf(status.ErrUnimplemented, "agent %q: the server publishes no %s route at %s", t.name, route, t.endpoint(suffix))
	case resp.StatusCode >= 300 && resp.StatusCode < 400:
		return status.Errorf(status.ErrFailedPrecondition, "agent %q: the server redirected the %s request (HTTP %d); redirects are refused", t.name, route, resp.StatusCode)
	}
	n := status.FromHTTPCode(resp.StatusCode)
	return &status.Error{Status: n, Message: fmt.Sprintf("agent %q: %s request failed: HTTP %d: %s", t.name, route, resp.StatusCode, strings.TrimSpace(string(raw)))}
}

// olderGenkitError reports whether a non-JSON error body is a Genkit server's
// own error, as servers sent it before error bodies were JSON: plain text with
// the error's message. Go's router answers a missing route in plain text too,
// with its fixed message.
func olderGenkitError(mediaType string, body []byte) bool {
	return mediaType == "text/plain" && strings.TrimSpace(string(body)) != "404 page not found"
}

// --- Registration ---

// Register registers the agent the handle reaches with r, so callers that
// hold only its name (the agents middleware, the Dev UI, [LookupAgent]) can
// call it. It is the handle's counterpart of [Agent.Register].
//
// For a handle over an agent in this process ([LookupAgent], [Agent.Handle])
// it registers the agent's own actions. For any other handle, such as one
// from [NewRemoteAgent], it registers an agent action, plus the snapshot
// companions the metadata allows (all three when it is unknown), that forward
// to the agent through the handle's transport. Such an agent action runs the
// inputs of an invocation one after another, each as its own turn on the
// transport, and carries the session from one turn to the next; a detach
// applies to the input that carries it. Serving layouts that list every agent
// (the genkit/exp package's AllAgentRoutes) skip it, so a server does not
// expose another service's agent with its own credentials.
//
// Like every registration, it panics if r already holds an agent by the same
// name.
func (h *AgentHandle) Register(r api.Registry) {
	if t, ok := h.transport.(*actionTransport); ok {
		t.run.Register(r)
		for _, companion := range []api.Action{t.getSnapshot, t.wait, t.abort} {
			if companion != nil {
				companion.Register(r)
			}
		}
		return
	}

	meta := h.Metadata()
	metadata := map[string]any{wire.RemoteAgentMetadataKey: true}
	if meta != nil {
		metadata["agent"] = *meta
	}
	if d := h.Description(); d != "" {
		metadata["description"] = d
	}
	core.NewBidiActionOf(api.ActionTypeAgent, h.name,
		&core.BidiActionOptions{Metadata: metadata}, h.forwardTurns).Register(r)

	st, ok := h.transport.(SnapshotTransport)
	if !ok {
		return
	}
	if meta == nil || meta.StateManagement == AgentStateManagementServer {
		core.NewActionOf(api.ActionTypeAgentSnapshot, h.name, nil,
			func(ctx context.Context, req *GetSnapshotRequest) (*SessionSnapshot[json.RawMessage], error) {
				if err := checkGetSnapshotRequest(req); err != nil {
					return nil, err
				}
				return st.GetSnapshot(ctx, req)
			}).Register(r)
		core.NewActionOf(api.ActionTypeAgentWait, h.name, nil,
			func(ctx context.Context, req *GetSnapshotRequest) (*SessionSnapshot[json.RawMessage], error) {
				if err := checkWaitRequest(req); err != nil {
					return nil, err
				}
				return st.WaitForSnapshot(ctx, req)
			}).Register(r)
	}
	if meta == nil || meta.Abortable {
		core.NewActionOf(api.ActionTypeAgentAbort, h.name, nil,
			func(ctx context.Context, req *AgentAbortRequest) (*AgentAbortResponse, error) {
				if err := checkAbortRequest(req); err != nil {
					return nil, err
				}
				return st.Abort(ctx, req)
			}).Register(r)
	}
}

// forwardTurns is the body of a registered remote agent's action. Each input
// becomes one turn on the transport, run to its end before the next input is
// read, and the turn's output names the session source of the next one. The
// invocation ends with the last turn's output and the usage of every turn,
// or early when a turn fails or detaches, as it does in this process.
func (h *AgentHandle) forwardTurns(ctx context.Context, init *AgentInit[json.RawMessage], inCh <-chan *AgentInput, outCh chan<- *AgentStreamChunk) (*AgentOutput[json.RawMessage], error) {
	forward := func(ctx context.Context, raw json.RawMessage) error {
		var chunk AgentStreamChunk
		if err := json.Unmarshal(raw, &chunk); err != nil {
			return status.Errorf(status.ErrInternal, "agent %q: malformed stream chunk: %w", h.name, err)
		}
		select {
		case outCh <- &chunk:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	var (
		out   *AgentOutput[json.RawMessage]
		usage *ai.GenerationUsage
	)
	for {
		var (
			in *AgentInput
			ok bool
		)
		select {
		case in, ok = <-inCh:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		if !ok {
			break
		}
		turn, err := h.runTurn(ctx, in, init, forward)
		if err != nil {
			return nil, err
		}
		usage = ai.SumUsage(usage, turn.Usage)
		out = turn
		switch turn.FinishReason {
		case AgentFinishReasonDetached:
			// A detached output reports no usage: the work continues.
			return turn, nil
		case AgentFinishReasonFailed:
			// A failed output counts every turn, as it does in this process.
			turn.Usage = usage
			return turn, nil
		}
		init = nextTurnInit(turn)
	}
	if out == nil {
		return &AgentOutput[json.RawMessage]{}, nil
	}
	out.Usage = usage
	return out, nil
}

// nextTurnInit returns the session source that continues the conversation a
// turn's output describes: its state for a client-managed agent, its snapshot
// for a server-managed one, with the session ID so the server checks that the
// snapshot belongs to that session.
func nextTurnInit(out *AgentOutput[json.RawMessage]) *AgentInit[json.RawMessage] {
	switch {
	case out.State != nil:
		return &AgentInit[json.RawMessage]{State: out.State}
	case out.SnapshotID != "":
		return &AgentInit[json.RawMessage]{SnapshotID: out.SnapshotID, SessionID: out.SessionID}
	case out.SessionID != "":
		return &AgentInit[json.RawMessage]{SessionID: out.SessionID}
	}
	return nil
}
