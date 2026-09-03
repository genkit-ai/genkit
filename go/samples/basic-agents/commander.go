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

// commander.go is the background sub-agent demo.
//
// The orchestrator agent delegates and waits: its turn is blocked for as long
// as the sub-agent runs. This agent cannot afford that. It is an incident
// commander, and an incident has two clocks that do not agree. The
// investigation takes as long as it takes. The status update is owed now.
//
// So every investigation starts with "background": true. The delegation tool
// returns a task ID at once and the sub-agent keeps running on a context that
// outlives the tool call, so the commander posts its first update while the
// investigators are still reading logs. It collects the results afterwards
// with wait_for_background_tasks: first with a short timeout, so a slow
// investigation turns into an interim update instead of silence, then without
// one.
//
// The parts worth reading:
//
//   - Agents.Async on the middleware. It adds the "background" flag to every
//     delegation tool and the three task tools that go with it: check, wait
//     and abort.
//   - Both sub-agents have a session store. A background delegation records a
//     pending snapshot as the durable task, so a store is what makes an agent
//     able to run in the background at all; a store-less agent is refused at
//     launch and the commander is told to delegate to it synchronously.
//   - The task ID ("<agent>:<snapshotId>") comes back in the delegation tool
//     result, so it lands in the conversation. Nothing else tracks the task,
//     which is why a commander rebuilt from its history alone can still
//     collect the answer.
//   - Neither investigator is told anything about this conversation. History
//     is never forwarded to a sub-agent that has a store, so the task text has
//     to stand alone and each investigator pulls its own input with tools.
//   - read_status_board is work the commander can do with the time background
//     delegation gives it back. A blocking orchestrator has no such time: its
//     turn is inside the sub-agent call. The board answers with one useful
//     line and one useless one, drawn at random, so the commander has to read
//     it more than once and has to judge what it gets.
//   - WithMaxTurns. Launching, posting, reading the board, waiting, posting
//     and waiting again are all tool rounds, so an orchestrator that collects
//     in the background needs a higher ceiling than one that blocks on each
//     delegation.
//
// Try it with:
//
//	checkout-api is throwing 500s and customers can't pay
//
// Watch the order of the tool call lines: two delegations, then post_status,
// then the waits, with read_status_board filling the time in between. The task
// IDs are visible in the wait tool's input.
//
// The snapshot outlives the process but the worker does not. Quit while a task
// is still pending and its heartbeat goes stale, so a later check reports it
// as expired rather than resuming it. Let the tasks settle first and their
// results are still on disk for the next run.

package main

import (
	"context"
	"fmt"
	"maps"
	"math/rand/v2"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/genkit"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
	"github.com/firebase/genkit/go/plugins/middleware"
	middlewarex "github.com/firebase/genkit/go/plugins/middleware/exp"
)

// The incident: checkout-api started failing at 14:03. The logs and the deploy
// history each hold one half of the answer, and neither investigator can reach
// the other's half. That is what makes the two of them worth running at the
// same time rather than one after the other.
var serviceLogs = map[string]string{
	"checkout-api": `14:02:58 INFO  rollout v2026.8.19-a1 complete (12/12 pods ready)
14:03:11 WARN  db pool: waited 1.9s for a connection (in use 5/5)
14:03:14 ERROR db pool: timed out after 5s (in use 5/5, waiters 63)
14:03:14 ERROR POST /v1/checkout 500 (db pool timeout)
14:04:02 ERROR db pool: timed out after 5s (in use 5/5, waiters 148)
14:05:30 WARN  p99 latency 8.4s (210ms at 13:55)
14:06:00 ERROR POST /v1/checkout 500 (db pool timeout) x412 in 60s`,
	"payments-api": `14:03:20 WARN  upstream checkout-api slow: 8.1s
14:03:44 INFO  circuit breaker half-open for checkout-api
14:05:10 WARN  upstream checkout-api slow: 9.6s`,
}

var serviceDeploys = map[string]string{
	"checkout-api": `v2026.8.19-a1  13:58  r.patel   cut the db pool from 20 to 5 to fit the smaller instance
v2026.8.18-c7  09:12  s.okafor  add a request id to the access logs`,
	"payments-api": `v2026.8.19-b3  13:41  m.lindqvist  update the vendor integration docs`,
}

// A real log scan reads tens of gigabytes and a real deploy lookup calls a
// release service. That slowness is the reason a delegation is worth
// backgrounding, so the demo keeps it. The scan is the slower of the two, so
// the investigations settle at different times.
const (
	logScanLatency      = 12 * time.Second
	deployLookupLatency = 5 * time.Second
)

// statusPage is the incident channel the commander posts to: process state,
// like the banker's account balance, and reset on restart.
var statusPage struct {
	sync.Mutex
	updates []string
}

// What other responders drop into the incident channel while the commander
// waits. A real channel carries both kinds at once, and the commander cannot
// tell which it is about to get, so read_status_board returns one of each at
// random: the point of the tool is that reading the board is worth a turn even
// though most reads are not.
var (
	// usefulChatter changes what the commander knows. Each line is a fact
	// neither investigator can reach: they read logs and deploys, not people.
	usefulChatter = []string{
		"@s.okafor: staging ran the same rollout an hour ago and is fine. staging runs 2 pods though, not 12.",
		"@r.patel: the pool change was mine. i sized it off the old instance type and nobody caught it in review.",
		"@m.lindqvist: payments-api is only failing on calls that go through checkout. its own health checks are green.",
		"@on-call-db: connection count on the primary is flat at 60. the pool is the ceiling, the database is bored.",
		"@t.abara: rolling back v2026.8.19-a1 in staging brought p99 back to 200ms in about a minute.",
	}
	// noiseChatter is the rest of the channel: real, well meant, and not
	// actionable. It is here so the commander has to sort signal from volume
	// rather than treating every read as a finding.
	noiseChatter = []string{
		"@j.reyes: is this why my dashboard looks weird?",
		"@k.tran: +1, seeing it too",
		"@d.singh: should we open a bridge call for this?",
		"@a.novak: who is running point? i lost the thread above.",
		"@support: three customer tickets so far, all about checkout. linking them here for later.",
		"@b.cho: reminder that the release freeze starts friday, unrelated but worth flagging",
	}
)

// defineCommanderAgent registers the incident tools, two investigator
// sub-agents, and the commander that runs them in the background.
func defineCommanderAgent(g *genkit.Genkit) *aix.Agent[any] {
	queryLogs := genkitx.DefineTool(g, "query_logs",
		"Scans the last 30 minutes of warnings and errors for one service. The scan is slow.",
		func(ctx context.Context, in struct {
			Service string `json:"service" jsonschema_description:"The service to scan e.g. checkout-api"`
		}) (string, error) {
			if err := slowBackend(ctx, logScanLatency); err != nil {
				return "", err
			}
			logs, ok := serviceLogs[in.Service]
			if !ok {
				return unknownService(in.Service, serviceLogs), nil
			}
			return logs, nil
		})

	recentDeploys := genkitx.DefineTool(g, "recent_deploys",
		"Lists what shipped to one service in the last 24 hours, newest first.",
		func(ctx context.Context, in struct {
			Service string `json:"service" jsonschema_description:"The service to list deploys for e.g. checkout-api"`
		}) (string, error) {
			if err := slowBackend(ctx, deployLookupLatency); err != nil {
				return "", err
			}
			deploys, ok := serviceDeploys[in.Service]
			if !ok {
				return unknownService(in.Service, serviceDeploys), nil
			}
			return deploys, nil
		})

	readStatusBoard := genkitx.DefineTool(g, "read_status_board",
		"Reads the incident channel: the updates posted so far, plus whatever other responders have said since. Cheap and safe to call while waiting on an investigation.",
		func(ctx context.Context, in struct{}) (string, error) {
			return renderStatusBoard(), nil
		})

	postStatus := genkitx.DefineTool(g, "post_status",
		"Posts an update to the incident channel. Use it to keep responders informed while the investigation runs.",
		func(ctx context.Context, in struct {
			Message string `json:"message" jsonschema_description:"The update to post, one or two sentences"`
		}) (string, error) {
			statusPage.Lock()
			defer statusPage.Unlock()
			statusPage.updates = append(statusPage.updates, in.Message)
			return fmt.Sprintf("Posted update #%d to the incident channel.", len(statusPage.updates)), nil
		})

	// Each investigator gets a session store. That is what lets the commander
	// launch it in the background: the runtime writes a pending snapshot for
	// the detached invocation and finalizes it in place when the work settles.
	// Both descriptions mention the latency, because they reach the
	// commander's model in the injected <sub-agents> block and tell it which
	// delegations are worth backgrounding.
	logAnalyst := genkitx.DefineAgent(g, "log_analyst",
		aix.InlinePrompt{
			ai.WithModel(model),
			ai.WithTools(queryLogs),
			ai.WithSystem("You are a log analyst on an incident call. Scan the logs of the " +
				"service you are given, and of any other service those logs implicate. " +
				"Report the failure signature, when it started, and how it spread. Report " +
				"only what the logs show: naming the cause is someone else's job. Answer " +
				"in at most five lines."),
			// A transient model error settles the task as failed, and the only
			// recovery from there is a fresh delegation.
			ai.WithUse(&middleware.Retry{MaxRetries: 5}),
		},
		aix.WithSessionStore(mustStore[any]("log_analyst")),
		aix.WithDescription[any]("Scans service logs and reports the failure signature. Slow: a scan takes tens of seconds."),
	)

	deployAuditor := genkitx.DefineAgent(g, "deploy_auditor",
		aix.InlinePrompt{
			ai.WithModel(model),
			ai.WithTools(recentDeploys),
			ai.WithSystem("You are a release auditor on an incident call. List what shipped " +
				"to the service you are given near the time of the incident. Name the " +
				"deploy most likely to be responsible, say what in it could cause the " +
				"symptoms, and name the deploys you rule out. Answer in at most five lines."),
			ai.WithUse(&middleware.Retry{MaxRetries: 5}),
		},
		aix.WithSessionStore(mustStore[any]("deploy_auditor")),
		aix.WithDescription[any]("Reviews recent deploys around an incident and names the likely culprit. Slow: a lookup takes several seconds."),
	)

	return genkitx.DefineAgent(g, "commander",
		aix.InlinePrompt{
			ai.WithModel(model),
			ai.WithTools(postStatus, readStatusBoard),
			// Collecting in the background costs tool rounds a blocking
			// delegation does not: launch, post, wait, read the board, post,
			// wait, post is seven before the commander says anything, and it
			// may read the board more than once. The default cap is five.
			ai.WithMaxTurns(15),
			// The middleware already explains how background delegation works,
			// so the runbook says only when to use it. Without a rule this
			// explicit the model often delegates synchronously, which is
			// correct but leaves the channel silent until the work ends.
			ai.WithSystem("You are the incident commander for an on-call rotation. You run the " +
				"incident. You do not investigate it yourself.\n\n" +
				"Your runbook:\n" +
				"1. Start every investigation in the background, with \"background\": true. " +
				"Start all of them in one turn. The investigators are slow and the incident " +
				"channel will not wait for them.\n" +
				"2. Post the first status update as soon as the investigations are running. " +
				"Never hold the first update back for a result.\n" +
				"3. Then wait for the tasks with timeoutSeconds 10. If the wait times out, " +
				"read the status board, post an interim update naming what is still " +
				"outstanding, and fold in anything the board told you. Then wait again " +
				"with no timeout.\n" +
				"4. When the results are in, post a final update with the cause and the fix, " +
				"then give the user the same answer in two or three lines.\n" +
				"5. If a task comes back failed or expired, say so and start that one again, " +
				"once. Do not restart a task that returned a result.\n\n" +
				"Say what you are about to do in one short sentence before each tool call. " +
				"Keep every message short enough to read on a phone."),
			ai.WithUse(
				// Async is the only difference from the orchestrator agent's
				// configuration. HistoryLength is left at 0 on purpose: both
				// sub-agents have stores, and history is never forwarded to a
				// sub-agent with a store, so setting it would suggest a
				// context the investigators never receive.
				&middlewarex.Agents{
					Agents:         []aix.AgentRef{logAnalyst.Ref(), deployAuditor.Ref()},
					Async:          true,
					MaxDelegations: 6,
				},
				&middleware.Retry{MaxRetries: 5},
			),
		},
		aix.WithSessionStore(mustStore[any]("commander")),
		aix.WithDescription[any]("Incident commander (runs sub-agents in the background via the agents middleware)"),
	)
}

// renderStatusBoard renders the incident channel: the updates the commander
// has posted, then one useful line and one useless one from the other
// responders, in an order it cannot predict. Reading the board is therefore
// worth doing more than once and never reliably worth doing twice in a row,
// which is the judgement call the tool exists to pose.
func renderStatusBoard() string {
	statusPage.Lock()
	posted := slices.Clone(statusPage.updates)
	statusPage.Unlock()

	var b strings.Builder
	b.WriteString("Updates you have posted:\n")
	if len(posted) == 0 {
		b.WriteString("  (none yet)\n")
	}
	for i, update := range posted {
		fmt.Fprintf(&b, "  %d. %s\n", i+1, update)
	}

	chatter := []string{
		usefulChatter[rand.IntN(len(usefulChatter))],
		noiseChatter[rand.IntN(len(noiseChatter))],
	}
	rand.Shuffle(len(chatter), func(i, j int) { chatter[i], chatter[j] = chatter[j], chatter[i] })
	b.WriteString("\nSince you last looked:\n")
	for _, line := range chatter {
		fmt.Fprintf(&b, "  %s\n", line)
	}
	return b.String()
}

// slowBackend blocks for d, or returns early if the task is aborted. An
// aborted background task should stop the work it started, so the wait
// watches the context rather than sleeping through it.
func slowBackend(ctx context.Context, d time.Duration) error {
	select {
	case <-time.After(d):
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// unknownService names the services that exist instead of failing the turn,
// so a guessed name costs one tool call rather than the investigation.
func unknownService(name string, known map[string]string) string {
	names := slices.Sorted(maps.Keys(known))
	return fmt.Sprintf("No records for %q. Known services: %s.", name, strings.Join(names, ", "))
}
