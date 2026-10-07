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

package livetest

import (
	"bytes"
	"encoding/base64"
	"errors"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"sync/atomic"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// The tools carry made-up names and made-up results, so a correct answer can
// only come from calling them.

type gablorkenInput struct {
	Value float64 `json:"value"`
	Over  float64 `json:"over"`
}

type transferInput struct {
	Amount float64 `json:"amount"`
	To     string  `json:"to"`
}

type transferResult struct {
	Status       string `json:"status"`
	Confirmation string `json:"confirmation,omitempty"`
}

type orderInput struct {
	OrderID string `json:"orderId"`
}

type orderResult struct {
	OrderID string `json:"orderId"`
	Status  string `json:"status"`
	Carrier string `json:"carrier"`
}

type diagnosticsInput struct {
	Target string `json:"target"`
}

type swatchInput struct {
	Name string `json:"name"`
}

// transferCode is the confirmation an approved transfer returns.
const transferCode = "ZX-41"

// orderCarrier is the carrier the order lookup reports.
const orderCarrier = "Quokka Express"

// fixtures holds the tools the cases share and the switches that steer the
// stateful ones.
type fixtures struct {
	gablorken *ai.ToolAction[gablorkenInput, float64]
	// transfer interrupts for approval. Restarted with "approved" set in its
	// resumed metadata, it completes with [transferCode].
	transfer *ai.ToolAction[transferInput, transferResult]
	// lookupOrder fails while failLookups is above zero, decrementing it on
	// each failure.
	lookupOrder *ai.ToolAction[orderInput, orderResult]
	failLookups atomic.Int32
	// diagnostics blocks until its context ends, after signaling started.
	diagnostics *ai.ToolAction[diagnosticsInput, string]
	started     chan struct{}
	// swatch returns a red image inside its tool response.
	swatch *ai.ToolAction[swatchInput, *ai.MultipartToolResponse]
}

// errLookupDown is the failure lookupOrder reports while failing.
var errLookupDown = errors.New("order service unavailable")

func defineFixtures(g *genkit.Genkit) *fixtures {
	f := &fixtures{started: make(chan struct{}, 1)}

	f.gablorken = genkit.DefineTool(g, "gablorken",
		"Calculates a gablorken. Use it whenever asked for a gablorken.",
		func(_ *ai.ToolContext, in gablorkenInput) (float64, error) {
			// One more than the power, so a model that has seen a result
			// cannot work out the next one without the tool.
			return math.Pow(in.Value, in.Over) + 1, nil
		})

	f.transfer = genkit.DefineTool(g, "transferFunds",
		"Sends money to a person and returns a confirmation code. The app pauses each transfer until the account owner confirms it.",
		func(tc *ai.ToolContext, in transferInput) (transferResult, error) {
			if !tc.IsResumed() {
				return transferResult{}, tc.Interrupt(&ai.InterruptOptions{
					Metadata: map[string]any{"reason": "needs approval"},
				})
			}
			if approved, _ := tc.Resumed["approved"].(bool); approved {
				return transferResult{Status: "completed", Confirmation: transferCode}, nil
			}
			return transferResult{Status: "rejected by the account owner"}, nil
		})

	f.lookupOrder = genkit.DefineTool(g, "lookupOrder",
		"Looks up the shipping status of an order by its ID.",
		func(_ *ai.ToolContext, in orderInput) (orderResult, error) {
			// Parallel tool calls run concurrently, so the claim on a
			// failure is a compare-and-swap, not a load then a decrement.
			for n := f.failLookups.Load(); n > 0; n = f.failLookups.Load() {
				if f.failLookups.CompareAndSwap(n, n-1) {
					return orderResult{}, errLookupDown
				}
			}
			return orderResult{OrderID: in.OrderID, Status: "shipped", Carrier: orderCarrier}, nil
		})

	f.diagnostics = genkit.DefineTool(g, "runDiagnostics",
		"Runs a diagnostic scan on a target system and returns its report.",
		func(tc *ai.ToolContext, _ diagnosticsInput) (string, error) {
			select {
			case f.started <- struct{}{}:
			default:
			}
			select {
			case <-tc.Done():
				return "", tc.Err()
			case <-time.After(time.Minute):
				return "", errors.New("diagnostics were never aborted")
			}
		})

	f.swatch = genkit.DefineMultipartTool(g, "fetchSwatch",
		"Fetches the color swatch with the given name as an image.",
		func(_ *ai.ToolContext, _ swatchInput) (*ai.MultipartToolResponse, error) {
			return &ai.MultipartToolResponse{
				Output:  map[string]any{"attached": true},
				Content: []*ai.Part{ai.NewMediaPart("image/png", redImage)},
			}, nil
		})

	return f
}

// redImage is a solid red 100x100 PNG as a data URL, big enough that every
// provider's minimum-size check accepts it.
var redImage = func() string {
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	draw.Draw(img, img.Bounds(), &image.Uniform{C: color.RGBA{R: 255, A: 255}}, image.Point{}, draw.Src)
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		panic(err)
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}()
