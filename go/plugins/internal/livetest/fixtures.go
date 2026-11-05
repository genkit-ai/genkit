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
	"encoding/binary"
	"errors"
	"fmt"
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
	// diagnostics blocks until its context ends, after signaling started,
	// while holdDiagnostics is set, and reports at once otherwise: a model
	// may run it again on a later turn, which must not stall.
	diagnostics     *ai.ToolAction[diagnosticsInput, string]
	started         chan struct{}
	holdDiagnostics atomic.Bool
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
			if !f.holdDiagnostics.Load() {
				return "All systems nominal.", nil
			}
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
				Content: []*ai.Part{ai.NewMediaPart("image/png", RedImage)},
			}, nil
		})

	return f
}

// RedImage is a solid red 100x100 PNG as a data URL, big enough that every
// provider's minimum-size check accepts it. Plugin tiers use it for their
// own image cases.
var RedImage = func() string {
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	draw.Draw(img, img.Bounds(), &image.Uniform{C: color.RGBA{R: 255, A: 255}}, image.Point{}, draw.Src)
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		panic(err)
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}()

// ToneAudio is one second of a 440 Hz sine tone as a 16 kHz mono 16-bit WAV
// data URL.
var ToneAudio = func() string {
	const rate, seconds = 16000, 1
	samples := make([]int16, rate*seconds)
	for i := range samples {
		samples[i] = int16(math.MaxInt16 / 2 * math.Sin(2*math.Pi*440*float64(i)/rate))
	}
	dataSize := uint32(len(samples) * 2)
	var buf bytes.Buffer
	for _, field := range []any{
		[]byte("RIFF"), 36 + dataSize, []byte("WAVE"),
		// fmt chunk: PCM, mono, rate, byte rate, block align, bits per sample.
		[]byte("fmt "), uint32(16), uint16(1), uint16(1), uint32(rate), uint32(rate * 2), uint16(2), uint16(16),
		[]byte("data"), dataSize, samples,
	} {
		if err := binary.Write(&buf, binary.LittleEndian, field); err != nil {
			panic(err)
		}
	}
	return "data:audio/wav;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}()

// secretWord is the word [SecretPDF] carries, made up so the model can only
// read it from the document.
const secretWord = "Quorblex"

// SecretPDF is a one-page PDF whose only text names [secretWord], as a data
// URL.
var SecretPDF = func() string {
	content := fmt.Sprintf("BT /F1 24 Tf 72 700 Td (The secret word is %s.) Tj ET", secretWord)
	objects := []string{
		"<< /Type /Catalog /Pages 2 0 R >>",
		"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
		"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
		fmt.Sprintf("<< /Length %d >>\nstream\n%s\nendstream", len(content), content),
		"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
	}
	var buf bytes.Buffer
	buf.WriteString("%PDF-1.4\n")
	offsets := make([]int, len(objects))
	for i, obj := range objects {
		offsets[i] = buf.Len()
		fmt.Fprintf(&buf, "%d 0 obj\n%s\nendobj\n", i+1, obj)
	}
	xref := buf.Len()
	fmt.Fprintf(&buf, "xref\n0 %d\n0000000000 65535 f \n", len(objects)+1)
	for _, off := range offsets {
		fmt.Fprintf(&buf, "%010d 00000 n \n", off)
	}
	fmt.Fprintf(&buf, "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n", len(objects)+1, xref)
	return "data:application/pdf;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}()
