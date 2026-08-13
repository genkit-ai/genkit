// Adversarial review scratch test: NOT for commit.
// A recursive Config type now infers to {"$ref":"#/$defs/...","$defs":{...}}.
// stripRequired and tolerateNulls only walk properties/items, so they no-op on
// the $ref root and the enforced config schema keeps its "required" list —
// partial configs (the documented contract) get rejected at the action
// boundary.
package ai

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/firebase/genkit/go/internal/registry"
)

type advRecursiveConfig struct {
	Temperature float64             `json:"temperature"` // no omitempty -> required in inferred schema
	Fallback    *advRecursiveConfig `json:"fallback,omitempty"`
}

func TestPartialRecursiveConfigAccepted(t *testing.T) {
	r := registry.New()
	model := NewModelAction("test/rec-config", &ModelOptions{},
		func(ctx context.Context, req *ModelRequest, cfg advRecursiveConfig, cb ModelStreamCallback) (*ModelResponse, error) {
			return &ModelResponse{
				Message:      NewModelTextMessage("ok"),
				FinishReason: FinishReasonStop,
			}, nil
		})
	model.Register(r)

	// Partial config: omits "temperature". Config is partial by nature
	// (effectiveConfigSchema's contract), so this must validate.
	req := `{"messages":[{"role":"user","content":[{"text":"hi"}]}],"config":{"fallback":{"temperature":0.5}}}`
	if _, err := model.RunJSON(context.Background(), json.RawMessage(req), nil); err != nil {
		t.Fatalf("partial recursive config rejected at action boundary: %v", err)
	}
}
