// Adversarial review scratch test: NOT for commit.
// Compares the pre-PR InferJSONSchema (DoNotReference + sub-reflector) against
// the new InferJSONSchemaMap on acyclic types, where the PR claims output is
// unchanged.
package base

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/invopop/jsonschema"
)

var jsonMarshalerTypeOld = reflect.TypeOf((*json.Marshaler)(nil)).Elem()

func anyStructSchemaOld(t reflect.Type) *jsonschema.Schema {
	if t.Implements(jsonMarshalerTypeOld) || reflect.PointerTo(t).Implements(jsonMarshalerTypeOld) {
		return &jsonschema.Schema{AdditionalProperties: jsonschema.TrueSchema}
	}
	return &jsonschema.Schema{
		Type:                 "object",
		AdditionalProperties: jsonschema.TrueSchema,
	}
}

func inferOldJSONSchema(x any) *jsonschema.Schema {
	inProgress := make(map[reflect.Type]bool)
	var mapper func(reflect.Type) *jsonschema.Schema
	mapper = func(t reflect.Type) *jsonschema.Schema {
		if t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Interface {
			return &jsonschema.Schema{
				Type:  "array",
				Items: &jsonschema.Schema{AdditionalProperties: jsonschema.TrueSchema},
			}
		}
		baseType := t
		if t.Kind() == reflect.Ptr {
			baseType = t.Elem()
		}
		if baseType.Kind() != reflect.Struct {
			return nil
		}
		if inProgress[baseType] {
			return anyStructSchemaOld(baseType)
		}
		inProgress[baseType] = true
		defer delete(inProgress, baseType)
		firstCall := true
		sub := jsonschema.Reflector{
			DoNotReference: true,
			Anonymous:      true,
			Mapper: func(st reflect.Type) *jsonschema.Schema {
				if firstCall && st == baseType {
					firstCall = false
					return nil
				}
				return mapper(st)
			},
		}
		s := sub.ReflectFromType(baseType)
		s.Version = ""
		return s
	}
	r := jsonschema.Reflector{DoNotReference: true, Anonymous: true, Mapper: mapper}
	s := r.Reflect(x)
	s.Version = ""
	return s
}

type reviewAddress struct {
	Street string `json:"street"`
	City   string `json:"city"`
}

type reviewPerson struct {
	Name string         `json:"name" jsonschema:"description=Full name"`
	Home reviewAddress  `json:"home" jsonschema:"description=Primary residence"`
	Work *reviewAddress `json:"work,omitempty" jsonschema:"title=Work address"`
}

type reviewMarshaler struct{ V string }

func (m reviewMarshaler) MarshalJSON() ([]byte, error) { return json.Marshal(m.V) }

type reviewMixed struct {
	When   time.Time       `json:"when"`
	Tags   []string        `json:"tags,omitempty"`
	Any    []any           `json:"any,omitempty"`
	Meta   map[string]int  `json:"meta,omitempty"`
	M      reviewMarshaler `json:"m"`
	Level  int             `json:"level" jsonschema:"minimum=0,description=depth"`
	Choice string          `json:"choice,omitempty" jsonschema:"enum=a,enum=b"`
}

type reviewEmbedded struct {
	reviewAddress
	Extra string `json:"extra"`
}

func TestAcyclicUnchangedClaim(t *testing.T) {
	cases := []struct {
		name string
		v    any
	}{
		{"nested struct with field descriptions", reviewPerson{}},
		{"mixed leaf types", reviewMixed{}},
		{"embedded struct", reviewEmbedded{}},
		{"plain address", reviewAddress{}},
		{"pointer root", &reviewPerson{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			oldMap := SchemaAsMap(inferOldJSONSchema(tc.v))
			newMap := InferJSONSchemaMap(tc.v)
			oldJSON, _ := json.MarshalIndent(oldMap, "", "  ")
			newJSON, _ := json.MarshalIndent(newMap, "", "  ")
			if diff := cmp.Diff(string(oldJSON), string(newJSON)); diff != "" {
				t.Errorf("acyclic schema changed (-old +new):\n%s", diff)
			}
		})
	}
}
