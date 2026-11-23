/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import * as assert from 'assert';
import { z } from 'genkit';
import { toJsonSchema } from 'genkit/schema';
import { describe, it } from 'node:test';
import {
  answersText,
  compileQuestions,
  enumQuestion,
  wireQuestions,
} from '../src/protocol.js';
import {
  answerKind,
  choice,
  label,
  level,
  margin,
  noul,
  ranked,
  responseInfo,
  schema,
  score,
} from '../src/questions.js';
import { stringifyBody } from '../src/state.js';

/** A wire value as the server reads it. */
const wire = (value: unknown) => JSON.parse(stringifyBody(value));

const Triage = z
  .object({
    department: choice({
      technical: 'Bugs, outages, integrations',
      billing: 'Payments, invoicing, refunds',
      other: 'None of the above',
    }).describe('Which team should handle this?'),
    isUrgent: noul().describe(
      'Does the ticket explicitly communicate time pressure?'
    ),
    frustration: score(['Calm', 'Concerned but civil', 'Very angry']).describe(
      'How frustrated is the customer?'
    ),
  })
  .describe('The state is a support ticket.');

const triageSchema = () => toJsonSchema({ schema: Triage });

describe('question helpers', () => {
  it('encode the questions on the output schema, through .describe()', () => {
    const s = triageSchema();
    const department = s.properties.department;
    assert.strictEqual(department['x-systemone'], 'choice');
    assert.strictEqual(
      department.description,
      'Which team should handle this?'
    );
    assert.strictEqual(department.additionalProperties, false);
    assert.strictEqual(s.properties.isUrgent['x-systemone'], 'noul');
    assert.strictEqual(s.properties.frustration['x-systemone'], 'score');
    assert.deepStrictEqual(s.properties.frustration['x-levels'], [
      'Calm',
      'Concerned but civil',
      'Very angry',
    ]);
    assert.strictEqual(s.properties.frustration.properties.score.maximum, 2);
  });

  it('compile into wire questions, a typed choice sorted by name', () => {
    const questions = compileQuestions(triageSchema(), 'You triage tickets.');
    assert.deepStrictEqual(wire(questions.department.wire), {
      type: 'choice',
      instructions:
        'You triage tickets.\n\nThe state is a support ticket.\n\nWhich team should handle this?',
      criteria: {
        technical: 'Bugs, outages, integrations',
        billing: 'Payments, invoicing, refunds',
        other: 'None of the above',
      },
    });
    assert.match(
      stringifyBody(questions.department.wire.criteria),
      /^\{"billing":.*"other":.*"technical":/
    );
    assert.deepStrictEqual(wire(questions.isUrgent.wire), {
      type: 'noul',
      instructions:
        'You triage tickets.\n\nThe state is a support ticket.\n\nDoes the ticket explicitly communicate time pressure?',
    });
    assert.deepStrictEqual(wire(questions.frustration.wire.criteria), [
      'Calm',
      'Concerned but civil',
      'Very angry',
    ]);
    assert.deepStrictEqual(questions.frustration.labels, [
      'Calm',
      'Concerned but civil',
      'Very angry',
    ]);
  });

  it('send guidance in place of the strings, with the string as its what', () => {
    const s = toJsonSchema({
      schema: z.object({
        dept: choice(
          { billing: 'Payments', other: 'Anything else' },
          {
            guidance: {
              billing: { not_for: 'Shipping', examples: ['Charged twice'] },
              other: undefined,
            },
          }
        ).describe('Which team?'),
        anger: score(['Calm', 'Angry'], {
          guidance: { 1: { what: 'Shouting', signals: ['caps'] } },
        }).describe('How angry?'),
        urgent: noul({
          yes: 'Names a deadline',
          no: 'No time pressure',
          noGuidance: ['No date', 'No "now"'],
        }).describe('Urgent?'),
      }),
    });
    const q = compileQuestions(s, '');
    assert.deepStrictEqual(wire(q.dept.wire.criteria), {
      billing: {
        not_for: 'Shipping',
        examples: ['Charged twice'],
        what: 'Payments',
      },
      other: 'Anything else',
    });
    assert.deepStrictEqual(wire(q.anger.wire.criteria), [
      'Calm',
      { what: 'Shouting', signals: ['caps'] },
    ]);
    assert.deepStrictEqual(wire(q.urgent.wire.criteria), {
      true: 'Names a deadline',
      false: ['No date', 'No "now"'],
    });
  });

  it('refuse a schema that is not a question set', () => {
    const cases: [string, any, RegExp][] = [
      ['no properties', { type: 'object' }, /has no properties/],
      [
        'plain field',
        {
          type: 'object',
          properties: { x: { type: 'string', description: 'x' } },
        },
        /field "x" is not a question/,
      ],
      [
        'no instructions',
        toJsonSchema({ schema: z.object({ x: noul() }) }),
        /question "x" has no instructions/,
      ],
      [
        'one level',
        toJsonSchema({
          schema: z.object({ x: score(['Only']).describe('x') }),
        }),
        /needs at least two levels/,
      ],
      [
        'half criteria',
        toJsonSchema({
          schema: z.object({ x: noul({ yes: 'Y' }).describe('x') }),
        }),
        /says what only one side means/,
      ],
      [
        'level guidance out of range',
        toJsonSchema({
          schema: z.object({
            x: score(['a', 'b'], { guidance: { 2: 'c' } }).describe('x'),
          }),
        }),
        /guidance for level 2/,
      ],
      [
        'unknown type',
        {
          type: 'object',
          properties: { x: { 'x-systemone': 'rank', description: 'x' } },
        },
        /unknown type "rank"/,
      ],
    ];
    for (const [name, s, message] of cases) {
      assert.throws(() => compileQuestions(s, ''), message, name);
    }
  });

  it('build runtime questions that compile like declared ones', () => {
    const s = schema({
      tool: {
        type: 'choice',
        instructions: 'Which tool serves the request?',
        options: [
          { name: 'search', criteria: 'Looks things up' },
          { name: 'calendar', guidance: { what: 'Dates and meetings' } },
          { name: 'none' },
        ],
      },
      effort: {
        type: 'score',
        instructions: {
          field: { name: 'request', description: 'What was asked' },
        },
        levels: ['One lookup', 'A few steps', 'A project'],
      },
      personal: {
        type: 'noul',
        instructions: ['Does it involve', 'the user'],
        yes: "Names the user's files",
        no: 'Asks about the world',
      },
    });
    assert.deepStrictEqual(s.required, ['effort', 'personal', 'tool']);
    const q = compileQuestions(s, 'Shared context.');
    assert.deepStrictEqual(wire(q.tool.wire.criteria), {
      search: 'Looks things up',
      calendar: { what: 'Dates and meetings' },
      none: 'none',
    });
    assert.deepStrictEqual(q.effort.wire.instructions, [
      'Shared context.',
      { field: { name: 'request', description: 'What was asked' } },
    ]);
    assert.deepStrictEqual(q.personal.wire.instructions, [
      'Shared context.',
      'Does it involve',
      'the user',
    ]);
    assert.deepStrictEqual(wire(q.personal.wire.criteria), {
      true: "Names the user's files",
      false: 'Asks about the world',
    });
  });

  it('refuse runtime questions the API cannot answer', () => {
    assert.throws(
      () =>
        compileQuestions(
          schema({ x: { type: 'choice', instructions: 'x', options: [] } }),
          ''
        ),
      /lists no options/
    );
    assert.throws(
      () =>
        compileQuestions(
          schema({
            x: {
              type: 'choice',
              instructions: 'x',
              options: [{ name: 'a' }, { name: 'a' }],
            },
          }),
          ''
        ),
      /lists option "a" twice/
    );
  });

  it('turn the enum format into one choice', () => {
    assert.deepStrictEqual(
      wire(
        enumQuestion({ type: 'string', enum: ['sales', 'billing'] }, '').choice
          .wire
      ),
      {
        type: 'choice',
        instructions: 'Choose the option that best describes the state.',
        criteria: { sales: 'sales', billing: 'billing' },
      }
    );
    assert.throws(
      () => enumQuestion({ type: 'string', enum: ['a', 'a'] }, ''),
      /lists "a" twice/
    );
    assert.throws(
      () => enumQuestion({ type: 'string', enum: ['a', ''] }, ''),
      /non-empty strings/
    );
  });
});

describe('option names and shapes', () => {
  it('keep the declared order, integer-like names included', () => {
    const q = compileQuestions(
      schema({
        rating: {
          type: 'choice',
          instructions: 'Stars?',
          options: [{ name: '10' }, { name: '2' }, { name: 'none' }],
        },
      }),
      ''
    );
    assert.strictEqual(
      stringifyBody(q.rating.wire.criteria),
      '{"10":"10","2":"2","none":"none"}'
    );
  });

  it('take any name, prototype names included', () => {
    const q = enumQuestion(
      { type: 'string', enum: ['constructor', '__proto__', 'toString'] },
      ''
    );
    assert.strictEqual(
      stringifyBody(q.choice.wire.criteria),
      '{"constructor":"constructor","__proto__":"__proto__","toString":"toString"}'
    );
    const c = compileQuestions(
      schema({
        x: {
          type: 'choice',
          instructions: 'x',
          options: [{ name: 'constructor', criteria: 'A builder' }],
        },
      }),
      ''
    );
    assert.deepStrictEqual(wire(c.x.wire.criteria), {
      constructor: 'A builder',
    });
  });

  it('take any question ID and option name, __proto__ included', () => {
    // A runtime question set read from JSON, as from a tenant's config.
    const questions = JSON.parse(
      '{"__proto__": {"type": "choice", "instructions": "Which?", "options": [{"name": "__proto__", "guidance": {"what": "The odd one"}}, {"name": "plain"}]}}'
    );
    const s = schema(questions);
    assert.deepStrictEqual(s.required, ['__proto__']);
    assert.ok(Object.hasOwn(s.properties, '__proto__'));
    const q = compileQuestions(s, '');
    assert.deepStrictEqual(Object.keys(q), ['__proto__']);
    assert.strictEqual(
      stringifyBody(wireQuestions(q)),
      '{"__proto__":{"type":"choice","instructions":"Which?","criteria":{"__proto__":{"what":"The odd one"},"plain":"plain"}}}'
    );
    const text = answersText(
      JSON.parse('{"__proto__": {"choice": "__proto__", "confidence": 0.9}}'),
      q,
      false
    );
    assert.strictEqual(
      text,
      '{"__proto__":{"choice":"__proto__","confidence":0.9}}'
    );
    assert.throws(
      () => answersText({}, q, false),
      /no answer for question "__proto__"/
    );
  });

  it('survive one choice() behind two fields', () => {
    const Dept = choice({ billing: 'Payments', other: 'Anything else' });
    const s = toJsonSchema({
      schema: z.object({
        primary: Dept.describe('Which team first?'),
        secondary: Dept.describe('Which team next?'),
      }),
    });
    assert.ok(s.properties.secondary.$ref, 'zod writes the reuse as a $ref');
    const q = compileQuestions(s, '');
    assert.deepStrictEqual(wire(q.secondary.wire), {
      type: 'choice',
      instructions: 'Which team next?',
      criteria: { billing: 'Payments', other: 'Anything else' },
    });
  });

  it('read options with no descriptions', () => {
    const q = compileQuestions(
      toJsonSchema({
        schema: z.object({
          x: choice(
            { a: '', b: '' },
            { guidance: { a: { examples: ['one'] } } }
          ).describe('Which?'),
        }),
      }),
      ''
    );
    assert.deepStrictEqual(wire(q.x.wire.criteria), {
      a: { examples: ['one'] },
      b: 'b',
    });
  });
});

describe('references', () => {
  const withRef = (ref: string, extra: object = {}) => ({
    type: 'object',
    properties: { x: { $ref: ref, description: 'x' } },
    ...extra,
  });

  it('refuses a $ref that does not point into the schema', () => {
    assert.throws(
      () =>
        compileQuestions(
          {
            type: 'object',
            properties: { x: { $ref: 'other.json#/x', description: 'x' } },
          },
          ''
        ),
      /a \$ref that is not a pointer into itself/
    );
  });

  it('refuses a $ref that loops, is malformed, or dangles', () => {
    assert.throws(
      () =>
        compileQuestions(
          withRef('#/$defs/a', {
            $defs: { a: { $ref: '#/$defs/b' }, b: { $ref: '#/$defs/a' } },
          }),
          ''
        ),
      /refers to itself/
    );
    assert.throws(
      () => compileQuestions(withRef('#/$defs/%E0%A4%A'), ''),
      /is malformed/
    );
    assert.throws(
      () => compileQuestions(withRef('#/$defs/missing'), ''),
      /references #\/\$defs\/missing, which it does not define/
    );
  });
});

describe('answers', () => {
  it('are projected onto the declared fields, legend rebuilt, score clamped', () => {
    const questions = compileQuestions(triageSchema(), '');
    const text = answersText(
      {
        department: {
          type: 'choice',
          choice: 'billing',
          probabilities: { billing: 0.8, technical: 0.1, other: 0.1 },
          confidence: 0.6,
          rationale: 'added later by a gateway',
        },
        isUrgent: { type: 'noul', noul: 0.2 },
        frustration: {
          type: 'score',
          score: 2.0000000001,
          confidence: 0.5,
          legend: { '0': { what: 'Calm' }, '1': 'x', '2': 'y' },
        },
      },
      questions,
      false
    );
    assert.deepStrictEqual(JSON.parse(text), {
      department: {
        choice: 'billing',
        probabilities: { billing: 0.8, technical: 0.1, other: 0.1 },
        confidence: 0.6,
      },
      isUrgent: { noul: 0.2 },
      frustration: {
        score: 2,
        confidence: 0.5,
        legend: { '0': 'Calm', '1': 'Concerned but civil', '2': 'Very angry' },
      },
    });
  });

  it('say when an answer is not an object', () => {
    const questions = compileQuestions(triageSchema(), '');
    assert.throws(
      () =>
        answersText(
          { department: 'billing', isUrgent: {}, frustration: {} },
          questions,
          false
        ),
      /the answer for question "department" is not an object/
    );
  });

  it('check confidence and probabilities, keeping declared keys only', () => {
    const questions = compileQuestions(triageSchema(), '');
    const answers = (department: object) => ({
      department: { choice: 'billing', ...department },
      isUrgent: { noul: 0.5 },
      frustration: { score: 1 },
    });
    const text = answersText(
      answers({
        confidence: 0.7,
        probabilities: { billing: 0.7, technical: 0.3, sales: 0.0 },
      }),
      questions,
      false
    );
    assert.deepStrictEqual(JSON.parse(text).department, {
      choice: 'billing',
      confidence: 0.7,
      probabilities: { billing: 0.7, technical: 0.3 },
    });
    // A server may leave confidence out.
    assert.deepStrictEqual(
      JSON.parse(answersText(answers({}), questions, false)).department,
      { choice: 'billing' }
    );
    for (const bad of [
      { confidence: 1.2 },
      { confidence: null },
      { confidence: Number.NaN },
      { probabilities: { billing: -0.1 } },
      { probabilities: { billing: null } },
      { probabilities: [0.5] },
    ]) {
      assert.throws(
        () => answersText(answers(bad), questions, false),
        /not a number from 0 to 1|probabilities that are not an object/,
        JSON.stringify(bad)
      );
    }
  });

  it('must answer every question', () => {
    const questions = compileQuestions(triageSchema(), '');
    assert.throws(
      () => answersText({ department: {}, isUrgent: {} }, questions, false),
      /no answer for question "frustration"/
    );
  });

  it('read ranked, margin, level, and label', () => {
    const answer = {
      choice: 'b' as const,
      probabilities: { a: 0.2, b: 0.6, c: 0.2 },
      confidence: 0.4,
    };
    assert.deepStrictEqual(ranked(answer), ['b', 'a', 'c']);
    assert.ok(Math.abs(margin(answer) - 0.4) < 1e-9);
    assert.strictEqual(margin({ choice: 'a', confidence: 1 }), 0);
    const levels = ['Calm', 'Concerned', 'Angry'];
    assert.strictEqual(level({ score: 1.4, confidence: 0 }, levels), 1);
    assert.strictEqual(level({ score: 7, confidence: 0 }, levels), 2);
    assert.strictEqual(label({ score: 1.6, confidence: 0 }, levels), 'Angry');
    assert.strictEqual(
      label({ score: 0.2, confidence: 0, legend: { '0': 'Calm', '1': 'x' } }),
      'Calm'
    );
    assert.strictEqual(answerKind({ noul: 0 }), 'noul');
    assert.strictEqual(answerKind({ score: 0, confidence: 0 }), 'score');
  });

  it('drop probabilities for undeclared options and levels unchecked', () => {
    const q = compileQuestions(
      toJsonSchema({
        schema: z.object({
          pick: choice({ a: 'A', b: 'B' }).describe('Which?'),
          rate: score(['Low', 'High']).describe('How much?'),
        }),
      }),
      ''
    );
    for (const extra of [1.5, null, 'x']) {
      const text = answersText(
        {
          pick: { choice: 'a', probabilities: { a: 0.7, b: 0.3, c: extra } },
          rate: {
            score: 0.3,
            probabilities: { '0': 0.7, '1': 0.3, '2': extra },
          },
        },
        q,
        false
      );
      const out = JSON.parse(text);
      assert.deepStrictEqual(
        out.pick.probabilities,
        { a: 0.7, b: 0.3 },
        String(extra)
      );
      assert.deepStrictEqual(
        out.rate.probabilities,
        { '0': 0.7, '1': 0.3 },
        String(extra)
      );
    }
  });

  it('copies the answers of the info', () => {
    const response = {
      raw: {
        model: 'jev',
        answers: { u: { noul: 0.5 } },
        systemOneInfo: true,
      },
    };
    const info = responseInfo(response);
    (info.answers!.u as any).noul = 1;
    assert.strictEqual(response.raw.answers.u.noul, 0.5);
  });

  it('carry the response info', () => {
    assert.deepStrictEqual(responseInfo(undefined), { model: '' });
    assert.deepStrictEqual(responseInfo({ raw: { text: 'hi' } }), {
      model: '',
    });
    assert.deepStrictEqual(
      responseInfo({
        raw: {
          model: 'jev-1.13.0',
          provider: 'TypeSafe',
          id: 'gen-1',
          answers: {},
          systemOneInfo: true,
        },
      }),
      { model: 'jev-1.13.0', provider: 'TypeSafe', id: 'gen-1', answers: {} }
    );
    // Another model's raw output that happens to name a model is not info.
    assert.deepStrictEqual(
      responseInfo({ raw: { model: 'gpt-x', answers: {} } }),
      { model: '' }
    );
  });
});
