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

import { annotateSchema, z, type JSONSchema } from 'genkit';
import {
  FALSE_KEYWORD,
  GUIDANCE_KEYWORD,
  INSTRUCTIONS_KEYWORD,
  KIND_KEYWORD,
  LEVELS_KEYWORD,
  TRUE_KEYWORD,
  own,
  setOwn,
} from './protocol.js';

/**
 * The answer to a choice question: one option, the probability of every
 * option, and how concentrated that distribution is.
 *
 * `probabilities` is the model's distribution over the options and sums to 1;
 * `choice` is the option with the most of it; `confidence`, from 0 to 1, is
 * how concentrated the distribution is. A server may leave `confidence` out,
 * as Genkit Go's schema allows, so read it with a default:
 * `(answer.confidence ?? 0) < 0.6`.
 */
export interface ChoiceAnswer<K extends string = string> {
  choice: K;
  probabilities?: Partial<Record<K, number>>;
  confidence?: number;
}

/**
 * The answer to a rubric question: the expected level, computed from the
 * probability of each level, so it falls between levels when the model is
 * split. `probabilities` is that distribution, keyed by level number;
 * `confidence`, from 0 to 1, is how concentrated it is; `legend` maps each
 * level number back to its description, the string of the rubric even when
 * the level was sent with guidance. A server may leave `confidence` out.
 */
export interface ScoreAnswer {
  score: number;
  probabilities?: Record<string, number>;
  confidence?: number;
  legend?: Record<string, string>;
}

/**
 * The answer to a yes/no question: the probability that the statement is
 * true. A value near 0.5 means the model could not tell, not that the answer
 * is "somewhat". There is no separate confidence; the probability is the whole
 * answer.
 */
export interface NoulAnswer {
  noul: number;
}

/** A question schema, the type of one field of a decision schema. */
export type QuestionSchema<T> = z.ZodType<T>;

/** The structured guidance of a choice's options, keyed by option. */
export type OptionGuidance<K extends string> = Partial<Record<K, unknown>>;

/**
 * A choice question: pick one of the options, which are the keys of
 * `criteria`, each with what it means. Describe the field with the question:
 *
 * ```ts
 * const Triage = z.object({
 *   department: choice({
 *     billing: 'Payments, invoicing, refunds',
 *     technical: 'Bugs, outages, integrations',
 *     other: 'None of the above',
 *   }).describe('Which team should handle this?'),
 * });
 * ```
 *
 * The options go out sorted by name, as Genkit Go, Java, and Dart send a typed
 * choice's options; a runtime question keeps the order given. `guidance` adds the
 * structured form of a description the wire format also takes, most often an
 * object with labeled parts such as `not_for` and `examples`. An object with
 * no `what` gets the option's string as its `what`, so guidance adds to the
 * description rather than replacing it; any other value is sent as it is. The
 * strings stay what the schema and the answers show.
 */
export function choice<const K extends string>(
  criteria: Record<K, string>,
  options?: { guidance?: OptionGuidance<K> }
): QuestionSchema<ChoiceAnswer<K>> {
  const keys = (Object.keys(criteria) as K[]).sort((a, b) =>
    a < b ? -1 : a > b ? 1 : 0
  );
  const literals = keys.map((key) => z.literal(key).describe(criteria[key]));
  const answer = z
    .object({
      choice: z.union(literals as unknown as [z.ZodTypeAny, z.ZodTypeAny]),
      probabilities: z.record(z.number()).optional(),
      confidence: z.number().optional(),
    })
    .strict();
  return question(answer, 'choice', {
    [GUIDANCE_KEYWORD]: guidanceOf(options?.guidance),
  });
}

/**
 * A score question: rate the state on the levels, lowest first. A level's
 * index is its number on the wire, so a three-level rubric scores 0 to 2.
 *
 * ```ts
 * frustration: score(['Calm', 'Concerned but civil', 'Very angry']).describe(
 *   'How frustrated is the customer?'
 * ),
 * ```
 *
 * `guidance` is keyed by level index, as a choice's is by option.
 */
export function score(
  levels: string[],
  options?: { guidance?: Record<number, unknown> }
): QuestionSchema<ScoreAnswer> {
  const answer = z
    .object({
      score: z
        .number()
        .min(0)
        .max(Math.max(levels.length - 1, 0)),
      probabilities: z.record(z.number()).optional(),
      confidence: z.number().optional(),
      legend: z.record(z.string()).optional(),
    })
    .strict();
  return question(answer, 'score', {
    [LEVELS_KEYWORD]: [...levels],
    [GUIDANCE_KEYWORD]: guidanceOf(options?.guidance),
  });
}

/**
 * What yes and no mean for a yes/no question, with optional structured
 * guidance for each side. Each side needs a string or guidance, or neither
 * does: the API takes the pair or nothing.
 */
export interface YesNo {
  yes?: string;
  no?: string;
  yesGuidance?: unknown;
  noGuidance?: unknown;
}

/**
 * A yes/no question: is the statement its description makes true? With
 * `criteria`, the question says what yes and no mean.
 *
 * ```ts
 * isUrgent: noul({
 *   yes: 'Names a deadline, or says now or today',
 *   no: 'No time pressure is expressed',
 * }).describe('Does the ticket explicitly communicate time pressure?'),
 * ```
 */
export function noul(criteria?: YesNo): QuestionSchema<NoulAnswer> {
  const answer = z.object({ noul: z.number().min(0).max(1) }).strict();
  return question(answer, 'noul', noulKeywords(criteria));
}

function noulKeywords(criteria: YesNo | undefined): Record<string, unknown> {
  const keywords: Record<string, unknown> = {};
  if (criteria?.yes || criteria?.no) {
    keywords[TRUE_KEYWORD] = criteria.yes ?? '';
    keywords[FALSE_KEYWORD] = criteria.no ?? '';
  }
  keywords[GUIDANCE_KEYWORD] = guidanceOf({
    true: criteria?.yesGuidance,
    false: criteria?.noGuidance,
  });
  return keywords;
}

/**
 * Marks an answer schema as a question. The keywords go on the inner object,
 * which is wrapped so that `.describe()` on the result, which builds a new
 * schema, keeps them.
 */
function question<T>(
  answer: z.ZodTypeAny,
  kind: 'choice' | 'score' | 'noul',
  keywords: Record<string, unknown>
): QuestionSchema<T> {
  const annotations: Record<string, unknown> = { [KIND_KEYWORD]: kind };
  for (const [key, value] of Object.entries(keywords)) {
    if (value !== undefined) annotations[key] = value;
  }
  return annotateSchema(answer, annotations).refine(
    () => true
  ) as unknown as QuestionSchema<T>;
}

/** The non-null guidance values, keyed as on the wire, or undefined. */
function guidanceOf(
  guidance: Record<string | number, unknown> | undefined
): Record<string, unknown> | undefined {
  if (!guidance) return undefined;
  const wire: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(guidance)) {
    if (value !== undefined && value !== null) setOwn(wire, key, value);
  }
  return Object.keys(wire).length > 0 ? wire : undefined;
}

/**
 * The options of a choice answer from most to least likely, ties broken by
 * name. It is empty when the answer carries no distribution.
 */
export function ranked<K extends string>(answer: ChoiceAnswer<K>): K[] {
  const probabilities = (answer.probabilities ?? {}) as Record<K, number>;
  return (Object.keys(probabilities) as K[]).sort(
    (a, b) =>
      probabilities[b] - probabilities[a] || (a < b ? -1 : a > b ? 1 : 0)
  );
}

/**
 * The probability gap between the two most likely options: how decisive the
 * choice is, where `confidence` reads the whole distribution. It is zero with
 * fewer than two options in the distribution.
 */
export function margin(answer: ChoiceAnswer): number {
  const order = ranked(answer);
  if (order.length < 2) return 0;
  const probabilities = answer.probabilities as Record<string, number>;
  return probabilities[order[0]] - probabilities[order[1]];
}

/**
 * The nearest whole level of a score answer, clamped to the rubric: the
 * levels given, or the answer's legend, which the plugin always fills.
 */
export function level(answer: ScoreAnswer, levels?: string[]): number {
  const n = levels?.length ?? Object.keys(answer.legend ?? {}).length;
  if (n === 0) return 0;
  return Math.min(Math.max(Math.round(answer.score), 0), n - 1);
}

/**
 * The rubric's description of {@link level}: from the levels given, or from
 * the answer's legend.
 */
export function label(answer: ScoreAnswer, levels?: string[]): string {
  const n = levels?.length ?? Object.keys(answer.legend ?? {}).length;
  if (n === 0) return '';
  const i = level(answer, levels);
  return levels ? levels[i] : (answer.legend?.[String(i)] ?? '');
}

/**
 * One question of a set built at run time, for questions whose options,
 * levels, or instructions come from data: the tools or products on hand, the
 * nodes of a taxonomy, a tenant's own categories. {@link schema} turns a set
 * of them into an output schema. A question known when the code is written
 * reads better as a field of a decision schema.
 */
export type Question = ChoiceQuestion | ScoreQuestion | NoulQuestion;

/** A runtime question that asks for one of its options, in the order given. */
export interface ChoiceQuestion {
  type: 'choice';
  /** The question: a string, or any JSON value. */
  instructions?: unknown;
  options: ChoiceOption[];
}

/**
 * One option of a {@link ChoiceQuestion}: the name that is the answer when it
 * is chosen, what it means, and optionally the structured form of that
 * meaning. An option with neither criteria nor guidance is described by its
 * name.
 */
export interface ChoiceOption {
  name: string;
  criteria?: string;
  guidance?: unknown;
}

/**
 * A runtime question that rates the state on its levels, lowest first, with
 * guidance keyed by level index.
 */
export interface ScoreQuestion {
  type: 'score';
  /** The question: a string, or any JSON value. */
  instructions?: unknown;
  levels: string[];
  guidance?: Record<number, unknown>;
}

/**
 * A runtime question that asks whether a statement is true, with what yes and
 * no mean: each side needs a string or guidance, or neither does.
 */
export interface NoulQuestion extends YesNo {
  type: 'noul';
  /** The question: a string, or any JSON value. */
  instructions?: unknown;
}

/**
 * The output schema that asks the given questions, each answered under its
 * key. Pass it as `output.jsonSchema` and read the answers as a record of
 * {@link Answer}:
 *
 * ```ts
 * const { output } = await ai.generate({
 *   model: 'typesafe/jev-1.13.0',
 *   prompt: request,
 *   output: {
 *     jsonSchema: systemOne.schema({
 *       tool: {
 *         type: 'choice',
 *         instructions: 'Which tool serves the request?',
 *         options: tools.map((t) => ({ name: t.name, criteria: t.description })),
 *       },
 *     }),
 *   },
 * });
 * const tool = (output as Record<string, Answer>).tool.choice;
 * ```
 *
 * A string instruction is the question's description; any other JSON value,
 * such as an object that describes the field of the state a question is about,
 * rides on the `x-instructions` keyword.
 */
export function schema(questions: Record<string, Question>): JSONSchema {
  const ids = Object.keys(questions).sort();
  const properties: Record<string, JSONSchema> = {};
  for (const id of ids) {
    const q = own(questions, id) as Question | undefined;
    // A missing question is left unmarked, so the request is refused as
    // asking nothing.
    setOwn(properties, id, q ? runtimeQuestionSchema(q) : {});
  }
  return {
    type: 'object',
    properties,
    required: ids,
    additionalProperties: false,
  };
}

function runtimeQuestionSchema(q: Question): JSONSchema {
  let s: JSONSchema;
  switch (q.type) {
    case 'choice':
      s = choiceSchema(q.options);
      break;
    case 'score':
      s = scoreSchema(q.levels, q.guidance);
      break;
    case 'noul':
      s = noulSchema(q);
      break;
    default:
      return {};
  }
  if (typeof q.instructions === 'string') {
    s.description = q.instructions;
  } else if (q.instructions !== undefined && q.instructions !== null) {
    s[INSTRUCTIONS_KEYWORD] = q.instructions;
  }
  return s;
}

function answerSchema(
  kind: 'choice' | 'score' | 'noul',
  properties: Record<string, JSONSchema>,
  required: string[],
  keywords: Record<string, unknown> = {}
): JSONSchema {
  const s: JSONSchema = {
    type: 'object',
    properties,
    required,
    additionalProperties: false,
    [KIND_KEYWORD]: kind,
  };
  for (const [key, value] of Object.entries(keywords)) {
    if (value !== undefined) s[key] = value;
  }
  return s;
}

const probabilitiesSchema = () => ({
  type: 'object',
  additionalProperties: { type: 'number' },
});

function choiceSchema(options: ChoiceOption[]): JSONSchema {
  const guidance: Record<string, unknown> = {};
  for (const option of options ?? []) {
    if (option.guidance !== undefined && option.guidance !== null) {
      setOwn(guidance, option.name, option.guidance);
    }
  }
  return answerSchema(
    'choice',
    {
      choice: {
        type: 'string',
        oneOf: (options ?? []).map((option) => ({
          const: option.name,
          ...(option.criteria ? { description: option.criteria } : {}),
        })),
      },
      probabilities: probabilitiesSchema(),
      confidence: { type: 'number' },
    },
    ['choice'],
    { [GUIDANCE_KEYWORD]: guidanceOf(guidance) }
  );
}

function scoreSchema(
  levels: string[],
  guidance: Record<number, unknown> | undefined
): JSONSchema {
  return answerSchema(
    'score',
    {
      score: {
        type: 'number',
        minimum: 0,
        maximum: Math.max((levels ?? []).length - 1, 0),
      },
      probabilities: probabilitiesSchema(),
      confidence: { type: 'number' },
      legend: { type: 'object', additionalProperties: { type: 'string' } },
    },
    ['score'],
    {
      [LEVELS_KEYWORD]: levels ? [...levels] : levels,
      [GUIDANCE_KEYWORD]: guidanceOf(guidance),
    }
  );
}

function noulSchema(q: NoulQuestion): JSONSchema {
  return answerSchema(
    'noul',
    { noul: { type: 'number', minimum: 0, maximum: 1 } },
    ['noul'],
    noulKeywords(q)
  );
}

/**
 * The answer to a question of a {@link schema}, with the fields of its kind
 * set: a choice sets `choice`, `probabilities` over the options, and
 * `confidence`; a score sets `score`, `probabilities` keyed by level number,
 * `confidence`, and `legend`; a noul sets `noul` alone. They mean what the same
 * fields of {@link ChoiceAnswer}, {@link ScoreAnswer}, and {@link NoulAnswer}
 * mean, and {@link answerKind} says which kind an answer is.
 */
export interface Answer {
  /**
   * The kind of question it answers, as Genkit Go and Dart write an answer.
   * The model's answers leave it out; {@link answerKind} reads it from the
   * fields.
   */
  type?: 'choice' | 'score' | 'noul';
  choice?: string;
  score?: number;
  noul?: number;
  probabilities?: Record<string, number>;
  confidence?: number;
  legend?: Record<string, string>;
}

/** The kind of an answer, read from the field every answer of a kind has. */
export function answerKind(
  answer: Answer
): 'choice' | 'score' | 'noul' | undefined {
  if (answer.type) return answer.type;
  if (answer.choice !== undefined) return 'choice';
  if (answer.score !== undefined) return 'score';
  if (answer.noul !== undefined) return 'noul';
  return undefined;
}

/**
 * Marks the info a decision model's response carries, so {@link responseInfo}
 * reads no other model's.
 */
export const INFO_MARKER = 'systemOneInfo';

/**
 * What a decision model's response carries beside the answers in the message:
 * the model version that answered, which is what confidence thresholds are
 * tuned against, a gateway's provider and generation ID, and the answers as
 * the API sent them, guidance echoes and fields the answer schemas do not
 * declare included. {@link responseInfo} reads it.
 */
export interface Info {
  model: string;
  provider?: string;
  id?: string;
  answers?: Record<string, Record<string, unknown>>;
}

/**
 * Reads the {@link Info} of a decision model's response, from a
 * `GenerateResponse` or from the JSON of one, such as `response.toJSON()`, a
 * flow's output, or a stored trace, which carry it as `custom`. It is
 * `{model: ''}` for a response from any other model, whatever its own raw
 * output holds, since the plugin marks its own.
 */
export function responseInfo(
  response: { raw?: unknown; custom?: unknown } | undefined | null
): Info {
  const raw = (response?.raw ?? response?.custom) as Partial<Info> | undefined;
  if (
    !raw ||
    typeof raw !== 'object' ||
    (raw as Record<string, unknown>)[INFO_MARKER] !== true ||
    typeof raw.model !== 'string'
  ) {
    return { model: '' };
  }
  const info: Info = { model: raw.model };
  if (typeof raw.provider === 'string' && raw.provider) {
    info.provider = raw.provider;
  }
  if (typeof raw.id === 'string' && raw.id) info.id = raw.id;
  if (
    raw.answers &&
    typeof raw.answers === 'object' &&
    !Array.isArray(raw.answers)
  ) {
    info.answers = raw.answers;
  }
  return info;
}
