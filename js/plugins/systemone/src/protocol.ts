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

/**
 * The System One protocol that decision models speak: a state and a set of
 * typed questions go out, and one typed answer per question comes back. This
 * module is internal to the plugin; the question helpers callers declare
 * decisions with are in `questions.ts`.
 *
 * @module
 */

import { GenkitError } from 'genkit';

/**
 * The schema keywords a question is encoded with. The output schema is the
 * one channel the generate loop hands a model besides the messages, so the
 * question set rides on it: each property is one question, its description is
 * the instructions, and these keywords carry what plain JSON Schema has no
 * keyword for. {@link compileQuestions} reads them back.
 */
export const KIND_KEYWORD = 'x-systemone';
/** Carries a score question's ordered level descriptions. */
export const LEVELS_KEYWORD = 'x-levels';
/** Carry a noul question's optional criteria for the yes and the no side. */
export const TRUE_KEYWORD = 'x-true';
export const FALSE_KEYWORD = 'x-false';
/**
 * Carries a question's structured guidance, keyed as on the wire: by option,
 * by level index, or by "true" and "false".
 */
export const GUIDANCE_KEYWORD = 'x-guidance';
/**
 * Carries a runtime question's instructions when they are structured rather
 * than text, which a description cannot hold.
 */
export const INSTRUCTIONS_KEYWORD = 'x-instructions';

/** The question types, as the API names them. */
export type QuestionKind = 'choice' | 'score' | 'noul';

/** One question on the wire. */
export interface WireQuestion {
  type: QuestionKind;
  /** The question's text, or a runtime question's structured value. */
  instructions: unknown;
  /**
   * A choice question's options and their descriptions, in order; a noul
   * question's map of side to description; a score question's ordered level
   * descriptions. A description is a string, or the guidance that replaces it
   * on the wire.
   */
  criteria?: unknown;
}

/** A compiled question and what the answer projection needs from it. */
export interface CompiledQuestion {
  wire: WireQuestion;
  /** A score question's level strings, which its legend is built from. */
  labels?: string[];
}

/** The questions of one request, keyed by question ID. */
export type CompiledQuestions = Record<string, CompiledQuestion>;

/**
 * A choice question's options and their descriptions, in the order they were
 * declared. A JS object would put integer-like names first, so the options are
 * pairs, and the request body writes them as a JSON object whose keys keep
 * that order. A description is a string, or the guidance that replaces it on
 * the wire.
 */
export class Options {
  constructor(readonly entries: [string, unknown][]) {}

  /** The description of an option, or undefined. */
  get(name: string): unknown {
    return this.entries.find(([key]) => key === name)?.[1];
  }

  /** The option names, in order. */
  names(): string[] {
    return this.entries.map(([key]) => key);
  }
}

/** The ID of the one question an enum-format request asks. */
export const ENUM_QUESTION_ID = 'choice';

type JsonObject = Record<string, any>;

function isObject(value: unknown): value is JsonObject {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function invalidSchema(message: string): GenkitError {
  return new GenkitError({
    status: 'INVALID_ARGUMENT',
    message: `systemone: ${message}`,
  });
}

/**
 * Reads the questions an output schema encodes: one per property, each marked
 * with the `x-systemone` keyword the question helpers emit. A property without
 * the marker is rejected, since the model has no way to answer a shape it does
 * not know. The preamble, the request's system text, goes in front of every
 * question's own instructions, followed by the schema's own description when
 * it has one, as it is for the enum format.
 */
export function compileQuestions(
  schema: JsonObject | undefined,
  preamble: string
): CompiledQuestions {
  const props = isObject(schema?.properties) ? schema.properties : {};
  if (Object.keys(props).length === 0) {
    throw invalidSchema(
      'the output schema has no properties; each question is a field of the output schema'
    );
  }
  const description =
    typeof schema?.description === 'string' ? schema.description : '';
  const lead = joinInstructions(preamble, description);
  const questions: CompiledQuestions = {};
  for (const [id, raw] of Object.entries(props)) {
    const prop = resolveRef(schema, isObject(raw) ? raw : {});
    const kind = prop[KIND_KEYWORD];
    if (typeof kind !== 'string' || kind === '') {
      throw invalidSchema(
        `field "${id}" is not a question; use a choice(), score(), or noul() field`
      );
    }
    const instructions = questionInstructions(id, prop, lead);
    switch (kind) {
      case 'choice':
        setOwn(questions, id, {
          wire: {
            type: kind,
            instructions,
            criteria: choiceCriteria(id, prop),
          },
        });
        break;
      case 'score': {
        const { criteria, labels } = scoreCriteria(id, prop);
        setOwn(questions, id, {
          wire: { type: kind, instructions, criteria },
          labels,
        });
        break;
      }
      case 'noul': {
        const criteria = noulCriteria(id, prop);
        setOwn(questions, id, {
          wire:
            criteria === undefined
              ? { type: kind, instructions }
              : { type: kind, instructions, criteria },
        });
        break;
      }
      default:
        throw invalidSchema(`question "${id}" has unknown type "${kind}"`);
    }
  }
  return questions;
}

/**
 * A property with its `$ref` resolved: Zod writes a schema used twice, such
 * as one `choice()` behind two fields, as a reference to the first use. The
 * property's own keywords, its description and the question keywords among
 * them, win over the target's.
 */
function resolveRef(
  root: JsonObject | undefined,
  prop: JsonObject
): JsonObject {
  const ref = prop.$ref;
  if (typeof ref !== 'string' || !ref.startsWith('#')) return prop;
  let target: unknown = root;
  for (const token of ref.slice(1).split('/').slice(1)) {
    const key = decodeURIComponent(token)
      .replace(/~1/g, '/')
      .replace(/~0/g, '~');
    target =
      isObject(target) && Object.hasOwn(target, key) ? target[key] : undefined;
  }
  if (!isObject(target)) return prop;
  const { $ref: _ref, ...own } = prop;
  return { ...resolveRef(root, target), ...own };
}

/**
 * Reads a question's instructions: the structured value a runtime question
 * carries on the `x-instructions` keyword, or the property's description. The
 * preamble goes in front: as a paragraph of the text, as the first element of
 * an array, or as the first element beside any other structured value.
 */
function questionInstructions(
  id: string,
  prop: JsonObject,
  preamble: string
): unknown {
  const structured = prop[INSTRUCTIONS_KEYWORD];
  if (structured !== undefined && structured !== null) {
    if (preamble === '') return structured;
    if (Array.isArray(structured)) return [preamble, ...structured];
    return [preamble, structured];
  }
  const text = typeof prop.description === 'string' ? prop.description : '';
  if (text === '') {
    throw invalidSchema(
      `question "${id}" has no instructions; describe the field with .describe(), or set instructions on a runtime question`
    );
  }
  return joinInstructions(preamble, text);
}

/**
 * Reads the options back out of the constants a choice lists, in order, with
 * the guidance of an option in place of its string. A runtime question lists
 * them in a oneOf, as Genkit Go does; Zod lists the literals of `choice()` in
 * an anyOf. An option with neither a description nor guidance is described by
 * its own name, since the wire format takes a description per option and some
 * gateways reject a null one. The options are an object whose keys keep their
 * order. Zod writes the literals as an enum when none of them has a
 * description, which reads as options with no description.
 */
function choiceCriteria(id: string, prop: JsonObject): Options {
  const choice = prop.properties?.choice;
  const oneOf =
    choice?.oneOf ??
    choice?.anyOf ??
    (Array.isArray(choice?.enum)
      ? choice.enum.map((value: unknown) => ({ const: value }))
      : undefined);
  if (!Array.isArray(oneOf) || oneOf.length === 0) {
    throw invalidSchema(`choice question "${id}" lists no options`);
  }
  const guidance = isObject(prop[GUIDANCE_KEYWORD])
    ? prop[GUIDANCE_KEYWORD]
    : {};
  const options: [string, unknown][] = [];
  const seen = new Set<string>();
  for (const raw of oneOf) {
    const key = isObject(raw) ? raw.const : undefined;
    if (typeof key !== 'string' || key === '') {
      throw invalidSchema(
        `choice question "${id}" has an option without a const`
      );
    }
    if (seen.has(key)) {
      throw invalidSchema(
        `choice question "${id}" lists option "${key}" twice`
      );
    }
    seen.add(key);
    const description =
      typeof raw.description === 'string' ? raw.description : '';
    const value = withWhat(own(guidance, key), description);
    options.push([key, value === '' ? key : value]);
  }
  for (const key of Object.keys(guidance).sort()) {
    if (!seen.has(key)) {
      throw invalidSchema(
        `choice question "${id}" has guidance for "${key}", which is not one of its options`
      );
    }
  }
  return new Options(options);
}

/**
 * Reads the ordered levels off the `x-levels` keyword, with the guidance of a
 * level in place of its string, and returns the strings too for the legend.
 * The API takes two to ten levels; the lower bound is checked here because one
 * level is not a scale, the upper bound is the API's to enforce.
 */
function scoreCriteria(
  id: string,
  prop: JsonObject
): { criteria: unknown[]; labels: string[] } {
  const levels = prop[LEVELS_KEYWORD];
  if (
    levels !== undefined &&
    (!Array.isArray(levels) || levels.some((l) => typeof l !== 'string'))
  ) {
    throw invalidSchema(
      `score question "${id}" has a level that is not a string`
    );
  }
  const labels: string[] = levels ?? [];
  if (labels.length < 2) {
    throw invalidSchema(`score question "${id}" needs at least two levels`);
  }
  const criteria: unknown[] = [...labels];
  const guidance = isObject(prop[GUIDANCE_KEYWORD])
    ? prop[GUIDANCE_KEYWORD]
    : {};
  for (const key of Object.keys(guidance).sort()) {
    const i = /^\d+$/.test(key) ? Number(key) : -1;
    if (i < 0 || i >= labels.length) {
      throw invalidSchema(
        `score question "${id}" has guidance for level ${key}, which its rubric does not have`
      );
    }
    criteria[i] = withWhat(own(guidance, key), labels[i]);
  }
  return { criteria, labels };
}

/**
 * Reads the optional yes and no criteria of a noul question, with the guidance
 * of a side in place of its string. They are sent as a pair or not at all: the
 * API describes the pair, and one side alone would leave the other implied. No
 * criteria is undefined, so the field stays absent rather than null, which the
 * gateways reject.
 */
function noulCriteria(id: string, prop: JsonObject): JsonObject | undefined {
  const guidance = isObject(prop[GUIDANCE_KEYWORD])
    ? prop[GUIDANCE_KEYWORD]
    : {};
  const criteria: JsonObject = {};
  for (const [side, keyword] of [
    ['true', TRUE_KEYWORD],
    ['false', FALSE_KEYWORD],
  ] as const) {
    const text = typeof prop[keyword] === 'string' ? prop[keyword] : '';
    const value = withWhat(own(guidance, side), text);
    if (value !== '') criteria[side] = value;
  }
  switch (Object.keys(criteria).length) {
    case 0:
      return undefined;
    case 1:
      throw invalidSchema(
        `noul question "${id}" says what only one side means; give both yes and no, as criteria or guidance`
      );
  }
  return criteria;
}

/** An own property of an object, never one it inherits. */
export function own(object: JsonObject, key: string): unknown {
  return Object.hasOwn(object, key) ? object[key] : undefined;
}

/**
 * Sets an own data property. A question ID or an option name is any string,
 * and assigning `__proto__` on a plain object would set its prototype
 * instead of adding the key.
 */
export function setOwn(object: JsonObject, key: string, value: unknown) {
  Object.defineProperty(object, key, {
    value,
    enumerable: true,
    writable: true,
    configurable: true,
  });
}

/**
 * A description as it goes on the wire: the guidance when there is some, the
 * string otherwise. An object with no "what" gets the string as its what, so
 * guidance adds to the description rather than replacing it; any other
 * guidance is sent as it is.
 */
function withWhat(guidance: unknown, description: string): unknown {
  if (guidance === undefined || guidance === null) return description;
  if (isObject(guidance)) {
    if ('what' in guidance || description === '') return guidance;
    return { ...guidance, what: description };
  }
  return guidance;
}

/**
 * Turns an enum output schema into a single choice question, so the built-in
 * enum format works on the model with no decision schema. The instructions are
 * the preamble, the request's system text, followed by the schema's
 * description when it has one; the options keep the order the values were
 * given in, and an option's description is its own name, since an enum carries
 * none.
 */
export function enumQuestion(
  schema: JsonObject | undefined,
  preamble: string
): CompiledQuestions {
  const values = schema?.enum;
  if (
    !Array.isArray(values) ||
    values.some((v) => typeof v !== 'string' || v === '')
  ) {
    throw invalidSchema('enum output values must be non-empty strings');
  }
  if (values.length === 0) {
    throw invalidSchema('the enum output schema lists no values');
  }
  const seen = new Set<string>();
  for (const value of values as string[]) {
    if (seen.has(value)) {
      throw invalidSchema(`the enum output schema lists "${value}" twice`);
    }
    seen.add(value);
  }
  const options = new Options((values as string[]).map((v) => [v, v]));
  const description =
    typeof schema?.description === 'string' ? schema.description : '';
  const instructions =
    joinInstructions(preamble, description) ||
    'Choose the option that best describes the state.';
  return {
    [ENUM_QUESTION_ID]: {
      wire: { type: 'choice', instructions, criteria: options },
    },
  };
}

/**
 * Puts the preamble in front of a question's own text as two paragraphs, or
 * returns whichever of the two is present.
 */
export function joinInstructions(preamble: string, text: string): string {
  if (preamble === '') return text;
  if (text === '') return preamble;
  return `${preamble}\n\n${text}`;
}

/** The wire questions of a compiled set, as the request body takes them. */
export function wireQuestions(
  questions: CompiledQuestions
): Record<string, WireQuestion> {
  return Object.fromEntries(
    Object.entries(questions).map(([id, q]) => [id, q.wire])
  );
}

/**
 * The fields of a wire answer per question type, which are the fields the
 * answer schemas declare. The answer schemas are closed, so a field the API or
 * a gateway adds later is dropped here rather than failing every call at
 * validation; the untouched answer is still on the response's info.
 */
const ANSWER_FIELDS: Record<QuestionKind, string[]> = {
  choice: ['choice', 'probabilities', 'confidence'],
  score: ['score', 'probabilities', 'confidence', 'legend'],
  noul: ['noul'],
};

/**
 * Renders the answers as the message text: for the enum format the chosen
 * option itself, otherwise a JSON object keyed by question ID whose values are
 * the answers projected onto the fields the output schema declares. A score's
 * legend is rebuilt from the rubric's strings: for a level sent with guidance
 * the API echoes the guidance, which the legend's strings cannot hold. A score
 * is clamped to the rubric, since float rounding in the expected level can
 * land a hair past the top level, which the schema's bounds would reject along
 * with every other answer in the call.
 */
export function answersText(
  answers: Record<string, unknown>,
  questions: CompiledQuestions,
  isEnum: boolean
): string {
  const projected: Record<string, JsonObject> = {};
  for (const id of Object.keys(questions).sort()) {
    const answer = own(answers, id);
    if (!isObject(answer)) {
      throw new GenkitError({
        status: 'INTERNAL',
        message: `systemone: no answer for question "${id}"`,
      });
    }
    const q = questions[id];
    const fields: JsonObject = {};
    for (const field of ANSWER_FIELDS[q.wire.type]) {
      if (field in answer) fields[field] = answer[field];
    }
    if (q.labels) {
      fields.legend = Object.fromEntries(
        q.labels.map((label, i) => [String(i), label])
      );
      if (typeof fields.score === 'number') {
        fields.score = Math.min(Math.max(fields.score, 0), q.labels.length - 1);
      }
    }
    setOwn(projected, id, fields);
  }
  if (isEnum) {
    const choice = projected[ENUM_QUESTION_ID]?.choice;
    if (typeof choice !== 'string' || choice === '') {
      throw new GenkitError({
        status: 'INTERNAL',
        message: 'systemone: the enum answer names no option',
      });
    }
    return choice;
  }
  return JSON.stringify(projected);
}
