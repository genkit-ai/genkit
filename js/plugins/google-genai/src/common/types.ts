/**
 * Copyright 2025 Google LLC
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

import { z } from 'genkit';

/** Function calling mode. */
export enum FunctionCallingMode {
  /** Unspecified function calling mode. This value should not be used. */
  MODE_UNSPECIFIED = 'MODE_UNSPECIFIED',
  /**
   * Default model behavior, model decides to predict either function calls
   * or natural language response.
   */
  AUTO = 'AUTO',
  /**
   * Model is constrained to always predicting function calls only.
   * If "allowedFunctionNames" are set, the predicted function calls will be
   * limited to any one of "allowedFunctionNames", else the predicted
   * function calls will be any one of the provided "function_declarations".
   */
  ANY = 'ANY',
  /**
   * Model will not predict any function calls. Model behavior is same as when
   * not passing any function declarations.
   */
  NONE = 'NONE',
}

export function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

/**
 * The reason why the response is blocked.
 */
export enum BlockReason {
  /** Unspecified block reason. */
  BLOCKED_REASON_UNSPECIFIED = 'BLOCKED_REASON_UNSPECIFIED', // GoogleAI
  BLOCK_REASON_UNSPECIFIED = 'BLOCK_REASON_UNSPECIFIED', // VertexAI

  /** Candidates blocked due to safety. */
  SAFETY = 'SAFETY',
  /** Candidates blocked due to other reason. */
  OTHER = 'OTHER',
  /** terminology blocklist. */
  BLOCKLIST = 'BLOCKLIST',
  /** Candidates blocked due to prohibited content. */
  PROHIBITED_CONTENT = 'PROHIBITED_CONTENT',
}

/**
 * Harm categories that would cause prompts or candidates to be blocked.
 * @public
 */
export enum HarmCategory {
  HARM_CATEGORY_UNSPECIFIED = 'HARM_CATEGORY_UNSPECIFIED',
  HARM_CATEGORY_HATE_SPEECH = 'HARM_CATEGORY_HATE_SPEECH',
  HARM_CATEGORY_SEXUALLY_EXPLICIT = 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
  HARM_CATEGORY_HARASSMENT = 'HARM_CATEGORY_HARASSMENT',
  HARM_CATEGORY_DANGEROUS_CONTENT = 'HARM_CATEGORY_DANGEROUS_CONTENT',
  HARM_CATEGORY_CIVIC_INTEGRITY = 'HARM_CATEGORY_CIVIC_INTEGRITY',
}

/**
 * Probability based thresholds levels for blocking.
 */
export enum HarmBlockThreshold {
  /** Unspecified harm block threshold. */
  HARM_BLOCK_THRESHOLD_UNSPECIFIED = 'HARM_BLOCK_THRESHOLD_UNSPECIFIED',
  /** Block low threshold and above (i.e. block more). */
  BLOCK_LOW_AND_ABOVE = 'BLOCK_LOW_AND_ABOVE',
  /** Block medium threshold and above. */
  BLOCK_MEDIUM_AND_ABOVE = 'BLOCK_MEDIUM_AND_ABOVE',
  /** Block only high threshold (i.e. block less). */
  BLOCK_ONLY_HIGH = 'BLOCK_ONLY_HIGH',
  /** Block none. */
  BLOCK_NONE = 'BLOCK_NONE',
  /** Turn off the safety filter. */
  OFF = 'OFF',
}

/**
 * Probability that a prompt or candidate matches a harm category.
 * @public
 */
export enum HarmProbability {
  /** Probability is unspecified. */
  HARM_PROBABILITY_UNSPECIFIED = 'HARM_PROBABILITY_UNSPECIFIED',
  /** Content has a negligible chance of being unsafe. */
  NEGLIGIBLE = 'NEGLIGIBLE',
  /** Content has a low chance of being unsafe. */
  LOW = 'LOW',
  /** Content has a medium chance of being unsafe. */
  MEDIUM = 'MEDIUM',
  /** Content has a high chance of being unsafe. */
  HIGH = 'HIGH',
}

/**
 * The mode of the predictor to be used in dynamic retrieval.
 * @public
 */
export enum DynamicRetrievalMode {
  // Unspecified function calling mode. This value should not be used.
  MODE_UNSPECIFIED = 'MODE_UNSPECIFIED',
  // Run retrieval only when system decides it is necessary.
  MODE_DYNAMIC = 'MODE_DYNAMIC',
}

/**
 * Specifies the dynamic retrieval configuration for the given source.
 * @public
 */
export declare interface DynamicRetrievalConfig {
  /**
   * The mode of the predictor to be used in dynamic retrieval.
   */
  mode?: DynamicRetrievalMode;
  /**
   * The threshold to be used in dynamic retrieval. If not set, a system default
   * value is used.
   */
  dynamicThreshold?: number;
}

/**
 * Defines a retrieval tool that model can call to access external knowledge.
 */
export declare interface GoogleSearchRetrievalTool {
  /** Optional. {@link GoogleSearchRetrieval}. */
  googleSearchRetrieval?: GoogleSearchRetrieval;
  googleSearch?: GoogleSearchRetrieval;
}
export function isGoogleSearchRetrievalTool(
  tool: Tool
): tool is GoogleSearchRetrievalTool {
  return (
    (tool as GoogleSearchRetrievalTool).googleSearchRetrieval !== undefined ||
    (tool as GoogleSearchRetrievalTool).googleSearch !== undefined
  );
}

export declare interface UrlContextTool {
  urlContext?: {};
}

/**
 * The FileSearch tool that retrieves knowledge from Semantic Retrieval corpora.
 * Files are imported to Semantic Retrieval corpora using the ImportFile API
 */
export declare interface FileSearchTool {
  fileSearch: FileSearch;
}

export declare interface FileSearch {
  /**
   * The names of the fileSearchStores to retrieve from.
   * Example: fileSearchStores/my-file-search-store-123
   */
  fileSearchStoreNames: string[];
  /**
   * Metadata filter to apply to the semantic retrieval documents and chunks.
   */
  metadataFilter?: string;
  /**
   * The number of semantic retrieval chunks to retrieve.
   */
  topK?: number;
}
export function isFileSearchTool(tool: Tool): tool is FileSearchTool {
  return (tool as FileSearchTool).fileSearch !== undefined;
}

/**
 * Grounding support.
 */
export declare interface GroundingSupport {
  /** Optional. Segment of the content this support belongs to. */
  segment?: GroundingSupportSegment;
  /**
   * Optional. A array of indices (into {@link GroundingChunk}) specifying the
   * citations associated with the claim. For instance [1,3,4] means
   * that grounding_chunk[1], grounding_chunk[3],
   * grounding_chunk[4] are the retrieved content attributed to the claim.
   */
  groundingChunkIndices?: number[];
  /**
   * Confidence score of the support references. Ranges from 0 to 1. 1 is the
   * most confident. This list must have the same size as the
   * groundingChunkIndices.
   */
  confidenceScores?: number[];
}

/**
 * Grounding support segment.
 */
export declare interface GroundingSupportSegment {
  /** Optional. The index of a Part object within its parent Content object. */
  partIndex?: number;
  /**
   * Optional. Start index in the given Part, measured in bytes.
   * Offset from the start of the Part, inclusive, starting at zero.
   */
  startIndex?: number;
  /**
   * Optional. End index in the given Part, measured in bytes.
   * Offset from the start of the Part, exclusive, starting at zero.
   */
  endIndex?: number;
  /** Optional. The text corresponding to the segment from the response. */
  text?: string;
}

/**
 * Harm severity levels
 */
export enum HarmSeverity {
  /** Harm severity unspecified. */
  HARM_SEVERITY_UNSPECIFIED = 'HARM_SEVERITY_UNSPECIFIED',
  /** Negligible level of harm severity. */
  HARM_SEVERITY_NEGLIGIBLE = 'HARM_SEVERITY_NEGLIGIBLE',
  /** Low level of harm severity. */
  HARM_SEVERITY_LOW = 'HARM_SEVERITY_LOW',
  /** Medium level of harm severity. */
  HARM_SEVERITY_MEDIUM = 'HARM_SEVERITY_MEDIUM',
  /** High level of harm severity. */
  HARM_SEVERITY_HIGH = 'HARM_SEVERITY_HIGH',
}

/**
 * Safety rating corresponding to the generated content.
 */
export declare interface SafetyRating {
  /** The harm category. {@link HarmCategory} */
  category?: HarmCategory;
  /** The harm probability. {@link HarmProbability} */
  probability?: HarmProbability;
  /** The harm probability score. */
  probabilityScore?: number;
  /** The harm severity.level {@link HarmSeverity} */
  severity?: HarmSeverity;
  /** The harm severity score. */
  severityScore?: number;
}

/**
 * If the prompt was blocked, this will be populated with `blockReason` and
 * the relevant `safetyRatings`.
 */
export declare interface PromptFeedback {
  /** The reason why the response is blocked. {@link BlockReason}. */
  blockReason: BlockReason;
  /** Array of {@link SafetyRating} */
  safetyRatings: SafetyRating[];
  /** A readable block reason message. */
  blockReasonMessage?: string;
}

/**
 * URI based data.
 */
export declare interface FileData {
  /** The IANA standard MIME type of the source data. */
  mimeType: string;
  /** URI of the file. */
  fileUri: string;
}

/**
 * Raw media bytes sent directly in the request. Text should not be sent as
 * raw bytes.
 */
export declare interface GenerativeContentBlob {
  /**
   * The MIME type of the source data. The only accepted values: "image/png" or
   * "image/jpeg".
   */
  mimeType: string;
  /** Base64 encoded data. */
  data: string;
}

/**
 * A predicted FunctionCall returned from the model that contains a string
 * representating the FunctionDeclaration.name with the parameters and their
 * values.
 */
export declare interface FunctionCall {
  /**
   * The unique id of the function call. If populated, the client to execute the
   * `function_call` and return the response with the matching `id`.
   */
  id?: string;
  /** The name of the function specified in FunctionDeclaration.name. */
  name?: string;
  /** The arguments to pass to the function. */
  args?: object;
  /** Optional. The partial argument value of the function call. If provided, represents the arguments/fields that are streamed incrementally. */
  partialArgs?: PartialArg[];
  /** Optional. Whether this is the last part of the FunctionCall. If true, another partial message for the current FunctionCall is expected to follow. */
  willContinue?: boolean;
}

/** Partial argument value of the function call. This data type is not supported in Gemini API. */
export declare interface PartialArg {
  /** Optional. Represents a null value. */
  nullValue?: 'NULL_VALUE';
  /** Optional. Represents a double value. */
  numberValue?: number;
  /** Optional. Represents a string value. */
  stringValue?: string;
  /** Optional. Represents a boolean value. */
  boolValue?: boolean;
  /** Required. A JSON Path (RFC 9535) to the argument being streamed. https://datatracker.ietf.org/doc/html/rfc9535. e.g. "$.foo.bar[0].data". */
  jsonPath?: string;
  /** Optional. Whether this is not the last part of the same json_path. If true, another PartialArg message for the current json_path is expected to follow. */
  willContinue?: boolean;
}
/**
 * The result output of a FunctionCall that contains a string representing
 * the FunctionDeclaration.name and a structured JSON object containing any
 * output from the function call. It is used as context to the model.
 */
export declare interface FunctionResponse {
  /** Optional. The id of the function call this response is for. Populated by the client to match the corresponding function call `id`. */
  id?: string;
  /** The name of the function specified in FunctionDeclaration.name. */
  name: string;
  /** The expected response from the model. */
  response: object;
  /** List of parts that constitute a function response. Each part may
      have a different IANA MIME type. */
  parts?: FunctionResponsePart[];
}

/**
 * A datatype containing media that is part of a `FunctionResponse` message.
 *
 * A `FunctionResponsePart` consists of data which has an associated datatype. A
 * `FunctionResponsePart` can only contain one of the accepted types in
 * `FunctionResponsePart.data`.
 *
 * A `FunctionResponsePart` must have a fixed IANA MIME type identifying the
 * type and subtype of the media if the `inline_data` field is filled with raw
 * bytes.
 */
export class FunctionResponsePart {
  /** Optional. Inline media bytes. */
  inlineData?: FunctionResponseBlob;
}

/**
 * Raw media bytes for function response.
 *
 * Text should not be sent as raw bytes, use the FunctionResponse.response field.
 */
export class FunctionResponseBlob {
  /** Required. The IANA standard MIME type of the source data. */
  mimeType?: string;
  /** Required. Inline media bytes.
   * @remarks Encoded as base64 string. */
  data?: string;
  /** Optional. Display name of the blob.
      Used to provide a label or filename to distinguish blobs. */
  displayName?: string;
}

/**
 * The list of OpenAPI data types
 * as defined by https://swagger.io/docs/specification/data-models/data-types/
 */
export enum SchemaType {
  /** String type. */
  STRING = 'STRING',
  /** Number type. */
  NUMBER = 'NUMBER',
  /** Integer type. */
  INTEGER = 'INTEGER',
  /** Boolean type. */
  BOOLEAN = 'BOOLEAN',
  /** Array type. */
  ARRAY = 'ARRAY',
  /** Object type. */
  OBJECT = 'OBJECT',
}

export declare interface Schema {
  type?: SchemaType;
  format?: string;
  title?: string;
  description?: string;
  nullable?: boolean;
  items?: Schema;
  minItems?: number;
  maxItems?: number;
  properties?: Record<string, Schema>;
  enum?: string[];
  required?: string[];
  example?: unknown;
}

/**
 * Schema for parameters passed to {@link FunctionDeclaration.parameters}.
 */
export declare interface FunctionDeclarationSchema {
  /** The type of the parameter. */
  type: SchemaType;
  /** The format of the parameter. */
  properties: Record<string, Schema>;
  /** Optional. Description of the parameter. */
  description?: string;
  /** Optional. Array of required parameters. */
  required?: string[];
}

export declare interface FunctionDeclaration {
  /**
   * The name of the function to call. Must start with a letter or an
   * underscore. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with
   * a max length of 64.
   */
  name: string;
  /**
   * Optional. Description and purpose of the function. Model uses it to decide
   * how and whether to call the function.
   */
  description?: string;
  /**
   * Optional. Describes the parameters to this function in JSON Schema Object
   * format. Reflects the Open API 3.03 Parameter Object. string Key: the name
   * of the parameter. Parameter names are case sensitive. Schema Value: the
   * Schema defining the type used for the parameter. For function with no
   * parameters, this can be left unset.
   *
   * @example with 1 required and 1 optional parameter: type: OBJECT properties:
   * ```
   * param1:
   *
   *   type: STRING
   * param2:
   *
   *  type: INTEGER
   * required:
   *
   *   - param1
   * ```
   */
  parameters?: FunctionDeclarationSchema;
}

/**
 * Metadata on the generation request's token usage.
 */
export declare interface UsageMetadata {
  /** Optional. Number of tokens in the request. */
  promptTokenCount?: number;
  /** Optional. Number of tokens in the response(s). */
  candidatesTokenCount?: number;
  /** Optional. Total number of tokens. */
  totalTokenCount?: number;
  /** Optional. Number of tokens in the cached content. */
  cachedContentTokenCount?: number;
  /** Optional. Number of tokens present in thoughts output. */
  thoughtsTokenCount?: number;
}

export const TaskTypeSchema = z.enum([
  'RETRIEVAL_DOCUMENT',
  'RETRIEVAL_QUERY',
  'SEMANTIC_SIMILARITY',
  'CLASSIFICATION',
  'CLUSTERING',
  'CODE_RETRIEVAL_QUERY',
  'QUESTION_ANSWERING',
  'FACT_VERIFICATION',
]);

export type TaskType = z.infer<typeof TaskTypeSchema>;

/**
 * Reason that a candidate finished.
 * @public
 */
export enum FinishReason {
  // Default value. This value is unused.
  FINISH_REASON_UNSPECIFIED = 'FINISH_REASON_UNSPECIFIED',
  // Natural stop point of the model or provided stop sequence.
  STOP = 'STOP',
  // The maximum number of tokens as specified in the request was reached.
  MAX_TOKENS = 'MAX_TOKENS',
  // The candidate content was flagged for safety reasons.
  SAFETY = 'SAFETY',
  // The candidate content was flagged for recitation reasons.
  RECITATION = 'RECITATION',
  // The candidate content was flagged for using an unsupported language.
  LANGUAGE = 'LANGUAGE',
  // Token generation stopped because the content contains forbidden terms.
  BLOCKLIST = 'BLOCKLIST',
  // Token generation stopped for potentially containing prohibited content.
  PROHIBITED_CONTENT = 'PROHIBITED_CONTENT',
  // Token generation stopped because the content potentially contains Sensitive Personally Identifiable Information (SPII).
  SPII = 'SPII',
  // The function call generated by the model is invalid.
  MALFORMED_FUNCTION_CALL = 'MALFORMED_FUNCTION_CALL',
  // At least one thought signature from a previous call is missing.
  MISSING_THOUGHT_SIGNATURE = 'MISSING_THOUGHT_SIGNATURE',
  // Unknown reason.
  OTHER = 'OTHER',
}

/**
 * Represents a whole or partial calendar date, such as a birthday. The time of
 * day and time zone are either specified elsewhere or are insignificant. The
 * date is relative to the Gregorian Calendar. This can represent one of the
 * following:
 *
 *   A full date, with non-zero year, month, and day values.
 *   A month and day, with a zero year (for example, an anniversary).
 *   A year on its own, with a zero month and a zero day.
 *   A year and month, with a zero day (for example, a credit card expiration
 *   date).
 */
export declare interface GoogleDate {
  /**
   * Year of the date. Must be from 1 to 9999, or 0 to specify a date without a
   * year.
   */
  year?: number;
  /**
   * Month of the date. Must be from 1 to 12, or 0 to specify a year without a
   * month and day.
   */
  month?: number;
  /**
   * Day of the date. Must be from 1 to 31 and valid for the year and month.
   * or 0 to specify a year by itself or a year and month where the day isn't
   * significant
   */
  day?: number;
}

/**
 * Source attributions for content.
 */
export declare interface CitationSource {
  /** Optional. Start index into the content. */
  startIndex?: number;
  /** Optional. End index into the content. */
  endIndex?: number;
  /** Optional. Url reference of the attribution. */
  uri?: string;
  /** Optional. License of the attribution. */
  license?: string;
  /** Optional. Title of the attribution. VertexAI only.*/
  title?: string;
  /** Optional. Publication date of the attribution. VertexAI only */
  publicationDate?: GoogleDate;
}

/**
 * A collection of source attributions for a piece of content.
 */
export declare interface CitationMetadata {
  /** Array of {@link CitationSource}. */
  citations?: CitationSource[]; // VertexAI
  citationSources?: CitationSource[]; // GoogleAI
}

/**
 * Google search entry point.
 */
export declare interface SearchEntryPoint {
  /**
   * Optional. Web content snippet that can be embedded in a web page or an app
   * webview.
   */
  renderedContent?: string;
  /** Optional. Base64 encoded JSON representing array of tuple. */
  sdkBlob?: string;
}

/**
 * Grounding chunk from the web.
 */
export declare interface GroundingChunkWeb {
  /** Optional. URI reference of the grounding chunk. */
  uri?: string;
  /** Optional. Title of the grounding chunk. */
  title?: string;
}

/**
 * Grounding chunk from context retrieved by the retrieval tools.
 */
export declare interface GroundingChunkRetrievedContext {
  /** Optional. URI reference of the attribution. */
  uri?: string;
  /** Optional. Title of the attribution. */
  title?: string;
}

/**
 * Grounding chunk.
 */
export declare interface GroundingChunk {
  /** Optional. Grounding chunk from the web. */
  web?: GroundingChunkWeb;
  /**
   * Optional. Grounding chunk from context retrieved by the retrieval tools. (VertexAI only)
   */
  retrievedContext?: GroundingChunkRetrievedContext;
}

/**
 * Metadata related to retrieval in the grounding flow. GoogleAI only.
 * @public
 */
export declare interface RetrievalMetadata {
  /**
   * Score indicating how likely information from google search could help
   * answer the prompt. The score is in the range [0, 1], where 0 is the least
   * likely and 1 is the most likely. This score is only populated when google
   * search grounding and dynamic retrieval is enabled. It will becompared to
   * the threshold to determine whether to trigger google search.
   */
  googleSearchDynamicRetrievalScore?: number;
}

/**
 * A collection of grounding attributions for a piece of content.
 */
export declare interface GroundingMetadata {
  /** Optional. Google search entry for the following-up web searches. {@link SearchEntryPoint} */
  searchEntryPoint?: SearchEntryPoint;
  /**
   * Optional. Array of supporting references retrieved from specified
   * grounding source. {@link GroundingChunk}.
   */
  groundingChunks?: GroundingChunk[];
  /** Optional. Array of grounding support. {@link GroundingSupport}. */
  groundingSupports?: GroundingSupport[];
  /** Optional. Web search queries for the following-up web search. */
  webSearchQueries?: string[];
  /** Optional. Queries executed by the retrieval tools. VertexAI only*/
  retrievalQueries?: string[];
  /**
   * Optional. Metadata related to retrieval in the grounding flow. GoogleAI only.
   */
  retrievalMetadata?: RetrievalMetadata;
}

/**
 * @public
 */
export enum ExecutableCodeLanguage {
  LANGUAGE_UNSPECIFIED = 'LANGUAGE_UNSPECIFIED',
  PYTHON = 'PYTHON',
}

/**
 * Code generated by the model that is meant to be executed, where the result
 * is returned to the model.
 * Only generated when using the code execution tool, in which the code will
 * be automatically executed, and a corresponding `CodeExecutionResult` will
 * also be generated.
 *
 * @public
 */
export declare interface ExecutableCode {
  /**
   * Programming language of the `code`.
   */
  language: ExecutableCodeLanguage;
  /**
   * The code to be executed.
   */
  code: string;
}

/**
 * Possible outcomes of code execution.
 * @public
 */
export enum Outcome {
  /**
   * Unspecified status. This value should not be used.
   */
  OUTCOME_UNSPECIFIED = 'OUTCOME_UNSPECIFIED',
  /**
   * Code execution completed successfully.
   */
  OUTCOME_OK = 'OUTCOME_OK',
  /**
   * Code execution finished but with a failure. `stderr` should contain the
   * reason.
   */
  OUTCOME_FAILED = 'OUTCOME_FAILED',
  /**
   * Code execution ran for too long, and was cancelled. There may or may not
   * be a partial output present.
   */
  OUTCOME_DEADLINE_EXCEEDED = 'OUTCOME_DEADLINE_EXCEEDED',
}

/**
 * Result of executing the `ExecutableCode`.
 * Only generated when using code execution, and always follows a `Part`
 * containing the `ExecutableCode`.
 * @public
 */
export declare interface CodeExecutionResult {
  /**
   * Outcome of the code execution.
   */
  outcome: Outcome;
  /**
   * Contains stdout when code execution is successful, stderr or other
   * description otherwise.
   */
  output: string;
}

/**
 * Can be added in the same part as video media to specify
 * which part of the video to consider and how many frames
 * per second to analyze. VertexAI only.
 */
export declare interface VideoMetadata {
  /**
   * The video offset to start at. e.g. '3.5s'
   */
  startOffset?: string;
  /**
   * The video offset to end at e.g. '10.5s'
   */
  endOffset?: string;
  /**
   * The number of frames to consider per second
   * 0.0 to 24.0.
   */
  fps?: number;
}

export enum MediaResolutionLevel {
  MEDIA_RESOUTION_LOW = 'MEDIA_RESOUTION_LOW',
  MEDIA_RESOLUTION_MEDIUM = 'MEDIA_RESOLUTION_MEDIUM',
  MEDIA_RESOLUTION_HIGH = 'MEDIA_RESOLUTION_HIGH',
}

export declare interface MediaResolution {
  level?: MediaResolutionLevel;
}

/**
 * This is a Gemini Part. (Users never see this
 * structure, it is just built by the converters.)
 */
export declare interface Part {
  text?: string;
  inlineData?: GenerativeContentBlob;
  functionCall?: FunctionCall;
  functionResponse?: FunctionResponse;
  fileData?: FileData;
  thought?: boolean;
  thoughtSignature?: string;
  executableCode?: ExecutableCode;
  codeExecutionResult?: CodeExecutionResult;
  videoMetadata?: VideoMetadata;
  mediaResolution?: MediaResolution;
}

/**
 * The base structured datatype containing multi-part content of a message.
 */
export declare interface Content {
  /** The producer of the content. */
  role: string;
  /** Array of {@link Part}. */
  parts: Part[];
}

/**
 * Candidate for the logprobs token and score.
 * @public
 */
export declare interface LogprobsCandidate {
  /** The candidate's token string value. */
  token: string;
  /** The candidate's token id value. */
  tokenID: number;
  /** The candidate's log probability. */
  logProbability: number;
}

/**
 * Candidates with top log probabilities at each decoding step
 */
export declare interface TopCandidates {
  /** Sorted by log probability in descending order. */
  candidates: LogprobsCandidate[];
}

/**
 * Logprobs Result
 * @public
 */
export declare interface LogprobsResult {
  /** Length = total number of decoding steps. */
  topCandidates: TopCandidates[];
  /**
   * Length = total number of decoding steps.
   * The chosen candidates may or may not be in topCandidates.
   */
  chosenCandidates: LogprobsCandidate[];
}

/**
 * A candidate returned as part of a GenerateContentResponse.
 * @public
 */
export declare interface GenerateContentCandidate {
  index: number;
  content: Content;
  finishReason?: FinishReason;
  finishMessage?: string;
  safetyRatings?: SafetyRating[];
  citationMetadata?: CitationMetadata;
  /** Average log probability score of the candidate. GoogleAI only*/
  avgLogprobs?: number;
  /** Log-likelihood scores for the response tokens and top tokens. GoogleAI only*/
  logprobsResult?: LogprobsResult;
  /** Search grounding metadata. */
  groundingMetadata?: GroundingMetadata;
}

/**
 * Individual response from generateContent and generateContentStream.
 * `generateContentStream()` will return one in each chunk until
 * the stream is done.
 * @public
 */
export declare interface GenerateContentResponse {
  /** Candidate responses from the model. */
  candidates?: GenerateContentCandidate[];
  /** The prompt's feedback related to the content filters. */
  promptFeedback?: PromptFeedback;
  /** Metadata on the generation request's token usage. */
  usageMetadata?: UsageMetadata;
}

/**
 * A FunctionDeclarationsTool is a piece of code that enables the system to
 * interact with external systems to perform an action, or set of actions,
 * outside of knowledge and scope of the model.
 * @public
 */
export declare interface FunctionDeclarationsTool {
  /**
   * Optional. One or more function declarations
   * to be passed to the model along with the current user query. Model may
   * decide to call a subset of these functions by populating
   * [FunctionCall][content.part.functionCall] in the response. User should
   * provide a [FunctionResponse][content.part.functionResponse] for each
   * function call in the next turn. Based on the function responses, Model will
   * generate the final response back to the user. Maximum 64 function
   * declarations can be provided.
   */
  functionDeclarations?: FunctionDeclaration[];
}
export function isFunctionDeclarationsTool(
  tool: Tool
): tool is FunctionDeclarationsTool {
  return (tool as FunctionDeclarationsTool).functionDeclarations !== undefined;
}

/**
 * Google AI Only. Enables the model to execute code as part of generation.
 * @public
 */
export declare interface CodeExecutionTool {
  /**
   * Provide an empty object to enable code execution. This field may have
   * subfields added in the future.
   */
  codeExecution: {};
}
export function isCodeExecutionTool(tool: Tool): tool is CodeExecutionTool {
  return (tool as CodeExecutionTool).codeExecution !== undefined;
}

/**
 * Vertex AI Only. Retrieve from Vertex AI Search datastore for grounding.
 */
export declare interface VertexAISearch {
  /**
   * Fully-qualified Vertex AI Search's datastore resource ID. See
   * https://cloud.google.com/vertex-ai-search-and-conversation
   *
   * @example
   * "projects/<>/locations/<>/collections/<>/dataStores/<>"
   */
  datastore: string;
}

/**
 * Vertex AI Only. Config of Vertex RagStore grounding checking.
 */
export declare interface RagResource {
  /**
   * Optional. Vertex RAG Store corpus resource name.
   *
   * @example
   * `projects/{project}/locations/{location}/ragCorpora/{rag_corpus}`
   */
  ragCorpus?: string;

  /**
   * Optional. Set this field to select the files under the ragCorpora for
   * retrieval.
   */
  ragFileIds?: string[];
}

/** Vertex AI Only. */
export declare interface VertexRagStore {
  /**
   * Optional. List of corpora for retrieval. Currently only support one corpus
   * or multiple files from one corpus. In the future we may open up multiple
   * corpora support.
   */
  ragResources?: RagResource[];

  /** Optional. Number of top k results to return from the selected corpora. */
  similarityTopK?: number;

  /**
   * Optional. If set this field, results with vector distance smaller than
   * this threshold will be returned.
   */
  vectorDistanceThreshold?: number;
}

/**
 * Vertex AI Only. Defines a retrieval tool that model can call to access external knowledge.
 */
export declare interface Retrieval {
  /**
   * Optional. Set to use data source powered by Vertex AI Search. {@link
   * VertexAISearch}.
   */
  vertexAiSearch?: VertexAISearch;

  /** Optional. Set to use data source powered by Vertex RAG store. */
  vertexRagStore?: VertexRagStore;

  /**
   * Optional. Disable using the result from this tool in detecting grounding
   * attribution. This does not affect how the result is given to the model for
   * generation.
   */
  disableAttribution?: boolean;
}

/**
 * Vertex AI Only. Defines a retrieval tool that model can call to access external knowledge.
 */
export declare interface RetrievalTool {
  /** Optional. {@link Retrieval}. */
  retrieval?: Retrieval;
}
export function isRetrievalTool(tool: Tool): tool is RetrievalTool {
  return (tool as RetrievalTool).retrieval !== undefined;
}

export declare interface GoogleMaps {
  enableWidget: boolean;
}
export declare interface GoogleMapsTool {
  googleMaps?: GoogleMaps;
}
export function isGoogleMapsTool(tool: Tool): tool is GoogleMapsTool {
  return (tool as GoogleMapsTool).googleMaps !== undefined;
}

/**
 * Tool to retrieve public web data for grounding, powered by Google.
 */
export declare interface GoogleSearchRetrieval {
  /** Specifies the dynamic retrieval configuration for the given source. */
  dynamicRetrievalConfig?: DynamicRetrievalConfig;
}

/**
 * Defines a tool that model can call to access external knowledge.
 * @public
 */
export declare type Tool =
  | FunctionDeclarationsTool
  | RetrievalTool // Vertex AI Only
  | GoogleMapsTool // Vertex AI Only
  | CodeExecutionTool // Google AI Only
  | FileSearchTool // Google AI Only
  | UrlContextTool // Google AI Only
  | GoogleSearchRetrievalTool;

/**
 * Configuration options for model generation and outputs.
 */
export declare interface GenerationConfig {
  /** Optional. If true, the timestamp of the audio will be included in the response. */
  audioTimestamp?: boolean;
  /** Optional. Number of candidates to generate. */
  candidateCount?: number;
  /** Optional. Stop sequences. */
  stopSequences?: string[];
  /** Optional. The maximum number of output tokens to generate per message. */
  maxOutputTokens?: number;
  /** Optional. Controls the randomness of predictions. */
  temperature?: number;
  /** Optional. If specified, nucleus sampling will be used. */
  topP?: number;
  /** Optional. If specified, topK sampling will be used. */
  topK?: number;
  /**
   * Google AI only. Presence penalty applied to the next token's logprobs if the token has
   * already been seen in the response.
   */
  presencePenalty?: number;
  /**
   * Optional. Positive values penalize tokens that repeatedly appear in the generated text, decreasing the probability of repeating content.
   * This maximum value for frequencyPenalty is up to, but not including, 2.0. Its minimum value is -2.0.
  frequencyPenalty?: number;
  /**
   * Google AI Only. If True, export the logprobs results in response.
   */
  responseLogprobs?: boolean;
  /**
   * Google AI Only. Valid if responseLogProbs is set to True. This will set the number of top
   * logprobs to return at each decoding step in the logprobsResult.
   */
  logprobs?: number;
  /**
   * Optional. Output response mimetype of the generated candidate text.
   * Supported mimetype:
   * - `text/plain`: (default) Text output.
   * - `application/json`: JSON response in the candidates.
   * The model needs to be prompted to output the appropriate response type,
   * otherwise the behavior is undefined.
   */
  responseMimeType?: string;

  /**
   * Optional. The schema that generated candidate text must follow.  For more
   * information, see
   * https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/control-generated-output.
   * If set, a compatible responseMimeType must also be set.
   */
  responseSchema?: Schema;

  /**
   * Optional. Output schema of the generated response. This is an alternative to
   * `response_schema` that accepts [JSON Schema](https://json-schema.org/).
   *
   * If set, `response_schema` must be omitted, but `response_mime_type` is
   * required.
   *
   * While the full JSON Schema may be sent, not all features are supported.
   * Specifically, only the following properties are supported:
   *
   * - `$id`
   * - `$defs`
   * - `$ref`
   * - `$anchor`
   * - `type`
   * - `format`
   * - `title`
   * - `description`
   * - `enum` (for strings and numbers)
   * - `items`
   * - `prefixItems`
   * - `minItems`
   * - `maxItems`
   * - `minimum`
   * - `maximum`
   * - `anyOf`
   * - `oneOf` (interpreted the same as `anyOf`)
   * - `properties`
   * - `additionalProperties`
   * - `required`
   *
   * The non-standard `propertyOrdering` property may also be set.
   *
   * Cyclic references are unrolled to a limited degree and, as such, may only
   * be used within non-required properties. (Nullable properties are not
   * sufficient.) If `$ref` is set on a sub-schema, no other properties, except
   * for than those starting as a `$`, may be set.
   */
  responseJsonSchema?: Record<string, any>;
}

/**
 * Safety setting that can be sent as part of request parameters.
 * @public
 */
export declare interface SafetySetting {
  category: HarmCategory;
  threshold: HarmBlockThreshold;
}

export declare interface FunctionCallingConfig {
  /** Optional. Function calling mode. */
  mode?: FunctionCallingMode;

  /**
   * Optional. Function names to call. Only set when the Mode is ANY. Function
   * names should match [FunctionDeclaration.name]. With mode set to ANY, model
   * will predict a function call from the set of function names provided.
   */
  allowedFunctionNames?: string[];

  /**
   * When set to true, arguments of a single function call will be streamed out
   * in multiple parts/contents/responses. Partial parameter results will be
   * returned in the [FunctionCall.partial_args] field.
   */
  streamFunctionCallArguments?: boolean;
}

export declare interface LatLng {
  latitude?: number;
  longitude?: number;
}

export declare interface RetrievalConfig {
  latLng?: LatLng;
  languageCode?: string;
}

/** This config is shared for all tools provided in the request. */
export declare interface ToolConfig {
  /** Function calling config. */
  functionCallingConfig?: FunctionCallingConfig;
  /** Retrieval config */
  retrievalConfig?: RetrievalConfig;
}

export declare interface GenerateContentRequest {
  /** Array of {@link Content}.*/
  contents: Content[];
  /**
   * Optional. The name of the cached content used as context to serve the prediction.
   * This is the name of a `CachedContent` and not the cache object itself.
   */
  cachedContent?: string;
  /** Optional.  {@link GenerationConfig}. */
  generationConfig?: GenerationConfig;
  /**
   * Optional. Vertex AI Only. Custom metadata labels for organizing API calls and managing costs at scale. See
   * https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/add-labels-to-api-calls
   */
  labels?: Record<string, string>;
  /** Optional. Array of {@link SafetySetting}. */
  safetySettings?: SafetySetting[];
  /**
   * Optional. The user provided system instructions for the model.
   * Note: only text should be used in parts of {@link Content}
   */
  systemInstruction?: string | Part | Content;
  /** Optional. Array of {@link Tool}. */
  tools?: Tool[];
  /** Optional. This config is shared for all tools provided in the request. */
  toolConfig?: ToolConfig;
}

/**
 * Result from calling generateContentStream.
 * It contains both the stream and the final aggregated response.
 * @public
 */
export declare interface GenerateContentStreamResult {
  stream: AsyncGenerator<GenerateContentResponse>;
  response: Promise<GenerateContentResponse>;
}

export declare interface ImagenParameters {
  sampleCount?: number;
  aspectRatio?: string;
  negativePrompt?: string; // Vertex only
  seed?: number; // Vertex only
  language?: string; // Vertex only
  personGeneration?: string;
  safetySetting?: string; // Vertex only
  addWatermark?: boolean; // Vertex only
  storageUri?: string; // Vertex only
  baseSteps?: number; // Vertex Try On only
  outputOptions?: {
    // Vertex Try On only
    mimeType?: string;
    compressionQuality?: number;
  };
}

export declare interface ImagenPredictRequest {
  instances: ImagenInstance[];
  parameters: ImagenParameters;
}

export declare interface ImagenPredictResponse {
  predictions: ImagenPrediction[];
}

export declare interface ImagenPrediction {
  bytesBase64Encoded: string;
  mimeType: string;
}

export declare interface ImagenInstance {
  prompt?: string;
  image?: { bytesBase64Encoded: string };
  mask?: { image?: { bytesBase64Encoded: string } };
  personImage?: {
    // Vertex Try On Only
    image: { bytesBase64Encoded?: string; gcsUri?: string };
  };
  productImages?: {
    // Vertex Try On Only
    image: { bytesBase64Encoded?: string; gcsUri?: string };
  }[];
}
