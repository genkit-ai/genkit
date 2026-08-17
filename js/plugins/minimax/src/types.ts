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
 * Service regions that expose the MiniMax API.
 *
 * - `global`: the international endpoint.
 * - `cn`: the mainland China endpoint.
 */
export const MINIMAX_REGIONS = ['global', 'cn'] as const;

/**
 * Service region used to reach the MiniMax API.
 */
export type MiniMaxRegion = (typeof MINIMAX_REGIONS)[number];

/**
 * Base URL of each supported {@link MiniMaxRegion}.
 */
export const REGION_BASE_URLS: Record<MiniMaxRegion, string> = {
  global: 'https://api.minimax.io',
  cn: 'https://api.minimaxi.com',
};

/**
 * Default region used when the plugin is configured without one.
 */
export const DEFAULT_REGION: MiniMaxRegion = 'global';

/**
 * Resource path of the music generation endpoint.
 */
export const MUSIC_GENERATION_PATH = '/v1/music_generation';

/**
 * Request fields that are only accepted by the `cn` region.
 */
export const CN_ONLY_REQUEST_FIELDS = ['aigc_watermark'] as const;

/**
 * Number of hours a generated audio URL stays reachable when
 * `output_format: 'url'` is used.
 */
export const AUDIO_URL_TTL_HOURS = 24;

/**
 * Options accepted by the `minimax` plugin.
 */
export interface MiniMaxPluginOptions {
  /**
   * MiniMax API key. Defaults to the `MINIMAX_API_KEY` environment variable.
   *
   * Pass `false` to skip the check at plugin initialization and provide the key
   * per request through the model config instead.
   */
  apiKey?: string | false;

  /**
   * Region whose endpoint should be called. Defaults to `global`.
   */
  region?: MiniMaxRegion;

  /**
   * Overrides the base URL derived from {@link MiniMaxPluginOptions.region}.
   */
  baseUrl?: string;

  /**
   * Extra headers added to every request.
   */
  customHeaders?: Record<string, string>;
}

/**
 * Resolved per-request client configuration.
 */
export interface ClientOptions {
  apiKey: string;
  region: MiniMaxRegion;
  baseUrl?: string;
  customHeaders?: Record<string, string>;
  signal?: AbortSignal;
}

/**
 * Output formats supported by the music generation endpoint.
 *
 * - `url`: a temporary download URL, see {@link AUDIO_URL_TTL_HOURS}.
 * - `hex`: the audio bytes as a hexadecimal string.
 */
export const MUSIC_OUTPUT_FORMATS = ['url', 'hex'] as const;

/**
 * Encoding of the generated audio returned by the API.
 */
export type MusicOutputFormat = (typeof MUSIC_OUTPUT_FORMATS)[number];

/**
 * Output formats the endpoint supports while streaming.
 */
export const MUSIC_STREAM_OUTPUT_FORMATS = ['hex'] as const;

/**
 * Output format used when the request does not specify one.
 */
export const DEFAULT_OUTPUT_FORMAT: MusicOutputFormat = 'hex';

/**
 * Audio container formats supported by the music generation endpoint.
 */
export const MUSIC_AUDIO_FORMATS = ['mp3', 'wav', 'pcm'] as const;

/**
 * Container format of the generated audio.
 */
export type MusicAudioFormat = (typeof MUSIC_AUDIO_FORMATS)[number];

/**
 * Container format used when the request does not specify one.
 */
export const DEFAULT_AUDIO_FORMAT: MusicAudioFormat = 'mp3';

/**
 * Sample rates accepted by `audio_setting.sample_rate`.
 */
export const MUSIC_SAMPLE_RATES = [16000, 24000, 32000, 44100] as const;

/**
 * Bitrates accepted by `audio_setting.bitrate`.
 */
export const MUSIC_BITRATES = [32000, 64000, 128000, 256000] as const;

/**
 * Audio output configuration sent as `audio_setting`.
 *
 * Allowed values are enforced by the model config schema, see
 * {@link MUSIC_SAMPLE_RATES} and {@link MUSIC_BITRATES}.
 */
export interface MusicAudioSetting {
  sample_rate?: number;
  bitrate?: number;
  format?: MusicAudioFormat;
}

/**
 * Body of a `POST /v1/music_generation` request.
 */
export interface MusicGenerationRequest {
  /** Music model to run. The only field the API always requires. */
  model: string;
  /** Style, mood or scenario description. */
  prompt?: string;
  /** Lyrics to sing, with lines separated by `\n`. */
  lyrics?: string;
  /** Whether the API should stream the audio back. */
  stream?: boolean;
  /** Encoding of the returned audio. */
  output_format?: MusicOutputFormat;
  /** Audio output configuration. */
  audio_setting?: MusicAudioSetting;
  /** Let the service derive lyrics from `prompt`. */
  lyrics_optimizer?: boolean;
  /** Generate vocal-free audio, which makes `lyrics` unnecessary. */
  is_instrumental?: boolean;
  /** Embed an AIGC watermark. Only accepted by the `cn` region. */
  aigc_watermark?: boolean;
}

/**
 * Generation status reported by `data.status`.
 */
export const MUSIC_TASK_STATUS = {
  /** The service is still working on the audio. */
  IN_PROGRESS: 1,
  /** The audio is complete and present in `data.audio`. */
  COMPLETED: 2,
} as const;

/**
 * `status_code` returned by `base_resp` when a request succeeded.
 */
export const SUCCESS_STATUS_CODE = 0;

/**
 * Payload of a `POST /v1/music_generation` response.
 */
export interface MusicGenerationResponse {
  data?: {
    /** See {@link MUSIC_TASK_STATUS}. */
    status?: number;
    /**
     * Generated audio. A hexadecimal string when `output_format` is `hex`,
     * or a temporary download URL when it is `url`.
     */
    audio?: string;
  };
  base_resp?: {
    /** See {@link SUCCESS_STATUS_CODE}. */
    status_code?: number;
    status_msg?: string;
  };
  trace_id?: string;
  extra_info?: Record<string, unknown>;
}
