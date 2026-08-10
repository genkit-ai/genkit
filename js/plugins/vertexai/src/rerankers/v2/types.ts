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
import { GoogleAuth, GoogleAuthOptions } from 'google-auth-library';

// These are the options you pass to the plugin during config
export interface VertexRerankerPluginOptions {
  googleAuth?: GoogleAuthOptions;
  location?: string;
  projectId?: string;
}

// These are the resolved options from the Request
export interface VertexRerankerClientOptions {
  authClient: GoogleAuth;
  location?: string;
  projectId: string;
}

export type RerankRequestRecord = {
  id: string;
  title?: string;
  content: string;
};

export type RerankRequest = {
  model: string;
  query: string;
  records: RerankRequestRecord[];
  topN?: number;
  ignoreRecordDetailsInResponse?: boolean;
};

export type RerankResponseRecord = RerankRequestRecord & { score: number };

export type RerankResponse = {
  records: RerankResponseRecord[];
};
