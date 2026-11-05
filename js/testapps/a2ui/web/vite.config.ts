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

import { defineConfig } from 'vite';

// Proxy /api to the backend (Express JS backend or the Go agent server, both on
// :8080) for both the dev server (`pnpm dev`) and the preview server
// (`pnpm preview`), so the built app talks to whichever backend is running.
const proxy = {
  '/api': {
    target: 'http://localhost:8080',
    changeOrigin: true,
  },
};

export default defineConfig({
  server: {
    port: 5173,
    proxy,
  },
  preview: {
    port: 4173,
    proxy,
  },
});
