import react from '@vitejs/plugin-react';
import path from 'path';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
  resolve: {
    alias: {
      // Point directly at TS sources to avoid CJS/ESM interop issues
      // with the built output in the monorepo.
      'genkit/beta/client': path.resolve(
        __dirname,
        '../../../genkit/src/client/client.ts'
      ),
      '@genkit-ai/core/async': path.resolve(
        __dirname,
        '../../../core/src/async.ts'
      ),
    },
  },
});
