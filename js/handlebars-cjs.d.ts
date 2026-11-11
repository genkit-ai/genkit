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

// Non-published workspace module shim for dotprompt/dist/index.d.ts, which
// imports 'handlebars/dist/cjs/handlebars.js' while handlebars ships its
// declarations at types/index.d.ts. Keeping this shim at the js/ root lets
// js/tsconfig.json keep skipLibCheck off so tsc validates Genkit's emitted
// .d.ts files across workspace packages.
// Note: 'handlebars' cannot be re-exported directly here because pnpm does not
// hoist transitive dependencies to js/node_modules/.
// TODO(https://github.com/google/dotprompt): Fix dotprompt's emitted index.d.ts
// so it imports 'handlebars' instead of 'handlebars/dist/cjs/handlebars.js'
// (which also hits downstream consumers using strict + skipLibCheck: false),
// then remove this shim.
declare module 'handlebars/dist/cjs/handlebars.js' {
  namespace Handlebars {
    export interface TemplateDelegate<T = any> {
      (context: T, options?: any): string;
    }
    export interface HelperOptions {
      fn: TemplateDelegate;
      inverse: TemplateDelegate;
      hash: any;
      data?: any;
    }
    export interface HelperDelegate {
      (
        context?: any,
        arg1?: any,
        arg2?: any,
        arg3?: any,
        arg4?: any,
        arg5?: any,
        options?: HelperOptions
      ): any;
    }
  }
  export default Handlebars;
}
