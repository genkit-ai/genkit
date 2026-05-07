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

import {
  orchestratorWithA2A,
  testDirectA2AWeather,
  testSubAgentA2AWeather,
  weatherA2AAgent,
} from './a2a-weather.js';

console.log('Loaded A2A weather agent:', weatherA2AAgent.__action.name);
console.log('Loaded orchestrator agent:', orchestratorWithA2A.__action.name);
console.log('Loaded direct A2A flow:', testDirectA2AWeather.__action.name);
console.log('Loaded sub-agent A2A flow:', testSubAgentA2AWeather.__action.name);

export * from './a2a-weather.js';
