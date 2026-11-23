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

import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { streamFlow } from 'genkit/beta/client';

// This sample uses an Angular dev-server proxy to connect to the flow
// see the config in genkit-app/proxy.conf.json
const url = '/streamCharacters';

@Component({
  selector: 'app-streaming-json',
  imports: [FormsModule, MatButtonModule],
  templateUrl: './streaming-json.component.html',
  styleUrl: './streaming-json.component.scss',
})
export class StreamingJSONComponent {
  count = '3';
  characters: any = undefined;
  error?: string = undefined;
  loading = false;

  async callFlow() {
    this.characters = undefined;
    this.error = undefined;
    this.loading = true;
    try {
      const response = streamFlow({
        url,
        input: Number.parseInt(this.count),
      });
      for await (const chunk of response.stream) {
        this.characters = chunk;
      }
      console.log('streamConsumer done', await response.output);
      this.loading = false;
    } catch (e) {
      this.loading = false;
      if ((e as any).cause) {
        this.error = `${(e as any).cause}`;
      } else {
        this.error = `${e}`;
      }
    }
  }
}
