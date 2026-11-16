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
import {
  FormControl,
  FormsModule,
  ReactiveFormsModule,
  Validators,
} from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { provideNativeDateAdapter } from '@angular/material/core';
import { MatDatepickerModule } from '@angular/material/datepicker';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatRadioModule } from '@angular/material/radio';
import { streamFlow } from 'genkit/beta/client';
import { MarkdownModule } from 'ngx-markdown';

// This sample uses an angular proxy to connect to the flow
// see the config in genkit-app/proxy.conf.json
const url = '/chatbotFlow';

interface ToolResponse {
  name: string;
  ref: string;
  output?: unknown;
}

interface InputSchema {
  role: 'user';
  text?: string;
  toolResponse?: ToolResponse;
}

interface ToolRequest {
  name: string;
  ref: string;
  input?: unknown;
}
interface OutputSchema {
  role: 'model';
  text?: string;
  toolRequest?: ToolRequest;
}

@Component({
  selector: 'app-chatbot',
  providers: [provideNativeDateAdapter()],
  imports: [
    FormsModule,
    MatFormFieldModule,
    MatInputModule,
    ReactiveFormsModule,
    MatButtonModule,
    MatIconModule,
    MatProgressBarModule,
    MatDatepickerModule,
    MarkdownModule,
    MatRadioModule,
  ],
  templateUrl: './chatbot.component.html',
  styleUrl: './chatbot.component.scss',
})
export class ChatbotComponent {
  history: (InputSchema | OutputSchema)[] = [];
  error?: string;
  input?: string;
  loading = false;
  id = Date.now() + '' + Math.floor(Math.random() * 1000000000);
  llmIndex: number | undefined;

  llmNames = ['Gemini Hydrogen', 'Llama 3.1 405b'];

  chatFormControl = new FormControl(
    'write a function that adds two number together',
    [Validators.required]
  );

  ask(input?: string) {
    const text = this.chatFormControl.value!.trim();
    if (!text) return;
    this.history.push({ role: 'user', text: text });
    this.chatFormControl.setValue('');
    this.chatFormControl.disable();
    this.callFlow({ role: 'user', text });
    this.loading = true;
  }

  async callFlow(input: InputSchema) {
    this.error = undefined;
    this.loading = true;
    try {
      const response = await streamFlow({
        url,
        input: {
          prompt: input,
          conversationId: this.id,
          llmIndex: this.llmIndex,
        },
      });

      let textBlock: OutputSchema | undefined = undefined;
      for await (const chunk of response.stream) {
        for (const content of chunk.content) {
          if (content.text) {
            if (!textBlock) {
              textBlock = { role: 'model', text: content.text! };
              this.history.push(textBlock);
            } else {
              textBlock.text += content.text!;
            }
          }
        }
      }

      this.loading = false;
      this.chatFormControl.enable();

      await response.output;
    } catch (e) {
      this.loading = false;
      this.chatFormControl.enable();
      if ((e as any).cause) {
        this.error = `${(e as any).cause}`;
      } else {
        this.error = `${e}`;
      }
    }
  }

  keyPress(event: KeyboardEvent) {
    if (event.key === 'Enter') {
      if (event.ctrlKey || event.shiftKey) {
        this.input += '\n';
      } else {
        this.ask(this.input);
      }
    }
  }
}
