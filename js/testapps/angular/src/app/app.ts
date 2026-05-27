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

import { Component, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { runFlow, streamFlow } from 'genkit/beta/client';

interface MenuItem {
  category: string;
  name: string;
  description: string;
  price: string;
}

interface Menu {
  restaurantName: string;
  items: MenuItem[];
}

@Component({
  selector: 'app-root',
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  themeInput = '';
  restaurantName = signal('');
  menuItems = signal<MenuItem[]>([]);
  isLoading = signal(false);
  error = signal('');

  async run() {
    const theme = this.themeInput || null;
    this.reset();
    this.isLoading.set(true);
    try {
      const result = await runFlow<Menu>({
        url: '/api/menu',
        input: theme,
      });
      this.restaurantName.set(result.restaurantName);
      this.menuItems.set(result.items);
    } catch (e: any) {
      this.error.set(e.message);
    } finally {
      this.isLoading.set(false);
    }
  }

  async stream() {
    const theme = this.themeInput || null;
    this.reset();
    this.isLoading.set(true);
    try {
      const { stream, output } = streamFlow<Menu, MenuItem>({
        url: '/api/menu',
        input: theme,
      });

      // Each chunk is a complete MenuItem — show them one by one.
      // Cast needed: streamFlow's chunk type is a union that includes the
      // output type's primitives (string | null), but at runtime each
      // chunk is always a full MenuItem object from sendChunk().
      for await (const item of stream) {
        this.menuItems.update((items) => [
          ...items,
          item as unknown as MenuItem,
        ]);
      }

      // Set the restaurant name from the final output
      const result = await output;
      this.restaurantName.set(result.restaurantName);
    } catch (e: any) {
      this.error.set(e.message);
    } finally {
      this.isLoading.set(false);
    }
  }

  private reset() {
    this.restaurantName.set('');
    this.menuItems.set([]);
    this.error.set('');
  }
}
