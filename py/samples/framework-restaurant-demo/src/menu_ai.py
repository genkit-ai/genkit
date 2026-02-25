# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


"""AI flows and tools for the menu sample."""

import os

from genkit.ai import Genkit
from genkit.plugins.dev_local_vectorstore import define_dev_local_vector_store
from genkit.plugins.google_genai import GoogleAI

if 'GEMINI_API_KEY' not in os.environ:
    raise ValueError(
        'GEMINI_API_KEY environment variable is required to run this sample.\n'
        'Please set it before starting the application:\n'
        '  export GEMINI_API_KEY=your_api_key_here\n'
        'You can get an API key from https://aistudio.google.com/'
    )


ai = Genkit(plugins=[GoogleAI()])

# Define dev local vector store
define_dev_local_vector_store(
    ai,
    name='menu-items',
    embedder='googleai/gemini-embedding-001',
)
