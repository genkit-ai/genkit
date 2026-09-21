# Copyright 2026 Google LLC
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

"""Vertex image generation through the same generate(). Uses ADC, not a Gemini key."""

from genkit_google_genai import VertexAI

from genkit import Genkit

ai = Genkit(plugins=[VertexAI()])


async def main() -> None:
    response = await ai.generate(
        prompt='Draw a cat in a hat',
        model=VertexAI.gemini_image_model('gemini-2.5-flash-image'),
    )
    print(response.media[0].url if response.media else response.text)


if __name__ == '__main__':
    ai.run_main(main())
