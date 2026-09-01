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

"""Gemini writes and runs the Python. You call generate()."""

from genkit_google_genai import GoogleAI

from genkit import Genkit

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini_model('gemini-flash-latest'))


async def main() -> None:
    # code_execution=True is the whole feature. The model has to compute
    # this, not guess.
    response = await ai.generate(
        prompt=(
            'Monthly revenue ($k): Jan 120, Feb 135, Mar 128, Apr 150, May 162, '
            'Jun 158, Jul 175, Aug 180, Sep 195, Oct 210, Nov 205, Dec 230. '
            'What is the 12-month CAGR, and which quarter grew fastest?'
        ),
        config={'code_execution': True},
    )
    print(response.text)


if __name__ == '__main__':
    ai.run_main(main())
