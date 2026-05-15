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

{pkgs, language ? "js", tos ? "false", ... }: {  
  packages = [
    pkgs.nodejs
  ];
  bootstrap = ''
    mkdir "$out"
    mkdir "$out"/.idx
    cp -r ${./.idx}/. "$out/.idx/"
    cp -f ${./package.json} "$out/package.json"
    cp -f ${./package-lock.json} "$out/package-lock.json"
    cp -f ${./index.ts} "$out/index.ts"
    cp -f ${./.gitignore} "$out/.gitignore"
    cp ${./README_IDX.md} "$out"/README.md
    chmod -R u+w "$out"
  '';
}