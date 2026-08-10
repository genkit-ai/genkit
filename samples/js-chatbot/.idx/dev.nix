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

## Default Nix Environment for Typescript + Gemini Examples
## Requires the sample to be started with npx run genkit:dev

# To learn more about how to use Nix to configure your environment
# see: https://developers.google.com/idx/guides/customize-idx-env
{ pkgs, ... }: {
  # Which nixpkgs channel to use.
  channel = "stable-24.05"; # or "unstable"
  # Use https://search.nixos.org/packages to find packages
  packages = [
    pkgs.nodejs_20
    pkgs.util-linux
  ];
  # Sets environment variables in the workspace
  env = {
  };
  idx = {
    # Search for the extensions you want on https://open-vsx.org/ and use "publisher.id"
    extensions = [
    ];

    # Workspace lifecycle hooks
    workspace = {
      # Runs when a workspace is first created
      onCreate = {
        npm-install = "npm run setup";
        default.openFiles = [ "README.md" "server/src/index.ts" ];
      };
      onStart = {
        npm-run-server = "npm run start:server:idx";
      };
    };

    previews = {
      enable = true;
      previews = {
        web = {
          cwd = "genkit-app";
          command = ["npm" "run" "start" "--" "--port" "$PORT"];
          env = {
            PORT = "$PORT";
          };
          manager = "web";
        };
      };
    };
  };
}