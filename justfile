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
# Genkit Monorepo — run `just` to see all available commands.
#
# Install just: https://github.com/casey/just#installation
#   brew install just        # macOS
#   cargo install just       # Rust
#
# Language-specific commands live under submodules:
#   just py <command>        # Python SDK commands

set dotenv-load := true
set shell := ["bash", "-euo", "pipefail", "-c"]

# Python SDK subcommands (just py <command>).
mod py

# Default: show available commands.
default:
    @just --list --unsorted

# Format all code (Python, TOML).
fmt:
    ./bin/fmt

# Run all linters and type checkers.
lint:
    ./bin/lint

# Add Apache 2.0 license headers to all files.
add-license:
    ./bin/add_license

# Check license headers and compliance.
check-license:
    ./bin/check_license

# Format all pyproject.toml files.
format-toml:
    ./bin/format_toml_files

# Kill processes on common development ports.
killports:
    ./bin/killports 3100..3105 4000 8080
