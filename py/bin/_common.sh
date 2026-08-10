#!/usr/bin/env bash
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

# Shared helpers for py/bin scripts. Source with: . "$(dirname "$0")/_common.sh"

# Colors (CYAN exported for scripts that source this file)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
# shellcheck disable=SC2034
CYAN='\033[0;36m'
NC='\033[0m'

# Paths (set by caller or default)
: "${SCRIPT_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
: "${PY_DIR:=$(cd "${SCRIPT_DIR}/.." && pwd)}"
: "${TOP_DIR:=$(cd "${PY_DIR}/.." && pwd)}"

# Extracts field from pyproject.toml. $1 = dir or path to .toml
get_pyproject() {
  local f="$1" k="$2"
  [[ "$f" == *.toml ]] || f="$f/pyproject.toml"
  grep "^$k" "$f" 2>/dev/null | head -1 | sed 's/.*= *"//;s/".*//'
}
get_version() { get_pyproject "$1" version; }
get_name() { get_pyproject "$1" name; }
get_requires_python() { local f="$1"; [[ "$f" == *.toml ]] || f="$f/pyproject.toml"; grep 'requires-python' "$f" 2>/dev/null | sed 's/.*= *"//;s/".*//' || echo ""; }

# Status reporting
ok() { echo -e "  ${GREEN}✓${NC} $*"; }
fail() { echo -e "  ${RED}$1${NC} $2"; ERRORS=$((ERRORS + ${3:-1})); }
fail_n() { ERRORS=$((ERRORS + $1)); }
warn() { echo -e "  ${YELLOW}$1${NC} $2"; WARNINGS=$((WARNINGS + ${3:-1})); }
header() { echo -e "${BLUE}[$1/$2] $3...${NC}"; }

# Run command, add to ERRORS on failure
run_check() {
  if "$@" > /dev/null 2>&1; then
    ok "$1 OK"
  else
    fail "ERROR" "$1 failed"
    return 1
  fi
}
