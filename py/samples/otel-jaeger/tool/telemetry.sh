#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
# Local telemetry stack for the OTel sample.
#
# Downloads Jaeger and otelcol-contrib into .otel/ (skips if cached), writes
# a collector config, then keeps both running:
#
#   app --OTLP:4318--> otelcol-contrib --OTLP:14317--> jaeger (UI :16686)
#                             \--debug--> collector.log (metrics)
#
# Env overrides for locked-down networks:
#   JAEGER_BIN / OTEL_COLLECTOR_BIN         use an existing binary, skip download
#   JAEGER_VERSION / OTEL_COLLECTOR_VERSION pin a release tag instead of latest

set -euo pipefail

JAEGER_UI_PORT=16686
COLLECTOR_OTLP_GRPC_PORT=4317
COLLECTOR_OTLP_HTTP_PORT=4318
JAEGER_OTLP_GRPC_PORT=14317
JAEGER_OTLP_HTTP_PORT=14318
USER_AGENT='genkit-py-telemetry'

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OTEL_DIR="$ROOT/.otel"
BIN_DIR="$OTEL_DIR/bin"
JAEGER_LOG="$OTEL_DIR/jaeger.log"
COLLECTOR_LOG="$OTEL_DIR/collector.log"
CONFIG_FILE="$OTEL_DIR/collector.yaml"

JAEGER_PID=''
COLLECTOR_PID=''

die() {
  echo "$*" >&2
  exit 1
}

platform_arch() {
  local system machine
  system="$(uname -s | tr '[:upper:]' '[:lower:]')"
  machine="$(uname -m | tr '[:upper:]' '[:lower:]')"
  case "$system" in
    darwin) PLAT=darwin ;;
    linux) PLAT=linux ;;
    *) die "This helper is for macOS/Linux. Set JAEGER_BIN and OTEL_COLLECTOR_BIN, or run on darwin/linux." ;;
  esac
  case "$machine" in
    x86_64 | x64 | amd64) ARCH=amd64 ;;
    arm64 | aarch64) ARCH=arm64 ;;
    *) ARCH="$machine" ;;
  esac
}

latest_tag() {
  local repo="$1"
  local url
  url="$(curl -fsSLI -A "$USER_AGENT" -o /dev/null -w '%{url_effective}' \
    "https://github.com/${repo}/releases/latest")"
  printf '%s' "${url##*/}"
}

ensure_binary() {
  local name="$1"
  local env_var="$2"
  local version_var="$3"
  local repo="$4"
  local kind="$5"
  local override="${!env_var:-}"
  local target pinned tag version asset url tmp found

  if [[ -n "$override" ]]; then
    echo "Using ${name} from ${env_var}=${override}" >&2
    printf '%s' "$override"
    return
  fi

  target="$BIN_DIR/$name"
  if [[ -x "$target" ]]; then
    echo "${name} already cached: ${target}" >&2
    printf '%s' "$target"
    return
  fi

  echo "${name} not found; resolving release from ${repo}..." >&2
  pinned="${!version_var:-}"
  if [[ -n "$pinned" ]]; then
    tag="$pinned"
  else
    tag="$(latest_tag "$repo")"
  fi

  if [[ "$kind" == jaeger ]]; then
    [[ "$tag" == v2.* ]] || die "Need a Jaeger v2 tag (got ${tag}). Set JAEGER_VERSION=v2.x.y"
    version="${tag#v}"
    asset="jaeger-${version}-${PLAT}-${ARCH}.tar.gz"
    url="https://github.com/${repo}/releases/download/${tag}/${asset}"
  else
    version="${tag#v}"
    asset="otelcol-contrib_${version}_${PLAT}_${ARCH}.tar.gz"
    url="https://github.com/${repo}/releases/download/${tag}/${asset}"
  fi

  tmp="$(mktemp -d "${TMPDIR:-/tmp}/genkit-otel.XXXXXX")"
  echo "Downloading ${asset}..." >&2
  curl -fL -A "$USER_AGENT" -o "$tmp/$asset" "$url"
  echo "Extracting..." >&2
  tar -xzf "$tmp/$asset" -C "$tmp"
  found="$(find "$tmp" -type f -name "$name" | head -n 1)"
  if [[ -z "$found" ]]; then
    rm -rf "$tmp"
    die "Binary \"${name}\" not found in ${asset}"
  fi
  cp "$found" "$target"
  chmod +x "$target"
  rm -rf "$tmp"
  echo "Installed ${name}: ${target}" >&2
  printf '%s' "$target"
}

write_collector_config() {
  cat >"$CONFIG_FILE" <<EOF
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: "0.0.0.0:${COLLECTOR_OTLP_GRPC_PORT}"
      http:
        endpoint: "0.0.0.0:${COLLECTOR_OTLP_HTTP_PORT}"
processors:
  batch:
    timeout: 1s
exporters:
  otlp:
    endpoint: "127.0.0.1:${JAEGER_OTLP_GRPC_PORT}"
    tls:
      insecure: true
  debug:
    verbosity: detailed
service:
  telemetry:
    logs:
      level: "info"
    metrics:
      level: "none"
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug]
EOF
  echo "Wrote collector config: ${CONFIG_FILE}"
}

port_open() {
  local port="$1"
  (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1
}

wait_until_ready() {
  local pid="$1"
  local port="$2"
  local name="$3"
  local log="$4"
  local i
  for i in $(seq 1 60); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "${name} failed to start (port ${port})." >&2
      [[ -f "$log" ]] && tail -c 2000 "$log" >&2
      return 1
    fi
    if port_open "$port"; then
      return 0
    fi
    sleep 0.5
  done
  echo "${name} failed to start (port ${port})." >&2
  [[ -f "$log" ]] && tail -c 2000 "$log" >&2
  return 1
}

shutdown() {
  local code="${1:-0}"
  echo
  echo 'Shutting down...'
  [[ -n "$COLLECTOR_PID" ]] && kill "$COLLECTOR_PID" 2>/dev/null || true
  [[ -n "$JAEGER_PID" ]] && kill "$JAEGER_PID" 2>/dev/null || true
  wait "$COLLECTOR_PID" 2>/dev/null || true
  wait "$JAEGER_PID" 2>/dev/null || true
  exit "$code"
}

mkdir -p "$BIN_DIR"
platform_arch
echo "Platform: ${PLAT}/${ARCH}"

command -v curl >/dev/null || die 'curl is required to download Jaeger and otelcol-contrib.'

OTELCOL_PATH="$(ensure_binary otelcol-contrib OTEL_COLLECTOR_BIN OTEL_COLLECTOR_VERSION \
  open-telemetry/opentelemetry-collector-releases otelcol)"
JAEGER_PATH="$(ensure_binary jaeger JAEGER_BIN JAEGER_VERSION jaegertracing/jaeger jaeger)"

pkill -f "$OTELCOL_PATH" 2>/dev/null || true
pkill -f "$JAEGER_PATH" 2>/dev/null || true

write_collector_config

trap 'shutdown 0' INT TERM

echo "Starting Jaeger... logs: ${JAEGER_LOG}"
: >"$JAEGER_LOG"
"$JAEGER_PATH" \
  "--set=receivers.otlp.protocols.grpc.endpoint=127.0.0.1:${JAEGER_OTLP_GRPC_PORT}" \
  "--set=receivers.otlp.protocols.http.endpoint=127.0.0.1:${JAEGER_OTLP_HTTP_PORT}" \
  >>"$JAEGER_LOG" 2>&1 &
JAEGER_PID=$!
wait_until_ready "$JAEGER_PID" "$JAEGER_UI_PORT" Jaeger "$JAEGER_LOG" || shutdown 1
echo 'Jaeger is up.'

echo "Starting otelcol-contrib... logs: ${COLLECTOR_LOG}"
: >"$COLLECTOR_LOG"
"$OTELCOL_PATH" --config "$CONFIG_FILE" >>"$COLLECTOR_LOG" 2>&1 &
COLLECTOR_PID=$!
wait_until_ready "$COLLECTOR_PID" "$COLLECTOR_OTLP_HTTP_PORT" Collector "$COLLECTOR_LOG" || shutdown 1
echo 'Collector is up.'

cat <<EOF

Local telemetry environment is running.

  Jaeger UI:  http://localhost:${JAEGER_UI_PORT}
  OTLP in:    http://localhost:${COLLECTOR_OTLP_HTTP_PORT} (http)  |  localhost:${COLLECTOR_OTLP_GRPC_PORT} (grpc)
  Metrics:    tail -f ${COLLECTOR_LOG}

Run the sample in another terminal:
  export GEMINI_API_KEY=...
  python src/main.py

Press Ctrl+C to stop.
EOF

while true; do
  kill -0 "$JAEGER_PID" 2>/dev/null || shutdown 0
  kill -0 "$COLLECTOR_PID" 2>/dev/null || shutdown 0
  sleep 0.5
done
